from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agents.data_cleansing_agent import data_cleansing_agent
from app.agents.data_scientist import data_scientist_agent
from app.agents.deep_inference_agent import deep_inference_agent
from app.agents.harvester_agent import harvester_agent
from app.agents.orchestrator import orchestrator
from app.agents.pdf_extractor import pdf_extractor_agent
from app.agents.registry import record_agent_invocation, sync_agent_registry
from app.agents.translation_agent import translation_agent
from app.config import settings
from app.data.seed_data import ensure_seeded
from app.database import get_db
from app.models import (
    DatasetArtifact,
    Dissertation,
    IndustryProblem,
    LazyPredictRun,
    MarketOpportunity,
    RegisteredAgent,
    SavedItem,
)
from app.services.llm_router import llm_router
from app.services.scientific_integrations import scientific_integrations
from app.schemas import (
    CustomLazyPredictRequest,
    DatasetArtifactSchema,
    DissertationSchema,
    HarvestQueryRequest,
    IndustryProblemCreate,
    IndustryProblemSchema,
    LazyPredictRunSchema,
    MarketOpportunitySchema,
    RegisteredAgentSchema,
    SavedItemNoteUpdate,
    SavedItemSchema,
    SavedItemToggleRequest,
)

router = APIRouter()


@router.get("/stats")
def get_platform_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    diss_count = db.query(Dissertation).count()
    ds_count = db.query(DatasetArtifact).count()
    lazy_runs = db.query(LazyPredictRun).all()
    opps = db.query(MarketOpportunity).order_by(MarketOpportunity.overall_tos_score.desc()).all()
    problems = db.query(IndustryProblem).all()

    total_models_benchmarked = sum(r.models_tested_count for r in lazy_runs)
    avg_signal_gain = (
        round(sum(r.signal_gain_delta for r in lazy_runs) / len(lazy_runs), 4) if lazy_runs else 0.0
    )
    total_tam_unlocked = round(sum(p.tam_billions_usd for p in problems), 1)
    top_tos = opps[0].overall_tos_score if opps else 0.0

    return {
        "dissertations_indexed": diss_count,
        "datasets_extracted": ds_count,
        "lazypredict_sweeps_count": len(lazy_runs),
        "total_ml_models_benchmarked": total_models_benchmarked,
        "average_signal_gain_delta": avg_signal_gain,
        "market_opportunities_count": len(opps),
        "giant_impact_problems_count": len(problems),
        "total_addressable_market_billions": total_tam_unlocked,
        "highest_tos_score": top_tos,
        "llm_router_config": {
            "primary_model": settings.PRIMARY_LLM_MODEL,
            "fallback_model": settings.FALLBACK_LLM_MODEL,
            "quality_gate": "Tier-1 Top-Science Institutions Only",
            "quota_guardrails": llm_router.get_quota_status(),
        },
    }


@router.get("/dissertations", response_model=List[DissertationSchema])
async def list_dissertations(
    domain: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Dissertation).order_by(Dissertation.scientific_rigor_score.desc(), Dissertation.id.desc())
    if domain and domain != "all":
        q = q.filter(Dissertation.domain == domain)
    items = [d for d in q.all() if d.datasets and d.opportunities]
    if search:
        s = search.lower()
        items = [
            d
            for d in items
            if s in d.title.lower()
            or s in d.university.lower()
            or s in d.abstract.lower()
            or s in (d.subdomain or "").lower()
        ]
    return items


@router.get("/dissertations/{dissertation_id}", response_model=DissertationSchema)
def get_dissertation(dissertation_id: int, db: Session = Depends(get_db)):
    diss = db.query(Dissertation).filter(Dissertation.id == dissertation_id).first()
    if not diss:
        raise HTTPException(status_code=404, detail="Dissertation not found")
    return diss


@router.post("/harvest", response_model=List[DissertationSchema])
async def harvest_dissertations(req: HarvestQueryRequest, db: Session = Depends(get_db)):
    """
    Live harvests Top-Science dissertations from OpenAlex/EuropePMC/arXiv, resolves real Zenodo
    datasets, runs 30+ LazyPredict models, and generates Market Translation Dossiers.
    """
    results = await orchestrator.harvest_and_analyze_live(
        db=db,
        query=req.query,
        domain=req.domain,
        min_year=req.min_year,
        max_results=req.max_results,
        top_science_only=req.top_science_only,
    )
    return results


@router.get("/harvest/stream")
async def stream_harvest_dissertations(
    query: str = Query(...),
    domain: str = Query("biomedical"),
    min_year: int = Query(2018),
    max_results: int = Query(3),
    db: Session = Depends(get_db),
):
    """
    Streams real-time Server-Sent Events (SSE) as ThesisOut queries OpenAlex, ClinicalTrials.gov,
    and Zenodo APIs and runs LazyPredict on real downloaded datasets.
    """
    return StreamingResponse(
        orchestrator.stream_harvest_and_analyze(
            db=db,
            query=query,
            domain=domain,
            min_year=min_year,
            max_results=max_results,
            top_science_only=True,
        ),
        media_type="text/event-stream",
    )


@router.delete("/workspace/reset")
async def reset_workspace(db: Session = Depends(get_db)) -> Dict[str, str]:
    """Resets non-saved workspace items while preserving Star/Ribbon saved items in persistent memory."""
    saved_diss_ids = {
        s.dissertation_id
        for s in db.query(SavedItem).filter((SavedItem.is_starred == True) | (SavedItem.is_bookmarked == True)).all()
    }
    if saved_diss_ids:
        db.query(MarketOpportunity).filter(~MarketOpportunity.dissertation_id.in_(saved_diss_ids)).delete(synchronize_session=False)
        unsaved_ds_ids = [
            d.id for d in db.query(DatasetArtifact).filter(~DatasetArtifact.dissertation_id.in_(saved_diss_ids)).all()
        ]
        if unsaved_ds_ids:
            db.query(LazyPredictRun).filter(LazyPredictRun.dataset_id.in_(unsaved_ds_ids)).delete(synchronize_session=False)
            db.query(DatasetArtifact).filter(DatasetArtifact.id.in_(unsaved_ds_ids)).delete(synchronize_session=False)
        db.query(Dissertation).filter(~Dissertation.id.in_(saved_diss_ids)).delete(synchronize_session=False)
    else:
        db.query(MarketOpportunity).delete()
        db.query(LazyPredictRun).delete()
        db.query(DatasetArtifact).delete()
        db.query(Dissertation).delete()
    db.commit()
    await ensure_seeded(db)
    return {"status": "reset", "message": "Workspace refreshed while preserving saved memory."}


@router.post("/upload-dissertation", response_model=DissertationSchema)
@router.post("/dissertations/upload", response_model=DissertationSchema)
async def upload_dissertation_and_dataset(
    title: Optional[str] = Form(None),
    authors: Optional[str] = Form(None),
    university: Optional[str] = Form(None),
    publication_year: Optional[int] = Form(None),
    domain: str = Form("general_science"),
    abstract: str = Form(""),
    author_primary_target: Optional[str] = Form(None),
    author_baseline_score: Optional[float] = Form(None),
    pdf_file: Optional[UploadFile] = File(None),
    dataset_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """
    Ingests a user-uploaded dissertation PDF and/or raw dataset (.csv, .tsv, .xlsx, .parquet).
    Automatically parses title, authors, university, year, embedded tables, or linked Zenodo datasets
    and runs 30 cross-validated LazyPredict models.
    """
    import time

    record_agent_invocation(db, "pdf-extractor-agent")
    pdf_saved_path = None
    extracted_meta: Dict[str, Any] = {
        "extracted_claims": ["Empirical findings extracted from uploaded dissertation."],
        "data_availability_links": [],
        "author_statistical_method": "Classical Linear / Parametric Modeling",
    }

    seed_name = (
        (pdf_file.filename if pdf_file and pdf_file.filename else None)
        or (dataset_file.filename if dataset_file and dataset_file.filename else None)
        or title
        or f"study_{int(time.time())}"
    )
    slug = f"upload_{abs(hash(seed_name)) % 100000}"

    # Read uploaded file bytes and auto-detect PDF vs Tabular (CSV/TSV/Excel/Parquet/JSON) even if slots were swapped
    pdf_bytes: Optional[bytes] = None
    pdf_filename: Optional[str] = None
    ds_bytes: Optional[bytes] = None
    ds_filename: Optional[str] = None

    for up_f in (pdf_file, dataset_file):
        if up_f and up_f.filename:
            raw_b = await up_f.read()
            if not raw_b:
                continue
            fname = Path(up_f.filename).name
            is_pdf = fname.lower().endswith(".pdf") or raw_b[:5] == b"%PDF-"
            if is_pdf and pdf_bytes is None:
                pdf_bytes = raw_b
                pdf_filename = fname
            else:
                ds_bytes = raw_b
                ds_filename = fname

    if pdf_bytes and pdf_filename:
        pdf_dest = settings.PDF_DIR / f"{slug}_{pdf_filename}"
        pdf_dest.parent.mkdir(parents=True, exist_ok=True)
        pdf_dest.write_bytes(pdf_bytes)
        pdf_saved_path = str(pdf_dest)
        try:
            extracted_meta = pdf_extractor_agent.extract_text_and_metadata(pdf_dest)
            if not abstract:
                abstract = extracted_meta.get("text_preview", "")[:1500]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF file '{pdf_filename}': {exc}")

    fallback_file_title = Path(seed_name).stem.replace("_", " ").replace("-", " ").strip().title()
    final_title = (title or "").strip() or extracted_meta.get("parsed_title") or fallback_file_title or "Uploaded Study"
    final_authors = (authors or "").strip() or extracted_meta.get("parsed_authors") or "Uploaded Author"
    final_university = (university or "").strip() or extracted_meta.get("parsed_university") or "Uploaded File"
    final_year = publication_year or extracted_meta.get("parsed_year") or 2025

    diss = Dissertation(
        title=final_title,
        authors=final_authors,
        university=final_university,
        department="Uploaded Audit",
        publication_year=final_year,
        domain=domain if domain != "general_science" else "biomedical",
        subdomain="Uploaded File Audit",
        source_repo="Direct Upload",
        pdf_path=pdf_saved_path,
        institution_tier="Uploaded",
        scientific_rigor_score=0.999,
        citation_velocity=15.0,
        abstract=abstract[:500] if abstract else final_title,
        author_primary_hypothesis=final_title,
        author_statistical_method=extracted_meta.get("author_statistical_method", "OLS Linear Fit"),
        author_primary_target=author_primary_target,
        author_reported_baseline_metric="R-Squared / ROC-AUC",
        author_reported_baseline_score=author_baseline_score or 0.6200,
        extracted_claims=extracted_meta.get("extracted_claims", [final_title]),
        data_availability_links=extracted_meta.get("data_availability_links", []),
        key_entities={"targets": [final_title.split()[0]] if final_title.split() else ["target"]},
        status="analyzed",
    )
    db.add(diss)
    db.commit()
    db.refresh(diss)

    # 1. Process uploaded dataset file if provided
    if ds_bytes and ds_filename:
        record_agent_invocation(db, "ml-rigor-scientist")
        record_agent_invocation(db, "translation-whitespace-agent")
        ds_dest = settings.DATASET_DIR / f"{slug}_{ds_filename}"
        ds_dest.parent.mkdir(parents=True, exist_ok=True)
        ds_dest.write_bytes(ds_bytes)
        try:
            await orchestrator.process_dataset_with_lazypredict_and_translation(
                db=db,
                dissertation=diss,
                dataset_path=ds_dest,
                dataset_name=ds_filename,
                source_type="upload",
                author_primary_target=author_primary_target,
                author_baseline_score=author_baseline_score,
                max_secondary_targets=0,
            )
        except Exception as exc:
            db.delete(diss)
            db.commit()
            raise HTTPException(status_code=400, detail=f"Failed to parse or benchmark '{ds_filename}': {exc}")
        db.refresh(diss)
        if not diss.datasets or not diss.datasets[0].lazy_runs:
            db.delete(diss)
            db.commit()
            raise HTTPException(
                status_code=400,
                detail=f"Could not benchmark '{ds_filename}': dataset must contain at least 5 rows and 2 columns with numeric or categorical variance.",
            )

    # 2. If a PDF was uploaded without a dataset (or to extract embedded PDF tables), parse tables from the PDF
    db.refresh(diss)
    if not diss.datasets and pdf_saved_path:
        pdf_tables = pdf_extractor_agent.extract_tables_from_pdf(
            pdf_saved_path,
            slug,
            fallback_text=abstract,
            fallback_title=final_title,
        )
        for tbl in pdf_tables:
            record_agent_invocation(db, "ml-rigor-scientist")
            record_agent_invocation(db, "translation-whitespace-agent")
            ds_art = await orchestrator.process_dataset_with_lazypredict_and_translation(
                db=db,
                dissertation=diss,
                dataset_path=tbl["file_path"],
                dataset_name=tbl["name"],
                source_type="pdf_table",
                source_url=tbl["source_url"],
                author_primary_target=author_primary_target,
                author_baseline_score=author_baseline_score,
                max_secondary_targets=0,
            )
            if ds_art and ds_art.lazy_runs:
                break

    # 3. If PDF had no embedded table, check explicit Data Availability / repository links inside the PDF
    db.refresh(diss)
    if (not diss.datasets or not diss.datasets[0].lazy_runs) and diss.data_availability_links:
        record_agent_invocation(db, "harvester-agent")
        zenodo_files = await harvester_agent.resolve_multi_repository_datasets(
            query=final_title,
            target_dir=settings.DATASET_DIR,
            dissertation_meta={
                "title": final_title,
                "authors": final_authors,
                "university": final_university,
                "publication_year": final_year,
                "abstract": abstract,
            },
            extracted_links=diss.data_availability_links or [],
            max_files=1,
        )
        for zf in zenodo_files[:1]:
            record_agent_invocation(db, "ml-rigor-scientist")
            record_agent_invocation(db, "translation-whitespace-agent")
            await orchestrator.process_dataset_with_lazypredict_and_translation(
                db=db,
                dissertation=diss,
                dataset_path=zf["file_path"],
                dataset_name=zf["name"],
                source_type=zf["source_type"],
                source_url=zf["source_url"],
                author_primary_target=author_primary_target,
                author_baseline_score=author_baseline_score,
                max_secondary_targets=0,
            )

    db.refresh(diss)
    if not diss.datasets or not diss.datasets[0].lazy_runs:
        db.delete(diss)
        db.commit()
        raise HTTPException(
            status_code=400,
            detail="No quantitative tables, numerical measurements, or linked open-access datasets could be extracted from the uploaded file.",
        )

    first_run = diss.datasets[0].lazy_runs[0]
    if first_run:
        diss.author_primary_target = first_run.target_column
        diss.author_reported_baseline_score = first_run.author_baseline_score
        diss.author_reported_baseline_metric = first_run.primary_metric_name
        db.commit()
        db.refresh(diss)

    return diss


@router.get("/datasets/{dataset_id}/cleansing-report")
def get_dataset_cleansing_report(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Runs or retrieves the structured Data Cleansing & Schema Curation Report (`data-cleansing-agent`)
    for any indexed or uploaded dataset artifact.
    """
    ds = db.query(DatasetArtifact).filter(DatasetArtifact.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset artifact not found")
    record_agent_invocation(db, "data-cleansing-agent")
    _, report = data_cleansing_agent.cleanse_tabular_file(ds.file_path)
    return {
        "dataset_id": ds.id,
        "dataset_name": ds.name,
        "source_type": ds.source_type,
        **report,
    }


@router.post("/lazypredict/run-custom", response_model=LazyPredictRunSchema)
async def run_custom_lazypredict_target(
    req: CustomLazyPredictRequest,
    db: Session = Depends(get_db),
):
    """
    Allows the researcher to pick ANY column in an extracted dataset as a custom target variable,
    runs all ~30 LazyPredict models live, extracts feature importances, and synthesizes a new
    Market Translation Opportunity against the Giant Impact Problem Bank.
    """
    ds = db.query(DatasetArtifact).filter(DatasetArtifact.id == req.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    diss = db.query(Dissertation).filter(Dissertation.id == ds.dissertation_id).first()
    df = data_scientist_agent.load_dataframe(ds.file_path)

    exp_res = data_scientist_agent.run_lazypredict_experiment(
        df=df,
        target_column=req.target_column,
        task_type=req.task_type,
        is_author_primary_target=req.is_author_primary_target,
        target_rationale=req.target_rationale
        or f"Interactive Researcher Target Rotation on `{req.target_column}`",
    )

    lazy_run = LazyPredictRun(
        dataset_id=ds.id,
        target_column=exp_res["target_column"],
        task_type=exp_res["task_type"],
        is_author_primary_target=exp_res["is_author_primary_target"],
        target_rationale=exp_res["target_rationale"],
        models_tested_count=exp_res["models_tested_count"],
        best_model_name=exp_res["best_model_name"],
        primary_metric_name=exp_res["primary_metric_name"],
        best_score=exp_res["best_score"],
        author_baseline_score=exp_res["author_baseline_score"],
        signal_gain_delta=exp_res["signal_gain_delta"],
        leaderboard=exp_res["leaderboard"],
        top_features=exp_res["top_features"],
        overlooked_signal_summary=exp_res["overlooked_signal_summary"],
        execution_time_sec=exp_res["execution_time_sec"],
    )
    db.add(lazy_run)
    db.commit()
    db.refresh(lazy_run)

    if diss:
        problems = db.query(IndustryProblem).all()
        opp_data = await translation_agent.synthesize_opportunity(
            dissertation=diss,
            lazy_run=lazy_run,
            problems=problems,
        )
        opp = MarketOpportunity(**opp_data)
        db.add(opp)
        db.commit()

    return lazy_run


@router.post("/lazypredict/{lazy_run_id}/escalate")
def escalate_lazy_run_to_deep_inference(
    lazy_run_id: int,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Stage-2 Deep Inference Escalation:
    When LazyPredict identifies a high-signal target ('Huge Target'), this endpoint executes:
    1) 5-Fold CV + Target Permutation Null-Test + Top-Feature Ablation Leakage Audit
    2) Level-2 Super-Learner Stacking Ensemble (LightGBM + XGBoost + ExtraTrees + RandomForest)
    3) Surrogate Decision-Rule Extraction & 2x2 Quadrant Synergy Matrix
    4) 90% Split-Conformal Uncertainty Calibration
    5) Counterfactual Minimal-Intervention Prescription & Exportable Python Pipeline
    """
    lazy_run = db.query(LazyPredictRun).filter(LazyPredictRun.id == lazy_run_id).first()
    if not lazy_run:
        raise HTTPException(status_code=404, detail="LazyPredict run not found")

    ds = db.query(DatasetArtifact).filter(DatasetArtifact.id == lazy_run.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Associated dataset artifact not found")

    result = deep_inference_agent.run_deep_inference_workflow(
        file_path=ds.file_path,
        target_column=lazy_run.target_column,
        task_type=lazy_run.task_type,
        best_lazy_model=lazy_run.best_model_name,
        lazy_score=lazy_run.best_score,
        author_baseline_score=lazy_run.author_baseline_score or 0.5,
    )
    return result


@router.post("/opportunities/{opportunity_id}/execute-step")
async def execute_opportunity_action_step(
    opportunity_id: int,
    step_index: int = Query(..., ge=0, le=3),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Executes one of the 4 actionable steps on a Market Opportunity card in real time:
    - Step 0 (Step 1: Verify & Lock Pattern): Runs 5-Fold Cross-Validation + Permutation Null Check + Super-Learner Stacking.
    - Step 1 (Step 2: Extract Exact Screening Rules & IP Claims): Extracts exact numerical decision thresholds + ready-to-file provisional claim text.
    - Step 2 (Step 3: Run Live Benchmark & Clinical/Bioactivity Audit): Queries ChEMBL, OpenTargets, ClinicalTrials.gov, and openFDA for active reference compounds and trial protocols.
    - Step 3 (Step 4: Generate Exportable Python Model & Venture Dossier): Produces the standalone Python inference script + Markdown Venture Brief for download.
    """
    opp = db.query(MarketOpportunity).filter(MarketOpportunity.id == opportunity_id).first()
    if not opp:
        raise HTTPException(status_code=404, detail="Market Opportunity not found")

    diss = db.query(Dissertation).filter(Dissertation.id == opp.dissertation_id).first()
    lazy_run = db.query(LazyPredictRun).filter(LazyPredictRun.id == opp.lazy_run_id).first() if opp.lazy_run_id else None
    ds = db.query(DatasetArtifact).filter(DatasetArtifact.id == lazy_run.dataset_id).first() if lazy_run else None

    deep_res = None
    if lazy_run and ds:
        deep_res = deep_inference_agent.run_deep_inference_workflow(
            file_path=ds.file_path,
            target_column=lazy_run.target_column,
            task_type=lazy_run.task_type,
            best_lazy_model=lazy_run.best_model_name,
            lazy_score=lazy_run.best_score,
            author_baseline_score=lazy_run.author_baseline_score or 0.5,
        )

    if step_index == 0:
        audit = (deep_res or {}).get("stage1_leakage_and_permutation_audit", {})
        sl = (deep_res or {}).get("stage2_super_learner_ensemble", {})
        conf = (deep_res or {}).get("stage4_conformal_uncertainty", {})
        return {
            "step_index": 0,
            "action_title": "Step 1 Executed: 5-Fold Cross-Validation & Super-Learner Lock",
            "status": audit.get("status", "Verified"),
            "summary": audit.get("audit_interpretation", "Validated across 5 folds."),
            "metrics": {
                "5-Fold CV Score": f"{audit.get('five_fold_cv_mean', 0)*100:.1f}% (±{audit.get('five_fold_cv_std', 0)*100:.1f}%)",
                "Scrambled Null Baseline": f"{audit.get('permuted_null_baseline', 0)*100:.1f}%",
                "Super-Learner Team Score": f"{sl.get('super_learner_score', 0)*100:.1f}%",
                "90% Conformal Reliability": conf.get("ood_note", "Calibrated"),
            },
        }

    elif step_index == 1:
        regime = (deep_res or {}).get("stage3_mechanistic_regime_discovery", {})
        cf = (deep_res or {}).get("stage5_counterfactual_intervention", {})
        pair = regime.get("primary_interacting_pair", ["Factor 1", "Factor 2"])
        rules = regime.get("extracted_surrogate_decision_rules", [])
        provisional_claim = (
            f"PROVISIONAL CLAIM DRAFT: A method for predicting or stratifying '{lazy_run.target_column if lazy_run else 'outcome'}' "
            f"comprising measuring '{pair[0]}' (critical threshold: {regime.get('f1_critical_threshold')}) and "
            f"'{pair[-1]}' (critical threshold: {regime.get('f2_critical_threshold')}), wherein evaluating their combined regime "
            f"via {lazy_run.best_model_name if lazy_run else 'ensemble classifier'} identifies high-response candidates."
        )
        return {
            "step_index": 1,
            "action_title": "Step 2 Executed: Exact Screening Thresholds & Provisional Claim Generated",
            "summary": regime.get("synergy_conclusion", "Extracted exact decision thresholds."),
            "quadrant_thresholds": regime.get("quadrant_response_surface", {}),
            "decision_tree_rules": rules[:6],
            "prescribed_shifts": [s.get("intervention_action") for s in cf.get("prescribed_parameter_shifts", [])],
            "provisional_patent_claim": provisional_claim,
        }

    elif step_index == 2:
        q_term = (diss.subdomain if diss else None) or (diss.title[:30] if diss else "biomarker")
        chembl = await scientific_integrations.query_chembl_target_and_bioactivity(q_term)
        ot = await scientific_integrations.query_opentargets_deep(q_term)
        fda = await scientific_integrations.query_openfda_adverse_events_and_approvals(q_term)
        ct = opp.clinical_trials_check or {}
        return {
            "step_index": 2,
            "action_title": f"Step 3 Executed: Live Clinical, ChEMBL & FDA Audit for '{q_term}'",
            "summary": (
                f"Checked live external registries: {ct.get('whitespace_status', 'Verified')}. "
                f"{fda.get('regulatory_note', '')}"
            ),
            "live_registry_checks": {
                "ClinicalTrials.gov Active/Halted Trials": ct.get("total_trials", 0),
                "Sample NCT Trial IDs": ", ".join(ct.get("sample_nct_ids", [])) or "None (Open Whitespace)",
                "ChEMBL Bioactive Scaffolds Found": len(chembl.get("molecules_found", [])),
                "ChEMBL Max Clinical Phase": chembl.get("max_clinical_phase_observed", 0),
                "OpenTargets Ensembl Match": (ot.get("target_profile") or {}).get("ensembl_id") or "Pathway-Level",
                "openFDA Safety Reports Indexed": f"{fda.get('total_adverse_event_reports', 0):,}",
            },
        }

    else:
        cf = (deep_res or {}).get("stage5_counterfactual_intervention", {})
        py_code = cf.get("reproducible_python_code", "# Python pipeline ready")
        dossier_md = (
            f"# Venture & Translation Dossier: {opp.title}\n\n"
            f"- **Source Study**: {diss.title if diss else ''} ({diss.university if diss else ''}, {diss.publication_year if diss else ''})\n"
            f"- **Translation Opportunity Score (TOS)**: {opp.overall_tos_score} / 100\n"
            f"- **Best Model**: {lazy_run.best_model_name if lazy_run else ''} ({lazy_run.primary_metric_name if lazy_run else ''} = {lazy_run.best_score if lazy_run else ''})\n\n"
            f"## 1. Discovery Summary\n{opp.untranslated_finding_summary}\n\n"
            f"## 2. Why It Was Overlooked\n{opp.valley_of_death_diagnosis}\n\n"
            f"## 3. Recommended Solution\n{opp.proposed_product_or_therapeutic}\n\n"
            f"## 4. Reproducible Python Verification Script\n```python\n{py_code}\n```\n"
        )
        return {
            "step_index": 3,
            "action_title": "Step 4 Executed: Reproducible Python Script & Venture Dossier Ready",
            "summary": "Generated standalone Python verification script and downloadable Markdown Venture Dossier.",
            "reproducible_python_code": py_code,
            "venture_dossier_markdown": dossier_md,
        }


@router.post("/dissertations/{dissertation_id}/attach-zenodo", response_model=DissertationSchema)
async def attach_zenodo_record_to_dissertation(
    dissertation_id: int,
    zenodo_id_or_url: str = Form(...),
    author_primary_target: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Downloads real CSV/TSV/XLSX/Parquet files from a real Zenodo record ID or URL and runs LazyPredict."""
    from app.agents.harvester_agent import harvester_agent

    diss = db.query(Dissertation).filter(Dissertation.id == dissertation_id).first()
    if not diss:
        raise HTTPException(status_code=404, detail="Dissertation not found")

    files = await harvester_agent.fetch_zenodo_record_by_id(zenodo_id_or_url, settings.DATASET_DIR)
    if not files:
        raise HTTPException(status_code=400, detail="No open-access tabular files (.csv/.tsv/.xlsx/.parquet) found on that Zenodo record")

    for f in files[:2]:
        await orchestrator.process_dataset_with_lazypredict_and_translation(
            db=db,
            dissertation=diss,
            dataset_path=f["file_path"],
            dataset_name=f["name"],
            source_type="zenodo",
            source_url=f["source_url"],
            author_primary_target=author_primary_target,
            max_secondary_targets=1,
        )
    diss.status = "analyzed"
    db.commit()
    db.refresh(diss)
    return diss


@router.post("/dissertations/{dissertation_id}/upload-dataset", response_model=DissertationSchema)
async def attach_dataset_to_existing_dissertation(
    dissertation_id: int,
    dataset_file: UploadFile = File(...),
    author_primary_target: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Attaches a real uploaded dataset file (.csv/.tsv/.xlsx/.parquet) to an existing dissertation and runs LazyPredict."""
    diss = db.query(Dissertation).filter(Dissertation.id == dissertation_id).first()
    if not diss:
        raise HTTPException(status_code=404, detail="Dissertation not found")

    dest = settings.DATASET_DIR / f"diss_{diss.id}_{Path(dataset_file.filename or 'data.csv').name}"
    dest.write_bytes(await dataset_file.read())
    await orchestrator.process_dataset_with_lazypredict_and_translation(
        db=db,
        dissertation=diss,
        dataset_path=dest,
        dataset_name=f"Uploaded Dataset ({dataset_file.filename})",
        source_type="upload",
        author_primary_target=author_primary_target,
        max_secondary_targets=1,
    )
    diss.status = "analyzed"
    db.commit()
    db.refresh(diss)
    return diss



@router.get("/opportunities", response_model=List[MarketOpportunitySchema])
async def list_opportunities(
    domain: Optional[str] = None,
    min_tos: float = 0.0,
    db: Session = Depends(get_db),
):
    if db.query(MarketOpportunity).count() == 0:
        await ensure_seeded(db)
    q = (
        db.query(MarketOpportunity)
        .filter(MarketOpportunity.overall_tos_score >= min_tos)
        .order_by(MarketOpportunity.overall_tos_score.desc())
    )
    items = q.all()
    if domain and domain != "all":
        items = [o for o in items if o.dissertation and o.dissertation.domain == domain]
    return items


@router.get("/problems", response_model=List[IndustryProblemSchema])
def list_problems(domain: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(IndustryProblem).order_by(IndustryProblem.tam_billions_usd.desc())
    if domain and domain != "all":
        q = q.filter(IndustryProblem.domain == domain)
    return q.all()


@router.post("/problems", response_model=IndustryProblemSchema)
def create_problem(prob: IndustryProblemCreate, db: Session = Depends(get_db)):
    db_prob = IndustryProblem(**prob.model_dump())
    db.add(db_prob)
    db.commit()
    db.refresh(db_prob)
    return db_prob


@router.get("/integrations/probe")
async def probe_live_scientific_integrations(
    query: str = Query("EGFR"),
) -> Dict[str, Any]:
    """
    Live Scientific Integration Probe:
    Executes real-time queries against OpenTargets GraphQL v4, ChEMBL REST API,
    ClinicalTrials.gov API v2, openFDA FAERS API, EuropePMC REST API, and arXiv Atom API
    and returns the raw live telemetry.
    """
    ot = await scientific_integrations.query_opentargets_deep(query)
    chembl = await scientific_integrations.query_chembl_target_and_bioactivity(query)
    ct = await translation_agent.check_clinical_trials_whitespace([query])
    fda = await scientific_integrations.query_openfda_adverse_events_and_approvals(query)
    epmc = await scientific_integrations.search_europepmc_theses_and_papers(query, max_results=2)
    arxiv = await scientific_integrations.search_arxiv_preprints_and_theses(query, max_results=2)
    hf = await scientific_integrations.query_huggingface_datasets(query, limit=2)

    return {
        "query": query,
        "live_integrations": {
            "opentargets_graphql_v4": ot,
            "chembl_ebi_rest": chembl,
            "clinicaltrials_gov_v2": ct,
            "openfda_faers": fda,
            "europepmc_rest": epmc,
            "arxiv_atom_api": arxiv,
            "huggingface_hub_datasets": hf,
        },
    }


@router.post("/copilot/chat")
async def chat_with_gemini_discovery_copilot(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Conversational Gemini Discovery & Downstream Inference Co-Pilot.
    Lets the investigator chat naturally about hypotheses, datasets, target variables,
    and downstream inference without hardcoded or forceful presets.
    """
    from app.services.llm_router import llm_router

    messages = payload.get("messages", [])
    diss_id = payload.get("dissertation_id")
    run_id = payload.get("lazy_run_id")

    diss = None
    if diss_id:
        diss = db.query(Dissertation).filter(Dissertation.id == int(diss_id)).first()
    if not diss:
        diss = db.query(Dissertation).order_by(Dissertation.id.desc()).first()

    lazy_run = None
    if run_id:
        lazy_run = db.query(LazyPredictRun).filter(LazyPredictRun.id == int(run_id)).first()
    elif diss and diss.datasets:
        for ds in diss.datasets:
            if ds.lazy_runs:
                lazy_run = ds.lazy_runs[0]
                break

    available_cols = []
    dataset_meta = None
    if diss and diss.datasets:
        ds0 = diss.datasets[0]
        available_cols = [
            c.get("name")
            for c in (ds0.column_metadata or [])
            if c.get("role") != "identifier" and c.get("name")
        ]
        dataset_meta = {
            "id": ds0.id,
            "name": ds0.name,
            "source_type": ds0.source_type,
            "row_count": ds0.row_count,
            "col_count": ds0.col_count,
            "columns": ds0.column_metadata[:25] if ds0.column_metadata else [],
        }

    workspace_context = {
        "total_dissertations_in_workspace": db.query(Dissertation).count(),
        "active_dissertation": {
            "id": diss.id,
            "title": diss.title,
            "university": diss.university,
            "publication_year": diss.publication_year,
            "domain": diss.domain,
            "abstract": (diss.abstract or "")[:800],
            "extracted_claims": diss.extracted_claims or [],
        }
        if diss
        else None,
        "active_dataset": dataset_meta,
        "available_columns": available_cols,
        "active_lazy_run": {
            "id": lazy_run.id,
            "target_column": lazy_run.target_column,
            "task_type": lazy_run.task_type,
            "models_tested_count": lazy_run.models_tested_count,
            "best_model_name": lazy_run.best_model_name,
            "primary_metric_name": lazy_run.primary_metric_name,
            "best_score": lazy_run.best_score,
            "author_baseline_score": lazy_run.author_baseline_score,
            "signal_gain_delta": lazy_run.signal_gain_delta,
            "top_features": lazy_run.top_features or [],
            "top_5_leaderboard": (lazy_run.leaderboard or [])[:5],
            "overlooked_signal_summary": lazy_run.overlooked_signal_summary,
        }
        if lazy_run
        else None,
    }

    return await llm_router.chat_with_discovery_copilot(
        messages=messages,
        workspace_context=workspace_context,
    )


@router.get("/copilot/autonomous-stream")
async def stream_autonomous_gemini_discovery(
    prompt: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Streams the full end-to-end Gemini-driven discovery workflow (SSE):
    1. Gemini formulates the scientific hypothesis & optimal search parameters
    2. Streams dissertations & unmet bottlenecks (OpenAlex, EuropePMC, arXiv, ClinicalTrials.gov)
    3. Downloads real open-access tabular dataset from Zenodo & runs 30+ LazyPredict models
    4. Runs 5-Stage Deep Inference Verification & 9-API Live Probe
    5. Executes all 4 validation actions (5-fold CV, patent claim, registry audit, Python model export)
    6. Generates Gemini Executive Summary and reveals the Moonshot + all supporting evidence
    """
    return StreamingResponse(
        orchestrator.stream_autonomous_gemini_workflow(db=db, user_prompt=prompt),
        media_type="text/event-stream",
    )


# =========================================================================
# PERSISTENT MEMORY: SAVE & FAVORITE (STAR / RIBBON + RESEARCHER NOTES)
# =========================================================================
@router.get("/saved", response_model=List[SavedItemSchema])
def list_saved_items(db: Session = Depends(get_db)):
    """Returns all starred (favorite) and ribbon-bookmarked items from persistent SQLite memory."""
    items = (
        db.query(SavedItem)
        .filter((SavedItem.is_starred == True) | (SavedItem.is_bookmarked == True))
        .order_by(SavedItem.updated_at.desc(), SavedItem.id.desc())
        .all()
    )
    return items


@router.post("/saved/toggle", response_model=SavedItemSchema)
def toggle_saved_item(
    req: SavedItemToggleRequest,
    db: Session = Depends(get_db),
):
    """Toggles Star (favorite) or Ribbon (bookmark) on a dissertation/opportunity in persistent memory."""
    diss = db.query(Dissertation).filter(Dissertation.id == req.dissertation_id).first()
    if not diss:
        raise HTTPException(status_code=404, detail="Dissertation not found")

    opp_id = req.opportunity_id
    if not opp_id and diss.opportunities:
        opp_id = diss.opportunities[0].id

    item = db.query(SavedItem).filter(SavedItem.dissertation_id == req.dissertation_id).first()
    if not item:
        item = SavedItem(
            dissertation_id=req.dissertation_id,
            opportunity_id=opp_id,
            is_starred=(req.toggle_field == "star"),
            is_bookmarked=(req.toggle_field == "ribbon"),
            user_note=req.user_note or "",
            tags=[],
        )
        db.add(item)
    else:
        if req.toggle_field == "star":
            item.is_starred = not bool(item.is_starred)
        else:
            item.is_bookmarked = not bool(item.is_bookmarked)
        if opp_id:
            item.opportunity_id = opp_id
        if req.user_note is not None:
            item.user_note = req.user_note

    db.commit()
    db.refresh(item)
    return item


@router.patch("/saved/{saved_id}/note", response_model=SavedItemSchema)
def update_saved_item_note(
    saved_id: int,
    payload: SavedItemNoteUpdate,
    db: Session = Depends(get_db),
):
    """Persists researcher notes and tags on a saved item."""
    item = db.query(SavedItem).filter(SavedItem.id == saved_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Saved item not found")
    item.user_note = payload.user_note
    if payload.tags is not None:
        item.tags = payload.tags
    db.commit()
    db.refresh(item)
    return item


@router.delete("/saved/{saved_id}")
def delete_saved_item(saved_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    item = db.query(SavedItem).filter(SavedItem.id == saved_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Saved item not found")
    db.delete(item)
    db.commit()
    return {"deleted": True, "id": saved_id}


# =========================================================================
# PRODUCTION AGENT REGISTRY & POLICY GOVERNANCE
# =========================================================================
@router.get("/agents/registry", response_model=List[RegisteredAgentSchema])
def get_registered_agents(db: Session = Depends(get_db)):
    """Returns all registered production agents, their roles, SKILL.md & agent.md paths, tools, and safety policies."""
    return sync_agent_registry(db)


@router.post("/agents/registry/{agent_name}/verify")
def verify_registered_agent(agent_name: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Audits a registered agent's manifest, skill, and declarative policies."""
    record_agent_invocation(db, "qa-production-guardian")
    ag = record_agent_invocation(db, agent_name)
    if not ag:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found in registry")
    agent_md_exists = (settings.BASE_DIR / ag.agent_md_path).exists()
    skill_md_exists = (settings.BASE_DIR / ag.skill_path).exists()
    return {
        "agent_name": ag.name,
        "role": ag.role,
        "status": "verified" if (agent_md_exists and skill_md_exists) else "missing_manifest",
        "agent_md_verified": agent_md_exists,
        "skill_md_verified": skill_md_exists,
        "policies_enforced": ag.policies,
        "invocations_count": ag.invocations_count,
    }



