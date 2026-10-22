from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.agents.data_scientist import data_scientist_agent
from app.agents.deep_inference_agent import deep_inference_agent
from app.agents.orchestrator import orchestrator
from app.agents.pdf_extractor import pdf_extractor_agent
from app.agents.translation_agent import translation_agent
from app.config import settings
from app.database import get_db
from app.models import DatasetArtifact, Dissertation, IndustryProblem, LazyPredictRun, MarketOpportunity
from app.schemas import (
    CustomLazyPredictRequest,
    DatasetArtifactSchema,
    DissertationSchema,
    HarvestQueryRequest,
    IndustryProblemCreate,
    IndustryProblemSchema,
    LazyPredictRunSchema,
    MarketOpportunitySchema,
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
        },
    }


@router.get("/dissertations", response_model=List[DissertationSchema])
def list_dissertations(
    domain: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Dissertation).order_by(Dissertation.scientific_rigor_score.desc(), Dissertation.id.desc())
    if domain and domain != "all":
        q = q.filter(Dissertation.domain == domain)
    items = q.all()
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
    Live harvests Top-Science dissertations from OpenAlex/EuropePMC/arXiv, resolves or synthesizes
    empirical datasets, runs 30+ LazyPredict models (Primary + Rotated Secondary targets),
    and generates Giant-Impact Market Translation Dossiers.
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


@router.post("/upload-dissertation", response_model=DissertationSchema)
async def upload_dissertation_and_dataset(
    title: str = Form(...),
    authors: str = Form("Doctoral Researcher"),
    university: str = Form("Tier-1 Research University"),
    publication_year: int = Form(2024),
    domain: str = Form("biomedical"),
    abstract: str = Form(""),
    author_primary_target: Optional[str] = Form(None),
    author_baseline_score: Optional[float] = Form(None),
    pdf_file: Optional[UploadFile] = File(None),
    dataset_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """
    Ingests a user-uploaded dissertation PDF and/or raw dataset (.csv, .tsv, .xlsx, .parquet).
    If only a PDF is uploaded, automatically extracts embedded tables from the PDF using `pdfplumber`
    and runs LazyPredict + Market Translation on them.
    """
    pdf_saved_path = None
    extracted_meta = {
        "extracted_claims": ["Empirical findings extracted from uploaded dissertation."],
        "data_availability_links": [],
        "author_statistical_method": "Classical Linear / Parametric Modeling",
    }

    slug = f"upload_{abs(hash(title)) % 100000}"

    if pdf_file and pdf_file.filename:
        pdf_dest = settings.PDF_DIR / f"{slug}_{Path(pdf_file.filename).name}"
        pdf_dest.write_bytes(await pdf_file.read())
        pdf_saved_path = str(pdf_dest)
        extracted_meta = pdf_extractor_agent.extract_text_and_metadata(pdf_dest)
        if not abstract:
            abstract = extracted_meta.get("text_preview", "")[:1500]

    diss = Dissertation(
        title=title,
        authors=authors,
        university=university,
        department="Graduate Research Division",
        publication_year=publication_year,
        domain=domain,
        subdomain="Custom Uploaded Dissertation",
        source_repo="Direct Researcher Upload",
        pdf_path=pdf_saved_path,
        institution_tier="Tier-1 Global Research",
        scientific_rigor_score=0.95,
        citation_velocity=3.5,
        abstract=abstract or f"Uploaded doctoral dissertation: {title}",
        author_primary_hypothesis=f"Author's primary thesis evaluation on `{author_primary_target or 'primary outcome'}`.",
        author_Statistical_method=extracted_meta["author_statistical_method"],
        author_primary_target=author_primary_target,
        author_reported_baseline_metric="ROC-AUC / R-Squared",
        author_reported_baseline_score=author_baseline_score,
        extracted_claims=extracted_meta["extracted_claims"],
        data_availability_links=extracted_meta["data_availability_links"],
        key_entities={"targets": [title.split()[0]]},
        status="analyzed",
    )
    db.add(diss)
    db.commit()
    db.refresh(diss)

    # 1. Process uploaded dataset file if provided
    if dataset_file and dataset_file.filename:
        ds_dest = settings.DATASET_DIR / f"{slug}_{Path(dataset_file.filename).name}"
        ds_dest.write_bytes(await dataset_file.read())
        await orchestrator.process_dataset_with_lazypredict_and_translation(
            db=db,
            dissertation=diss,
            dataset_path=ds_dest,
            dataset_name=f"Uploaded Dataset ({dataset_file.filename})",
            source_type="upload",
            author_primary_target=author_primary_target,
            author_baseline_score=author_baseline_score,
            max_secondary_targets=1,
        )
    elif pdf_saved_path:
        # 2. Extract embedded quantitative tables directly from the PDF
        pdf_tables = pdf_extractor_agent.extract_tables_from_pdf(pdf_saved_path, slug)
        for tbl in pdf_tables[:2]:
            await orchestrator.process_dataset_with_lazypredict_and_translation(
                db=db,
                dissertation=diss,
                dataset_path=tbl["file_path"],
                dataset_name=tbl["name"],
                source_type="pdf_table",
                source_url=tbl["source_url"],
                author_primary_target=author_primary_target,
                author_baseline_score=author_baseline_score,
                max_secondary_targets=1,
            )

    db.refresh(diss)
    return diss


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
def list_opportunities(
    domain: Optional[str] = None,
    min_tos: float = 0.0,
    db: Session = Depends(get_db),
):
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
