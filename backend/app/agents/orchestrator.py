"""
Multi-Agent Orchestrator for ThesisOut (100% Real Data Only).
Coordinates:
1. Real Dissertation Ingestion (via OpenAlex Live API or Direct PDF + CSV/TSV/XLSX/Parquet Upload)
2. Real Dataset Resolution from Zenodo API or Embedded PDF Table Extraction (`pdfplumber`)
3. Dataset Profiling & Multi-Target LazyPredict Sweeps (Primary Target + Secondary Rotated Targets)
4. Market Translation & Giant-Impact Reverse Matching (via Gemini 4 Argon / Opus 5.5 + ClinicalTrials.gov & OpenTargets APIs)
"""
import logging
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session

from app.agents.data_scientist import data_scientist_agent
from app.agents.harvester_agent import harvester_agent
from app.agents.pdf_extractor import pdf_extractor_agent
from app.agents.translation_agent import translation_agent
from app.config import settings
from app.models import DatasetArtifact, Dissertation, IndustryProblem, LazyPredictRun, MarketOpportunity

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    async def process_dataset_with_lazypredict_and_translation(
        self,
        db: Session,
        dissertation: Dissertation,
        dataset_path: str | Path,
        dataset_name: str,
        source_type: str = "upload",
        source_url: Optional[str] = None,
        author_primary_target: Optional[str] = None,
        author_baseline_score: Optional[float] = None,
        author_discussed_features: Optional[List[str]] = None,
        max_secondary_targets: int = 1,
    ) -> Optional[DatasetArtifact]:
        """
        Loads a real dataset from disk, profiles its schema, runs LazyPredict on the primary target
        plus rotated secondary targets, and generates Market Translation Opportunities.
        """
        df = data_scientist_agent.load_dataframe(dataset_path)
        if len(df) < 10 or len(df.columns) < 2:
            logger.warning("Dataset %s has insufficient rows/columns (%s) for LazyPredict", dataset_name, df.shape)
            return None

        profile = data_scientist_agent.profile_dataset(df)

        ds_artifact = DatasetArtifact(
            dissertation_id=dissertation.id,
            name=dataset_name,
            source_type=source_type,
            source_url=source_url,
            file_path=str(dataset_path),
            row_count=profile["row_count"],
            col_count=profile["col_count"],
            column_metadata=profile["column_metadata"],
            sample_rows=profile["sample_rows"],
        )
        db.add(ds_artifact)
        db.commit()
        db.refresh(ds_artifact)

        targets = data_scientist_agent.select_targets_for_sweep(
            df,
            author_primary_target=author_primary_target or dissertation.author_primary_target,
            max_secondary_targets=max_secondary_targets,
        )

        problems = db.query(IndustryProblem).all()

        for t_info in targets:
            t_col = t_info["target_column"]
            is_primary = t_info["is_author_primary_target"]
            baseline_to_pass = (
                (author_baseline_score or dissertation.author_reported_baseline_score)
                if is_primary
                else None
            )

            try:
                exp_res = data_scientist_agent.run_lazypredict_experiment(
                    df=df,
                    target_column=t_col,
                    is_author_primary_target=is_primary,
                    target_rationale=t_info["target_rationale"],
                    author_baseline_score=baseline_to_pass,
                    author_discussed_features=author_discussed_features,
                )
            except Exception as exc:
                logger.warning("LazyPredict skipped target %s on %s: %s", t_col, dataset_name, exc)
                continue

            lazy_run = LazyPredictRun(
                dataset_id=ds_artifact.id,
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

            opp_data = await translation_agent.synthesize_opportunity(
                dissertation=dissertation,
                lazy_run=lazy_run,
                problems=problems,
            )
            opp = MarketOpportunity(**opp_data)
            db.add(opp)
            db.commit()

        db.refresh(ds_artifact)
        return ds_artifact

    async def harvest_and_analyze_live(
        self,
        db: Session,
        query: str,
        domain: str = "biomedical",
        min_year: int = 2018,
        max_results: int = 3,
        top_science_only: bool = True,
    ) -> List[Dissertation]:
        """
        Searches OpenAlex for real Top-Science dissertations matching `query` and searches Zenodo's live API
        for real open-access CSV datasets associated with the dissertation or query.
        Never generates synthetic data.
        """
        harvested_items = await harvester_agent.search_openalex_dissertations(
            query=query,
            domain=domain,
            min_year=min_year,
            max_results=max_results,
            top_science_only=top_science_only,
        )
        created_dissertations: List[Dissertation] = []

        # Also query Zenodo once for real open-access CSV datasets matching this research topic
        real_zenodo_files = await harvester_agent.fetch_real_zenodo_datasets(
            query=query,
            target_dir=settings.DATASET_DIR,
            max_files=1,
        )

        for idx, item in enumerate(harvested_items):
            ext_id = item.get("external_id")
            if ext_id:
                existing = db.query(Dissertation).filter(Dissertation.external_id == ext_id).first()
                if existing:
                    created_dissertations.append(existing)
                    continue

            links = pdf_extractor_agent.extract_repository_links(item.get("abstract") or "")
            claims = pdf_extractor_agent.extract_key_claims(item.get("abstract") or "")
            stat_method = pdf_extractor_agent.detect_author_statistical_method(item.get("abstract") or "")

            diss = Dissertation(
                external_id=ext_id,
                title=item["title"],
                authors=item["authors"],
                university=item["university"],
                publication_year=item["publication_year"],
                domain=domain,
                subdomain=query,
                source_repo=item["source_repo"],
                source_url=item["source_url"],
                institution_tier=item["institution_tier"],
                scientific_rigor_score=item["scientific_rigor_score"],
                citation_velocity=item["citation_velocity"],
                abstract=item["abstract"],
                author_Statistical_method=stat_method,
                extracted_claims=claims,
                data_availability_links=links,
                key_entities={"targets": [w for w in query.split() if len(w) > 3][:3]},
                status="ingested",
            )
            db.add(diss)
            db.commit()
            db.refresh(diss)

            # If real Zenodo datasets were found for this query (attach to the top dissertation), run LazyPredict
            if idx == 0 and real_zenodo_files:
                zf = real_zenodo_files[0]
                ds_art = await self.process_dataset_with_lazypredict_and_translation(
                    db=db,
                    dissertation=diss,
                    dataset_path=zf["file_path"],
                    dataset_name=zf["name"],
                    source_type="zenodo",
                    source_url=zf["source_url"],
                    max_secondary_targets=1,
                )
                if ds_art:
                    diss.status = "analyzed"
                    db.commit()

            db.refresh(diss)
            created_dissertations.append(diss)

        return created_dissertations


orchestrator = PipelineOrchestrator()
