from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class LazyPredictRunSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    target_column: str
    task_type: str
    is_author_primary_target: bool
    target_rationale: Optional[str] = None
    models_tested_count: int
    best_model_name: str
    primary_metric_name: str
    best_score: float
    author_baseline_score: Optional[float] = None
    signal_gain_delta: float
    leaderboard: List[Dict[str, Any]]
    top_features: List[Dict[str, Any]]
    overlooked_signal_summary: Optional[str] = None
    execution_time_sec: float
    created_at: datetime


class DatasetArtifactSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dissertation_id: int
    name: str
    source_type: str
    source_url: Optional[str] = None
    file_path: str
    row_count: int
    col_count: int
    column_metadata: List[Dict[str, Any]]
    sample_rows: List[Dict[str, Any]]
    created_at: datetime
    lazy_runs: List[LazyPredictRunSchema] = []


class IndustryProblemSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str
    title: str
    domain: str
    category: str
    sponsor_or_source: str
    tam_billions_usd: float
    impact_scale: str
    urgency_score: float
    description: str
    technical_bottleneck: str
    desired_solution_profile: str
    keywords: List[str]
    created_at: datetime


class IndustryProblemCreate(BaseModel):
    code: str
    title: str
    domain: str
    category: str
    sponsor_or_source: str
    tam_billions_usd: float
    impact_scale: str = "Civilizational / Giant Impact"
    urgency_score: float = 0.95
    description: str
    technical_bottleneck: str
    desired_solution_profile: str
    keywords: List[str] = []


class MarketOpportunitySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dissertation_id: int
    lazy_run_id: Optional[int] = None
    matched_problem_id: Optional[int] = None
    title: str
    opportunity_type: str
    untranslated_finding_summary: str
    valley_of_death_category: str
    valley_of_death_diagnosis: str
    clinical_trials_check: Dict[str, Any]
    opentargets_chembl_check: Dict[str, Any]
    patent_and_market_check: Dict[str, Any]
    data_signal_score: float
    scientific_rigor_score: float
    commercial_whitespace_score: float
    giant_impact_tam_score: float
    overall_tos_score: float
    proposed_product_or_therapeutic: str
    validation_roadmap: List[str]
    llm_model_used: str
    created_at: datetime
    matched_problem: Optional[IndustryProblemSchema] = None


class DissertationSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    external_id: Optional[str] = None
    title: str
    authors: str
    university: str
    department: Optional[str] = None
    publication_year: int
    domain: str
    subdomain: Optional[str] = None
    source_repo: str
    source_url: Optional[str] = None
    pdf_path: Optional[str] = None
    institution_tier: str
    scientific_rigor_score: float
    citation_velocity: float
    abstract: str
    author_primary_hypothesis: Optional[str] = None
    author_Statistical_method: Optional[str] = None
    author_primary_target: Optional[str] = None
    author_reported_baseline_metric: Optional[str] = None
    author_reported_baseline_score: Optional[float] = None
    extracted_claims: List[str]
    data_availability_links: List[Dict[str, str]]
    key_entities: Dict[str, Any]
    status: str
    created_at: datetime
    datasets: List[DatasetArtifactSchema] = []
    opportunities: List[MarketOpportunitySchema] = []


class HarvestQueryRequest(BaseModel):
    query: str
    domain: str = "biomedical"
    min_year: int = 2018
    max_results: int = 5
    top_science_only: bool = True
    auto_run_lazypredict: bool = True


class CustomLazyPredictRequest(BaseModel):
    dataset_id: int
    target_column: str
    task_type: Optional[str] = None  # Auto-detected if None
    is_author_primary_target: bool = False
    target_rationale: Optional[str] = None
