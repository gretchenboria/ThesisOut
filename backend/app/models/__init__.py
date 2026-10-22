from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship, synonym
from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Dissertation(Base):
    __tablename__ = "dissertations"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, index=True, nullable=True)
    title = Column(String(600), nullable=False)
    authors = Column(String(400), nullable=False)
    university = Column(String(300), nullable=False)
    department = Column(String(250), nullable=True)
    publication_year = Column(Integer, nullable=False)
    domain = Column(String(100), nullable=False)  # "biomedical" | "cs_ai"
    subdomain = Column(String(200), nullable=True)
    source_repo = Column(String(100), default="OpenAlex / ETD")
    source_url = Column(String(600), nullable=True)
    pdf_path = Column(String(600), nullable=True)

    # Top-Science Quality Metrics
    institution_tier = Column(String(50), default="Tier-1 Global Research")
    scientific_rigor_score = Column(Float, default=0.90)  # 0.0 - 1.0
    citation_velocity = Column(Float, default=0.0)

    abstract = Column(Text, nullable=False)
    author_primary_hypothesis = Column(Text, nullable=True)
    author_Statistical_method = Column(String(255), nullable=True)
    author_statistical_method = synonym("author_Statistical_method")
    author_primary_target = Column(String(150), nullable=True)
    author_reported_baseline_metric = Column(String(100), nullable=True)
    author_reported_baseline_score = Column(Float, nullable=True)
    extracted_claims = Column(JSON, default=list)
    data_availability_links = Column(JSON, default=list)
    key_entities = Column(JSON, default=dict)  # targets, genes, compounds, algorithms, architectures

    status = Column(String(50), default="analyzed")  # ingested, data_extracted, analyzed
    created_at = Column(DateTime, default=utc_now)

    datasets = relationship("DatasetArtifact", back_populates="dissertation", cascade="all, delete-orphan")
    opportunities = relationship("MarketOpportunity", back_populates="dissertation", cascade="all, delete-orphan")


class DatasetArtifact(Base):
    __tablename__ = "dataset_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    dissertation_id = Column(Integer, ForeignKey("dissertations.id"), nullable=False)
    name = Column(String(300), nullable=False)
    source_type = Column(String(100), nullable=False)  # zenodo, figshare, github, geo, pdf_table, upload
    source_url = Column(String(600), nullable=True)
    file_path = Column(String(600), nullable=False)
    row_count = Column(Integer, nullable=False)
    col_count = Column(Integer, nullable=False)
    column_metadata = Column(JSON, default=list)  # [{name, dtype, missing_pct, unique_count, role}]
    sample_rows = Column(JSON, default=list)
    created_at = Column(DateTime, default=utc_now)

    dissertation = relationship("Dissertation", back_populates="datasets")
    lazy_runs = relationship("LazyPredictRun", back_populates="dataset", cascade="all, delete-orphan")


class LazyPredictRun(Base):
    __tablename__ = "lazypredict_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("dataset_artifacts.id"), nullable=False)
    target_column = Column(String(200), nullable=False)
    task_type = Column(String(50), nullable=False)  # classification | regression
    is_author_primary_target = Column(Boolean, default=True)
    target_rationale = Column(Text, nullable=True)

    models_tested_count = Column(Integer, default=0)
    best_model_name = Column(String(150), nullable=False)
    primary_metric_name = Column(String(80), nullable=False)  # ROC-AUC, Balanced Accuracy, R-Squared
    best_score = Column(Float, nullable=False)
    author_baseline_score = Column(Float, nullable=True)
    signal_gain_delta = Column(Float, default=0.0)  # Improvement over author's classical method

    leaderboard = Column(JSON, default=list)  # Full LazyPredict table (25-35 models)
    top_features = Column(JSON, default=list)  # [{feature, importance, discussed_by_author: bool, discovery_note}]
    overlooked_signal_summary = Column(Text, nullable=True)
    execution_time_sec = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utc_now)

    dataset = relationship("DatasetArtifact", back_populates="lazy_runs")
    opportunities = relationship("MarketOpportunity", back_populates="lazy_run")


class IndustryProblem(Base):
    __tablename__ = "industry_problems"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(60), unique=True, index=True, nullable=False)
    title = Column(String(400), nullable=False)
    domain = Column(String(100), nullable=False)  # biomedical | cs_ai
    category = Column(String(200), nullable=False)
    sponsor_or_source = Column(String(250), nullable=False)  # ARPA-H, DARPA, Grand Challenge, Top Pharma R&D, Hyperscaler
    tam_billions_usd = Column(Float, nullable=False)  # e.g., 45.0 ($45B)
    impact_scale = Column(String(100), default="Civilizational / Giant Impact")
    urgency_score = Column(Float, default=0.95)
    description = Column(Text, nullable=False)
    technical_bottleneck = Column(Text, nullable=False)
    desired_solution_profile = Column(Text, nullable=False)
    keywords = Column(JSON, default=list)
    created_at = Column(DateTime, default=utc_now)


class MarketOpportunity(Base):
    __tablename__ = "market_opportunities"

    id = Column(Integer, primary_key=True, index=True)
    dissertation_id = Column(Integer, ForeignKey("dissertations.id"), nullable=False)
    lazy_run_id = Column(Integer, ForeignKey("lazypredict_runs.id"), nullable=True)
    matched_problem_id = Column(Integer, ForeignKey("industry_problems.id"), nullable=True)

    title = Column(String(500), nullable=False)
    opportunity_type = Column(String(120), nullable=False)  # "Hidden Data Signal", "Uncommercialized Mechanism", "Cross-Domain Transfer"
    untranslated_finding_summary = Column(Text, nullable=False)
    valley_of_death_category = Column(String(200), nullable=False)
    valley_of_death_diagnosis = Column(Text, nullable=False)

    # Commercial & Scientific Validation Checks
    clinical_trials_check = Column(JSON, default=dict)  # {total_trials, phase3_count, whitespace_status, notes}
    opentargets_chembl_check = Column(JSON, default=dict)  # {target_tractability, known_drugs, novel_mechanism_flag}
    patent_and_market_check = Column(JSON, default=dict)  # {freedom_to_operate, startup_competition, commercial_status}

    # Scoring (0-100 scale)
    data_signal_score = Column(Float, nullable=False)
    scientific_rigor_score = Column(Float, nullable=False)
    commercial_whitespace_score = Column(Float, nullable=False)
    giant_impact_tam_score = Column(Float, nullable=False)
    overall_tos_score = Column(Float, nullable=False, index=True)

    # Actionable Venture & Translation Blueprint
    proposed_product_or_therapeutic = Column(Text, nullable=False)
    validation_roadmap = Column(JSON, default=list)
    llm_model_used = Column(String(120), default="gemini-4-argon")
    created_at = Column(DateTime, default=utc_now)

    dissertation = relationship("Dissertation", back_populates="opportunities")
    lazy_run = relationship("LazyPredictRun", back_populates="opportunities")
    matched_problem = relationship("IndustryProblem")


class SavedItem(Base):
    __tablename__ = "saved_items"

    id = Column(Integer, primary_key=True, index=True)
    dissertation_id = Column(Integer, ForeignKey("dissertations.id"), nullable=False, index=True)
    opportunity_id = Column(Integer, ForeignKey("market_opportunities.id"), nullable=True, index=True)
    is_starred = Column(Boolean, default=False, nullable=False)      # Star / Favorite
    is_bookmarked = Column(Boolean, default=False, nullable=False)   # Ribbon / Saved to Memory
    user_note = Column(Text, default="", nullable=False)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    dissertation = relationship("Dissertation")
    opportunity = relationship("MarketOpportunity")


class RegisteredAgent(Base):
    __tablename__ = "agent_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), unique=True, index=True, nullable=False)
    role = Column(String(250), nullable=False)
    description = Column(Text, nullable=False)
    skill_name = Column(String(150), nullable=False)
    skill_path = Column(String(500), nullable=False)
    agent_md_path = Column(String(500), nullable=False)
    tools_allowed = Column(JSON, default=list)
    policies = Column(JSON, default=list)
    status = Column(String(50), default="active")
    invocations_count = Column(Integer, default=0)
    last_invoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now)

