from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ThesisOut: Top-Science Agentic Dissertation & Market Translation Engine"
    VERSION: str = "1.0.0"

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STORAGE_DIR: Path = BASE_DIR / "storage"
    PDF_DIR: Path = STORAGE_DIR / "pdfs"
    DATASET_DIR: Path = STORAGE_DIR / "datasets"
    DB_PATH: Path = STORAGE_DIR / "thesisout.db"

    # LLM Router Configuration: Gemini 4 Argon (Primary) -> Claude Opus 5.5 (Fallback)
    PRIMARY_LLM_MODEL: str = "gemini-4-argon"
    FALLBACK_LLM_MODEL: str = "claude-opus-5-5"
    GEMINI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Top-Science Quality Gates
    MIN_INSTITUTION_TIER_SCORE: float = 0.75  # Strict R1 / Top Global Research Institution filter
    MIN_STATISTICAL_POWER_ROWS: int = 25
    MAX_LAZYPREDICT_ROWS: int = 10000
    MAX_ROTATED_TARGETS: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.PDF_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
