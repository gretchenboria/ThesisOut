"""
Zero hardcoded seed data.
All dissertations, datasets, LazyPredict runs, and unmet industry/clinical challenges
are dynamically streamed from live external APIs (OpenAlex, Zenodo, ClinicalTrials.gov, OpenTargets)
or uploaded directly by the researcher.
"""
from sqlalchemy.orm import Session


async def ensure_seeded(db: Session) -> None:
    """No-op: Zero hardcoded records."""
    return None
