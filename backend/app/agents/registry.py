"""
Production Multi-Agent Registry & Policy Governance Engine for ThesisOut.
Registers and governs all specialized agents defined in `.agents/agents/*/agent.md`
and their corresponding skills in `.agents/skills/*/SKILL.md`.
"""
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.models import RegisteredAgent

logger = logging.getLogger(__name__)

PRODUCTION_AGENT_SPECS: List[Dict[str, Any]] = [
    {
        "name": "harvester-agent",
        "role": "Corpus & Open-Access Dataset Harvester Agent",
        "description": "Queries live OpenAlex, EuropePMC, arXiv, ClinicalTrials.gov, and Zenodo REST APIs for verified doctoral dissertations and real empirical tabular datasets.",
        "skill_name": "thesis-corpus-harvester",
        "skill_path": ".agents/skills/thesis-corpus-harvester/SKILL.md",
        "agent_md_path": ".agents/agents/harvester-agent/agent.md",
        "tools_allowed": ["view_file", "read_url_content", "search_web", "send_message"],
        "policies": [
            {"rule": "deny", "target": "run_command", "reason": "External harvester is strictly read-only for shell execution."},
            {"rule": "enforce", "target": "repository_provenance_only", "reason": "Requires verified DOI or repository record URL on all ingested datasets."},
            {"rule": "enforce", "target": "provenance_url_required", "reason": "Requires exact external_id and source_url on all records."},
        ],
    },
    {
        "name": "pdf-extractor-agent",
        "role": "PDF Bibliographic, Claim, DOI & Tabular Extraction Agent",
        "description": "Parses uploaded dissertation PDFs using pypdf and pdfplumber (grid and text-alignment strategies) to extract metadata, statistical baselines, DOIs, and quantitative tables.",
        "skill_name": "pdf-table-extractor",
        "skill_path": ".agents/skills/pdf-table-extractor/SKILL.md",
        "agent_md_path": ".agents/agents/pdf-extractor-agent/agent.md",
        "tools_allowed": ["view_file", "write_file", "send_message"],
        "policies": [
            {"rule": "deny", "target": "run_command", "reason": "PDF parser cannot execute arbitrary shell commands."},
            {"rule": "workspace_only", "target": "storage/", "reason": "File writes restricted to storage/pdfs/ and storage/datasets/."},
            {"rule": "deny", "target": "np_random_fallback", "reason": "Never synthesize fake fallback tables when a PDF lacks tabular data."},
        ],
    },
    {
        "name": "ml-rigor-scientist",
        "role": "Principal ML & Biostatistical Rigor Scientist Agent",
        "description": "Executes leak-free 30-model out-of-fold LazyPredict sweeps with inside-fold imputation/scaling, single/multi-feature proxy pruning, and 95% confidence intervals.",
        "skill_name": "lazypredict-oof-benchmark",
        "skill_path": ".agents/skills/lazypredict-oof-benchmark/SKILL.md",
        "agent_md_path": ".agents/agents/ml-rigor-scientist/agent.md",
        "tools_allowed": ["view_file", "write_file", "edit_file", "run_command", "send_message"],
        "policies": [
            {"rule": "enforce", "target": "inside_fold_preprocessing_only", "reason": "SimpleImputer, OrdinalEncoder, and StandardScaler fit strictly on X_train inside each CV fold."},
            {"rule": "enforce", "target": "prune_target_leakage_proxies", "reason": "Strips IDs, boolean complements, single-feature tautologies (|r|>=0.88), and multi-feature post-hoc composites."},
            {"rule": "enforce", "target": "live_oof_evaluation", "reason": "All 30 model scores and 95% CIs are computed from out-of-fold cross-validation predictions."},
        ],
    },
    {
        "name": "deep-inference-auditor",
        "role": "Stage-2 Deep Inference & Conformal Verification Agent",
        "description": "Runs 5-fold permutation null tests, top-feature ablation audits, Level-2 Super-Learner stacking, surrogate decision-tree rule extraction, and 90% split-conformal calibration.",
        "skill_name": "deep-conformal-escalation",
        "skill_path": ".agents/skills/deep-conformal-escalation/SKILL.md",
        "agent_md_path": ".agents/agents/deep-inference-auditor/agent.md",
        "tools_allowed": ["view_file", "write_file", "run_command", "send_message"],
        "policies": [
            {"rule": "enforce", "target": "permutation_null_test", "reason": "Requires scrambled target permutation baseline to verify empirical signal separation."},
            {"rule": "enforce", "target": "train_split_thresholds_only", "reason": "Surrogate decision rules and 2x2 quadrant medians fit strictly on X_train."},
            {"rule": "enforce", "target": "reproducible_pipeline_export", "reason": "Generates standalone leak-free Python verification code."},
        ],
    },
    {
        "name": "translation-whitespace-agent",
        "role": "Clinical, Bioactivity, Patent Whitespace & Venture Synthesis Agent",
        "description": "Cross-references empirical discoveries against 9 live scientific APIs (ClinicalTrials.gov, OpenTargets, ChEMBL, openFDA, EuropePMC, arXiv, HuggingFace) and synthesizes TOS dossiers via Gemini 3.8 Flash.",
        "skill_name": "clinical-patent-translation",
        "skill_path": ".agents/skills/clinical-patent-translation/SKILL.md",
        "agent_md_path": ".agents/agents/translation-whitespace-agent/agent.md",
        "tools_allowed": ["view_file", "read_url_content", "search_web", "send_message"],
        "policies": [
            {"rule": "deny", "target": "run_command", "reason": "Translation synthesizer operates strictly over HTTP APIs and structured LLM routing."},
            {"rule": "enforce", "target": "honest_signal_scoring", "reason": "Computes TOS (0-100) from real out-of-fold gain, registry whitespace, and rigor."},
            {"rule": "enforce", "target": "zero_markdown_noise", "reason": "Strips raw asterisks, backticks, and filler prose from UI fields."},
        ],
    },
    {
        "name": "qa-production-guardian",
        "role": "Pipeline Integrity & Statistical Audit Agent",
        "description": "Audits database schema consistency, out-of-fold validation diagnostics, shortlist persistence, and API endpoint health.",
        "skill_name": "production-qa-audit",
        "skill_path": ".agents/skills/production-qa-audit/SKILL.md",
        "agent_md_path": ".agents/agents/qa-production-guardian/agent.md",
        "tools_allowed": ["view_file", "write_file", "edit_file", "run_command", "send_message"],
        "policies": [
            {"rule": "enforce", "target": "schema_and_pipeline_integrity", "reason": "Audits relational foreign keys, dataset artifacts, and out-of-fold evaluation metrics."},
            {"rule": "enforce", "target": "shortlist_persistence_check", "reason": "Verifies saved shortlist items and researcher notes persist across sessions."},
            {"rule": "workspace_only", "target": "/Users/dr.gretchenboria/thesisout", "reason": "Restricts all edits and test runs to the project workspace."},
        ],
    },
]


def sync_agent_registry(db: Session) -> List[RegisteredAgent]:
    """
    Synchronizes the 6 production agents with the persistent `agent_registry` table in SQLite
    and verifies their `.agents/agents/*/agent.md` and `.agents/skills/*/SKILL.md` files.
    """
    registered: List[RegisteredAgent] = []
    base_dir: Path = settings.BASE_DIR

    for spec in PRODUCTION_AGENT_SPECS:
        agent_md_full = base_dir / spec["agent_md_path"]
        skill_md_full = base_dir / spec["skill_path"]
        files_verified = agent_md_full.exists() and skill_md_full.exists()
        status_val = "active" if files_verified else "manifest_missing"

        existing = db.query(RegisteredAgent).filter(RegisteredAgent.name == spec["name"]).first()
        if existing:
            existing.role = spec["role"]
            existing.description = spec["description"]
            existing.skill_name = spec["skill_name"]
            existing.skill_path = spec["skill_path"]
            existing.agent_md_path = spec["agent_md_path"]
            existing.tools_allowed = spec["tools_allowed"]
            existing.policies = spec["policies"]
            existing.status = status_val
            db.commit()
            db.refresh(existing)
            registered.append(existing)
        else:
            new_agent = RegisteredAgent(
                name=spec["name"],
                role=spec["role"],
                description=spec["description"],
                skill_name=spec["skill_name"],
                skill_path=spec["skill_path"],
                agent_md_path=spec["agent_md_path"],
                tools_allowed=spec["tools_allowed"],
                policies=spec["policies"],
                status=status_val,
                invocations_count=1,
                last_invoked_at=datetime.now(timezone.utc),
            )
            db.add(new_agent)
            db.commit()
            db.refresh(new_agent)
            registered.append(new_agent)

    return registered


def record_agent_invocation(db: Session, agent_name: str) -> Optional[RegisteredAgent]:
    """Increments invocation count and updates last_invoked_at timestamp for an agent in the registry."""
    ag = db.query(RegisteredAgent).filter(RegisteredAgent.name == agent_name).first()
    if not ag:
        sync_agent_registry(db)
        ag = db.query(RegisteredAgent).filter(RegisteredAgent.name == agent_name).first()
    if ag:
        ag.invocations_count = int(ag.invocations_count or 0) + 1
        ag.last_invoked_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(ag)
    return ag
