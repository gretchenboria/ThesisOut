---
name: clinical-patent-translation
description: Use this skill when auditing empirical discoveries against 9 live external scientific registries (ClinicalTrials.gov, OpenTargets, ChEMBL, openFDA, EuropePMC, arXiv, HuggingFace) and synthesizing Translation Opportunity Scores (TOS) with Gemini 3.8 Flash.
---

# Clinical, Bioactivity & Patent Whitespace Translation Skill

## Overview
Connects validated dataset signals to unmet clinical and engineering bottlenecks and audits commercial whitespace across live registries.

## Execution Steps
1. Query `live_problem_harvester` (OpenAlex Grand Challenges + ClinicalTrials.gov `TERMINATED|WITHDRAWN` studies) to match the discovery to a real industry bottleneck.
2. Query live scientific registries in parallel (`ClinicalTrials.gov API v2`, `OpenTargets GraphQL v4`, `ChEMBL REST API`, `openFDA FAERS`, `HuggingFace Hub`).
3. Compute the composite Translation Opportunity Score (`overall_tos_score` 0-100) from empirical `data_signal_score`, `commercial_whitespace_score`, `giant_impact_tam_score`, and `scientific_rigor_score`.
4. Synthesize zero-fluff commercial opportunity dossiers and provisional patent claims via `llm_router` (`Gemini 3.8 Flash`).
