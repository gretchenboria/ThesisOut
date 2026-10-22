# ThesisOut Project Briefing (Production Architecture & Agentic Governance)

## 1. Mission & System Overview
ThesisOut is a production multi-agent scientific discovery platform that extracts empirical tabular datasets from doctoral dissertations and open scientific repositories (`OpenAlex`, `EuropePMC`, `arXiv`, `Zenodo`), runs leak-free 30-model out-of-fold cross-validated benchmarks (`LazyPredict` + `Scikit-Learn` + `XGBoost` + `LightGBM` + `CatBoost`), verifies discoveries with 5-stage conformal and permutation null tests, and audits commercial/clinical whitespace across 9 live scientific registries using `Gemini 3.8 Flash`.

## 2. Production Data Integrity (Zero Dummy Data)
- **100% Real Empirical Data**: All datasets stored in `storage/datasets/` and indexed in `storage/thesisout.db` are real open-access scientific CSV/TSV datasets downloaded from Zenodo REST API (`https://zenodo.org/api/records`) or extracted from uploaded PDFs/CSVs.
- **100% Real Model Execution**: Every leaderboard in `lazypredict_runs` is computed live by `DataScientistAgent.run_lazypredict_experiment` using out-of-fold cross-validation (`StratifiedKFold`, `RepeatedStratifiedKFold`, `KFold`, `RepeatedKFold`).

## 3. Registered Multi-Agent Architecture & Policies
Managed via `./agent-cli` (`agy` compatible) and persisted in the `agent_registry` SQLite table:

| Agent ID | Role | Primary Skill | Active Policies |
| :--- | :--- | :--- | :--- |
| `harvester-agent` | Corpus & Open-Access Dataset Harvester | `thesis-corpus-harvester` | `deny("run_command")`, `allow_live_apis_only`, `require_provenance_url` |
| `pdf-extractor-agent` | PDF Claim, DOI & Tabular Parser | `pdf-table-extractor` | `deny("run_command")`, `workspace_only("storage/")`, `deny_synthetic_fallback` |
| `ml-rigor-scientist` | 30-Model Out-of-Fold Benchmark Engine | `lazypredict-oof-benchmark` | `require_inside_fold_preprocessing`, `prune_target_proxies`, `require_oof_ci` |
| `deep-inference-auditor` | 5-Stage Conformal & Super-Learner Auditor | `deep-conformal-escalation` | `require_permutation_null_test`, `train_split_thresholds_only` |
| `translation-whitespace-agent` | 9-API Regulatory & Venture Synthesizer | `clinical-patent-translation` | `deny("run_command")`, `honest_signal_scoring`, `zero_markdown_noise` |
| `qa-production-guardian` | Zero-Dummy-Data & QA Verification Guardian | `production-qa-audit` | `enforce_zero_dummy_data`, `verify_persistent_memory`, `workspace_only` |

## 4. Persistent Memory & Relational Schema (`storage/thesisout.db`)
- `dissertations`: Indexed academic dissertations with provenance (`external_id`, `source_url`, `university`, `author_statistical_method`, `author_reported_baseline_score`).
- `dataset_artifacts`: Real tabular CSV/TSV files on disk (`file_path`, `source_url`, `row_count`, `col_count`, `column_metadata`).
- `lazypredict_runs`: 30-model out-of-fold cross-validation leaderboards, 95% confidence intervals, `signal_gain_delta`, and `top_features`.
- `market_opportunities`: Synthesized commercial opportunities, `overall_tos_score`, live registry checks (`ClinicalTrials.gov`, `OpenTargets`, `ChEMBL`, `openFDA`), and validation roadmaps.
- `saved_items`: Persistent user curation memory supporting Star (`is_starred` Favorite) and Ribbon (`is_bookmarked` Saved to Memory) plus persistent researcher notes (`user_note`).
- `agent_registry`: Persistent registry of all production agents, roles, skills, tool permissions, and declarative safety policies.
