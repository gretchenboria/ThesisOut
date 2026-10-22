# ThesisOut Workspace Rules & Agent Governance

## 1. Zero Dummy or Hardcoded Data Mandate
- Never use `np.random` or synthetic arrays to generate datasets, cohorts, or fallback PDF tables.
- Never hardcode model leaderboards (`_build_30_model_leaderboard`). Every `LazyPredictRun` must be produced by running `data_scientist_agent.run_lazypredict_experiment` on a real tabular file on disk.
- Every dissertation and dataset in production must originate from a live repository (`OpenAlex`, `EuropePMC`, `arXiv`, `Zenodo`) or a direct user upload.

## 2. Biostatistical & ML Rigor Rules
- All imputation (`SimpleImputer`), categorical encoding (`OrdinalEncoder`), and scaling (`StandardScaler`) must be fit strictly inside each cross-validation fold (`X_train` only).
- Identifier columns (`Unnamed: 0`, monotonic row indices, sample IDs), boolean complements, single-feature tautologies (`|r| >= 0.88`, Spearman `|rho| >= 0.90`, depth-2 stump CV `>= 0.92`), and multi-feature post-hoc composite columns must be stripped prior to model evaluation.

## 3. Persistent Memory & User Curation
- User favorites (`is_starred` via Star icon) and bookmarks (`is_bookmarked` via Ribbon icon) are stored persistently in the `saved_items` SQLite table along with user notes (`user_note`).
- Workspace resets and server restarts must never delete saved items or their underlying dissertations and opportunities.

## 4. Registered Agents & Skills
- Agents are registered in `.agents/agents/` and persisted in the `agent_registry` table via `./agent-cli`:
  - `harvester-agent` ( Skill: `thesis-corpus-harvester` )
  - `pdf-extractor-agent` ( Skill: `pdf-table-extractor` )
  - `ml-rigor-scientist` ( Skill: `lazypredict-oof-benchmark` )
  - `deep-inference-auditor` ( Skill: `deep-conformal-escalation` )
  - `translation-whitespace-agent` ( Skill: `clinical-patent-translation` )
  - `qa-production-guardian` ( Skill: `production-qa-audit` )
