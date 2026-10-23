# ThesisOut

**ThesisOut** (`https://thesisout.com`) is a production multi-agent scientific discovery and translational whitespace platform powered by **Google Cloud AI** (**Gemini 3.8 Flash** via `google-genai`, **Google Antigravity SDK** `agent-cli`, and **Google Cloud Run**). It harvests doctoral dissertations from global scholarly repositories, resolves open-access empirical datasets from **Zenodo**, executes leak-free **30-model out-of-fold cross-validated benchmarks (`LazyPredict`)** with **5-Stage Deep Conformal Verification**, and audits commercial and clinical whitespace across **9 live external scientific registries**.

---

## License & Commercial Pricing Schedule

**Dual License: Free for Personal & Academic Research • Paid Commercial License Available** — Copyright (c) 2026 **Dr. Gretchen Boria**.

This repository (`https://github.com/gretchenboria/ThesisOut`) is **public** and licensed under the dual licensing terms in [`LICENSE`](LICENSE):

### 1. Free Personal, Academic & Non-Commercial Research License (`$0`)
Individuals (personal use), academic researchers, doctoral students, faculty, accredited universities, and non-profit research institutions may freely inspect, run, fork, and modify ThesisOut for **personal, educational, and non-commercial scientific research** with attribution to Dr. Gretchen Boria.

### 2. Commercial License Pricing Schedule (For-Profit Entities)
Using, hosting, integrating, or deploying ThesisOut within a for-profit company, venture capital fund, biotechnology/pharmaceutical enterprise, or commercial consulting practice requires a paid **Commercial License**:

| Tier | Target Organization | Seats / Deployment | Annual License | Perpetual / Monthly Option |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0: Personal, Academic & Non-Profit** | Individuals, Universities, PhD Researchers, Non-Profit Labs | Unlimited (Non-Commercial) | **FREE (`$0`)** | Free with Attribution |
| **Tier 1: Startup & Boutique Scout** | Seed Startups ($\le 10$ staff), Boutique IP Scouts, Independent Consultants | Up to 3 seats • 1 Cloud Run / VPC instance | **`$4,800 / yr`** | `$450 / mo` |
| **Tier 2: Venture & Biotech R&D** | Venture Capital Funds, Mid-Market Biotech/MedTech R&D, Corporate BD | Up to 15 seats • Private VPC deployment + 9-API Connectors | **`$18,000 / yr`** | `$1,750 / mo` |
| **Tier 3: Enterprise & OEM** | Global Pharma, Enterprise R&D, White-Label / Commercial Redistribution | Unlimited seats • Multi-region VPC + Full OEM Source Rights | **`$48,000 / yr`** | **`$125,000`** One-Time Buyout |

- **Commercial Licensing Inquiries**: Contact **Dr. Gretchen Boria** via GitHub ([`@gretchenboria`](https://github.com/gretchenboria)) or [`https://thesisout.com`](https://thesisout.com).

---

## Google Cloud AI Architecture & Value

ThesisOut is built on three core **Google Cloud & AI** pillars:
1. **Google Gemini 3.8 Flash (`google-genai` SDK)**:
   - Powers the `translation-whitespace-agent` to synthesize empirical out-of-fold machine learning signal gains (`Δ`) with live evidence from 9 external scientific registries (`ClinicalTrials.gov`, `OpenTargets`, `ChEMBL`, `openFDA`, `EuropePMC`, `arXiv`, `HuggingFace`, `OpenAlex`, `Zenodo`).
   - Computes calibrated **Translation Opportunity Scores (TOS)**, patent/clinical whitespace assessments, and validation roadmaps grounded in real cross-validated metrics.
2. **Google Antigravity SDK (`agent-cli` & Declarative Agent Governance)**:
   - Orchestrates 6 specialized production agents defined in `.agents/agents/*/agent.md` and `.agents/skills/*/SKILL.md`, persisted in the SQLite `agent_registry` table and verified via `./agent-cli`.
   - Enforces strict declarative policies (`enforce_zero_dummy_data`, `require_inside_fold_preprocessing`, `prune_target_proxies`, `require_permutation_null_test`, `verify_persistent_memory`).
3. **Google Cloud Run Serverless Container Runtime**:
   - Multi-stage Docker build serving the compiled React 19 + TypeScript + Vite SPA and FastAPI async backend on port `8080` with autoscaling on `us-central1`.

---

## System Architecture: Models, APIs, Agents, Libraries & Storage

### 1. AI & Machine Learning Models
- **Generative AI Synthesis**: `Gemini 3.8 Flash` (`google-genai` SDK) with deterministic empirical fallback.
- **30-Model Out-of-Fold Benchmark (`LazyPredict`)**:
  - **Gradient Boosted Trees**: `XGBoost`, `LightGBM`, `CatBoost`, `HistGradientBoosting`, `GradientBoosting`, `AdaBoost`.
  - **Bagging & Tree Ensembles**: `RandomForest`, `ExtraTrees`, `Bagging`, `DecisionTree`, `ExtraTree`.
  - **Linear, Kernel & Neural Estimators**: `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`, `SVC`/`SVR` (`RBF`), `NuSVC`, `LinearSVC`, `MLPClassifier`/`MLPRegressor`, `KNeighbors`, `GaussianNB`, `CalibratedClassifierCV`.
- **Stage-2 Deep Conformal Escalation**:
  - Level-2 **Super-Learner Ensemble** (`CalibratedClassifierCV` / `RidgeCV` meta-learner stacking top 3 diverse model families).
  - **5-Fold Permutation Null Test** ($p$-value against shuffled target labels), **Top-Feature Ablation Audit**, **Depth-2 Surrogate Decision Rules** (fit strictly on `X_train`), and **90% Split-Conformal Prediction Calibration** (`alpha = 0.10`).

### 2. Live External Scientific APIs (9 Registries)
| Registry | Endpoint / Protocol | Purpose in ThesisOut |
| :--- | :--- | :--- |
| **OpenAlex** | `https://api.openalex.org/works` | Doctoral dissertation harvesting, institution tier scoring, and citation velocity. |
| **Zenodo** | `https://zenodo.org/api/records` | Open-access empirical tabular dataset resolution and download (`.csv`, `.tsv`, `.xlsx`, `.parquet`). |
| **ClinicalTrials.gov** | `https://clinicaltrials.gov/api/v2/studies` | Active & recruiting human clinical trial landscape and NCT registry lookup. |
| **OpenTargets** | `https://api.platform.opentargets.org/api/v4/graphql` | Target-disease association scores and genetic tractability evidence. |
| **ChEMBL** | `https://www.ebi.ac.uk/chembl/api/data` | Small-molecule bioactivity, IC50/Ki assays, and approved drug mechanisms. |
| **openFDA** | `https://api.fda.gov/drug/event.json` | Post-market adverse event pharmacovigilance and regulatory safety signals. |
| **EuropePMC** | `https://www.ebi.ac.uk/europepmc/webservices/rest` | Full-text biomedical theses, supplemental data links, and literature verification. |
| **arXiv** | `https://export.arxiv.org/api/query` | Preprint surveillance across quantitative biology, physics, and machine learning. |
| **HuggingFace** | `https://huggingface.co/api/models` | Open-weights scientific foundation model and dataset availability check. |

### 3. Registered Agents & Governance Policies (`./agent-cli`)
| Agent ID | Role | Skill (`SKILL.md`) | Declarative Policies |
| :--- | :--- | :--- | :--- |
| `harvester-agent` | Corpus & Open-Access Dataset Harvester | `thesis-corpus-harvester` | `deny("run_command")`, `allow_live_apis_only`, `require_provenance_url` |
| `pdf-extractor-agent` | PDF Claim, DOI & Tabular Parser | `pdf-table-extractor` | `deny("run_command")`, `workspace_only("storage/")`, `deny_synthetic_fallback` |
| `data-cleansing-agent` | Scientific Data Cleansing, Schema Curation & Leakage Pruning | `scientific-data-cleanser` | `deny("run_command")`, `strip_identifier_and_proxy_columns`, `inside_fold_imputation_only` |
| `ml-rigor-scientist` | 30-Model Out-of-Fold Benchmark Engine | `lazypredict-oof-benchmark` | `require_inside_fold_preprocessing`, `prune_target_proxies`, `require_oof_ci` |
| `deep-inference-auditor` | 5-Stage Conformal & Super-Learner Auditor | `deep-conformal-escalation` | `require_permutation_null_test`, `train_split_thresholds_only` |
| `translation-whitespace-agent` | 9-API Regulatory & Venture Synthesizer | `clinical-patent-translation` | `deny("run_command")`, `honest_signal_scoring`, `zero_markdown_noise` |
| `qa-production-guardian` | Zero-Dummy-Data & QA Verification Guardian | `production-qa-audit` | `enforce_zero_dummy_data`, `verify_persistent_memory`, `workspace_only` |

### 4. Libraries & Tools
- **Backend & ML**: `Python 3.12`, `FastAPI`, `SQLAlchemy 2.0`, `Pydantic v2`, `httpx`, `Scikit-Learn`, `LazyPredict`, `XGBoost`, `LightGBM`, `CatBoost`, `SciPy`, `Statsmodels`, `Pandas`, `NumPy`, `pdfplumber`, `pypdf`, `google-genai`.
- **Frontend**: `React 19`, `TypeScript`, `Vite`, `Tailwind CSS`, `Lucide React`.
- **Agent CLI**: `./agent-cli` (`list`, `inspect`, `skills`, `verify-policies`, `sync-db`, `audit`).

### 5. Relational Database Schema (`storage/thesisout.db`)
- `dissertations`: Indexed dissertations (`external_id`, `title`, `authors`, `university`, `publication_year`, `domain`, `source_url`, `author_statistical_method`, `author_reported_baseline_score`).
- `dataset_artifacts`: Real tabular files in `storage/datasets/` (`file_path`, `source_url`, `row_count`, `col_count`, `column_metadata`, `missing_pct`).
- `lazypredict_runs`: 30-model out-of-fold cross-validation results (`best_model_name`, `best_model_score`, `author_baseline_score`, `signal_gain_delta`, `all_models_leaderboard`, `top_features`, `deep_verification_json`).
- `market_opportunities`: Commercial & clinical synthesis (`overall_tos_score`, `matched_problem_title`, `scientific_merit_narrative`, `commercial_whitespace_assessment`, `external_scientific_signals`).
- `saved_items`: Persistent user curation memory (`item_type`, `item_id`, `is_starred`, `is_bookmarked`, `user_note`) preserved across workspace resets.
- `agent_registry`: Registered agents, skills, tool permissions, policies, and live invocation telemetry.

---

## Running Locally & Verifying QA

```bash
# 1. Verify all 6 registered agents and zero-dummy-data production audit
./agent-cli verify-policies
./agent-cli audit

# 2. Run end-to-end automated pipeline & edge-case tests
./backend/.venv/bin/pytest backend/tests/test_pipeline.py -v

# 3. Start backend + frontend locally
./start.sh
```

## Deploying to Google Cloud Run

```bash
gcloud run deploy thesisout \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```
