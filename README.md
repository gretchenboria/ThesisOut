# ThesisOut 🌈🔬

**ThesisOut** is an agentic research-to-impact web application that ingests real doctoral dissertations and associated datasets, automatically benchmarks **30+ machine learning models (`LazyPredict`)** alongside a **5-Stage Deep Inference Verification Pipeline**, and matches overlooked findings to **$2.5T+ Giant-Impact Moonshot Problems** using **Gemini 4 Argon** (with **Claude Opus 5.5** fallback).

---

## ✨ Friendly 3-Step Guided Workflow

1. **Step 1 — Pick or Add a Thesis (Cheer Pink)**:
   - Search live university repositories (**OpenAlex API**) and open-access dataset archives (**Zenodo API**) with 1-click topic chips, or upload your own dissertation PDF and CSV/Excel dataset.
2. **Step 2 — 30-Model Data Check (`LazyPredict` + Stage-2 Deep Verification)**:
   - View a plain-English takeaway card and 3 simple summary cards first (*Standard Linear Baseline*, *Best Model Found*, and *Top 3 Clues*).
   - Expand progressive-disclosure drawers only when you want to peek inside all 30+ models, test another column, or run the **5-Stage Deep Verification** (5-Fold CV + Permutation Null Check + Level-2 Super-Learner Stacking + 2×2 Sweet-Spot Thresholds + Conformal Reliability).
3. **Step 3 — Market & Impact Match (Share Lavender & Peach)**:
   - Explore how the hidden signal solves major real-world clinical and engineering problems, verified against **ClinicalTrials.gov** and **OpenTargets**.

---

## ☁️ Deploying to Google Cloud Run

This repository includes a production multi-stage [`Dockerfile`](Dockerfile) and [`cloudbuild.yaml`](cloudbuild.yaml) that compile the React SPA and serve both the frontend and FastAPI backend on port `8080`.

### One-Command Cloud Run Deploy
```bash
gcloud run deploy thesisout \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

---

## 💻 Running Locally

```bash
./start.sh
```
- **Web App**: `http://127.0.0.1:5173`
- **API Docs**: `http://127.0.0.1:8000/docs`
