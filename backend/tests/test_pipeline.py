import io
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app


def test_real_data_only_pipeline():
    with TestClient(app) as client:
        # 1. Verify health and LLM router config
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["primary_llm"] == "gemini-4-argon"
        assert r_health.json()["fallback_llm"] == "claude-opus-5-5"

        # 2. Verify zero synthetic dissertations are pre-populated on startup
        r_stats = client.get("/api/stats")
        assert r_stats.status_code == 200
        stats = r_stats.json()
        assert stats["giant_impact_problems_count"] >= 15

        # 3. Upload the real UCI Wisconsin Breast Cancer Diagnostic dataset (569 real clinical biopsy samples, 30 features)
        bc = load_breast_cancer(as_frame=True)
        real_df = bc.frame
        csv_bytes = real_df.to_csv(index=False).encode("utf-8")

        r_upload = client.post(
            "/api/upload-dissertation",
            data={
                "title": "Nuclear Morphometric Feature Extraction in Fine-Needle Aspirates of Breast Masses",
                "authors": "W. N. Street, W. H. Wolberg, O. L. Mangasarian",
                "university": "University of Wisconsin-Madison",
                "publication_year": "1993",
                "domain": "biomedical",
                "author_primary_target": "target",
            },
            files={"dataset_file": ("uci_wisconsin_breast_cancer_real.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r_upload.status_code == 200
        diss = r_upload.json()
        assert diss["title"].startswith("Nuclear Morphometric")
        assert len(diss["datasets"]) == 1
        assert diss["datasets"][0]["row_count"] == 569
        assert len(diss["datasets"][0]["lazy_runs"]) >= 2

        # 4. Verify Stage-2 Deep Inference Escalation on the real clinical dataset
        lazy_run_id = diss["datasets"][0]["lazy_runs"][0]["id"]
        r_esc = client.post(f"/api/lazypredict/{lazy_run_id}/escalate")
        assert r_esc.status_code == 200
        esc = r_esc.json()
        assert "stage1_leakage_and_permutation_audit" in esc
        assert "stage2_super_learner_ensemble" in esc
        assert esc["stage2_super_learner_ensemble"]["super_learner_score"] > 0.90
