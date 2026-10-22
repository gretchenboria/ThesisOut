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

        # 2. Verify zero synthetic dissertations or problems are hardcoded on startup
        r_stats = client.get("/api/stats")
        assert r_stats.status_code == 200
        stats = r_stats.json()
        assert stats["dissertations_indexed"] == 0

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

        # 3b. Verify leaderboard statistical rigor: zero 1.0 degeneracy & distinct Precision/Recall/F1/ROC-AUC/Accuracy
        first_run = diss["datasets"][0]["lazy_runs"][0]
        lb = first_run["leaderboard"]
        assert len(lb) >= 15
        for row in lb[:10]:
            if first_run["task_type"] == "classification":
                assert row["ROC AUC"] < 1.0
                assert row["Accuracy"] < 1.0
                assert row["Recall"] != row["Accuracy"] or row["Precision"] != row["Accuracy"]
                assert row["ROC AUC"] != row["Balanced Accuracy"]
            else:
                assert row["R-Squared"] < 1.0
            assert "95% CI" in row and not row["95% CI"].endswith("1.0000]")

        # 4. Verify Stage-2 Deep Inference Escalation on the real clinical dataset
        lazy_run_id = diss["datasets"][0]["lazy_runs"][0]["id"]
        r_esc = client.post(f"/api/lazypredict/{lazy_run_id}/escalate")
        assert r_esc.status_code == 200
        esc = r_esc.json()
        assert "stage1_leakage_and_permutation_audit" in esc
        assert "stage2_super_learner_ensemble" in esc
        assert esc["stage2_super_learner_ensemble"]["super_learner_score"] > 0.90

        # 4b. Verify all 4 interactive Actionable Steps on the Market Opportunity
        opp_id = diss["opportunities"][0]["id"]
        for s_idx in range(4):
            r_step = client.post(f"/api/opportunities/{opp_id}/execute-step?step_index={s_idx}")
            assert r_step.status_code == 200
            assert r_step.json()["step_index"] == s_idx

        # 5. Test live scientific integrations probe (OpenTargets, ChEMBL, ClinicalTrials, openFDA, EuropePMC, arXiv, HuggingFace)
        r_probe = client.get("/api/integrations/probe?query=EGFR")
        assert r_probe.status_code == 200
        probe = r_probe.json()
        assert "opentargets_graphql_v4" in probe["live_integrations"]
        assert "chembl_ebi_rest" in probe["live_integrations"]
        assert "clinicaltrials_gov_v2" in probe["live_integrations"]
        assert "openfda_faers" in probe["live_integrations"]
        assert "europepmc_rest" in probe["live_integrations"]
        assert "arxiv_atom_api" in probe["live_integrations"]

        # 5b. Test Gemini Discovery & Downstream Inference Co-Pilot Chat
        r_chat = client.post(
            "/api/copilot/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Which morphometric variables in this biopsy dataset drive nonlinear separation and what column should we test next?",
                    }
                ],
                "dissertation_id": diss["id"],
                "lazy_run_id": lazy_run_id,
            },
        )
        assert r_chat.status_code == 200
        chat_data = r_chat.json()
        assert "reply" in chat_data and len(chat_data["reply"]) > 20
        assert "suggested_actions" in chat_data and len(chat_data["suggested_actions"]) > 0


        # 6. Reset workspace so DB remains completely clean
        r_reset = client.delete("/api/workspace/reset")
        assert r_reset.status_code == 200
        assert client.get("/api/stats").json()["dissertations_indexed"] == 0
