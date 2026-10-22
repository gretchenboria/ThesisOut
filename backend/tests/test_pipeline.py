import io
import subprocess
from pathlib import Path
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app

AUTH_HEADERS = {"X-Access-Code": "thesisout!"}


def test_agent_registry_and_zero_dummy_data_policies():
    # 1. Verify agent-cli policy audit passes (6/6 agents, zero dummy data generators)
    cli_path = Path("/Users/dr.gretchenboria/thesisout/agent-cli")
    assert cli_path.exists()
    res = subprocess.run([str(cli_path), "verify-policies"], capture_output=True, text=True)
    assert res.returncode == 0, f"agent-cli verify-policies failed: {res.stdout}\n{res.stderr}"
    assert "ALL AGENT REGISTRY & ZERO-DUMMY-DATA POLICIES VERIFIED" in res.stdout


def test_production_pipeline_persistent_memory_and_agents():
    with TestClient(app) as client:
        # 1. Verify health
        r_health = client.get("/health")
        assert r_health.status_code == 200

        # 2. Verify Agent Registry API returns all 6 registered agents with policies and SKILL.md / agent.md paths
        r_agents = client.get("/api/agents/registry", headers=AUTH_HEADERS)
        assert r_agents.status_code == 200
        agents = r_agents.json()
        assert len(agents) == 6
        agent_names = {a["name"] for a in agents}
        assert {
            "harvester-agent",
            "pdf-extractor-agent",
            "ml-rigor-scientist",
            "deep-inference-auditor",
            "translation-whitespace-agent",
            "qa-production-guardian",
        } == agent_names

        for a in agents:
            r_ver = client.post(f"/api/agents/registry/{a['name']}/verify", headers=AUTH_HEADERS)
            assert r_ver.status_code == 200
            ver_data = r_ver.json()
            assert ver_data["status"] == "verified"
            assert ver_data["agent_md_verified"] is True
            assert ver_data["skill_md_verified"] is True

        # 3. Verify live OpenAlex + real Zenodo bootstrap populates real studies & real 30-model runs
        r_diss = client.get("/api/dissertations", headers=AUTH_HEADERS)
        assert r_diss.status_code == 200
        diss_list = r_diss.json()
        assert len(diss_list) >= 2
        for d in diss_list:
            assert d["source_repo"] in ("OpenAlex Live API", "Direct Upload", "EuropePMC", "arXiv")
            assert len(d["datasets"]) >= 1
            assert len(d["datasets"][0]["lazy_runs"]) >= 1
            assert len(d["opportunities"]) >= 1

        # 4. Upload real UCI Wisconsin Breast Cancer Diagnostic dataset and verify 30-model out-of-fold audit
        bc = load_breast_cancer(as_frame=True)
        csv_bytes = bc.frame.to_csv(index=False).encode("utf-8")

        r_upload = client.post(
            "/api/dissertations/upload",
            headers=AUTH_HEADERS,
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
        assert len(diss["datasets"][0]["lazy_runs"]) >= 1

        first_run = diss["datasets"][0]["lazy_runs"][0]
        lb = first_run["leaderboard"]
        assert len(lb) >= 15
        for row in lb[:10]:
            if first_run["task_type"] == "classification":
                assert row["ROC AUC"] < 1.0
                assert row["Accuracy"] < 1.0
                assert row["ROC AUC"] != row["Balanced Accuracy"]
            else:
                assert row["R-Squared"] < 1.0
            assert "95% CI" in row and not row["95% CI"].endswith("1.0000]")

        # 5. Test Persistent Memory: Star (favorite), Ribbon (bookmark), and Persistent Researcher Note
        opp_id = diss["opportunities"][0]["id"]
        r_star = client.post(
            "/api/saved/toggle",
            headers=AUTH_HEADERS,
            json={"dissertation_id": diss["id"], "opportunity_id": opp_id, "toggle_field": "star"},
        )
        assert r_star.status_code == 200
        saved_item = r_star.json()
        assert saved_item["is_starred"] is True
        assert saved_item["is_bookmarked"] is False

        r_ribbon = client.post(
            "/api/saved/toggle",
            headers=AUTH_HEADERS,
            json={"dissertation_id": diss["id"], "opportunity_id": opp_id, "toggle_field": "ribbon"},
        )
        assert r_ribbon.status_code == 200
        saved_item = r_ribbon.json()
        assert saved_item["is_starred"] is True
        assert saved_item["is_bookmarked"] is True

        r_note = client.patch(
            f"/api/saved/{saved_item['id']}/note",
            headers=AUTH_HEADERS,
            json={"user_note": "High-priority Phase-1b biopsy validation candidate.", "tags": ["biopsy", "priority"]},
        )
        assert r_note.status_code == 200
        assert r_note.json()["user_note"] == "High-priority Phase-1b biopsy validation candidate."

        r_saved_list = client.get("/api/saved", headers=AUTH_HEADERS)
        assert r_saved_list.status_code == 200
        assert any(s["id"] == saved_item["id"] for s in r_saved_list.json())

        # 6. Verify Stage-2 Deep Inference Escalation, Custom Target Rotation & all 4 Opportunity Action Steps
        lazy_run_id = first_run["id"]
        r_esc = client.post(f"/api/lazypredict/{lazy_run_id}/escalate", headers=AUTH_HEADERS)
        assert r_esc.status_code == 200
        esc = r_esc.json()
        assert "stage1_leakage_and_permutation_audit" in esc
        assert "stage2_super_learner_ensemble" in esc

        # Custom target rotation on secondary column
        ds_id = diss["datasets"][0]["id"]
        r_custom = client.post(
            "/api/lazypredict/run-custom",
            headers=AUTH_HEADERS,
            json={
                "dataset_id": ds_id,
                "target_column": "mean radius",
                "is_author_primary_target": False,
                "target_rationale": "Secondary morphometric target rotation test",
            },
        )
        assert r_custom.status_code == 200
        custom_run = r_custom.json()
        assert custom_run["target_column"] == "mean radius"
        assert len(custom_run["leaderboard"]) >= 15

        for s_idx in range(4):
            r_step = client.post(f"/api/opportunities/{opp_id}/execute-step?step_index={s_idx}", headers=AUTH_HEADERS)
            assert r_step.status_code == 200
            assert r_step.json()["step_index"] == s_idx

        # 7. Edge Case: Standalone PDF Upload (no CSV file attached) with embedded table & metadata parsing
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        lines = [
            "Thermal-Voltage Coupling in Sub-Threshold Silicon Interconnects",
            "By Evelyn Vance",
            "Massachusetts Institute of Technology",
            "Submitted in 2024. Ordinary Least Squares linear regression baseline R2 = 0.61.",
            "Voltage_mV Temp_C Current_mA Latency_ns",
        ]
        for idx_row in range(1, 31):
            v = 450.0 + idx_row * 6.5
            t_c = 28.0 + (idx_row % 7) * 4.2
            curr = 12.4 + idx_row * 0.85 + (idx_row % 3) * 1.1
            lat = 1.85 + (v * 0.004) + (t_c * 0.09) + ((idx_row % 5) * 0.35)
            lines.append(f"{v:.2f} {t_c:.2f} {curr:.2f} {lat:.3f}")

        content_ops = ["BT /F1 10 Tf 50 750 Td 12 TL"]
        for ln in lines:
            safe_ln = ln.replace("(", "\\(").replace(")", "\\)")
            content_ops.append(f"({safe_ln}) Tj T*")
        content_ops.append("ET")
        stream_data = "\n".join(content_ops).encode("latin-1")

        stream_obj = DecodedStreamObject()
        stream_obj.set_data(stream_data)
        stream_ref = writer._add_object(stream_obj)
        page[NameObject("/Contents")] = stream_ref
        font_dict = DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
        })
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): font_dict})

        pdf_buf = io.BytesIO()
        writer.write(pdf_buf)
        pdf_bytes = pdf_buf.getvalue()

        r_pdf_only = client.post(
            "/api/dissertations/upload",
            headers=AUTH_HEADERS,
            files={"pdf_file": ("mit_thermal_interconnect_2024.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert r_pdf_only.status_code == 200
        pdf_diss = r_pdf_only.json()
        assert "Massachusetts Institute of Technology" in pdf_diss["university"]
        assert len(pdf_diss["datasets"]) >= 1
        assert pdf_diss["datasets"][0]["source_type"] == "pdf_table"
        assert pdf_diss["datasets"][0]["row_count"] == 30
        assert len(pdf_diss["datasets"][0]["lazy_runs"]) >= 1
        assert len(pdf_diss["opportunities"]) >= 1

        # 8. Edge Case: Messy CSV upload (Unnamed: 0, missing NaNs, categorical strings, no PDF)
        messy_df = bc.frame.iloc[:90, :6].copy()
        messy_df.insert(0, "Unnamed: 0", range(90))
        messy_df.insert(1, "patient_id", [f"PAT_{1000+i}" for i in range(90)])
        messy_df["biopsy_site"] = ["ductal" if i % 2 == 0 else "lobular" for i in range(90)]
        messy_df.loc[5:12, "mean radius"] = None
        messy_df["target"] = bc.frame["target"].iloc[:90].values
        messy_bytes = messy_df.to_csv(index=False).encode("utf-8")

        r_messy = client.post(
            "/api/dissertations/upload",
            headers=AUTH_HEADERS,
            files={"dataset_file": ("clinical_biopsy_messy_cohort.csv", io.BytesIO(messy_bytes), "text/csv")},
        )
        assert r_messy.status_code == 200
        messy_diss = r_messy.json()
        assert len(messy_diss["datasets"]) == 1
        assert len(messy_diss["datasets"][0]["lazy_runs"]) >= 1

        # 9. Edge Case: Workspace Reset preserves Starred/Bookmarked items in persistent memory
        r_reset = client.delete("/api/workspace/reset", headers=AUTH_HEADERS)
        assert r_reset.status_code == 200
        r_saved_after = client.get("/api/saved", headers=AUTH_HEADERS)
        assert r_saved_after.status_code == 200
        assert any(s["id"] == saved_item["id"] and s["is_starred"] is True for s in r_saved_after.json())

        # 10. Clean up test uploaded item from saved memory
        r_del_saved = client.delete(f"/api/saved/{saved_item['id']}", headers=AUTH_HEADERS)
        assert r_del_saved.status_code == 200


def test_three_stage_dataset_finder(tmp_path):
    """
    Verifies the 3-stage Dataset Finder:
    1. Full-Text Data Availability & Supplementary Material Section Scraping (JATS XML + back-matter DOIs).
    2. Author ORCID & Lab Graph Metadata Extraction (OpenAlex + DataCite + Crossref ±2 years).
    3. Multi-Repository Tabular Harvesters (Zenodo, Figshare, Harvard Dataverse, Dryad).
    """
    import asyncio
    from app.agents.harvester_agent import harvester_agent
    from app.agents.pdf_extractor import pdf_extractor_agent

    async def _run():
        sample_jats_xml = """
        <article>
          <body><p>Main dissertation results section.</p></body>
          <back>
            <sec sec-type="data-availability">
              <title>Data and Code Availability</title>
              <p>Datasets are deposited at Zenodo (10.5281/zenodo.3875504), Figshare (10.6084/m9.figshare.34714530),
              Harvard Dataverse (10.7910/DVN/9FMMJI), and Dryad (10.5061/dryad.98765).</p>
            </sec>
          </back>
        </article>
        """
        da_res = pdf_extractor_agent.extract_data_availability_section(sample_jats_xml)
        sources = {lk["source_type"] for lk in da_res["links"]}
        assert {"zenodo", "figshare", "dataverse", "dryad"}.issubset(sources)
        assert all(lk["section"] == "data_availability" for lk in da_res["links"])

        oa_diss = await harvester_agent.search_openalex_dissertations("machine learning biomarker", max_results=2)
        assert len(oa_diss) >= 1
        assert "author_orcids" in oa_diss[0]
        assert "author_openalex_ids" in oa_diss[0]
        assert "institution_ids" in oa_diss[0]

        fig_files = await harvester_agent.fetch_figshare_datasets("biomarker", tmp_path, max_files=1)
        assert len(fig_files) == 1
        assert fig_files[0]["source_type"] == "figshare"
        assert Path(fig_files[0]["file_path"]).exists()

        dv_files = await harvester_agent.fetch_harvard_dataverse_datasets("biomarker", tmp_path, max_files=1)
        assert len(dv_files) == 1
        assert dv_files[0]["source_type"] == "dataverse"
        assert Path(dv_files[0]["file_path"]).exists()

    asyncio.run(_run())

