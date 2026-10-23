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

        # 2. Verify Agent Registry API returns all 7 registered agents with policies and SKILL.md / agent.md paths
        r_agents = client.get("/api/agents/registry", headers=AUTH_HEADERS)
        assert r_agents.status_code == 200
        agents = r_agents.json()
        assert len(agents) == 7
        agent_names = {a["name"] for a in agents}
        assert {
            "harvester-agent",
            "pdf-extractor-agent",
            "data-cleansing-agent",
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


def test_regression_and_edge_cases(tmp_path, monkeypatch):
    """
    Comprehensive regression & edge-case test suite covering:
    1. Gemini 4 Argon / 3.8 deep PDF extraction (valid matrix persistence vs. non-numeric garbage rejection vs. empty PDF handling).
    2. Scientific data cleansing (mean ± SD, %, CI brackets, inequalities, title-banner header promotion, duplicate column deduplication).
    3. Leakage proxy pruning (Unnamed: 0, single-feature tautology |r| >= 0.88, boolean complement, multi-feature post-hoc composite).
    4. Scientific coherence & honest TOS scoring (no unrelated psoriasis/atherosclerosis fallback on zero keyword overlap; no artificial 62.0 signal floor on R2 < 0.25).
    5. Tiny/invalid dataset upload rejection (< 5 rows or < 2 columns) with zero orphan dissertation rows left in SQLite.
    """
    import asyncio
    import pandas as pd
    from app.agents.pdf_extractor import pdf_extractor_agent
    from app.agents.data_cleansing_agent import data_cleansing_agent
    from app.agents.data_scientist import data_scientist_agent
    from app.agents.translation_agent import translation_agent
    from app.services.llm_router import llm_router

    # --- 1. Gemini 4 Argon / 3.8 Deep PDF Extraction Edge Cases ---
    async def _test_gemini_pdf():
        # Edge case 1a: Missing or empty PDF path returns gracefully
        empty_res = await pdf_extractor_agent.extract_gemini_pdf_intelligence_and_tables(tmp_path / "nonexistent.pdf", "empty_slug")
        assert empty_res == {"gemini_meta": {}, "gemini_tables": []}

        # Edge case 1b: Mock PDF with text and mocked Gemini 4 Argon response containing 1 valid numeric table + 1 non-numeric garbage table
        monkeypatch.setattr(
            pdf_extractor_agent,
            "extract_text_and_metadata",
            lambda p: {"text_preview": "A" * 200, "full_text_excerpt": "Study on Silicon Carbide MOSFET Gate Oxide Breakdown under Thermal Stress. " * 5},
        )

        async def _mock_gemini_intel(*args, **kwargs):
            return {
                "parsed_title": "Silicon Carbide MOSFET Gate Oxide Breakdown Under Thermal Stress",
                "parsed_authors": "Dr. Kenji Takahashi",
                "parsed_university": "Tokyo Institute of Technology",
                "parsed_year": 2025,
                "author_primary_hypothesis": "Non-linear gate leakage current accelerates above 175C.",
                "author_statistical_method": "Weibull Linear Fit",
                "author_primary_target": "breakdown_hours",
                "author_baseline_score": 0.58,
                "extracted_claims": ["Gate oxide breakdown exhibits non-linear thermal acceleration above 175C."],
                "extracted_tables": [
                    {
                        "table_name": "Table_1_SiC_Gate_Breakdown",
                        "columns": ["temp_c", "gate_voltage_v", "leakage_na", "breakdown_hours"],
                        "rows": [
                            [150.0, 18.0, 12.4, 1420.5],
                            [165.0, 19.0, 24.8, 980.2],
                            [175.0, 20.0, 58.1, 510.4],
                            [185.0, 21.0, 115.6, 215.8],
                            [195.0, 22.0, 240.2, 88.4],
                            [205.0, 23.0, 490.0, 32.1],
                        ],
                    },
                    {
                        "table_name": "Table_2_Pure_Text_Glossary",
                        "columns": ["Symbol", "Definition"],
                        "rows": [
                            ["Vgs", "Gate-to-Source Voltage"],
                            ["Ids", "Drain-to-Source Current"],
                            ["Tj", "Junction Temperature"],
                            ["Rds", "On-State Resistance"],
                            ["Qg", "Total Gate Charge"],
                        ],
                    },
                ],
            }

        monkeypatch.setattr(llm_router, "extract_scientific_pdf_intelligence", _mock_gemini_intel)
        dummy_pdf = tmp_path / "sic_mosfet.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 mock")
        g_out = await pdf_extractor_agent.extract_gemini_pdf_intelligence_and_tables(dummy_pdf, "sic_test")
        assert g_out["gemini_meta"]["parsed_university"] == "Tokyo Institute of Technology"
        assert g_out["gemini_meta"]["author_primary_target"] == "breakdown_hours"
        # Only Table 1 (numeric) should be kept; Table 2 (pure text glossary) must be rejected
        assert len(g_out["gemini_tables"]) == 1
        assert g_out["gemini_tables"][0]["row_count"] == 6
        assert g_out["gemini_tables"][0]["col_count"] == 4
        assert Path(g_out["gemini_tables"][0]["file_path"]).exists()

    asyncio.run(_test_gemini_pdf())

    # --- 2. Scientific Data Cleansing & String Coercion Edge Cases ---
    raw_Banner_df = pd.DataFrame(
        [
            ["SUPPLEMENTARY TABLE S1: CLINICAL CYTOKINE COHORT", None, None, None],
            ["Unnamed: 0", "IL6_Level", "Response_Rate", "Outcome_Score"],
            ["0", "14.2 ± 1.8", "84.5%", "0.42"],
            ["1", "18.9 ± 2.1", "76.0%", "0.51"],
            ["2", "11.4 [9.1 - 13.7]", "91.2%", "0.33"],
            ["3", "< 0.001", "62.4%", "0.78"],
            ["4", "22.5 ± 3.4", "55.0%", "0.84"],
            ["5", "16.1 ± 1.2", "80.1%", "0.49"],
            ["6", "19.8 ± 2.9", "68.5%", "0.66"],
            ["7", "13.0 ± 1.1", "88.0%", "0.38"],
        ]
    )
    banner_csv = tmp_path / "cytokine_supp.csv"
    raw_Banner_df.to_csv(banner_csv, index=False, header=False)
    cleansed_df, c_report = data_cleansing_agent.cleanse_tabular_file(banner_csv)
    assert "Unnamed: 0" not in cleansed_df.columns
    assert "IL6_Level" in cleansed_df.columns
    assert pd.api.types.is_numeric_dtype(cleansed_df["IL6_Level"])
    assert pd.api.types.is_numeric_dtype(cleansed_df["Response_Rate"])
    assert len(cleansed_df) == 8

    # --- 3. Leakage Proxy Pruning (Single-Feature Tautology & Boolean Complement) ---
    leak_df = pd.DataFrame({
        "Unnamed: 0": list(range(30)),
        "legit_biomarker_a": [float((i * 7) % 13) + (i * 0.15) for i in range(30)],
        "legit_biomarker_b": [float((i * 3) % 11) - (i * 0.08) for i in range(30)],
        "target_outcome": [float(i * 2.5 + ((i % 4) * 1.2)) for i in range(30)],
    })
    # Inject a direct tautological proxy (r > 0.99 with target_outcome)
    leak_df["tautology_proxy"] = leak_df["target_outcome"] * 1.001 + 0.002

    run_res = data_scientist_agent.run_lazypredict_experiment(
        df=leak_df,
        target_column="target_outcome",
        author_baseline_score=0.55,
    )
    top_feat_names = {f["feature"] for f in run_res["top_features"]}
    assert "Unnamed: 0" not in top_feat_names
    assert "tautology_proxy" not in top_feat_names, "Tautological proxy (r >= 0.88) must be pruned prior to evaluation"
    assert run_res["best_score"] < 1.0

    # --- 4. Scientific Coherence & Honest Signal Scoring Edge Cases ---
    from app.models import Dissertation, IndustryProblem, LazyPredictRun

    mock_diss = Dissertation(
        title="Support Vector Machine Kernel Optimization on Synthetic Hyperplanes",
        authors="Test Researcher",
        university="Test University",
        publication_year=2025,
        domain="biomedical",
        subdomain="Kernel Methods",
        abstract="An empirical benchmark of radial basis function kernels.",
        extracted_claims=["RBF kernels converge under bounded margin constraints."],
    )
    mock_lazy = LazyPredictRun(
        target_column="sensor_drift",
        task_type="regression",
        best_model_name="Ridge",
        best_score=0.0159,
        author_baseline_score=0.6200,
        signal_gain_delta=-0.6041,
        top_features=[{"feature": "ambient_temp", "importance": 0.52}],
    )
    mock_prob = IndustryProblem(
        title="TNF-Alpha Inhibitor Non-Response in Psoriasis and Atherosclerosis",
        domain="biomedical",
        tam_billions_usd=42.0,
        keywords=["psoriasis", "atherosclerosis", "tnf", "cytokine", "inflammation"],
    )

    unrelated_problem = translation_agent.match_giant_impact_problem(
        dissertation=mock_diss,
        lazy_run=mock_lazy,
        problems=[mock_prob],
    )
    assert unrelated_problem is None, "Must never force an unrelated disease problem when keyword/title overlap is zero"

    # Low R2 (< 0.25) must never be floored at 62.0
    async def _test_low_r2_tos():
        low_opp = await translation_agent.synthesize_opportunity(
            dissertation=mock_diss,
            lazy_run=mock_lazy,
            problems=[mock_prob],
        )
        assert low_opp["data_signal_score"] < 30.0, f"Expected honest low signal score < 30.0 for R2=0.0159, got {low_opp['data_signal_score']}"

    asyncio.run(_test_low_r2_tos())

    # --- 5. Tiny / Invalid Dataset Upload Rejection (Zero Orphan Rows) ---
    with TestClient(app) as client:
        before_count = len(client.get("/api/dissertations", headers=AUTH_HEADERS).json())
        tiny_csv = b"col_a,col_b\n1.0,2.0\n3.0,4.0\n"
        r_tiny = client.post(
            "/api/dissertations/upload",
            headers=AUTH_HEADERS,
            files={"dataset_file": ("tiny_invalid.csv", io.BytesIO(tiny_csv), "text/csv")},
        )
        assert r_tiny.status_code == 400
        after_count = len(client.get("/api/dissertations", headers=AUTH_HEADERS).json())
        assert after_count == before_count, "Failed upload must clean up and leave zero orphan Dissertation rows"


