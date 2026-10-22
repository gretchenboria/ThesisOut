import React, { useEffect, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Compass,
  Heart,
  Play,
  RefreshCw,
  Rocket,
  Search,
  ShieldCheck,
  Sparkles,
  Star,
  Upload,
  Wand2,
} from 'lucide-react';

interface TopFeature {
  feature: string;
  importance: number;
  discussed_by_author: boolean;
  discovery_note: string;
}

interface LazyPredictRun {
  id: number;
  dataset_id: number;
  target_column: string;
  task_type: string;
  is_author_primary_target: boolean;
  target_rationale: string;
  models_tested_count: number;
  best_model_name: string;
  primary_metric_name: string;
  best_score: number;
  author_baseline_score: number;
  signal_gain_delta: number;
  leaderboard: Record<string, any>[];
  top_features: TopFeature[];
  overlooked_signal_summary: string;
  execution_time_sec: number;
}

interface DatasetArtifact {
  id: number;
  dissertation_id: number;
  name: string;
  source_type: string;
  source_url?: string;
  file_path: string;
  row_count: number;
  col_count: number;
  column_metadata: {
    name: string;
    dtype: string;
    missing_pct: number;
    unique_count: number;
    role: string;
  }[];
  sample_rows: Record<string, any>[];
  lazy_runs: LazyPredictRun[];
}

interface IndustryProblem {
  id: number;
  code: string;
  title: string;
  domain: string;
  category: string;
  sponsor_or_source: string;
  tam_billions_usd: number;
  impact_scale: string;
  urgency_score: number;
  description: string;
  technical_bottleneck: string;
  desired_solution_profile: string;
  keywords: string[];
}

interface MarketOpportunity {
  id: number;
  dissertation_id: number;
  lazy_run_id?: number;
  matched_problem_id?: number;
  title: string;
  opportunity_type: string;
  untranslated_finding_summary: string;
  valley_of_death_category: string;
  valley_of_death_diagnosis: string;
  clinical_trials_check: Record<string, any>;
  opentargets_chembl_check: Record<string, any>;
  patent_and_market_check: Record<string, any>;
  data_signal_score: number;
  scientific_rigor_score: number;
  commercial_whitespace_score: number;
  giant_impact_tam_score: number;
  overall_tos_score: number;
  proposed_product_or_therapeutic: string;
  validation_roadmap: string[];
  llm_model_used: string;
  matched_problem?: IndustryProblem;
}

interface Dissertation {
  id: number;
  external_id?: string;
  title: string;
  authors: string;
  university: string;
  department?: string;
  publication_year: number;
  domain: string;
  subdomain?: string;
  source_repo: string;
  source_url?: string;
  institution_tier: string;
  scientific_rigor_score: number;
  citation_velocity: number;
  abstract: string;
  author_primary_hypothesis?: string;
  author_Statistical_method?: string;
  author_primary_target?: string;
  author_reported_baseline_metric?: string;
  author_reported_baseline_score?: number;
  extracted_claims: string[];
  data_availability_links: { source_type: string; identifier: string; url: string }[];
  key_entities: Record<string, any>;
  datasets: DatasetArtifact[];
  opportunities: MarketOpportunity[];
}

interface DeepInferenceResult {
  target_column: string;
  task_type: string;
  execution_time_sec: number;
  stage1_leakage_and_permutation_audit: {
    status: string;
    five_fold_cv_mean: number;
    five_fold_cv_std: number;
    permuted_null_baseline: number;
    empirical_signal_separation: number;
    top_feature_ablation_score: number;
    single_feature_dominance_share: number;
    audit_interpretation: string;
  };
  stage2_super_learner_ensemble: {
    architecture: string;
    metric_name: string;
    author_classical_baseline: number;
    lazypredict_best_single_model: string;
    super_learner_score: number;
    gain_over_lazypredict: number;
    total_gain_over_author: number;
  };
  stage3_mechanistic_regime_discovery: {
    primary_interacting_pair: string[];
    f1_critical_threshold: number;
    f2_critical_threshold: number;
    quadrant_response_surface: Record<string, number>;
    extracted_surrogate_decision_rules: string[];
    synergy_conclusion: string;
  };
  stage4_conformal_uncertainty: Record<string, any>;
  stage5_counterfactual_intervention: {
    prescribed_parameter_shifts: {
      feature: string;
      population_median: number;
      optimal_target_regime: number;
      intervention_action: string;
    }[];
    predicted_optimal_outcome: number;
    reproducible_python_code: string;
  };
}

export function App() {
  // 3-Step Friendly Guided Wizard State
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3>(1);
  const [step1Mode, setStep1Mode] = useState<'search' | 'upload' | 'saved'>('search');

  // Progressive Disclosure Toggles (Keep UI calm and uncluttered by default)
  const [showAllModels, setShowAllModels] = useState<boolean>(false);
  const [showDeepCheck, setShowDeepCheck] = useState<boolean>(false);
  const [showMoreDataTools, setShowMoreDataTools] = useState<boolean>(false);
  const [showAllMoonshots, setShowAllMoonshots] = useState<boolean>(false);

  const [dissertations, setDissertations] = useState<Dissertation[]>([]);
  const [opportunities, setOpportunities] = useState<MarketOpportunity[]>([]);
  const [problems, setProblems] = useState<IndustryProblem[]>([]);
  const [selectedDissId, setSelectedDissId] = useState<number | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Stage-2 Deep Verification state
  const [escalationCache, setEscalationCache] = useState<Record<number, DeepInferenceResult>>({});
  const [escalatingRunId, setEscalatingRunId] = useState<number | null>(null);

  // Custom Target Rotation state
  const [customTargetCol, setCustomTargetCol] = useState<string>('');
  const [runningCustomTarget, setRunningCustomTarget] = useState<boolean>(false);

  // Attach Zenodo / Dataset to existing dissertation
  const [zenodoInput, setZenodoInput] = useState<string>('');
  const [attachingData, setAttachingData] = useState<boolean>(false);

  // Live Harvester state
  const [harvestQuery, setHarvestQuery] = useState<string>('');
  const [harvestDomain, setHarvestDomain] = useState<string>('biomedical');
  const [harvesting, setHarvesting] = useState<boolean>(false);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);
  const [probeData, setProbeData] = useState<any>(null);
  const [probing, setProbing] = useState<boolean>(false);

  // Upload state
  const [uploadTitle, setUploadTitle] = useState<string>('');
  const [uploadUni, setUploadUni] = useState<string>('');
  const [uploadDomain, setUploadDomain] = useState<string>('biomedical');
  const [uploadTarget, setUploadTarget] = useState<string>('');
  const [uploadPdf, setUploadPdf] = useState<File | null>(null);
  const [uploadCsv, setUploadCsv] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);

  const fetchAllData = async () => {
    setErrorMsg(null);
    try {
      const [dRes, oRes, pRes] = await Promise.all([
        fetch('/api/dissertations'),
        fetch('/api/opportunities'),
        fetch('/api/problems'),
      ]);
      const dData: Dissertation[] = await dRes.json();
      const oData: MarketOpportunity[] = await oRes.json();
      const pData: IndustryProblem[] = await pRes.json();

      setDissertations(dData);
      setOpportunities(oData);
      setProblems(pData);

      if (dData.length > 0 && !selectedDissId) {
        setSelectedDissId(dData[0].id);
        const firstRun = dData[0].datasets?.[0]?.lazy_runs?.[0];
        if (firstRun) setSelectedRunId(firstRun.id);
      }
    } catch (err) {
      console.error('Error loading platform data:', err);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  const selectedDiss = dissertations.find((d) => d.id === selectedDissId) || dissertations[0];
  const activeDataset = selectedDiss?.datasets?.[0];
  const activeRun =
    activeDataset?.lazy_runs?.find((r) => r.id === selectedRunId) ||
    activeDataset?.lazy_runs?.[0];

  const handleSelectDissertation = (diss: Dissertation) => {
    setSelectedDissId(diss.id);
    const firstRun = diss.datasets?.[0]?.lazy_runs?.[0];
    if (firstRun) {
      setSelectedRunId(firstRun.id);
    }
    setCurrentStep(2);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleEscalateDeepInference = async (lazyRunId: number) => {
    setShowDeepCheck(true);
    if (escalationCache[lazyRunId]) return;
    setEscalatingRunId(lazyRunId);
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/lazypredict/${lazyRunId}/escalate`, {
        method: 'POST',
      });
      if (res.ok) {
        const data: DeepInferenceResult = await res.json();
        setEscalationCache((prev) => ({ ...prev, [lazyRunId]: data }));
      }
    } catch (err) {
      console.error('Deep inference escalation error:', err);
    } finally {
      setEscalatingRunId(null);
    }
  };

  const handleRunCustomTarget = async () => {
    if (!activeDataset || !customTargetCol) return;
    setRunningCustomTarget(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/lazypredict/run-custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: activeDataset.id,
          target_column: customTargetCol,
          is_author_primary_target: false,
          target_rationale: `Exploring column ${customTargetCol}`,
        }),
      });
      if (res.ok) {
        const newRun: LazyPredictRun = await res.json();
        await fetchAllData();
        setSelectedRunId(newRun.id);
        setShowMoreDataTools(false);
      }
    } catch (err) {
      setErrorMsg('Could not run LazyPredict on that column.');
    } finally {
      setRunningCustomTarget(false);
    }
  };

  const handleAttachZenodo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDiss || !zenodoInput.trim()) return;
    setAttachingData(true);
    setErrorMsg(null);
    try {
      const fd = new FormData();
      fd.append('zenodo_id_or_url', zenodoInput.trim());
      const res = await fetch(`/api/dissertations/${selectedDiss.id}/attach-zenodo`, {
        method: 'POST',
        body: fd,
      });
      if (!res.ok) {
        const err = await res.json();
        setErrorMsg(err.detail || 'Could not find a CSV/TSV file on that Zenodo record.');
      } else {
        await fetchAllData();
        setZenodoInput('');
      }
    } catch (err) {
      setErrorMsg('Failed to connect to Zenodo.');
    } finally {
      setAttachingData(false);
    }
  };

  const handleAttachDatasetFile = async (file: File) => {
    if (!selectedDiss) return;
    setAttachingData(true);
    setErrorMsg(null);
    try {
      const fd = new FormData();
      fd.append('dataset_file', file);
      const res = await fetch(`/api/dissertations/${selectedDiss.id}/upload-dataset`, {
        method: 'POST',
        body: fd,
      });
      if (res.ok) {
        await fetchAllData();
      }
    } catch (err) {
      setErrorMsg('Dataset upload failed.');
    } finally {
      setAttachingData(false);
    }
  };

  const handleLiveHarvest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!harvestQuery.trim()) return;
    setHarvesting(true);
    setErrorMsg(null);
    setStreamEvents(['Connecting to live OpenAlex, ClinicalTrials.gov & Zenodo streams...']);

    const url = `/api/harvest/stream?query=${encodeURIComponent(harvestQuery.trim())}&domain=${encodeURIComponent(harvestDomain)}&min_year=2018&max_results=3`;
    const evtSource = new EventSource(url);

    evtSource.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.message) {
          setStreamEvents((prev) => [...prev, data.message]);
        }
        if (data.step === 'complete') {
          evtSource.close();
          await fetchAllData();
          setHarvesting(false);
          const createdIds: number[] = data.payload?.dissertation_ids || [];
          if (createdIds.length > 0) {
            setSelectedDissId(createdIds[0]);
            setCurrentStep(2);
          }
        }
      } catch (err) {
        console.error('Stream parse error:', err);
      }
    };

    evtSource.onerror = async () => {
      evtSource.close();
      await fetchAllData();
      setHarvesting(false);
    };
  };

  const handleResetWorkspace = async () => {
    await fetch('/api/workspace/reset', { method: 'DELETE' });
    setDissertations([]);
    setOpportunities([]);
    setProblems([]);
    setSelectedDissId(null);
    setSelectedRunId(null);
    setStreamEvents([]);
    setProbeData(null);
    setCurrentStep(1);
  };

  const handleRunLiveProbe = async (q: string) => {
    const targetQ = q.trim() || 'EGFR';
    setProbing(true);
    try {
      const res = await fetch(`/api/integrations/probe?query=${encodeURIComponent(targetQ)}`);
      if (res.ok) {
        setProbeData(await res.json());
      }
    } finally {
      setProbing(false);
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadTitle.trim()) return;
    setUploading(true);
    setErrorMsg(null);
    try {
      const formData = new FormData();
      formData.append('title', uploadTitle);
      formData.append('university', uploadUni || 'Research University');
      formData.append('domain', uploadDomain);
      if (uploadTarget) formData.append('author_primary_target', uploadTarget);
      if (uploadPdf) formData.append('pdf_file', uploadPdf);
      if (uploadCsv) formData.append('dataset_file', uploadCsv);

      const res = await fetch('/api/upload-dissertation', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const created: Dissertation = await res.json();
        await fetchAllData();
        handleSelectDissertation(created);
      }
    } catch (err) {
      setErrorMsg('Upload failed. Please check your file format.');
    } finally {
      setUploading(false);
    }
  };

  const activeOpportunities = selectedDiss
    ? opportunities.filter((o) => o.dissertation_id === selectedDiss.id)
    : opportunities;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top Care Bears Rainbow Ribbon */}
      <div className="h-2 w-full care-rainbow-bar" />

      {/* Calm, Friendly Header */}
      <header className="bg-white/85 backdrop-blur-md border-b border-pink-100 sticky top-0 z-30">
        <div className="max-w-5xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-pink-400 via-amber-300 to-sky-400 flex items-center justify-center shadow-sm">
              <Heart className="w-5 h-5 text-white fill-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-slate-800">ThesisOut</span>
                <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-pink-50 text-pink-600 border border-pink-200">
                  Guided Discovery
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Your friendly companion for finding hidden gems in dissertations & datasets
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs">
            {dissertations.length > 0 && (
              <button
                onClick={handleResetWorkspace}
                className="px-3 py-1.5 rounded-full bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 font-medium transition cursor-pointer"
              >
                Clear Workspace
              </button>
            )}
            <span className="px-3 py-1.5 rounded-full bg-purple-50 text-purple-700 border border-purple-200 font-medium flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-purple-500" />
              Gemini 4 Argon + Opus 5.5
            </span>
          </div>
        </div>

        {/* 3-Step Care Bears Guided Stepper */}
        <div className="max-w-5xl mx-auto px-6 pb-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Step 1: Cheer Bear Pink */}
            <button
              onClick={() => setCurrentStep(1)}
              className={`text-left p-3.5 rounded-2xl border-2 transition flex items-center gap-3 cursor-pointer ${
                currentStep === 1
                  ? 'bg-pink-50/90 border-pink-400 shadow-xs'
                  : 'bg-white border-slate-100 hover:border-pink-200'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
                  currentStep === 1 ? 'bg-pink-500 text-white' : 'bg-pink-100 text-pink-600'
                }`}
              >
                1
              </div>
              <div>
                <div className="text-xs font-bold text-slate-800">Pick or Add a Thesis</div>
                <div className="text-[11px] text-slate-500">Search online or upload yours</div>
              </div>
            </button>

            {/* Step 2: Grumpy Bear Sky Blue */}
            <button
              onClick={() => setCurrentStep(2)}
              className={`text-left p-3.5 rounded-2xl border-2 transition flex items-center gap-3 cursor-pointer ${
                currentStep === 2
                  ? 'bg-sky-50/90 border-sky-400 shadow-xs'
                  : 'bg-white border-slate-100 hover:border-sky-200'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
                  currentStep === 2 ? 'bg-sky-500 text-white' : 'bg-sky-100 text-sky-600'
                }`}
              >
                2
              </div>
              <div>
                <div className="text-xs font-bold text-slate-800">30-Model Data Check</div>
                <div className="text-[11px] text-slate-500">Let LazyPredict spot hidden patterns</div>
              </div>
            </button>

            {/* Step 3: Share Bear Lavender */}
            <button
              onClick={() => setCurrentStep(3)}
              className={`text-left p-3.5 rounded-2xl border-2 transition flex items-center gap-3 cursor-pointer ${
                currentStep === 3
                  ? 'bg-purple-50/90 border-purple-400 shadow-xs'
                  : 'bg-white border-slate-100 hover:border-purple-200'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
                  currentStep === 3 ? 'bg-purple-500 text-white' : 'bg-purple-100 text-purple-600'
                }`}
              >
                3
              </div>
              <div>
                <div className="text-xs font-bold text-slate-800">Market & Impact Match</div>
                <div className="text-[11px] text-slate-500">See real-world problems it solves</div>
              </div>
            </button>
          </div>
        </div>
      </header>

      {/* Main Guided Workspace */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8">
        {errorMsg && (
          <div className="mb-6 bg-rose-50 border-2 border-rose-200 text-rose-700 px-5 py-3.5 rounded-2xl text-xs flex items-center justify-between">
            <span>{errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="font-bold underline cursor-pointer">
              Dismiss
            </button>
          </div>
        )}

        {/* Active Thesis Context Pill (Shown on Steps 2 & 3 so user never feels lost) */}
        {selectedDiss && currentStep > 1 && (
          <div className="mb-6 bg-white border-2 border-sky-100 rounded-2xl px-5 py-3.5 flex flex-wrap items-center justify-between gap-3 shadow-2xs">
            <div className="flex items-center gap-3">
              <span className="px-2.5 py-1 rounded-xl bg-sky-50 text-sky-700 font-semibold text-xs border border-sky-200">
                Active Thesis
              </span>
              <div>
                <div className="text-xs font-bold text-slate-800 line-clamp-1">{selectedDiss.title}</div>
                <div className="text-[11px] text-slate-500 flex flex-wrap items-center gap-2 mt-0.5">
                  <span>
                    {selectedDiss.authors} • {selectedDiss.university} ({selectedDiss.publication_year})
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-semibold">
                    Source: {selectedDiss.source_repo}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-purple-50 text-purple-700 border border-purple-200 font-semibold">
                    Rigor Confidence: {(selectedDiss.scientific_rigor_score * 100).toFixed(0)}%
                  </span>
                  {selectedDiss.source_url && (
                    <a
                      href={selectedDiss.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sky-600 font-bold underline"
                    >
                      Verify Source Link ↗
                    </a>
                  )}
                </div>
              </div>
            </div>

            {dissertations.length > 1 && (
              <select
                value={selectedDiss.id}
                onChange={(e) => {
                  const found = dissertations.find((d) => d.id === Number(e.target.value));
                  if (found) handleSelectDissertation(found);
                }}
                className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 text-xs text-slate-700"
              >
                {dissertations.map((d) => (
                  <option key={d.id} value={d.id}>
                    Switch: {d.title.slice(0, 45)}...
                  </option>
                ))}
              </select>
            )}
          </div>
        )}

        {/* ===================================================================== */}
        {/* STEP 1: PICK OR ADD A THESIS (CHEER BEAR PINK ACCENTS) */}
        {/* ===================================================================== */}
        {currentStep === 1 && (
          <div className="space-y-6">
            <div className="text-center max-w-xl mx-auto">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-pink-100 text-pink-700 text-xs font-semibold mb-2">
                <Star className="w-3.5 h-3.5 fill-pink-500 text-pink-500" />
                Step 1 of 3 • Let's Get Started
              </span>
              <h1 className="text-2xl font-bold text-slate-800">
                How would you like to start your discovery?
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Pick one simple option below. We will automatically handle the dataset matching and model testing for you!
              </p>
            </div>

            {/* Friendly 3-Option Mode Switcher */}
            <div className="flex justify-center">
              <div className="inline-flex bg-white p-1.5 rounded-2xl border-2 border-pink-100 shadow-xs gap-1">
                <button
                  onClick={() => setStep1Mode('search')}
                  className={`px-4 py-2 rounded-xl text-xs font-bold transition cursor-pointer ${
                    step1Mode === 'search'
                      ? 'bg-pink-500 text-white shadow-xs'
                      : 'text-slate-600 hover:bg-pink-50'
                  }`}
                >
                  🔍 Search Online (OpenAlex + Zenodo)
                </button>
                <button
                  onClick={() => setStep1Mode('upload')}
                  className={`px-4 py-2 rounded-xl text-xs font-bold transition cursor-pointer ${
                    step1Mode === 'upload'
                      ? 'bg-sky-500 text-white shadow-xs'
                      : 'text-slate-600 hover:bg-sky-50'
                  }`}
                >
                  📄 Upload My Own PDF / CSV
                </button>
                {dissertations.length > 0 && (
                  <button
                    onClick={() => setStep1Mode('saved')}
                    className={`px-4 py-2 rounded-xl text-xs font-bold transition cursor-pointer ${
                      step1Mode === 'saved'
                        ? 'bg-emerald-500 text-white shadow-xs'
                        : 'text-slate-600 hover:bg-emerald-50'
                    }`}
                  >
                    📚 My Saved Theses ({dissertations.length})
                  </button>
                )}
              </div>
            </div>

            {/* MODE A: Search Online */}
            {step1Mode === 'search' && (
              <div className="care-card border-2 border-pink-200 rounded-3xl p-8 max-w-2xl mx-auto">
                <div className="flex items-center gap-2.5 text-pink-600 font-bold text-base mb-1">
                  <Search className="w-5 h-5" />
                  Find Real Dissertations & Open Datasets
                </div>
                <p className="text-xs text-slate-500 mb-5 leading-relaxed">
                  Type any research topic or click a quick suggestion below. We'll search live university repositories (OpenAlex) and open data archives (Zenodo) for real datasets, then test 30+ models automatically.
                </p>

                {/* Quick Topic Chips */}
                <div className="flex flex-wrap items-center gap-2 mb-5">
                  <span className="text-[11px] font-semibold text-slate-400">Try a topic:</span>
                  {[
                    { label: '🎗️ Breast Cancer', q: 'breast cancer', dom: 'biomedical' },
                    { label: '🩺 Sepsis Biomarkers', q: 'sepsis', dom: 'biomedical' },
                    { label: '🧬 CRISPR Delivery', q: 'CRISPR delivery', dom: 'biomedical' },
                    { label: '🔋 Battery Degradation', q: 'battery degradation', dom: 'cs_ai' },
                    { label: '🤖 Transformer Inference', q: 'transformer attention', dom: 'cs_ai' },
                  ].map((chip) => (
                    <button
                      key={chip.label}
                      type="button"
                      onClick={() => {
                        setHarvestQuery(chip.q);
                        setHarvestDomain(chip.dom);
                      }}
                      className="px-3 py-1 rounded-full bg-pink-50 hover:bg-pink-100 text-pink-700 border border-pink-200 text-xs font-medium transition cursor-pointer"
                    >
                      {chip.label}
                    </button>
                  ))}
                </div>

                <form onSubmit={handleLiveHarvest} className="space-y-4 text-xs">
                  <div>
                    <label className="block font-bold text-slate-700 mb-1.5">What topic are you exploring?</label>
                    <input
                      type="text"
                      value={harvestQuery}
                      onChange={(e) => setHarvestQuery(e.target.value)}
                      placeholder="e.g., breast cancer, Alzheimer's, solid-state battery..."
                      className="w-full bg-white border-2 border-slate-200 focus:border-pink-400 rounded-2xl px-4 py-3 text-sm text-slate-800 focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block font-bold text-slate-700 mb-1.5">Research Field</label>
                    <select
                      value={harvestDomain}
                      onChange={(e) => setHarvestDomain(e.target.value)}
                      className="w-full bg-white border-2 border-slate-200 focus:border-pink-400 rounded-2xl px-4 py-3 text-sm text-slate-800 focus:outline-none"
                    >
                      <option value="biomedical">🧬 Biomedical, Life Sciences & Healthcare</option>
                      <option value="cs_ai">💻 Computer Science, AI & Engineering</option>
                    </select>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-2.5">
                    <button
                      type="submit"
                      disabled={harvesting || !harvestQuery.trim()}
                      className="flex-1 py-3.5 px-6 rounded-2xl bg-gradient-to-r from-pink-500 to-rose-400 hover:from-pink-600 hover:to-rose-500 disabled:opacity-50 text-white font-bold text-sm flex items-center justify-center gap-2 shadow-md shadow-pink-200 transition cursor-pointer"
                    >
                      {harvesting ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Streaming OpenAlex, EuropePMC, arXiv, ClinicalTrials & Zenodo...
                        </>
                      ) : (
                        <>
                          <Wand2 className="w-4 h-4" />
                          Stream Real Theses & Auto-Test Data →
                        </>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={() => handleRunLiveProbe(harvestQuery || 'EGFR')}
                      disabled={probing}
                      className="py-3.5 px-4 rounded-2xl bg-purple-50 hover:bg-purple-100 text-purple-700 border-2 border-purple-200 font-bold text-xs flex items-center justify-center gap-1.5 transition cursor-pointer shrink-0"
                    >
                      {probing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                      Probe 9 Live Science APIs
                    </button>
                  </div>
                </form>

                {/* Live SSE Stream Progress Log */}
                {streamEvents.length > 0 && (
                  <div className="mt-5 bg-sky-50/70 border border-sky-200 rounded-2xl p-4 space-y-1.5 text-xs">
                    <div className="font-bold text-sky-800 uppercase tracking-wider text-[10px]">
                      Live Multi-API Stream Telemetry
                    </div>
                    {streamEvents.map((msg, i) => (
                      <div key={i} className="flex items-center gap-2 text-slate-700">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
                        <span>{msg}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Live Multi-Database Integration Telemetry Drawer */}
                {probeData && (
                  <div className="mt-5 bg-purple-50/60 border-2 border-purple-200 rounded-2xl p-4 space-y-3 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-purple-900">
                        Live Scientific API Telemetry for "{probeData.query}"
                      </span>
                      <button
                        onClick={() => setProbeData(null)}
                        className="text-purple-600 font-semibold underline cursor-pointer"
                      >
                        Close
                      </button>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                      <div className="bg-white border border-purple-100 rounded-xl p-3">
                        <div className="font-bold text-pink-600">OpenTargets GraphQL v4</div>
                        <div className="text-slate-700 mt-0.5">
                          Target: {probeData.live_integrations?.opentargets_graphql_v4?.target_profile?.symbol || 'Queried'} (
                          {probeData.live_integrations?.opentargets_graphql_v4?.target_profile?.ensembl_id || 'Live Hit'})
                        </div>
                        <div className="text-slate-500 text-[11px]">
                          Known Drugs: {probeData.live_integrations?.opentargets_graphql_v4?.target_profile?.unique_known_drugs ?? 0}
                        </div>
                      </div>
                      <div className="bg-white border border-purple-100 rounded-xl p-3">
                        <div className="font-bold text-emerald-600">ChEMBL EBI REST API</div>
                        <div className="text-slate-700 mt-0.5">
                          Targets Found: {probeData.live_integrations?.chembl_ebi_rest?.targets_found?.length || 0} • Molecules:{' '}
                          {probeData.live_integrations?.chembl_ebi_rest?.molecules_found?.length || 0}
                        </div>
                        <div className="text-slate-500 text-[11px]">
                          Max Clinical Phase: {probeData.live_integrations?.chembl_ebi_rest?.max_clinical_phase_observed ?? 0}
                        </div>
                      </div>
                      <div className="bg-white border border-purple-100 rounded-xl p-3">
                        <div className="font-bold text-sky-600">ClinicalTrials.gov API v2 & openFDA</div>
                        <div className="text-slate-700 mt-0.5">
                          Trials: {probeData.live_integrations?.clinicaltrials_gov_v2?.total_trials ?? 0}
                        </div>
                        <div className="text-slate-500 text-[11px]">
                          openFDA FAERS Reports: {probeData.live_integrations?.openfda_faers?.total_adverse_event_reports ?? 0}
                        </div>
                      </div>
                      <div className="bg-white border border-purple-100 rounded-xl p-3">
                        <div className="font-bold text-amber-600">EuropePMC, arXiv & HuggingFace</div>
                        <div className="text-slate-700 mt-0.5">
                          EuropePMC Hits: {probeData.live_integrations?.europepmc_rest?.length || 0} • arXiv:{' '}
                          {probeData.live_integrations?.arxiv_atom_api?.length || 0}
                        </div>
                        <div className="text-slate-500 text-[11px]">
                          HF Datasets: {probeData.live_integrations?.huggingface_hub_datasets?.length || 0}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* MODE B: Upload My Own PDF / CSV */}
            {step1Mode === 'upload' && (
              <div className="care-card border-2 border-sky-200 rounded-3xl p-8 max-w-2xl mx-auto">
                <div className="flex items-center gap-2.5 text-sky-600 font-bold text-base mb-1">
                  <Upload className="w-5 h-5" />
                  Upload Your Dissertation PDF or Dataset
                </div>
                <p className="text-xs text-slate-500 mb-5 leading-relaxed">
                  Have a CSV/Excel dataset or a dissertation PDF on your computer? Drop it here and we'll automatically run all 30+ LazyPredict models and take you to Step 2.
                </p>

                <form onSubmit={handleUploadSubmit} className="space-y-4 text-xs">
                  <div>
                    <label className="block font-bold text-slate-700 mb-1">Title of the Study *</label>
                    <input
                      type="text"
                      required
                      value={uploadTitle}
                      onChange={(e) => setUploadTitle(e.target.value)}
                      placeholder="Give your study or dissertation a title"
                      className="w-full bg-white border-2 border-slate-200 focus:border-sky-400 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block font-bold text-slate-700 mb-1">University (Optional)</label>
                      <input
                        type="text"
                        value={uploadUni}
                        onChange={(e) => setUploadUni(e.target.value)}
                        placeholder="e.g., MIT, Stanford"
                        className="w-full bg-white border-2 border-slate-200 focus:border-sky-400 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block font-bold text-slate-700 mb-1">Field</label>
                      <select
                        value={uploadDomain}
                        onChange={(e) => setUploadDomain(e.target.value)}
                        className="w-full bg-white border-2 border-slate-200 focus:border-sky-400 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none"
                      >
                        <option value="biomedical">🧬 Biomedical & Healthcare</option>
                        <option value="cs_ai">💻 CS, AI & Engineering</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                    <div className="border-2 border-dashed border-sky-200 rounded-2xl p-4 bg-sky-50/40">
                      <label className="block font-bold text-sky-800 mb-1">📊 Dataset File (.csv, .xlsx, .tsv)</label>
                      <input
                        type="file"
                        accept=".csv,.tsv,.xlsx,.parquet"
                        onChange={(e) => setUploadCsv(e.target.files?.[0] || null)}
                        className="w-full text-slate-600 text-[11px]"
                      />
                    </div>
                    <div className="border-2 border-dashed border-pink-200 rounded-2xl p-4 bg-pink-50/40">
                      <label className="block font-bold text-pink-800 mb-1">📄 Dissertation PDF (Optional)</label>
                      <input
                        type="file"
                        accept=".pdf"
                        onChange={(e) => setUploadPdf(e.target.files?.[0] || null)}
                        className="w-full text-slate-600 text-[11px]"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block font-bold text-slate-700 mb-1">
                      Column to Predict (Leave blank and we'll pick it automatically!)
                    </label>
                    <input
                      type="text"
                      value={uploadTarget}
                      onChange={(e) => setUploadTarget(e.target.value)}
                      placeholder="Optional column name"
                      className="w-full bg-white border-2 border-slate-200 focus:border-sky-400 rounded-2xl px-4 py-2.5 text-slate-800 focus:outline-none"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={uploading}
                    className="w-full py-3.5 px-6 rounded-2xl bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-600 hover:to-cyan-600 text-white font-bold text-sm flex items-center justify-center gap-2 shadow-md shadow-sky-200 transition cursor-pointer"
                  >
                    {uploading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Analyzing Dataset with 30+ Models...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4" />
                        Upload & Go to Step 2 →
                      </>
                    )}
                  </button>
                </form>
              </div>
            )}

            {/* MODE C: Saved Theses */}
            {step1Mode === 'saved' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {dissertations.map((diss) => {
                  const run = diss.datasets?.[0]?.lazy_runs?.[0];
                  return (
                    <div
                      key={diss.id}
                      onClick={() => handleSelectDissertation(diss)}
                      className="care-card border-2 border-emerald-200 hover:border-emerald-400 rounded-3xl p-6 transition cursor-pointer flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex items-center justify-between text-xs mb-2">
                          <span className="px-2.5 py-0.5 rounded-full bg-pink-50 text-pink-700 font-semibold border border-pink-200">
                            {diss.university}
                          </span>
                          <span className="text-slate-400 font-medium">{diss.publication_year}</span>
                        </div>
                        <h3 className="text-base font-bold text-slate-800 leading-snug">{diss.title}</h3>
                        <p className="text-xs text-slate-500 mt-2 line-clamp-2">{diss.abstract}</p>
                      </div>

                      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                        {run ? (
                          <span className="px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200">
                            ✨ Best Model: {run.best_model_name} ({run.best_score})
                          </span>
                        ) : (
                          <span className="text-slate-400">Click to attach data</span>
                        )}
                        <span className="font-bold text-sky-600 flex items-center gap-1">
                          See Step 2 <ArrowRight className="w-3.5 h-3.5" />
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ===================================================================== */}
        {/* STEP 2: 30-MODEL DATA CHECK (SKY BLUE, MINT & FUNSHINE YELLOW) */}
        {/* ===================================================================== */}
        {currentStep === 2 && (
          <div className="space-y-6">
            {!selectedDiss ? (
              <div className="care-card border-2 border-sky-200 rounded-3xl p-10 text-center max-w-lg mx-auto">
                <BookOpen className="w-10 h-10 text-sky-400 mx-auto mb-3" />
                <h3 className="text-lg font-bold text-slate-800">Let's Pick a Thesis First!</h3>
                <p className="text-xs text-slate-500 mt-1 mb-5">
                  Head back to Step 1 to search for a real dissertation or upload your dataset.
                </p>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="px-5 py-2.5 rounded-2xl bg-pink-500 text-white text-xs font-bold cursor-pointer"
                >
                  ← Go to Step 1
                </button>
              </div>
            ) : !activeRun ? (
              <div className="care-card border-2 border-amber-200 rounded-3xl p-8 max-w-xl mx-auto space-y-5">
                <div className="text-center">
                  <span className="px-3 py-1 rounded-full bg-amber-100 text-amber-800 text-xs font-bold">
                    Almost Ready!
                  </span>
                  <h3 className="text-lg font-bold text-slate-800 mt-2">
                    Attach a Dataset to Test 30+ Models on This Thesis
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Upload a CSV/Excel table or paste a Zenodo Record ID below and LazyPredict will test 30+ machine learning models automatically.
                  </p>
                </div>

                <form onSubmit={handleAttachZenodo} className="flex gap-2">
                  <input
                    type="text"
                    value={zenodoInput}
                    onChange={(e) => setZenodoInput(e.target.value)}
                    placeholder="Paste Zenodo Record ID or URL..."
                    className="flex-1 bg-white border-2 border-slate-200 rounded-2xl px-4 py-2.5 text-xs"
                  />
                  <button
                    type="submit"
                    disabled={attachingData}
                    className="px-4 py-2.5 rounded-2xl bg-sky-500 text-white text-xs font-bold cursor-pointer"
                  >
                    {attachingData ? 'Fetching...' : 'Fetch Zenodo CSV'}
                  </button>
                </form>

                <div className="border-2 border-dashed border-emerald-200 rounded-2xl p-4 bg-emerald-50/40 text-center">
                  <label className="block text-xs font-bold text-emerald-800 mb-2">
                    Or Upload a Dataset File (.csv, .tsv, .xlsx, .parquet)
                  </label>
                  <input
                    type="file"
                    accept=".csv,.tsv,.xlsx,.parquet"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleAttachDatasetFile(f);
                    }}
                    className="text-xs text-slate-600"
                  />
                </div>
              </div>
            ) : (
              <>
                {/* Funshine Bear Yellow "Plain-English Takeaway" Hero Banner */}
                <div className="bg-amber-50/90 border-2 border-amber-200 rounded-3xl p-6 shadow-xs">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="space-y-1.5">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-200/70 text-amber-900 text-xs font-bold">
                          <Sparkles className="w-3.5 h-3.5" />
                          Discovery & Confidence Summary
                        </div>
                        {activeDataset && (
                          <span className="px-3 py-1 rounded-full bg-white border border-amber-300 text-slate-700 text-xs font-semibold">
                            Dataset Source: {activeDataset.source_type.toUpperCase()} ({activeDataset.row_count} rows × {activeDataset.col_count} cols)
                          </span>
                        )}
                        {activeDataset?.source_url && (
                          <a
                            href={activeDataset.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="px-3 py-1 rounded-full bg-sky-100 text-sky-800 text-xs font-bold underline"
                          >
                            Dataset Origin Link ↗
                          </a>
                        )}
                      </div>
                      <h2 className="text-lg font-bold text-slate-800">
                        We tested {activeRun.models_tested_count} machine learning models on{' '}
                        <code className="px-2 py-0.5 rounded-lg bg-white border border-amber-300 text-pink-600">
                          {activeRun.target_column}
                        </code>
                      </h2>
                      <p className="text-sm text-slate-700 leading-relaxed">
                        {activeRun.overlooked_signal_summary}
                      </p>
                    </div>

                    <button
                      onClick={() => {
                        setCurrentStep(3);
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                      className="px-5 py-3.5 rounded-2xl bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-bold text-xs flex items-center gap-2 shrink-0 shadow-md shadow-purple-200 transition cursor-pointer"
                    >
                      <Rocket className="w-4 h-4" />
                      Next: See Market & Impact Match →
                    </button>
                  </div>

                  {/* Simple Target Switcher Pills if multiple columns were tested */}
                  {activeDataset && activeDataset.lazy_runs.length > 1 && (
                    <div className="mt-4 pt-3 border-t border-amber-200/70 flex flex-wrap items-center gap-2 text-xs">
                      <span className="font-semibold text-amber-900">Tested Columns:</span>
                      {activeDataset.lazy_runs.map((run) => (
                        <button
                          key={run.id}
                          onClick={() => setSelectedRunId(run.id)}
                          className={`px-3 py-1 rounded-full font-bold transition cursor-pointer ${
                            activeRun.id === run.id
                              ? 'bg-slate-800 text-white'
                              : 'bg-white text-slate-700 border border-amber-300 hover:bg-amber-100'
                          }`}
                        >
                          {run.target_column} ({run.best_score})
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* 3 Friendly Summary Cards (No Wall of Numbers!) */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  {/* Card 1: Grumpy Sky Blue — Classical Baseline */}
                  <div className="bg-sky-50/70 border-2 border-sky-200 rounded-3xl p-5 flex flex-col justify-between">
                    <div>
                      <span className="text-[11px] font-bold uppercase tracking-wider text-sky-700">
                        1. Standard Linear Baseline
                      </span>
                      <div className="text-3xl font-extrabold text-slate-800 mt-2">
                        {activeRun.author_baseline_score}
                      </div>
                      <div className="text-xs font-medium text-sky-800 mt-0.5">
                        Metric: {activeRun.primary_metric_name}
                      </div>
                    </div>
                    <p className="text-xs text-slate-600 mt-4 leading-relaxed">
                      What standard linear statistical methods achieve on this column without nonlinear pattern recognition.
                    </p>
                  </div>

                  {/* Card 2: Good Luck Mint — Winning LazyPredict Model */}
                  <div className="bg-emerald-50/80 border-2 border-emerald-300 rounded-3xl p-5 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-700">
                          2. Best Model Found
                        </span>
                        <span className="px-2.5 py-0.5 rounded-full bg-emerald-200 text-emerald-900 text-[11px] font-extrabold">
                          +{(activeRun.signal_gain_delta * 100).toFixed(1)}% Boost
                        </span>
                      </div>
                      <div className="text-3xl font-extrabold text-emerald-700 mt-2">
                        {activeRun.best_score}
                      </div>
                      <div className="text-xs font-bold text-slate-800 mt-0.5">
                        Winner: {activeRun.best_model_name}
                      </div>
                    </div>
                    <p className="text-xs text-slate-600 mt-4 leading-relaxed">
                      Automatically selected after testing {activeRun.models_tested_count} different machine learning algorithms side-by-side.
                    </p>
                  </div>

                  {/* Card 3: Cheer Bear Pink — Top 3 Clues */}
                  <div className="bg-pink-50/70 border-2 border-pink-200 rounded-3xl p-5">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-pink-700">
                      3. Top 3 Clues Driving the Result
                    </span>
                    <div className="space-y-3 mt-3">
                      {activeRun.top_features.slice(0, 3).map((tf, i) => (
                        <div key={tf.feature} className="bg-white/90 border border-pink-100 rounded-2xl p-2.5">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-bold text-slate-800 truncate pr-2">
                              #{i + 1} {tf.feature}
                            </span>
                            <span className="font-bold text-pink-600 shrink-0">
                              {(tf.importance * 100).toFixed(0)}% impact
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-pink-100 rounded-full overflow-hidden mt-1.5">
                            <div
                              className="h-full bg-pink-400 rounded-full"
                              style={{ width: `${Math.min(100, tf.importance * 180)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* PROGRESSIVE DISCLOSURE DRAWERS (Optional Deep-Dives) */}
                <div className="space-y-3 pt-2">
                  {/* Drawer 1: Double-Check This Discovery (Stage-2 Deep Verification) */}
                  <div className="bg-white border-2 border-purple-200 rounded-3xl overflow-hidden">
                    <button
                      onClick={() => handleEscalateDeepInference(activeRun.id)}
                      className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-purple-50/50 transition cursor-pointer"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-xl bg-purple-100 text-purple-600 flex items-center justify-center">
                          <ShieldCheck className="w-5 h-5" />
                        </div>
                        <div>
                          <div className="text-sm font-bold text-slate-800">
                            🛡️ Double-Check This Discovery (Run 5-Stage Deep Verification)
                          </div>
                          <div className="text-xs text-slate-500">
                            Makes sure the signal is real (not random luck), combines top models into a team, and finds sweet-spot thresholds
                          </div>
                        </div>
                      </div>
                      <span className="px-3 py-1.5 rounded-xl bg-purple-100 text-purple-700 text-xs font-bold shrink-0">
                        {escalatingRunId === activeRun.id
                          ? 'Checking...'
                          : escalationCache[activeRun.id]
                          ? 'Verified ✓'
                          : 'Run Check'}
                      </span>
                    </button>

                    {showDeepCheck && escalationCache[activeRun.id] && (
                      <div className="px-6 pb-6 pt-2 border-t border-purple-100 bg-purple-50/30 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        {(() => {
                          const esc = escalationCache[activeRun.id];
                          return (
                            <>
                              <div className="bg-white border border-purple-200 rounded-2xl p-4">
                                <div className="font-bold text-purple-700 mb-1">
                                  ✓ Real Signal vs. Random Luck Check
                                </div>
                                <p className="text-slate-600 leading-relaxed">
                                  {esc.stage1_leakage_and_permutation_audit.audit_interpretation}
                                </p>
                              </div>

                              <div className="bg-white border border-emerald-200 rounded-2xl p-4">
                                <div className="font-bold text-emerald-700 mb-1">
                                  🤝 Super-Learner Team Score: {esc.stage2_super_learner_ensemble.super_learner_score}
                                </div>
                                <p className="text-slate-600 leading-relaxed">
                                  Combining LightGBM, XGBoost, ExtraTrees, and Random Forest together reached{' '}
                                  <strong>{esc.stage2_super_learner_ensemble.super_learner_score}</strong>.{' '}
                                  {esc.stage4_conformal_uncertainty.ood_note}
                                </p>
                              </div>

                              <div className="bg-white border border-amber-200 rounded-2xl p-4">
                                <div className="font-bold text-amber-700 mb-1">
                                  🎯 Sweet-Spot Thresholds Found
                                </div>
                                <p className="text-slate-600 leading-relaxed">
                                  {esc.stage3_mechanistic_regime_discovery.synergy_conclusion}
                                </p>
                              </div>

                              <div className="bg-white border border-pink-200 rounded-2xl p-4">
                                <div className="font-bold text-pink-700 mb-1">
                                  💡 Recommended "What-If" Adjustments
                                </div>
                                <ul className="space-y-1 text-slate-600 mt-1">
                                  {esc.stage5_counterfactual_intervention.prescribed_parameter_shifts.map((s, idx) => (
                                    <li key={idx}>• {s.intervention_action}</li>
                                  ))}
                                </ul>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                    )}
                  </div>

                  {/* Drawer 2: Peek inside all 30+ tested models */}
                  <div className="bg-white border-2 border-sky-200 rounded-3xl overflow-hidden">
                    <button
                      onClick={() => setShowAllModels(!showAllModels)}
                      className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-sky-50/50 transition cursor-pointer"
                    >
                      <div className="text-sm font-bold text-slate-800">
                        📊 Peek Inside All {activeRun.leaderboard.length} Tested Models (Full Leaderboard)
                      </div>
                      {showAllModels ? (
                        <ChevronUp className="w-5 h-5 text-slate-400" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-slate-400" />
                      )}
                    </button>

                    {showAllModels && (
                      <div className="px-6 pb-6 pt-2 border-t border-sky-100 overflow-x-auto max-h-80">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="border-b border-slate-200 text-slate-500">
                              <th className="py-2 px-3">Rank</th>
                              {Object.keys(activeRun.leaderboard[0] || {}).map((col) => (
                                <th key={col} className="py-2 px-3">
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 font-mono">
                            {activeRun.leaderboard.map((row, idx) => (
                              <tr
                                key={idx}
                                className={idx === 0 ? 'bg-emerald-50 font-bold text-emerald-900' : 'text-slate-700'}
                              >
                                <td className="py-2 px-3">#{idx + 1}</td>
                                {Object.entries(row).map(([k, val]) => (
                                  <td key={k} className="py-2 px-3">
                                    {val === null ? '—' : String(val)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>

                  {/* Drawer 3: Try predicting a different column */}
                  {activeDataset && (
                    <div className="bg-white border-2 border-amber-200 rounded-3xl overflow-hidden">
                      <button
                        onClick={() => setShowMoreDataTools(!showMoreDataTools)}
                        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-amber-50/50 transition cursor-pointer"
                      >
                        <div className="text-sm font-bold text-slate-800">
                          🔄 Curious about another column in this dataset? Test it here
                        </div>
                        {showMoreDataTools ? (
                          <ChevronUp className="w-5 h-5 text-slate-400" />
                        ) : (
                          <ChevronDown className="w-5 h-5 text-slate-400" />
                        )}
                      </button>

                      {showMoreDataTools && (
                        <div className="px-6 pb-6 pt-3 border-t border-amber-100 flex flex-wrap items-center gap-3 text-xs">
                          <select
                            value={customTargetCol}
                            onChange={(e) => setCustomTargetCol(e.target.value)}
                            className="flex-1 bg-slate-50 border-2 border-slate-200 rounded-2xl px-4 py-2.5 text-slate-800"
                          >
                            <option value="">Choose any column from {activeDataset.name}...</option>
                            {activeDataset.column_metadata
                              .filter((c) => c.role !== 'identifier')
                              .map((c) => (
                                <option key={c.name} value={c.name}>
                                  {c.name} ({c.dtype})
                                </option>
                              ))}
                          </select>
                          <button
                            onClick={handleRunCustomTarget}
                            disabled={!customTargetCol || runningCustomTarget}
                            className="px-5 py-2.5 rounded-2xl bg-amber-500 hover:bg-amber-600 disabled:opacity-40 text-white font-bold flex items-center gap-1.5 cursor-pointer"
                          >
                            {runningCustomTarget ? (
                              <RefreshCw className="w-4 h-4 animate-spin" />
                            ) : (
                              <Play className="w-4 h-4" />
                            )}
                            Run 30-Model Check
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Friendly Bottom Step Navigation */}
                <div className="flex items-center justify-between pt-4">
                  <button
                    onClick={() => setCurrentStep(1)}
                    className="px-4 py-2.5 rounded-2xl bg-white border-2 border-slate-200 text-slate-700 text-xs font-bold flex items-center gap-1.5 cursor-pointer"
                  >
                    <ArrowLeft className="w-4 h-4" /> Back to Step 1
                  </button>
                  <button
                    onClick={() => {
                      setCurrentStep(3);
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }}
                    className="px-6 py-3 rounded-2xl bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs font-bold flex items-center gap-2 shadow-md shadow-purple-200 cursor-pointer"
                  >
                    Continue to Step 3: Market & Impact Match <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* ===================================================================== */}
        {/* STEP 3: MARKET & IMPACT MATCH (SHARE BEAR LAVENDER & PEACH) */}
        {/* ===================================================================== */}
        {currentStep === 3 && (
          <div className="space-y-6">
            <div className="text-center max-w-xl mx-auto">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-semibold mb-2">
                <Compass className="w-3.5 h-3.5 text-purple-600" />
                Step 3 of 3 • Turning Data into Real-World Solutions
              </span>
              <h2 className="text-2xl font-bold text-slate-800">
                Untranslated Market Solutions & Moonshot Matches
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Here is how the hidden signal found in Step 2 can solve major real-world industry problems that haven't been commercialized yet.
              </p>
            </div>

            {activeOpportunities.length === 0 ? (
              <div className="care-card border-2 border-purple-200 rounded-3xl p-10 text-center max-w-lg mx-auto">
                <Rocket className="w-10 h-10 text-purple-400 mx-auto mb-3" />
                <h3 className="text-base font-bold text-slate-800">No Market Dossiers Yet</h3>
                <p className="text-xs text-slate-500 mt-1 mb-5">
                  Run a dataset through Step 2 first, and we will automatically match its findings to unmet clinical and engineering problems!
                </p>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="px-5 py-2.5 rounded-2xl bg-pink-500 text-white text-xs font-bold cursor-pointer"
                >
                  Start at Step 1
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                {activeOpportunities.map((opp) => (
                  <div
                    key={opp.id}
                    className="care-card border-2 border-purple-200 rounded-3xl p-6 space-y-5"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <span className="px-3 py-1 rounded-full bg-emerald-100 text-emerald-800 text-xs font-extrabold border border-emerald-300">
                        🌟 Translation Opportunity Score: {opp.overall_tos_score} / 100
                      </span>
                      <span className="px-3 py-1 rounded-full bg-purple-50 text-purple-700 text-xs font-semibold border border-purple-200">
                        {opp.opportunity_type}
                      </span>
                    </div>

                    <h3 className="text-lg font-bold text-slate-800 leading-snug">{opp.title}</h3>

                    {/* 3 Friendly Story Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                      <div className="bg-pink-50/60 border border-pink-200 rounded-2xl p-4">
                        <div className="font-bold text-pink-700 mb-1.5">✨ 1. What the Data Revealed</div>
                        <p className="text-slate-700 leading-relaxed">{opp.untranslated_finding_summary}</p>
                      </div>

                      <div className="bg-amber-50/60 border border-amber-200 rounded-2xl p-4">
                        <div className="font-bold text-amber-800 mb-1.5">🌉 2. Why It Got Stuck in Academia</div>
                        <p className="text-slate-700 leading-relaxed">{opp.valley_of_death_diagnosis}</p>
                      </div>

                      <div className="bg-sky-50/60 border border-sky-200 rounded-2xl p-4">
                        <div className="font-bold text-sky-800 mb-1.5">🚀 3. Market & Clinical Whitespace</div>
                        <p className="text-slate-700 leading-relaxed">
                          <strong>ClinicalTrials.gov:</strong> {opp.clinical_trials_check?.whitespace_status}
                        </p>
                        <p className="text-slate-700 mt-1.5 leading-relaxed">
                          <strong>Target / Modality:</strong> {opp.opentargets_chembl_check?.tractability_modality}
                        </p>
                      </div>
                    </div>

                    {/* Proposed Solution & 4-Step Action Plan */}
                    <div className="bg-emerald-50/60 border border-emerald-200 rounded-2xl p-5 text-xs">
                      <div className="font-bold text-emerald-800 uppercase tracking-wider text-[11px] mb-1">
                        Recommended Real-World Solution & Next Steps
                      </div>
                      <p className="text-slate-800 font-semibold text-sm mb-3">
                        {opp.proposed_product_or_therapeutic}
                      </p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                        {opp.validation_roadmap.map((step, i) => (
                          <div
                            key={i}
                            className="bg-white border border-emerald-200 rounded-xl p-3 text-slate-700 flex items-start gap-2"
                          >
                            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                            <span>{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Collapsible Moonshot Problem Bank Drawer (Tenderheart Peach) */}
            <div className="bg-white border-2 border-orange-200 rounded-3xl overflow-hidden">
              <button
                onClick={() => setShowAllMoonshots(!showAllMoonshots)}
                className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-orange-50/50 transition cursor-pointer"
              >
                <div>
                  <div className="text-sm font-bold text-slate-800">
                    🌍 Browse All {problems.length} Giant-Impact Moonshot Problems ($2.5T+ Market Need)
                  </div>
                  <div className="text-xs text-slate-500">
                    See the full list of major biomedical and engineering challenges we match dissertations against
                  </div>
                </div>
                {showAllMoonshots ? (
                  <ChevronUp className="w-5 h-5 text-slate-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-slate-400" />
                )}
              </button>

              {showAllMoonshots && (
                <div className="p-6 border-t border-orange-100 bg-orange-50/20 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  {problems.map((prob) => (
                    <div key={prob.id} className="bg-white border border-orange-200 rounded-2xl p-4">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="px-2.5 py-0.5 rounded-full bg-orange-100 text-orange-800 font-bold">
                          {prob.code} • ${prob.tam_billions_usd}B Need
                        </span>
                        <span className="text-slate-400">{prob.sponsor_or_source}</span>
                      </div>
                      <div className="font-bold text-slate-800 text-sm">{prob.title}</div>
                      <p className="text-slate-600 mt-1.5 leading-relaxed">{prob.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-2">
              <button
                onClick={() => setCurrentStep(2)}
                className="px-4 py-2.5 rounded-2xl bg-white border-2 border-slate-200 text-slate-700 text-xs font-bold flex items-center gap-1.5 cursor-pointer"
              >
                <ArrowLeft className="w-4 h-4" /> Back to Step 2 (Data Check)
              </button>
              <button
                onClick={() => setCurrentStep(1)}
                className="px-5 py-2.5 rounded-2xl bg-pink-500 hover:bg-pink-600 text-white text-xs font-bold cursor-pointer"
              >
                ✨ Explore Another Dissertation (Step 1)
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
