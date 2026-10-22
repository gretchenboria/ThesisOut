import React, { useEffect, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Compass,
  Download,
  ExternalLink,
  FileText,
  MessageSquare,
  Play,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Upload,
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
  author_statistical_method?: string;
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

interface CopilotAction {
  action_type:
    | 'harvest_stream'
    | 'probe_apis'
    | 'run_lazypredict_column'
    | 'run_deep_verification'
    | 'follow_up_prompt';
  label: string;
  query_or_column: string;
  domain?: string;
}

interface CopilotMessage {
  role: 'user' | 'assistant';
  content: string;
  hypothesis_formulation?: string;
  downstream_inference_plan?: string[];
  suggested_actions?: CopilotAction[];
  model_used?: string;
}

function inferDomainFromText(text: string): string {
  const lower = text.toLowerCase();
  const csKeywords = [
    'transformer',
    'battery',
    'neural',
    'compiler',
    'quantum',
    'robotics',
    'silicon',
    'algorithm',
    'latency',
    'gpu',
    'semiconductor',
    'grid',
  ];
  return csKeywords.some((k) => lower.includes(k)) ? 'cs_ai' : 'biomedical';
}

export function App() {
  const [currentStep, setCurrentStep] = useState<1 | 2 | 3>(1);
  const [step1Mode, setStep1Mode] = useState<'copilot' | 'upload' | 'saved'>('copilot');

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

  const [escalationCache, setEscalationCache] = useState<Record<number, DeepInferenceResult>>({});
  const [escalatingRunId, setEscalatingRunId] = useState<number | null>(null);
  const [stepActionResults, setStepActionResults] = useState<Record<string, any>>({});
  const [executingStepKey, setExecutingStepKey] = useState<string | null>(null);

  const [customTargetCol, setCustomTargetCol] = useState<string>('');
  const [runningCustomTarget, setRunningCustomTarget] = useState<boolean>(false);

  const [zenodoInput, setZenodoInput] = useState<string>('');
  const [attachingData, setAttachingData] = useState<boolean>(false);

  const [harvestQuery, setHarvestQuery] = useState<string>('');
  const [harvesting, setHarvesting] = useState<boolean>(false);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);
  const [probeData, setProbeData] = useState<any>(null);
  const [probing, setProbing] = useState<boolean>(false);

  // Gemini Discovery & Downstream Inference Co-Pilot state
  const [chatInput, setChatInput] = useState<string>('');
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<CopilotMessage[]>([
    {
      role: 'assistant',
      content:
        'Describe the phenomenon, biological mechanism, dataset anomaly, or engineering bottleneck you want to investigate. I will help formulate the hypothesis, identify what empirical signals to look for, and stream live dissertations, datasets, and 30-model audits based on our dialogue.',
      model_used: 'Gemini Discovery Co-Pilot',
    },
  ]);

  const [uploadTitle, setUploadTitle] = useState<string>('');
  const [uploadUni, setUploadUni] = useState<string>('');
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

  const handleRunCustomTargetColumn = async (colName: string) => {
    if (!activeDataset || !colName) return;
    setRunningCustomTarget(true);
    setErrorMsg(null);
    try {
      const res = await fetch('/api/lazypredict/run-custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: activeDataset.id,
          target_column: colName,
          is_author_primary_target: false,
          target_rationale: `Investigating column ${colName} via Gemini Co-Pilot`,
        }),
      });
      if (res.ok) {
        const newRun: LazyPredictRun = await res.json();
        await fetchAllData();
        setSelectedRunId(newRun.id);
        setShowMoreDataTools(false);
        setCurrentStep(2);
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

  const executeLiveStreamHarvest = (queryStr: string, explicitDomain?: string) => {
    const cleanQ = queryStr.trim();
    if (!cleanQ) return;
    const domain = explicitDomain || inferDomainFromText(cleanQ);
    setHarvestQuery(cleanQ);
    setHarvesting(true);
    setErrorMsg(null);
    setStreamEvents([`Connecting to live OpenAlex, ClinicalTrials.gov, and Zenodo streams for "${cleanQ}"...`]);

    const url = `/api/harvest/stream?query=${encodeURIComponent(cleanQ)}&domain=${encodeURIComponent(domain)}&min_year=2018&max_results=3`;
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

  const handleLiveHarvest = async (e: React.FormEvent) => {
    e.preventDefault();
    executeLiveStreamHarvest(harvestQuery);
  };

  const handleSendCopilotMessage = async (customPrompt?: string) => {
    const text = (customPrompt ?? chatInput).trim();
    if (!text || chatLoading) return;

    const updatedHistory: CopilotMessage[] = [...chatMessages, { role: 'user', content: text }];
    setChatMessages(updatedHistory);
    if (!customPrompt) setChatInput('');
    setChatLoading(true);
    setErrorMsg(null);

    try {
      const res = await fetch('/api/copilot/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updatedHistory.map((m) => ({ role: m.role, content: m.content })),
          dissertation_id: selectedDiss?.id ?? null,
          lazy_run_id: activeRun?.id ?? null,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setChatMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.reply || 'Analysis complete.',
            hypothesis_formulation: data.hypothesis_formulation,
            downstream_inference_plan: data.downstream_inference_plan,
            suggested_actions: data.suggested_actions,
            model_used: data.model_used || 'Gemini 2.5 (Live API)',
          },
        ]);
      } else {
        setErrorMsg('Gemini Co-Pilot request encountered an error.');
      }
    } catch (err) {
      setErrorMsg('Failed to reach Gemini Co-Pilot endpoint.');
    } finally {
      setChatLoading(false);
    }
  };

  const handleCopilotActionClick = (action: CopilotAction) => {
    if (action.action_type === 'harvest_stream') {
      executeLiveStreamHarvest(action.query_or_column, action.domain);
    } else if (action.action_type === 'probe_apis') {
      handleRunLiveProbe(action.query_or_column);
    } else if (action.action_type === 'run_lazypredict_column') {
      handleRunCustomTargetColumn(action.query_or_column);
    } else if (action.action_type === 'run_deep_verification' && activeRun) {
      setCurrentStep(2);
      handleEscalateDeepInference(activeRun.id);
    } else if (action.action_type === 'follow_up_prompt') {
      handleSendCopilotMessage(action.query_or_column);
    }
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
    const targetQ = q.trim() || harvestQuery.trim() || selectedDiss?.title?.split(' ')[0] || 'EGFR';
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

  const handleExecuteOpportunityStep = async (oppId: number, stepIdx: number) => {
    const key = `${oppId}_${stepIdx}`;
    setExecutingStepKey(key);
    try {
      const res = await fetch(`/api/opportunities/${oppId}/execute-step?step_index=${stepIdx}`, {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        setStepActionResults((prev) => ({ ...prev, [key]: data }));
      }
    } finally {
      setExecutingStepKey(null);
    }
  };

  const downloadTextFile = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
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
      formData.append('domain', inferDomainFromText(uploadTitle));
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
      setErrorMsg('Upload failed. Please verify the file format.');
    } finally {
      setUploading(false);
    }
  };

  const activeOpportunities = selectedDiss
    ? opportunities.filter((o) => o.dissertation_id === selectedDiss.id)
    : opportunities;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Top Investigative Rule Bar */}
      <div className="h-2 w-full investigative-header-rule" />

      {/* Editorial Magnifying Glass Header */}
      <header className="bg-white border-b-2 border-slate-900 sticky top-0 z-30">
        <div className="max-w-5xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div className="w-11 h-11 rounded-lg bg-slate-900 border-2 border-amber-500 flex items-center justify-center shadow-xs">
              <Search className="w-5 h-5 text-amber-400 stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="font-black text-xl tracking-tight text-slate-900 uppercase">
                  ThesisOut
                </span>
                <span className="text-[11px] font-bold uppercase tracking-wider px-2.5 py-0.5 bg-amber-400 text-slate-950 border border-slate-900">
                  Investigative Discovery Engine
                </span>
              </div>
              <p className="text-xs text-slate-600 font-medium">
                Conversational Gemini reasoning paired with 30-model empirical screening and 9 live scientific APIs
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5 text-xs">
            {dissertations.length > 0 && (
              <button
                onClick={handleResetWorkspace}
                className="px-3 py-1.5 bg-red-700 hover:bg-red-800 text-white border border-slate-900 font-bold uppercase tracking-wider transition cursor-pointer"
              >
                Clear Case Files
              </button>
            )}
            <span className="px-3 py-1.5 bg-slate-900 text-amber-400 border border-slate-900 font-bold uppercase tracking-wider flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5 text-amber-400" />
              Gemini Live API Active
            </span>
          </div>
        </div>

        {/* 3-Step Investigative Stepper */}
        <div className="max-w-5xl mx-auto px-6 pb-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <button
              onClick={() => setCurrentStep(1)}
              className={`text-left p-3.5 border-2 transition flex items-center gap-3 cursor-pointer ${
                currentStep === 1
                  ? 'bg-slate-900 text-white border-slate-900 shadow-[3px_3px_0px_#D97706]'
                  : 'bg-stone-50 text-slate-800 border-slate-300 hover:border-slate-900'
              }`}
            >
              <div
                className={`w-8 h-8 flex items-center justify-center font-black text-sm shrink-0 border ${
                  currentStep === 1
                    ? 'bg-amber-400 text-slate-950 border-slate-950'
                    : 'bg-white text-slate-900 border-slate-400'
                }`}
              >
                01
              </div>
              <div>
                <div className="text-xs font-extrabold uppercase tracking-wider">
                  Gemini Discovery Dialogue
                </div>
                <div
                  className={`text-[11px] ${
                    currentStep === 1 ? 'text-slate-300' : 'text-slate-500'
                  }`}
                >
                  Chat to formulate hypotheses & stream data
                </div>
              </div>
            </button>

            <button
              onClick={() => setCurrentStep(2)}
              className={`text-left p-3.5 border-2 transition flex items-center gap-3 cursor-pointer ${
                currentStep === 2
                  ? 'bg-slate-900 text-white border-slate-900 shadow-[3px_3px_0px_#D97706]'
                  : 'bg-stone-50 text-slate-800 border-slate-300 hover:border-slate-900'
              }`}
            >
              <div
                className={`w-8 h-8 flex items-center justify-center font-black text-sm shrink-0 border ${
                  currentStep === 2
                    ? 'bg-amber-400 text-slate-950 border-slate-950'
                    : 'bg-white text-slate-900 border-slate-400'
                }`}
              >
                02
              </div>
              <div>
                <div className="text-xs font-extrabold uppercase tracking-wider">
                  30-Model Signal Audit
                </div>
                <div
                  className={`text-[11px] ${
                    currentStep === 2 ? 'text-slate-300' : 'text-slate-500'
                  }`}
                >
                  Inspect empirical patterns & steer inference
                </div>
              </div>
            </button>

            <button
              onClick={() => setCurrentStep(3)}
              className={`text-left p-3.5 border-2 transition flex items-center gap-3 cursor-pointer ${
                currentStep === 3
                  ? 'bg-slate-900 text-white border-slate-900 shadow-[3px_3px_0px_#D97706]'
                  : 'bg-stone-50 text-slate-800 border-slate-300 hover:border-slate-900'
              }`}
            >
              <div
                className={`w-8 h-8 flex items-center justify-center font-black text-sm shrink-0 border ${
                  currentStep === 3
                    ? 'bg-amber-400 text-slate-950 border-slate-950'
                    : 'bg-white text-slate-900 border-slate-400'
                }`}
              >
                03
              </div>
              <div>
                <div className="text-xs font-extrabold uppercase tracking-wider">
                  Actionable Dossier
                </div>
                <div
                  className={`text-[11px] ${
                    currentStep === 3 ? 'text-slate-300' : 'text-slate-500'
                  }`}
                >
                  Execute verification steps & export code
                </div>
              </div>
            </button>
          </div>
        </div>
      </header>

      {/* Main Investigative Workspace */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8">
        {errorMsg && (
          <div className="mb-6 bg-red-50 border-2 border-red-800 text-red-900 px-5 py-3.5 text-xs flex items-center justify-between font-medium">
            <span>{errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="font-bold underline cursor-pointer">
              Dismiss
            </button>
          </div>
        )}

        {/* Active Case File Context Banner */}
        {selectedDiss && (
          <div className="mb-6 bg-white border-2 border-slate-900 px-5 py-4 flex flex-wrap items-center justify-between gap-3 shadow-[3px_3px_0px_#0F172A]">
            <div className="flex items-center gap-3">
              <span className="px-2.5 py-1 bg-slate-900 text-amber-400 font-bold uppercase tracking-wider text-[11px]">
                Active Case File
              </span>
              <div>
                <div className="text-xs font-bold text-slate-900 line-clamp-1">
                  {selectedDiss.title}
                </div>
                <div className="text-[11px] text-slate-600 flex flex-wrap items-center gap-2 mt-1">
                  <span>
                    {selectedDiss.authors} | {selectedDiss.university} ({selectedDiss.publication_year})
                  </span>
                  <span className="px-2 py-0.5 bg-stone-100 text-slate-900 border border-slate-400 font-semibold">
                    Source: {selectedDiss.source_repo}
                  </span>
                  <span className="px-2 py-0.5 bg-amber-100 text-slate-900 border border-amber-500 font-semibold">
                    Rigor Confidence: {(selectedDiss.scientific_rigor_score * 100).toFixed(0)}%
                  </span>
                  {selectedDiss.source_url && (
                    <a
                      href={selectedDiss.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-700 font-bold underline inline-flex items-center gap-1"
                    >
                      Verify Source Record <ExternalLink className="w-3 h-3" />
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
                className="bg-stone-50 border-2 border-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-900"
              >
                {dissertations.map((d) => (
                  <option key={d.id} value={d.id}>
                    Switch Case: {d.title.slice(0, 45)}...
                  </option>
                ))}
              </select>
            )}
          </div>
        )}

        {/* ===================================================================== */}
        {/* STEP 1: CONVERSATIONAL GEMINI DISCOVERY & EVIDENCE ACQUISITION */}
        {/* ===================================================================== */}
        {currentStep === 1 && (
          <div className="space-y-6">
            <div className="text-center max-w-2xl mx-auto">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-slate-900 text-amber-400 text-[11px] font-bold uppercase tracking-wider mb-2">
                <Search className="w-3.5 h-3.5" />
                Stage 01 | Conversational Hypothesis & Evidence Formulation
              </span>
              <h1 className="text-2xl font-black text-slate-900 tracking-tight">
                Collaborate with Gemini to Guide Discovery & Downstream Inference
              </h1>
              <p className="text-xs text-slate-600 mt-1">
                Discuss what you believe is happening in a biological or engineering system. Gemini will help structure the hypothesis, recommend what empirical evidence to stream, and configure downstream 30-model inference.
              </p>
            </div>

            {/* Mode Switcher */}
            <div className="flex justify-center">
              <div className="inline-flex bg-white p-1 border-2 border-slate-900 shadow-[3px_3px_0px_#0F172A] gap-1">
                <button
                  onClick={() => setStep1Mode('copilot')}
                  className={`px-4 py-2 text-xs font-bold uppercase tracking-wider transition cursor-pointer flex items-center gap-1.5 ${
                    step1Mode === 'copilot'
                      ? 'bg-slate-900 text-amber-400'
                      : 'text-slate-700 hover:bg-stone-100'
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Gemini Discovery Co-Pilot & Live Stream
                </button>
                <button
                  onClick={() => setStep1Mode('upload')}
                  className={`px-4 py-2 text-xs font-bold uppercase tracking-wider transition cursor-pointer flex items-center gap-1.5 ${
                    step1Mode === 'upload'
                      ? 'bg-slate-900 text-amber-400'
                      : 'text-slate-700 hover:bg-stone-100'
                  }`}
                >
                  <Upload className="w-3.5 h-3.5" />
                  Upload Own PDF / CSV
                </button>
                {dissertations.length > 0 && (
                  <button
                    onClick={() => setStep1Mode('saved')}
                    className={`px-4 py-2 text-xs font-bold uppercase tracking-wider transition cursor-pointer flex items-center gap-1.5 ${
                      step1Mode === 'saved'
                        ? 'bg-slate-900 text-amber-400'
                        : 'text-slate-700 hover:bg-stone-100'
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    Active Case Files ({dissertations.length})
                  </button>
                )}
              </div>
            </div>

            {/* MODE A: Conversational Gemini Discovery Co-Pilot + Open-Ended Live Stream */}
            {step1Mode === 'copilot' && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Left 7 Columns: Conversational Gemini Co-Pilot */}
                <div className="lg:col-span-7 dossier-card p-6 flex flex-col justify-between space-y-4">
                  <div>
                    <div className="flex items-center justify-between border-b-2 border-slate-900 pb-3 mb-4">
                      <div className="flex items-center gap-2 text-slate-900 font-black text-sm uppercase tracking-wider">
                        <MessageSquare className="w-4 h-4 text-red-700" />
                        Gemini Scientific Discovery & Inference Dialogue
                      </div>
                      <span className="px-2 py-0.5 bg-amber-400 text-slate-950 border border-slate-900 text-[10px] font-bold uppercase">
                        Context-Aware
                      </span>
                    </div>

                    {/* Chat Transcript */}
                    <div className="space-y-4 max-h-[420px] overflow-y-auto pr-1">
                      {chatMessages.map((msg, idx) => (
                        <div
                          key={idx}
                          className={`p-4 border-2 text-xs space-y-3 ${
                            msg.role === 'user'
                              ? 'bg-slate-900 text-white border-slate-900 ml-8'
                              : 'bg-stone-50 text-slate-900 border-slate-900 mr-4'
                          }`}
                        >
                          <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider opacity-75">
                            <span>{msg.role === 'user' ? 'Investigator' : msg.model_used || 'Gemini Co-Pilot'}</span>
                          </div>

                          <div className="leading-relaxed whitespace-pre-line">{msg.content}</div>

                          {msg.hypothesis_formulation && (
                            <div className="bg-amber-50 border border-slate-900 p-2.5 text-slate-900">
                              <div className="font-black uppercase tracking-wider text-[10px] text-amber-900">
                                Formulated Hypothesis
                              </div>
                              <div className="mt-0.5 font-medium">{msg.hypothesis_formulation}</div>
                            </div>
                          )}

                          {msg.downstream_inference_plan && msg.downstream_inference_plan.length > 0 && (
                            <div className="bg-white border border-slate-900 p-2.5 text-slate-900 space-y-1">
                              <div className="font-black uppercase tracking-wider text-[10px] text-slate-700">
                                Downstream Inference Strategy
                              </div>
                              <ul className="list-disc list-inside space-y-0.5 text-[11px] text-slate-700">
                                {msg.downstream_inference_plan.map((step, sIdx) => (
                                  <li key={sIdx}>{step}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {msg.suggested_actions && msg.suggested_actions.length > 0 && (
                            <div className="pt-1 space-y-1.5">
                              <div className="text-[10px] font-black uppercase tracking-wider text-red-700">
                                Recommended Actions (Click to Execute):
                              </div>
                              <div className="flex flex-wrap gap-2">
                                {msg.suggested_actions.map((act, aIdx) => (
                                  <button
                                    key={aIdx}
                                    type="button"
                                    onClick={() => handleCopilotActionClick(act)}
                                    disabled={harvesting || probing || runningCustomTarget}
                                    className="px-3 py-1.5 bg-slate-900 hover:bg-red-700 text-amber-400 hover:text-white border border-slate-900 font-bold text-[11px] flex items-center gap-1.5 transition cursor-pointer"
                                  >
                                    <Play className="w-3 h-3" />
                                    {act.label}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}

                      {chatLoading && (
                        <div className="p-4 bg-stone-50 border-2 border-slate-900 text-xs flex items-center gap-2 text-slate-700">
                          <RefreshCw className="w-4 h-4 animate-spin text-red-700" />
                          Gemini is analyzing your hypothesis and workspace state...
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Chat Input Box */}
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleSendCopilotMessage();
                    }}
                    className="pt-3 border-t-2 border-slate-900 flex gap-2"
                  >
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder="Ask Gemini about a hypothesis, mechanism, or which variables to test next..."
                      className="flex-1 bg-white border-2 border-slate-900 px-3.5 py-2.5 text-xs text-slate-900 focus:outline-none"
                    />
                    <button
                      type="submit"
                      disabled={chatLoading || !chatInput.trim()}
                      className="px-4 py-2.5 bg-red-700 hover:bg-red-800 disabled:opacity-50 text-white border-2 border-slate-900 font-black uppercase tracking-wider text-xs flex items-center gap-1.5 cursor-pointer"
                    >
                      <Send className="w-3.5 h-3.5" />
                      Send
                    </button>
                  </form>
                </div>

                {/* Right 5 Columns: Direct Live Repository Stream & 9-API Telemetry */}
                <div className="lg:col-span-5 dossier-card p-6 flex flex-col justify-between space-y-5">
                  <div className="space-y-4">
                    <div className="border-b-2 border-slate-900 pb-3">
                      <div className="flex items-center gap-2 text-slate-900 font-black text-sm uppercase tracking-wider">
                        <Search className="w-4 h-4 text-red-700" />
                        Direct Multi-Repository Stream
                      </div>
                      <p className="text-[11px] text-slate-600 mt-1 leading-relaxed">
                        Or enter any free-form scientific query directly. Domain classification, dataset matching (OpenAlex + Zenodo), and 30-model LazyPredict screening run automatically without rigid presets.
                      </p>
                    </div>

                    <form onSubmit={handleLiveHarvest} className="space-y-3 text-xs">
                      <div>
                        <label className="block font-bold uppercase tracking-wider text-slate-800 mb-1.5">
                          Free-Form Scientific Query
                        </label>
                        <input
                          type="text"
                          value={harvestQuery}
                          onChange={(e) => setHarvestQuery(e.target.value)}
                          placeholder="Enter any mechanism, target, disease, or system..."
                          className="w-full bg-white border-2 border-slate-900 px-3.5 py-2.5 text-xs text-slate-900 focus:outline-none"
                        />
                      </div>

                      <div className="flex flex-col gap-2.5 pt-1">
                        <button
                          type="submit"
                          disabled={harvesting || !harvestQuery.trim()}
                          className="w-full py-3 px-4 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-amber-400 border-2 border-slate-900 font-black uppercase tracking-wider text-xs flex items-center justify-center gap-2 shadow-[3px_3px_0px_#D97706] transition cursor-pointer"
                        >
                          {harvesting ? (
                            <>
                              <RefreshCw className="w-4 h-4 animate-spin" />
                              Streaming Live Repositories...
                            </>
                          ) : (
                            <>
                              <Search className="w-4 h-4" />
                              Stream Theses & Auto-Audit Datasets
                            </>
                          )}
                        </button>

                        <button
                          type="button"
                          onClick={() => handleRunLiveProbe(harvestQuery)}
                          disabled={probing || !harvestQuery.trim()}
                          className="w-full py-2.5 px-4 bg-amber-400 hover:bg-amber-500 disabled:opacity-50 text-slate-950 border-2 border-slate-900 font-black uppercase tracking-wider text-xs flex items-center justify-center gap-1.5 transition cursor-pointer"
                        >
                          {probing ? (
                            <RefreshCw className="w-4 h-4 animate-spin" />
                          ) : (
                            <Search className="w-4 h-4" />
                          )}
                          Probe 9 Live Science APIs for Query
                        </button>
                      </div>
                    </form>

                    {/* Live SSE Stream Progress Log */}
                    {streamEvents.length > 0 && (
                      <div className="bg-stone-100 border-2 border-slate-900 p-3.5 space-y-1.5 text-[11px] font-mono">
                        <div className="font-bold text-slate-900 uppercase tracking-wider text-[10px]">
                          Live Multi-Repository Stream Telemetry
                        </div>
                        {streamEvents.map((msg, i) => (
                          <div key={i} className="flex items-start gap-2 text-slate-800">
                            <span className="w-2 h-2 bg-red-700 shrink-0 mt-1" />
                            <span>{msg}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Live Multi-Database Integration Telemetry Drawer */}
                    {probeData && (
                      <div className="bg-stone-100 border-2 border-slate-900 p-3.5 space-y-2.5 text-[11px]">
                        <div className="flex items-center justify-between">
                          <span className="font-black uppercase tracking-wider text-slate-900">
                            9-API Telemetry: "{probeData.query}"
                          </span>
                          <button
                            onClick={() => setProbeData(null)}
                            className="text-red-700 font-bold underline cursor-pointer"
                          >
                            Close
                          </button>
                        </div>
                        <div className="space-y-2">
                          <div className="bg-white border border-slate-900 p-2.5">
                            <div className="font-bold text-red-700">OpenTargets GraphQL v4</div>
                            <div className="text-slate-800">
                              Target: {probeData.live_integrations?.opentargets_graphql_v4?.target_profile?.symbol || 'Queried'} | Known Drugs:{' '}
                              {probeData.live_integrations?.opentargets_graphql_v4?.target_profile?.unique_known_drugs ?? 0}
                            </div>
                          </div>
                          <div className="bg-white border border-slate-900 p-2.5">
                            <div className="font-bold text-emerald-800">ChEMBL EBI REST API</div>
                            <div className="text-slate-800">
                              Targets: {probeData.live_integrations?.chembl_ebi_rest?.targets_found?.length || 0} | Max Phase:{' '}
                              {probeData.live_integrations?.chembl_ebi_rest?.max_clinical_phase_observed ?? 0}
                            </div>
                          </div>
                          <div className="bg-white border border-slate-900 p-2.5">
                            <div className="font-bold text-blue-800">ClinicalTrials.gov v2 & openFDA</div>
                            <div className="text-slate-800">
                              Trials: {probeData.live_integrations?.clinicaltrials_gov_v2?.total_trials ?? 0} | FAERS Reports:{' '}
                              {probeData.live_integrations?.openfda_faers?.total_adverse_event_reports ?? 0}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* MODE B: Upload PDF / CSV */}
            {step1Mode === 'upload' && (
              <div className="dossier-card p-8 max-w-2xl mx-auto">
                <div className="flex items-center gap-2.5 text-slate-900 font-black text-base uppercase tracking-tight mb-1">
                  <Upload className="w-5 h-5 text-red-700" />
                  Submit Dissertation PDF or Tabular Dataset
                </div>
                <p className="text-xs text-slate-600 mb-5 leading-relaxed">
                  Upload a local CSV, TSV, Excel, or Parquet dataset along with an optional dissertation PDF. The engine will execute 30+ LazyPredict models and advance to Stage 02.
                </p>

                <form onSubmit={handleUploadSubmit} className="space-y-4 text-xs">
                  <div>
                    <label className="block font-bold uppercase tracking-wider text-slate-800 mb-1">
                      Study or Dissertation Title *
                    </label>
                    <input
                      type="text"
                      required
                      value={uploadTitle}
                      onChange={(e) => setUploadTitle(e.target.value)}
                      placeholder="Enter study or dissertation title"
                      className="w-full bg-white border-2 border-slate-900 px-4 py-2.5 text-slate-900 focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block font-bold uppercase tracking-wider text-slate-800 mb-1">
                      Institution (Optional)
                    </label>
                    <input
                      type="text"
                      value={uploadUni}
                      onChange={(e) => setUploadUni(e.target.value)}
                      placeholder="e.g., MIT, Stanford"
                      className="w-full bg-white border-2 border-slate-900 px-4 py-2.5 text-slate-900 focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                    <div className="border-2 border-dashed border-slate-900 p-4 bg-stone-50">
                      <label className="block font-bold uppercase tracking-wider text-slate-900 mb-1">
                        Dataset File (.csv, .xlsx, .tsv)
                      </label>
                      <input
                        type="file"
                        accept=".csv,.tsv,.xlsx,.parquet"
                        onChange={(e) => setUploadCsv(e.target.files?.[0] || null)}
                        className="w-full text-slate-700 text-[11px]"
                      />
                    </div>
                    <div className="border-2 border-dashed border-slate-900 p-4 bg-stone-50">
                      <label className="block font-bold uppercase tracking-wider text-slate-900 mb-1">
                        Dissertation PDF (Optional)
                      </label>
                      <input
                        type="file"
                        accept=".pdf"
                        onChange={(e) => setUploadPdf(e.target.files?.[0] || null)}
                        className="w-full text-slate-700 text-[11px]"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block font-bold uppercase tracking-wider text-slate-800 mb-1">
                      Target Column to Predict (Leave blank for automatic selection)
                    </label>
                    <input
                      type="text"
                      value={uploadTarget}
                      onChange={(e) => setUploadTarget(e.target.value)}
                      placeholder="Optional target column name"
                      className="w-full bg-white border-2 border-slate-900 px-4 py-2.5 text-slate-900 focus:outline-none"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={uploading}
                    className="w-full py-3.5 px-6 bg-slate-900 hover:bg-slate-800 text-amber-400 border-2 border-slate-900 font-black uppercase tracking-wider text-xs flex items-center justify-center gap-2 shadow-[3px_3px_0px_#D97706] transition cursor-pointer"
                  >
                    {uploading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Auditing Dataset Across 30+ Models...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4" />
                        Ingest Evidence & Proceed to Stage 02
                      </>
                    )}
                  </button>
                </form>
              </div>
            )}

            {/* MODE C: Active Case Files */}
            {step1Mode === 'saved' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {dissertations.map((diss) => {
                  const run = diss.datasets?.[0]?.lazy_runs?.[0];
                  return (
                    <div
                      key={diss.id}
                      onClick={() => handleSelectDissertation(diss)}
                      className="dossier-card p-6 transition cursor-pointer flex flex-col justify-between hover:bg-stone-50"
                    >
                      <div>
                        <div className="flex items-center justify-between text-xs mb-2">
                          <span className="px-2.5 py-0.5 bg-slate-900 text-amber-400 font-bold uppercase tracking-wider">
                            {diss.university}
                          </span>
                          <span className="text-slate-600 font-bold">{diss.publication_year}</span>
                        </div>
                        <h3 className="text-base font-bold text-slate-900 leading-snug">{diss.title}</h3>
                        <p className="text-xs text-slate-600 mt-2 line-clamp-2">{diss.abstract}</p>
                      </div>

                      <div className="mt-4 pt-3 border-t border-slate-300 flex items-center justify-between text-xs">
                        {run ? (
                          <span className="px-2.5 py-1 bg-emerald-50 text-emerald-900 font-bold border border-emerald-700">
                            Best Model: {run.best_model_name} ({run.best_score})
                          </span>
                        ) : (
                          <span className="text-slate-500">Click to attach dataset</span>
                        )}
                        <span className="font-bold text-red-700 uppercase tracking-wider flex items-center gap-1">
                          Open Audit <ArrowRight className="w-3.5 h-3.5" />
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
        {/* STEP 2: 30-MODEL SIGNAL AUDIT */}
        {/* ===================================================================== */}
        {currentStep === 2 && (
          <div className="space-y-6">
            {!selectedDiss ? (
              <div className="dossier-card p-10 text-center max-w-lg mx-auto">
                <BookOpen className="w-10 h-10 text-slate-900 mx-auto mb-3" />
                <h3 className="text-lg font-black uppercase text-slate-900">
                  No Active Case Selected
                </h3>
                <p className="text-xs text-slate-600 mt-1 mb-5">
                  Return to Stage 01 to converse with Gemini or stream a live dissertation dataset.
                </p>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="px-5 py-2.5 bg-slate-900 text-amber-400 text-xs font-bold uppercase tracking-wider cursor-pointer"
                >
                  Return to Stage 01
                </button>
              </div>
            ) : !activeRun ? (
              <div className="dossier-card p-8 max-w-xl mx-auto space-y-5">
                <div className="text-center">
                  <span className="px-3 py-1 bg-amber-400 text-slate-950 text-xs font-bold uppercase tracking-wider border border-slate-900">
                    Dataset Required
                  </span>
                  <h3 className="text-lg font-black text-slate-900 mt-2">
                    Attach a Dataset to Execute the 30-Model Audit
                  </h3>
                  <p className="text-xs text-slate-600 mt-1">
                    Upload a CSV/Excel table or enter a Zenodo Record ID below to benchmark 30+ machine learning models automatically.
                  </p>
                </div>

                <form onSubmit={handleAttachZenodo} className="flex gap-2">
                  <input
                    type="text"
                    value={zenodoInput}
                    onChange={(e) => setZenodoInput(e.target.value)}
                    placeholder="Enter Zenodo Record ID or URL..."
                    className="flex-1 bg-white border-2 border-slate-900 px-4 py-2.5 text-xs"
                  />
                  <button
                    type="submit"
                    disabled={attachingData}
                    className="px-4 py-2.5 bg-slate-900 text-amber-400 text-xs font-bold uppercase tracking-wider cursor-pointer"
                  >
                    {attachingData ? 'Fetching...' : 'Fetch Zenodo CSV'}
                  </button>
                </form>

                <div className="border-2 border-dashed border-slate-900 p-4 bg-stone-50 text-center">
                  <label className="block text-xs font-bold uppercase tracking-wider text-slate-900 mb-2">
                    Or Upload a Local Dataset File (.csv, .tsv, .xlsx, .parquet)
                  </label>
                  <input
                    type="file"
                    accept=".csv,.tsv,.xlsx,.parquet"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleAttachDatasetFile(f);
                    }}
                    className="text-xs text-slate-700"
                  />
                </div>
              </div>
            ) : (
              <>
                {/* Investigative Signal Takeaway Banner */}
                <div className="dossier-card p-6">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-amber-400 text-slate-950 border border-slate-900 text-xs font-bold uppercase tracking-wider">
                          <Search className="w-3.5 h-3.5" />
                          Empirical Signal & Confidence Report
                        </div>
                        {activeDataset && (
                          <span className="px-3 py-1 bg-stone-100 border border-slate-900 text-slate-900 text-xs font-semibold">
                            Dataset Provenance: {activeDataset.source_type.toUpperCase()} ({activeDataset.row_count} rows x {activeDataset.col_count} cols)
                          </span>
                        )}
                        {activeDataset?.source_url && (
                          <a
                            href={activeDataset.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="px-3 py-1 bg-blue-50 border border-blue-800 text-blue-900 text-xs font-bold underline inline-flex items-center gap-1"
                          >
                            Dataset Origin Link <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                      <h2 className="text-lg font-black text-slate-900">
                        Audited {activeRun.models_tested_count} models on target column{' '}
                        <code className="px-2 py-0.5 bg-stone-100 border border-slate-900 text-red-700 font-mono">
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
                      className="px-5 py-3.5 bg-red-700 hover:bg-red-800 text-white border-2 border-slate-900 font-black uppercase tracking-wider text-xs flex items-center gap-2 shrink-0 shadow-[3px_3px_0px_#0F172A] transition cursor-pointer"
                    >
                      Proceed to Actionable Dossier <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Target Switcher Pills */}
                  {activeDataset && activeDataset.lazy_runs.length > 1 && (
                    <div className="mt-4 pt-3 border-t border-slate-300 flex flex-wrap items-center gap-2 text-xs">
                      <span className="font-bold uppercase tracking-wider text-slate-700">
                        Audited Columns:
                      </span>
                      {activeDataset.lazy_runs.map((run) => (
                        <button
                          key={run.id}
                          onClick={() => setSelectedRunId(run.id)}
                          className={`px-3 py-1 font-bold border transition cursor-pointer ${
                            activeRun.id === run.id
                              ? 'bg-slate-900 text-amber-400 border-slate-900'
                              : 'bg-stone-50 text-slate-800 border-slate-400 hover:border-slate-900'
                          }`}
                        >
                          {run.target_column} ({run.best_score})
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* Inline Gemini Downstream Inference Bar inside Stage 02 */}
                <div className="dossier-card p-5 bg-stone-50 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-xs font-black uppercase tracking-wider text-slate-900">
                      <MessageSquare className="w-4 h-4 text-red-700" />
                      Ask Gemini Co-Pilot About This 30-Model Audit & Downstream Inference
                    </div>
                    <span className="text-[10px] font-bold uppercase text-slate-500">
                      Has full visibility into {activeDataset?.name} & leaderboard
                    </span>
                  </div>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleSendCopilotMessage();
                    }}
                    className="flex gap-2"
                  >
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder={`Ask why ${activeRun.best_model_name} won, which confounder to ablate, or which column in ${activeDataset?.name} to test next...`}
                      className="flex-1 bg-white border-2 border-slate-900 px-3.5 py-2 text-xs text-slate-900 focus:outline-none"
                    />
                    <button
                      type="submit"
                      disabled={chatLoading || !chatInput.trim()}
                      className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-amber-400 border-2 border-slate-900 font-bold uppercase tracking-wider text-xs flex items-center gap-1.5 cursor-pointer"
                    >
                      {chatLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                      Ask Gemini
                    </button>
                  </form>
                  {chatMessages.length > 1 && (
                    <div className="bg-white border-2 border-slate-900 p-3.5 text-xs space-y-2">
                      <div className="font-bold text-slate-900">
                        {chatMessages[chatMessages.length - 1].model_used || 'Gemini Co-Pilot'}:
                      </div>
                      <p className="text-slate-800 whitespace-pre-line leading-relaxed">
                        {chatMessages[chatMessages.length - 1].content}
                      </p>
                      {chatMessages[chatMessages.length - 1].suggested_actions && (
                        <div className="flex flex-wrap gap-2 pt-1">
                          {chatMessages[chatMessages.length - 1].suggested_actions?.map((act, i) => (
                            <button
                              key={i}
                              onClick={() => handleCopilotActionClick(act)}
                              className="px-3 py-1 bg-slate-900 text-amber-400 text-[11px] font-bold flex items-center gap-1 cursor-pointer"
                            >
                              <Play className="w-3 h-3" /> {act.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* 3 High-Contrast Investigative Evidence Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  <div className="dossier-card p-5 flex flex-col justify-between">
                    <div>
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                        01. Classical Linear Baseline
                      </span>
                      <div className="text-3xl font-black text-slate-900 mt-2">
                        {activeRun.author_baseline_score}
                      </div>
                      <div className="text-xs font-bold text-slate-700 mt-0.5">
                        Metric: {activeRun.primary_metric_name}
                      </div>
                    </div>
                    <p className="text-xs text-slate-600 mt-4 leading-relaxed">
                      Baseline performance of conventional linear statistical methods prior to nonlinear ensemble screening.
                    </p>
                  </div>

                  <div className="dossier-card p-5 flex flex-col justify-between bg-amber-50/40">
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-800">
                          02. Top Model Identified
                        </span>
                        <span className="px-2.5 py-0.5 bg-emerald-700 text-white text-[11px] font-bold uppercase">
                          +{(activeRun.signal_gain_delta * 100).toFixed(1)}% Gain
                        </span>
                      </div>
                      <div className="text-3xl font-black text-emerald-800 mt-2">
                        {activeRun.best_score}
                      </div>
                      <div className="text-xs font-bold text-slate-900 mt-0.5">
                        Model: {activeRun.best_model_name}
                      </div>
                    </div>
                    <p className="text-xs text-slate-600 mt-4 leading-relaxed">
                      Selected after benchmarking {activeRun.models_tested_count} distinct machine learning architectures on identical folds.
                    </p>
                  </div>

                  <div className="dossier-card p-5">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-slate-800">
                      03. Primary Predictive Variables
                    </span>
                    <div className="space-y-3 mt-3">
                      {activeRun.top_features.slice(0, 3).map((tf, i) => (
                        <div key={tf.feature} className="bg-stone-50 border border-slate-400 p-2.5">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-bold text-slate-900 truncate pr-2">
                              #{i + 1} {tf.feature}
                            </span>
                            <span className="font-bold text-red-700 shrink-0">
                              {(tf.importance * 100).toFixed(0)}% weight
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-slate-200 overflow-hidden mt-1.5">
                            <div
                              className="h-full bg-slate-900"
                              style={{ width: `${Math.min(100, tf.importance * 180)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* PROGRESSIVE DISCLOSURE DRAWERS */}
                <div className="space-y-3 pt-2">
                  {/* Drawer 1: 5-Stage Deep Verification */}
                  <div className="dossier-card overflow-hidden">
                    <button
                      onClick={() => handleEscalateDeepInference(activeRun.id)}
                      className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-stone-50 transition cursor-pointer"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-slate-900 text-amber-400 flex items-center justify-center">
                          <ShieldCheck className="w-5 h-5" />
                        </div>
                        <div>
                          <div className="text-sm font-black uppercase tracking-tight text-slate-900">
                            Verify Discovery (5-Stage Deep Verification Audit)
                          </div>
                          <div className="text-xs text-slate-600">
                            Executes 5-fold cross-validation, target-permutation null testing, Super-Learner stacking, and regime threshold extraction
                          </div>
                        </div>
                      </div>
                      <span className="px-3 py-1.5 bg-amber-400 text-slate-950 border border-slate-900 text-xs font-bold uppercase tracking-wider shrink-0">
                        {escalatingRunId === activeRun.id
                          ? 'Auditing...'
                          : escalationCache[activeRun.id]
                          ? 'Verified'
                          : 'Run Audit'}
                      </span>
                    </button>

                    {showDeepCheck && escalationCache[activeRun.id] && (
                      <div className="px-6 pb-6 pt-3 border-t-2 border-slate-900 bg-stone-50 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        {(() => {
                          const esc = escalationCache[activeRun.id];
                          return (
                            <>
                              <div className="bg-white border border-slate-900 p-4">
                                <div className="font-bold uppercase tracking-wider text-slate-900 mb-1">
                                  01. Permutation & Leakage Audit
                                </div>
                                <p className="text-slate-700 leading-relaxed">
                                  {esc.stage1_leakage_and_permutation_audit.audit_interpretation}
                                </p>
                              </div>

                              <div className="bg-white border border-slate-900 p-4">
                                <div className="font-bold uppercase tracking-wider text-emerald-800 mb-1">
                                  02. Super-Learner Stacking Score: {esc.stage2_super_learner_ensemble.super_learner_score}
                                </div>
                                <p className="text-slate-700 leading-relaxed">
                                  Stacking LightGBM, XGBoost, ExtraTrees, and Random Forest achieved{' '}
                                  <strong>{esc.stage2_super_learner_ensemble.super_learner_score}</strong>.{' '}
                                  {esc.stage4_conformal_uncertainty.ood_note}
                                </p>
                              </div>

                              <div className="bg-white border border-slate-900 p-4">
                                <div className="font-bold uppercase tracking-wider text-amber-800 mb-1">
                                  03. Mechanistic Regime Thresholds
                                </div>
                                <p className="text-slate-700 leading-relaxed">
                                  {esc.stage3_mechanistic_regime_discovery.synergy_conclusion}
                                </p>
                              </div>

                              <div className="bg-white border border-slate-900 p-4">
                                <div className="font-bold uppercase tracking-wider text-red-700 mb-1">
                                  04. Counterfactual Parameter Shifts
                                </div>
                                <ul className="space-y-1 text-slate-700 mt-1">
                                  {esc.stage5_counterfactual_intervention.prescribed_parameter_shifts.map((s, idx) => (
                                    <li key={idx}>- {s.intervention_action}</li>
                                  ))}
                                </ul>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                    )}
                  </div>

                  {/* Drawer 2: Full 30-Model Leaderboard */}
                  <div className="dossier-card overflow-hidden">
                    <button
                      onClick={() => setShowAllModels(!showAllModels)}
                      className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-stone-50 transition cursor-pointer"
                    >
                      <div className="text-sm font-black uppercase tracking-tight text-slate-900">
                        Inspect Full {activeRun.leaderboard.length}-Model Leaderboard & Confidence Intervals
                      </div>
                      {showAllModels ? (
                        <ChevronUp className="w-5 h-5 text-slate-700" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-slate-700" />
                      )}
                    </button>

                    {showAllModels && (
                      <div className="px-6 pb-6 pt-2 border-t-2 border-slate-900 overflow-x-auto max-h-80">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="border-b-2 border-slate-900 text-slate-900 font-bold">
                              <th className="py-2 px-3">Rank</th>
                              {Object.keys(activeRun.leaderboard[0] || {}).map((col) => (
                                <th key={col} className="py-2 px-3">
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-200 font-mono">
                            {activeRun.leaderboard.map((row, idx) => (
                              <tr
                                key={idx}
                                className={idx === 0 ? 'bg-amber-100 font-bold text-slate-950' : 'text-slate-800'}
                              >
                                <td className="py-2 px-3">#{idx + 1}</td>
                                {Object.entries(row).map(([k, val]) => (
                                  <td key={k} className="py-2 px-3">
                                    {val === null ? '-' : String(val)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>

                  {/* Drawer 3: Audit Another Column */}
                  {activeDataset && (
                    <div className="dossier-card overflow-hidden">
                      <button
                        onClick={() => setShowMoreDataTools(!showMoreDataTools)}
                        className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-stone-50 transition cursor-pointer"
                      >
                        <div className="text-sm font-black uppercase tracking-tight text-slate-900">
                          Audit Another Target Variable in This Dataset
                        </div>
                        {showMoreDataTools ? (
                          <ChevronUp className="w-5 h-5 text-slate-700" />
                        ) : (
                          <ChevronDown className="w-5 h-5 text-slate-700" />
                        )}
                      </button>

                      {showMoreDataTools && (
                        <div className="px-6 pb-6 pt-3 border-t-2 border-slate-900 flex flex-wrap items-center gap-3 text-xs">
                          <select
                            value={customTargetCol}
                            onChange={(e) => setCustomTargetCol(e.target.value)}
                            className="flex-1 bg-stone-50 border-2 border-slate-900 px-4 py-2.5 text-slate-900"
                          >
                            <option value="">Select column from {activeDataset.name}...</option>
                            {activeDataset.column_metadata
                              .filter((c) => c.role !== 'identifier')
                              .map((c) => (
                                <option key={c.name} value={c.name}>
                                  {c.name} ({c.dtype})
                                </option>
                              ))}
                          </select>
                          <button
                            onClick={() => handleRunCustomTargetColumn(customTargetCol)}
                            disabled={!customTargetCol || runningCustomTarget}
                            className="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 disabled:opacity-40 text-amber-400 font-bold uppercase tracking-wider flex items-center gap-1.5 cursor-pointer"
                          >
                            {runningCustomTarget ? (
                              <RefreshCw className="w-4 h-4 animate-spin" />
                            ) : (
                              <Play className="w-4 h-4" />
                            )}
                            Run 30-Model Audit
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Bottom Step Navigation */}
                <div className="flex items-center justify-between pt-4">
                  <button
                    onClick={() => setCurrentStep(1)}
                    className="px-4 py-2.5 bg-white border-2 border-slate-900 text-slate-900 text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 cursor-pointer"
                  >
                    <ArrowLeft className="w-4 h-4" /> Stage 01
                  </button>
                  <button
                    onClick={() => {
                      setCurrentStep(3);
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }}
                    className="px-6 py-3 bg-red-700 hover:bg-red-800 text-white border-2 border-slate-900 text-xs font-black uppercase tracking-wider flex items-center gap-2 shadow-[3px_3px_0px_#0F172A] cursor-pointer"
                  >
                    Continue to Stage 03: Actionable Dossier <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* ===================================================================== */}
        {/* STEP 3: ACTIONABLE DOSSIER & MARKET TRANSLATION */}
        {/* ===================================================================== */}
        {currentStep === 3 && (
          <div className="space-y-6">
            <div className="text-center max-w-xl mx-auto">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-slate-900 text-amber-400 text-[11px] font-bold uppercase tracking-wider mb-2">
                <Compass className="w-3.5 h-3.5" />
                Stage 03 | Commercial & Clinical Translation
              </span>
              <h2 className="text-2xl font-black text-slate-900 tracking-tight">
                Investigative Translation Dossiers & Executable Roadmap
              </h2>
              <p className="text-xs text-slate-600 mt-1">
                Every discovered empirical signal is cross-examined against live clinical trials, bioactivity registries, and high-urgency industry bottlenecks.
              </p>
            </div>

            {activeOpportunities.length === 0 ? (
              <div className="dossier-card p-10 text-center max-w-lg mx-auto">
                <Search className="w-10 h-10 text-slate-900 mx-auto mb-3" />
                <h3 className="text-base font-black uppercase text-slate-900">
                  No Translation Dossiers Generated Yet
                </h3>
                <p className="text-xs text-slate-600 mt-1 mb-5">
                  Audit a dataset in Stage 02 first to synthesize market translation dossiers.
                </p>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="px-5 py-2.5 bg-slate-900 text-amber-400 text-xs font-bold uppercase tracking-wider cursor-pointer"
                >
                  Return to Stage 01
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                {activeOpportunities.map((opp) => (
                  <div key={opp.id} className="dossier-card p-6 space-y-5">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <span className="px-3 py-1 bg-amber-400 text-slate-950 text-xs font-black uppercase tracking-wider border border-slate-900">
                        Translation Opportunity Score: {opp.overall_tos_score} / 100
                      </span>
                      <span className="px-3 py-1 bg-slate-900 text-white text-xs font-bold uppercase tracking-wider">
                        {opp.opportunity_type}
                      </span>
                    </div>

                    <h3 className="text-lg font-black text-slate-900 leading-snug">{opp.title}</h3>

                    {/* 3 Investigative Case Notes */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                      <div className="dossier-subcard p-4">
                        <div className="font-black uppercase tracking-wider text-red-700 mb-1.5">
                          01. Empirical Evidence Uncovered
                        </div>
                        <p className="text-slate-800 leading-relaxed">{opp.untranslated_finding_summary}</p>
                      </div>

                      <div className="dossier-subcard p-4">
                        <div className="font-black uppercase tracking-wider text-amber-800 mb-1.5">
                          02. Academic Bottleneck Diagnosis
                        </div>
                        <p className="text-slate-800 leading-relaxed">{opp.valley_of_death_diagnosis}</p>
                      </div>

                      <div className="dossier-subcard p-4">
                        <div className="font-black uppercase tracking-wider text-blue-800 mb-1.5">
                          03. Live Registry Whitespace
                        </div>
                        <p className="text-slate-800 leading-relaxed">
                          <strong>ClinicalTrials.gov:</strong> {opp.clinical_trials_check?.whitespace_status}
                        </p>
                        <p className="text-slate-800 mt-1.5 leading-relaxed">
                          <strong>Target / Modality:</strong> {opp.opentargets_chembl_check?.tractability_modality}
                        </p>
                      </div>
                    </div>

                    {/* Proposed Solution & Interactive 4-Step Action Plan */}
                    <div className="bg-stone-100 border-2 border-slate-900 p-5 text-xs space-y-4">
                      <div>
                        <div className="font-black text-slate-900 uppercase tracking-wider text-[11px] mb-1">
                          Recommended Translation Strategy & Executable Verification Steps
                        </div>
                        <p className="text-slate-900 font-semibold text-sm">
                          {opp.proposed_product_or_therapeutic}
                        </p>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {opp.validation_roadmap.map((step, i) => {
                          const key = `${opp.id}_${i}`;
                          const isRunning = executingStepKey === key;
                          const stepRes = stepActionResults[key];
                          const buttonLabels = [
                            'Run 5-Fold Verification Now',
                            'Generate Thresholds & Patent Claim',
                            'Run Live Clinical & ChEMBL Audit',
                            'Generate Python Script & Dossier',
                          ];

                          return (
                            <div
                              key={i}
                              className="bg-white border-2 border-slate-900 p-4 text-slate-800 flex flex-col justify-between gap-3"
                            >
                              <div className="flex items-start gap-2">
                                <CheckCircle2 className="w-4 h-4 text-red-700 shrink-0 mt-0.5" />
                                <span className="leading-relaxed">{step}</span>
                              </div>

                              <button
                                onClick={() => handleExecuteOpportunityStep(opp.id, i)}
                                disabled={isRunning}
                                className="self-start px-3.5 py-2 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-amber-400 font-bold uppercase tracking-wider text-[11px] flex items-center gap-1.5 transition cursor-pointer"
                              >
                                {isRunning ? (
                                  <>
                                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                    Executing Step {i + 1}...
                                  </>
                                ) : (
                                  <>
                                    <Play className="w-3 h-3" />
                                    {buttonLabels[i] || `Execute Step ${i + 1}`}
                                  </>
                                )}
                              </button>

                              {/* Live Action Result Box */}
                              {stepRes && (
                                <div className="mt-1 pt-3 border-t border-slate-300 space-y-2 text-[11px]">
                                  <div className="font-bold text-slate-900 uppercase">{stepRes.action_title}</div>
                                  <p className="text-slate-700">{stepRes.summary}</p>

                                  {stepRes.metrics && (
                                    <div className="grid grid-cols-2 gap-1.5 bg-stone-100 border border-slate-400 p-2.5">
                                      {Object.entries(stepRes.metrics).map(([k, v]) => (
                                        <div key={k}>
                                          <span className="text-slate-600">{k}: </span>
                                          <strong className="text-slate-900">{String(v)}</strong>
                                        </div>
                                      ))}
                                    </div>
                                  )}

                                  {stepRes.provisional_patent_claim && (
                                    <div className="bg-amber-50 border border-slate-900 p-2.5 text-slate-900 font-mono text-[10px]">
                                      {stepRes.provisional_patent_claim}
                                    </div>
                                  )}

                                  {stepRes.live_registry_checks && (
                                    <div className="bg-stone-100 border border-slate-900 p-2.5 space-y-1">
                                      {Object.entries(stepRes.live_registry_checks).map(([k, v]) => (
                                        <div key={k}>
                                          <span className="text-slate-600">{k}: </span>
                                          <strong className="text-slate-900">{String(v)}</strong>
                                        </div>
                                      ))}
                                    </div>
                                  )}

                                  {stepRes.reproducible_python_code && (
                                    <div className="flex flex-wrap gap-2 pt-1">
                                      <button
                                        onClick={() =>
                                          downloadTextFile(
                                            `thesisout_model_opp_${opp.id}.py`,
                                            stepRes.reproducible_python_code
                                          )
                                        }
                                        className="px-3 py-1.5 bg-slate-900 text-amber-400 font-bold uppercase tracking-wider inline-flex items-center gap-1 cursor-pointer"
                                      >
                                        <Download className="w-3 h-3" /> Download Python Model (.py)
                                      </button>
                                      <button
                                        onClick={() =>
                                          downloadTextFile(
                                            `thesisout_dossier_opp_${opp.id}.md`,
                                            stepRes.venture_dossier_markdown
                                          )
                                        }
                                        className="px-3 py-1.5 bg-red-700 text-white font-bold uppercase tracking-wider inline-flex items-center gap-1 cursor-pointer"
                                      >
                                        <Download className="w-3 h-3" /> Download Dossier (.md)
                                      </button>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Collapsible Moonshot Problem Bank Drawer */}
            <div className="dossier-card overflow-hidden">
              <button
                onClick={() => setShowAllMoonshots(!showAllMoonshots)}
                className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-stone-50 transition cursor-pointer"
              >
                <div>
                  <div className="text-sm font-black uppercase tracking-tight text-slate-900">
                    Inspect All {problems.length} Live Unmet Industry & Clinical Bottlenecks
                  </div>
                  <div className="text-xs text-slate-600">
                    Streamed from terminated ClinicalTrials.gov records and high-urgency engineering challenges
                  </div>
                </div>
                {showAllMoonshots ? (
                  <ChevronUp className="w-5 h-5 text-slate-700" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-slate-700" />
                )}
              </button>

              {showAllMoonshots && (
                <div className="p-6 border-t-2 border-slate-900 bg-stone-50 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  {problems.map((prob) => (
                    <div key={prob.id} className="bg-white border-2 border-slate-900 p-4">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="px-2.5 py-0.5 bg-amber-400 text-slate-950 border border-slate-900 font-bold">
                          {prob.code} | ${prob.tam_billions_usd}B Scale
                        </span>
                        <span className="text-slate-600 font-medium">{prob.sponsor_or_source}</span>
                      </div>
                      <div className="font-bold text-slate-900 text-sm">{prob.title}</div>
                      <p className="text-slate-700 mt-1.5 leading-relaxed">{prob.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-2">
              <button
                onClick={() => setCurrentStep(2)}
                className="px-4 py-2.5 bg-white border-2 border-slate-900 text-slate-900 text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 cursor-pointer"
              >
                <ArrowLeft className="w-4 h-4" /> Return to Stage 02
              </button>
              <button
                onClick={() => setCurrentStep(1)}
                className="px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-amber-400 text-xs font-bold uppercase tracking-wider cursor-pointer"
              >
                Investigate Another Dissertation (Stage 01)
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
