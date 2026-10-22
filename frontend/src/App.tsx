import React, { useEffect, useState } from 'react';
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  ExternalLink,
  FileText,
  MessageSquare,
  Play,
  RefreshCw,
  Search,
  Send,
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
  const [showDeepCheck, setShowDeepCheck] = useState<boolean>(true);

  const [dissertations, setDissertations] = useState<Dissertation[]>([]);
  const [opportunities, setOpportunities] = useState<MarketOpportunity[]>([]);
  const [, setProblems] = useState<IndustryProblem[]>([]);
  const [selectedDissId, setSelectedDissId] = useState<number | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [escalationCache, setEscalationCache] = useState<Record<number, DeepInferenceResult>>({});
  const [escalatingRunId, setEscalatingRunId] = useState<number | null>(null);
  const [stepActionResults, setStepActionResults] = useState<Record<string, any>>({});
  const [executingStepKey, setExecutingStepKey] = useState<string | null>(null);

  // Autonomous Gemini Executive Summary & Hypothesis
  const [executiveSummary, setExecutiveSummary] = useState<string>('');
  const [activeHypothesis, setActiveHypothesis] = useState<string>('');

  const [, setRunningCustomTarget] = useState<boolean>(false);

  const [zenodoInput, setZenodoInput] = useState<string>('');
  const [attachingData, setAttachingData] = useState<boolean>(false);

  const [harvesting, setHarvesting] = useState<boolean>(false);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);
  const [probeData, setProbeData] = useState<any>(null);
  const [, setProbing] = useState<boolean>(false);

  const [chatInput, setChatInput] = useState<string>('');
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<CopilotMessage[]>([]);

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
    setCurrentStep(3);
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

  /**
   * Autonomous End-to-End Gemini Workflow:
   * Ask Gemini -> Formulates hypothesis -> Streams dissertations & datasets -> Runs 30+ LazyPredict models ->
   * Escalates to 5-Stage Deep Inference -> Probes 9 live APIs -> Executes all 4 actions -> Reveals Moonshot & Supporting Evidence!
   */
  const handleRunAutonomousGeminiWorkflow = (promptOverride?: string) => {
    const promptText = (promptOverride ?? chatInput).trim();
    if (!promptText || harvesting) return;

    if (!promptOverride) setChatInput('');
    setHarvesting(true);
    setErrorMsg(null);
    setStreamEvents([`Submitting inquiry to Gemini Autonomous Discovery Engine: "${promptText}"...`]);
    setChatMessages((prev) => [...prev, { role: 'user', content: promptText }]);

    const url = `/api/copilot/autonomous-stream?prompt=${encodeURIComponent(promptText)}`;
    const evtSource = new EventSource(url);

    evtSource.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.message) {
          setStreamEvents((prev) => [...prev, data.message]);
        }
        if (data.step === 'gemini_plan_ready' && data.payload?.plan?.hypothesis) {
          setActiveHypothesis(data.payload.plan.hypothesis);
        }
        if (data.step === 'complete') {
          evtSource.close();
          await fetchAllData();
          setHarvesting(false);

          const payload = data.payload || {};
          if (payload.selected_dissertation_id) {
            setSelectedDissId(payload.selected_dissertation_id);
          }
          if (payload.selected_lazy_run_id) {
            setSelectedRunId(payload.selected_lazy_run_id);
            if (payload.deep_inference) {
              setEscalationCache((prev) => ({
                ...prev,
                [payload.selected_lazy_run_id]: payload.deep_inference,
              }));
            }
          }
          if (payload.probe_data) {
            setProbeData(payload.probe_data);
          }
          if (payload.executed_steps) {
            setStepActionResults((prev) => ({ ...prev, ...payload.executed_steps }));
          }
          if (payload.executive_summary) {
            setExecutiveSummary(payload.executive_summary);
            setChatMessages((prev) => [
              ...prev,
              {
                role: 'assistant',
                content: payload.executive_summary,
                hypothesis_formulation: payload.plan?.hypothesis,
                model_used: 'Gemini Autonomous Discovery Engine',
              },
            ]);
          }
          // Automatically reveal the Moonshot + Supporting Evidence view (Stage 03)
          setCurrentStep(3);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      } catch (err) {
        console.error('Autonomous stream parse error:', err);
      }
    };

    evtSource.onerror = async () => {
      evtSource.close();
      await fetchAllData();
      setHarvesting(false);
    };
  };

  /**
   * Conversational Follow-up Chat with Gemini on already-loaded data
   */
  const handleFollowUpChat = async (customPrompt?: string) => {
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
      handleRunAutonomousGeminiWorkflow(action.query_or_column);
    } else if (action.action_type === 'probe_apis') {
      handleRunLiveProbe(action.query_or_column);
    } else if (action.action_type === 'run_lazypredict_column') {
      handleRunCustomTargetColumn(action.query_or_column);
    } else if (action.action_type === 'run_deep_verification' && activeRun) {
      handleEscalateDeepInference(activeRun.id);
    } else if (action.action_type === 'follow_up_prompt') {
      handleFollowUpChat(action.query_or_column);
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
    setExecutiveSummary('');
    setActiveHypothesis('');
    setChatMessages([]);
    setCurrentStep(1);
  };

  const handleRunLiveProbe = async (q: string) => {
    const targetQ = q.trim() || selectedDiss?.title?.split(' ')[0] || 'EGFR';
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
    <div className="min-h-screen flex flex-col relative">
      {/* Subtle Animated Background with Trajectory Dash Lines & Drifting Magnifying Glass Optics */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" aria-hidden="true">
        <svg
          className="w-full h-full opacity-60"
          viewBox="0 0 1440 900"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          preserveAspectRatio="xMidYMid slice"
        >
          <defs>
            <linearGradient id="trajGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#2563EB" stopOpacity="0.28" />
              <stop offset="50%" stopColor="#4F46E5" stopOpacity="0.22" />
              <stop offset="100%" stopColor="#E11D48" stopOpacity="0.18" />
            </linearGradient>
            <linearGradient id="trajGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#6366F1" stopOpacity="0.24" />
              <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.08" />
            </linearGradient>
            <radialGradient id="lensGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.12" />
              <stop offset="70%" stopColor="#6366F1" stopOpacity="0.04" />
              <stop offset="100%" stopColor="#6366F1" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Animated Curved Trajectory Dash Lines */}
          <path
            d="M -80 680 C 260 520, 480 200, 860 280 C 1140 340, 1290 140, 1520 90"
            stroke="url(#trajGrad1)"
            strokeWidth="2"
            className="trajectory-dash"
          />
          <path
            d="M -60 220 C 320 380, 640 120, 1020 440 C 1220 600, 1380 490, 1540 580"
            stroke="url(#trajGrad2)"
            strokeWidth="1.75"
            className="trajectory-dash-slow"
          />
          <path
            d="M 180 940 C 420 640, 780 690, 1120 320 C 1260 170, 1390 220, 1500 180"
            stroke="url(#trajGrad1)"
            strokeWidth="1.5"
            className="trajectory-dash"
          />

          {/* Trajectory Waypoint Nodes */}
          <circle cx="860" cy="280" r="4" fill="#2563EB" fillOpacity="0.35" />
          <circle cx="860" cy="280" r="11" stroke="#2563EB" strokeOpacity="0.2" strokeWidth="1" />
          <circle cx="1020" cy="440" r="4" fill="#4F46E5" fillOpacity="0.35" />
          <circle cx="1020" cy="440" r="11" stroke="#4F46E5" strokeOpacity="0.2" strokeWidth="1" />
          <circle cx="430" cy="360" r="3.5" fill="#E11D48" fillOpacity="0.3" />

          {/* Floating Magnifying Glass Assembly 1 */}
          <g className="floating-magnifying-lens-1">
            <circle cx="1080" cy="250" r="120" fill="url(#lensGlow)" />
            <circle
              cx="1080"
              cy="250"
              r="76"
              stroke="#3B82F6"
              strokeOpacity="0.22"
              strokeWidth="2"
            />
            <circle
              cx="1080"
              cy="250"
              r="64"
              stroke="#6366F1"
              strokeOpacity="0.16"
              strokeWidth="1"
              strokeDasharray="6 6"
            />
            <path
              d="M 1035 215 A 56 56 0 0 1 1105 198"
              stroke="#2563EB"
              strokeOpacity="0.28"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
            <line
              x1="1134"
              y1="304"
              x2="1192"
              y2="362"
              stroke="#3B82F6"
              strokeOpacity="0.24"
              strokeWidth="6"
              strokeLinecap="round"
            />
          </g>

          {/* Floating Magnifying Glass Assembly 2 */}
          <g className="floating-magnifying-lens-2">
            <circle cx="290" cy="540" r="105" fill="url(#lensGlow)" />
            <circle
              cx="290"
              cy="540"
              r="58"
              stroke="#6366F1"
              strokeOpacity="0.2"
              strokeWidth="1.75"
            />
            <circle
              cx="290"
              cy="540"
              r="46"
              stroke="#3B82F6"
              strokeOpacity="0.14"
              strokeWidth="1"
              strokeDasharray="4 8"
            />
            <path
              d="M 256 512 A 42 42 0 0 1 310 500"
              stroke="#4F46E5"
              strokeOpacity="0.25"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <line
              x1="331"
              y1="581"
              x2="376"
              y2="626"
              stroke="#6366F1"
              strokeOpacity="0.22"
              strokeWidth="5"
              strokeLinecap="round"
            />
          </g>
        </svg>
      </div>

      {/* Top Sapphire-Indigo-Crimson Accent Line */}
      <div className="h-1.5 w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-rose-600 relative z-20" />

      {/* Sleek Glass Header with ThesisOut in Sentence/CamelCase */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-slate-200/80 sticky top-0 z-30">
        <div className="max-w-5xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center shadow-md shadow-blue-500/20">
              <Search className="w-5 h-5 text-white stroke-[2.2]" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="font-extrabold text-xl tracking-tight text-slate-900">
                  Thesis<span className="text-blue-600">Out</span>
                </span>
                <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                  Taking Thesis Discoveries Out Into the World
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Ask Gemini any research question — it autonomously streams theses & datasets, tests 30+ models, and reveals moonshots
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5 text-xs">
            {dissertations.length > 0 && (
              <button
                onClick={handleResetWorkspace}
                className="px-3 py-1.5 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 font-semibold transition cursor-pointer"
              >
                Clear Workspace
              </button>
            )}
            <span className="px-3 py-1.5 rounded-xl bg-indigo-50 text-indigo-700 border border-indigo-200 font-semibold flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5 text-indigo-600" />
              Gemini Autonomous Engine
            </span>
          </div>
        </div>

        {/* 3-Stage Trajectory Stepper */}
        <div className="max-w-5xl mx-auto px-6 pb-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <button
              onClick={() => setCurrentStep(1)}
              className={`text-left p-3.5 rounded-2xl border transition flex items-center gap-3 cursor-pointer ${
                currentStep === 1
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-transparent shadow-md shadow-blue-500/20'
                  : 'bg-white/90 text-slate-800 border-slate-200/90 hover:border-blue-300'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-xs shrink-0 ${
                  currentStep === 1 ? 'bg-white/20 text-white' : 'bg-blue-50 text-blue-700'
                }`}
              >
                01
              </div>
              <div>
                <div className="text-xs font-bold">Ask Gemini (Full Run)</div>
                <div className={`text-[11px] ${currentStep === 1 ? 'text-blue-100' : 'text-slate-500'}`}>
                  Describe what to investigate & launch
                </div>
              </div>
            </button>

            <button
              onClick={() => setCurrentStep(2)}
              className={`text-left p-3.5 rounded-2xl border transition flex items-center gap-3 cursor-pointer ${
                currentStep === 2
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-transparent shadow-md shadow-blue-500/20'
                  : 'bg-white/90 text-slate-800 border-slate-200/90 hover:border-blue-300'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-xs shrink-0 ${
                  currentStep === 2 ? 'bg-white/20 text-white' : 'bg-blue-50 text-blue-700'
                }`}
              >
                02
              </div>
              <div>
                <div className="text-xs font-bold">30-Model & 5-Stage Evidence</div>
                <div className={`text-[11px] ${currentStep === 2 ? 'text-blue-100' : 'text-slate-500'}`}>
                  Inspect leaderboard, CIs & deep verification
                </div>
              </div>
            </button>

            <button
              onClick={() => setCurrentStep(3)}
              className={`text-left p-3.5 rounded-2xl border transition flex items-center gap-3 cursor-pointer ${
                currentStep === 3
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-transparent shadow-md shadow-blue-500/20'
                  : 'bg-white/90 text-slate-800 border-slate-200/90 hover:border-blue-300'
              }`}
            >
              <div
                className={`w-8 h-8 rounded-xl flex items-center justify-center font-bold text-xs shrink-0 ${
                  currentStep === 3 ? 'bg-white/20 text-white' : 'bg-blue-50 text-blue-700'
                }`}
              >
                03
              </div>
              <div>
                <div className="text-xs font-bold">Moonshot & Supporting Dossier</div>
                <div className={`text-[11px] ${currentStep === 3 ? 'text-blue-100' : 'text-slate-500'}`}>
                  Executive summary, actions & full evidence
                </div>
              </div>
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8 relative z-10">
        {errorMsg && (
          <div className="mb-6 bg-rose-50 border border-rose-200 text-rose-800 px-5 py-3.5 rounded-2xl text-xs flex items-center justify-between font-medium">
            <span>{errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="font-bold underline cursor-pointer">
              Dismiss
            </button>
          </div>
        )}

        {/* Active Study Context Banner */}
        {selectedDiss && (
          <div className="mb-6 glass-panel rounded-2xl px-5 py-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className="px-2.5 py-1 rounded-lg bg-blue-50 text-blue-700 border border-blue-200 font-semibold text-xs">
                Active Study
              </span>
              <div>
                <div className="text-xs font-bold text-slate-900 line-clamp-1">
                  {selectedDiss.title}
                </div>
                <div className="text-[11px] text-slate-500 flex flex-wrap items-center gap-2 mt-1">
                  <span>
                    {selectedDiss.authors} • {selectedDiss.university} ({selectedDiss.publication_year})
                  </span>
                  <span className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 border border-slate-200 font-medium">
                    Source: {selectedDiss.source_repo}
                  </span>
                  <span className="px-2 py-0.5 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-200 font-medium">
                    Rigor Confidence: {(selectedDiss.scientific_rigor_score * 100).toFixed(0)}%
                  </span>
                  {selectedDiss.source_url && (
                    <a
                      href={selectedDiss.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-blue-600 font-semibold underline inline-flex items-center gap-1"
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
                className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 text-xs font-medium text-slate-700"
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
        {/* STAGE 01: ASK GEMINI TO RUN THE ENTIRE END-TO-END DISCOVERY WORKFLOW */}
        {/* ===================================================================== */}
        {currentStep === 1 && (
          <div className="space-y-6">
            <div className="text-center max-w-2xl mx-auto">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs font-semibold mb-2">
                <Search className="w-3.5 h-3.5" />
                ThesisOut • Autonomous Scientific Discovery
              </span>
              <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
                Tell Gemini What You Want to Discover
              </h1>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Ask any scientific question or describe a mechanism, disease, or engineering challenge. Gemini will autonomously formulate the hypothesis, stream dissertations and open datasets, run 30+ models and 5-stage deep verification, execute validation steps, and reveal the moonshot with all supporting evidence.
              </p>
            </div>

            {/* Mode Switcher */}
            <div className="flex justify-center">
              <div className="inline-flex bg-white/90 p-1.5 rounded-2xl border border-slate-200/90 shadow-xs gap-1">
                <button
                  onClick={() => setStep1Mode('copilot')}
                  className={`px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer flex items-center gap-1.5 ${
                    step1Mode === 'copilot'
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Ask Gemini (Autonomous Full Workflow)
                </button>
                <button
                  onClick={() => setStep1Mode('upload')}
                  className={`px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer flex items-center gap-1.5 ${
                    step1Mode === 'upload'
                      ? 'bg-blue-600 text-white shadow-xs'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  <Upload className="w-3.5 h-3.5" />
                  Upload Own PDF / CSV
                </button>
                {dissertations.length > 0 && (
                  <button
                    onClick={() => setStep1Mode('saved')}
                    className={`px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer flex items-center gap-1.5 ${
                      step1Mode === 'saved'
                        ? 'bg-blue-600 text-white shadow-xs'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    Saved Studies ({dissertations.length})
                  </button>
                )}
              </div>
            </div>

            {/* MODE A: Single Unified Gemini Command & Conversation Box */}
            {step1Mode === 'copilot' && (
              <div className="glass-panel rounded-3xl p-8 max-w-3xl mx-auto space-y-6">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleRunAutonomousGeminiWorkflow();
                  }}
                  className="space-y-4"
                >
                  <label className="block text-sm font-bold text-slate-900">
                    What scientific question, biological mechanism, or technological bottleneck should Gemini investigate?
                  </label>
                  <div className="flex flex-col sm:flex-row gap-3">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder="Describe your idea or question in plain English..."
                      className="flex-1 bg-white border border-slate-300 focus:border-blue-600 rounded-2xl px-4 py-3.5 text-sm text-slate-900 focus:outline-none shadow-2xs"
                    />
                    <button
                      type="submit"
                      disabled={harvesting || !chatInput.trim()}
                      className="px-6 py-3.5 rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-md shadow-blue-500/20 transition cursor-pointer shrink-0"
                    >
                      {harvesting ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Running Full Discovery Workflow...
                        </>
                      ) : (
                        <>
                          <Send className="w-4 h-4" />
                          Ask Gemini & Run Full Discovery
                        </>
                      )}
                    </button>
                  </div>
                </form>

                {/* Live Autonomous Telemetry Log */}
                {streamEvents.length > 0 && (
                  <div className="glass-subpanel rounded-2xl p-5 space-y-2.5 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-blue-700 uppercase tracking-wider text-[11px]">
                        Autonomous Gemini Workflow Execution Log
                      </span>
                      {harvesting && (
                        <span className="text-[11px] text-indigo-600 font-semibold flex items-center gap-1.5">
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Executing Pipeline...
                        </span>
                      )}
                    </div>
                    <div className="space-y-1.5 font-mono text-[11px]">
                      {streamEvents.map((msg, i) => (
                        <div key={i} className="flex items-start gap-2.5 text-slate-700">
                          <span className="w-2 h-2 rounded-full bg-blue-600 shrink-0 mt-1.5" />
                          <span>{msg}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* MODE B: Upload PDF / CSV */}
            {step1Mode === 'upload' && (
              <div className="glass-panel rounded-3xl p-8 max-w-2xl mx-auto">
                <div className="flex items-center gap-2.5 text-slate-900 font-bold text-base mb-1">
                  <Upload className="w-5 h-5 text-blue-600" />
                  Submit Dissertation PDF or Tabular Dataset
                </div>
                <p className="text-xs text-slate-500 mb-5 leading-relaxed">
                  Upload a local CSV, TSV, Excel, or Parquet dataset along with an optional dissertation PDF. ThesisOut will execute 30+ LazyPredict models and synthesize the moonshot dossier.
                </p>

                <form onSubmit={handleUploadSubmit} className="space-y-4 text-xs">
                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">
                      Study or Dissertation Title *
                    </label>
                    <input
                      type="text"
                      required
                      value={uploadTitle}
                      onChange={(e) => setUploadTitle(e.target.value)}
                      placeholder="Enter study or dissertation title"
                      className="w-full bg-white border border-slate-300 focus:border-blue-500 rounded-xl px-4 py-2.5 text-slate-900 focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">
                      Institution (Optional)
                    </label>
                    <input
                      type="text"
                      value={uploadUni}
                      onChange={(e) => setUploadUni(e.target.value)}
                      placeholder="e.g., MIT, Stanford"
                      className="w-full bg-white border border-slate-300 focus:border-blue-500 rounded-xl px-4 py-2.5 text-slate-900 focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                    <div className="border border-dashed border-blue-300 rounded-2xl p-4 bg-blue-50/40">
                      <label className="block font-semibold text-blue-900 mb-1">
                        Dataset File (.csv, .xlsx, .tsv)
                      </label>
                      <input
                        type="file"
                        accept=".csv,.tsv,.xlsx,.parquet"
                        onChange={(e) => setUploadCsv(e.target.files?.[0] || null)}
                        className="w-full text-slate-600 text-[11px]"
                      />
                    </div>
                    <div className="border border-dashed border-indigo-300 rounded-2xl p-4 bg-indigo-50/40">
                      <label className="block font-semibold text-indigo-900 mb-1">
                        Dissertation PDF (Optional)
                      </label>
                      <input
                        type="file"
                        accept=".pdf"
                        onChange={(e) => setUploadPdf(e.target.files?.[0] || null)}
                        className="w-full text-slate-600 text-[11px]"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block font-semibold text-slate-700 mb-1">
                      Target Column to Predict (Leave blank for automatic selection)
                    </label>
                    <input
                      type="text"
                      value={uploadTarget}
                      onChange={(e) => setUploadTarget(e.target.value)}
                      placeholder="Optional target column name"
                      className="w-full bg-white border border-slate-300 focus:border-blue-500 rounded-xl px-4 py-2.5 text-slate-900 focus:outline-none"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={uploading}
                    className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-sm shadow-blue-500/20 transition cursor-pointer"
                  >
                    {uploading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Auditing Dataset Across 30+ Models...
                      </>
                    ) : (
                      <>
                        <Upload className="w-4 h-4" />
                        Ingest Evidence & Reveal Moonshot
                      </>
                    )}
                  </button>
                </form>
              </div>
            )}

            {/* MODE C: Saved Studies */}
            {step1Mode === 'saved' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {dissertations.map((diss) => {
                  const run = diss.datasets?.[0]?.lazy_runs?.[0];
                  return (
                    <div
                      key={diss.id}
                      onClick={() => handleSelectDissertation(diss)}
                      className="glass-panel rounded-3xl p-6 transition cursor-pointer flex flex-col justify-between hover:border-blue-400"
                    >
                      <div>
                        <div className="flex items-center justify-between text-xs mb-2">
                          <span className="px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 font-semibold">
                            {diss.university}
                          </span>
                          <span className="text-slate-500 font-medium">{diss.publication_year}</span>
                        </div>
                        <h3 className="text-base font-bold text-slate-900 leading-snug">{diss.title}</h3>
                        <p className="text-xs text-slate-500 mt-2 line-clamp-2">{diss.abstract}</p>
                      </div>

                      <div className="mt-4 pt-3 border-t border-slate-200/80 flex items-center justify-between text-xs">
                        {run ? (
                          <span className="px-2.5 py-1 rounded-lg bg-indigo-50 text-indigo-700 font-semibold border border-indigo-200">
                            Best Model: {run.best_model_name} ({run.best_score})
                          </span>
                        ) : (
                          <span className="text-slate-400">Click to attach dataset</span>
                        )}
                        <span className="font-semibold text-blue-600 flex items-center gap-1">
                          Open Dossier <ArrowRight className="w-3.5 h-3.5" />
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
        {/* STAGE 02: 30-MODEL & 5-STAGE EMPIRICAL EVIDENCE */}
        {/* ===================================================================== */}
        {currentStep === 2 && (
          <div className="space-y-6">
            {!selectedDiss ? (
              <div className="glass-panel rounded-3xl p-10 text-center max-w-lg mx-auto">
                <BookOpen className="w-10 h-10 text-blue-600 mx-auto mb-3" />
                <h3 className="text-lg font-bold text-slate-900">No Active Study Selected</h3>
                <p className="text-xs text-slate-500 mt-1 mb-5">
                  Return to Stage 01 and ask Gemini a question to run the discovery pipeline.
                </p>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="px-5 py-2.5 rounded-xl bg-blue-600 text-white text-xs font-semibold cursor-pointer"
                >
                  Return to Stage 01
                </button>
              </div>
            ) : !activeRun ? (
              <div className="glass-panel rounded-3xl p-8 max-w-xl mx-auto space-y-5">
                <div className="text-center">
                  <span className="px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs font-semibold">
                    Dataset Required
                  </span>
                  <h3 className="text-lg font-bold text-slate-900 mt-2">
                    Attach a Dataset to Execute the 30-Model Audit
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Upload a CSV/Excel table or enter a Zenodo Record ID below to benchmark 30+ machine learning models automatically.
                  </p>
                </div>

                <form onSubmit={handleAttachZenodo} className="flex gap-2">
                  <input
                    type="text"
                    value={zenodoInput}
                    onChange={(e) => setZenodoInput(e.target.value)}
                    placeholder="Enter Zenodo Record ID or URL..."
                    className="flex-1 bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-xs"
                  />
                  <button
                    type="submit"
                    disabled={attachingData}
                    className="px-4 py-2.5 rounded-xl bg-blue-600 text-white text-xs font-semibold cursor-pointer"
                  >
                    {attachingData ? 'Fetching...' : 'Fetch Zenodo CSV'}
                  </button>
                </form>

                <div className="border border-dashed border-indigo-300 rounded-2xl p-4 bg-indigo-50/40 text-center">
                  <label className="block text-xs font-semibold text-indigo-900 mb-2">
                    Or Upload a Local Dataset File (.csv, .tsv, .xlsx, .parquet)
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
                <div className="glass-panel rounded-3xl p-6">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs font-semibold">
                          <Search className="w-3.5 h-3.5" />
                          Empirical Signal & Confidence Report
                        </div>
                        {activeDataset && (
                          <span className="px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 text-xs font-medium">
                            Dataset Provenance: {activeDataset.source_type.toUpperCase()} ({activeDataset.row_count} rows × {activeDataset.col_count} cols)
                          </span>
                        )}
                        {activeDataset?.source_url && (
                          <a
                            href={activeDataset.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="px-3 py-1 rounded-full bg-indigo-50 border border-indigo-200 text-indigo-700 text-xs font-semibold underline inline-flex items-center gap-1"
                          >
                            Dataset Origin Link <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                      <h2 className="text-lg font-bold text-slate-900">
                        Audited {activeRun.models_tested_count} models on target column{' '}
                        <code className="px-2 py-0.5 rounded-md bg-blue-50 border border-blue-200 text-blue-700 font-mono">
                          {activeRun.target_column}
                        </code>
                      </h2>
                      <p className="text-sm text-slate-600 leading-relaxed">
                        {activeRun.overlooked_signal_summary}
                      </p>
                    </div>

                    <button
                      onClick={() => {
                        setCurrentStep(3);
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                      className="px-5 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold text-xs flex items-center gap-2 shrink-0 shadow-sm shadow-blue-500/20 transition cursor-pointer"
                    >
                      View Moonshot & Full Dossier <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>

                  {activeDataset && activeDataset.lazy_runs.length > 1 && (
                    <div className="mt-4 pt-3 border-t border-slate-200/80 flex flex-wrap items-center gap-2 text-xs">
                      <span className="font-semibold text-slate-600">Audited Columns:</span>
                      {activeDataset.lazy_runs.map((run) => (
                        <button
                          key={run.id}
                          onClick={() => setSelectedRunId(run.id)}
                          className={`px-3 py-1 rounded-full font-semibold border transition cursor-pointer ${
                            activeRun.id === run.id
                              ? 'bg-blue-600 text-white border-blue-600'
                              : 'bg-white text-slate-700 border-slate-200 hover:border-blue-300'
                          }`}
                        >
                          {run.target_column} ({run.best_score})
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {/* 3 Evidence Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  <div className="glass-panel rounded-3xl p-5 flex flex-col justify-between">
                    <div>
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                        01. Classical Linear Baseline
                      </span>
                      <div className="text-3xl font-extrabold text-slate-900 mt-2">
                        {activeRun.author_baseline_score}
                      </div>
                      <div className="text-xs font-medium text-slate-600 mt-0.5">
                        Metric: {activeRun.primary_metric_name}
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 mt-4 leading-relaxed">
                      Baseline performance of conventional linear statistical methods prior to nonlinear ensemble screening.
                    </p>
                  </div>

                  <div className="glass-panel rounded-3xl p-5 flex flex-col justify-between border-blue-300/80">
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-blue-700">
                          02. Top Model Identified
                        </span>
                        <span className="px-2.5 py-0.5 rounded-full bg-blue-600 text-white text-[11px] font-bold">
                          +{(activeRun.signal_gain_delta * 100).toFixed(1)}% Gain
                        </span>
                      </div>
                      <div className="text-3xl font-extrabold text-blue-700 mt-2">
                        {activeRun.best_score}
                      </div>
                      <div className="text-xs font-bold text-slate-900 mt-0.5">
                        Model: {activeRun.best_model_name}
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 mt-4 leading-relaxed">
                      Selected after benchmarking {activeRun.models_tested_count} distinct machine learning architectures on identical folds.
                    </p>
                  </div>

                  <div className="glass-panel rounded-3xl p-5">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-700">
                      03. Primary Predictive Variables
                    </span>
                    <div className="space-y-3 mt-3">
                      {activeRun.top_features.slice(0, 3).map((tf, i) => (
                        <div key={tf.feature} className="glass-subpanel rounded-xl p-2.5">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-bold text-slate-800 truncate pr-2">
                              #{i + 1} {tf.feature}
                            </span>
                            <span className="font-semibold text-indigo-600 shrink-0">
                              {(tf.importance * 100).toFixed(0)}% weight
                            </span>
                          </div>
                          <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden mt-1.5">
                            <div
                              className="h-full bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full"
                              style={{ width: `${Math.min(100, tf.importance * 180)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Full Leaderboard Drawer */}
                <div className="glass-panel rounded-3xl overflow-hidden">
                  <button
                    onClick={() => setShowAllModels(!showAllModels)}
                    className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-slate-50/70 transition cursor-pointer"
                  >
                    <div className="text-sm font-bold text-slate-900">
                      Inspect Full {activeRun.leaderboard.length}-Model Leaderboard & 95% Confidence Intervals
                    </div>
                    {showAllModels ? (
                      <ChevronUp className="w-5 h-5 text-slate-500" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-slate-500" />
                    )}
                  </button>

                  {showAllModels && (
                    <div className="px-6 pb-6 pt-2 border-t border-slate-200/80 overflow-x-auto max-h-80">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 text-slate-600 font-semibold">
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
                              className={idx === 0 ? 'bg-blue-50/80 font-bold text-blue-900' : 'text-slate-700'}
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
              </>
            )}
          </div>
        )}

        {/* ===================================================================== */}
        {/* STAGE 03: UNIFIED MOONSHOT REVEAL + GEMINI SUMMARY + ALL SUPPORTING EVIDENCE */}
        {/* ===================================================================== */}
        {currentStep === 3 && (
          <div className="space-y-6">
            {/* 1. Gemini Executive Summary & Formulated Hypothesis Banner */}
            {(executiveSummary || activeHypothesis) && (
              <div className="glass-panel rounded-3xl p-6 border-blue-300/90 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/80 pb-3">
                  <div className="flex items-center gap-2 text-sm font-extrabold text-slate-900">
                    <MessageSquare className="w-4 h-4 text-blue-600" />
                    Gemini Executive Discovery Summary & Moonshot Synthesis
                  </div>
                  <span className="px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-[11px] font-semibold">
                    Full Workflow Executed Automatically
                  </span>
                </div>

                {activeHypothesis && (
                  <div className="bg-indigo-50/70 border border-indigo-200 rounded-2xl p-3.5 text-xs">
                    <div className="font-bold uppercase tracking-wider text-[10px] text-indigo-700">
                      Formulated Scientific Hypothesis
                    </div>
                    <div className="font-semibold text-slate-900 mt-0.5">{activeHypothesis}</div>
                  </div>
                )}

                {executiveSummary && (
                  <p className="text-xs text-slate-700 leading-relaxed whitespace-pre-line">
                    {executiveSummary}
                  </p>
                )}
              </div>
            )}

            {/* 2. Revealed Moonshot Dossier & Pre-Executed Actions */}
            {activeOpportunities.length === 0 ? (
              <div className="glass-panel rounded-3xl p-10 text-center max-w-lg mx-auto">
                <Search className="w-10 h-10 text-blue-600 mx-auto mb-3" />
                <h3 className="text-base font-bold text-slate-900">
                  No Moonshot Dossiers Generated Yet
                </h3>
                <p className="text-xs text-slate-500 mt-1 mb-5">
                  Go to Stage 01 and ask Gemini a research question—it will run the entire workflow and reveal the moonshot here automatically.
                </p>
                <button
                  onClick={() => setCurrentStep(1)}
                  className="px-5 py-2.5 rounded-xl bg-blue-600 text-white text-xs font-semibold cursor-pointer"
                >
                  Ask Gemini in Stage 01
                </button>
              </div>
            ) : (
              <div className="space-y-6">
                {activeOpportunities.map((opp) => (
                  <div key={opp.id} className="glass-panel rounded-3xl p-6 space-y-5">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <span className="px-3 py-1 rounded-full bg-blue-600 text-white text-xs font-bold shadow-xs">
                        Moonshot Translation Score: {opp.overall_tos_score} / 100
                      </span>
                      <span className="px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 text-xs font-semibold border border-indigo-200">
                        {opp.opportunity_type}
                      </span>
                    </div>

                    <h3 className="text-xl font-extrabold text-slate-900 leading-snug">{opp.title}</h3>

                    {/* 3 Case Notes */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                      <div className="glass-subpanel rounded-2xl p-4">
                        <div className="font-bold text-blue-700 mb-1.5">
                          01. Empirical Signal Uncovered
                        </div>
                        <p className="text-slate-700 leading-relaxed">{opp.untranslated_finding_summary}</p>
                      </div>

                      <div className="glass-subpanel rounded-2xl p-4">
                        <div className="font-bold text-indigo-700 mb-1.5">
                          02. Why It Remained Untranslated
                        </div>
                        <p className="text-slate-700 leading-relaxed">{opp.valley_of_death_diagnosis}</p>
                      </div>

                      <div className="glass-subpanel rounded-2xl p-4">
                        <div className="font-bold text-rose-700 mb-1.5">
                          03. Live Registry Whitespace
                        </div>
                        <p className="text-slate-700 leading-relaxed">
                          <strong>ClinicalTrials.gov:</strong> {opp.clinical_trials_check?.whitespace_status}
                        </p>
                        <p className="text-slate-700 mt-1.5 leading-relaxed">
                          <strong>Target / Modality:</strong> {opp.opentargets_chembl_check?.tractability_modality}
                        </p>
                      </div>
                    </div>

                    {/* Proposed Solution & 4 Executed Action Steps */}
                    <div className="glass-subpanel rounded-2xl p-5 text-xs space-y-4">
                      <div>
                        <div className="font-bold text-indigo-700 uppercase tracking-wider text-[11px] mb-1">
                          Revealed Moonshot Solution & Executed Actions
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
                              className="bg-white border border-slate-200/90 rounded-2xl p-4 text-slate-700 flex flex-col justify-between gap-3 shadow-2xs"
                            >
                              <div className="flex items-start gap-2">
                                <CheckCircle2 className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
                                <span className="leading-relaxed">{step}</span>
                              </div>

                              <button
                                onClick={() => handleExecuteOpportunityStep(opp.id, i)}
                                disabled={isRunning}
                                className="self-start px-3.5 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 text-white font-semibold text-[11px] flex items-center gap-1.5 transition cursor-pointer"
                              >
                                {isRunning ? (
                                  <>
                                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                    Executing Step {i + 1}...
                                  </>
                                ) : (
                                  <>
                                    <Play className="w-3 h-3" />
                                    {stepRes ? `Re-Run Step ${i + 1}` : buttonLabels[i] || `Execute Step ${i + 1}`}
                                  </>
                                )}
                              </button>

                              {stepRes && (
                                <div className="mt-1 pt-3 border-t border-slate-200 space-y-2 text-[11px]">
                                  <div className="font-bold text-blue-700">{stepRes.action_title}</div>
                                  <p className="text-slate-600">{stepRes.summary}</p>

                                  {stepRes.metrics && (
                                    <div className="grid grid-cols-2 gap-1.5 bg-slate-50 border border-slate-200 rounded-xl p-2.5">
                                      {Object.entries(stepRes.metrics).map(([k, v]) => (
                                        <div key={k}>
                                          <span className="text-slate-500">{k}: </span>
                                          <strong className="text-slate-900">{String(v)}</strong>
                                        </div>
                                      ))}
                                    </div>
                                  )}

                                  {stepRes.provisional_patent_claim && (
                                    <div className="bg-indigo-50/70 border border-indigo-200 rounded-xl p-2.5 text-slate-800 font-mono text-[10px]">
                                      {stepRes.provisional_patent_claim}
                                    </div>
                                  )}

                                  {stepRes.live_registry_checks && (
                                    <div className="bg-blue-50/60 border border-blue-200 rounded-xl p-2.5 space-y-1">
                                      {Object.entries(stepRes.live_registry_checks).map(([k, v]) => (
                                        <div key={k}>
                                          <span className="text-slate-500">{k}: </span>
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
                                        className="px-3 py-1.5 rounded-lg bg-blue-600 text-white font-semibold inline-flex items-center gap-1 cursor-pointer"
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
                                        className="px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-semibold inline-flex items-center gap-1 cursor-pointer"
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

            {/* 3. Supporting Empirical Evidence Section (30-Model Audit + 5-Stage Deep Verification + 9-API Telemetry) */}
            {activeRun && (
              <div className="glass-panel rounded-3xl p-6 space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/80 pb-3">
                  <div>
                    <div className="text-sm font-extrabold text-slate-900">
                      Supporting Empirical Evidence & Statistical Verification
                    </div>
                    <div className="text-xs text-slate-500">
                      Backed by {activeRun.models_tested_count} benchmarked models on{' '}
                      <code>{activeRun.target_column}</code>{' '}
                      {activeDataset ? `(${activeDataset.name}, ${activeDataset.row_count} rows)` : ''}
                    </div>
                  </div>
                  <button
                    onClick={() => setShowAllModels(!showAllModels)}
                    className="px-3.5 py-1.5 rounded-xl bg-blue-50 text-blue-700 border border-blue-200 text-xs font-semibold cursor-pointer"
                  >
                    {showAllModels ? 'Hide Full 30-Model Table' : 'Show Full 30-Model Leaderboard'}
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                  <div className="glass-subpanel rounded-2xl p-4">
                    <div className="text-slate-500 font-semibold">Classical Baseline</div>
                    <div className="text-2xl font-extrabold text-slate-900 mt-1">
                      {activeRun.author_baseline_score}
                    </div>
                    <div className="text-slate-500 mt-0.5">{activeRun.primary_metric_name}</div>
                  </div>

                  <div className="glass-subpanel rounded-2xl p-4 border-blue-300">
                    <div className="text-blue-700 font-bold">
                      Winning Model ({activeRun.best_model_name})
                    </div>
                    <div className="text-2xl font-extrabold text-blue-700 mt-1">
                      {activeRun.best_score}{' '}
                      <span className="text-xs font-bold text-indigo-600">
                        (+{(activeRun.signal_gain_delta * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div className="text-slate-500 mt-0.5">
                      Top Clues: {activeRun.top_features.slice(0, 3).map((f) => f.feature).join(', ')}
                    </div>
                  </div>

                  <div className="glass-subpanel rounded-2xl p-4">
                    <div className="text-indigo-700 font-bold">5-Stage Deep Verification</div>
                    {escalationCache[activeRun.id] ? (
                      <>
                        <div className="text-2xl font-extrabold text-indigo-700 mt-1">
                          {escalationCache[activeRun.id].stage2_super_learner_ensemble.super_learner_score}
                        </div>
                        <div className="text-slate-500 mt-0.5">
                          Super-Learner Stacking • Permutation Verified
                        </div>
                      </>
                    ) : (
                      <button
                        onClick={() => handleEscalateDeepInference(activeRun.id)}
                        className="mt-2 px-3 py-1.5 rounded-xl bg-indigo-600 text-white font-semibold cursor-pointer"
                      >
                        {escalatingRunId === activeRun.id ? 'Verifying...' : 'Run 5-Stage Verification'}
                      </button>
                    )}
                  </div>
                </div>

                {showDeepCheck && escalationCache[activeRun.id] && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-1">
                    <div className="bg-white border border-slate-200 rounded-2xl p-3.5">
                      <div className="font-bold text-blue-700 mb-1">Permutation & Leakage Audit</div>
                      <p className="text-slate-600">
                        {escalationCache[activeRun.id].stage1_leakage_and_permutation_audit.audit_interpretation}
                      </p>
                    </div>
                    <div className="bg-white border border-slate-200 rounded-2xl p-3.5">
                      <div className="font-bold text-indigo-700 mb-1">Mechanistic Regime Thresholds</div>
                      <p className="text-slate-600">
                        {escalationCache[activeRun.id].stage3_mechanistic_regime_discovery.synergy_conclusion}
                      </p>
                    </div>
                  </div>
                )}

                {showAllModels && (
                  <div className="overflow-x-auto max-h-72 border-t border-slate-200 pt-3">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-600 font-semibold">
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
                            className={idx === 0 ? 'bg-blue-50/80 font-bold text-blue-900' : 'text-slate-700'}
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
            )}

            {/* 4. Supporting 9-API Live Scientific Telemetry */}
            {probeData && (
              <div className="glass-panel rounded-3xl p-6 space-y-3 text-xs">
                <div className="font-extrabold text-slate-900 text-sm">
                  Supporting 9-Database Live Scientific Telemetry ("{probeData.query}")
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="glass-subpanel rounded-2xl p-3.5">
                    <div className="font-bold text-blue-700">OpenTargets GraphQL v4</div>
                    <div className="text-slate-700 mt-1">
                      Target: {probeData.live_integrations?.opentargets_graphql_v4?.target_profile?.symbol || 'Queried'} • Known Drugs:{' '}
                      {probeData.live_integrations?.opentargets_graphql_v4?.target_profile?.unique_known_drugs ?? 0}
                    </div>
                  </div>
                  <div className="glass-subpanel rounded-2xl p-3.5">
                    <div className="font-bold text-indigo-700">ChEMBL EBI REST API</div>
                    <div className="text-slate-700 mt-1">
                      Targets Found: {probeData.live_integrations?.chembl_ebi_rest?.targets_found?.length || 0} • Max Clinical Phase:{' '}
                      {probeData.live_integrations?.chembl_ebi_rest?.max_clinical_phase_observed ?? 0}
                    </div>
                  </div>
                  <div className="glass-subpanel rounded-2xl p-3.5">
                    <div className="font-bold text-rose-700">ClinicalTrials.gov v2 & openFDA</div>
                    <div className="text-slate-700 mt-1">
                      Trials: {probeData.live_integrations?.clinicaltrials_gov_v2?.total_trials ?? 0} • FAERS Reports:{' '}
                      {probeData.live_integrations?.openfda_faers?.total_adverse_event_reports ?? 0}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 5. Follow-Up Conversation with Gemini */}
            <div className="glass-panel rounded-3xl p-6 space-y-4">
              <div className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-blue-600" />
                Continue Conversing with Gemini or Launch a New Investigation
              </div>
              <div className="flex flex-col sm:flex-row gap-2.5">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask a follow-up question about this moonshot, or enter a new topic to run..."
                  className="flex-1 bg-white border border-slate-300 focus:border-blue-600 rounded-xl px-4 py-2.5 text-xs text-slate-900 focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => handleFollowUpChat()}
                  disabled={chatLoading || !chatInput.trim()}
                  className="px-4 py-2.5 rounded-xl bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 font-semibold text-xs cursor-pointer"
                >
                  {chatLoading ? 'Thinking...' : 'Ask Follow-Up on This Study'}
                </button>
                <button
                  type="button"
                  onClick={() => handleRunAutonomousGeminiWorkflow()}
                  disabled={harvesting || !chatInput.trim()}
                  className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold text-xs cursor-pointer"
                >
                  Run Full Workflow on New Topic
                </button>
              </div>

              {chatMessages.length > 0 && (
                <div className="space-y-2.5 max-h-60 overflow-y-auto pt-2">
                  {chatMessages.slice(-4).map((m, idx) => (
                    <div
                      key={idx}
                      className={`p-3.5 rounded-2xl text-xs ${
                        m.role === 'user'
                          ? 'bg-blue-600 text-white ml-12'
                          : 'glass-subpanel text-slate-800 mr-6'
                      }`}
                    >
                      <div className="font-bold text-[10px] uppercase opacity-75 mb-1">
                        {m.role === 'user' ? 'You' : m.model_used || 'Gemini'}
                      </div>
                      <div className="whitespace-pre-line leading-relaxed">{m.content}</div>
                      {m.suggested_actions && m.suggested_actions.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-2">
                          {m.suggested_actions.map((act, i) => (
                            <button
                              key={i}
                              onClick={() => handleCopilotActionClick(act)}
                              className="px-2.5 py-1 rounded-lg bg-blue-600 text-white text-[11px] font-semibold cursor-pointer"
                            >
                              {act.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
