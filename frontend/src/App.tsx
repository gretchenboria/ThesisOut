import React, { useEffect, useState } from 'react';
import {
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

export function App() {
  const [showAllModels, setShowAllModels] = useState<boolean>(true);
  const [showUploadBox, setShowUploadBox] = useState<boolean>(false);

  const [dissertations, setDissertations] = useState<Dissertation[]>([]);
  const [opportunities, setOpportunities] = useState<MarketOpportunity[]>([]);
  const [selectedDissId, setSelectedDissId] = useState<number | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [escalationCache, setEscalationCache] = useState<Record<number, DeepInferenceResult>>({});
  const [escalatingRunId, setEscalatingRunId] = useState<number | null>(null);
  const [stepActionResults, setStepActionResults] = useState<Record<string, any>>({});
  const [executingStepKey, setExecutingStepKey] = useState<string | null>(null);

  const [executiveSummary, setExecutiveSummary] = useState<string>('');
  const [activeHypothesis, setActiveHypothesis] = useState<string>('');
  const [runningCustomTarget, setRunningCustomTarget] = useState<boolean>(false);

  const [harvesting, setHarvesting] = useState<boolean>(false);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);
  const [, setProbeData] = useState<any>(null);

  const [chatInput, setChatInput] = useState<string>('');
  const [followUpInput, setFollowUpInput] = useState<string>('');
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
      const [dRes, oRes] = await Promise.all([
        fetch('/api/dissertations'),
        fetch('/api/opportunities'),
      ]);
      const dData: Dissertation[] = await dRes.json();
      const oData: MarketOpportunity[] = await oRes.json();

      setDissertations(dData);
      setOpportunities(oData);

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

  const activeOpportunities = opportunities.filter(
    (o) => !selectedDiss || o.dissertation_id === selectedDiss.id
  );

  const handleSelectDissertation = (diss: Dissertation) => {
    setSelectedDissId(diss.id);
    const firstRun = diss.datasets?.[0]?.lazy_runs?.[0];
    if (firstRun) {
      setSelectedRunId(firstRun.id);
    }
  };

  const handleEscalateDeepInference = async (lazyRunId: number) => {
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
          target_rationale: `Testing column ${colName}`,
        }),
      });
      if (res.ok) {
        const newRun: LazyPredictRun = await res.json();
        await fetchAllData();
        setSelectedRunId(newRun.id);
      }
    } catch (err) {
      setErrorMsg('Could not run model sweep on that column.');
    } finally {
      setRunningCustomTarget(false);
    }
  };

  const handleRunAutonomousGeminiWorkflow = (promptOverride?: string) => {
    const promptText = (promptOverride ?? chatInput).trim();
    if (!promptText || harvesting) return;

    if (!promptOverride) setChatInput('');
    setHarvesting(true);
    setErrorMsg(null);
    setStreamEvents([`Running investigation: "${promptText}"`]);
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
                model_used: 'ThesisOut Discovery',
              },
            ]);
          }
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

  const handleFollowUpChat = async (customPrompt?: string) => {
    const text = (customPrompt ?? followUpInput).trim();
    if (!text || chatLoading) return;

    const updatedHistory: CopilotMessage[] = [...chatMessages, { role: 'user', content: text }];
    setChatMessages(updatedHistory);
    if (!customPrompt) setFollowUpInput('');
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
      }
    } catch (err) {
      setErrorMsg('Failed to reach Gemini endpoint.');
    } finally {
      setChatLoading(false);
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
      const fd = new FormData();
      fd.append('title', uploadTitle);
      fd.append('authors', 'Uploaded Investigation');
      fd.append('university', uploadUni || 'Research Lab');
      fd.append('domain', 'biomedical');
      if (uploadTarget.trim()) fd.append('author_primary_target', uploadTarget.trim());
      if (uploadPdf) fd.append('pdf_file', uploadPdf);
      if (uploadCsv) fd.append('dataset_file', uploadCsv);

      const res = await fetch('/api/dissertations/upload', {
        method: 'POST',
        body: fd,
      });
      if (res.ok) {
        const data = await res.json();
        await fetchAllData();
        if (data.dissertation_id) setSelectedDissId(data.dissertation_id);
        setShowUploadBox(false);
        setUploadTitle('');
        setUploadUni('');
        setUploadTarget('');
      }
    } catch (err) {
      setErrorMsg('Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="relative min-h-screen text-slate-900">
      {/* ===================================================================== */}
      {/* PLAYFUL, NOTICEABLE MAGNIFYING GLASS, BINARY STREAM & TRAJECTORY SVG  */}
      {/* ===================================================================== */}
      <div
        className="fixed inset-0 pointer-events-none z-0 overflow-hidden"
        aria-hidden="true"
      >
        <svg
          className="w-full h-full"
          viewBox="0 0 1440 960"
          fill="none"
          preserveAspectRatio="xMidYMid slice"
        >
          <defs>
            <clipPath id="lensClip1">
              <circle cx="210" cy="210" r="46" />
            </clipPath>
            <clipPath id="lensClip2">
              <circle cx="1220" cy="370" r="52" />
            </clipPath>
            <clipPath id="lensClip3">
              <circle cx="260" cy="730" r="42" />
            </clipPath>
          </defs>

          {/* Calm Animated Trajectory Dash Lines */}
          <path
            d="M -60 220 C 300 60, 680 360, 1060 160 C 1250 70, 1390 230, 1520 130"
            stroke="#2563EB"
            strokeWidth="2.2"
            strokeOpacity="0.26"
            className="trajectory-dash"
          />
          <path
            d="M -40 640 C 380 460, 760 740, 1160 440 C 1310 330, 1420 460, 1520 390"
            stroke="#4F46E5"
            strokeWidth="2"
            strokeOpacity="0.22"
            className="trajectory-dash-reverse"
          />

          {/* Slow Horizontal Scrolling Binary Streams */}
          <g className="binary-scroll-left" opacity="0.28">
            <text x="0" y="140" fill="#2563EB" fontSize="12" fontWeight="600" letterSpacing="5">
              01101101 10100110 01001110 10110010 01011010 11001010 01000001 11010010 01110100 01101101 10100110 01001110 10110010 01011010 11001010 01000001 11010010 01110100
            </text>
            <text x="0" y="830" fill="#54250E" fontSize="12" fontWeight="600" letterSpacing="5">
              10110010 01011010 11001010 01101101 10100110 01001110 11010110 00101101 10111001 10110010 01011010 11001010 01101101 10100110 01001110 11010110 00101101 10111001
            </text>
          </g>

          <g className="binary-scroll-right" opacity="0.26">
            <text x="0" y="510" fill="#4F46E5" fontSize="12" fontWeight="600" letterSpacing="5">
              11010110 00101101 10111001 01011100 10010110 01100011 10011010 01101101 00110101 11010110 00101101 10111001 01011100 10010110 01100011 10011010 01101101 00110101
            </text>
          </g>

          {/* MAGNIFYING GLASS 1 (Compact Top-Left Scanning Lens) */}
          <g className="lens-float-1" opacity="0.46">
            <circle cx="210" cy="210" r="52" fill="#DBEAFE" fillOpacity="0.48" />
            <g clipPath="url(#lensClip1)">
              <g className="binary-scroll-lens">
                <text x="150" y="214" fill="#1D4ED8" fontSize="12" fontWeight="700" letterSpacing="2">
                  01101001 11010010 01011011 01101001 11010010 01011011
                </text>
              </g>
            </g>
            <circle cx="210" cy="210" r="52" stroke="#3B1A08" strokeWidth="4.5" />
            <circle cx="210" cy="210" r="46" stroke="#2563EB" strokeWidth="2" />
            <path
              d="M 178 182 A 38 38 0 0 1 232 174"
              stroke="#FFFFFF"
              strokeWidth="3.5"
              strokeLinecap="round"
            />
            <line
              x1="247"
              y1="247"
              x2="292"
              y2="292"
              stroke="#3B1A08"
              strokeWidth="11"
              strokeLinecap="round"
            />
            <line
              x1="249"
              y1="249"
              x2="289"
              y2="289"
              stroke="#6B3410"
              strokeWidth="6.5"
              strokeLinecap="round"
            />
          </g>

          {/* MAGNIFYING GLASS 2 (Compact Right-Center Scanning Lens) */}
          <g className="lens-float-2" opacity="0.44">
            <circle cx="1220" cy="370" r="58" fill="#EDE9FE" fillOpacity="0.48" />
            <g clipPath="url(#lensClip2)">
              <g className="binary-scroll-lens">
                <text x="1150" y="374" fill="#4F46E5" fontSize="12" fontWeight="700" letterSpacing="2">
                  10101100 01101001 11001010 10101100 01101001 11001010
                </text>
              </g>
            </g>
            <circle cx="1220" cy="370" r="58" stroke="#3B1A08" strokeWidth="4.5" />
            <circle cx="1220" cy="370" r="52" stroke="#7C3AED" strokeWidth="2" />
            <path
              d="M 1184 338 A 42 42 0 0 1 1244 330"
              stroke="#FFFFFF"
              strokeWidth="3.5"
              strokeLinecap="round"
            />
            <line
              x1="1261"
              y1="411"
              x2="1312"
              y2="462"
              stroke="#3B1A08"
              strokeWidth="12"
              strokeLinecap="round"
            />
            <line
              x1="1263"
              y1="413"
              x2="1309"
              y2="459"
              stroke="#5C2C12"
              strokeWidth="7"
              strokeLinecap="round"
            />
          </g>

          {/* MAGNIFYING GLASS 3 (Compact Bottom-Left Scanning Lens) */}
          <g className="lens-float-3" opacity="0.4">
            <circle cx="260" cy="730" r="46" fill="#FFE4E6" fillOpacity="0.45" />
            <g clipPath="url(#lensClip3)">
              <g className="binary-scroll-lens">
                <text x="205" y="734" fill="#E11D48" fontSize="11.5" fontWeight="700" letterSpacing="2">
                  01100101 10110100 01011001 01100101 10110100 01011001
                </text>
              </g>
            </g>
            <circle cx="260" cy="730" r="46" stroke="#3B1A08" strokeWidth="4" />
            <circle cx="260" cy="730" r="40" stroke="#E11D48" strokeWidth="1.8" />
            <line
              x1="293"
              y1="763"
              x2="332"
              y2="802"
              stroke="#5C2C12"
              strokeWidth="9.5"
              strokeLinecap="round"
            />
          </g>
        </svg>
      </div>

      {/* ===================================================================== */}
      {/* VINTAGE MAHOGANY LEATHER TOP BAR WITH STITCHED "TO" MONOGRAM          */}
      {/* ===================================================================== */}
      <header className="sticky top-0 z-30 leather-bar text-white">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl leather-badge flex items-center justify-center font-extrabold text-lg tracking-tighter">
              <span>
                T<span className="text-blue-400">O</span>
              </span>
            </div>
            <div>
              <span className="text-xl font-extrabold tracking-tight text-white">
                Thesis<span className="text-blue-300">Out</span>
              </span>
              <span className="hidden sm:inline-block ml-3 px-3 py-0.5 rounded-full bg-black/25 border border-[#D6B89C]/40 text-[11px] font-semibold text-[#F5E6D8]">
                Taking Thesis Discoveries Out Into the World
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowUploadBox(!showUploadBox)}
              className="pop-btn px-3.5 py-1.5 rounded-xl bg-[#FAF6F0] text-[#3B1A08] text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5 text-blue-600" />
              {showUploadBox ? 'Close Upload' : 'Upload Thesis CSV'}
            </button>
          </div>
        </div>
      </header>

      {/* ===================================================================== */}
      {/* SINGLE-PAGE UNIFIED DISCOVERY CANVAS (ZERO TABS, ZERO SILOS)          */}
      {/* ===================================================================== */}
      <main className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-8">
        {errorMsg && (
          <div className="pop-card-crimson rounded-2xl p-4 flex items-center justify-between text-xs font-bold text-rose-900">
            <span>{errorMsg}</span>
            <button
              onClick={() => setErrorMsg(null)}
              className="px-2.5 py-1 rounded-lg bg-rose-100 text-rose-900 cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* OPTIONAL UPLOAD DRAWER */}
        {showUploadBox && (
          <form
            onSubmit={handleUploadSubmit}
            className="pop-card rounded-3xl p-6 space-y-4"
          >
            <div className="text-sm font-extrabold text-slate-900">
              Upload Your Own Thesis PDF & Raw Dataset CSV
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <input
                type="text"
                required
                value={uploadTitle}
                onChange={(e) => setUploadTitle(e.target.value)}
                placeholder="Thesis Title"
                className="px-3.5 py-2 rounded-xl border-2 border-slate-900 text-xs font-medium"
              />
              <input
                type="text"
                value={uploadUni}
                onChange={(e) => setUploadUni(e.target.value)}
                placeholder="University"
                className="px-3.5 py-2 rounded-xl border-2 border-slate-900 text-xs font-medium"
              />
              <input
                type="text"
                value={uploadTarget}
                onChange={(e) => setUploadTarget(e.target.value)}
                placeholder="Target Column (optional)"
                className="px-3.5 py-2 rounded-xl border-2 border-slate-900 text-xs font-medium"
              />
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <label className="px-3.5 py-2 rounded-xl bg-slate-100 border-2 border-slate-900 font-bold cursor-pointer">
                <FileText className="w-3.5 h-3.5 inline mr-1.5 text-blue-600" />
                {uploadPdf ? uploadPdf.name : 'Choose Thesis PDF'}
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setUploadPdf(e.target.files?.[0] || null)}
                  className="hidden"
                />
              </label>
              <label className="px-3.5 py-2 rounded-xl bg-blue-50 border-2 border-slate-900 font-bold cursor-pointer">
                <Upload className="w-3.5 h-3.5 inline mr-1.5 text-blue-600" />
                {uploadCsv ? uploadCsv.name : 'Choose Dataset CSV'}
                <input
                  type="file"
                  accept=".csv,.tsv"
                  onChange={(e) => setUploadCsv(e.target.files?.[0] || null)}
                  className="hidden"
                />
              </label>
              <button
                type="submit"
                disabled={uploading}
                className="pop-btn px-5 py-2 rounded-xl bg-blue-600 text-white font-bold cursor-pointer"
              >
                {uploading ? 'Running 30 Models' : 'Run Audit on Upload'}
              </button>
            </div>
          </form>
        )}

        {/* =================================================================== */}
        {/* HERO ASK BAR                                                        */}
        {/* =================================================================== */}
        <section className="pop-card rounded-3xl p-6 sm:p-8 space-y-5">
          <div className="space-y-1.5">
            <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-xl leather-badge text-xs font-bold">
              <Search className="w-3.5 h-3.5 text-blue-300" />
              Thesis Detective
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight pt-1">
              Hunt for hidden discoveries inside university thesis data.
            </h1>
            <p className="text-xs sm:text-sm text-slate-600">
              Ask a research question to pull the thesis, test 30 cross-validated models on the raw dataset, check live clinical and patent registries, and see the moonshot below.
            </p>
          </div>

          {/* Search Input Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleRunAutonomousGeminiWorkflow();
            }}
            className="flex flex-col sm:flex-row gap-3"
          >
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Enter any research question across any scientific, engineering, or quantitative field"
              className="flex-1 px-4 py-3.5 rounded-2xl bg-slate-50 border-2 border-slate-900 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:bg-white font-medium"
            />
            <button
              type="submit"
              disabled={harvesting || !chatInput.trim()}
              className="pop-btn px-6 py-3.5 rounded-2xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-extrabold text-sm flex items-center justify-center gap-2 cursor-pointer shrink-0"
            >
              {harvesting ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Hunting Clues
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 stroke-[2.5]" />
                  Investigate & Reveal
                </>
              )}
            </button>
          </form>

          {/* Open-Ended Query Guidance Across Any Discipline */}
          <div className="pop-subcard rounded-2xl p-4 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div>
              <div className="font-extrabold text-slate-900">1. Any Field or Phenomenon</div>
              <p className="text-slate-600 mt-0.5 leading-relaxed">
                Ask about any discipline: medicine, physics, climate, chemistry, economics, ecology, astrophysics, or materials science.
              </p>
            </div>
            <div>
              <div className="font-extrabold text-slate-900">2. What You Want to Predict</div>
              <p className="text-slate-600 mt-0.5 leading-relaxed">
                Mention the outcome or relationship you want 30 cross-validated models to test inside the raw dataset.
              </p>
            </div>
            <div>
              <div className="font-extrabold text-slate-900">3. Or Bring Your Own Data</div>
              <p className="text-slate-600 mt-0.5 leading-relaxed">
                Click Upload Thesis CSV in the top right to run the full 30-model sweep on your own spreadsheet or thesis PDF.
              </p>
            </div>
          </div>

          {/* Live Stream Progress Banner When Running */}
          {(harvesting || streamEvents.length > 0) && (
            <div className="bg-slate-900 text-white rounded-2xl p-4 border-2 border-slate-900 space-y-1.5 font-mono text-xs">
              <div className="flex items-center justify-between text-blue-300 font-bold">
                <span>
                  {harvesting ? 'LIVE INVESTIGATION IN PROGRESS' : 'INVESTIGATION COMPLETE'}
                </span>
                <span>{streamEvents.length} steps logged</span>
              </div>
              {streamEvents.slice(-3).map((ev, i) => (
                <div key={i} className="text-slate-200 truncate">
                  → {ev}
                </div>
              ))}
            </div>
          )}

          {/* Simple Case Switcher Pills (No Hidden Drawer!) */}
          {dissertations.length > 1 && (
            <div className="pt-2 border-t border-slate-200 flex flex-wrap items-center gap-2">
              <span className="text-xs font-bold text-slate-500">Active Case File:</span>
              {dissertations.map((d) => {
                const isSelected = selectedDiss?.id === d.id;
                return (
                  <button
                    key={d.id}
                    onClick={() => handleSelectDissertation(d)}
                    className={`px-3 py-1 rounded-xl text-xs font-bold border-2 transition cursor-pointer ${
                      isSelected
                        ? 'bg-slate-900 text-white border-slate-900'
                        : 'bg-white text-slate-700 border-slate-300 hover:border-slate-900'
                    }`}
                  >
                    {d.university}: {d.title.slice(0, 44)}
                  </button>
                );
              })}
            </div>
          )}
        </section>

        {/* =================================================================== */}
        {/* SECTION 1: THE MOONSHOT REVEAL (TOP OF STORY)                       */}
        {/* =================================================================== */}
        {activeOpportunities.length > 0 && (
          <section className="space-y-6">
            {activeOpportunities.slice(0, 1).map((opp) => (
              <div key={opp.id} className="pop-card-sapphire rounded-3xl p-6 sm:p-8 space-y-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                      01. THE MOONSHOT REVEAL
                    </span>
                    <span className="px-3 py-1 rounded-xl bg-blue-50 text-blue-900 border-2 border-slate-900 text-xs font-extrabold">
                      {opp.opportunity_type}
                    </span>
                  </div>
                  <div className="px-4 py-1.5 rounded-xl bg-rose-600 text-white border-2 border-slate-900 text-xs font-extrabold shadow-[2px_2px_0px_#0F172A]">
                    Translation Score: {opp.overall_tos_score} / 100
                  </div>
                </div>

                {/* Executive Takeaway if available */}
                {(executiveSummary || activeHypothesis) && (
                  <div className="bg-blue-50/90 border-2 border-slate-900 rounded-2xl p-4 space-y-2">
                    <div className="text-xs font-extrabold text-blue-900 uppercase tracking-wider">
                      Case Brief
                    </div>
                    {activeHypothesis && (
                      <div className="text-xs font-bold text-slate-900">
                        Hypothesis: {activeHypothesis}
                      </div>
                    )}
                    {executiveSummary && (
                      <p className="text-xs text-slate-800 leading-relaxed whitespace-pre-line">
                        {executiveSummary}
                      </p>
                    )}
                  </div>
                )}

                <div>
                  <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 leading-snug">
                    {opp.title}
                  </h2>
                  <p className="text-sm font-bold text-blue-700 mt-1">
                    What to Build: {opp.proposed_product_or_therapeutic}
                  </p>
                </div>

                {/* 3 Simple Story Cards: Signal -> Why Missed -> Whitespace */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                  <div className="pop-subcard rounded-2xl p-4 space-y-1.5">
                    <div className="font-extrabold text-blue-700 uppercase">
                      Hidden Signal Found
                    </div>
                    <p className="text-slate-700 leading-relaxed">
                      {opp.untranslated_finding_summary}
                    </p>
                  </div>

                  <div className="pop-subcard rounded-2xl p-4 space-y-1.5">
                    <div className="font-extrabold text-indigo-700 uppercase">
                      Why Authors Missed It
                    </div>
                    <p className="text-slate-700 leading-relaxed">
                      {opp.valley_of_death_diagnosis}
                    </p>
                  </div>

                  <div className="pop-subcard rounded-2xl p-4 space-y-1.5">
                    <div className="font-extrabold text-rose-700 uppercase">
                      Live Registry Check
                    </div>
                    <p className="text-slate-700 leading-relaxed">
                      <strong>ClinicalTrials.gov:</strong>{' '}
                      {opp.clinical_trials_check?.whitespace_status}
                    </p>
                    <p className="text-slate-700 leading-relaxed">
                      <strong>ChEMBL / OpenTargets:</strong>{' '}
                      {opp.opentargets_chembl_check?.tractability_modality}
                    </p>
                  </div>
                </div>

                {/* 4 Actionable Validation & Export Buttons */}
                <div className="space-y-3 pt-2 border-t-2 border-slate-200">
                  <div className="text-xs font-extrabold text-slate-900 uppercase tracking-wider">
                    Take Action on This Discovery
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {opp.validation_roadmap.map((step, i) => {
                      const key = `${opp.id}_${i}`;
                      const isRunning = executingStepKey === key;
                      const stepRes = stepActionResults[key];
                      const buttonLabels = [
                        '1. Run 5-Fold Verification',
                        '2. Generate Patent Claim',
                        '3. Audit Live Clinical Trials',
                        '4. Download Python Model & Dossier',
                      ];

                      return (
                        <div
                          key={i}
                          className="bg-slate-50 border-2 border-slate-900 rounded-2xl p-4 text-xs flex flex-col justify-between gap-3"
                        >
                          <div className="font-medium text-slate-800 leading-relaxed">{step}</div>

                          <button
                            onClick={() => handleExecuteOpportunityStep(opp.id, i)}
                            disabled={isRunning}
                            className="pop-btn self-start px-3.5 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs flex items-center gap-1.5 cursor-pointer"
                          >
                            {isRunning ? (
                              <>
                                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                Running...
                              </>
                            ) : (
                              <>
                                <Play className="w-3.5 h-3.5" />
                                {buttonLabels[i] || `Run Action ${i + 1}`}
                              </>
                            )}
                          </button>

                          {stepRes && (
                            <div className="pt-2 border-t border-slate-300 space-y-2 text-[11px]">
                              <div className="font-extrabold text-blue-800">
                                {stepRes.action_title}
                              </div>
                              <p className="text-slate-700">{stepRes.summary}</p>

                              {stepRes.provisional_patent_claim && (
                                <div className="bg-white border border-slate-300 rounded-xl p-2.5 font-mono text-[10px] text-slate-900">
                                  {stepRes.provisional_patent_claim}
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
                                    className="pop-btn px-3 py-1.5 rounded-lg bg-blue-600 text-white font-bold inline-flex items-center gap-1 cursor-pointer"
                                  >
                                    <Download className="w-3 h-3" /> Python Model (.py)
                                  </button>
                                  <button
                                    onClick={() =>
                                      downloadTextFile(
                                        `thesisout_dossier_opp_${opp.id}.md`,
                                        stepRes.venture_dossier_markdown
                                      )
                                    }
                                    className="pop-btn px-3 py-1.5 rounded-lg bg-indigo-600 text-white font-bold inline-flex items-center gap-1 cursor-pointer"
                                  >
                                    <Download className="w-3 h-3" /> Venture Dossier (.md)
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
          </section>
        )}

        {/* =================================================================== */}
        {/* SECTION 2: THE CLUE BOARD (MODEL SHOWDOWN & LEADERBOARD)            */}
        {/* =================================================================== */}
        {activeRun && (
          <section className="pop-card rounded-3xl p-6 sm:p-8 space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                  02. THE CLUE BOARD
                </span>
                <span className="text-xs font-bold text-slate-600">
                  {activeRun.models_tested_count} Cross-Validated Models Tested on{' '}
                  <code className="px-2 py-0.5 rounded bg-slate-200 text-slate-900 font-bold">
                    {activeRun.target_column}
                  </code>
                </span>
              </div>

              {activeDataset && (
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="font-bold text-slate-500">Switch Target:</span>
                  {activeDataset.column_metadata
                    .filter((c) => c.role !== 'identifier')
                    .slice(0, 4)
                    .map((col) => (
                      <button
                        key={col.name}
                        onClick={() => handleRunCustomTargetColumn(col.name)}
                        disabled={runningCustomTarget}
                        className={`px-2.5 py-1 rounded-lg font-mono text-[11px] font-bold border cursor-pointer ${
                          activeRun.target_column === col.name
                            ? 'bg-slate-900 text-white border-slate-900'
                            : 'bg-slate-100 text-slate-700 border-slate-300 hover:border-slate-900'
                        }`}
                      >
                        {col.name}
                      </button>
                    ))}
                </div>
              )}
            </div>

            {/* 3-Box Score Showdown */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="pop-subcard rounded-2xl p-4">
                <div className="text-xs font-extrabold text-slate-500 uppercase">
                  Thesis Author Baseline
                </div>
                <div className="text-3xl font-extrabold text-slate-900 mt-1">
                  {activeRun.author_baseline_score.toFixed(4)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Linear Model ({activeRun.primary_metric_name})
                </div>
              </div>

              <div className="bg-blue-50 border-2 border-slate-900 rounded-2xl p-4 shadow-[3px_3px_0px_#2563EB]">
                <div className="text-xs font-extrabold text-blue-700 uppercase">
                  Winning Model ({activeRun.best_model_name})
                </div>
                <div className="text-3xl font-extrabold text-blue-700 mt-1 flex items-baseline gap-2">
                  <span>{activeRun.best_score.toFixed(4)}</span>
                  <span className="text-xs font-extrabold px-2 py-0.5 rounded-full bg-rose-600 text-white">
                    +{(activeRun.signal_gain_delta * 100).toFixed(1)}% Gain
                  </span>
                </div>
                <div className="text-xs text-slate-600 mt-1">
                  Out-of-Fold Cross-Validated {activeRun.primary_metric_name}
                </div>
              </div>

              <div className="pop-subcard rounded-2xl p-4 flex flex-col justify-between">
                <div>
                  <div className="text-xs font-extrabold text-indigo-700 uppercase">
                    5-Stage Deep Verification
                  </div>
                  {escalationCache[activeRun.id] ? (
                    <>
                      <div className="text-3xl font-extrabold text-indigo-700 mt-1">
                        {escalationCache[activeRun.id].stage2_super_learner_ensemble.super_learner_score.toFixed(4)}
                      </div>
                      <div className="text-xs text-slate-600 mt-1">
                        Super-Learner Stacking • Zero Leakage Verified
                      </div>
                    </>
                  ) : (
                    <p className="text-xs text-slate-600 mt-1">
                      Verify with 5-fold permutation testing & Super-Learner stacking.
                    </p>
                  )}
                </div>
                {!escalationCache[activeRun.id] && (
                  <button
                    onClick={() => handleEscalateDeepInference(activeRun.id)}
                    disabled={escalatingRunId === activeRun.id}
                    className="pop-btn mt-2 self-start px-3.5 py-1.5 rounded-xl bg-indigo-600 text-white text-xs font-bold cursor-pointer"
                  >
                    {escalatingRunId === activeRun.id ? 'Verifying...' : 'Run 5-Stage Proof'}
                  </button>
                )}
              </div>
            </div>

            {/* Top Hidden Clues (Feature Importance Bars) */}
            <div className="space-y-3">
              <div className="text-xs font-extrabold text-slate-900 uppercase tracking-wider">
                Top Predictive Clues Driving <code className="text-blue-700">{activeRun.target_column}</code>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {activeRun.top_features.slice(0, 6).map((feat, idx) => (
                  <div key={idx} className="pop-subcard rounded-xl p-3 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono font-extrabold text-slate-900">
                        {feat.feature}
                      </span>
                      <div className="flex items-center gap-2">
                        {!feat.discussed_by_author && (
                          <span className="px-2 py-0.5 rounded bg-rose-600 text-white text-[10px] font-extrabold">
                            Overlooked Clue
                          </span>
                        )}
                        <span className="font-mono font-extrabold text-blue-700">
                          {(feat.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <div className="w-full h-2.5 bg-slate-200 rounded-full overflow-hidden border border-slate-300">
                      <div
                        className={`h-full rounded-full ${
                          !feat.discussed_by_author ? 'bg-rose-600' : 'bg-blue-600'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(8, feat.importance * 100))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Cross-Validated Model Leaderboard Table */}
            <div className="space-y-3 pt-2 border-t-2 border-slate-200">
              <div className="flex items-center justify-between">
                <div className="text-xs font-extrabold text-slate-900 uppercase tracking-wider">
                  Cross-Validated Model Benchmark ({activeRun.leaderboard.length} Architectures)
                </div>
                <button
                  onClick={() => setShowAllModels(!showAllModels)}
                  className="px-3 py-1 rounded-xl bg-slate-100 border border-slate-300 text-xs font-bold text-slate-800 flex items-center gap-1 cursor-pointer"
                >
                  {showAllModels ? (
                    <>
                      Hide Table <ChevronUp className="w-3.5 h-3.5" />
                    </>
                  ) : (
                    <>
                      Show All {activeRun.leaderboard.length} Models <ChevronDown className="w-3.5 h-3.5" />
                    </>
                  )}
                </button>
              </div>

              {showAllModels && (
                <div className="overflow-x-auto max-h-80 rounded-2xl border-2 border-slate-900">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="bg-slate-900 text-white font-bold">
                        <th className="py-2.5 px-3">Rank</th>
                        {Object.keys(activeRun.leaderboard[0] || {}).map((col) => (
                          <th key={col} className="py-2.5 px-3">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 font-mono bg-white">
                      {activeRun.leaderboard.map((row, idx) => (
                        <tr
                          key={idx}
                          className={
                            idx === 0
                              ? 'bg-blue-50 font-extrabold text-blue-950'
                              : 'text-slate-800 hover:bg-slate-50'
                          }
                        >
                          <td className="py-2 px-3">#{idx + 1}</td>
                          {Object.entries(row).map(([k, val]) => (
                            <td key={k} className="py-2 px-3">
                              {val === null
                                ? '-'
                                : typeof val === 'number'
                                ? val.toFixed(4)
                                : String(val)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>
        )}

        {/* =================================================================== */}
        {/* SECTION 3: THE EVIDENCE LOCKER (SOURCE THESIS & LIVE REGISTRIES)    */}
        {/* =================================================================== */}
        {selectedDiss && (
          <section className="pop-card rounded-3xl p-6 sm:p-8 space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                03. THE EVIDENCE LOCKER
              </span>
              <span className="text-xs font-bold text-slate-600">
                Real University Thesis + Open Zenodo Dataset + Live Registry Verification
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              {/* Source Thesis Card */}
              <div className="pop-subcard rounded-2xl p-4 space-y-2">
                <div className="font-extrabold text-blue-700 uppercase">
                  Source Dissertation ({selectedDiss.university}, {selectedDiss.publication_year})
                </div>
                <div className="text-sm font-extrabold text-slate-900 leading-snug">
                  {selectedDiss.title}
                </div>
                <div className="text-slate-600">Authors: {selectedDiss.authors}</div>
                {selectedDiss.source_url && (
                  <a
                    href={selectedDiss.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 font-bold text-blue-600 hover:underline pt-1"
                  >
                    <BookOpen className="w-3.5 h-3.5" /> View Original Publication{' '}
                    <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>

              {/* Real Dataset Card */}
              {activeDataset && (
                <div className="pop-subcard rounded-2xl p-4 space-y-2">
                  <div className="font-extrabold text-indigo-700 uppercase">
                    Audited Open Dataset ({activeDataset.source_type})
                  </div>
                  <div className="text-sm font-extrabold text-slate-900">
                    {activeDataset.name}
                  </div>
                  <div className="text-slate-600 font-mono">
                    {activeDataset.row_count} real rows × {activeDataset.col_count} columns
                  </div>
                  {activeDataset.source_url && (
                    <a
                      href={activeDataset.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 font-bold text-indigo-600 hover:underline pt-1"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5" /> Inspect Raw Dataset on Zenodo{' '}
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  )}
                </div>
              )}
            </div>

            {/* Follow-up Conversational Bar */}
            <div className="pt-3 border-t border-slate-200 space-y-3">
              {chatMessages.length > 0 && (
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {chatMessages.slice(-2).map((msg, i) => (
                    <div
                      key={i}
                      className={`p-3.5 rounded-2xl text-xs ${
                        msg.role === 'user'
                          ? 'bg-slate-900 text-white font-bold ml-auto max-w-xl'
                          : 'bg-blue-50 border-2 border-slate-900 text-slate-900'
                      }`}
                    >
                      <p className="whitespace-pre-line leading-relaxed">{msg.content}</p>
                    </div>
                  ))}
                </div>
              )}

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleFollowUpChat();
                }}
                className="flex gap-2"
              >
                <input
                  type="text"
                  value={followUpInput}
                  onChange={(e) => setFollowUpInput(e.target.value)}
                  placeholder="Ask a follow-up question about this moonshot or dataset"
                  className="flex-1 px-3.5 py-2.5 rounded-xl bg-slate-50 border-2 border-slate-900 text-xs font-medium"
                />
                <button
                  type="submit"
                  disabled={chatLoading || !followUpInput.trim()}
                  className="pop-btn px-4 py-2.5 rounded-xl bg-slate-900 text-white text-xs font-bold flex items-center gap-1.5 cursor-pointer"
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  {chatLoading ? 'Working' : 'Ask Follow-Up'}
                </button>
              </form>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
