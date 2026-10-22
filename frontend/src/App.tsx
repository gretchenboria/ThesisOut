import React, { useEffect, useState } from 'react';
import {
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  ExternalLink,
  FileText,
  Globe,
  Layers,
  Lock,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  Sparkles,
  Upload,
  X,
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

const DOMAIN_LABELS: Record<string, string> = {
  all: 'All Global Findings',
  cs_ai: 'AI & Hardware Systems',
  biomedical: 'Biomedical & Clinical',
  energy_materials: 'Energy & Solid-State',
  climate_earth: 'Climate & Ocean Physics',
};

const GLOBAL_PULL_TOPICS = [
  { label: 'Pull Global: AI Memory & Silicon Telemetry', query: 'HBM3e memory latency thermal telemetry' },
  { label: 'Pull Global: Solid Tumor Immunotherapy', query: 'pancreatic tumor microenvironment lactate CD8' },
  { label: 'Pull Global: Solid-State Battery Dendrites', query: 'solid state battery impedance dendrite cycling' },
  { label: 'Pull Global: Sleep & Metabolic Biomarkers', query: 'sleep restriction metabolic biomarker cohort' },
];

export function App() {
  const [accessToken, setAccessToken] = useState<string>(
    () => sessionStorage.getItem('thesisout_access_token') || ''
  );
  const [isUnlocked, setIsUnlocked] = useState<boolean>(
    () => sessionStorage.getItem('thesisout_access_token') === 'thesisout!'
  );
  const [gateInput, setGateInput] = useState<string>('');
  const [botHoneypot, setBotHoneypot] = useState<string>('');
  const [gateError, setGateError] = useState<string>('');

  const [showAllModels, setShowAllModels] = useState<boolean>(true);
  const [showUploadBox, setShowUploadBox] = useState<boolean>(false);
  const [showArchDiagram, setShowArchDiagram] = useState<boolean>(false);

  const [dissertations, setDissertations] = useState<Dissertation[]>([]);
  const [opportunities, setOpportunities] = useState<MarketOpportunity[]>([]);
  const [selectedDissId, setSelectedDissId] = useState<number | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [domainFilter, setDomainFilter] = useState<string>('all');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [escalationCache, setEscalationCache] = useState<Record<number, DeepInferenceResult>>({});
  const [escalatingRunId, setEscalatingRunId] = useState<number | null>(null);
  const [stepActionResults, setStepActionResults] = useState<Record<string, any>>({});
  const [executingStepKey, setExecutingStepKey] = useState<string | null>(null);

  const [executiveSummary, setExecutiveSummary] = useState<string>('');
  const [runningCustomTarget, setRunningCustomTarget] = useState<boolean>(false);

  const [harvesting, setHarvesting] = useState<boolean>(false);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);

  const [searchInput, setSearchInput] = useState<string>('');

  const [uploadPdf, setUploadPdf] = useState<File | null>(null);
  const [uploadCsv, setUploadCsv] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);

  const authHeaders = (extra?: Record<string, string>) => ({
    'X-Access-Code': accessToken || 'thesisout!',
    ...(extra || {}),
  });

  const handleUnlockSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (botHoneypot) return;
    setGateError('');
    try {
      const res = await fetch('/api/auth/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: gateInput.trim(), website_url: botHoneypot }),
      });
      if (res.ok) {
        sessionStorage.setItem('thesisout_access_token', gateInput.trim());
        setAccessToken(gateInput.trim());
        setIsUnlocked(true);
      } else {
        setGateError('Invalid access code.');
      }
    } catch {
      if (gateInput.trim() === 'thesisout!') {
        sessionStorage.setItem('thesisout_access_token', 'thesisout!');
        setAccessToken('thesisout!');
        setIsUnlocked(true);
      } else {
        setGateError('Invalid access code.');
      }
    }
  };

  const handleLockWorkspace = () => {
    sessionStorage.removeItem('thesisout_access_token');
    setAccessToken('');
    setIsUnlocked(false);
    setGateInput('');
  };

  const fetchAllData = async () => {
    if (!isUnlocked) return;
    setErrorMsg(null);
    try {
      const [dRes, oRes] = await Promise.all([
        fetch('/api/dissertations', { headers: authHeaders() }),
        fetch('/api/opportunities', { headers: authHeaders() }),
      ]);
      if (dRes.status === 401 || oRes.status === 401) {
        setIsUnlocked(false);
        return;
      }
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
    if (isUnlocked) {
      fetchAllData();
    }
  }, [isUnlocked]);

  const filteredDissertations = dissertations.filter((d) => {
    if (domainFilter !== 'all' && d.domain !== domainFilter) return false;
    return true;
  });

  const visibleDissertations =
    filteredDissertations.length > 0 ? filteredDissertations : dissertations;

  const selectedDiss =
    visibleDissertations.find((d) => d.id === selectedDissId) ||
    dissertations.find((d) => d.id === selectedDissId) ||
    visibleDissertations[0] ||
    dissertations[0];

  const activeDataset = selectedDiss?.datasets?.[0];
  const activeRun =
    activeDataset?.lazy_runs?.find((r) => r.id === selectedRunId) ||
    activeDataset?.lazy_runs?.[0];

  const activeOpportunities = opportunities.filter(
    (o) => !selectedDiss || o.dissertation_id === selectedDiss.id
  );

  const handleSelectDissertation = (diss: Dissertation, scrollToDetail = false) => {
    setSelectedDissId(diss.id);
    const firstRun = diss.datasets?.[0]?.lazy_runs?.[0];
    if (firstRun) {
      setSelectedRunId(firstRun.id);
    }
    if (scrollToDetail) {
      const el = document.getElementById('active-finding-inspector');
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleResetToGlobalView = () => {
    setDomainFilter('all');
    setSearchInput('');
    setStreamEvents([]);
    setExecutiveSummary('');
    setShowUploadBox(false);
    setErrorMsg(null);
    if (dissertations.length > 0) {
      setSelectedDissId(dissertations[0].id);
      const firstRun = dissertations[0].datasets?.[0]?.lazy_runs?.[0];
      if (firstRun) setSelectedRunId(firstRun.id);
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleEscalateDeepInference = async (lazyRunId: number) => {
    if (escalationCache[lazyRunId]) return;
    setEscalatingRunId(lazyRunId);
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/lazypredict/${lazyRunId}/escalate`, {
        method: 'POST',
        headers: authHeaders(),
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
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dataset_id: activeDataset.id,
          target_column: colName,
          is_author_primary_target: false,
          target_rationale: `Global sweep on column ${colName}`,
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

  const handleRunGlobalSearch = (queryOverride?: string) => {
    const queryText = (queryOverride ?? searchInput).trim();
    if (!queryText || harvesting) return;

    if (!queryOverride) setSearchInput('');
    setHarvesting(true);
    setErrorMsg(null);
    setDomainFilter('all');
    setStreamEvents([`Global repository search initiated for "${queryText}"`]);

    const url = `/api/copilot/autonomous-stream?prompt=${encodeURIComponent(queryText)}&access_code=${encodeURIComponent(accessToken || 'thesisout!')}`;
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
          if (payload.executed_steps) {
            setStepActionResults((prev) => ({ ...prev, ...payload.executed_steps }));
          }
          if (payload.executive_summary) {
            setExecutiveSummary(payload.executive_summary);
          }
        }
      } catch (err) {
        console.error('Global stream parse error:', err);
      }
    };

    evtSource.onerror = async () => {
      evtSource.close();
      await fetchAllData();
      setHarvesting(false);
    };
  };

  const handleExecuteOpportunityStep = async (oppId: number, stepIdx: number) => {
    const key = `${oppId}_${stepIdx}`;
    setExecutingStepKey(key);
    try {
      const res = await fetch(`/api/opportunities/${oppId}/execute-step?step_index=${stepIdx}`, {
        method: 'POST',
        headers: authHeaders(),
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
    if (!uploadPdf && !uploadCsv) return;
    setUploading(true);
    setErrorMsg(null);
    try {
      const fd = new FormData();
      if (uploadPdf) fd.append('pdf_file', uploadPdf);
      if (uploadCsv) fd.append('dataset_file', uploadCsv);

      const res = await fetch('/api/dissertations/upload', {
        method: 'POST',
        headers: authHeaders(),
        body: fd,
      });
      if (res.ok) {
        const data = await res.json();
        await fetchAllData();
        if (data.id) setSelectedDissId(data.id);
        setShowUploadBox(false);
        setUploadPdf(null);
        setUploadCsv(null);
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
      {/* CALM MAGNIFYING GLASS, BINARY STREAM, TRAJECTORY & EPISODIC GEMS SVG  */}
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

          <g className="gem-episodic-1">
            <polygon
              points="1080,155 1098,170 1091,194 1069,194 1062,170"
              fill="#DBEAFE"
              stroke="#2563EB"
              strokeWidth="2.2"
            />
            <polygon
              points="1080,161 1092,172 1080,189 1068,172"
              fill="#3B82F6"
              fillOpacity="0.65"
            />
            <line x1="1062" y1="170" x2="1098" y2="170" stroke="#1D4ED8" strokeWidth="1.4" />
            <line x1="1080" y1="155" x2="1080" y2="194" stroke="#1D4ED8" strokeWidth="1.2" />
            <path
              d="M 1106 150 L 1108 156 L 1114 158 L 1108 160 L 1106 166 L 1104 160 L 1098 158 L 1104 156 Z"
              fill="#2563EB"
            />
          </g>

          <g className="gem-episodic-2">
            <polygon
              points="1130,680 1146,694 1140,716 1120,716 1114,694"
              fill="#FFE4E6"
              stroke="#E11D48"
              strokeWidth="2.2"
            />
            <polygon
              points="1130,685 1141,695 1130,711 1119,695"
              fill="#E11D48"
              fillOpacity="0.58"
            />
            <line x1="1114" y1="694" x2="1146" y2="694" stroke="#BE123C" strokeWidth="1.4" />
            <line x1="1130" y1="680" x2="1130" y2="716" stroke="#BE123C" strokeWidth="1.2" />
            <path
              d="M 1153 675 L 1155 680 L 1160 682 L 1155 684 L 1153 689 L 1151 684 L 1146 682 L 1151 680 Z"
              fill="#4F46E5"
            />
          </g>
        </svg>
      </div>

      {/* ===================================================================== */}
      {/* 3D PEBBLED MAHOGANY LEATHER TOP BAR WITH CLEAR GLOBAL NAVIGATION      */}
      {/* ===================================================================== */}
      <header className="sticky top-0 z-30 leather-bar text-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <button
            type="button"
            onClick={handleResetToGlobalView}
            className="flex items-center gap-3 text-left cursor-pointer group"
            title="Return to All Global Findings"
          >
            <div className="w-11 h-11 rounded-xl leather-badge flex items-center justify-center font-extrabold text-lg tracking-tighter">
              <span>
                T<span className="text-blue-400">O</span>
              </span>
            </div>
            <div>
              <span className="text-xl font-extrabold tracking-tight text-white group-hover:text-blue-200 transition">
                Thesis<span className="text-blue-300">Out</span>
              </span>
              <span className="hidden md:inline-block ml-3 px-3 py-0.5 rounded-full bg-black/25 border border-[#D6B89C]/40 text-[11px] font-semibold text-[#F5E6D8]">
                Global University Thesis &amp; High-Value Opportunity Engine
              </span>
            </div>
          </button>

          {isUnlocked && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleResetToGlobalView}
                className="pop-btn px-3.5 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
              >
                <Globe className="w-3.5 h-3.5" />
                All Global Findings ({dissertations.length})
              </button>

              <button
                type="button"
                onClick={() => setShowUploadBox(!showUploadBox)}
                className="pop-btn px-3.5 py-1.5 rounded-xl bg-[#FAF6F0] text-[#3B1A08] text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5 text-blue-600" />
                {showUploadBox ? 'Close Upload' : 'Upload Thesis / CSV'}
              </button>

              <button
                type="button"
                onClick={handleLockWorkspace}
                className="px-2.5 py-1.5 rounded-xl bg-black/30 hover:bg-black/45 border border-white/20 text-xs font-bold text-[#F5E6D8] flex items-center gap-1 cursor-pointer"
                title="Lock Workspace"
              >
                <Lock className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Lock</span>
              </button>
            </div>
          )}
        </div>
      </header>

      {/* ===================================================================== */}
      {/* ACCESS CODE GATE & BOT SHIELD ("thesisout!")                          */}
      {/* ===================================================================== */}
      {!isUnlocked ? (
        <main className="relative z-10 max-w-md mx-auto px-4 pt-20 pb-12">
          <form
            onSubmit={handleUnlockSubmit}
            className="pop-card rounded-3xl p-7 space-y-5"
          >
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl leather-badge flex items-center justify-center font-extrabold text-lg">
                <Lock className="w-5 h-5 text-blue-300" />
              </div>
              <div>
                <h1 className="text-lg font-extrabold text-slate-900">
                  Private Research Workspace
                </h1>
                <p className="text-xs text-slate-600">
                  Enter the access code to unlock ThesisOut.
                </p>
              </div>
            </div>

            <input
              type="text"
              name="website_url"
              tabIndex={-1}
              autoComplete="off"
              value={botHoneypot}
              onChange={(e) => setBotHoneypot(e.target.value)}
              className="hidden"
              aria-hidden="true"
            />

            <div className="space-y-2">
              <input
                type="password"
                value={gateInput}
                onChange={(e) => setGateInput(e.target.value)}
                placeholder="Access code"
                autoFocus
                className="w-full px-4 py-3 rounded-2xl bg-slate-50 border-2 border-slate-900 text-sm font-bold text-slate-900 focus:outline-none focus:bg-white"
              />
              {gateError && (
                <div className="text-xs font-bold text-rose-600">{gateError}</div>
              )}
            </div>

            <button
              type="submit"
              className="pop-btn w-full py-3 rounded-2xl bg-blue-600 hover:bg-blue-700 text-white font-extrabold text-sm cursor-pointer"
            >
              Unlock ThesisOut
            </button>
          </form>
        </main>
      ) : (
        <main className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 py-7 space-y-8">
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

          {/* ZERO-TEXT FILE UPLOAD DRAWER (AUTO-PARSES METADATA & COLUMNS) */}
          {showUploadBox && (
            <form
              onSubmit={handleUploadSubmit}
              className="pop-card rounded-3xl p-6 space-y-4"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-extrabold text-slate-900">
                    Drop Your Thesis PDF or Raw Dataset CSV
                  </div>
                  <p className="text-xs text-slate-600 mt-0.5">
                    Title, authors, university, publication year, and target variables are parsed automatically from your file.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setShowUploadBox(false)}
                  className="p-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <label className="px-4 py-2.5 rounded-xl bg-slate-100 border-2 border-slate-900 font-bold cursor-pointer">
                  <FileText className="w-3.5 h-3.5 inline mr-1.5 text-blue-600" />
                  {uploadPdf ? uploadPdf.name : 'Choose Thesis PDF'}
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={(e) => setUploadPdf(e.target.files?.[0] || null)}
                    className="hidden"
                  />
                </label>
                <label className="px-4 py-2.5 rounded-xl bg-blue-50 border-2 border-slate-900 font-bold cursor-pointer">
                  <Upload className="w-3.5 h-3.5 inline mr-1.5 text-blue-600" />
                  {uploadCsv ? uploadCsv.name : 'Choose Dataset CSV / TSV'}
                  <input
                    type="file"
                    accept=".csv,.tsv"
                    onChange={(e) => setUploadCsv(e.target.files?.[0] || null)}
                    className="hidden"
                  />
                </label>
                <button
                  type="submit"
                  disabled={uploading || (!uploadPdf && !uploadCsv)}
                  className="pop-btn px-5 py-2.5 rounded-xl bg-blue-600 disabled:opacity-50 text-white font-bold cursor-pointer"
                >
                  {uploading ? 'Parsing & Running 30 Models' : 'Parse & Audit File'}
                </button>
              </div>
            </form>
          )}

          {/* =================================================================== */}
          {/* GLOBAL SEARCH & ONE-CLICK REPOSITORY SWEEP (NO HYPOTHESIS NEEDED)   */}
          {/* =================================================================== */}
          <section className="pop-card rounded-3xl p-6 sm:p-7 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="space-y-1">
                <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-xl leather-badge text-xs font-bold">
                  <Globe className="w-3.5 h-3.5 text-blue-300" />
                  Global University Thesis &amp; Dataset Search
                </div>
                <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight pt-1">
                  Global search across university thesis data. The highest-value findings are pulled automatically.
                </h1>
                <p className="text-xs sm:text-sm text-slate-600">
                  No need to formulate a hypothesis. Browse the top-ranked discoveries pulled from MIT, Stanford, ETH Zurich, and Oxford below, or trigger a global repository sweep on any topic.
                </p>
              </div>
            </div>

            {/* Global Search Bar + Reset */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleRunGlobalSearch();
              }}
              className="flex flex-col sm:flex-row gap-2.5"
            >
              <input
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="Global search by keyword, phenomenon, university, or dataset topic (optional)"
                className="flex-1 px-4 py-3 rounded-2xl bg-slate-50 border-2 border-slate-900 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:bg-white font-medium"
              />
              <button
                type="submit"
                disabled={harvesting || !searchInput.trim()}
                className="pop-btn px-5 py-3 rounded-2xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-extrabold text-sm flex items-center justify-center gap-2 cursor-pointer shrink-0"
              >
                {harvesting ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Pulling Global Findings
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4 stroke-[2.5]" />
                    Pull Top Global Findings
                  </>
                )}
              </button>
              {(domainFilter !== 'all' || streamEvents.length > 0 || searchInput) && (
                <button
                  type="button"
                  onClick={handleResetToGlobalView}
                  className="pop-btn px-4 py-3 rounded-2xl bg-slate-100 hover:bg-slate-200 text-slate-900 font-extrabold text-xs flex items-center justify-center gap-1.5 cursor-pointer shrink-0"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Show All Findings
                </button>
              )}
            </form>

            {/* One-Click Global Domain Filters & Instant Live Repository Sweeps */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] font-extrabold text-slate-500 uppercase mr-1">
                  Filter Pulled Catalog:
                </span>
                {Object.entries(DOMAIN_LABELS).map(([key, label]) => {
                  const active = domainFilter === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setDomainFilter(key)}
                      className={`px-3 py-1 rounded-xl text-xs font-extrabold border-2 transition cursor-pointer ${
                        active
                          ? 'bg-slate-900 text-white border-slate-900'
                          : 'bg-white text-slate-700 border-slate-300 hover:border-slate-900'
                      }`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] font-extrabold text-blue-700 uppercase mr-1">
                  One-Click Live Sweep:
                </span>
                {GLOBAL_PULL_TOPICS.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    disabled={harvesting}
                    onClick={() => handleRunGlobalSearch(item.query)}
                    className="px-2.5 py-1 rounded-xl bg-blue-50 hover:bg-blue-100 text-blue-900 border border-blue-300 text-[11px] font-bold cursor-pointer transition"
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Live Stream Progress Banner with Clear Dismiss Button */}
            {(harvesting || streamEvents.length > 0) && (
              <div className="bg-slate-900 text-white rounded-2xl p-4 border-2 border-slate-900 space-y-1.5 font-mono text-xs">
                <div className="flex items-center justify-between text-blue-300 font-bold">
                  <span>
                    {harvesting
                      ? 'GLOBAL REPOSITORY SWEEP IN PROGRESS'
                      : 'GLOBAL FINDINGS PULLED & RANKED BELOW'}
                  </span>
                  <div className="flex items-center gap-3">
                    <span>{streamEvents.length} events</span>
                    {!harvesting && (
                      <button
                        type="button"
                        onClick={() => setStreamEvents([])}
                        className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-sans text-[11px] cursor-pointer"
                      >
                        Dismiss Log
                      </button>
                    )}
                  </div>
                </div>
                {streamEvents.slice(-3).map((ev, i) => (
                  <div key={i} className="text-slate-200 truncate">
                    → {ev}
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* =================================================================== */}
          {/* SECTION 01: REVEALED DISSERTATIONS & HIGH-VALUE OPPORTUNITIES       */}
          {/* =================================================================== */}
          <section className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 px-1">
              <div className="flex items-center gap-2.5">
                <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                  01. REVEALED DISSERTATIONS &amp; HIGH-VALUE OPPORTUNITIES
                </span>
                <span className="text-xs font-extrabold text-slate-700">
                  Ranked by 30-Model Empirical Signal Gain &amp; Translation Score ({visibleDissertations.length} Pulled)
                </span>
              </div>
              <span className="text-xs font-bold text-blue-700">
                Click any card to inspect its 30-model leaderboard &amp; venture blueprint below
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {visibleDissertations.map((diss) => {
                const isSelected = selectedDiss?.id === diss.id;
                const ds = diss.datasets?.[0];
                const run = ds?.lazy_runs?.[0];
                const opp =
                  opportunities.find((o) => o.dissertation_id === diss.id) ||
                  diss.opportunities?.[0];
                const topOverlooked = run?.top_features?.find((f) => !f.discussed_by_author) || run?.top_features?.[0];

                return (
                  <div
                    key={diss.id}
                    onClick={() => handleSelectDissertation(diss, true)}
                    className={`rounded-3xl p-5 sm:p-6 transition cursor-pointer flex flex-col justify-between gap-4 ${
                      isSelected
                        ? 'pop-card-sapphire ring-4 ring-blue-600/25'
                        : 'pop-card hover:translate-y-[-2px]'
                    }`}
                  >
                    <div className="space-y-3.5">
                      {/* Card Header Badges: University + TOS Score + Signal Gain */}
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="px-3 py-1 rounded-xl bg-slate-900 text-white text-xs font-extrabold">
                          {diss.university} • {diss.publication_year}
                        </span>
                        <div className="flex items-center gap-1.5">
                          {run && (
                            <span className="px-2.5 py-1 rounded-xl bg-blue-600 text-white text-[11px] font-extrabold">
                              +{(run.signal_gain_delta * 100).toFixed(1)}% Model Gain
                            </span>
                          )}
                          {opp && (
                            <span className="px-2.5 py-1 rounded-xl bg-rose-600 text-white text-[11px] font-extrabold">
                              TOS {opp.overall_tos_score} / 100
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Block A: Revealed University Dissertation */}
                      <div className="bg-slate-50 border-2 border-slate-900 rounded-2xl p-3.5 space-y-1.5">
                        <div className="flex items-center justify-between text-[11px] font-extrabold uppercase tracking-wider text-slate-600">
                          <span>Revealed Dissertation ({diss.subdomain || diss.domain})</span>
                          {run && (
                            <span className="font-mono text-blue-700">
                              {run.best_model_name}: {run.best_score.toFixed(4)} vs {run.author_baseline_score.toFixed(4)}
                            </span>
                          )}
                        </div>
                        <h3 className="text-sm sm:text-base font-extrabold text-slate-900 leading-snug">
                          {diss.title}
                        </h3>
                        <div className="text-xs text-slate-600 font-medium">
                          Authors: {diss.authors}
                        </div>
                        {topOverlooked && (
                          <div className="pt-1 text-xs text-rose-700 font-bold">
                            Overlooked Clue Pulled: <code className="px-1.5 py-0.5 rounded bg-rose-100 text-rose-900">{topOverlooked.feature}</code> ({(topOverlooked.importance * 100).toFixed(1)}% importance)
                          </div>
                        )}
                      </div>

                      {/* Block B: High-Value Commercial Opportunity */}
                      {opp && (
                        <div className="bg-blue-50/90 border-2 border-slate-900 rounded-2xl p-3.5 space-y-1.5">
                          <div className="flex items-center justify-between text-[11px] font-extrabold uppercase tracking-wider text-blue-800">
                            <span className="flex items-center gap-1">
                              <Sparkles className="w-3.5 h-3.5 text-rose-600" />
                              High-Value Opportunity • {opp.opportunity_type}
                            </span>
                          </div>
                          <div className="text-sm font-extrabold text-slate-900 leading-snug">
                            {opp.title}
                          </div>
                          <p className="text-xs text-slate-700 leading-relaxed">
                            <span className="font-extrabold text-blue-900">What to Build:</span>{' '}
                            {opp.proposed_product_or_therapeutic}
                          </p>
                          {opp.opentargets_chembl_check?.tractability_modality && (
                            <div className="text-[11px] font-bold text-indigo-800 pt-0.5">
                              Market &amp; Modality: {opp.opentargets_chembl_check.tractability_modality}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Card Footer */}
                    <div className="flex items-center justify-between pt-2 border-t border-slate-200 text-xs">
                      <span
                        className={`font-extrabold px-3 py-1 rounded-xl ${
                          isSelected
                            ? 'bg-blue-600 text-white'
                            : 'bg-slate-100 text-slate-800'
                        }`}
                      >
                        {isSelected
                          ? 'Active in Deep Inspector Below'
                          : 'Inspect Full 30-Model Proof & Blueprint'}
                      </span>

                      <div className="flex items-center gap-3" onClick={(e) => e.stopPropagation()}>
                        {diss.source_url && (
                          <a
                            href={diss.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="font-bold text-blue-700 hover:underline inline-flex items-center gap-0.5"
                          >
                            Thesis <ArrowUpRight className="w-3.5 h-3.5" />
                          </a>
                        )}
                        {ds?.source_url && (
                          <a
                            href={ds.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="font-bold text-indigo-700 hover:underline inline-flex items-center gap-0.5"
                          >
                            Dataset <ArrowUpRight className="w-3.5 h-3.5" />
                          </a>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* =================================================================== */}
          {/* SECTION 02: ACTIVE HIGH-VALUE OPPORTUNITY DOSSIER & ACTIONS         */}
          {/* =================================================================== */}
          <div id="active-finding-inspector" className="scroll-mt-20 space-y-8">
            {activeOpportunities.length > 0 && (
              <section className="space-y-6">
                {activeOpportunities.slice(0, 1).map((opp) => (
                  <div key={opp.id} className="pop-card-sapphire rounded-3xl p-6 sm:p-8 space-y-6">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                          02. HIGH-VALUE OPPORTUNITY DOSSIER
                        </span>
                        <span className="px-3 py-1 rounded-xl bg-slate-900 text-white text-xs font-extrabold">
                          {selectedDiss?.university} ({selectedDiss?.publication_year})
                        </span>
                        <span className="px-3 py-1 rounded-xl bg-blue-50 text-blue-900 border-2 border-slate-900 text-xs font-extrabold">
                          {opp.opportunity_type}
                        </span>
                      </div>
                      <div className="px-4 py-1.5 rounded-xl bg-rose-600 text-white border-2 border-slate-900 text-xs font-extrabold shadow-[2px_2px_0px_#0F172A]">
                        Translation Score: {opp.overall_tos_score} / 100
                      </div>
                    </div>

                    {executiveSummary && (
                      <div className="bg-blue-50/90 border-2 border-slate-900 rounded-2xl p-4 space-y-1.5">
                        <div className="text-xs font-extrabold text-blue-900 uppercase tracking-wider">
                          Global Pull Summary
                        </div>
                        <p className="text-xs text-slate-800 leading-relaxed whitespace-pre-line">
                          {executiveSummary}
                        </p>
                      </div>
                    )}

                    <div>
                      <div className="text-xs font-extrabold uppercase tracking-wider text-slate-500">
                        Selected Dissertation: {selectedDiss?.title}
                      </div>
                      <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 leading-snug mt-1">
                        {opp.title}
                      </h2>
                      <p className="text-sm font-bold text-blue-700 mt-1">
                        What to Build: {opp.proposed_product_or_therapeutic}
                      </p>
                    </div>

                    {/* 3 Story Cards: Overlooked Signal -> Why Missed -> Live Registry Whitespace */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                      <div className="pop-subcard rounded-2xl p-4 space-y-1.5">
                        <div className="font-extrabold text-blue-700 uppercase">
                          Overlooked Empirical Finding
                        </div>
                        <p className="text-slate-700 leading-relaxed">
                          {opp.untranslated_finding_summary}
                        </p>
                      </div>

                      <div className="pop-subcard rounded-2xl p-4 space-y-1.5">
                        <div className="font-extrabold text-indigo-700 uppercase">
                          Why the Thesis Missed It
                        </div>
                        <p className="text-slate-700 leading-relaxed">
                          {opp.valley_of_death_diagnosis}
                        </p>
                      </div>

                      <div className="pop-subcard rounded-2xl p-4 space-y-1.5">
                        <div className="font-extrabold text-rose-700 uppercase">
                          Patent &amp; Registry Whitespace
                        </div>
                        <p className="text-slate-700 leading-relaxed">
                          <strong>Registry Status:</strong>{' '}
                          {opp.clinical_trials_check?.whitespace_status}
                        </p>
                        <p className="text-slate-700 leading-relaxed">
                          <strong>Commercial Modality:</strong>{' '}
                          {opp.opentargets_chembl_check?.tractability_modality}
                        </p>
                      </div>
                    </div>

                    {/* 4 Actionable Validation & Export Buttons */}
                    <div className="space-y-3 pt-2 border-t-2 border-slate-200">
                      <div className="text-xs font-extrabold text-slate-900 uppercase tracking-wider">
                        Execute Verification &amp; Commercialization Actions
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {opp.validation_roadmap.map((step, i) => {
                          const key = `${opp.id}_${i}`;
                          const isRunning = executingStepKey === key;
                          const stepRes = stepActionResults[key];
                          const buttonLabels = [
                            '1. Run 5-Fold Verification',
                            '2. Generate Patent Claim',
                            '3. Audit Live Registries',
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
                                    Running
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
            {/* SECTION 03: THE CLUE BOARD (30-MODEL SHOWDOWN & LEADERBOARD)        */}
            {/* =================================================================== */}
            {activeRun && (
              <section className="pop-card rounded-3xl p-6 sm:p-8 space-y-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                      03. 30-MODEL EMPIRICAL CLUE BOARD
                    </span>
                    <span className="text-xs font-bold text-slate-600">
                      {activeRun.models_tested_count} Cross-Validated Models Audited on{' '}
                      <code className="px-2 py-0.5 rounded bg-slate-200 text-slate-900 font-bold">
                        {activeRun.target_column}
                      </code>
                    </span>
                  </div>

                  {activeDataset && (
                    <div className="flex flex-wrap items-center gap-1.5 text-xs">
                      <span className="font-bold text-slate-500">Audit Another Column:</span>
                      {activeDataset.column_metadata
                        .filter((c) => c.role !== 'identifier')
                        .slice(0, 5)
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
                      {selectedDiss?.author_Statistical_method || 'Classical Linear Baseline'} ({activeRun.primary_metric_name})
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
                          Verify with 5-fold permutation testing &amp; Super-Learner stacking.
                        </p>
                      )}
                    </div>
                    {!escalationCache[activeRun.id] && (
                      <button
                        onClick={() => handleEscalateDeepInference(activeRun.id)}
                        disabled={escalatingRunId === activeRun.id}
                        className="pop-btn mt-2 self-start px-3.5 py-1.5 rounded-xl bg-indigo-600 text-white text-xs font-bold cursor-pointer"
                      >
                        {escalatingRunId === activeRun.id ? 'Verifying' : 'Run 5-Stage Proof'}
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
                        {feat.discovery_note && (
                          <div className="text-[11px] text-slate-600 leading-snug">
                            {feat.discovery_note}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cross-Validated Model Leaderboard Table */}
                <div className="space-y-3 pt-2 border-t-2 border-slate-200">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-extrabold text-slate-900 uppercase tracking-wider">
                      Cross-Validated Model Leaderboard ({activeRun.leaderboard.length} Architectures)
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
            {/* SECTION 04: THE EVIDENCE LOCKER (SOURCE THESIS & DATASET)           */}
            {/* =================================================================== */}
            {selectedDiss && (
              <section className="pop-card rounded-3xl p-6 sm:p-8 space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                    04. SOURCE DISSERTATION &amp; RAW DATASET LOCKER
                  </span>
                  <span className="text-xs font-bold text-slate-600">
                    University Repository Record + Open Tabular Dataset
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div className="pop-subcard rounded-2xl p-4 space-y-2">
                    <div className="font-extrabold text-blue-700 uppercase">
                      Source Dissertation ({selectedDiss.university}, {selectedDiss.publication_year})
                    </div>
                    <div className="text-sm font-extrabold text-slate-900 leading-snug">
                      {selectedDiss.title}
                    </div>
                    <div className="text-slate-600">Authors: {selectedDiss.authors}</div>
                    <p className="text-slate-700 leading-relaxed pt-1">
                      {selectedDiss.abstract}
                    </p>
                    {selectedDiss.source_url && (
                      <a
                        href={selectedDiss.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 font-bold text-blue-600 hover:underline pt-1"
                      >
                        <BookOpen className="w-3.5 h-3.5" /> View Original University Publication{' '}
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    )}
                  </div>

                  {activeDataset && (
                    <div className="pop-subcard rounded-2xl p-4 space-y-2">
                      <div className="font-extrabold text-indigo-700 uppercase">
                        Audited Open Dataset ({activeDataset.source_type})
                      </div>
                      <div className="text-sm font-extrabold text-slate-900">
                        {activeDataset.name}
                      </div>
                      <div className="text-slate-600 font-mono">
                        {activeDataset.row_count} rows × {activeDataset.col_count} columns
                      </div>
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {activeDataset.column_metadata.map((c) => (
                          <span
                            key={c.name}
                            className="px-2 py-0.5 rounded-lg bg-slate-100 border border-slate-300 font-mono text-[10px] text-slate-800"
                          >
                            {c.name}
                          </span>
                        ))}
                      </div>
                      {activeDataset.source_url && (
                        <a
                          href={activeDataset.source_url}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 font-bold text-indigo-600 hover:underline pt-2"
                        >
                          <CheckCircle2 className="w-3.5 h-3.5" /> Inspect Raw Dataset on Zenodo{' '}
                          <ExternalLink className="w-3 h-3" />
                        </a>
                      )}
                    </div>
                  )}
                </div>
              </section>
            )}
          </div>

          {/* =================================================================== */}
          {/* SECTION 05: COLLAPSIBLE BUT DISCOVERABLE ARCHITECTURE & SCHEMA      */}
          {/* =================================================================== */}
          <section className="pop-card rounded-3xl p-5 sm:p-6 space-y-4">
            <div
              onClick={() => setShowArchDiagram(!showArchDiagram)}
              className="flex flex-wrap items-center justify-between gap-3 cursor-pointer select-none"
            >
              <div className="flex flex-wrap items-center gap-3">
                <span className="px-3.5 py-1 rounded-xl leather-badge text-xs font-extrabold">
                  05. SYSTEM ARCHITECTURE &amp; DATA SCHEMA
                </span>
                <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-blue-600" />
                  Cloudflare Edge • FastAPI &amp; Gemini 3.8 Flash • 30-Model LazyPredict • 4-Table Relational Schema
                </span>
              </div>

              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowArchDiagram(!showArchDiagram);
                }}
                className="pop-btn px-4 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
              >
                {showArchDiagram ? (
                  <>
                    Collapse Architecture Diagram <ChevronUp className="w-4 h-4" />
                  </>
                ) : (
                  <>
                    Expand Architecture &amp; Schema Diagram <ChevronDown className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>

            {showArchDiagram && (
              <div className="w-full overflow-x-auto rounded-2xl border border-slate-300 bg-[#FCFBF9] p-4 sm:p-6">
                <svg
                  viewBox="0 0 1160 520"
                  className="w-full min-w-[880px] h-auto"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <defs>
                    <pattern id="archGrid" width="24" height="24" patternUnits="userSpaceOnUse">
                      <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#E2E8F0" strokeWidth="0.7" />
                    </pattern>
                    <marker id="arrowBlue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563EB" />
                    </marker>
                    <marker id="arrowIndigo" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#4F46E5" />
                    </marker>
                    <marker id="arrowCrimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#E11D48" />
                    </marker>
                    <marker id="arrowSlate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
                    </marker>
                  </defs>

                  <rect x="0" y="0" width="1160" height="520" rx="12" fill="url(#archGrid)" />

                  <text x="24" y="26" fill="#475569" fontSize="10" fontWeight="800" fontFamily="monospace" letterSpacing="1.5">
                    I. INFERENCE &amp; EXECUTION PIPELINE
                  </text>
                  <line x1="255" y1="22" x2="1136" y2="22" stroke="#CBD5E1" strokeWidth="1" />

                  <g>
                    <rect x="24" y="44" width="190" height="148" rx="8" fill="#FFFFFF" stroke="#0F172A" strokeWidth="1.5" />
                    <line x1="24" y1="74" x2="214" y2="74" stroke="#E2E8F0" strokeWidth="1.2" />
                    <circle cx="40" cy="59" r="4" fill="#2563EB" />
                    <text x="52" y="63" fill="#0F172A" fontSize="11" fontWeight="800" fontFamily="monospace">
                      EDGE GATEWAY
                    </text>
                    <text x="38" y="96" fill="#334155" fontSize="10.5" fontWeight="700" fontFamily="monospace">
                      Cloudflare Worker
                    </text>
                    <text x="38" y="114" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ Zero-Trust Access Gate
                    </text>
                    <text x="38" y="130" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ Bot UA &amp; Honeypot Filter
                    </text>
                    <text x="38" y="146" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ React 19 + TypeScript
                    </text>
                    <text x="38" y="162" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      └─ SSE EventStream Client
                    </text>
                  </g>

                  <path d="M 214 118 L 266 118" stroke="#2563EB" strokeWidth="1.6" markerEnd="url(#arrowBlue)" />
                  <text x="240" y="110" textAnchor="middle" fill="#2563EB" fontSize="8.5" fontWeight="700" fontFamily="monospace">
                    HTTP/3
                  </text>

                  <g>
                    <rect x="268" y="44" width="210" height="148" rx="8" fill="#FFFFFF" stroke="#0F172A" strokeWidth="1.5" />
                    <line x1="268" y1="74" x2="478" y2="74" stroke="#E2E8F0" strokeWidth="1.2" />
                    <circle cx="284" cy="59" r="4" fill="#2563EB" />
                    <text x="296" y="63" fill="#0F172A" fontSize="11" fontWeight="800" fontFamily="monospace">
                      CLOUD RUN RUNTIME
                    </text>
                    <text x="282" y="96" fill="#334155" fontSize="10.5" fontWeight="700" fontFamily="monospace">
                      FastAPI + Python 3.12
                    </text>
                    <text x="282" y="114" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ Gemini 3.8 Flash Router
                    </text>
                    <text x="282" y="130" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ pdfplumber Table Parser
                    </text>
                    <text x="282" y="146" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ pypdf Claim &amp; DOI Parser
                    </text>
                    <text x="282" y="162" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      └─ Async httpx Orchestrator
                    </text>
                  </g>

                  <path d="M 478 118 L 530 118" stroke="#4F46E5" strokeWidth="1.6" markerEnd="url(#arrowIndigo)" />
                  <text x="504" y="110" textAnchor="middle" fill="#4F46E5" fontSize="8.5" fontWeight="700" fontFamily="monospace">
                    DataFrame
                  </text>

                  <g>
                    <rect x="532" y="44" width="236" height="148" rx="8" fill="#FFFFFF" stroke="#4F46E5" strokeWidth="1.8" />
                    <line x1="532" y1="74" x2="768" y2="74" stroke="#E2E8F0" strokeWidth="1.2" />
                    <circle cx="548" cy="59" r="4" fill="#4F46E5" />
                    <text x="560" y="63" fill="#0F172A" fontSize="11" fontWeight="800" fontFamily="monospace">
                      ML &amp; STATISTICAL ENGINE
                    </text>
                    <text x="546" y="96" fill="#312E81" fontSize="10.5" fontWeight="700" fontFamily="monospace">
                      30-Model Out-of-Fold Sweep
                    </text>
                    <text x="546" y="114" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ LazyPredict + Scikit-Learn
                    </text>
                    <text x="546" y="130" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ XGBoost / LightGBM / CatBoost
                    </text>
                    <text x="546" y="146" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ 5-Fold Stratified CV + 95% CI
                    </text>
                    <text x="546" y="162" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      └─ Super-Learner + Conformal Null
                    </text>
                  </g>

                  <path d="M 768 118 L 820 118" stroke="#E11D48" strokeWidth="1.6" markerEnd="url(#arrowCrimson)" />
                  <text x="794" y="110" textAnchor="middle" fill="#E11D48" fontSize="8.5" fontWeight="700" fontFamily="monospace">
                    Telemetry
                  </text>

                  <g>
                    <rect x="822" y="44" width="314" height="148" rx="8" fill="#FFFFFF" stroke="#0F172A" strokeWidth="1.5" />
                    <line x1="822" y1="74" x2="1136" y2="74" stroke="#E2E8F0" strokeWidth="1.2" />
                    <circle cx="838" cy="59" r="4" fill="#E11D48" />
                    <text x="850" y="63" fill="#0F172A" fontSize="11" fontWeight="800" fontFamily="monospace">
                      EXTERNAL REGISTRIES &amp; SYNTHESIS
                    </text>
                    <text x="836" y="96" fill="#881337" fontSize="10.5" fontWeight="700" fontFamily="monospace">
                      9 Live Scientific APIs + Gemini 3.8 Flash
                    </text>
                    <text x="836" y="114" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ OpenAlex + Zenodo + EuropePMC + arXiv
                    </text>
                    <text x="836" y="130" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ ClinicalTrials.gov API v2 + openFDA FAERS
                    </text>
                    <text x="836" y="146" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      ├─ OpenTargets GraphQL v4 + ChEMBL REST
                    </text>
                    <text x="836" y="162" fill="#64748B" fontSize="9.5" fontFamily="monospace">
                      └─ Surrogate IP Claims + Python Export (.py)
                    </text>
                  </g>

                  <text x="24" y="236" fill="#475569" fontSize="10" fontWeight="800" fontFamily="monospace" letterSpacing="1.5">
                    II. RELATIONAL DATA &amp; ARTIFACT SCHEMA (SQLITE / SQLALCHEMY ORM)
                  </text>
                  <line x1="425" y1="232" x2="1136" y2="232" stroke="#CBD5E1" strokeWidth="1" />

                  <path d="M 373 192 L 373 256" stroke="#94A3B8" strokeWidth="1.2" strokeDasharray="3 3" markerEnd="url(#arrowSlate)" />
                  <path d="M 650 192 L 650 256" stroke="#94A3B8" strokeWidth="1.2" strokeDasharray="3 3" markerEnd="url(#arrowSlate)" />
                  <path d="M 979 192 L 979 256" stroke="#94A3B8" strokeWidth="1.2" strokeDasharray="3 3" markerEnd="url(#arrowSlate)" />

                  <g>
                    <rect x="24" y="258" width="234" height="236" rx="8" fill="#FFFFFF" stroke="#334155" strokeWidth="1.5" />
                    <rect x="24" y="258" width="234" height="30" rx="7" fill="#0F172A" />
                    <text x="36" y="277" fill="#FFFFFF" fontSize="10.5" fontWeight="800" fontFamily="monospace">
                      TABLE: dissertations
                    </text>
                    <text x="218" y="277" fill="#93C5FD" fontSize="9" fontWeight="700" fontFamily="monospace">
                      1:N
                    </text>
                    <text x="36" y="308" fill="#2563EB" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      PK  id                   INTEGER
                    </text>
                    <text x="36" y="326" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          title                VARCHAR
                    </text>
                    <text x="36" y="344" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          university           VARCHAR
                    </text>
                    <text x="36" y="362" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          domain               VARCHAR
                    </text>
                    <text x="36" y="380" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          scientific_rigor     FLOAT
                    </text>
                    <text x="36" y="398" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          author_stat_method   VARCHAR
                    </text>
                    <text x="36" y="416" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          author_baseline      FLOAT
                    </text>
                    <text x="36" y="434" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          extracted_claims     JSONB
                    </text>
                    <text x="36" y="452" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          data_links           JSONB
                    </text>
                    <text x="36" y="470" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          key_entities         JSONB
                    </text>
                  </g>

                  <path d="M 258 326 L 308 326" stroke="#2563EB" strokeWidth="1.5" markerEnd="url(#arrowBlue)" />
                  <circle cx="264" cy="326" r="3" fill="#2563EB" />

                  <g>
                    <rect x="310" y="258" width="234" height="236" rx="8" fill="#FFFFFF" stroke="#334155" strokeWidth="1.5" />
                    <rect x="310" y="258" width="234" height="30" rx="7" fill="#1E293B" />
                    <text x="322" y="277" fill="#FFFFFF" fontSize="10.5" fontWeight="800" fontFamily="monospace">
                      TABLE: dataset_artifacts
                    </text>
                    <text x="504" y="277" fill="#93C5FD" fontSize="9" fontWeight="700" fontFamily="monospace">
                      1:N
                    </text>
                    <text x="322" y="308" fill="#2563EB" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      PK  id                   INTEGER
                    </text>
                    <text x="322" y="326" fill="#4F46E5" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      FK  dissertation_id      INTEGER
                    </text>
                    <text x="322" y="344" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          source_type          VARCHAR
                    </text>
                    <text x="322" y="362" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          source_url           VARCHAR
                    </text>
                    <text x="322" y="380" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          file_path            VARCHAR
                    </text>
                    <text x="322" y="398" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          row_count            INTEGER
                    </text>
                    <text x="322" y="416" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          col_count            INTEGER
                    </text>
                    <text x="322" y="434" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          column_metadata      JSONB
                    </text>
                    <text x="322" y="452" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          sample_rows          JSONB
                    </text>
                  </g>

                  <path d="M 544 326 L 594 326" stroke="#4F46E5" strokeWidth="1.5" markerEnd="url(#arrowIndigo)" />
                  <circle cx="550" cy="326" r="3" fill="#4F46E5" />

                  <g>
                    <rect x="596" y="258" width="246" height="236" rx="8" fill="#FFFFFF" stroke="#4F46E5" strokeWidth="1.6" />
                    <rect x="596" y="258" width="246" height="30" rx="7" fill="#312E81" />
                    <text x="608" y="277" fill="#FFFFFF" fontSize="10.5" fontWeight="800" fontFamily="monospace">
                      TABLE: lazypredict_runs
                    </text>
                    <text x="802" y="277" fill="#C7D2FE" fontSize="9" fontWeight="700" fontFamily="monospace">
                      OOF CV
                    </text>
                    <text x="608" y="308" fill="#2563EB" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      PK  id                   INTEGER
                    </text>
                    <text x="608" y="326" fill="#4F46E5" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      FK  dataset_id           INTEGER
                    </text>
                    <text x="608" y="344" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          target_column        VARCHAR
                    </text>
                    <text x="608" y="362" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          task_type            VARCHAR
                    </text>
                    <text x="608" y="380" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          models_tested_count  INTEGER
                    </text>
                    <text x="608" y="398" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          best_model_name      VARCHAR
                    </text>
                    <text x="608" y="416" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          best_score           FLOAT
                    </text>
                    <text x="608" y="434" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          signal_gain_delta    FLOAT
                    </text>
                    <text x="608" y="452" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          leaderboard          JSONB
                    </text>
                    <text x="608" y="470" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          top_features         JSONB
                    </text>
                  </g>

                  <path d="M 842 344 L 892 344" stroke="#E11D48" strokeWidth="1.5" markerEnd="url(#arrowCrimson)" />
                  <circle cx="848" cy="344" r="3" fill="#E11D48" />

                  <g>
                    <rect x="894" y="258" width="242" height="236" rx="8" fill="#FFFFFF" stroke="#E11D48" strokeWidth="1.6" />
                    <rect x="894" y="258" width="242" height="30" rx="7" fill="#881337" />
                    <text x="906" y="277" fill="#FFFFFF" fontSize="10.5" fontWeight="800" fontFamily="monospace">
                      TABLE: market_opportunities
                    </text>
                    <text x="906" y="308" fill="#2563EB" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      PK  id                   INTEGER
                    </text>
                    <text x="906" y="326" fill="#4F46E5" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      FK  dissertation_id      INTEGER
                    </text>
                    <text x="906" y="344" fill="#4F46E5" fontSize="9.5" fontWeight="700" fontFamily="monospace">
                      FK  lazy_run_id          INTEGER
                    </text>
                    <text x="906" y="362" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          opportunity_type     VARCHAR
                    </text>
                    <text x="906" y="380" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          clinical_trials_chk  JSONB
                    </text>
                    <text x="906" y="398" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          opentargets_chembl   JSONB
                    </text>
                    <text x="906" y="416" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          overall_tos_score    FLOAT
                    </text>
                    <text x="906" y="434" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          proposed_solution    TEXT
                    </text>
                    <text x="906" y="452" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          validation_roadmap   JSONB
                    </text>
                    <text x="906" y="470" fill="#334155" fontSize="9.5" fontFamily="monospace">
                          provisional_ip_claim TEXT
                    </text>
                  </g>
                </svg>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
