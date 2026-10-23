import React, { useEffect, useState } from 'react';
import {
  ArrowUpRight,
  Bookmark,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
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
  ShieldCheck,
  Sparkles,
  Star,
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

interface SavedItem {
  id: number;
  dissertation_id: number;
  opportunity_id?: number | null;
  is_starred: boolean;
  is_bookmarked: boolean;
  user_note: string;
  tags: string[];
  created_at: string;
  updated_at?: string;
  dissertation?: Dissertation;
  opportunity?: MarketOpportunity;
}

interface RegisteredAgent {
  id: number;
  name: string;
  role: string;
  description: string;
  skill_name: string;
  skill_path: string;
  agent_md_path: string;
  tools_allowed: string[];
  policies: { rule: string; target: string; reason: string }[];
  status: string;
  invocations_count: number;
  last_invoked_at?: string;
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

const DOMAIN_NAMES: Record<string, string> = {
  all: 'All Domains',
  biomedical: 'Biomedical',
  cs_ai: 'AI / Systems',
  energy_materials: 'Energy / Materials',
  climate_earth: 'Climate / Earth',
};

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

  const [activeNav, setActiveNav] = useState<'discoveries' | 'saved' | 'docs' | 'agents'>('discoveries');
  const [docsSection, setDocsSection] = useState<'about' | 'how' | 'arch'>('about');
  const [savedFilter, setSavedFilter] = useState<'all' | 'star' | 'ribbon'>('all');

  const [showAllModels, setShowAllModels] = useState<boolean>(true);
  const [showUploadBox, setShowUploadBox] = useState<boolean>(false);

  const [dissertations, setDissertations] = useState<Dissertation[]>([]);
  const [opportunities, setOpportunities] = useState<MarketOpportunity[]>([]);
  const [savedItems, setSavedItems] = useState<SavedItem[]>([]);
  const [registeredAgents, setRegisteredAgents] = useState<RegisteredAgent[]>([]);
  const [noteDrafts, setNoteDrafts] = useState<Record<number, string>>({});
  const [verifyingAgent, setVerifyingAgent] = useState<string | null>(null);
  const [agentVerifyResult, setAgentVerifyResult] = useState<Record<string, any>>({});

  const [selectedDissId, setSelectedDissId] = useState<number | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [domainFilter, setDomainFilter] = useState<string>('all');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [escalationCache, setEscalationCache] = useState<Record<number, DeepInferenceResult>>({});
  const [escalatingRunId, setEscalatingRunId] = useState<number | null>(null);
  const [stepActionResults, setStepActionResults] = useState<Record<string, any>>({});
  const [executingStepKey, setExecutingStepKey] = useState<string | null>(null);
  const [runningCustomTarget, setRunningCustomTarget] = useState<boolean>(false);

  const [harvesting, setHarvesting] = useState<boolean>(false);
  const [streamEvents, setStreamEvents] = useState<string[]>([]);
  const [searchInput, setSearchInput] = useState<string>('');

  const [uploadPdf, setUploadPdf] = useState<File | null>(null);
  const [uploadCsv, setUploadCsv] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadedDiss, setUploadedDiss] = useState<Dissertation | null>(null);

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
      const [dRes, oRes, sRes, aRes] = await Promise.all([
        fetch('/api/dissertations', { headers: authHeaders() }),
        fetch('/api/opportunities', { headers: authHeaders() }),
        fetch('/api/saved', { headers: authHeaders() }),
        fetch('/api/agents/registry', { headers: authHeaders() }),
      ]);
      if (dRes.status === 401 || oRes.status === 401) {
        setIsUnlocked(false);
        return;
      }
      const dData: Dissertation[] = await dRes.json();
      const oData: MarketOpportunity[] = await oRes.json();
      const sData: SavedItem[] = sRes.ok ? await sRes.json() : [];
      const aData: RegisteredAgent[] = aRes.ok ? await aRes.json() : [];

      setDissertations(dData);
      setOpportunities(oData);
      setSavedItems(sData);
      setRegisteredAgents(aData);

      const drafts: Record<number, string> = {};
      sData.forEach((s) => {
        drafts[s.id] = s.user_note || '';
      });
      setNoteDrafts((prev) => ({ ...drafts, ...prev }));

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

  const savedByDissId = (dissId: number): SavedItem | undefined =>
    savedItems.find((s) => s.dissertation_id === dissId);

  const handleToggleSave = async (
    e: React.MouseEvent,
    dissId: number,
    oppId: number | undefined,
    toggleField: 'star' | 'ribbon'
  ) => {
    e.stopPropagation();
    try {
      const res = await fetch('/api/saved/toggle', {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          dissertation_id: dissId,
          opportunity_id: oppId,
          toggle_field: toggleField,
        }),
      });
      if (res.ok) {
        const sRes = await fetch('/api/saved', { headers: authHeaders() });
        if (sRes.ok) {
          const sData: SavedItem[] = await sRes.json();
          setSavedItems(sData);
        }
      }
    } catch (err) {
      console.error('Toggle save error:', err);
    }
  };

  const handleSaveNote = async (savedId: number) => {
    const noteText = noteDrafts[savedId] ?? '';
    try {
      const res = await fetch(`/api/saved/${savedId}/note`, {
        method: 'PATCH',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ user_note: noteText }),
      });
      if (res.ok) {
        const updated: SavedItem = await res.json();
        setSavedItems((prev) => prev.map((x) => (x.id === savedId ? updated : x)));
      }
    } catch (err) {
      console.error('Save note error:', err);
    }
  };

  const handleVerifyAgent = async (agentName: string) => {
    setVerifyingAgent(agentName);
    try {
      const res = await fetch(`/api/agents/registry/${agentName}/verify`, {
        method: 'POST',
        headers: authHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        setAgentVerifyResult((prev) => ({ ...prev, [agentName]: data }));
        const aRes = await fetch('/api/agents/registry', { headers: authHeaders() });
        if (aRes.ok) setRegisteredAgents(await aRes.json());
      }
    } finally {
      setVerifyingAgent(null);
    }
  };

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
    setActiveNav('discoveries');
    setSelectedDissId(diss.id);
    const firstRun = diss.datasets?.[0]?.lazy_runs?.[0];
    if (firstRun) {
      setSelectedRunId(firstRun.id);
    }
    if (scrollToDetail) {
      setTimeout(() => {
        const el = document.getElementById('active-finding-inspector');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 60);
    }
  };

  const handleResetToGlobalView = () => {
    setActiveNav('discoveries');
    setDomainFilter('all');
    setSearchInput('');
    setStreamEvents([]);
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
    setActiveNav('discoveries');
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

  const executeFileUpload = async (pdfToSend: File | null, csvToSend: File | null) => {
    if (!pdfToSend && !csvToSend) return;
    setUploading(true);
    setErrorMsg(null);
    setActiveNav('discoveries');
    setDomainFilter('all');
    try {
      const fd = new FormData();
      if (pdfToSend) fd.append('pdf_file', pdfToSend);
      if (csvToSend) fd.append('dataset_file', csvToSend);

      const res = await fetch('/api/dissertations/upload', {
        method: 'POST',
        headers: authHeaders(),
        body: fd,
      });
      if (res.ok) {
        const data: Dissertation = await res.json();
        setDissertations((prev) => [data, ...prev.filter((d) => d.id !== data.id)]);
        if (data.opportunities?.length) {
          setOpportunities((prev) => [
            ...data.opportunities,
            ...prev.filter((o) => o.dissertation_id !== data.id),
          ]);
        }
        await fetchAllData();
        if (data.id) {
          setSelectedDissId(data.id);
          const firstRun = data.datasets?.[0]?.lazy_runs?.[0];
          if (firstRun) setSelectedRunId(firstRun.id);
        }
        setUploadedDiss(data);
        setShowUploadBox(false);
        setUploadPdf(null);
        setUploadCsv(null);
      } else {
        const errBody = await res.json().catch(() => ({}));
        setErrorMsg(errBody.detail || `Upload failed (HTTP ${res.status}).`);
      }
    } catch (err: any) {
      setErrorMsg(err?.message || 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await executeFileUpload(uploadPdf, uploadCsv);
  };

  // Dynamic domain counts from live indexed dissertations
  const domainCounts: Record<string, number> = { all: dissertations.length };
  dissertations.forEach((d) => {
    domainCounts[d.domain] = (domainCounts[d.domain] || 0) + 1;
  });

  const filteredSavedItems = savedItems.filter((item) => {
    if (savedFilter === 'star') return item.is_starred;
    if (savedFilter === 'ribbon') return item.is_bookmarked;
    return item.is_starred || item.is_bookmarked;
  });

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
            stroke="#2B4257"
            strokeWidth="2.2"
            strokeOpacity="0.26"
            className="trajectory-dash"
          />
          <path
            d="M -40 640 C 380 460, 760 740, 1160 440 C 1310 330, 1420 460, 1520 390"
            stroke="#4A2E40"
            strokeWidth="2"
            strokeOpacity="0.22"
            className="trajectory-dash-reverse"
          />

          <g className="binary-scroll-left" opacity="0.28">
            <text x="0" y="140" fill="#2B4257" fontSize="12" fontWeight="600" letterSpacing="5">
              01101101 10100110 01001110 10110010 01011010 11001010 01000001 11010010 01110100 01101101 10100110 01001110 10110010 01011010 11001010 01000001 11010010 01110100
            </text>
            <text x="0" y="830" fill="#54250E" fontSize="12" fontWeight="600" letterSpacing="5">
              10110010 01011010 11001010 01101101 10100110 01001110 11010110 00101101 10111001 10110010 01011010 11001010 01101101 10100110 01001110 11010110 00101101 10111001
            </text>
          </g>

          <g className="binary-scroll-right" opacity="0.26">
            <text x="0" y="510" fill="#4A2E40" fontSize="12" fontWeight="600" letterSpacing="5">
              11010110 00101101 10111001 01011100 10010110 01100011 10011010 01101101 00110101 11010110 00101101 10111001 01011100 10010110 01100011 10011010 01101101 00110101
            </text>
          </g>

          <g className="lens-float-1" opacity="0.46">
            <circle cx="210" cy="210" r="52" fill="#EDF0F2" fillOpacity="0.48" />
            <g clipPath="url(#lensClip1)">
              <g className="binary-scroll-lens">
                <text x="150" y="214" fill="#1F3142" fontSize="12" fontWeight="700" letterSpacing="2">
                  01101001 11010010 01011011 01101001 11010010 01011011
                </text>
              </g>
            </g>
            <circle cx="210" cy="210" r="52" stroke="#3B1A08" strokeWidth="4.5" />
            <circle cx="210" cy="210" r="46" stroke="#2B4257" strokeWidth="2" />
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
                <text x="1150" y="374" fill="#4A2E40" fontSize="12" fontWeight="700" letterSpacing="2">
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
            <circle cx="260" cy="730" r="46" fill="#F7ECED" fillOpacity="0.45" />
            <g clipPath="url(#lensClip3)">
              <g className="binary-scroll-lens">
                <text x="205" y="734" fill="#7A1C24" fontSize="11.5" fontWeight="700" letterSpacing="2">
                  01100101 10110100 01011001 01100101 10110100 01011001
                </text>
              </g>
            </g>
            <circle cx="260" cy="730" r="46" stroke="#3B1A08" strokeWidth="4" />
            <circle cx="260" cy="730" r="40" stroke="#7A1C24" strokeWidth="1.8" />
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
              fill="#EDF0F2"
              stroke="#2B4257"
              strokeWidth="2.2"
            />
            <polygon
              points="1080,161 1092,172 1080,189 1068,172"
              fill="#364F66"
              fillOpacity="0.65"
            />
            <line x1="1062" y1="170" x2="1098" y2="170" stroke="#1F3142" strokeWidth="1.4" />
            <line x1="1080" y1="155" x2="1080" y2="194" stroke="#1F3142" strokeWidth="1.2" />
            <path
              d="M 1106 150 L 1108 156 L 1114 158 L 1108 160 L 1106 166 L 1104 160 L 1098 158 L 1104 156 Z"
              fill="#2B4257"
            />
          </g>

          <g className="gem-episodic-2">
            <polygon
              points="1130,680 1146,694 1140,716 1120,716 1114,694"
              fill="#F7ECED"
              stroke="#7A1C24"
              strokeWidth="2.2"
            />
            <polygon
              points="1130,685 1141,695 1130,711 1119,695"
              fill="#7A1C24"
              fillOpacity="0.58"
            />
            <line x1="1114" y1="694" x2="1146" y2="694" stroke="#61141B" strokeWidth="1.4" />
            <line x1="1130" y1="680" x2="1130" y2="716" stroke="#61141B" strokeWidth="1.2" />
            <path
              d="M 1153 675 L 1155 680 L 1160 682 L 1155 684 L 1153 689 L 1151 684 L 1146 682 L 1151 680 Z"
              fill="#4A2E40"
            />
          </g>
        </svg>
      </div>

      {/* ===================================================================== */}
      {/* 3D PEBBLED MAHOGANY LEATHER TOP BAR WITH GLOBAL NAVIGATION & DOCS     */}
      {/* ===================================================================== */}
      <header className="sticky top-0 z-30 leather-bar text-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={handleResetToGlobalView}
            className="flex items-center gap-2.5 text-left cursor-pointer group shrink-0"
            title="Discoveries"
          >
            <div className="w-9 h-9 rounded-xl leather-badge flex items-center justify-center font-extrabold text-base tracking-tighter">
              <span>
                T<span className="text-blue-400">O</span>
              </span>
            </div>
            <span className="text-lg font-extrabold tracking-tight text-white group-hover:text-[#E6D2BE] transition">
              Thesis<span className="text-[#D8B99C]">Out</span>
            </span>
          </button>

          {isUnlocked && (
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => setActiveNav('discoveries')}
                className={`pop-btn px-3 py-1.5 rounded-xl text-xs font-extrabold flex items-center gap-1.5 cursor-pointer ${
                  activeNav === 'discoveries'
                    ? 'bg-[#2B4257] text-white'
                    : 'bg-black/35 hover:bg-black/50 text-white'
                }`}
              >
                <Globe className="w-3.5 h-3.5" />
                Discoveries ({dissertations.length})
              </button>

              <button
                type="button"
                onClick={() => setActiveNav('saved')}
                className={`pop-btn px-3 py-1.5 rounded-xl text-xs font-extrabold flex items-center gap-1.5 cursor-pointer ${
                  activeNav === 'saved'
                    ? 'bg-[#7A1C24] text-white'
                    : 'bg-black/35 hover:bg-black/50 text-white'
                }`}
              >
                <Bookmark className="w-3.5 h-3.5" />
                Saved ({savedItems.length})
              </button>

              <button
                type="button"
                onClick={() => setActiveNav('agents')}
                className={`pop-btn px-3 py-1.5 rounded-xl text-xs font-extrabold flex items-center gap-1.5 cursor-pointer ${
                  activeNav === 'agents'
                    ? 'bg-[#4A2E40] text-white'
                    : 'bg-black/35 hover:bg-black/50 text-white'
                }`}
              >
                <Cpu className="w-3.5 h-3.5" />
                Agents ({registeredAgents.length})
              </button>

              <button
                type="button"
                onClick={() => setActiveNav('docs')}
                className={`pop-btn px-3 py-1.5 rounded-xl text-xs font-extrabold flex items-center gap-1.5 cursor-pointer ${
                  activeNav === 'docs'
                    ? 'bg-[#2B4257] text-white'
                    : 'bg-black/35 hover:bg-black/50 text-white'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                Docs
              </button>

              <button
                type="button"
                onClick={() => {
                  setActiveNav('discoveries');
                  setShowUploadBox(!showUploadBox);
                }}
                className="pop-btn px-3 py-1.5 rounded-xl bg-[#FAF6F0] text-[#3B1A08] text-xs font-extrabold flex items-center gap-1.5 cursor-pointer"
              >
                <Upload className="w-3.5 h-3.5 text-[#2B4257]" />
                {showUploadBox ? 'Close' : 'Upload PDF / CSV'}
              </button>

              <button
                type="button"
                onClick={handleLockWorkspace}
                className="px-2.5 py-1.5 rounded-xl bg-black/30 hover:bg-black/45 border border-white/20 text-xs font-bold text-[#F5E6D8] flex items-center gap-1 cursor-pointer"
                title="Lock"
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
        <main className="relative z-10 max-w-sm mx-auto px-4 pt-24 pb-12">
          <form
            onSubmit={handleUnlockSubmit}
            className="pop-card rounded-3xl p-6 space-y-4"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl leather-badge flex items-center justify-center font-extrabold text-lg">
                <Lock className="w-4 h-4 text-[#D8B99C]" />
              </div>
              <h1 className="text-base font-extrabold text-slate-900">
                Access Code
              </h1>
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

            <div className="space-y-1.5">
              <input
                type="password"
                value={gateInput}
                onChange={(e) => setGateInput(e.target.value)}
                placeholder="Access code"
                autoFocus
                className="w-full px-4 py-2.5 rounded-2xl bg-slate-50 border-2 border-slate-900 text-sm font-bold text-slate-900 focus:outline-none focus:bg-white"
              />
              {gateError && (
                <div className="text-xs font-bold text-[#7A1C24]">{gateError}</div>
              )}
            </div>

            <button
              type="submit"
              className="pop-btn w-full py-2.5 rounded-2xl bg-[#2B4257] hover:bg-[#1F3142] text-white font-extrabold text-sm cursor-pointer"
            >
              Unlock
            </button>
          </form>
        </main>
      ) : (
        <main className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 py-5 space-y-5">
          {errorMsg && (
            <div className="pop-card-crimson rounded-2xl p-3 flex items-center justify-between text-xs font-bold text-[#4D1016]">
              <span>{errorMsg}</span>
              <button
                onClick={() => setErrorMsg(null)}
                className="px-2 py-0.5 rounded-lg bg-[#EFE0E2] text-[#4D1016] cursor-pointer"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* ================================================================= */}
          {/* VIEW 1: DOCS SECTION (ABOUT, HOW IT WORKS, ARCHITECTURE & SCHEMA) */}
          {/* ================================================================= */}
          {activeNav === 'docs' && (
            <section className="pop-card rounded-2xl p-5 space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
                <div className="flex items-center gap-2.5">
                  <span className="px-3 py-1 rounded-xl leather-badge text-xs font-extrabold">
                    DOCUMENTATION
                  </span>
                  <span className="text-xs font-mono font-bold text-slate-600">
                    About • How It Works • Architecture &amp; Schema
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {[
                    { id: 'about', label: '1. About ThesisOut' },
                    { id: 'how', label: '2. How It Works' },
                    { id: 'arch', label: '3. Architecture & Schema' },
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setDocsSection(tab.id as any)}
                      className={`pop-btn px-3 py-1.5 rounded-xl text-xs font-extrabold cursor-pointer ${
                        docsSection === tab.id
                          ? 'bg-[#2B4257] text-white'
                          : 'bg-slate-100 text-slate-800 hover:bg-slate-200'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              {docsSection === 'about' && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                  <div className="pop-subcard rounded-xl p-4 space-y-2">
                    <div className="text-[11px] font-extrabold text-[#2B4257] uppercase">
                      Mission &amp; Scope
                    </div>
                    <div className="text-sm font-extrabold text-slate-900">
                      Empirical Discovery Across Global Doctoral Research
                    </div>
                    <div className="text-slate-700 leading-relaxed font-medium">
                      Doctoral dissertations frequently publish rich experimental and clinical datasets analyzed only with classical linear or univariate models. ThesisOut harvests live dissertations and open-access datasets, re-analyzes every tabular variable across 30 machine learning architectures, and surfaces non-linear interactions with high commercial and clinical value.
                    </div>
                  </div>

                  <div className="pop-subcard rounded-xl p-4 space-y-2">
                    <div className="text-[11px] font-extrabold text-[#4A2E40] uppercase">
                      Methodology &amp; Statistical Protocol
                    </div>
                    <div className="text-sm font-extrabold text-slate-900">
                      Nested Cross-Validation &amp; Conformal Calibration
                    </div>
                    <div className="text-slate-700 leading-relaxed font-medium">
                      Dissertations are indexed from OpenAlex, EuropePMC, and arXiv alongside empirical tabular datasets from Zenodo, Figshare, Harvard Dataverse, and Dryad. Imputation, categorical encoding, and scaling are fit strictly inside cross-validation folds to preserve out-of-fold integrity.
                    </div>
                  </div>

                  <div className="pop-subcard rounded-xl p-4 space-y-2">
                    <div className="text-[11px] font-extrabold text-[#7A1C24] uppercase">
                      Translational Dossiers &amp; Reproducibility
                    </div>
                    <div className="text-sm font-extrabold text-slate-900">
                      Multi-Registry Evidence &amp; Standalone Code
                    </div>
                    <div className="text-slate-700 leading-relaxed font-medium">
                      Each empirical signal is cross-referenced against ClinicalTrials.gov, OpenTargets, ChEMBL, and openFDA to quantify clinical and patent whitespace. Researchers can maintain due-diligence shortlists with persistent notes and export standalone Python pipelines.
                    </div>
                  </div>
                </div>
              )}

              {docsSection === 'how' && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 text-xs">
                  <div className="pop-subcard rounded-xl p-4 space-y-2">
                    <span className="px-2 py-0.5 rounded bg-slate-900 text-white font-mono text-[10px] font-bold">
                      STAGE 01 • 3-STAGE DATASET FINDER
                    </span>
                    <div className="text-sm font-extrabold text-slate-900">
                      ORCID Graph, Full-Text &amp; 4-Repo Harvester
                    </div>
                    <div className="text-slate-700 leading-relaxed font-medium">
                      Scrapes full-text Data Availability sections from EuropePMC/OA PDFs, matches author ORCIDs &amp; lab graphs (±2 yrs) across OpenAlex, DataCite &amp; Crossref, and resolves tabular files from Zenodo, Figshare, Harvard Dataverse, and Dryad.
                    </div>
                  </div>

                  <div className="pop-subcard rounded-xl p-4 space-y-2">
                    <span className="px-2 py-0.5 rounded bg-[#2B4257] text-white font-mono text-[10px] font-bold">
                      STAGE 02 • ML RIGOR SCIENTIST
                    </span>
                    <div className="text-sm font-extrabold text-slate-900">
                      30-Model Out-of-Fold Benchmark
                    </div>
                    <div className="text-slate-700 leading-relaxed font-medium">
                      Strips identifiers and target proxies, fits preprocessing strictly inside each fold, evaluates 30 classifiers/regressors, and computes 95% confidence intervals and feature importances.
                    </div>
                  </div>

                  <div className="pop-subcard rounded-xl p-4 space-y-2">
                    <span className="px-2 py-0.5 rounded bg-[#4A2E40] text-white font-mono text-[10px] font-bold">
                      STAGE 03 • DEEP INFERENCE AUDITOR
                    </span>
                    <div className="text-sm font-extrabold text-slate-900">
                      5-Stage Super-Learner &amp; Conformal Proof
                    </div>
                    <div className="text-slate-700 leading-relaxed font-medium">
                      Runs 5-fold target permutation null tests, top-feature ablation, Level-2 Super-Learner stacking, surrogate decision-tree threshold extraction, and 90% split-conformal calibration.
                    </div>
                  </div>

                  <div className="pop-subcard rounded-xl p-4 space-y-2">
                    <span className="px-2 py-0.5 rounded bg-[#7A1C24] text-white font-mono text-[10px] font-bold">
                      STAGE 04 • TRANSLATION &amp; WHITESPACE
                    </span>
                    <div className="text-sm font-extrabold text-slate-900">
                      9-API Regulatory Audit &amp; Code Export
                    </div>
                    <div className="text-slate-700 leading-relaxed font-medium">
                      Cross-checks ClinicalTrials.gov, OpenTargets, ChEMBL, and openFDA FAERS, synthesizes the Translation Opportunity Score (TOS) via Gemini 4 Argon (budget-capped at 10 RPM / 200 calls/day, with Gemini 3.8 Flash fallback), and exports standalone Python code.
                    </div>
                  </div>
                </div>
              )}

              {docsSection === 'arch' && (
                <div className="rounded-2xl border-2 border-[#2B1406] bg-[#FAF8F5] p-5 space-y-5">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                    <div className="pop-subcard rounded-xl p-4 space-y-2.5 border-l-4 border-l-[#2B4257]">
                      <div className="text-[11px] font-mono font-extrabold text-[#2B4257] uppercase">
                        01. AI &amp; ML MODELS
                      </div>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px] font-bold">
                        {[
                          'Gemini 4 Argon (10 RPM / 200/day Cap)',
                          'Gemini 3.8 Flash (Fallback)',
                          'LazyPredict (30 Models)',
                          'ExtraTrees / RandomForest',
                          'XGBoost',
                          'LightGBM',
                          'CatBoost',
                          'Super-Learner Stacking',
                          '90% Split-Conformal',
                        ].map((item) => (
                          <span key={item} className="px-2.5 py-1 rounded-lg bg-[#EDF0F2] text-[#1E2E3D] border border-[#B8C6D1]">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="pop-subcard rounded-xl p-4 space-y-2.5 border-l-4 border-l-[#7A1C24]">
                      <div className="text-[11px] font-mono font-extrabold text-[#7A1C24] uppercase">
                        02. LIVE SCIENTIFIC APIs
                      </div>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px] font-bold">
                        {[
                          'OpenAlex + ORCID Graph',
                          'DataCite + Crossref',
                          'Zenodo REST API',
                          'Figshare API v2',
                          'Harvard Dataverse API',
                          'Dryad API v2',
                          'EuropePMC FullTextXML',
                          'ClinicalTrials.gov v2',
                          'OpenTargets + ChEMBL + openFDA',
                        ].map((item) => (
                          <span key={item} className="px-2.5 py-1 rounded-lg bg-[#F7ECED] text-[#4D1016] border border-[#DEC0C3]">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="pop-subcard rounded-xl p-4 space-y-2.5 border-l-4 border-l-[#4A2E40]">
                      <div className="text-[11px] font-mono font-extrabold text-[#4A2E40] uppercase">
                        03. REGISTERED AGENTS
                      </div>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px] font-bold">
                        {[
                          'harvester-agent',
                          'pdf-extractor-agent',
                          'data-cleansing-agent',
                          'ml-rigor-scientist',
                          'deep-inference-auditor',
                          'translation-whitespace-agent',
                          'qa-production-guardian',
                        ].map((item) => (
                          <span key={item} className="px-2.5 py-1 rounded-lg bg-[#F3EDF1] text-[#2D1B27] border border-[#D6C5D0]">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="pop-subcard rounded-xl p-4 space-y-2.5 border-l-4 border-l-[#58270E]">
                      <div className="text-[11px] font-mono font-extrabold text-[#58270E] uppercase">
                        04. LIBRARIES &amp; TOOLS
                      </div>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px] font-bold">
                        {[
                          'FastAPI + Pydantic',
                          'Scikit-Learn',
                          'Pandas / NumPy / SciPy',
                          'Statsmodels',
                          'pdfplumber + pypdf',
                          'React 19 + Vite',
                          'Tailwind CSS',
                        ].map((item) => (
                          <span key={item} className="px-2.5 py-1 rounded-lg bg-[#EFEAE1] text-[#1E140F] border border-[#D6C7B8]">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="pop-subcard rounded-xl p-4 space-y-2.5 border-l-4 border-l-[#2B4257]">
                      <div className="text-[11px] font-mono font-extrabold text-[#2B4257] uppercase">
                        05. DATABASES &amp; STORAGE
                      </div>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px] font-bold">
                        {[
                          'SQLite + SQLAlchemy ORM',
                          'dissertations',
                          'dataset_artifacts',
                          'lazypredict_runs',
                          'market_opportunities',
                          'saved_items',
                          'agent_registry',
                        ].map((item) => (
                          <span key={item} className="px-2.5 py-1 rounded-lg bg-[#EDF0F2] text-[#1E2E3D] border border-[#B8C6D1]">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="pop-subcard rounded-xl p-4 space-y-2.5 border-l-4 border-l-[#4A2E40]">
                      <div className="text-[11px] font-mono font-extrabold text-[#4A2E40] uppercase">
                        06. INFRASTRUCTURE &amp; RUNTIME
                      </div>
                      <div className="flex flex-wrap gap-1.5 font-mono text-[11px] font-bold">
                        {[
                          'Cloudflare Worker Edge',
                          'Google Cloud Run',
                          'Docker Multi-Stage',
                          'SSE Real-Time Stream',
                          '5-Fold Out-of-Fold CV',
                        ].map((item) => (
                          <span key={item} className="px-2.5 py-1 rounded-lg bg-[#F3EDF1] text-[#2D1B27] border border-[#D6C5D0]">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* ================================================================= */}
          {/* VIEW 2: PERSISTENT SAVED MEMORY (STAR & RIBBON + NOTES)           */}
          {/* ================================================================= */}
          {activeNav === 'saved' && (
            <section className="pop-card rounded-2xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
                <div className="flex items-center gap-2.5">
                  <span className="px-3 py-1 rounded-xl leather-badge text-xs font-extrabold">
                    SAVED MEMORY ({filteredSavedItems.length})
                  </span>
                  <span className="text-xs font-mono font-bold text-slate-600">
                    Saved Shortlist • Favorites, Bookmarks &amp; Research Notes
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  {[
                    { id: 'all', label: `All (${savedItems.length})` },
                    { id: 'star', label: `Favorites (${savedItems.filter((x) => x.is_starred).length})` },
                    { id: 'ribbon', label: `Bookmarked (${savedItems.filter((x) => x.is_bookmarked).length})` },
                  ].map((f) => (
                    <button
                      key={f.id}
                      type="button"
                      onClick={() => setSavedFilter(f.id as any)}
                      className={`px-3 py-1 rounded-xl text-xs font-extrabold border-2 cursor-pointer ${
                        savedFilter === f.id
                          ? 'bg-slate-900 text-white border-slate-900'
                          : 'bg-white text-slate-700 border-slate-300'
                      }`}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>

              {filteredSavedItems.length === 0 ? (
                <div className="pop-subcard rounded-xl p-6 text-center space-y-2">
                  <div className="text-sm font-extrabold text-slate-900">
                    No Saved Discoveries Yet
                  </div>
                  <div className="text-xs text-slate-600 font-medium">
                    Use the Favorite or Bookmark icon on any discovery card to add it to your shortlist.
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {filteredSavedItems.map((item) => {
                    const diss = item.dissertation || dissertations.find((d) => d.id === item.dissertation_id);
                    const opp =
                      item.opportunity ||
                      opportunities.find((o) => o.dissertation_id === item.dissertation_id);
                    if (!diss) return null;
                    return (
                      <div key={item.id} className="pop-subcard rounded-2xl p-4 space-y-3">
                        <div className="flex items-center justify-between gap-2">
                          <span className="px-2.5 py-0.5 rounded-lg bg-slate-900 text-white text-xs font-extrabold">
                            {diss.university} • {diss.publication_year}
                          </span>
                          <div className="flex items-center gap-1.5">
                            <button
                              type="button"
                              onClick={(e) => handleToggleSave(e, diss.id, opp?.id, 'star')}
                              className={`p-1.5 rounded-lg border cursor-pointer ${
                                item.is_starred
                                  ? 'bg-[#7A1C24] text-white border-slate-900'
                                  : 'bg-white text-slate-500 border-slate-300'
                              }`}
                              title="Favorite"
                            >
                              <Star className="w-3.5 h-3.5 fill-current" />
                            </button>
                            <button
                              type="button"
                              onClick={(e) => handleToggleSave(e, diss.id, opp?.id, 'ribbon')}
                              className={`p-1.5 rounded-lg border cursor-pointer ${
                                item.is_bookmarked
                                  ? 'bg-[#2B4257] text-white border-slate-900'
                                  : 'bg-white text-slate-500 border-slate-300'
                              }`}
                              title="Bookmark"
                            >
                              <Bookmark className="w-3.5 h-3.5 fill-current" />
                            </button>
                          </div>
                        </div>

                        <div className="text-sm font-extrabold text-slate-900">{diss.title}</div>
                        {opp && (
                          <div className="text-xs font-bold text-[#2B4257]">
                            Opportunity (TOS {opp.overall_tos_score}): {opp.title}
                          </div>
                        )}

                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={noteDrafts[item.id] ?? item.user_note ?? ''}
                            onChange={(e) =>
                              setNoteDrafts((prev) => ({ ...prev, [item.id]: e.target.value }))
                            }
                            placeholder="Persistent researcher note..."
                            className="flex-1 px-3 py-1.5 rounded-xl bg-white border border-slate-300 text-xs font-bold text-slate-900"
                          />
                          <button
                            type="button"
                            onClick={() => handleSaveNote(item.id)}
                            className="pop-btn px-3 py-1.5 rounded-xl bg-[#2B4257] text-white text-xs font-extrabold cursor-pointer"
                          >
                            Save Note
                          </button>
                          <button
                            type="button"
                            onClick={() => handleSelectDissertation(diss, true)}
                            className="pop-btn px-3 py-1.5 rounded-xl bg-slate-900 text-white text-xs font-extrabold cursor-pointer"
                          >
                            Inspect
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          )}

          {/* ================================================================= */}
          {/* VIEW 3: PRODUCTION AGENT REGISTRY & POLICY GOVERNANCE             */}
          {/* ================================================================= */}
          {activeNav === 'agents' && (
            <section className="pop-card rounded-2xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-3">
                <div className="flex items-center gap-2.5">
                  <span className="px-3 py-1 rounded-xl leather-badge text-xs font-extrabold">
                    AGENT REGISTRY ({registeredAgents.length} ACTIVE)
                  </span>
                  <span className="text-xs font-mono font-bold text-slate-600">
                    Multi-Agent Pipeline • Statistical &amp; Execution Policies
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {registeredAgents.map((ag) => {
                  const vRes = agentVerifyResult[ag.name];
                  return (
                    <div key={ag.id} className="pop-subcard rounded-2xl p-4 space-y-2.5 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <span className="px-2.5 py-0.5 rounded-lg bg-slate-900 text-white font-mono font-extrabold">
                          {ag.name}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded bg-[#E2E8EC] text-[#1E2E3D] font-mono font-bold text-[10px]">
                            Runs: {ag.invocations_count}
                          </span>
                          <button
                            type="button"
                            onClick={() => handleVerifyAgent(ag.name)}
                            disabled={verifyingAgent === ag.name}
                            className="pop-btn px-2.5 py-1 rounded-lg bg-[#2B4257] text-white font-extrabold text-[11px] flex items-center gap-1 cursor-pointer"
                          >
                            <ShieldCheck className="w-3 h-3" />
                            {verifyingAgent === ag.name ? 'Auditing' : 'Verify Policy'}
                          </button>
                        </div>
                      </div>

                      <div className="text-sm font-extrabold text-slate-900">{ag.role}</div>
                      <div className="text-slate-700 font-medium">{ag.description}</div>

                      <div className="bg-slate-50 border border-slate-300 rounded-xl p-2.5 space-y-1 font-mono text-[10px]">
                        <div>
                          <span className="text-slate-500">SKILL.md:</span>{' '}
                          <span className="font-bold text-[#2B4257]">{ag.skill_path}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">agent.md:</span>{' '}
                          <span className="font-bold text-[#4A2E40]">{ag.agent_md_path}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Tools:</span>{' '}
                          <span className="font-bold text-slate-900">{ag.tools_allowed.join(', ')}</span>
                        </div>
                      </div>

                      <div className="flex flex-wrap gap-1.5">
                        {ag.policies.map((pol, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-0.5 rounded bg-[#F7ECED] border border-[#DEC0C3] text-[#4D1016] font-mono text-[10px] font-bold"
                            title={pol.reason}
                          >
                            {pol.rule}: {pol.target}
                          </span>
                        ))}
                      </div>

                      {vRes && (
                        <div className="bg-[#EDF0F2] border border-[#B8C6D1] rounded-lg px-2.5 py-1.5 font-mono text-[10px] text-[#1E2E3D] font-bold">
                          Audit Status: {vRes.status.toUpperCase()} • agent.md={String(vRes.agent_md_verified)} • SKILL.md={String(vRes.skill_md_verified)}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* ================================================================= */}
          {/* VIEW 4: MAIN DISCOVERIES WORKSPACE                                */}
          {/* ================================================================= */}
          {activeNav === 'discoveries' && (
            <>
              {/* FILE UPLOAD BAR */}
              {showUploadBox && (
                <form
                  onSubmit={handleUploadSubmit}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const files = Array.from(e.dataTransfer.files || []);
                    let nextPdf = uploadPdf;
                    let nextCsv = uploadCsv;
                    for (const f of files) {
                      if (f.name.toLowerCase().endsWith('.pdf')) {
                        nextPdf = f;
                        setUploadPdf(f);
                      } else {
                        nextCsv = f;
                        setUploadCsv(f);
                      }
                    }
                    if (nextPdf || nextCsv) {
                      void executeFileUpload(nextPdf, nextCsv);
                    }
                  }}
                  className="pop-card rounded-2xl p-4 flex flex-wrap items-center justify-between gap-3 text-xs"
                >
                  <div className="flex flex-wrap items-center gap-2.5">
                    <label className="px-3.5 py-2 rounded-xl bg-[#EDF0F2] border-2 border-slate-900 font-bold cursor-pointer hover:bg-slate-200 transition-colors">
                      <Upload className="w-3.5 h-3.5 inline mr-1.5 text-[#2B4257]" />
                      <FileText className="w-3.5 h-3.5 inline mr-1.5 text-[#7A1C24]" />
                      {uploading
                        ? `Auditing ${(uploadCsv?.name || uploadPdf?.name || 'File')} across 30 Models...`
                        : uploadCsv && uploadPdf
                          ? `${uploadPdf.name} + ${uploadCsv.name}`
                          : uploadCsv
                            ? uploadCsv.name
                            : uploadPdf
                              ? uploadPdf.name
                              : 'Select or Drop CSV, TSV, Excel, or Thesis PDF'}
                      <input
                        type="file"
                        multiple
                        onChange={(e) => {
                          const files = Array.from(e.target.files || []);
                          let nextPdf = uploadPdf;
                          let nextCsv = uploadCsv;
                          for (const f of files) {
                            if (f.name.toLowerCase().endsWith('.pdf')) {
                              nextPdf = f;
                              setUploadPdf(f);
                            } else {
                              nextCsv = f;
                              setUploadCsv(f);
                            }
                          }
                          e.currentTarget.value = '';
                          if (nextPdf || nextCsv) {
                            void executeFileUpload(nextPdf, nextCsv);
                          }
                        }}
                        className="sr-only"
                      />
                    </label>
                    {(uploadPdf || uploadCsv) && (
                      <button
                        type="button"
                        onClick={() => {
                          setUploadPdf(null);
                          setUploadCsv(null);
                        }}
                        className="px-2.5 py-2 rounded-xl bg-slate-100 border border-slate-300 text-slate-600 font-bold cursor-pointer"
                      >
                        Clear
                      </button>
                    )}
                    <button
                      type="submit"
                      disabled={uploading || (!uploadPdf && !uploadCsv)}
                      className="pop-btn px-4 py-2 rounded-xl bg-[#2B4257] disabled:opacity-50 text-white font-extrabold cursor-pointer"
                    >
                      {uploading ? 'Running 30 Models...' : 'Audit File'}
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowUploadBox(false)}
                    className="p-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </form>
              )}

              {/* UPLOADED FILE 30-MODEL AUDIT RESULT BANNER */}
              {uploadedDiss && (
                <section className="pop-card-sapphire rounded-2xl p-4 space-y-3 ring-4 ring-[#2B4257]/30">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="px-3 py-1 rounded-lg bg-[#7A1C24] text-white text-xs font-extrabold font-mono">
                        UPLOADED FILE AUDITED (30 MODELS)
                      </span>
                      <span className="text-sm font-extrabold text-slate-900">
                        {uploadedDiss.title}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-slate-900 text-white text-xs font-bold">
                        {uploadedDiss.university} • {uploadedDiss.publication_year}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => handleSelectDissertation(uploadedDiss, true)}
                        className="pop-btn px-3 py-1 rounded-lg bg-[#2B4257] text-white text-xs font-extrabold cursor-pointer"
                      >
                        View 30-Model Leaderboard ↓
                      </button>
                      <button
                        type="button"
                        onClick={() => setUploadedDiss(null)}
                        className="p-1 rounded-lg bg-slate-200 hover:bg-slate-300 text-slate-800 cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {(() => {
                    const uDs = uploadedDiss.datasets?.[0];
                    const uRun = uDs?.lazy_runs?.[0];
                    const uOpp = uploadedDiss.opportunities?.[0];
                    const uFeat =
                      uRun?.top_features?.find((f) => !f.discussed_by_author) ||
                      uRun?.top_features?.[0];
                    return (
                      <>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono">
                        <div className="bg-white border-2 border-slate-900 rounded-xl p-2.5">
                          <div className="text-[10px] font-extrabold text-slate-500 uppercase">
                            TARGET AUDITED
                          </div>
                          <div className="font-extrabold text-slate-900 truncate mt-0.5">
                            {uRun?.target_column || 'Auto-Parsed'}
                          </div>
                          <div className="text-[10px] text-slate-600">
                            {uDs ? `${uDs.row_count} rows × ${uDs.col_count} cols` : '30 Models'}
                          </div>
                        </div>

                        <div className="bg-[#EDF0F2] border-2 border-slate-900 rounded-xl p-2.5">
                          <div className="text-[10px] font-extrabold text-[#2B4257] uppercase">
                            WINNING MODEL
                          </div>
                          <div className="font-extrabold text-[#1E2E3D] truncate mt-0.5">
                            {uRun ? `${uRun.best_model_name} (${uRun.best_score.toFixed(4)})` : 'Evaluated'}
                          </div>
                          <div className="text-[10px] font-bold text-[#7A1C24]">
                            {uRun
                              ? `+${(uRun.signal_gain_delta * 100).toFixed(1)}% vs ${uRun.author_baseline_score.toFixed(4)} Baseline`
                              : ''}
                          </div>
                        </div>

                        <div className="bg-[#F7ECED] border-2 border-slate-900 rounded-xl p-2.5">
                          <div className="text-[10px] font-extrabold text-[#7A1C24] uppercase">
                            OVERLOOKED CLUE
                          </div>
                          <div className="font-extrabold text-rose-950 truncate mt-0.5">
                            {uFeat?.feature || 'Primary Factor'}
                          </div>
                          <div className="text-[10px] font-bold text-[#7A1C24]">
                            {uFeat ? `${(uFeat.importance * 100).toFixed(1)}% Importance` : ''}
                          </div>
                        </div>

                        <div className="bg-[#F3EDF1] border-2 border-slate-900 rounded-xl p-2.5">
                          <div className="text-[10px] font-extrabold text-[#4A2E40] uppercase">
                            OPPORTUNITY SCORE
                          </div>
                          <div className="font-extrabold text-[#2D1B27] truncate mt-0.5">
                            {uOpp ? `TOS ${uOpp.overall_tos_score} / 100` : 'Synthesized'}
                          </div>
                          <div className="text-[10px] font-sans font-bold text-slate-700 truncate">
                            {uOpp?.title || ''}
                          </div>
                        </div>
                      </div>
                      {(() => {
                        const cSummary = (uDs?.column_metadata?.[0] as any)?.cleansing_summary;
                        if (!cSummary?.actions?.length) return null;
                        return (
                          <div className="bg-white/80 border border-slate-300 rounded-xl px-3 py-2 text-[11px] font-mono text-slate-800 space-y-1">
                            <div className="font-extrabold text-[#2B4257] uppercase text-[10px]">
                              DATA CLEANSING AUDIT (data-cleansing-agent • {cSummary.detected_format || 'Tabular'})
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              {cSummary.actions.map((act: string, i: number) => (
                                <span
                                  key={i}
                                  className="px-2 py-0.5 rounded bg-[#EDF0F2] border border-[#B8C6D1] text-[#1E2E3D] font-bold"
                                >
                                  {act}
                                </span>
                              ))}
                            </div>
                          </div>
                        );
                      })()}
                    </>
                    );
                  })()}
                </section>
              )}

              {/* GLOBAL SEARCH & DYNAMIC DOMAIN FILTER BAR */}
              <section className="pop-card rounded-2xl p-4 space-y-3">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleRunGlobalSearch();
                  }}
                  className="flex flex-col sm:flex-row gap-2"
                >
                  <input
                    type="text"
                    value={searchInput}
                    onChange={(e) => setSearchInput(e.target.value)}
                    placeholder="Global search: live OpenAlex, EuropePMC, arXiv, and Zenodo datasets"
                    className="flex-1 px-3.5 py-2 rounded-xl bg-slate-50 border-2 border-slate-900 text-xs sm:text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:bg-white font-bold"
                  />
                  <button
                    type="submit"
                    disabled={harvesting || !searchInput.trim()}
                    className="pop-btn px-4 py-2 rounded-xl bg-[#2B4257] hover:bg-[#1F3142] disabled:opacity-50 text-white font-extrabold text-xs flex items-center justify-center gap-1.5 cursor-pointer shrink-0"
                  >
                    {harvesting ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        Pulling Live
                      </>
                    ) : (
                      <>
                        <Search className="w-3.5 h-3.5 stroke-[2.5]" />
                        Pull Global
                      </>
                    )}
                  </button>
                  {(domainFilter !== 'all' || streamEvents.length > 0 || searchInput) && (
                    <button
                      type="button"
                      onClick={handleResetToGlobalView}
                      className="pop-btn px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-900 font-extrabold text-xs flex items-center justify-center gap-1 cursor-pointer shrink-0"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                      Reset
                    </button>
                  )}
                </form>

                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-1.5">
                    {Object.entries(DOMAIN_NAMES).map(([key, label]) => {
                      const cnt = domainCounts[key] || 0;
                      if (key !== 'all' && cnt === 0) return null;
                      const active = domainFilter === key;
                      return (
                        <button
                          key={key}
                          type="button"
                          onClick={() => setDomainFilter(key)}
                          className={`px-2.5 py-1 rounded-lg text-[11px] font-extrabold border-2 transition cursor-pointer ${
                            active
                              ? 'bg-slate-900 text-white border-slate-900'
                              : 'bg-white text-slate-700 border-slate-300 hover:border-slate-900'
                          }`}
                        >
                          {label} ({cnt})
                        </button>
                      );
                    })}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {[
                      'Oncology Biomarkers',
                      'Battery Degradation',
                      'Semiconductor Telemetry',
                      'Ocean Carbon Flux',
                    ].map((topic) => (
                      <button
                        key={topic}
                        type="button"
                        disabled={harvesting}
                        onClick={() => handleRunGlobalSearch(topic)}
                        className="px-2.5 py-1 rounded-lg bg-[#EDF0F2] hover:bg-[#E2E8EC] border border-[#B8C6D1] text-[#2B4257] text-[11px] font-bold cursor-pointer"
                      >
                        {topic}
                      </button>
                    ))}
                  </div>
                </div>

                {(harvesting || streamEvents.length > 0) && (
                  <div className="bg-slate-900 text-white rounded-xl px-3.5 py-2.5 border-2 border-slate-900 flex items-center justify-between gap-3 font-mono text-[11px]">
                    <span className="text-[#D8B99C] font-bold truncate">
                      {streamEvents[streamEvents.length - 1] || 'Running global sweep'}
                    </span>
                    {!harvesting && (
                      <button
                        type="button"
                        onClick={() => setStreamEvents([])}
                        className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-sans text-[10px] shrink-0 cursor-pointer"
                      >
                        Dismiss
                      </button>
                    )}
                  </div>
                )}
              </section>

              {/* SECTION 01: REVEALED DISSERTATIONS & HIGH-VALUE OPPORTUNITIES */}
              <section className="space-y-3">
                <div className="flex items-center justify-between px-1">
                  <span className="px-3 py-1 rounded-xl leather-badge text-xs font-extrabold">
                    01. REVEALED DISSERTATIONS &amp; HIGH-VALUE OPPORTUNITIES
                  </span>
                  <span className="text-xs font-mono font-bold text-slate-700">
                    {visibleDissertations.length} PULLED
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {visibleDissertations.map((diss) => {
                    const isSelected = selectedDiss?.id === diss.id;
                    const ds = diss.datasets?.[0];
                    const run = ds?.lazy_runs?.[0];
                    const opp =
                      opportunities.find((o) => o.dissertation_id === diss.id) ||
                      diss.opportunities?.[0];
                    const topOverlooked =
                      run?.top_features?.find((f) => !f.discussed_by_author) ||
                      run?.top_features?.[0];
                    const savedState = savedByDissId(diss.id);

                    return (
                      <div
                        key={diss.id}
                        onClick={() => handleSelectDissertation(diss, true)}
                        className={`rounded-2xl p-4 transition cursor-pointer flex flex-col justify-between gap-3 ${
                          isSelected
                            ? 'pop-card-sapphire ring-4 ring-[#2B4257]/25'
                            : 'pop-card hover:translate-y-[-2px]'
                        }`}
                      >
                        <div className="space-y-2.5">
                          <div className="flex items-center justify-between gap-2">
                            <span className="px-2.5 py-0.5 rounded-lg bg-slate-900 text-white text-xs font-extrabold truncate max-w-[55%]">
                              {diss.university} • {diss.publication_year}
                            </span>
                            <div className="flex items-center gap-1.5 shrink-0">
                              {run && (
                                <span className="px-2 py-0.5 rounded-lg bg-[#2B4257] text-white text-[11px] font-extrabold font-mono">
                                  +{(run.signal_gain_delta * 100).toFixed(1)}% Gain
                                </span>
                              )}
                              {opp && (
                                <span className="px-2 py-0.5 rounded-lg bg-[#7A1C24] text-white text-[11px] font-extrabold font-mono">
                                  TOS {opp.overall_tos_score}
                                </span>
                              )}
                              <button
                                type="button"
                                onClick={(e) => handleToggleSave(e, diss.id, opp?.id, 'star')}
                                className={`p-1 rounded-lg border transition cursor-pointer ${
                                  savedState?.is_starred
                                    ? 'bg-[#7A1C24] text-white border-slate-900'
                                    : 'bg-white text-slate-500 border-slate-300 hover:text-[#7A1C24]'
                                }`}
                                title="Favorite"
                              >
                                <Star className="w-3.5 h-3.5 fill-current" />
                              </button>
                              <button
                                type="button"
                                onClick={(e) => handleToggleSave(e, diss.id, opp?.id, 'ribbon')}
                                className={`p-1 rounded-lg border transition cursor-pointer ${
                                  savedState?.is_bookmarked
                                    ? 'bg-[#2B4257] text-white border-slate-900'
                                    : 'bg-white text-slate-500 border-slate-300 hover:text-[#2B4257]'
                                }`}
                                title="Bookmark"
                              >
                                <Bookmark className="w-3.5 h-3.5 fill-current" />
                              </button>
                            </div>
                          </div>

                          <div className="bg-slate-50 border-2 border-slate-900 rounded-xl p-3 space-y-1.5">
                            <div className="flex items-center justify-between text-[10px] font-extrabold uppercase tracking-wider text-slate-500">
                              <span className="truncate max-w-[50%]">
                                THESIS • {diss.subdomain || diss.domain}
                              </span>
                              {run && (
                                <span className="font-mono text-[#2B4257]">
                                  {run.author_baseline_score.toFixed(2)} → {run.best_score.toFixed(2)} ({run.best_model_name})
                                </span>
                              )}
                            </div>
                            <div className="text-sm font-extrabold text-slate-900 leading-snug">
                              {diss.title}
                            </div>
                            {topOverlooked && (
                              <div className="flex flex-wrap items-center gap-1.5 pt-0.5 text-[11px] font-mono">
                                <span className="px-1.5 py-0.5 rounded bg-[#7A1C24] text-white font-bold">
                                  Overlooked
                                </span>
                                <span className="font-bold text-slate-900">
                                  {topOverlooked.feature}
                                </span>
                                <span className="text-[#7A1C24] font-bold">
                                  ({(topOverlooked.importance * 100).toFixed(1)}%)
                                </span>
                              </div>
                            )}
                          </div>

                          {opp && (
                            <div className="bg-[#EDF0F2]/90 border-2 border-slate-900 rounded-xl p-3 space-y-1.5">
                              <div className="flex items-center justify-between text-[10px] font-extrabold uppercase tracking-wider text-[#2B4257]">
                                <span className="flex items-center gap-1">
                                  <Sparkles className="w-3 h-3 text-[#7A1C24]" />
                                  OPPORTUNITY • {opp.opportunity_type}
                                </span>
                              </div>
                              <div className="text-sm font-extrabold text-slate-900 leading-snug">
                                {opp.title}
                              </div>
                              <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-bold">
                                {opp.opentargets_chembl_check?.tractability_modality && (
                                  <span className="px-2 py-0.5 rounded bg-white border border-slate-300 text-[#382130] truncate max-w-full">
                                    {opp.opentargets_chembl_check.tractability_modality}
                                  </span>
                                )}
                                {opp.clinical_trials_check?.whitespace_status && (
                                  <span className="px-2 py-0.5 rounded bg-[#EFE0E2] text-[#4D1016] truncate max-w-full">
                                    {opp.clinical_trials_check.whitespace_status}
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center justify-between pt-1.5 border-t border-slate-200 text-xs">
                          <span
                            className={`font-extrabold px-2.5 py-0.5 rounded-lg text-[11px] ${
                              isSelected
                                ? 'bg-[#2B4257] text-white'
                                : 'bg-slate-100 text-slate-800'
                            }`}
                          >
                            {isSelected ? 'Selected' : 'Inspect'}
                          </span>

                          <div className="flex items-center gap-3" onClick={(e) => e.stopPropagation()}>
                            {diss.source_url && (
                              <a
                                href={diss.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="font-bold text-[#2B4257] hover:underline inline-flex items-center gap-0.5 text-[11px]"
                              >
                                Thesis <ArrowUpRight className="w-3 h-3" />
                              </a>
                            )}
                            {ds?.source_url && (
                              <a
                                href={ds.source_url}
                                target="_blank"
                                rel="noreferrer"
                                className="font-bold text-[#4A2E40] hover:underline inline-flex items-center gap-0.5 text-[11px]"
                              >
                                Zenodo CSV <ArrowUpRight className="w-3 h-3" />
                              </a>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>

              {/* SECTION 02: ACTIVE OPPORTUNITY DOSSIER & ACTIONS */}
              <div id="active-finding-inspector" className="scroll-mt-16 space-y-5">
                {activeOpportunities.length > 0 && (
                  <section>
                    {activeOpportunities.slice(0, 1).map((opp) => {
                      const savedState = selectedDiss ? savedByDissId(selectedDiss.id) : undefined;
                      return (
                        <div key={opp.id} className="pop-card-sapphire rounded-2xl p-5 space-y-4">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="px-3 py-1 rounded-xl leather-badge text-xs font-extrabold">
                                02. OPPORTUNITY DOSSIER
                              </span>
                              <span className="px-2.5 py-1 rounded-lg bg-slate-900 text-white text-xs font-extrabold">
                                {selectedDiss?.university} • {selectedDiss?.publication_year}
                              </span>
                              <span className="px-2.5 py-1 rounded-lg bg-[#EDF0F2] text-[#1E2E3D] border border-slate-900 text-xs font-extrabold">
                                {opp.opportunity_type}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              {selectedDiss && (
                                <>
                                  <button
                                    type="button"
                                    onClick={(e) => handleToggleSave(e, selectedDiss.id, opp.id, 'star')}
                                    className={`px-2.5 py-1 rounded-xl border-2 text-xs font-extrabold flex items-center gap-1 cursor-pointer ${
                                      savedState?.is_starred
                                        ? 'bg-[#7A1C24] text-white border-slate-900'
                                        : 'bg-white text-slate-800 border-slate-900'
                                    }`}
                                  >
                                    <Star className="w-3.5 h-3.5 fill-current" />
                                    {savedState?.is_starred ? 'Starred' : 'Star'}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => handleToggleSave(e, selectedDiss.id, opp.id, 'ribbon')}
                                    className={`px-2.5 py-1 rounded-xl border-2 text-xs font-extrabold flex items-center gap-1 cursor-pointer ${
                                      savedState?.is_bookmarked
                                        ? 'bg-[#2B4257] text-white border-slate-900'
                                        : 'bg-white text-slate-800 border-slate-900'
                                    }`}
                                  >
                                    <Bookmark className="w-3.5 h-3.5 fill-current" />
                                    {savedState?.is_bookmarked ? 'Bookmarked' : 'Bookmark'}
                                  </button>
                                </>
                              )}
                              <div className="px-3 py-1 rounded-xl bg-[#7A1C24] text-white border-2 border-slate-900 text-xs font-extrabold font-mono">
                                TOS {opp.overall_tos_score} / 100
                              </div>
                            </div>
                          </div>

                          <div className="flex flex-wrap items-baseline justify-between gap-2">
                            <h2 className="text-lg sm:text-xl font-extrabold text-slate-900">
                              {opp.title}
                            </h2>
                            <span className="text-xs font-extrabold text-[#2B4257]">
                              {opp.proposed_product_or_therapeutic}
                            </span>
                          </div>

                          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                            <div className="pop-subcard rounded-xl p-3 space-y-1">
                              <div className="text-[10px] font-extrabold text-[#2B4257] uppercase">
                                Overlooked Clue
                              </div>
                              <div className="font-mono font-bold text-slate-900 text-[11px]">
                                {opp.untranslated_finding_summary}
                              </div>
                            </div>

                            <div className="pop-subcard rounded-xl p-3 space-y-1">
                              <div className="text-[10px] font-extrabold text-[#4A2E40] uppercase">
                                Thesis Blindspot
                              </div>
                              <div className="font-bold text-slate-900 text-[11px]">
                                {opp.valley_of_death_diagnosis}
                              </div>
                            </div>

                            <div className="pop-subcard rounded-xl p-3 space-y-1">
                              <div className="text-[10px] font-extrabold text-[#7A1C24] uppercase">
                                Registry Whitespace
                              </div>
                              <div className="font-bold text-slate-900 text-[11px]">
                                {opp.clinical_trials_check?.whitespace_status}
                              </div>
                            </div>

                            <div className="pop-subcard rounded-xl p-3 space-y-1">
                              <div className="text-[10px] font-extrabold text-slate-600 uppercase">
                                Modality &amp; Validation
                              </div>
                              <div className="font-bold text-slate-900 text-[11px]">
                                {opp.opentargets_chembl_check?.tractability_modality}
                              </div>
                            </div>
                          </div>

                          <div className="space-y-3 pt-2 border-t border-slate-200">
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                              {[
                                '1. 5-Fold CV Proof',
                                '2. Patent Claim',
                                '3. Registry Audit',
                                '4. Export .py & .md',
                              ].map((label, i) => {
                                const key = `${opp.id}_${i}`;
                                const isRunning = executingStepKey === key;
                                const hasDone = Boolean(stepActionResults[key]);
                                return (
                                  <button
                                    key={i}
                                    onClick={() => handleExecuteOpportunityStep(opp.id, i)}
                                    disabled={isRunning}
                                    className={`pop-btn px-3 py-2 rounded-xl font-extrabold text-xs flex items-center justify-center gap-1.5 cursor-pointer ${
                                      hasDone
                                        ? 'bg-[#4A2E40] text-white'
                                        : 'bg-[#2B4257] hover:bg-[#1F3142] text-white'
                                    }`}
                                  >
                                    {isRunning ? (
                                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                    ) : (
                                      <Play className="w-3.5 h-3.5" />
                                    )}
                                    {label}
                                  </button>
                                );
                              })}
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                              {[0, 1, 2, 3].map((i) => {
                                const key = `${opp.id}_${i}`;
                                const stepRes = stepActionResults[key];
                                if (!stepRes) return null;
                                return (
                                  <div
                                    key={key}
                                    className="bg-slate-50 border-2 border-slate-900 rounded-xl p-3 space-y-1.5 text-[11px]"
                                  >
                                    <div className="font-extrabold text-[#1E2E3D]">
                                      {stepRes.action_title}
                                    </div>
                                    {stepRes.metrics && (
                                      <div className="flex flex-wrap gap-1.5 font-mono">
                                        {Object.entries(stepRes.metrics).map(([k, v]) => (
                                          <span
                                            key={k}
                                            className="px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-900"
                                          >
                                            {k}: <strong>{String(v)}</strong>
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                    {stepRes.provisional_patent_claim && (
                                      <div className="bg-white border border-slate-300 rounded-lg p-2 font-mono text-[10px] text-slate-900">
                                        {stepRes.provisional_patent_claim}
                                      </div>
                                    )}
                                    {stepRes.live_registry_checks && (
                                      <div className="flex flex-wrap gap-1.5 font-mono">
                                        {Object.entries(stepRes.live_registry_checks).map(([k, v]) => (
                                          <span
                                            key={k}
                                            className="px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-900"
                                          >
                                            {k}: <strong>{String(v)}</strong>
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                    {stepRes.reproducible_python_code && (
                                      <div className="flex flex-wrap gap-2 pt-0.5">
                                        <button
                                          onClick={() =>
                                            downloadTextFile(
                                              `thesisout_model_opp_${opp.id}.py`,
                                              stepRes.reproducible_python_code
                                            )
                                          }
                                          className="pop-btn px-3 py-1 rounded-lg bg-[#2B4257] text-white font-bold inline-flex items-center gap-1 cursor-pointer"
                                        >
                                          <Download className="w-3 h-3" /> Model (.py)
                                        </button>
                                        <button
                                          onClick={() =>
                                            downloadTextFile(
                                              `thesisout_dossier_opp_${opp.id}.md`,
                                              stepRes.venture_dossier_markdown
                                            )
                                          }
                                          className="pop-btn px-3 py-1 rounded-lg bg-[#4A2E40] text-white font-bold inline-flex items-center gap-1 cursor-pointer"
                                        >
                                          <Download className="w-3 h-3" /> Dossier (.md)
                                        </button>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </section>
                )}

                {/* SECTION 03: 30-MODEL EMPIRICAL CLUE BOARD */}
                {activeRun && (
                  <section className="pop-card rounded-2xl p-5 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="px-3 py-1 rounded-xl leather-badge text-xs font-extrabold">
                          03. 30-MODEL LEADERBOARD
                        </span>
                        <code className="px-2 py-0.5 rounded bg-slate-200 text-slate-900 text-xs font-bold">
                          Target: {activeRun.target_column}
                        </code>
                      </div>

                      {activeDataset && (
                        <div className="flex flex-wrap items-center gap-1 text-xs">
                          <span className="font-bold text-slate-500 mr-1">Target:</span>
                          {activeDataset.column_metadata
                            .filter((c) => c.role !== 'identifier')
                            .slice(0, 5)
                            .map((col) => (
                              <button
                                key={col.name}
                                onClick={() => handleRunCustomTargetColumn(col.name)}
                                disabled={runningCustomTarget}
                                className={`px-2 py-0.5 rounded-lg font-mono text-[11px] font-bold border cursor-pointer ${
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

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div className="pop-subcard rounded-xl p-3.5">
                        <div className="text-[10px] font-extrabold text-slate-500 uppercase">
                          Author Baseline ({activeRun.primary_metric_name})
                        </div>
                        <div className="text-2xl font-extrabold font-mono text-slate-900 mt-0.5">
                          {activeRun.author_baseline_score.toFixed(4)}
                        </div>
                        <div className="text-[11px] font-bold text-slate-500 mt-0.5">
                          {selectedDiss?.author_Statistical_method || 'Linear Fit'}
                        </div>
                      </div>

                      <div className="bg-[#EDF0F2] border-2 border-slate-900 rounded-xl p-3.5 shadow-[3px_3px_0px_#2B4257]">
                        <div className="text-[10px] font-extrabold text-[#2B4257] uppercase">
                          Best Model • {activeRun.best_model_name}
                        </div>
                        <div className="text-2xl font-extrabold font-mono text-[#2B4257] mt-0.5 flex items-baseline gap-2">
                          <span>{activeRun.best_score.toFixed(4)}</span>
                          <span className="text-[11px] font-extrabold px-2 py-0.5 rounded-full bg-[#7A1C24] text-white">
                            +{(activeRun.signal_gain_delta * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-[11px] font-bold text-slate-600 mt-0.5">
                          5-Fold Out-of-Fold {activeRun.primary_metric_name}
                        </div>
                      </div>

                      <div className="pop-subcard rounded-xl p-3.5 flex items-center justify-between gap-2">
                        <div>
                          <div className="text-[10px] font-extrabold text-[#4A2E40] uppercase">
                            5-Stage Super-Learner
                          </div>
                          <div className="text-2xl font-extrabold font-mono text-[#4A2E40] mt-0.5">
                            {escalationCache[activeRun.id]
                              ? escalationCache[activeRun.id].stage2_super_learner_ensemble.super_learner_score.toFixed(4)
                              : 'Ready'}
                          </div>
                          <div className="text-[11px] font-bold text-slate-500 mt-0.5">
                            {escalationCache[activeRun.id] ? '0.00% Leakage Verified' : 'Permutation + Stacking'}
                          </div>
                        </div>
                        {!escalationCache[activeRun.id] && (
                          <button
                            onClick={() => handleEscalateDeepInference(activeRun.id)}
                            disabled={escalatingRunId === activeRun.id}
                            className="pop-btn px-3 py-1.5 rounded-xl bg-[#4A2E40] text-white text-xs font-bold cursor-pointer shrink-0"
                          >
                            {escalatingRunId === activeRun.id ? 'Running' : 'Verify'}
                          </button>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                      {activeRun.top_features.slice(0, 4).map((feat, idx) => (
                        <div key={idx} className="pop-subcard rounded-xl p-2.5 space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-mono font-extrabold text-slate-900 truncate">
                              {feat.feature}
                            </span>
                            <div className="flex items-center gap-1.5 shrink-0">
                              {!feat.discussed_by_author && (
                                <span className="px-1.5 py-0.5 rounded bg-[#7A1C24] text-white text-[10px] font-extrabold">
                                  Overlooked
                                </span>
                              )}
                              <span className="font-mono font-extrabold text-[#2B4257]">
                                {(feat.importance * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden border border-slate-300">
                            <div
                              className={`h-full rounded-full ${
                                !feat.discussed_by_author ? 'bg-[#7A1C24]' : 'bg-[#2B4257]'
                              }`}
                              style={{ width: `${Math.min(100, Math.max(8, feat.importance * 100))}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="space-y-2 pt-1 border-t border-slate-200">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-extrabold text-slate-900 uppercase">
                          {activeRun.leaderboard.length} Cross-Validated Models
                        </span>
                        <button
                          onClick={() => setShowAllModels(!showAllModels)}
                          className="px-2.5 py-1 rounded-lg bg-slate-100 border border-slate-300 text-xs font-bold text-slate-800 flex items-center gap-1 cursor-pointer"
                        >
                          {showAllModels ? (
                            <>
                              Hide <ChevronUp className="w-3.5 h-3.5" />
                            </>
                          ) : (
                            <>
                              Show {activeRun.leaderboard.length} Models <ChevronDown className="w-3.5 h-3.5" />
                            </>
                          )}
                        </button>
                      </div>

                      {showAllModels && (
                        <div className="overflow-x-auto max-h-64 rounded-xl border-2 border-slate-900">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-900 text-white font-bold">
                                <th className="py-2 px-3">#</th>
                                {Object.keys(activeRun.leaderboard[0] || {}).map((col) => (
                                  <th key={col} className="py-2 px-3">
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
                                      ? 'bg-[#EDF0F2] font-extrabold text-[#1E2E3D]'
                                      : 'text-slate-800 hover:bg-slate-50'
                                  }
                                >
                                  <td className="py-1.5 px-3">#{idx + 1}</td>
                                  {Object.entries(row).map(([k, val]) => (
                                    <td key={k} className="py-1.5 px-3">
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

                {/* SECTION 04: SOURCE THESIS & DATASET METADATA */}
                {selectedDiss && (
                  <section className="pop-card rounded-2xl p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="px-3 py-1 rounded-xl leather-badge text-xs font-extrabold">
                        04. SOURCE THESIS &amp; DATASET
                      </span>
                      <span className="text-xs font-mono font-bold text-slate-600">
                        {selectedDiss.source_repo}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                      <div className="pop-subcard rounded-xl p-3.5 flex flex-col justify-between gap-2">
                        <div className="space-y-1">
                          <div className="text-[10px] font-extrabold text-[#2B4257] uppercase">
                            {selectedDiss.university} ({selectedDiss.publication_year}) • {selectedDiss.authors}
                          </div>
                          <div className="text-sm font-extrabold text-slate-900">
                            {selectedDiss.title}
                          </div>
                          <div className="font-mono text-[11px] text-slate-600">
                            Baseline: {selectedDiss.author_Statistical_method} ({selectedDiss.author_reported_baseline_score?.toFixed(4)})
                          </div>
                        </div>
                        {selectedDiss.source_url && (
                          <a
                            href={selectedDiss.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 font-extrabold text-[#2B4257] hover:underline"
                          >
                            <BookOpen className="w-3.5 h-3.5" /> Open Repository{' '}
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                      </div>

                      {activeDataset && (
                        <div className="pop-subcard rounded-xl p-3.5 flex flex-col justify-between gap-2">
                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-extrabold text-[#4A2E40] uppercase">
                                {activeDataset.name}
                              </span>
                              <span className="font-mono font-bold text-slate-700">
                                {activeDataset.row_count} × {activeDataset.col_count}
                              </span>
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {activeDataset.column_metadata.map((c) => (
                                <span
                                  key={c.name}
                                  className="px-2 py-0.5 rounded bg-slate-100 border border-slate-300 font-mono text-[10px] text-slate-800"
                                >
                                  {c.name}
                                </span>
                              ))}
                            </div>
                          </div>
                          {activeDataset.source_url && (
                            <a
                              href={activeDataset.source_url}
                              target="_blank"
                              rel="noreferrer"
                              className="inline-flex items-center gap-1 font-extrabold text-[#4A2E40] hover:underline"
                            >
                              <CheckCircle2 className="w-3.5 h-3.5" /> Open Zenodo CSV{' '}
                              <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  </section>
                )}
              </div>
            </>
          )}
        </main>
      )}
    </div>
  );
}

export default App;
