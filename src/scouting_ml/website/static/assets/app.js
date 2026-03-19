const numberFmt = new Intl.NumberFormat("en-GB");
const currencyFmt = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 0,
});
const pctFmt = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const decimalFmt = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const DEFAULT_API = localStorage.getItem("scoutml_api_base") || "http://localhost:8000";
const BIG5_LEAGUES = new Set([
  "english premier league",
  "premier league",
  "laliga",
  "la liga",
  "serie a",
  "bundesliga",
  "ligue 1",
]);

const state = {
  apiBase: DEFAULT_API,
  connected: false,
  loading: false,

  view: "overview",

  mode: "shortlist",
  split: "test",
  season: "",
  league: "",
  position: "",
  roleNeed: "",
  search: "",
  minMinutes: 900,
  minAge: 18,
  maxAge: 23,
  budgetBand: "10000000",
  maxContractYearsLeft: 2,
  minConfidence: 0.5,
  minGapEur: 1_000_000,
  nonBig5Only: true,
  undervaluedOnly: true,
  sortBy: "value_gap_conservative_eur",
  sortOrder: "desc",
  limit: 50,
  offset: 0,
  shortlistTopN: 100,

  rows: [],
  total: 0,
  count: 0,
  selectedRow: null,
  selectedProfile: null,
  selectedReport: null,
  selectedHistory: null,
  activeDetailTab: "overview",
  profileModalOpen: false,
  detailRequestId: 0,
  reportCache: new Map(),
  profileCache: new Map(),

  health: null,
  metrics: null,
  modelManifest: null,
  benchmark: null,
  activeArtifacts: null,
  coverageRows: [],
  queryDiagnostics: null,
  funnelDiagnostics: null,

  funnelRows: [],
  funnelTopRows: [],
  watchlistRows: [],
  watchlistTotal: 0,
};

const el = {
  apiBase: document.getElementById("api-base"),
  connectBtn: document.getElementById("connect-btn"),
  apiStatus: document.getElementById("api-status"),

  tabButtons: Array.from(document.querySelectorAll(".tab-btn")),
  views: {
    overview: document.getElementById("view-overview"),
    workbench: document.getElementById("view-workbench"),
    funnel: document.getElementById("view-funnel"),
  },

  overviewPosturePill: document.getElementById("overview-posture-pill"),
  overviewPostureTitle: document.getElementById("overview-posture-title"),
  overviewPostureCopy: document.getElementById("overview-posture-copy"),
  overviewPricingStatus: document.getElementById("overview-pricing-status"),
  overviewPricingCopy: document.getElementById("overview-pricing-copy"),
  overviewRankingStatus: document.getElementById("overview-ranking-status"),
  overviewRankingCopy: document.getElementById("overview-ranking-copy"),
  overviewCoverageStatus: document.getElementById("overview-coverage-status"),
  overviewCoverageCopy: document.getElementById("overview-coverage-copy"),

  trustModelVersion: document.getElementById("trust-model-version"),
  trustUpdated: document.getElementById("trust-updated"),
  trustDataset: document.getElementById("trust-dataset"),
  trustSplits: document.getElementById("trust-splits"),
  trustRows: document.getElementById("trust-rows"),
  artifactTest: document.getElementById("artifact-test"),
  artifactVal: document.getElementById("artifact-val"),
  artifactMetrics: document.getElementById("artifact-metrics"),
  championValuation: document.getElementById("champion-valuation"),
  championShortlist: document.getElementById("champion-shortlist"),
  championRuntime: document.getElementById("champion-runtime"),
  trustNote: document.getElementById("trust-note"),

  metricTestR2: document.getElementById("metric-test-r2"),
  metricTestMae: document.getElementById("metric-test-mae"),
  metricTestMape: document.getElementById("metric-test-mape"),
  metricValR2: document.getElementById("metric-val-r2"),
  metricValMae: document.getElementById("metric-val-mae"),
  metricValMape: document.getElementById("metric-val-mape"),
  metricsMeta: document.getElementById("metrics-meta"),

  segmentBody: document.getElementById("segment-body"),
  segmentWarning: document.getElementById("segment-warning"),
  coverageBody: document.getElementById("coverage-body"),
  benchmarkMeta: document.getElementById("benchmark-meta"),
  benchmarkBody: document.getElementById("benchmark-body"),
  experimentBestOverall: document.getElementById("experiment-best-overall"),
  experimentBestCheap: document.getElementById("experiment-best-cheap"),
  experimentOnboarding: document.getElementById("experiment-onboarding"),
  experimentWeakest: document.getElementById("experiment-weakest"),

  mode: document.getElementById("mode-select"),
  split: document.getElementById("split-select"),
  season: document.getElementById("season-select"),
  league: document.getElementById("league-select"),
  position: document.getElementById("position-select"),
  roleNeed: document.getElementById("role-need-select"),
  search: document.getElementById("search-input"),
  minMinutes: document.getElementById("min-minutes"),
  minAge: document.getElementById("min-age"),
  maxAge: document.getElementById("max-age"),
  budgetBand: document.getElementById("budget-band"),
  maxContractYears: document.getElementById("max-contract-years"),
  minConfidence: document.getElementById("min-confidence"),
  minGap: document.getElementById("min-gap"),
  topN: document.getElementById("top-n"),
  sort: document.getElementById("sort-select"),
  sortDir: document.getElementById("sort-direction"),
  limit: document.getElementById("limit-input"),
  outsideBig5Only: document.getElementById("outside-big5-only"),
  undervaluedOnly: document.getElementById("undervalued-only"),
  refresh: document.getElementById("refresh-btn"),
  reset: document.getElementById("reset-btn"),
  exportBtn: document.getElementById("export-btn"),
  exportPackBtn: document.getElementById("export-pack-btn"),

  title: document.getElementById("results-title"),
  resultCount: document.getElementById("result-count"),
  resultRange: document.getElementById("result-range"),
  resultsNote: document.getElementById("results-note"),
  tbody: document.getElementById("results-body"),
  prevBtn: document.getElementById("prev-btn"),
  nextBtn: document.getElementById("next-btn"),

  detailPlaceholder: document.getElementById("detail-placeholder"),
  detailContent: document.getElementById("detail-content"),
  detailOpenProfile: document.getElementById("detail-open-profile"),
  detailTabButtons: Array.from(document.querySelectorAll("[data-detail-tab]")),
  detailPanels: Array.from(document.querySelectorAll("[data-detail-panel]")),
  detailDecisionPill: document.getElementById("detail-decision-pill"),
  detailDecisionNext: document.getElementById("detail-decision-next"),
  detailDecisionReason: document.getElementById("detail-decision-reason"),
  detailName: document.getElementById("detail-name"),
  detailMeta: document.getElementById("detail-meta"),
  detailGap: document.getElementById("detail-gap"),
  detailGapNote: document.getElementById("detail-gap-note"),
  detailConfidenceScore: document.getElementById("detail-confidence-score"),
  detailConfidenceNote: document.getElementById("detail-confidence-note"),
  detailPriceStance: document.getElementById("detail-price-stance"),
  detailPriceContext: document.getElementById("detail-price-context"),
  detailMarket: document.getElementById("detail-market"),
  detailExpected: document.getElementById("detail-expected"),
  detailLower: document.getElementById("detail-lower"),
  detailBadges: document.getElementById("detail-badges"),
  detailScoreDriver: document.getElementById("detail-score-driver"),
  detailWhyRanked: document.getElementById("detail-why-ranked"),
  detailCoverageSummary: document.getElementById("detail-coverage-summary"),
  detailCoverageList: document.getElementById("detail-coverage-list"),
  detailList: document.getElementById("detail-list"),
  detailSummary: document.getElementById("detail-summary"),
  detailRoleSummary: document.getElementById("detail-role-summary"),
  detailRoleMetrics: document.getElementById("detail-role-metrics"),
  detailStrengths: document.getElementById("detail-strengths"),
  detailWeaknesses: document.getElementById("detail-weaknesses"),
  detailLevers: document.getElementById("detail-levers"),
  detailHistory: document.getElementById("detail-history"),
  detailStatsSummary: document.getElementById("detail-stats-summary"),
  detailStatGroups: document.getElementById("detail-stat-groups"),
  detailArchetype: document.getElementById("detail-archetype"),
  detailArchetypeCandidates: document.getElementById("detail-archetype-candidates"),
  detailFormationSummary: document.getElementById("detail-formation-summary"),
  detailFormations: document.getElementById("detail-formations"),
  detailProviderTacticalSummary: document.getElementById("detail-provider-tactical-summary"),
  detailProviderTactical: document.getElementById("detail-provider-tactical"),
  detailSimilarSummary: document.getElementById("detail-similar-summary"),
  detailSimilar: document.getElementById("detail-similar"),
  detailRadar: document.getElementById("detail-radar"),
  detailRadarMeta: document.getElementById("detail-radar-meta"),
  detailConfidence: document.getElementById("detail-confidence"),
  detailRisks: document.getElementById("detail-risks"),
  detailAvailabilitySummary: document.getElementById("detail-availability-summary"),
  detailAvailabilityList: document.getElementById("detail-availability-list"),
  detailMarketContextSummary: document.getElementById("detail-market-context-summary"),
  detailMarketContextList: document.getElementById("detail-market-context-list"),
  detailExportJson: document.getElementById("detail-export-json"),
  detailExportCsv: document.getElementById("detail-export-csv"),
  watchlistTag: document.getElementById("watchlist-tag"),
  watchlistNotes: document.getElementById("watchlist-notes"),
  watchlistAddBtn: document.getElementById("watchlist-add-btn"),
  watchlistRefreshBtn: document.getElementById("watchlist-refresh-btn"),
  watchlistExportBtn: document.getElementById("watchlist-export-btn"),
  watchlistExportJsonBtn: document.getElementById("watchlist-export-json-btn"),
  watchlistMeta: document.getElementById("watchlist-meta"),
  watchlistBody: document.getElementById("watchlist-body"),
  barMarket: document.getElementById("bar-market"),
  barExpected: document.getElementById("bar-expected"),
  barLower: document.getElementById("bar-lower"),

  funnelSplit: document.getElementById("funnel-split"),
  funnelRoleNeed: document.getElementById("funnel-role-need"),
  funnelMinAge: document.getElementById("funnel-min-age"),
  funnelMaxAge: document.getElementById("funnel-max-age"),
  funnelMinMinutes: document.getElementById("funnel-min-minutes"),
  funnelMinConfidence: document.getElementById("funnel-min-confidence"),
  funnelMinGap: document.getElementById("funnel-min-gap"),
  funnelBudgetBand: document.getElementById("funnel-budget-band"),
  funnelMaxContractYears: document.getElementById("funnel-max-contract-years"),
  funnelTopN: document.getElementById("funnel-top-n"),
  funnelLowerOnly: document.getElementById("funnel-lower-only"),
  funnelRunBtn: document.getElementById("funnel-run-btn"),
  funnelExportBtn: document.getElementById("funnel-export-btn"),
  funnelMeta: document.getElementById("funnel-meta"),
  funnelSummaryTitle: document.getElementById("funnel-summary-title"),
  funnelSummaryLeague: document.getElementById("funnel-summary-league"),
  funnelSummaryCount: document.getElementById("funnel-summary-count"),
  funnelSummaryGap: document.getElementById("funnel-summary-gap"),
  funnelSummaryConfidence: document.getElementById("funnel-summary-confidence"),
  funnelSummaryStatus: document.getElementById("funnel-summary-status"),
  funnelSummaryCopy: document.getElementById("funnel-summary-copy"),
  funnelBody: document.getElementById("funnel-body"),
  funnelLeagueBody: document.getElementById("funnel-league-body"),

  profileModal: document.getElementById("profile-modal"),
  profileModalScrim: document.getElementById("profile-modal-scrim"),
  profileModalTitle: document.getElementById("profile-modal-title"),
  profileModalMeta: document.getElementById("profile-modal-meta"),
  profileModalBody: document.getElementById("profile-modal-body"),
  profileModalCloseBtn: document.getElementById("profile-modal-close-btn"),
  profileModalExportJson: document.getElementById("profile-modal-export-json"),
  profileModalExportCsv: document.getElementById("profile-modal-export-csv"),
};

function formatCurrency(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return currencyFmt.format(n);
}

function formatInt(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return numberFmt.format(Math.round(n));
}

function formatPct(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return `${pctFmt.format(n * 100)}%`;
}

function formatNumber(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return decimalFmt.format(n);
}

function parseNumberOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function parseOptionalPositive(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function safeText(value) {
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function leagueKey(value) {
  return String(value || "").trim().toLowerCase();
}

function isBig5LeagueValue(value) {
  return BIG5_LEAGUES.has(leagueKey(value));
}

function budgetLabel(maxBudget) {
  const n = Number(maxBudget);
  if (!Number.isFinite(n) || n <= 0) return "any budget";
  return `<= ${formatCurrency(n)}`;
}

function humanizeExperimentLabel(value) {
  const text = String(value || "").trim();
  if (!text) return "-";
  return text
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\b([a-z])/g, (match) => match.toUpperCase());
}

const SCORE_COLUMN_LABELS = {
  future_scout_blend_score: "Future scout blend",
  future_growth_probability: "Future growth probability",
  scout_target_score: "Scout target score",
  shortlist_score: "Shortlist score",
  undervaluation_score: "Undervaluation score",
  value_gap_capped_eur: "Capped conservative gap",
  value_gap_conservative_eur: "Conservative gap",
};

const RANKING_BASIS_LABELS = {
  future_target_tuned_blend: "Future-tuned blend of growth signal and current undervaluation",
  future_target_probability: "Future growth probability only",
  guardrailed_gap_confidence_history: "Guardrailed gap x confidence x history",
  guardrailed_gap_confidence_history_efficiency: "Guardrailed gap x confidence x history x value efficiency",
  manual_sort: "Manual workbench sort",
  funnel_rank: "Talent funnel rank",
};

const ROLE_LENS_CONFIG = {
  GK: {
    label: "Goalkeeper",
    summary: "Focus on shot-stopping, box command, and distribution.",
    metrics: [
      { label: "Saves", keys: ["sofa_saves"] },
      { label: "Box saves", keys: ["sofa_savedShotsFromInsideTheBox", "sofa_savedShotsFromOutsideTheBox"] },
      { label: "High claims", keys: ["sofa_highClaims"] },
      { label: "Runs out", keys: ["sofa_successfulRunsOut"] },
      { label: "Long-ball accuracy", keys: ["sofa_accurateLongBallsPercentage"] },
    ],
  },
  CB: {
    label: "Centre-back",
    summary: "Focus on aerial control, duel security, defending volume, and build-up passing.",
    metrics: [
      { label: "Aerial win rate", keys: ["sb_aerial_win_rate", "sofa_aerialDuelsWonPercentage"] },
      { label: "Duel win rate", keys: ["sb_duel_win_rate", "sofa_totalDuelsWonPercentage"] },
      { label: "Interceptions/90", keys: ["sofa_interceptions_per90"] },
      { label: "Clearances/90", keys: ["sofa_clearances_per90"] },
      { label: "Progressive passes/90", keys: ["sb_progressive_passes_per90", "sofa_accurateLongBallsPercentage"] },
    ],
  },
  FB: {
    label: "Fullback / Wing-back",
    summary: "Focus on transition defending, carrying, crossing, and wide progression.",
    metrics: [
      { label: "Tackles/90", keys: ["sofa_tackles_per90"] },
      { label: "Interceptions/90", keys: ["sofa_interceptions_per90"] },
      { label: "Progressive carries/90", keys: ["sb_progressive_carries_per90", "sofa_successfulDribbles_per90"] },
      { label: "Progressive passes/90", keys: ["sb_progressive_passes_per90", "sofa_accurateFinalThirdPasses_per90"] },
      { label: "Cross accuracy", keys: ["sofa_accurateCrossesPercentage", "sofa_accurateCrosses"] },
    ],
  },
  DM: {
    label: "Defensive midfielder",
    summary: "Focus on regains, duel control, safe circulation, and early progression.",
    metrics: [
      { label: "Tackles/90", keys: ["sofa_tackles_per90"] },
      { label: "Interceptions/90", keys: ["sofa_interceptions_per90"] },
      { label: "Duel win rate", keys: ["sb_duel_win_rate", "sofa_totalDuelsWonPercentage"] },
      { label: "Progressive passes/90", keys: ["sb_progressive_passes_per90"] },
      { label: "Pass accuracy", keys: ["sofa_accuratePassesPercentage"] },
    ],
  },
  CM: {
    label: "Central midfielder",
    summary: "Focus on progression, tempo, final-third delivery, and balanced two-way output.",
    metrics: [
      { label: "Progressive passes/90", keys: ["sb_progressive_passes_per90"] },
      { label: "Final-third passes/90", keys: ["sofa_accurateFinalThirdPasses_per90"] },
      { label: "Key passes/90", keys: ["sofa_keyPasses_per90"] },
      { label: "Pass accuracy", keys: ["sofa_accuratePassesPercentage"] },
      { label: "Progressive carries/90", keys: ["sb_progressive_carries_per90", "sofa_successfulDribbles_per90"] },
    ],
  },
  AM: {
    label: "Attacking midfielder",
    summary: "Focus on chance creation, box access, dribbling, and final-third passing.",
    metrics: [
      { label: "Key passes/90", keys: ["sofa_keyPasses_per90"] },
      { label: "Shot assists/90", keys: ["sb_shot_assists_per90", "sofa_passToAssist"] },
      { label: "Passes into box/90", keys: ["sb_passes_into_box_per90", "sofa_accurateFinalThirdPasses_per90"] },
      { label: "Dribbles/90", keys: ["sofa_successfulDribbles_per90"] },
      { label: "xG/90", keys: ["sofa_expectedGoals_per90"] },
    ],
  },
  W: {
    label: "Winger",
    summary: "Focus on dribbling, carrying, creation, and final-third access.",
    metrics: [
      { label: "Dribbles/90", keys: ["sofa_successfulDribbles_per90"] },
      { label: "Progressive carries/90", keys: ["sb_progressive_carries_per90"] },
      { label: "Shot assists/90", keys: ["sb_shot_assists_per90", "sofa_keyPasses_per90"] },
      { label: "Passes into box/90", keys: ["sb_passes_into_box_per90", "sofa_accurateCrossesPercentage"] },
      { label: "xG/90", keys: ["sofa_expectedGoals_per90"] },
    ],
  },
  SS: {
    label: "Support forward",
    summary: "Focus on hybrid creation and finishing rather than pure box-volume scoring.",
    metrics: [
      { label: "Key passes/90", keys: ["sofa_keyPasses_per90", "sb_shot_assists_per90"] },
      { label: "Dribbles/90", keys: ["sofa_successfulDribbles_per90"] },
      { label: "xG/90", keys: ["sofa_expectedGoals_per90"] },
      { label: "Shots on target/90", keys: ["sofa_shotsOnTarget_per90"] },
      { label: "Passes into box/90", keys: ["sb_passes_into_box_per90"] },
    ],
  },
  ST: {
    label: "Striker",
    summary: "Focus on shot volume, shot quality, finishing, and central duel strength.",
    metrics: [
      { label: "xG/90", keys: ["sofa_expectedGoals_per90"] },
      { label: "Shots/90", keys: ["sofa_totalShots_per90"] },
      { label: "Shots on target/90", keys: ["sofa_shotsOnTarget_per90"] },
      { label: "Conversion %", keys: ["sofa_goalConversionPercentage"] },
      { label: "Aerial win rate", keys: ["sb_aerial_win_rate", "sofa_aerialDuelsWonPercentage"] },
    ],
  },
  DF: {
    label: "Defender",
    summary: "Focus on duel security, defensive actions, and reliable distribution.",
    metrics: [
      { label: "Duel win rate", keys: ["sb_duel_win_rate", "sofa_totalDuelsWonPercentage"] },
      { label: "Interceptions/90", keys: ["sofa_interceptions_per90"] },
      { label: "Tackles/90", keys: ["sofa_tackles_per90"] },
      { label: "Clearances/90", keys: ["sofa_clearances_per90"] },
      { label: "Pass accuracy", keys: ["sofa_accuratePassesPercentage"] },
    ],
  },
  MF: {
    label: "Midfielder",
    summary: "Focus on progression, passing reliability, and balanced contribution across phases.",
    metrics: [
      { label: "Progressive passes/90", keys: ["sb_progressive_passes_per90"] },
      { label: "Key passes/90", keys: ["sofa_keyPasses_per90"] },
      { label: "Pass accuracy", keys: ["sofa_accuratePassesPercentage"] },
      { label: "Tackles/90", keys: ["sofa_tackles_per90"] },
      { label: "Interceptions/90", keys: ["sofa_interceptions_per90"] },
    ],
  },
  FW: {
    label: "Forward",
    summary: "Focus on chance volume, finishing threat, and attacking creation.",
    metrics: [
      { label: "xG/90", keys: ["sofa_expectedGoals_per90"] },
      { label: "Shots/90", keys: ["sofa_totalShots_per90"] },
      { label: "Key passes/90", keys: ["sofa_keyPasses_per90", "sb_shot_assists_per90"] },
      { label: "Dribbles/90", keys: ["sofa_successfulDribbles_per90"] },
      { label: "Conversion %", keys: ["sofa_goalConversionPercentage"] },
    ],
  },
};

function humanizeScoreColumn(scoreColumn) {
  return SCORE_COLUMN_LABELS[String(scoreColumn || "").trim()] || humanizeKey(scoreColumn || "score");
}

function rankingBasisLabel(rankingBasis) {
  return RANKING_BASIS_LABELS[String(rankingBasis || "").trim()] || safeText(rankingBasis || "Ranking");
}

function inferRoleKey(row) {
  const primary = `${row?.position_main || ""} ${row?.position || ""} ${row?.position_alt || ""}`.toLowerCase();
  const group = getPosition(row);
  if (group === "GK" || primary.includes("goalkeeper")) return "GK";
  if (group === "DF") {
    if (/wing-back|wing back|fullback|full back|left back|right back|defender,\s*(left|right)/.test(primary)) return "FB";
    if (/centre|center|back/.test(primary)) return "CB";
    return "DF";
  }
  if (group === "MF") {
    if (/defensive/.test(primary)) return "DM";
    if (/attacking/.test(primary)) return "AM";
    if (/left midfield|right midfield|wing/.test(primary)) return "W";
    if (/central/.test(primary)) return "CM";
    return "MF";
  }
  if (group === "FW") {
    if (/winger/.test(primary)) return "W";
    if (/second striker/.test(primary)) return "SS";
    if (/forward|striker|attack,\s*centre/.test(primary)) return "ST";
    return "FW";
  }
  return group || "FW";
}

function roleNeedLabel(roleKey) {
  const key = String(roleKey || "").trim().toUpperCase();
  if (!key) return "Any role";
  return ROLE_LENS_CONFIG[key]?.label || key;
}

function resolveRoleMetricSnapshot(row, spec) {
  for (const key of spec.keys || []) {
    const value = row?.[key];
    if (typeof value === "string" && value.trim() !== "") {
      const asNum = Number(value);
      if (Number.isFinite(asNum)) {
        return { key, label: spec.label, displayValue: formatStatValue(key, asNum), rawValue: asNum };
      }
    } else if (Number.isFinite(Number(value))) {
      const asNum = Number(value);
      return { key, label: spec.label, displayValue: formatStatValue(key, asNum), rawValue: asNum };
    }
  }
  return null;
}

function buildRoleLens(row, profile = null) {
  const backendRoleKey = String(profile?.player_type?.position_key || "").trim();
  const roleKey = backendRoleKey || inferRoleKey(row);
  const config = ROLE_LENS_CONFIG[roleKey] || ROLE_LENS_CONFIG[getPosition(row)] || ROLE_LENS_CONFIG.FW;
  const metrics = (config.metrics || [])
    .map((spec) => resolveRoleMetricSnapshot(row, spec))
    .filter(Boolean)
    .slice(0, 4);
  const label = safeText(profile?.player_type?.position_label || config.label);
  const archetype = safeText(profile?.player_type?.archetype || "");
  const summary = `${label} lens: ${config.summary}${archetype && archetype !== "-" ? ` Archetype fit currently leans ${archetype}.` : ""}`;
  return {
    key: roleKey,
    label,
    summary,
    metrics,
  };
}

function slugifyLeagueName(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function hasFiniteSignal(value) {
  return Number.isFinite(Number(value));
}

function hasAnyFiniteSignal(row, keys) {
  return keys.some((key) => hasFiniteSignal(row?.[key]));
}

function rowSignalCoverage(row) {
  const statsbomb = Number(row?.sb_has_data) > 0 || hasAnyFiniteSignal(row, [
    "sb_progressive_passes_per90",
    "sb_progressive_carries_per90",
    "sb_shot_assists_per90",
    "sb_pressures_per90",
    "sb_high_regains_per90",
  ]);
  const availability = hasAnyFiniteSignal(row, [
    "avail_reports",
    "avail_start_share",
    "avail_bench_share",
    "avail_injury_count",
    "avail_expected_start_rate",
  ]);
  const market = hasAnyFiniteSignal(row, [
    "fixture_matches",
    "fixture_mean_rest_days",
    "fixture_congestion_share",
    "odds_implied_team_strength",
    "odds_expected_total_goals",
  ]);
  const future = hasAnyFiniteSignal(row, [
    "future_scout_blend_score",
    "future_growth_probability",
    "future_scout_score",
  ]);
  return { statsbomb, availability, market, future };
}

function buildBadgeChipMarkup(label, tone = "neutral") {
  return `<span class="badge-chip badge-chip--${escapeHtml(tone)}">${escapeHtml(label)}</span>`;
}

function buildDecisionPillMarkup(decision) {
  const tone = safeText(decision?.tone || "neutral").toLowerCase();
  const label = safeText(decision?.label || "Waiting");
  return `<span class="decision-pill decision-pill--${escapeHtml(tone)}">${escapeHtml(label)}</span>`;
}

function classifyConfidenceSignal(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return { band: "unknown", label: "Unknown", tone: "neutral", note: "Confidence unavailable." };
  }
  if (n <= 1.2) {
    if (n >= 0.72) return { band: "strong", label: "Strong", tone: "pursue", note: "Strong confidence signal." };
    if (n >= 0.58) return { band: "decent", label: "Decent", tone: "watch", note: "Decent confidence signal." };
    if (n >= 0.42) return { band: "thin", label: "Thin", tone: "price", note: "Thin confidence signal." };
    return { band: "weak", label: "Weak", tone: "pass", note: "Weak confidence signal." };
  }
  if (n >= 5) return { band: "strong", label: "Strong", tone: "pursue", note: "Strong confidence signal." };
  if (n >= 3.5) return { band: "decent", label: "Decent", tone: "watch", note: "Decent confidence signal." };
  if (n >= 2.2) return { band: "thin", label: "Thin", tone: "price", note: "Thin confidence signal." };
  return { band: "weak", label: "Weak", tone: "pass", note: "Weak confidence signal." };
}

function summarizeRecruitmentDecision(row, report = null, { source = "workbench" } = {}) {
  const gaps = deriveGapValues(row, report);
  const market = firstFiniteNumber(report?.valuation_guardrails?.market_value_eur, row?.market_value_eur);
  const expected = firstFiniteNumber(report?.valuation_guardrails?.fair_value_eur, row?.expected_value_eur);
  const confidenceScore = firstFiniteNumber(report?.confidence?.score_0_to_1, row?.undervaluation_confidence);
  const confidence = classifyConfidenceSignal(confidenceScore);
  const leagueStatus = summarizeLeagueStatus(row?.league);
  const minutes = getMinutes(row);
  const futureProb = firstFiniteNumber(row?.future_growth_probability);
  const gap = Number(gaps.capped);
  const gapRatio = Number.isFinite(gap) && Number.isFinite(market) && market > 0 ? gap / market : NaN;
  let score = 0;

  if (Number.isFinite(gap)) {
    if (gap >= 2_500_000 || gapRatio >= 1.2) score += 2;
    else if (gap >= 1_000_000 || gapRatio >= 0.6) score += 1;
    else if (gap <= 0) score -= 2;
  }

  if (confidence.band === "strong") score += 2;
  else if (confidence.band === "decent") score += 1;
  else if (confidence.band === "weak") score -= 1;

  if (Number.isFinite(minutes) && minutes >= 1800) score += 1;
  else if (Number.isFinite(minutes) && minutes < 600) score -= 1;

  if (Number.isFinite(futureProb) && futureProb >= 0.55) score += 1;

  if (leagueStatus.tone === "bad") score -= 2;
  else if (leagueStatus.tone === "warn") score -= 1;

  let label = "Pass";
  let tone = "pass";
  let reason = "There is not enough investable upside or confidence to justify attention right now.";
  let nextAction = "Do not prioritize live scouting.";
  let priceStance = "Too thin for an active push";

  if (Number.isFinite(gap) && gap > 0) {
    if (score >= 4) {
      label = "Pursue";
      tone = "pursue";
      reason = "The guardrailed upside is strong enough to justify live follow-up, not just passive monitoring.";
      nextAction = "Open the memo and assign a live scouting step.";
      priceStance = "Investable upside";
    } else if (score >= 2) {
      label = "Watch";
      tone = "watch";
      reason = "There is real upside here, but the case still needs another layer of validation before it becomes a push target.";
      nextAction = "Keep on shortlist and validate through the memo.";
      priceStance = "Positive but not fully proven";
    } else {
      label = "Price Only";
      tone = "price";
      reason = "The valuation delta is useful for pricing discipline, but the recruitment case is too thin for an active chase.";
      nextAction = "Use as a pricing reference or low-priority comparable.";
      priceStance = "Pricing reference";
    }
  }

  if (source === "predictions" && label === "Pursue") {
    label = "Price Only";
    tone = "price";
    reason = "This view is manually sorted for valuation work. Treat the signal as pricing guidance, not a live pursuit order.";
    nextAction = "Use for pricing discipline, then switch back to Recruitment Board for live ranking.";
    priceStance = "Valuation view";
  }

  const gapNote = Number.isFinite(gap)
    ? `${formatCurrency(gap)} capped gap${Number.isFinite(gaps.conservative) && gaps.capApplied ? ` from ${formatCurrency(gaps.conservative)} raw` : ""}`
    : "No guardrailed gap available";
  const confidenceNote = Number.isFinite(confidenceScore)
    ? `${confidence.label} confidence | score ${formatNumber(confidenceScore)}`
    : "Confidence unavailable";
  const priceContext = [
    Number.isFinite(market) ? `Market ${formatCurrency(market)}` : null,
    Number.isFinite(expected) ? `Expected ${formatCurrency(expected)}` : null,
  ]
    .filter(Boolean)
    .join(" | ");

  return {
    label,
    tone,
    reason,
    nextAction,
    priceStance,
    gap,
    gapNote,
    confidenceScore,
    confidenceLabel: confidence.label,
    confidenceTone: confidence.tone,
    confidenceNote,
    priceContext: priceContext || "Market and expected value unavailable.",
    leagueTone: leagueStatus.tone,
    leagueNote: leagueStatus.note,
  };
}

function buildProvenanceBadges(row) {
  const coverage = rowSignalCoverage(row);
  const badges = [];
  if (coverage.future) {
    badges.push({ label: "Future-scored", tone: "future" });
  }
  if (coverage.statsbomb || coverage.availability || coverage.market) {
    badges.push({ label: "Provider-enriched", tone: "provider" });
  }
  if (!badges.length) {
    badges.push({ label: "Valuation only", tone: "neutral" });
  }
  if (coverage.statsbomb) badges.push({ label: "StatsBomb", tone: "provider-soft" });
  if (coverage.availability) badges.push({ label: "Availability", tone: "provider-soft" });
  if (coverage.market) badges.push({ label: "Fixture/Market", tone: "provider-soft" });
  return badges;
}

function scoreValueForRow(row, scoreColumn) {
  const n = Number(row?.[scoreColumn]);
  return Number.isFinite(n) ? n : NaN;
}

function resolveRowScoreContext(row, diagnostics = null, source = "workbench") {
  const scoreColumn =
    diagnostics?.score_column ||
    diagnostics?.scoreColumn ||
    (source === "predictions" ? state.sortBy : null) ||
    (hasFiniteSignal(row?.future_scout_blend_score)
      ? "future_scout_blend_score"
      : hasFiniteSignal(row?.future_growth_probability)
      ? "future_growth_probability"
      : hasFiniteSignal(row?.scout_target_score)
      ? "scout_target_score"
      : hasFiniteSignal(row?.shortlist_score)
      ? "shortlist_score"
      : "value_gap_capped_eur");
  const rankingBasis =
    diagnostics?.ranking_basis ||
    diagnostics?.rankingBasis ||
    (source === "predictions" ? "manual_sort" : source === "funnel" ? "funnel_rank" : null) ||
    (scoreColumn === "future_scout_blend_score"
      ? "future_target_tuned_blend"
      : scoreColumn === "future_growth_probability"
      ? "future_target_probability"
      : scoreColumn === "scout_target_score"
      ? "guardrailed_gap_confidence_history_efficiency"
      : "guardrailed_gap_confidence_history");
  return {
    scoreColumn,
    scoreLabel: humanizeScoreColumn(scoreColumn),
    rankingBasis,
    rankingBasisLabel: rankingBasisLabel(rankingBasis),
    scoreValue: scoreValueForRow(row, scoreColumn),
  };
}

function benchmarkLeagueRows() {
  const holdoutRows = Array.isArray(state.benchmark?.league_holdout?.rows) ? state.benchmark.league_holdout.rows : [];
  const predictionRows = Array.isArray(state.benchmark?.prediction_league?.rows) ? state.benchmark.prediction_league.rows : [];
  return {
    holdoutRows,
    predictionRows,
    onboardingItems: Array.isArray(state.benchmark?.onboarding?.items) ? state.benchmark.onboarding.items : [],
  };
}

function findLeagueDiagnostics(league) {
  const name = String(league || "").trim();
  const slug = slugifyLeagueName(name);
  const { holdoutRows, predictionRows, onboardingItems } = benchmarkLeagueRows();
  return {
    holdout: holdoutRows.find((row) => String(row?.league || "").trim().toLowerCase() === name.toLowerCase()) || null,
    prediction:
      predictionRows.find((row) => String(row?.league || "").trim().toLowerCase() === name.toLowerCase()) || null,
    onboarding:
      onboardingItems.find((item) => String(item?.league_slug || "").trim().toLowerCase() === slug.toLowerCase()) || null,
  };
}

function summarizeLeagueStatus(league) {
  const { holdout, prediction, onboarding } = findLeagueDiagnostics(league);
  const notes = [];
  let tone = "ok";
  let label = "usable";

  if (onboarding?.status === "blocked") {
    tone = "bad";
    label = "blocked";
    notes.push(`Onboarding blocked: ${safeText(onboarding.reasons)}`);
  } else if (onboarding?.status === "watch") {
    tone = "warn";
    label = "watch";
    notes.push(`Onboarding watch: ${safeText(onboarding.reasons)}`);
  }

  const wmape = Number(holdout?.wmape);
  const r2 = Number(holdout?.r2);
  if (holdout) {
    if ((Number.isFinite(wmape) && wmape >= 0.7) || (Number.isFinite(r2) && r2 < 0.3)) {
      tone = "bad";
      label = "high risk";
      notes.push(`Holdout error elevated (${Number.isFinite(wmape) ? formatPct(wmape) : "n/a"} WMAPE, ${Number.isFinite(r2) ? formatPct(r2) : "n/a"} R²).`);
    } else if ((Number.isFinite(wmape) && wmape >= 0.5) || (Number.isFinite(r2) && r2 < 0.45)) {
      tone = tone === "bad" ? tone : "warn";
      label = tone === "bad" ? label : "watch";
      notes.push(`Holdout is usable but noisy (${Number.isFinite(wmape) ? formatPct(wmape) : "n/a"} WMAPE).`);
    }
  } else if (prediction) {
    const predWmape = Number(prediction?.wmape);
    const predR2 = Number(prediction?.r2);
    if ((Number.isFinite(predWmape) && predWmape >= 0.5) || (Number.isFinite(predR2) && predR2 < 0.45)) {
      tone = tone === "bad" ? tone : "warn";
      label = tone === "bad" ? label : "watch";
      notes.push(`Prediction-league diagnostics are noisy (${Number.isFinite(predWmape) ? formatPct(predWmape) : "n/a"} WMAPE).`);
    }
  }

  if (!notes.length) {
    notes.push("Coverage and benchmark diagnostics are acceptable for shortlist review.");
  }
  return { tone, label, note: notes[0], notes, holdout, prediction, onboarding };
}

function detailCoverageWarnings(row, profile = null) {
  const warnings = [];
  const leagueStatus = summarizeLeagueStatus(row?.league);
  if (leagueStatus.note) {
    warnings.push({
      severity: leagueStatus.tone === "bad" ? "high" : leagueStatus.tone === "warn" ? "medium" : "low",
      code: "league_coverage",
      message: leagueStatus.note,
    });
  }

  const coverage = rowSignalCoverage({ ...row, ...(profile?.player || {}) });
  if (!coverage.statsbomb) {
    warnings.push({
      severity: "low",
      code: "statsbomb_missing",
      message: "No StatsBomb tactical signal matched this player in the active artifact.",
    });
  }
  if (!coverage.availability) {
    warnings.push({
      severity: "low",
      code: "availability_missing",
      message: "No provider availability context matched this player.",
    });
  }
  if (!coverage.market) {
    warnings.push({
      severity: "low",
      code: "schedule_market_missing",
      message: "No fixture or market-context enrichment matched this player.",
    });
  }
  const historyCoverage = Number(row?.history_strength_coverage);
  if (Number.isFinite(historyCoverage) && historyCoverage < 0.35) {
    warnings.push({
      severity: "medium",
      code: "history_sparse",
      message: `History-strength coverage is sparse (${formatPct(historyCoverage)}).`,
    });
  }
  return warnings;
}

function whyRankedItems(row, profile = null, scoreContext = null) {
  const context = scoreContext || resolveRowScoreContext(row, state.queryDiagnostics, state.mode);
  const gaps = deriveGapValues(row, profile);
  const roleLens = buildRoleLens(row, profile);
  const items = [];
  const market = Number(row?.market_value_eur);
  const confidence = Number(row?.undervaluation_confidence);
  const futureProb = Number(row?.future_growth_probability);
  const history = Number(row?.history_strength_score);
  const minutes = getMinutes(row);
  const age = Number(row?.age);

  items.push({
    label: "Score driver",
    message: `${context.scoreLabel} is the active ranking driver for this view${Number.isFinite(context.scoreValue) ? ` (${context.scoreColumn.endsWith("_eur") ? formatCurrency(context.scoreValue) : formatNumber(context.scoreValue)})` : ""}. ${context.rankingBasisLabel}.`,
  });
  if (Number.isFinite(gaps.capped)) {
    items.push({
      label: "Guardrailed value gap",
      message: `Capped conservative gap is ${formatCurrency(gaps.capped)}${Number.isFinite(gaps.conservative) && gaps.capApplied ? ` after trimming raw conservative upside of ${formatCurrency(gaps.conservative)}` : ""}.`,
    });
  }
  if (Number.isFinite(confidence)) {
    items.push({
      label: "Confidence",
      message: `Model confidence is ${formatNumber(confidence)}${Number.isFinite(market) && market > 0 && Number.isFinite(gaps.capped) ? ` against a current market price of ${formatCurrency(market)}` : ""}.`,
    });
  }
  if (Number.isFinite(futureProb)) {
    items.push({
      label: "Future signal",
      message: `Future growth probability is ${formatPct(futureProb)}, so this player is not only cheap on valuation but also favored by the future-outcome layer.`,
    });
  }
  if (Number.isFinite(age) || Number.isFinite(minutes)) {
    items.push({
      label: "Sample quality",
      message: `${Number.isFinite(age) ? `Age ${formatNumber(age)}` : "Age unavailable"} | ${Number.isFinite(minutes) ? `${formatInt(minutes)} minutes` : "minutes unavailable"}. Younger players with real minutes get a stronger scouting rank.`,
    });
  }
  if (Number.isFinite(history)) {
    items.push({
      label: "History context",
      message: `History-strength score is ${formatNumber(history)}/100, which adjusts how aggressively the rank trusts one-season performance.`,
    });
  }
  if (Array.isArray(roleLens.metrics) && roleLens.metrics.length) {
    const metricsText = roleLens.metrics
      .slice(0, 3)
      .map((metric) => `${metric.label} ${metric.displayValue}`)
      .join(" | ");
    items.push({
      label: `${roleLens.label} lens`,
      message: `${roleLens.summary} Priority signals in the active artifact: ${metricsText}.`,
    });
  }
  return items.slice(0, 5);
}

function providerContextFallbackSummary(kind, row, context = null) {
  if (context?.summary_text) return safeText(context.summary_text);
  const coverage = rowSignalCoverage(row);
  if (kind === "tactical") {
    return coverage.statsbomb
      ? "Provider coverage exists, but no tactical signal list was returned for this player."
      : "No StatsBomb tactical signal matched this player in the active artifact.";
  }
  if (kind === "availability") {
    return coverage.availability
      ? "Availability coverage exists, but no detailed availability signal list was returned."
      : "No provider availability context matched this player.";
  }
  if (kind === "market") {
    return coverage.market
      ? "Schedule or market coverage exists, but no detailed signal list was returned."
      : "No fixture or market-context enrichment matched this player.";
  }
  return "No provider signals matched this player.";
}

function buildNarrativeListMarkup(items, emptyMessage, { labelKey = "label", messageKey = "message", toneKey = "tone" } = {}) {
  if (!Array.isArray(items) || !items.length) {
    return `<li class="risk-item risk-item--none">${escapeHtml(emptyMessage)}</li>`;
  }
  return items
    .map((item) => {
      const tone = safeText(item?.[toneKey] || item?.severity || "low").toLowerCase();
      const label = safeText(item?.[labelKey] || item?.code || "context");
      const message = safeText(item?.[messageKey] || "");
      return `
        <li class="risk-item">
          <div class="risk-head">
            <span class="risk-severity risk-severity--${escapeHtml(tone)}">${escapeHtml(tone)}</span>
            <span class="risk-code">${escapeHtml(label)}</span>
          </div>
          <p class="risk-message">${escapeHtml(message)}</p>
        </li>
      `;
    })
    .join("");
}

const DETAIL_EMPTY_LIST_ITEM = '<li class="risk-item risk-item--none">No data loaded.</li>';
const PROFILE_CONTEXT_KEYS = [
  "player_id",
  "name",
  "club",
  "league",
  "season",
  "model_position",
  "position_group",
  "position_main",
  "country",
  "nationality",
  "age",
  "height",
  "foot",
  "contract_years_left",
];
const STAT_SKIP_KEYS = new Set([
  "_funnelScore",
  "name",
  "club",
  "league",
  "season",
  "player_id",
  "split",
  "source",
  "source_file",
  "league_norm",
  "position_norm",
  "value_segment",
  "position_main",
  "position_group",
  "model_position",
  "expected_value_raw_eur",
  "expected_value_low_raw_eur",
  "expected_value_high_raw_eur",
  "value_gap_raw_eur",
  "value_gap_eur",
]);
const STAT_GROUP_ORDER = [
  "Profile & Context",
  "Value & Model",
  "Attacking",
  "Passing & Progression",
  "Defending & Duels",
  "Goalkeeping",
  "Availability & Physical",
  "History & Context",
  "Other Metrics",
];

function rowKey(row) {
  const playerId = String(row?.player_id || "").trim();
  const season = String(row?.season || "").trim();
  return `${playerId}::${season}`;
}

function humanizeKey(key) {
  if (!key) return "-";
  const raw = String(key)
    .replace(/^sofa_/, "")
    .replace(/^clubctx_/, "club ")
    .replace(/^history_/, "history ")
    .replace(/^prior_/, "prior ")
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ")
    .trim();
  return raw
    .replace(/\beur\b/i, "EUR")
    .replace(/\bper90\b/i, "/90")
    .replace(/\b0 to 1\b/i, "0-1")
    .replace(/\bpct\b/i, "%")
    .replace(/\bxa\b/i, "xA")
    .replace(/\bxg\b/i, "xG")
    .replace(/\bnp xg\b/i, "npxG")
    .replace(/\b([a-z])/g, (match) => match.toUpperCase());
}

function isRenderableStatValue(key, value) {
  if (STAT_SKIP_KEYS.has(key) || String(key).startsWith("_")) return false;
  if (typeof value === "boolean") return true;
  if (typeof value === "string") {
    return PROFILE_CONTEXT_KEYS.includes(key) && value.trim() !== "";
  }
  const n = Number(value);
  return Number.isFinite(n);
}

function formatStatValue(key, value) {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return safeText(value);
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  const lowerKey = String(key).toLowerCase();
  if (lowerKey.endsWith("_eur")) return formatCurrency(n);
  if (lowerKey.endsWith("_0_to_1")) return formatPct(n);
  if (lowerKey.includes("percentile")) return formatPct(n);
  if (lowerKey.includes("percentage")) return `${formatNumber(n)}%`;
  if (lowerKey.includes("minutes") || lowerKey.includes("count") || lowerKey.includes("caps") || lowerKey.endsWith("_n")) {
    return formatInt(n);
  }
  if (Number.isInteger(n) && Math.abs(n) >= 10) return formatInt(n);
  return formatNumber(n);
}

function classifyStatGroup(key) {
  const lowerKey = String(key).toLowerCase();
  if (PROFILE_CONTEXT_KEYS.includes(key)) return "Profile & Context";
  if (
    lowerKey.includes("market_value") ||
    lowerKey.includes("expected_value") ||
    lowerKey.includes("fair_value") ||
    lowerKey.includes("gap") ||
    lowerKey.includes("confidence") ||
    lowerKey.includes("interval") ||
    lowerKey.includes("calibration") ||
    lowerKey.includes("prior_")
  ) {
    return "Value & Model";
  }
  if (
    lowerKey.includes("goal") ||
    lowerKey.includes("assist") ||
    lowerKey.includes("shot") ||
    lowerKey.includes("xg") ||
    lowerKey.includes("xa") ||
    lowerKey.includes("dribble") ||
    lowerKey.includes("touchesinoppositionbox") ||
    lowerKey.includes("bigchance") ||
    lowerKey.includes("penalty")
  ) {
    return "Attacking";
  }
  if (
    lowerKey.includes("pass") ||
    lowerKey.includes("cross") ||
    lowerKey.includes("throughball") ||
    lowerKey.includes("longball") ||
    lowerKey.includes("keypass") ||
    lowerKey.includes("progressive") ||
    lowerKey.includes("chancecreated")
  ) {
    return "Passing & Progression";
  }
  if (
    lowerKey.includes("tackle") ||
    lowerKey.includes("interception") ||
    lowerKey.includes("clearance") ||
    lowerKey.includes("blocked") ||
    lowerKey.includes("duel") ||
    lowerKey.includes("aerial") ||
    lowerKey.includes("recovery") ||
    lowerKey.includes("ballrecovery") ||
    lowerKey.includes("possessionwon")
  ) {
    return "Defending & Duels";
  }
  if (
    lowerKey.includes("save") ||
    lowerKey.includes("highclaim") ||
    lowerKey.includes("runout") ||
    lowerKey.includes("goalsprevented") ||
    lowerKey.includes("cleansheet")
  ) {
    return "Goalkeeping";
  }
  if (
    lowerKey.includes("age") ||
    lowerKey.includes("minutes") ||
    lowerKey.includes("injury") ||
    lowerKey.includes("height") ||
    lowerKey.includes("weight") ||
    lowerKey.includes("contract") ||
    lowerKey.includes("foot")
  ) {
    return "Availability & Physical";
  }
  if (lowerKey.startsWith("clubctx_") || lowerKey.startsWith("history_") || lowerKey.includes("coeff")) {
    return "History & Context";
  }
  return "Other Metrics";
}

function buildStatGroups(row, report = null, history = null) {
  const reportPlayer = report?.player && typeof report.player === "object" ? report.player : {};
  const merged = { ...row, ...reportPlayer };
  const historyPayload =
    history?.history_strength && typeof history.history_strength === "object" ? history.history_strength : null;
  if (historyPayload) {
    merged.history_strength_score = historyPayload.score_0_to_100 ?? merged.history_strength_score;
    merged.history_strength_coverage_0_to_1 =
      historyPayload.coverage_0_to_1 ?? merged.history_strength_coverage_0_to_1;
  }

  const groups = new Map(STAT_GROUP_ORDER.map((name) => [name, []]));
  Object.entries(merged).forEach(([key, value]) => {
    if (!isRenderableStatValue(key, value)) return;
    const groupName = classifyStatGroup(key);
    groups.get(groupName).push({
      key,
      label: humanizeKey(key),
      value: formatStatValue(key, value),
    });
  });

  return STAT_GROUP_ORDER.map((name) => {
    const items = groups.get(name).sort((a, b) => a.label.localeCompare(b.label));
    return { name, items };
  }).filter((group) => group.items.length > 0);
}

function buildStatGroupsHtml(groups, { openCount = 2 } = {}) {
  if (!groups.length) {
    return '<p class="details-placeholder">No grouped stats available for this player.</p>';
  }
  return groups
    .map((group, idx) => {
      const rows = group.items
        .map(
          (item) => `
            <div class="stat-row">
              <span class="stat-label">${escapeHtml(item.label)}</span>
              <span class="stat-value">${escapeHtml(item.display_value ?? item.value)}</span>
            </div>
          `
        )
        .join("");
      return `
        <details class="stat-group" ${idx < openCount ? "open" : ""}>
          <summary>
            <span>${escapeHtml(group.name)}</span>
            <span class="stat-group__count">${formatInt(group.items.length)} metrics</span>
          </summary>
          <div class="stat-grid">${rows}</div>
        </details>
      `;
    })
    .join("");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function firstFiniteNumber(...values) {
  for (const value of values) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return NaN;
}

function deriveCapThreshold(row, guardrails = null) {
  const candidates = [];
  const priorQae = firstFiniteNumber(guardrails?.prior_qae_eur, row.prior_qae_eur);
  const priorP75ae = firstFiniteNumber(guardrails?.prior_p75ae_eur, row.prior_p75ae_eur);
  const priorMae = firstFiniteNumber(guardrails?.prior_mae_eur, row.prior_mae_eur);
  if (Number.isFinite(priorQae) && priorQae > 0) candidates.push(2.5 * priorQae);
  if (Number.isFinite(priorP75ae) && priorP75ae > 0) candidates.push(3.0 * priorP75ae);
  if (Number.isFinite(priorMae) && priorMae > 0) candidates.push(4.0 * priorMae);
  return candidates.length ? Math.min(...candidates) : NaN;
}

function deriveGapValues(row, report = null) {
  const guardrails = report?.valuation_guardrails || null;
  const market = firstFiniteNumber(guardrails?.market_value_eur, row.market_value_eur);
  const prediction = firstFiniteNumber(guardrails?.fair_value_eur, row.fair_value_eur, row.expected_value_eur);
  const impliedRaw = Number.isFinite(market) && Number.isFinite(prediction) ? prediction - market : NaN;
  const raw = firstFiniteNumber(guardrails?.value_gap_raw_eur, row.value_gap_eur, impliedRaw);
  const conservative = firstFiniteNumber(
    guardrails?.value_gap_conservative_eur,
    row.value_gap_conservative_eur,
    raw
  );
  const capThreshold = firstFiniteNumber(guardrails?.cap_threshold_eur, deriveCapThreshold(row, guardrails));
  let capped = firstFiniteNumber(guardrails?.value_gap_capped_eur, row.value_gap_capped_eur);
  if (!Number.isFinite(capped)) {
    if (Number.isFinite(conservative) && Number.isFinite(capThreshold) && conservative > 0) {
      capped = Math.min(conservative, capThreshold);
    } else {
      capped = conservative;
    }
  }
  const capAppliedFromReport = guardrails?.cap_applied;
  const capApplied =
    typeof capAppliedFromReport === "boolean"
      ? capAppliedFromReport
      : Number.isFinite(conservative) && Number.isFinite(capped) && capped < conservative - 1;
  return { raw, conservative, capped, capThreshold, capApplied };
}

function withComputedConservativeGap(row) {
  const gaps = deriveGapValues(row);
  return {
    ...row,
    value_gap_eur: Number.isFinite(gaps.raw) ? gaps.raw : row.value_gap_eur,
    value_gap_conservative_eur: Number.isFinite(gaps.conservative) ? gaps.conservative : row.value_gap_conservative_eur,
    value_gap_capped_eur: Number.isFinite(gaps.capped) ? gaps.capped : row.value_gap_capped_eur,
    cap_threshold_eur: Number.isFinite(gaps.capThreshold) ? gaps.capThreshold : row.cap_threshold_eur,
    cap_applied: gaps.capApplied,
  };
}

function conservativeGapForRanking(row) {
  const n = Number(row.value_gap_capped_eur);
  if (Number.isFinite(n)) return n;
  const gaps = deriveGapValues(row);
  return Number.isFinite(gaps.capped) ? gaps.capped : NaN;
}

function fileLabel(pathValue) {
  const path = String(pathValue || "");
  if (!path) return "-";
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function shortHash(hashValue) {
  const hash = String(hashValue || "");
  if (hash.length < 12) return "-";
  return hash.slice(0, 12);
}

function getMinutes(row) {
  const n = Number(row.minutes);
  if (Number.isFinite(n)) return n;
  const m = Number(row.sofa_minutesPlayed);
  if (Number.isFinite(m)) return m;
  return NaN;
}

function getPosition(row) {
  return (row.model_position || row.position_group || row.position_main || "").toUpperCase();
}

function getLeague(row) {
  return String(row.league || "");
}

function manifestRoleEntry(role) {
  const manifest = state.modelManifest || {};
  if (role === "valuation") return manifest.valuation_champion || null;
  if (role === "future_shortlist") return manifest.future_shortlist_champion || null;
  return null;
}

function manifestArtifactMeta(role, artifactKey) {
  return manifestRoleEntry(role)?.artifacts?.[artifactKey] || null;
}

function activeChampionRoutingSummary() {
  const active = state.activeArtifacts || {};
  const valuation = manifestRoleEntry("valuation");
  const futureShortlist = manifestRoleEntry("future_shortlist");
  return {
    valuationLabel: safeText(valuation?.label || fileLabel(active?.valuation?.metrics_path)),
    futureShortlistLabel: safeText(futureShortlist?.label || fileLabel(active?.future_shortlist?.metrics_path)),
    predictionBaseRole: safeText(active.prediction_service_base_role || "valuation"),
    shortlistOverlayRole: safeText(active.shortlist_overlay_role || "future_shortlist"),
  };
}

function matchesRecruitmentWorkflow(row, filters = {}) {
  const {
    minAge = null,
    maxAge = null,
    maxBudget = null,
    maxContractYearsLeft = null,
    roleNeed = "",
    nonBig5Only = false,
  } = filters;
  const age = Number(row.age);
  if (Number.isFinite(minAge) && minAge >= 0 && (!Number.isFinite(age) || age < minAge)) return false;
  if (Number.isFinite(maxAge) && maxAge >= 0 && (!Number.isFinite(age) || age > maxAge)) return false;

  const market = Number(row.market_value_eur);
  if (Number.isFinite(maxBudget) && maxBudget > 0 && (!Number.isFinite(market) || market > maxBudget)) return false;

  if (Number.isFinite(maxContractYearsLeft) && maxContractYearsLeft > 0) {
    const contractYears = Number(row.contract_years_left);
    if (!Number.isFinite(contractYears) || contractYears > maxContractYearsLeft) return false;
  }

  const wantedRole = String(roleNeed || "").trim().toUpperCase();
  if (wantedRole && inferRoleKey(row) !== wantedRole) return false;
  if (nonBig5Only && isBig5LeagueValue(getLeague(row))) return false;
  return true;
}

function currentWorkbenchWorkflowSummary() {
  return {
    minAge: state.minAge,
    maxAge: state.maxAge,
    maxBudget: parseOptionalPositive(state.budgetBand),
    maxContractYearsLeft: state.maxContractYearsLeft,
    roleNeed: state.roleNeed,
    nonBig5Only: state.nonBig5Only,
  };
}

function currentFunnelWorkflowSummary() {
  return {
    minAge: parseNumberOr(el.funnelMinAge?.value, -1),
    maxAge: parseNumberOr(el.funnelMaxAge?.value, -1),
    maxBudget: parseOptionalPositive(el.funnelBudgetBand?.value),
    maxContractYearsLeft: parseOptionalPositive(el.funnelMaxContractYears?.value),
    roleNeed: el.funnelRoleNeed?.value || "",
    nonBig5Only: Boolean(el.funnelLowerOnly?.checked),
  };
}

function buildRecruitmentExportRow(row, extras = {}) {
  const consGap = conservativeGapForRanking(row);
  const champion = activeChampionRoutingSummary();
  return {
    source: extras.source || "workbench",
    split: extras.split || state.split,
    rank: extras.rank ?? null,
    player_id: row.player_id ?? null,
    name: row.name ?? null,
    club: row.club ?? null,
    league: row.league ?? null,
    season: row.season ?? null,
    position_family: getPosition(row) || null,
    role_need_fit: inferRoleKey(row),
    age: row.age ?? null,
    market_value_eur: row.market_value_eur ?? null,
    expected_value_eur: row.expected_value_eur ?? row.fair_value_eur ?? null,
    value_gap_conservative_eur: Number.isFinite(consGap) ? consGap : null,
    value_gap_raw_eur: row.value_gap_eur ?? row.value_gap_raw_eur ?? null,
    undervaluation_confidence: row.undervaluation_confidence ?? null,
    minutes: getMinutes(row),
    contract_years_left: row.contract_years_left ?? null,
    shortlist_score: row.shortlist_score ?? null,
    future_scout_blend_score: row.future_scout_blend_score ?? null,
    scout_target_score: row.scout_target_score ?? row._funnelScore ?? null,
    ranking_driver: extras.rankingDriver || null,
    ranking_basis: extras.rankingBasis || null,
    tag: row.tag ?? null,
    notes: row.notes ?? null,
    valuation_champion: champion.valuationLabel,
    future_shortlist_champion: champion.futureShortlistLabel,
  };
}

function buildWindowPack(rows, options = {}) {
  const champion = activeChampionRoutingSummary();
  return {
    generated_at_utc: new Date().toISOString(),
    source: options.source || "workbench",
    split: options.split || state.split,
    filters: options.filters || {},
    diagnostics: options.diagnostics || null,
    champions: {
      valuation_label: champion.valuationLabel,
      future_shortlist_label: champion.futureShortlistLabel,
      prediction_service_base_role: champion.predictionBaseRole,
      shortlist_overlay_role: champion.shortlistOverlayRole,
    },
    items: (rows || []).map((row, idx) =>
      buildRecruitmentExportRow(row, {
        source: options.source || "workbench",
        split: options.split || state.split,
        rank: idx + 1,
        rankingDriver: options.rankingDriver || null,
        rankingBasis: options.rankingBasis || null,
      })
    ),
  };
}

function describeRecruitmentFilters(filters = {}) {
  const bits = [];
  if (filters.nonBig5Only) bits.push("outside Big 5");
  if (Number.isFinite(filters.minAge) && filters.minAge >= 0) bits.push(`age >= ${formatInt(filters.minAge)}`);
  if (Number.isFinite(filters.maxAge) && filters.maxAge >= 0) bits.push(`age <= ${formatInt(filters.maxAge)}`);
  if (Number.isFinite(filters.maxBudget) && filters.maxBudget > 0) bits.push(`budget ${budgetLabel(filters.maxBudget)}`);
  if (Number.isFinite(filters.maxContractYearsLeft) && filters.maxContractYearsLeft > 0) {
    bits.push(`contract <= ${formatNumber(filters.maxContractYearsLeft)}y`);
  }
  if (filters.roleNeed) bits.push(`role ${roleNeedLabel(filters.roleNeed)}`);
  return bits.length ? bits.join(" | ") : "no extra recruitment filters";
}

function seasonSortValue(season) {
  const s = String(season || "").trim();
  const m = s.match(/^(\d{4})\/(\d{2}|\d{4})$/);
  if (!m) return -1;
  return Number(m[1]);
}

function buildUrl(path, params = {}) {
  const base = state.apiBase.endsWith("/") ? state.apiBase : `${state.apiBase}/`;
  const url = new URL(path.replace(/^\//, ""), base);
  Object.entries(params).forEach(([k, v]) => {
    if (v === null || v === undefined || v === "") return;
    url.searchParams.set(k, String(v));
  });
  return url.toString();
}

async function getJson(path, params = {}) {
  const response = await fetch(buildUrl(path, params));
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    throw new Error(String(payload.detail || `HTTP ${response.status}`));
  }
  return payload;
}

async function requestJson(path, { method = "GET", params = {}, body = null } = {}) {
  const options = { method };
  if (body !== null && body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(buildUrl(path, params), options);
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    throw new Error(String(payload.detail || `HTTP ${response.status}`));
  }
  return payload;
}

async function fetchAllPredictions(params = {}) {
  const pageSize = 1000;
  let offset = 0;
  const out = [];
  while (true) {
    const payload = await getJson("/market-value/predictions", {
      limit: pageSize,
      offset,
      ...params,
    });
    const items = payload.items || [];
    out.push(...items);
    offset += items.length;
    if (!items.length || items.length < pageSize || offset >= Number(payload.total || 0)) {
      break;
    }
  }
  return out;
}

function setStatus(mode, message) {
  el.apiStatus.className = `status status--${mode}`;
  el.apiStatus.textContent = message;
}

function setLoading(loading) {
  state.loading = loading;
  el.connectBtn.disabled = loading;
  el.refresh.disabled = loading;
  el.prevBtn.disabled = loading;
  el.nextBtn.disabled = loading;
  if (loading) {
    el.tbody.innerHTML = "<tr><td colspan=\"9\">Loading data...</td></tr>";
  }
}

function setView(view) {
  state.view = view;
  Object.entries(el.views).forEach(([name, node]) => {
    if (!node) return;
    node.classList.toggle("view--active", name === view);
  });
  el.tabButtons.forEach((btn) => {
    const active = btn.dataset.view === view;
    btn.classList.toggle("is-active", active);
  });
}

function syncWorkbenchControlsFromState() {
  el.apiBase.value = state.apiBase;
  el.mode.value = state.mode;
  el.split.value = state.split;
  el.season.value = state.season;
  el.league.value = state.league;
  el.position.value = state.position;
  el.roleNeed.value = state.roleNeed;
  el.search.value = state.search;
  el.minMinutes.value = String(state.minMinutes);
  el.minAge.value = String(state.minAge);
  el.maxAge.value = String(state.maxAge);
  el.budgetBand.value = String(state.budgetBand || "");
  el.maxContractYears.value = state.maxContractYearsLeft ? String(state.maxContractYearsLeft) : "";
  el.minConfidence.value = String(state.minConfidence);
  el.minGap.value = String(state.minGapEur);
  el.topN.value = String(state.shortlistTopN);
  el.sort.value = state.sortBy;
  el.sortDir.value = state.sortOrder;
  el.limit.value = String(state.limit);
  el.outsideBig5Only.checked = state.nonBig5Only;
  el.undervaluedOnly.checked = state.undervaluedOnly;
  el.funnelSplit.value = state.split;
}

function readWorkbenchControlsToState() {
  state.apiBase = el.apiBase.value.trim() || DEFAULT_API;
  state.mode = el.mode.value;
  state.split = el.split.value;
  state.season = el.season.value;
  state.league = el.league.value;
  state.position = el.position.value;
  state.roleNeed = el.roleNeed.value;
  state.search = el.search.value.trim();
  state.minMinutes = Math.max(parseNumberOr(el.minMinutes.value, 0), 0);
  state.minAge = parseNumberOr(el.minAge.value, -1);
  state.maxAge = parseNumberOr(el.maxAge.value, -1);
  state.budgetBand = String(el.budgetBand.value || "");
  state.maxContractYearsLeft = parseOptionalPositive(el.maxContractYears.value);
  state.minConfidence = Math.max(parseNumberOr(el.minConfidence.value, 0), 0);
  state.minGapEur = Math.max(parseNumberOr(el.minGap.value, 0), 0);
  state.shortlistTopN = Math.max(Math.round(parseNumberOr(el.topN.value, 100)), 1);
  state.sortBy = el.sort.value;
  state.sortOrder = el.sortDir.value;
  state.limit = Math.max(Math.round(parseNumberOr(el.limit.value, 50)), 1);
  state.nonBig5Only = el.outsideBig5Only.checked;
  state.undervaluedOnly = el.undervaluedOnly.checked;
}

function renderTrustCard() {
  const health = state.health || {};
  const metrics = state.metrics || {};
  const artifacts = health.artifacts || {};
  const active = state.activeArtifacts || {};
  const champion = activeChampionRoutingSummary();
  const valuationMetricsMeta = manifestArtifactMeta("valuation", "metrics");
  const shortlistTestMeta = manifestArtifactMeta("future_shortlist", "test_predictions");
  const valuationUpdated = valuationMetricsMeta?.mtime_utc || "-";
  const shortlistUpdated = shortlistTestMeta?.mtime_utc || manifestArtifactMeta("future_shortlist", "metrics")?.mtime_utc || "-";

  const versionBits = [
    `test:${safeText(metrics.test_season)}`,
    `val:${safeText(metrics.val_season)}`,
  ];
  if (Number.isFinite(Number(metrics.trials_per_position))) {
    versionBits.push(`trials:${formatInt(metrics.trials_per_position)}`);
  }
  el.trustModelVersion.textContent = versionBits.join(" | ");

  el.trustUpdated.textContent = `valuation ${safeText(valuationUpdated)} | shortlist ${safeText(shortlistUpdated)}`;
  el.trustDataset.textContent = safeText(metrics.dataset || health.metrics_dataset || "-");
  el.trustSplits.textContent = `${safeText(metrics.val_season || health.metrics_val_season)} / ${safeText(metrics.test_season || health.metrics_test_season)}`;
  el.trustRows.textContent = `${formatInt(health.val_rows)} / ${formatInt(health.test_rows)}`;
  el.artifactTest.textContent = `${fileLabel(active.test_predictions_path || artifacts.test_predictions_path)} [${shortHash(
    active.test_predictions_sha256
  )}]`;
  el.artifactVal.textContent = `${fileLabel(active.val_predictions_path || artifacts.val_predictions_path)} [${shortHash(
    active.val_predictions_sha256
  )}]`;
  el.artifactMetrics.textContent = `${fileLabel(active.metrics_path || artifacts.metrics_path)} [${shortHash(
    active.metrics_sha256
  )}]`;
  if (el.championValuation) {
    el.championValuation.textContent = `${champion.valuationLabel} | ${fileLabel(active?.valuation?.metrics_path)}`;
  }
  if (el.championShortlist) {
    el.championShortlist.textContent = `${champion.futureShortlistLabel} | ${fileLabel(active?.future_shortlist?.test_predictions_path)}`;
  }
  if (el.championRuntime) {
    el.championRuntime.textContent = `${champion.predictionBaseRole} base -> ${champion.shortlistOverlayRole} overlay`;
  }

  const testSegments = metrics?.segments?.test || [];
  const under5 = testSegments.find((s) => s.segment === "under_5m");
  if (under5 && Number.isFinite(Number(under5.mape))) {
    const mape = Number(under5.mape);
    if (mape > 0.45) {
      el.trustNote.textContent =
        "For smaller-club recruitment, low-value (<€5m) prices are noisy. Treat the valuation champion as pricing guidance and the shortlist champion as ranking guidance.";
    } else {
      el.trustNote.textContent =
        "Runtime routing is healthy: valuation comes from the pricing champion and shortlist ranking keeps the future-scored overlay.";
    }
  } else {
    el.trustNote.textContent = "Connect API to load recruitment reliability diagnostics.";
  }
}

function renderMetrics() {
  const test = state.metrics?.overall?.test || {};
  const val = state.metrics?.overall?.val || {};
  const champion = activeChampionRoutingSummary();

  el.metricTestR2.textContent = Number.isFinite(test.r2) ? formatPct(test.r2) : "-";
  el.metricTestMae.textContent = Number.isFinite(test.mae_eur) ? formatCurrency(test.mae_eur) : "-";
  el.metricTestMape.textContent = Number.isFinite(test.mape) ? formatPct(test.mape) : "-";

  el.metricValR2.textContent = Number.isFinite(val.r2) ? formatPct(val.r2) : "-";
  el.metricValMae.textContent = Number.isFinite(val.mae_eur) ? formatCurrency(val.mae_eur) : "-";
  el.metricValMape.textContent = Number.isFinite(val.mape) ? formatPct(val.mape) : "-";

  const dataset = state.metrics?.dataset || "-";
  const valSeason = state.metrics?.val_season || "-";
  const testSeason = state.metrics?.test_season || "-";
  el.metricsMeta.textContent = `Dataset: ${dataset} | Val: ${valSeason} | Test: ${testSeason} | Valuation champion: ${champion.valuationLabel} | Shortlist champion: ${champion.futureShortlistLabel}`;
}

function renderSegmentTable() {
  const rows = state.metrics?.segments?.test || [];
  if (!rows.length) {
    el.segmentBody.innerHTML = "<tr><td colspan=\"5\">No segment metrics loaded.</td></tr>";
    el.segmentWarning.textContent = "-";
    return;
  }

  el.segmentBody.innerHTML = rows
    .map((r) => {
      const mapeBad = Number(r.mape) > 0.45;
      return `
        <tr>
          <td>${safeText(r.segment)}</td>
          <td class="num">${Number.isFinite(Number(r.r2)) ? formatPct(r.r2) : "-"}</td>
          <td class="num">${Number.isFinite(Number(r.mae_eur)) ? formatCurrency(r.mae_eur) : "-"}</td>
          <td class="num ${mapeBad ? "negative" : ""}">${Number.isFinite(Number(r.mape)) ? formatPct(r.mape) : "-"}</td>
          <td class="num">${Number.isFinite(Number(r.wmape)) ? formatPct(r.wmape) : "-"}</td>
        </tr>
      `;
    })
    .join("");

  const under5 = rows.find((r) => r.segment === "under_5m");
  if (under5 && Number.isFinite(Number(under5.mape)) && Number(under5.mape) > 0.45) {
    el.segmentWarning.textContent =
      "Warning: under_5m segment error is high. For recruitment, prioritize relative rank and conservative gap over exact pricing.";
  } else {
    el.segmentWarning.textContent = "Segment diagnostics are acceptable for ranking-based recruitment workflows.";
  }
}

function renderCoverageTable() {
  const benchmarkCoverage = Array.isArray(state.benchmark?.coverage?.rows) ? state.benchmark.coverage.rows : [];
  if (!state.coverageRows.length && !benchmarkCoverage.length) {
    el.coverageBody.innerHTML =
      "<tr><td colspan=\"6\">No coverage rows loaded yet. Connect the API and load benchmark diagnostics first.</td></tr>";
    return;
  }

  const rows = benchmarkCoverage.length
    ? benchmarkCoverage
    : (() => {
        const grouped = new Map();
        state.coverageRows.forEach((row) => {
          const league = getLeague(row) || "Unknown";
          if (!grouped.has(league)) {
            grouped.set(league, { league, rows: 0, undervalued: 0, confSum: 0, confN: 0 });
          }
          const g = grouped.get(league);
          g.rows += 1;

          const undervalued = Number(row.undervalued_flag);
          if (Number.isFinite(undervalued) ? undervalued > 0 : Number(row.value_gap_conservative_eur) > 0) {
            g.undervalued += 1;
          }

          const conf = Number(row.undervaluation_confidence);
          if (Number.isFinite(conf)) {
            g.confSum += conf;
            g.confN += 1;
          }
        });
        return Array.from(grouped.values())
          .map((row) => ({
            league: row.league,
            rows: row.rows,
            undervalued_share: row.rows > 0 ? row.undervalued / row.rows : NaN,
            avg_confidence: row.confN > 0 ? row.confSum / row.confN : NaN,
          }))
          .sort((a, b) => Number(b.rows || 0) - Number(a.rows || 0));
      })();
  el.coverageBody.innerHTML = rows
    .map((g) => {
      const status = summarizeLeagueStatus(g.league);
      return `
        <tr>
          <td>${safeText(g.league)}</td>
          <td class="num">${formatInt(g.rows)}</td>
          <td class="num">${Number.isFinite(Number(g.undervalued_share)) ? formatPct(Number(g.undervalued_share)) : "-"}</td>
          <td class="num">${Number.isFinite(Number(g.avg_confidence)) ? formatNumber(Number(g.avg_confidence)) : "-"}</td>
          <td>${buildBadgeChipMarkup(status.label, status.tone)}</td>
          <td>${escapeHtml(status.note)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderBenchmarkCards() {
  const payload = state.benchmark || {};
  const holdout = payload.league_holdout || {};
  const predictionLeague = payload.prediction_league || {};
  const hasHoldouts = Array.isArray(holdout.rows) && holdout.rows.length > 0;
  const summary = hasHoldouts ? holdout.summary || {} : predictionLeague.summary || {};
  const rows = hasHoldouts ? holdout.rows || [] : predictionLeague.rows || [];

  if (!rows.length) {
    el.benchmarkBody.innerHTML = "<tr><td colspan=\"5\">No benchmark data loaded.</td></tr>";
    el.benchmarkMeta.textContent = "No holdout benchmark rows available.";
  } else {
    el.benchmarkBody.innerHTML = rows
      .slice(0, 6)
      .map(
        (row) => `
        <tr>
          <td>${safeText(row.league)}</td>
          <td>${safeText(hasHoldouts ? row.status : "prediction")}</td>
          <td class="num">${Number.isFinite(Number(row.r2)) ? formatPct(Number(row.r2)) : "-"}</td>
          <td class="num">${Number.isFinite(Number(row.wmape)) ? formatPct(Number(row.wmape)) : "-"}</td>
          <td class="num">${
            hasHoldouts && Number.isFinite(Number(row.domain_shift_mean_abs_z))
              ? formatNumber(Number(row.domain_shift_mean_abs_z))
              : "-"
          }</td>
        </tr>
      `
      )
      .join("");
    el.benchmarkMeta.textContent = hasHoldouts
      ? `${formatInt(summary.ok_count || 0)} ok / ${formatInt(summary.total || 0)} holdouts | mean R² ${
          Number.isFinite(Number(summary.mean_r2)) ? formatPct(Number(summary.mean_r2)) : "-"
        } | median WMAPE ${
          Number.isFinite(Number(summary.median_wmape)) ? formatPct(Number(summary.median_wmape)) : "-"
        }`
      : `${formatInt(summary.total || 0)} prediction leagues | median R² ${
          Number.isFinite(Number(summary.median_r2)) ? formatPct(Number(summary.median_r2)) : "-"
        } | median WMAPE ${
          Number.isFinite(Number(summary.median_wmape)) ? formatPct(Number(summary.median_wmape)) : "-"
        }`;
  }

  const ablation = payload.ablation || {};
  const bestOverall = ablation.best_overall_test || {};
  const bestCheap = ablation.best_under_20m_test || {};
  const weakest = Array.isArray(ablation.weakest_full_slices_test) ? ablation.weakest_full_slices_test[0] : null;
  const onboardingCounts = payload.onboarding?.status_counts || {};

  el.experimentBestOverall.textContent = bestOverall.config
    ? `${humanizeExperimentLabel(bestOverall.config)} | ${humanizeKey(bestOverall.metric || "metric")} ${formatNumber(
        bestOverall.value
      )}`
    : "No ablation bundle loaded";
  el.experimentBestCheap.textContent = bestCheap.config
    ? `${humanizeExperimentLabel(bestCheap.config)} | ${humanizeKey(bestCheap.metric || "metric")} ${formatNumber(
        bestCheap.value
      )}`
    : "No under-20m winner loaded";
  el.experimentOnboarding.textContent = `${formatInt(onboardingCounts.ready || 0)} ready | ${formatInt(
    onboardingCounts.watch || 0
  )} watch | ${formatInt(onboardingCounts.blocked || 0)} blocked`;
  el.experimentWeakest.textContent = weakest
    ? `${safeText(weakest.slice_label)} | ${humanizeExperimentLabel(weakest.slice_type)} | WMAPE ${
        Number.isFinite(Number(weakest.wmape)) ? formatPct(Number(weakest.wmape)) : "-"
      }`
    : "No weak-slice diagnostics loaded";
}

function renderOverviewReadiness() {
  if (!el.overviewPosturePill) return;

  if (!state.metrics) {
    el.overviewPosturePill.className = "decision-pill decision-pill--neutral";
    el.overviewPosturePill.textContent = "Waiting";
    el.overviewPostureTitle.textContent = "Connect the API to judge whether the live recruitment workflow is ready.";
    el.overviewPostureCopy.textContent =
      "This page should answer, quickly, whether the platform is safe enough to use for live shortlist and pricing work today.";
    el.overviewPricingStatus.textContent = "-";
    el.overviewPricingCopy.textContent = "Connect API to load pricing diagnostics.";
    el.overviewRankingStatus.textContent = "-";
    el.overviewRankingCopy.textContent = "Connect API to load shortlist-routing posture.";
    el.overviewCoverageStatus.textContent = "-";
    el.overviewCoverageCopy.textContent =
      "Connect API to load where the workflow is investable and where it is still fragile.";
    return;
  }

  const overallTestMape = Number(state.metrics?.overall?.test?.mape);
  const overallTestR2 = Number(state.metrics?.overall?.test?.r2);
  const segments = state.metrics?.segments?.test || [];
  const under5 = segments.find((row) => row.segment === "under_5m");
  const under5Mape = Number(under5?.mape);
  const champion = activeChampionRoutingSummary();
  const onboardingCounts = state.benchmark?.onboarding?.status_counts || {};
  const blocked = Number(onboardingCounts.blocked) || 0;
  const watch = Number(onboardingCounts.watch) || 0;
  const ready = Number(onboardingCounts.ready) || 0;

  let pricingTone = "pursue";
  let pricingStatus = "Aggressive with guardrails";
  let pricingCopy = "Pricing is disciplined enough to support live shortlist work, provided you stay with conservative gaps.";
  if ((Number.isFinite(under5Mape) && under5Mape > 0.45) || (Number.isFinite(overallTestMape) && overallTestMape > 0.4)) {
    pricingTone = "watch";
    pricingStatus = "Guardrail-heavy";
    pricingCopy =
      "Low-value pricing is still noisy. Use the valuation layer for discipline, not for exact price certainty.";
  }
  if ((Number.isFinite(under5Mape) && under5Mape > 0.6) || (Number.isFinite(overallTestMape) && overallTestMape > 0.5)) {
    pricingTone = "price";
    pricingStatus = "Price with caution";
    pricingCopy = "Exact valuation is still too fragile in key segments. Treat the price layer as an anchor, not a verdict.";
  }

  let rankingTone = "pursue";
  let rankingStatus = "Shortlist-ready";
  let rankingCopy = `Valuation champion ${champion.valuationLabel} and shortlist champion ${champion.futureShortlistLabel} are live.`;
  if (!state.modelManifest || champion.futureShortlistLabel === "-") {
    rankingTone = "watch";
    rankingStatus = "Routing unclear";
    rankingCopy = "Shortlist routing is not clearly surfaced. Analysts will need to stay closer to diagnostics.";
  } else if ((Number.isFinite(overallTestR2) && overallTestR2 < 0.45) || champion.futureShortlistLabel === champion.valuationLabel) {
    rankingTone = "watch";
    rankingStatus = "Functional, not decisive";
    rankingCopy = "Ranking is live, but the workflow still leans heavily on pricing posture and manual judgement.";
  }

  let coverageTone = "pursue";
  let coverageStatus = "Deployable across active leagues";
  let coverageCopy = `${formatInt(ready)} ready | ${formatInt(watch)} watch | ${formatInt(blocked)} blocked in current onboarding diagnostics.`;
  if (watch > 0 || blocked > 0) {
    coverageTone = "watch";
    coverageStatus = "Selective deployment";
    coverageCopy = `${formatInt(ready)} ready leagues, ${formatInt(watch)} watch leagues, ${formatInt(blocked)} blocked leagues. Stay selective on coverage quality.`;
  }
  if (blocked > ready && blocked > 0) {
    coverageTone = "price";
    coverageStatus = "Coverage still patchy";
    coverageCopy = "Too many leagues remain blocked or noisy for broad live deployment. Lean on league guardrails before acting.";
  }

  const tones = [pricingTone, rankingTone, coverageTone];
  const overallTone = tones.includes("price")
    ? "price"
    : tones.includes("watch")
    ? "watch"
    : tones.includes("pass")
    ? "pass"
    : "pursue";
  const overallLabel =
    overallTone === "pursue"
      ? "Operational"
      : overallTone === "watch"
      ? "Selective"
      : overallTone === "price"
      ? "Cautious"
      : "Blocked";
  const overallCopy =
    overallTone === "pursue"
      ? "The workflow is good enough for live shortlist decisions, with the memo and guardrails acting as the final decision layer."
      : overallTone === "watch"
      ? "The workflow can support live work, but you should be selective about leagues and avoid over-trusting exact price points."
      : overallTone === "price"
      ? "The platform still behaves more like a pricing and research aid than a fully trusted recruitment operating surface."
      : "The workflow should not drive live recruitment decisions yet.";

  el.overviewPosturePill.className = `decision-pill decision-pill--${overallTone}`;
  el.overviewPosturePill.textContent = overallLabel;
  el.overviewPostureTitle.textContent =
    overallTone === "pursue"
      ? "The live recruitment workflow is ready for shortlist work."
      : overallTone === "watch"
      ? "The live workflow is usable, but only with visible caution."
      : "The live workflow is not yet fully trustworthy as a recruitment operating surface.";
  el.overviewPostureCopy.textContent = overallCopy;
  el.overviewPricingStatus.textContent = pricingStatus;
  el.overviewPricingCopy.textContent = pricingCopy;
  el.overviewRankingStatus.textContent = rankingStatus;
  el.overviewRankingCopy.textContent = rankingCopy;
  el.overviewCoverageStatus.textContent = coverageStatus;
  el.overviewCoverageCopy.textContent = coverageCopy;
}

function renderWatchlist() {
  if (!el.watchlistBody || !el.watchlistMeta) return;
  const rows = Array.isArray(state.watchlistRows) ? state.watchlistRows : [];
  if (!rows.length) {
    el.watchlistBody.innerHTML = '<tr><td colspan="7">No watchlist entries yet.</td></tr>';
    el.watchlistMeta.textContent = "Watchlist empty.";
    return;
  }
  el.watchlistBody.innerHTML = rows
    .map((row) => {
      const watchId = safeText(row.watch_id);
      return `
        <tr>
          <td>${safeText(row.name)}</td>
          <td>${safeText(row.league)}</td>
          <td class="num">${formatCurrency(row.value_gap_capped_eur)}</td>
          <td class="num">${formatNumber(row.undervaluation_confidence)}</td>
          <td>${safeText(row.tag)}</td>
          <td>${safeText(row.created_at_utc)}</td>
          <td><button type="button" class="btn-ghost watchlist-delete" data-watch-id="${watchId}">Remove</button></td>
        </tr>
      `;
    })
    .join("");
  el.watchlistMeta.textContent = `${formatInt(rows.length)} shown / ${formatInt(state.watchlistTotal)} saved entries`;
}

async function refreshWatchlist() {
  if (!backendReady()) {
    state.watchlistRows = [];
    state.watchlistTotal = 0;
    renderWatchlist();
    el.watchlistMeta.textContent = "Watchlist unavailable until backend artifacts are ready.";
    return;
  }
  try {
    const payload = await requestJson("/market-value/watchlist", {
      method: "GET",
      params: { split: state.split, limit: 50, offset: 0 },
    });
    state.watchlistRows = payload.items || [];
    state.watchlistTotal = Number(payload.total) || state.watchlistRows.length;
    renderWatchlist();
  } catch (err) {
    state.watchlistRows = [];
    state.watchlistTotal = 0;
    renderWatchlist();
    el.watchlistMeta.textContent = err instanceof Error ? err.message : String(err);
  }
}

async function addSelectedToWatchlist() {
  if (!state.selectedRow) {
    if (el.watchlistMeta) el.watchlistMeta.textContent = "Select a player first.";
    return;
  }
  const playerId = String(state.selectedRow.player_id || "").trim();
  if (!playerId) {
    if (el.watchlistMeta) el.watchlistMeta.textContent = "Selected row has no player_id.";
    return;
  }
  const season = String(state.selectedRow.season || state.season || "").trim();
  const tag = (el.watchlistTag?.value || "").trim();
  const notes = (el.watchlistNotes?.value || "").trim();
  if (el.watchlistMeta) el.watchlistMeta.textContent = "Saving watchlist entry...";
  try {
    await requestJson("/market-value/watchlist/items", {
      method: "POST",
      body: {
        player_id: playerId,
        split: state.split,
        season: season || null,
        tag: tag || null,
        notes: notes || null,
        source: "frontend_workbench",
      },
    });
    if (el.watchlistNotes) el.watchlistNotes.value = "";
    await refreshWatchlist();
    if (el.watchlistMeta) el.watchlistMeta.textContent = "Recruitment watchlist entry saved.";
  } catch (err) {
    if (el.watchlistMeta) el.watchlistMeta.textContent = err instanceof Error ? err.message : String(err);
  }
}

async function deleteWatchlistItem(watchId) {
  const id = String(watchId || "").trim();
  if (!id) return;
  try {
    await requestJson(`/market-value/watchlist/items/${encodeURIComponent(id)}`, { method: "DELETE" });
    await refreshWatchlist();
  } catch (err) {
    if (el.watchlistMeta) el.watchlistMeta.textContent = err instanceof Error ? err.message : String(err);
  }
}

function applyClientFilters(rows) {
  let out = [...rows];
  if (state.season) out = out.filter((r) => String(r.season || "") === state.season);
  if (state.league) {
    const leagueFold = state.league.toLowerCase();
    out = out.filter((r) => String(r.league || "").toLowerCase() === leagueFold);
  }
  if (state.position) out = out.filter((r) => getPosition(r) === state.position);
  out = out.filter((r) => matchesRecruitmentWorkflow(r, currentWorkbenchWorkflowSummary()));
  if (state.search) {
    const q = state.search.toLowerCase();
    out = out.filter((r) => {
      const name = String(r.name || "").toLowerCase();
      const club = String(r.club || "").toLowerCase();
      return name.includes(q) || club.includes(q);
    });
  }
  return out;
}

function applyClientSort(rows) {
  const out = [...rows];
  const key = state.sortBy;
  const factor = state.sortOrder === "asc" ? 1 : -1;
  out.sort((a, b) => {
    if (key === "value_gap_conservative_eur" || key === "value_gap_capped_eur") {
      const avGap = conservativeGapForRanking(a);
      const bvGap = conservativeGapForRanking(b);
      if (Number.isFinite(avGap) && Number.isFinite(bvGap) && avGap !== bvGap) {
        return (avGap - bvGap) * factor;
      }
      const avConf = Number(a.undervaluation_confidence);
      const bvConf = Number(b.undervaluation_confidence);
      if (Number.isFinite(avConf) && Number.isFinite(bvConf) && avConf !== bvConf) {
        return (avConf - bvConf) * factor;
      }
    }
    const avNum = Number(a[key]);
    const bvNum = Number(b[key]);
    if (Number.isFinite(avNum) && Number.isFinite(bvNum)) return (avNum - bvNum) * factor;
    return String(a[key] || "").localeCompare(String(b[key] || "")) * factor;
  });
  return out;
}

function renderRows() {
  if (!state.rows.length) {
    const emptyMessage =
      state.mode === "shortlist"
        ? "No recruitment targets matched the current filters. Lower the minutes / age thresholds or switch to Valuation Board to inspect the full artifact."
        : "No valuation rows matched the current filters. Relax the manual sort filters or switch to Recruitment Board for score-driven ranking.";
    el.tbody.innerHTML = `<tr><td colspan="9">${emptyMessage}</td></tr>`;
    el.resultCount.textContent = "0 rows";
    el.resultRange.textContent = "offset 0";
    return;
  }

  el.tbody.innerHTML = state.rows
    .map((row, idx) => {
      const consGap = conservativeGapForRanking(row);
      const selected = state.selectedRow && rowKey(state.selectedRow) === rowKey(row) ? " selected-row" : "";
      const decision = summarizeRecruitmentDecision(row, null, {
        source: state.mode === "predictions" ? "predictions" : "workbench",
      });
      const scoreContext = resolveRowScoreContext(
        row,
        state.mode === "shortlist" ? state.queryDiagnostics : null,
        state.mode === "shortlist" ? "shortlist" : "predictions"
      );
      const badges = buildProvenanceBadges(row)
        .map((badge) => buildBadgeChipMarkup(badge.label, badge.tone))
        .join("");
      return `
        <tr data-index="${idx}" class="${selected.trim()}">
          <td class="player-cell">
            <strong>${safeText(row.name)}</strong>
            <span class="player-cell__sub">${safeText(row.club)} | ${safeText(row.league)} | ${safeText(row.season)}</span>
            <div class="player-cell__badges">${badges}</div>
            <span class="player-cell__note">${escapeHtml(scoreContext.scoreLabel)}${
              Number.isFinite(scoreContext.scoreValue)
                ? ` | ${scoreContext.scoreColumn.endsWith("_eur") ? formatCurrency(scoreContext.scoreValue) : formatNumber(scoreContext.scoreValue)}`
                : ""
            }</span>
          </td>
          <td>
            <div class="table-status">
              ${buildDecisionPillMarkup(decision)}
              <span class="table-status__note">${escapeHtml(decision.gapNote)}</span>
            </div>
          </td>
          <td>${safeText(getPosition(row))}</td>
          <td class="num">${formatNumber(row.age)}</td>
          <td class="num">${formatCurrency(row.market_value_eur)}</td>
          <td class="num ${consGap >= 0 ? "positive" : "negative"}">${formatCurrency(consGap)}</td>
          <td class="num">${formatNumber(row.undervaluation_confidence)}</td>
          <td class="num">${formatInt(getMinutes(row))}</td>
          <td><span class="action-copy">${escapeHtml(decision.nextAction)}</span></td>
        </tr>
      `;
    })
    .join("");

  const start = state.total === 0 ? 0 : state.offset + 1;
  const end = Math.min(state.offset + state.count, state.total);
  el.resultCount.textContent = `${formatInt(state.count)} / ${formatInt(state.total)} rows`;
  el.resultRange.textContent = `showing ${start}-${end}`;
}

function renderPager() {
  el.prevBtn.disabled = state.loading || state.offset <= 0;
  el.nextBtn.disabled = state.loading || state.offset + state.count >= state.total;
}

function renderArchetypeProfile(playerType = null, { loading = false, error = "" } = {}) {
  if (!el.detailArchetype || !el.detailArchetypeCandidates) return;
  if (loading) {
    el.detailArchetype.textContent = "Loading archetype profile...";
    el.detailArchetypeCandidates.innerHTML = '<li class="risk-item risk-item--none">Loading archetype candidates...</li>';
    return;
  }
  if (error) {
    el.detailArchetype.textContent = "Archetype profile unavailable for this player.";
    el.detailArchetypeCandidates.innerHTML = '<li class="risk-item risk-item--none">No archetype candidates available.</li>';
    return;
  }
  if (!playerType || typeof playerType !== "object") {
    el.detailArchetype.textContent = "Select a player to estimate archetype fit.";
    el.detailArchetypeCandidates.innerHTML = '<li class="risk-item risk-item--none">No archetype candidates loaded.</li>';
    return;
  }

  const archetype = safeText(playerType.archetype);
  const tier = safeText(playerType.tier).toLowerCase();
  const conf = Number(playerType.confidence_0_to_1);
  const confText = Number.isFinite(conf) ? formatPct(conf) : "-";
  const summary = playerType.summary_text
    ? `${playerType.summary_text} Confidence: ${confText}.`
    : `${archetype} profile (${tier} confidence ${confText}).`;
  el.detailArchetype.textContent = summary;

  const candidates = Array.isArray(playerType.candidates) ? playerType.candidates : [];
  if (!candidates.length) {
    el.detailArchetypeCandidates.innerHTML = '<li class="risk-item risk-item--none">No archetype candidates available.</li>';
    return;
  }
  el.detailArchetypeCandidates.innerHTML = candidates
    .slice(0, 3)
    .map((cand) => {
      const score = Number(cand?.score_0_to_1);
      const coverage = Number(cand?.coverage_0_to_1);
      const scoreText = Number.isFinite(score) ? formatPct(score) : "-";
      const coverageText = Number.isFinite(coverage) ? formatPct(coverage) : "-";
      return `<li class="risk-item">${safeText(cand?.name)} (fit ${scoreText} | coverage ${coverageText})</li>`;
    })
    .join("");
}

function renderFormationFitProfile(formationFit = null, { loading = false, error = "" } = {}) {
  if (!el.detailFormations || !el.detailFormationSummary) return;
  if (loading) {
    el.detailFormationSummary.textContent = "Loading formation fit...";
    el.detailFormations.innerHTML = '<li class="risk-item risk-item--none">Loading formation fit...</li>';
    return;
  }
  if (error) {
    el.detailFormationSummary.textContent = "Formation fit unavailable for this player.";
    el.detailFormations.innerHTML = '<li class="risk-item risk-item--none">Formation fit unavailable for this player.</li>';
    return;
  }
  const recommended =
    formationFit && typeof formationFit === "object" && Array.isArray(formationFit.recommended)
      ? formationFit.recommended
      : [];
  el.detailFormationSummary.textContent = formationFit?.summary_text
    ? safeText(formationFit.summary_text)
    : "Select a player to estimate formation fit.";
  if (!recommended.length) {
    el.detailFormations.innerHTML = '<li class="risk-item risk-item--none">No formation fit recommendations available.</li>';
    return;
  }
  el.detailFormations.innerHTML = recommended
    .slice(0, 3)
    .map((rec) => {
      const fit = Number(rec?.fit_score_0_to_1);
      const coverage = Number(rec?.coverage_0_to_1);
      const fitText = Number.isFinite(fit) ? formatPct(fit) : "-";
      const coverageText = Number.isFinite(coverage) ? formatPct(coverage) : "-";
      const matches = Array.isArray(rec?.matched_metrics) ? rec.matched_metrics : [];
      const matchHtml = matches.length
        ? `<ul class="formation-match-list">${matches
            .slice(0, 4)
            .map((part) => {
              const score = Number(part?.observed);
              return `<li>${escapeHtml(safeText(part?.label))}: ${Number.isFinite(score) ? formatPct(score) : "-"}</li>`;
            })
            .join("")}</ul>`
        : "";
      return `
        <li class="risk-item">
          <div class="risk-head">
            <span class="risk-severity risk-severity--medium">${safeText(rec?.fit_tier)}</span>
            <span class="risk-code">${safeText(rec?.formation)} | ${safeText(rec?.role)}</span>
          </div>
          <p class="formation-summary">Fit ${fitText} | coverage ${coverageText}</p>
          ${matchHtml}
        </li>
      `;
    })
    .join("");
}

function renderDetailTab() {
  el.detailTabButtons.forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.detailTab === state.activeDetailTab);
  });
  el.detailPanels.forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.detailPanel === state.activeDetailTab);
  });
}

function setDetailTab(tab) {
  state.activeDetailTab = tab || "overview";
  renderDetailTab();
}

function buildConfidenceSummary(row, report = null) {
  const guardrails = report?.valuation_guardrails || {};
  const confidence = report?.confidence || {};
  const label = safeText(confidence.label || "medium");
  const signal = firstFiniteNumber(confidence.score_0_to_1, row.undervaluation_confidence);
  const capped = firstFiniteNumber(guardrails.value_gap_capped_eur, row.value_gap_capped_eur);
  const raw = firstFiniteNumber(guardrails.value_gap_conservative_eur, row.value_gap_conservative_eur);
  const capApplied = guardrails.cap_applied === true || firstFiniteNumber(row.value_gap_cap_applied) > 0;
  const parts = [
    `${label} confidence signal`,
    Number.isFinite(signal) ? `score ${formatNumber(signal)}` : null,
    Number.isFinite(capped) ? `capped gap ${formatCurrency(capped)}` : null,
    Number.isFinite(raw) ? `raw conservative gap ${formatCurrency(raw)}` : null,
    capApplied ? "guardrail cap applied" : "no cap applied",
  ].filter(Boolean);
  return parts.join(" | ");
}

function renderSimilarPlayers(similarPayload = null, { loading = false, error = "" } = {}) {
  if (!el.detailSimilar || !el.detailSimilarSummary) return;
  if (loading) {
    el.detailSimilarSummary.textContent = "Loading similar players...";
    el.detailSimilar.innerHTML = '<li class="risk-item risk-item--none">Loading similar players...</li>';
    return;
  }
  if (error) {
    el.detailSimilarSummary.textContent = `Similar-player search unavailable: ${error}`;
    el.detailSimilar.innerHTML = '<li class="risk-item risk-item--none">Similar-player matches unavailable.</li>';
    return;
  }
  const available = similarPayload?.available === true;
  const items = Array.isArray(similarPayload?.items) ? similarPayload.items : [];
  if (!available) {
    el.detailSimilarSummary.textContent = similarPayload?.reason
      ? `Similar-player search unavailable: ${safeText(similarPayload.reason)}`
      : "Similar-player search unavailable for this player.";
    el.detailSimilar.innerHTML = '<li class="risk-item risk-item--none">Similar-player matches unavailable.</li>';
    return;
  }
  el.detailSimilarSummary.textContent = items.length
    ? `Top ${items.length} nearest-profile matches from the similarity index.`
    : "No similar-player matches returned.";
  if (!items.length) {
    el.detailSimilar.innerHTML = '<li class="risk-item risk-item--none">No similar-player matches returned.</li>';
    return;
  }
  el.detailSimilar.innerHTML = items
    .map((item) => {
      const meta = [
        item?.club ? safeText(item.club) : null,
        item?.league ? safeText(item.league) : null,
        item?.position ? safeText(item.position) : null,
      ]
        .filter(Boolean)
        .join(" | ");
      return `
        <li class="risk-item">
          <div class="risk-head">
            <span class="risk-code">${escapeHtml(safeText(item?.name || item?.player_id))}</span>
            <span class="risk-severity risk-severity--medium">${Number.isFinite(Number(item?.score)) ? formatNumber(item.score) : "-"}</span>
          </div>
          <p class="risk-message">${escapeHtml(meta || safeText(item?.player_id))}</p>
          <p class="formation-summary">${escapeHtml(safeText(item?.justification || ""))}</p>
        </li>
      `;
    })
    .join("");
}

function renderRadarProfile(radarProfile = null, { loading = false, error = "" } = {}) {
  if (!el.detailRadar || !el.detailRadarMeta) return;
  if (loading) {
    el.detailRadar.innerHTML = "";
    el.detailRadarMeta.textContent = "Loading radar profile...";
    return;
  }
  if (error) {
    el.detailRadar.innerHTML = "";
    el.detailRadarMeta.textContent = "Radar profile unavailable for this player.";
    return;
  }
  if (!radarProfile || typeof radarProfile !== "object") {
    el.detailRadar.innerHTML = "";
    el.detailRadarMeta.textContent = "Select a player to render radar profile.";
    return;
  }

  const axes = Array.isArray(radarProfile.axes) ? radarProfile.axes : [];
  const usable = axes
    .filter((axis) => axis && axis.available)
    .map((axis) => {
      const n = Number(axis.normalized_0_to_100);
      return Number.isFinite(n)
        ? {
            label: safeText(axis.label),
            normalized: Math.max(0, Math.min(100, n)),
          }
        : null;
    })
    .filter(Boolean);

  const coverage = Number(radarProfile.coverage_0_to_1);
  const coverageText = Number.isFinite(coverage) ? formatPct(coverage) : "-";
  if (usable.length < 3) {
    el.detailRadar.innerHTML = "";
    el.detailRadarMeta.textContent = `Radar unavailable (coverage ${coverageText}).`;
    return;
  }

  const cx = 160;
  const cy = 160;
  const radius = 108;
  const levels = [0.25, 0.5, 0.75, 1.0];
  const step = (Math.PI * 2) / usable.length;
  const angleAt = (idx) => -Math.PI / 2 + idx * step;
  const pointAt = (idx, scale) => {
    const a = angleAt(idx);
    const r = radius * scale;
    return {
      x: cx + r * Math.cos(a),
      y: cy + r * Math.sin(a),
    };
  };
  const pointsToString = (scaleFn) =>
    usable
      .map((_, idx) => {
        const p = scaleFn(idx);
        return `${p.x.toFixed(2)},${p.y.toFixed(2)}`;
      })
      .join(" ");

  const grid = levels
    .map((lvl) => {
      const pts = pointsToString((idx) => pointAt(idx, lvl));
      return `<polygon points="${pts}" class="radar-grid" />`;
    })
    .join("");
  const axesLines = usable
    .map((_, idx) => {
      const p = pointAt(idx, 1);
      return `<line x1="${cx}" y1="${cy}" x2="${p.x.toFixed(2)}" y2="${p.y.toFixed(2)}" class="radar-axis" />`;
    })
    .join("");
  const labels = usable
    .map((axis, idx) => {
      const p = pointAt(idx, 1.16);
      const c = Math.cos(angleAt(idx));
      const anchor = Math.abs(c) < 0.18 ? "middle" : c > 0 ? "start" : "end";
      return `<text x="${p.x.toFixed(2)}" y="${p.y.toFixed(2)}" text-anchor="${anchor}" class="radar-label">${escapeHtml(
        axis.label
      )}</text>`;
    })
    .join("");
  const profilePts = pointsToString((idx) => pointAt(idx, usable[idx].normalized / 100.0));
  const profileNodes = usable
    .map((axis, idx) => {
      const p = pointAt(idx, axis.normalized / 100.0);
      return `<circle cx="${p.x.toFixed(2)}" cy="${p.y.toFixed(2)}" r="3.6" class="radar-node">
        <title>${escapeHtml(axis.label)}: ${formatNumber(axis.normalized)}/100</title>
      </circle>`;
    })
    .join("");
  el.detailRadar.innerHTML = `
    <g>
      ${grid}
      ${axesLines}
      <polygon points="${profilePts}" class="radar-profile"></polygon>
      ${profileNodes}
      ${labels}
    </g>
  `;
  const ready = radarProfile.ready_for_plot === true ? "ready" : "limited";
  el.detailRadarMeta.textContent = `Radar ${ready}. Coverage: ${coverageText}.`;
}

function renderNarrativeMetricList(target, items, emptyMessage, valueKey = "quality_percentile", valueLabel = "") {
  if (!target) return;
  if (!Array.isArray(items) || !items.length) {
    target.innerHTML = `<li class="risk-item risk-item--none">${escapeHtml(emptyMessage)}</li>`;
    return;
  }
  target.innerHTML = items
    .slice(0, 5)
    .map((item) => {
      const value = Number(item?.[valueKey]);
      const suffix = Number.isFinite(value)
        ? valueKey.endsWith("_0_to_1") || valueKey.includes("percentile")
          ? formatPct(value)
          : formatNumber(value)
        : "-";
      const descriptor = valueLabel ? `${valueLabel} ${suffix}` : suffix;
      return `<li class="risk-item">${escapeHtml(safeText(item?.label))}${descriptor !== "-" ? ` (${descriptor})` : ""}</li>`;
    })
    .join("");
}

function buildSignalListMarkup(items, emptyMessage = "No provider signals loaded.") {
  if (!Array.isArray(items) || !items.length) {
    return `<li class="risk-item risk-item--none">${escapeHtml(emptyMessage)}</li>`;
  }
  return items
    .slice(0, 6)
    .map(
      (item) => `
        <li class="risk-item">
          <div class="risk-head">
            <span class="risk-code">${escapeHtml(safeText(item?.label))}</span>
            <span class="risk-severity risk-severity--medium">${escapeHtml(safeText(item?.display_value || "-"))}</span>
          </div>
        </li>
      `
    )
    .join("");
}

function renderSignalList(target, payload, emptyMessage) {
  if (!target) return;
  target.innerHTML = buildSignalListMarkup(payload?.signals, emptyMessage);
}

function closeProfileModal() {
  state.profileModalOpen = false;
  if (el.profileModal) el.profileModal.hidden = true;
  document.body.classList.remove("modal-open");
}

function renderProfileModal(row, { profile = null, reportError = "" } = {}) {
  if (!el.profileModalBody || !el.profileModalTitle || !el.profileModalMeta) return;
  if (!row) {
    el.profileModalTitle.textContent = "Player Profile";
    el.profileModalMeta.textContent = "Select a player to inspect the full profile.";
    el.profileModalBody.innerHTML = '<p class="detail-summary">Select a player to inspect the full profile.</p>';
    return;
  }
  const reportPlayer = profile?.player && typeof profile.player === "object" ? profile.player : {};
  const mergedRow = { ...row, ...reportPlayer };
  const statGroups = Array.isArray(profile?.stat_groups) ? profile.stat_groups : buildStatGroups(mergedRow, profile, null);
  const gaps = deriveGapValues(mergedRow, profile);
  const summaryText = reportError
    ? `Recruitment memo unavailable: ${reportError}`
    : profile?.summary_text || "No memo summary loaded for this player.";
  const bestFormation =
    Array.isArray(profile?.formation_fit?.recommended) && profile.formation_fit.recommended.length
      ? profile.formation_fit.recommended[0]
      : null;
  const strengths = Array.isArray(profile?.strengths) ? profile.strengths : [];
  const weaknesses = Array.isArray(profile?.weaknesses) ? profile.weaknesses : [];
  const levers = Array.isArray(profile?.development_levers) ? profile.development_levers : [];
  const risks = Array.isArray(profile?.risk_flags) ? profile.risk_flags : [];
  const historyPayload = profile?.history_strength && typeof profile.history_strength === "object" ? profile.history_strength : null;
  const similar = profile?.similar_players && typeof profile.similar_players === "object" ? profile.similar_players : null;
  const tacticalContext = profile?.external_tactical_context && typeof profile.external_tactical_context === "object" ? profile.external_tactical_context : null;
  const availabilityContext = profile?.availability_context && typeof profile.availability_context === "object" ? profile.availability_context : null;
  const marketContext = profile?.market_context && typeof profile.market_context === "object" ? profile.market_context : null;
  const rankingDiagnostics =
    Number.isFinite(Number(row?._funnelScore)) && state.funnelDiagnostics ? state.funnelDiagnostics : state.queryDiagnostics;
  const scoreContext = resolveRowScoreContext(
    mergedRow,
    rankingDiagnostics,
    Number.isFinite(Number(row?._funnelScore)) ? "funnel" : state.mode === "shortlist" ? "shortlist" : "predictions"
  );
  const roleLens = buildRoleLens(mergedRow, profile);
  const provenanceBadges = buildProvenanceBadges(mergedRow)
    .map((badge) => buildBadgeChipMarkup(badge.label, badge.tone))
    .join("");
  const coverageWarnings = detailCoverageWarnings(mergedRow, profile);
  const whyRanked = whyRankedItems(mergedRow, profile, scoreContext);
  const radarMarkup = el.detailRadar ? el.detailRadar.innerHTML : "";
  const radarMeta = el.detailRadarMeta?.textContent || "Radar unavailable.";
  const formationMarkup = Array.isArray(profile?.formation_fit?.recommended)
    ? profile.formation_fit.recommended
        .map((rec) => {
          const matched = Array.isArray(rec?.matched_metrics) ? rec.matched_metrics : [];
          return `
            <li class="risk-item">
              <div class="risk-head">
                <span class="risk-severity risk-severity--${escapeHtml(safeText(rec?.fit_tier || "low").toLowerCase())}">${escapeHtml(
            safeText(rec?.fit_tier || "low")
          )}</span>
                <span class="risk-code">${escapeHtml(safeText(rec?.formation))} | ${escapeHtml(safeText(rec?.role))}</span>
              </div>
              <p class="formation-summary">Fit ${formatPct(rec?.fit_score_0_to_1)} | coverage ${formatPct(
            rec?.coverage_0_to_1
          )}</p>
              ${
                matched.length
                  ? `<ul class="formation-match-list">${matched
                      .slice(0, 5)
                      .map((part) => `<li>${escapeHtml(safeText(part?.label))}: ${formatPct(part?.observed)}</li>`)
                      .join("")}</ul>`
                  : ""
              }
            </li>
          `;
        })
        .join("")
    : DETAIL_EMPTY_LIST_ITEM;

  el.profileModalTitle.textContent = safeText(mergedRow.name);
  el.profileModalMeta.textContent = `${safeText(mergedRow.club)} | ${safeText(mergedRow.league)} | ${safeText(
    getPosition(mergedRow)
  )} | ${safeText(mergedRow.season)}`;
  el.profileModalBody.innerHTML = `
    <section class="profile-hero">
      <article class="profile-card">
        <h3>Recruitment Summary</h3>
        <p class="detail-summary">${escapeHtml(summaryText)}</p>
        <div class="badge-row">${provenanceBadges}</div>
        <p class="detail-summary profile-note">${escapeHtml(
          `${scoreContext.scoreLabel}${Number.isFinite(scoreContext.scoreValue) ? ` | ${scoreContext.scoreColumn.endsWith("_eur") ? formatCurrency(scoreContext.scoreValue) : formatNumber(scoreContext.scoreValue)}` : ""} | ${scoreContext.rankingBasisLabel}.`
        )}</p>
      </article>
      <article class="profile-card">
        <h3>Role Lens</h3>
        <p class="detail-summary">${escapeHtml(roleLens.summary)}</p>
        <ul class="risk-list">${
          roleLens.metrics.length
            ? roleLens.metrics
                .map(
                  (metric) => `
                    <li class="risk-item">
                      <div class="risk-head">
                        <span class="risk-code">${escapeHtml(metric.label)}</span>
                        <span class="risk-severity risk-severity--medium">${escapeHtml(metric.displayValue)}</span>
                      </div>
                    </li>
                  `
                )
                .join("")
            : DETAIL_EMPTY_LIST_ITEM
        }</ul>
      </article>
      <article class="profile-card">
        <h3>Guardrails</h3>
        <p class="detail-summary">${escapeHtml(buildConfidenceSummary(mergedRow, profile))}</p>
      </article>
      <article class="profile-card">
        <h3>Coverage Warnings</h3>
        <ul class="risk-list">${buildNarrativeListMarkup(coverageWarnings, "Coverage diagnostics look acceptable for shortlist review.")}</ul>
      </article>
    </section>

    <section class="profile-metric-grid">
      <article class="profile-metric">
        <p class="profile-metric__label">Market</p>
        <p class="profile-metric__value">${formatCurrency(mergedRow.market_value_eur)}</p>
      </article>
      <article class="profile-metric">
        <p class="profile-metric__label">Expected</p>
        <p class="profile-metric__value">${formatCurrency(
          firstFiniteNumber(profile?.valuation_guardrails?.fair_value_eur, mergedRow.expected_value_eur)
        )}</p>
      </article>
      <article class="profile-metric">
        <p class="profile-metric__label">Capped Gap</p>
        <p class="profile-metric__value">${formatCurrency(gaps.capped)}</p>
      </article>
      <article class="profile-metric">
        <p class="profile-metric__label">Confidence</p>
        <p class="profile-metric__value">${formatNumber(mergedRow.undervaluation_confidence)}</p>
      </article>
    </section>

    <section class="profile-grid">
      <article class="profile-card profile-grid__span-4">
        <h3>Strengths</h3>
        <ul class="risk-list">${
          strengths.length
            ? strengths
                .slice(0, 5)
                .map((item) => `<li class="risk-item">${escapeHtml(safeText(item.label))}</li>`)
                .join("")
            : DETAIL_EMPTY_LIST_ITEM
        }</ul>
      </article>
      <article class="profile-card profile-grid__span-4">
        <h3>Weaknesses</h3>
        <ul class="risk-list">${
          weaknesses.length
            ? weaknesses
                .slice(0, 5)
                .map((item) => `<li class="risk-item">${escapeHtml(safeText(item.label))}</li>`)
                .join("")
            : DETAIL_EMPTY_LIST_ITEM
        }</ul>
      </article>
      <article class="profile-card profile-grid__span-4">
        <h3>Development Levers</h3>
        <ul class="risk-list">${
          levers.length
            ? levers
                .slice(0, 5)
                .map((item) => `<li class="risk-item">${escapeHtml(safeText(item.label))}</li>`)
                .join("")
            : DETAIL_EMPTY_LIST_ITEM
        }</ul>
      </article>

      <article class="profile-card profile-grid__span-6">
        <h3>Formation Fit</h3>
        <p class="detail-summary">${escapeHtml(safeText(profile?.formation_fit?.summary_text || "No formation fit summary."))}</p>
        <ul class="risk-list">${formationMarkup || DETAIL_EMPTY_LIST_ITEM}</ul>
      </article>
      <article class="profile-card profile-grid__span-6">
        <h3>Role Radar</h3>
        <div class="radar-wrap">
          <svg class="radar-chart" viewBox="0 0 320 320" role="img" aria-label="Player role radar profile">${radarMarkup}</svg>
        </div>
        <p class="detail-summary">${escapeHtml(radarMeta)}</p>
      </article>

      <article class="profile-card profile-grid__span-4">
        <h3>History Strength</h3>
        <p class="detail-summary">${
          historyPayload?.summary_text
            ? `${escapeHtml(historyPayload.summary_text)} Coverage ${formatPct(historyPayload.coverage_0_to_1)} | score ${formatNumber(
                historyPayload.score_0_to_100
              )}/100`
            : "History-strength breakdown unavailable."
        }</p>
      </article>
      <article class="profile-card profile-grid__span-4">
        <h3>External Tactical Signals</h3>
        <p class="detail-summary">${escapeHtml(providerContextFallbackSummary("tactical", mergedRow, tacticalContext))}</p>
        <ul class="risk-list">${buildSignalListMarkup(tacticalContext?.signals, "No external tactical provider signals.")}</ul>
      </article>
      <article class="profile-card profile-grid__span-4">
        <h3>Availability Context</h3>
        <p class="detail-summary">${escapeHtml(providerContextFallbackSummary("availability", mergedRow, availabilityContext))}</p>
        <ul class="risk-list">${buildSignalListMarkup(availabilityContext?.signals, "No provider availability signals.")}</ul>
      </article>
      <article class="profile-card profile-grid__span-8">
        <h3>Similar Players</h3>
        <ul class="risk-list">${
          similar?.available === true && Array.isArray(similar?.items) && similar.items.length
            ? similar.items
                .map(
                  (item) => `
                    <li class="risk-item">
                      <div class="risk-head">
                        <span class="risk-code">${escapeHtml(safeText(item?.name || item?.player_id))}</span>
                        <span class="risk-severity risk-severity--medium">${Number.isFinite(Number(item?.score)) ? formatNumber(item.score) : "-"}</span>
                      </div>
                      <p class="risk-message">${escapeHtml(
                        [item?.club, item?.league, item?.position].filter(Boolean).map((value) => safeText(value)).join(" | ") ||
                          safeText(item?.player_id)
                      )}</p>
                      <p class="formation-summary">${escapeHtml(safeText(item?.justification || ""))}</p>
                    </li>
                  `
                )
                .join("")
            : DETAIL_EMPTY_LIST_ITEM
        }</ul>
      </article>
      <article class="profile-card profile-grid__span-4">
        <h3>Schedule + Market Context</h3>
        <p class="detail-summary">${escapeHtml(providerContextFallbackSummary("market", mergedRow, marketContext))}</p>
        <ul class="risk-list">${buildSignalListMarkup(marketContext?.signals, "No schedule or market context signals.")}</ul>
      </article>
      <article class="profile-card profile-grid__span-12">
        <h3>Why Ranked Here</h3>
        <ul class="risk-list">${buildNarrativeListMarkup(whyRanked, "Ranking explanation unavailable.")}</ul>
      </article>
      <article class="profile-card profile-grid__span-8">
        <h3>Risk Flags</h3>
        <ul class="risk-list">${
          risks.length
            ? risks
                .map(
                  (flag) => `
                    <li class="risk-item">
                      <div class="risk-head">
                        <span class="risk-severity risk-severity--${escapeHtml(safeText(flag?.severity || "low").toLowerCase())}">${escapeHtml(
                      safeText(flag?.severity || "low")
                    )}</span>
                        <span class="risk-code">${escapeHtml(safeText(flag?.code))}</span>
                      </div>
                      <p class="risk-message">${escapeHtml(safeText(flag?.message))}</p>
                    </li>
                  `
                )
                .join("")
            : DETAIL_EMPTY_LIST_ITEM
        }</ul>
      </article>

      <article class="profile-card profile-grid__span-12">
        <h3>All Available Stats</h3>
        ${buildStatGroupsHtml(statGroups, { openCount: 3 })}
      </article>
    </section>
  `;
}

function openProfileModal() {
  if (!state.selectedRow) return;
  state.profileModalOpen = true;
  renderProfileModal(state.selectedRow, {
    profile: state.selectedProfile,
  });
  if (el.profileModal) el.profileModal.hidden = false;
  document.body.classList.add("modal-open");
}

function clearDetail() {
  state.detailRequestId += 1;
  state.selectedRow = null;
  state.selectedProfile = null;
  state.selectedReport = null;
  state.selectedHistory = null;
  el.detailContent.hidden = true;
  el.detailPlaceholder.hidden = false;
  el.detailDecisionPill.className = "decision-pill decision-pill--neutral";
  el.detailDecisionPill.textContent = "Waiting";
  el.detailDecisionNext.textContent = "Select a player to see the recommended next move.";
  el.detailDecisionReason.textContent = "Select a player to evaluate price upside and actionability.";
  el.detailGap.textContent = "-";
  el.detailGapNote.textContent = "Guardrailed upside";
  el.detailConfidenceScore.textContent = "-";
  el.detailConfidenceNote.textContent = "Model confidence posture";
  el.detailPriceStance.textContent = "-";
  el.detailPriceContext.textContent = "Market vs expected value";
  el.detailName.textContent = "";
  el.detailMeta.textContent = "";
  el.detailMarket.textContent = "-";
  el.detailExpected.textContent = "-";
  el.detailLower.textContent = "-";
  el.barMarket.style.width = "0";
  el.barExpected.style.width = "0";
  el.barLower.style.width = "0";
  el.detailSummary.textContent = "Select a player to load scouting summary.";
  el.detailRoleSummary.textContent = "Select a player to inspect the position-specific metric lens.";
  el.detailRoleMetrics.innerHTML =
    '<li class="risk-item risk-item--none">Select a player to load position-specific evidence.</li>';
  el.detailBadges.innerHTML = buildBadgeChipMarkup("No player selected", "neutral");
  el.detailScoreDriver.textContent = "Select a player to inspect ranking context.";
  el.detailWhyRanked.innerHTML =
    '<li class="risk-item risk-item--none">Select a player to see why the current ranking placed him here.</li>';
  el.detailCoverageSummary.textContent = "Select a player to inspect league and data coverage warnings.";
  el.detailCoverageList.innerHTML = '<li class="risk-item risk-item--none">No player selected.</li>';
  el.detailHistory.textContent = "Select a player to load history-strength breakdown.";
  el.detailStrengths.innerHTML = '<li class="risk-item risk-item--none">Select a player to load strengths.</li>';
  el.detailWeaknesses.innerHTML = '<li class="risk-item risk-item--none">Select a player to load weaknesses.</li>';
  el.detailLevers.innerHTML = '<li class="risk-item risk-item--none">Select a player to load development levers.</li>';
  el.detailStatsSummary.textContent = "Select a player to inspect grouped stat coverage.";
  el.detailStatGroups.innerHTML = '<p class="details-placeholder">No stats loaded yet.</p>';
  el.detailRisks.innerHTML = '<li class="risk-item risk-item--none">Select a player to load risk flags.</li>';
  el.detailConfidence.textContent = "Select a player to inspect model guardrails.";
  el.detailFormationSummary.textContent = "Select a player to estimate formation fit.";
  el.detailProviderTacticalSummary.textContent = "Select a player to load external tactical provider context.";
  el.detailProviderTactical.innerHTML = '<li class="risk-item risk-item--none">No external tactical provider signals loaded.</li>';
  el.detailSimilarSummary.textContent = "Select a player to load similar-player matches.";
  el.detailSimilar.innerHTML = '<li class="risk-item risk-item--none">No similar players loaded.</li>';
  el.detailAvailabilitySummary.textContent = "Select a player to load provider availability context.";
  el.detailAvailabilityList.innerHTML = '<li class="risk-item risk-item--none">No provider availability signals loaded.</li>';
  el.detailMarketContextSummary.textContent = "Select a player to load schedule and market context.";
  el.detailMarketContextList.innerHTML = '<li class="risk-item risk-item--none">No schedule or market context loaded.</li>';
  renderArchetypeProfile(null);
  renderFormationFitProfile(null);
  renderSimilarPlayers(null);
  renderRadarProfile(null);
  el.detailExportJson.disabled = true;
  el.detailExportCsv.disabled = true;
  setDetailTab("overview");
  renderProfileModal(null);
  closeProfileModal();
  renderRows();
}

function renderDetail(row, { profile = null, reportLoading = false, reportError = "" } = {}) {
  if (!row) return clearDetail();
  const reportPlayer = profile?.player && typeof profile.player === "object" ? profile.player : {};
  const historyPayload = profile?.history_strength && typeof profile.history_strength === "object" ? profile.history_strength : null;
  const tacticalContext = profile?.external_tactical_context && typeof profile.external_tactical_context === "object" ? profile.external_tactical_context : null;
  const availabilityContext = profile?.availability_context && typeof profile.availability_context === "object" ? profile.availability_context : null;
  const marketContext = profile?.market_context && typeof profile.market_context === "object" ? profile.market_context : null;
  const mergedRow = { ...row, ...reportPlayer };
  const statGroups = Array.isArray(profile?.stat_groups) ? profile.stat_groups : buildStatGroups(mergedRow, profile, null);
  const gaps = deriveGapValues(mergedRow, profile);
  const rankingDiagnostics =
    Number.isFinite(Number(row?._funnelScore)) && state.funnelDiagnostics ? state.funnelDiagnostics : state.queryDiagnostics;
  const scoreContext = resolveRowScoreContext(
    mergedRow,
    rankingDiagnostics,
    Number.isFinite(Number(row?._funnelScore)) ? "funnel" : state.mode === "shortlist" ? "shortlist" : "predictions"
  );
  const decision = summarizeRecruitmentDecision(mergedRow, profile, {
    source: Number.isFinite(Number(row?._funnelScore)) ? "funnel" : state.mode === "predictions" ? "predictions" : "workbench",
  });
  const roleLens = buildRoleLens(mergedRow, profile);
  const leagueStatus = summarizeLeagueStatus(mergedRow.league);
  const coverageWarnings = detailCoverageWarnings(mergedRow, profile);
  const provenanceBadges = buildProvenanceBadges(mergedRow)
    .concat([{ label: leagueStatus.label, tone: leagueStatus.tone }])
    .map((badge) => buildBadgeChipMarkup(badge.label, badge.tone))
    .join("");
  const whyRanked = whyRankedItems(mergedRow, profile, scoreContext);
  state.selectedRow = mergedRow;
  state.selectedProfile = profile || null;
  state.selectedReport = profile || null;
  state.selectedHistory = profile?.history_strength ? { history_strength: profile.history_strength } : null;
  el.detailPlaceholder.hidden = true;
  el.detailContent.hidden = false;
  el.detailExportJson.disabled = false;
  el.detailExportCsv.disabled = false;
  renderRows();

  const market = firstFiniteNumber(profile?.valuation_guardrails?.market_value_eur, mergedRow.market_value_eur, 0);
  const expected = firstFiniteNumber(profile?.valuation_guardrails?.fair_value_eur, mergedRow.expected_value_eur, 0);
  const lower = firstFiniteNumber(mergedRow.expected_value_low_eur, 0);
  const scaleMax = Math.max(market, expected, lower, 1);

  el.detailName.textContent = safeText(mergedRow.name);
  el.detailMeta.textContent = `${safeText(mergedRow.club)} | ${safeText(mergedRow.league)} | ${safeText(
    getPosition(mergedRow)
  )} | age ${formatNumber(mergedRow.age)} | ${formatInt(getMinutes(mergedRow))} minutes`;
  el.detailDecisionPill.className = `decision-pill decision-pill--${decision.tone}`;
  el.detailDecisionPill.textContent = decision.label;
  el.detailDecisionNext.textContent = decision.nextAction;
  el.detailDecisionReason.textContent = decision.reason;
  el.detailGap.textContent = Number.isFinite(decision.gap) ? formatCurrency(decision.gap) : "-";
  el.detailGapNote.textContent = decision.gapNote;
  el.detailConfidenceScore.textContent = Number.isFinite(decision.confidenceScore)
    ? `${decision.confidenceLabel} | ${formatNumber(decision.confidenceScore)}`
    : decision.confidenceLabel;
  el.detailConfidenceNote.textContent = decision.confidenceNote;
  el.detailPriceStance.textContent = decision.priceStance;
  el.detailPriceContext.textContent = decision.priceContext;
  el.detailBadges.innerHTML = provenanceBadges;
  el.detailScoreDriver.textContent = `${scoreContext.scoreLabel}${
    Number.isFinite(scoreContext.scoreValue)
      ? ` | ${scoreContext.scoreColumn.endsWith("_eur") ? formatCurrency(scoreContext.scoreValue) : formatNumber(scoreContext.scoreValue)}`
      : ""
  } | ${scoreContext.rankingBasisLabel}.`;
  el.detailWhyRanked.innerHTML = buildNarrativeListMarkup(whyRanked, "Ranking explanation unavailable.");
  const actionableCoverageWarnings = coverageWarnings.filter((item) => {
    const severity = String(item?.severity || "").toLowerCase();
    return severity === "medium" || severity === "high";
  });
  el.detailCoverageSummary.textContent = actionableCoverageWarnings.length
    ? `${actionableCoverageWarnings.length} higher-risk coverage warnings flagged for this player and league context.`
    : "Coverage diagnostics are mostly informational for this player and league.";
  el.detailCoverageList.innerHTML = buildNarrativeListMarkup(
    coverageWarnings,
    "Coverage diagnostics look acceptable for shortlist review."
  );
  el.detailMarket.textContent = formatCurrency(market);
  el.detailExpected.textContent = formatCurrency(expected);
  el.detailLower.textContent = formatCurrency(lower);

  el.barMarket.style.width = `${Math.max((market / scaleMax) * 100, 1)}%`;
  el.barExpected.style.width = `${Math.max((expected / scaleMax) * 100, 1)}%`;
  el.barLower.style.width = `${Math.max((lower / scaleMax) * 100, 1)}%`;

  const rows = [
    ["League", safeText(mergedRow.league)],
    ["Price stance", decision.priceStance],
    ["Age", formatNumber(mergedRow.age)],
    ["Minutes", formatInt(getMinutes(mergedRow))],
    ["Role Lens", safeText(roleLens.label)],
    ["Conservative Gap (capped)", formatCurrency(decision.gap)],
    ["Cap Threshold", formatCurrency(gaps.capThreshold)],
    ["Confidence", Number.isFinite(decision.confidenceScore) ? `${decision.confidenceLabel} | ${formatNumber(decision.confidenceScore)}` : decision.confidenceLabel],
    ["Segment", safeText(mergedRow.value_segment)],
    ["Position Model", safeText(mergedRow.model_position)],
    ["Pred Low", formatCurrency(mergedRow.expected_value_low_eur)],
  ];

  el.detailList.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  el.detailStatsSummary.textContent = `${formatInt(
    statGroups.reduce((sum, group) => sum + group.items.length, 0)
  )} grouped metrics across ${formatInt(statGroups.length)} stat sections.`;
  el.detailStatGroups.innerHTML = buildStatGroupsHtml(statGroups, { openCount: 2 });
  el.detailConfidence.textContent = buildConfidenceSummary(mergedRow, profile);

  if (reportLoading) {
    el.detailSummary.textContent = "Loading scouting memo...";
    el.detailRoleSummary.textContent = "Loading position-specific metric lens...";
    el.detailRoleMetrics.innerHTML = '<li class="risk-item risk-item--none">Loading role evidence...</li>';
    el.detailHistory.textContent = "Loading history-strength breakdown...";
    el.detailStrengths.innerHTML = '<li class="risk-item risk-item--none">Loading strengths...</li>';
    el.detailWeaknesses.innerHTML = '<li class="risk-item risk-item--none">Loading weaknesses...</li>';
    el.detailLevers.innerHTML = '<li class="risk-item risk-item--none">Loading development levers...</li>';
    el.detailRisks.innerHTML = '<li class="risk-item risk-item--none">Loading risk flags...</li>';
    el.detailProviderTacticalSummary.textContent = "Loading external tactical provider context...";
    el.detailProviderTactical.innerHTML = '<li class="risk-item risk-item--none">Loading provider signals...</li>';
    el.detailAvailabilitySummary.textContent = "Loading provider availability context...";
    el.detailAvailabilityList.innerHTML = '<li class="risk-item risk-item--none">Loading provider signals...</li>';
    el.detailMarketContextSummary.textContent = "Loading schedule and market context...";
    el.detailMarketContextList.innerHTML = '<li class="risk-item risk-item--none">Loading provider signals...</li>';
    el.detailConfidence.textContent = "Loading confidence and guardrails...";
    renderArchetypeProfile(null, { loading: true });
    renderFormationFitProfile(null, { loading: true });
    renderSimilarPlayers(null, { loading: true });
    renderRadarProfile(null, { loading: true });
    if (state.profileModalOpen) {
      renderProfileModal(mergedRow, { profile, reportError });
    }
    return;
  }
  if (reportError) {
    el.detailSummary.textContent = `Recruitment memo unavailable: ${reportError}`;
    el.detailRoleSummary.textContent = roleLens.summary;
    el.detailRoleMetrics.innerHTML = buildNarrativeListMarkup(
      roleLens.metrics.map((metric) => ({
        label: metric.label,
        message: `Current active value: ${metric.displayValue}.`,
        tone: "medium",
      })),
      "Position-specific evidence unavailable for this player."
    );
    el.detailHistory.textContent = "History-strength breakdown unavailable for this player.";
    el.detailStrengths.innerHTML = '<li class="risk-item risk-item--none">Strengths unavailable for this player.</li>';
    el.detailWeaknesses.innerHTML = '<li class="risk-item risk-item--none">Weaknesses unavailable for this player.</li>';
    el.detailLevers.innerHTML = '<li class="risk-item risk-item--none">Development levers unavailable for this player.</li>';
    el.detailRisks.innerHTML = '<li class="risk-item risk-item--none">Risk flags unavailable for this player.</li>';
    el.detailProviderTacticalSummary.textContent = "External tactical provider context unavailable for this player.";
    el.detailProviderTactical.innerHTML = '<li class="risk-item risk-item--none">Provider signals unavailable for this player.</li>';
    el.detailAvailabilitySummary.textContent = "Provider availability context unavailable for this player.";
    el.detailAvailabilityList.innerHTML = '<li class="risk-item risk-item--none">Provider signals unavailable for this player.</li>';
    el.detailMarketContextSummary.textContent = "Schedule and market context unavailable for this player.";
    el.detailMarketContextList.innerHTML = '<li class="risk-item risk-item--none">Provider signals unavailable for this player.</li>';
    el.detailConfidence.textContent = `Confidence and guardrails unavailable: ${reportError}`;
    renderArchetypeProfile(null, { error: reportError });
    renderFormationFitProfile(null, { error: reportError });
    renderSimilarPlayers(null, { error: reportError });
    renderRadarProfile(null, { error: reportError });
    if (state.profileModalOpen) {
      renderProfileModal(mergedRow, { profile, reportError });
    }
    return;
  }

  if (profile?.summary_text) {
    el.detailSummary.textContent = profile.summary_text;
  } else {
    el.detailSummary.textContent =
      "No memo summary loaded. Use the export buttons to fetch a complete player report on demand.";
  }
  el.detailRoleSummary.textContent = roleLens.summary;
  el.detailRoleMetrics.innerHTML = buildNarrativeListMarkup(
    roleLens.metrics.map((metric) => ({
      label: metric.label,
      message: `Current active value: ${metric.displayValue}.`,
      tone: "medium",
    })),
    "No role-specific metrics were available in the active artifact for this player."
  );

  const strengths = Array.isArray(profile?.strengths) ? profile.strengths : [];
  renderNarrativeMetricList(el.detailStrengths, strengths, "No clear metric strengths for this cohort.");

  const weaknesses = Array.isArray(profile?.weaknesses) ? profile.weaknesses : [];
  renderNarrativeMetricList(el.detailWeaknesses, weaknesses, "No clear metric weaknesses for this cohort.");

  const levers = Array.isArray(profile?.development_levers) ? profile.development_levers : [];
  renderNarrativeMetricList(
    el.detailLevers,
    levers,
    "No high-impact development lever detected.",
    "impact_score",
    "impact"
  );

  if (historyPayload?.summary_text) {
    const score = Number(historyPayload?.score_0_to_100);
    const cov = Number(historyPayload?.coverage_0_to_1);
    const scoreText = Number.isFinite(score) ? `${formatNumber(score)}/100` : "-";
    const covText = Number.isFinite(cov) ? formatPct(cov) : "-";
    el.detailHistory.textContent = `${historyPayload.summary_text} Coverage: ${covText}. Score: ${scoreText}.`;
  } else {
    el.detailHistory.textContent = "History-strength breakdown unavailable for this player.";
  }

  renderArchetypeProfile(profile?.player_type);
  renderFormationFitProfile(profile?.formation_fit);
  renderSimilarPlayers(profile?.similar_players);
  renderRadarProfile(profile?.radar_profile);
  el.detailProviderTacticalSummary.textContent =
    providerContextFallbackSummary("tactical", mergedRow, tacticalContext);
  renderSignalList(el.detailProviderTactical, tacticalContext, "No external tactical provider signals loaded.");
  el.detailAvailabilitySummary.textContent =
    providerContextFallbackSummary("availability", mergedRow, availabilityContext);
  renderSignalList(el.detailAvailabilityList, availabilityContext, "No provider availability signals loaded.");
  el.detailMarketContextSummary.textContent =
    providerContextFallbackSummary("market", mergedRow, marketContext);
  renderSignalList(el.detailMarketContextList, marketContext, "No schedule or market context loaded.");

  const riskFlags = Array.isArray(profile?.risk_flags) ? profile.risk_flags : [];
  if (!riskFlags.length) {
    el.detailRisks.innerHTML = '<li class="risk-item risk-item--none">No risk flags triggered.</li>';
  } else {
    el.detailRisks.innerHTML = riskFlags
      .map((flag) => {
        const severity = String(flag.severity || "low").toLowerCase();
        return `
          <li class="risk-item">
            <div class="risk-head">
              <span class="risk-severity risk-severity--${safeText(severity)}">${safeText(severity)}</span>
              <span class="risk-code">${safeText(flag.code)}</span>
            </div>
            <p class="risk-message">${safeText(flag.message)}</p>
          </li>
        `;
      })
      .join("");
  }
  if (state.profileModalOpen) {
    renderProfileModal(mergedRow, { profile, reportError });
  }
}

function renderModeAffordances() {
  const shortlist = state.mode === "shortlist";
  el.title.textContent = shortlist ? "Recruitment Board" : "Valuation Board";
  el.sort.disabled = shortlist;
  el.sortDir.disabled = shortlist;
  el.minConfidence.disabled = shortlist;
  el.minGap.disabled = shortlist;
  el.topN.disabled = !shortlist;
  renderResultsNote();
}

function renderResultsNote() {
  if (!el.resultsNote) return;
  const filterSummary = describeRecruitmentFilters(currentWorkbenchWorkflowSummary());
  if (state.mode === "predictions") {
    el.resultsNote.textContent = `This is a valuation view, not a live pursuit order. Use it for price discipline under the active brief (${filterSummary}), then switch back to Recruitment Board or Target Funnel when you want decision-ready ranking.`;
    return;
  }
  const diagnostics = state.queryDiagnostics || {};
  const scoreColumn = diagnostics.score_column || diagnostics.scoreColumn || "shortlist_score";
  const rankingBasis = diagnostics.ranking_basis || diagnostics.rankingBasis || "guardrailed_gap_confidence_history";
  const precisionRows = diagnostics?.precision_at_k?.rows || [];
  const p25 = precisionRows.find((row) => Number(row.k) === 25);
  const precisionText =
    p25 && Number.isFinite(Number(p25.precision)) ? ` Precision@25 ${formatPct(Number(p25.precision))}.` : "";
  el.resultsNote.textContent = `Recruitment brief: ${filterSummary}. Scan pursue and watch calls first, then open one memo to decide the next action. Ranking driver: ${humanizeScoreColumn(
    scoreColumn
  )} | ${rankingBasisLabel(rankingBasis)}.${precisionText}`;
}

function updateSelectOptions(select, values, keepValue = "") {
  const allOption = select.querySelector("option[value='']");
  const allLabel = allOption ? allOption.textContent : "All";
  select.innerHTML = `<option value="">${allLabel}</option>`;
  values.forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    select.appendChild(opt);
  });
  if (keepValue && values.includes(keepValue)) {
    select.value = keepValue;
  }
}

function backendReady() {
  return Boolean(state.connected && state.health?.status === "ok");
}

async function refreshCoverageAndOptions() {
  if (!backendReady()) {
    state.coverageRows = [];
    renderCoverageTable();
    renderOverviewReadiness();
    return;
  }
  const rows = await fetchAllPredictions({
    split: state.split,
    columns:
      "season,league,undervalued_flag,undervaluation_confidence,value_gap_conservative_eur",
  });
  state.coverageRows = rows;
  renderCoverageTable();
  renderOverviewReadiness();

  const seasons = Array.from(new Set(rows.map((r) => String(r.season || "")).filter(Boolean))).sort(
    (a, b) => seasonSortValue(b) - seasonSortValue(a)
  );
  const leagues = Array.from(new Set(rows.map((r) => getLeague(r)).filter(Boolean))).sort((a, b) =>
    a.localeCompare(b)
  );

  const prevSeason = state.season;
  const prevLeague = state.league;
  updateSelectOptions(el.season, seasons, prevSeason);
  updateSelectOptions(el.league, leagues, prevLeague);

  if (prevSeason && !seasons.includes(prevSeason)) state.season = "";
  if (prevLeague && !leagues.includes(prevLeague)) state.league = "";
}

async function fetchPredictionsPage() {
  const maxBudget = parseOptionalPositive(state.budgetBand);
  const payload = await getJson("/market-value/predictions", {
    split: state.split,
    season: state.season || null,
    league: state.league || null,
    position: state.position || null,
    role_keys: state.roleNeed || null,
    min_minutes: state.minMinutes,
    min_age: state.minAge < 0 ? null : state.minAge,
    max_age: state.maxAge < 0 ? null : state.maxAge,
    max_market_value_eur: maxBudget,
    max_contract_years_left: state.maxContractYearsLeft,
    non_big5_only: state.nonBig5Only,
    undervalued_only: state.undervaluedOnly,
    min_confidence: state.minConfidence > 0 ? state.minConfidence : null,
    min_value_gap_eur: state.minGapEur > 0 ? state.minGapEur : null,
    sort_by: state.sortBy,
    sort_order: state.sortOrder,
    limit: state.limit,
    offset: state.offset,
  });

  let rows = (payload.items || []).map((r) => withComputedConservativeGap(r));
  if (state.search) {
    const q = state.search.toLowerCase();
    rows = rows.filter((r) => {
      const name = String(r.name || "").toLowerCase();
      const club = String(r.club || "").toLowerCase();
      return name.includes(q) || club.includes(q);
    });
  }

  state.rows = rows;
  state.total = Number(payload.total) || rows.length;
  state.count = rows.length;
  state.queryDiagnostics = {
    score_column: state.sortBy,
    ranking_basis: "manual_sort",
    sort_order: state.sortOrder,
  };
  renderResultsNote();
}

function sortShortlistRows(rows) {
  const out = [...rows];
  out.sort((a, b) => {
    const gapDiff = conservativeGapForRanking(b) - conservativeGapForRanking(a);
    if (Number.isFinite(gapDiff) && gapDiff !== 0) return gapDiff;
    const confDiff = (Number(b.undervaluation_confidence) || 0) - (Number(a.undervaluation_confidence) || 0);
    if (confDiff !== 0) return confDiff;
    const scoreDiff = (Number(b.undervaluation_score) || 0) - (Number(a.undervaluation_score) || 0);
    if (scoreDiff !== 0) return scoreDiff;
    return String(a.name || "").localeCompare(String(b.name || ""));
  });
  return out;
}

async function fetchShortlistPage() {
  const maxBudget = parseOptionalPositive(state.budgetBand);
  const payload = await getJson("/market-value/shortlist", {
    split: state.split,
    top_n: state.shortlistTopN,
    min_minutes: state.minMinutes,
    min_age: state.minAge < 0 ? null : state.minAge,
    max_age: state.maxAge < 0 ? -1 : state.maxAge,
    positions: state.position || null,
    role_keys: state.roleNeed || null,
    non_big5_only: state.nonBig5Only,
    max_market_value_eur: maxBudget,
    max_contract_years_left: state.maxContractYearsLeft,
  });

  const filtered = applyClientFilters((payload.items || []).map((r) => withComputedConservativeGap(r)));
  const sorted = sortShortlistRows(filtered);
  const page = sorted.slice(state.offset, state.offset + state.limit);

  state.rows = page;
  state.total = sorted.length;
  state.count = page.length;
  state.queryDiagnostics = payload.diagnostics || null;
  renderResultsNote();
}

async function runQuery() {
  readWorkbenchControlsToState();
  localStorage.setItem("scoutml_api_base", state.apiBase);
  renderModeAffordances();
  if (!backendReady()) {
    state.rows = [];
    state.total = 0;
    state.count = 0;
    state.queryDiagnostics = null;
    el.tbody.innerHTML =
      '<tr><td colspan="10">Backend artifacts are not ready. Review Overview for readiness details.</td></tr>';
    renderResultsNote();
    renderPager();
    clearDetail();
    return;
  }
  setLoading(true);
  try {
    if (state.mode === "shortlist") {
      await fetchShortlistPage();
    } else {
      await fetchPredictionsPage();
    }
    renderRows();
    renderPager();
    clearDetail();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    el.tbody.innerHTML = `<tr><td colspan=\"10\">${msg}</td></tr>`;
    state.rows = [];
    state.total = 0;
    state.count = 0;
    state.queryDiagnostics = null;
    renderResultsNote();
    renderPager();
  } finally {
    setLoading(false);
  }
}

function downloadCsv(rows, filename) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const lines = [cols.join(",")];
  rows.forEach((row) => {
    const values = cols.map((c) => `"${String(row[c] ?? "").replace(/"/g, '""')}"`);
    lines.push(values.join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function reportCacheKey(row) {
  const playerId = String(row?.player_id || "").trim();
  const season = String(row?.season || state.season || "");
  return `${state.apiBase}|${state.split}|${season}|${playerId}`;
}

async function fetchPlayerProfileForRow(row) {
  const playerId = String(row?.player_id || "").trim();
  if (!playerId) {
    throw new Error("Selected row has no player_id.");
  }
  const season = String(row?.season || state.season || "").trim();
  const key = reportCacheKey(row);
  if (state.profileCache.has(key)) {
    return state.profileCache.get(key);
  }
  const payload = await getJson(`/market-value/player/${encodeURIComponent(playerId)}/profile`, {
    split: state.split,
    season: season || null,
    top_metrics: 6,
    similar_top_k: 5,
  });
  const profile = payload.profile || null;
  if (profile) state.profileCache.set(key, profile);
  return profile;
}

async function fetchPlayerReportForRow(row) {
  const playerId = String(row?.player_id || "").trim();
  if (!playerId) {
    throw new Error("Selected row has no player_id.");
  }
  const season = String(row?.season || state.season || "").trim();
  const key = reportCacheKey(row);
  if (state.reportCache.has(key)) {
    return state.reportCache.get(key);
  }
  const payload = await getJson(`/market-value/player/${encodeURIComponent(playerId)}/report`, {
    split: state.split,
    season: season || null,
    top_metrics: 5,
  });
  const report = payload.report || null;
  if (report) state.reportCache.set(key, report);
  return report;
}

async function fetchPlayerHistoryForRow(row) {
  const playerId = String(row?.player_id || "").trim();
  if (!playerId) {
    throw new Error("Selected row has no player_id.");
  }
  const season = String(row?.season || state.season || "").trim();
  const payload = await getJson(`/market-value/player/${encodeURIComponent(playerId)}/history-strength`, {
    split: state.split,
    season: season || null,
  });
  return payload.breakdown || null;
}

async function loadDetailWithReport(row) {
  if (!row) return clearDetail();
  const requestId = ++state.detailRequestId;
  setDetailTab("overview");
  renderDetail(row, { reportLoading: true });
  try {
    const profile = await fetchPlayerProfileForRow(row);
    if (requestId !== state.detailRequestId) return;
    renderDetail(row, { profile });
  } catch (err) {
    if (requestId !== state.detailRequestId) return;
    const msg = err instanceof Error ? err.message : String(err);
    renderDetail(row, { reportError: msg });
  }
}

function sanitizeFileToken(value, fallback = "player") {
  const token = String(value || "").trim();
  if (!token) return fallback;
  return token.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function buildPlayerMemoPayload(row, report = null) {
  const historyBreakdown =
    state.selectedProfile?.history_strength && typeof state.selectedProfile.history_strength === "object"
      ? state.selectedProfile.history_strength
      : state.selectedHistory?.history_strength && typeof state.selectedHistory.history_strength === "object"
      ? state.selectedHistory.history_strength
      : null;
  const reportPlayer = report?.player && typeof report.player === "object" ? report.player : {};
  const merged = { ...row, ...reportPlayer };
  const gaps = deriveGapValues(merged, report);
  const market = firstFiniteNumber(report?.valuation_guardrails?.market_value_eur, merged.market_value_eur);
  const fair = firstFiniteNumber(report?.valuation_guardrails?.fair_value_eur, merged.fair_value_eur, merged.expected_value_eur);
  return {
    generated_at_utc: new Date().toISOString(),
    split: state.split,
    player: {
      player_id: merged.player_id,
      name: merged.name,
      club: merged.club,
      league: merged.league,
      season: merged.season,
      position: getPosition(merged),
      age: merged.age,
      market_value_eur: market,
      expected_value_eur: fair,
      expected_value_low_eur: firstFiniteNumber(merged.expected_value_low_eur),
      expected_value_high_eur: firstFiniteNumber(merged.expected_value_high_eur),
      undervaluation_confidence: firstFiniteNumber(merged.undervaluation_confidence),
    },
    valuation: {
      market_value_eur: market,
      fair_value_eur: fair,
      value_gap_raw_eur: gaps.raw,
      value_gap_conservative_eur: gaps.conservative,
      value_gap_capped_eur: gaps.capped,
      cap_threshold_eur: gaps.capThreshold,
      cap_applied: gaps.capApplied,
    },
    summary_text: report?.summary_text || "",
    risk_flags: Array.isArray(report?.risk_flags) ? report.risk_flags : [],
    strengths: Array.isArray(report?.strengths) ? report.strengths : [],
    weaknesses: Array.isArray(report?.weaknesses) ? report.weaknesses : [],
    development_levers: Array.isArray(report?.development_levers) ? report.development_levers : [],
    confidence: report?.confidence || null,
    valuation_guardrails: report?.valuation_guardrails || null,
    cohort: report?.cohort || null,
    history_strength: historyBreakdown,
    player_type: report?.player_type || null,
    formation_fit: report?.formation_fit || null,
    radar_profile: report?.radar_profile || null,
    stat_groups: Array.isArray(report?.stat_groups) ? report.stat_groups : [],
    similar_players: report?.similar_players || null,
  };
}

function buildPlayerMemoCsvRow(row, report = null) {
  const memo = buildPlayerMemoPayload(row, report);
  const topStrengths = memo.strengths.slice(0, 3).map((m) => m.label).join("|");
  const topLevers = memo.development_levers.slice(0, 3).map((m) => m.label).join("|");
  const riskCodes = memo.risk_flags.map((r) => r.code).join("|");
  const bestFormation =
    Array.isArray(memo.formation_fit?.recommended) && memo.formation_fit.recommended.length
      ? memo.formation_fit.recommended[0]
      : null;
  return {
    generated_at_utc: memo.generated_at_utc,
    split: memo.split,
    player_id: memo.player.player_id,
    name: memo.player.name,
    club: memo.player.club,
    league: memo.player.league,
    season: memo.player.season,
    position: memo.player.position,
    age: memo.player.age,
    market_value_eur: memo.valuation.market_value_eur,
    fair_value_eur: memo.valuation.fair_value_eur,
    value_gap_raw_eur: memo.valuation.value_gap_raw_eur,
    value_gap_conservative_eur: memo.valuation.value_gap_conservative_eur,
    value_gap_capped_eur: memo.valuation.value_gap_capped_eur,
    cap_threshold_eur: memo.valuation.cap_threshold_eur,
    cap_applied: memo.valuation.cap_applied,
    confidence: memo.player.undervaluation_confidence,
    player_archetype: memo.player_type?.archetype || null,
    player_archetype_confidence: memo.player_type?.confidence_0_to_1 ?? null,
    player_archetype_tier: memo.player_type?.tier || null,
    best_formation: bestFormation?.formation || null,
    best_role: bestFormation?.role || null,
    best_formation_fit_score: bestFormation?.fit_score_0_to_1 ?? null,
    radar_coverage_0_to_1: memo.radar_profile?.coverage_0_to_1 ?? null,
    risk_codes: riskCodes,
    top_strengths: topStrengths,
    development_levers: topLevers,
    summary_text: memo.summary_text,
  };
}

async function ensureSelectedReport() {
  if (!state.selectedRow) return null;
  if (state.selectedProfile) return state.selectedProfile;
  try {
    const profile = await fetchPlayerProfileForRow(state.selectedRow);
    state.selectedProfile = profile || null;
    state.selectedReport = profile || null;
    state.selectedHistory = profile?.history_strength ? { history_strength: profile.history_strength } : null;
    return state.selectedProfile;
  } catch {
    return null;
  }
}

function computeFunnelScore(row) {
  const gap = Math.max(conservativeGapForRanking(row) || 0, 0);
  const conf = Math.max(Number(row.undervaluation_confidence) || 0, 0);
  const minutes = Math.max(getMinutes(row) || 0, 0);
  const age = Number(row.age);
  const market = Math.max(Number(row.market_value_eur) || 0, 1_000_000);

  const minutesFactor = Math.min(Math.max(minutes / 1800, 0.45), 1.35);
  const ageFactor = Number.isFinite(age)
    ? age <= 20
      ? 1.24
      : age <= 23
      ? 1.12
      : age <= 25
      ? 1.0
      : 0.82
    : 1.0;
  const valueEfficiency = gap / market;

  return (gap / 1_000_000) * (1 + conf) * minutesFactor * ageFactor * (1 + 0.3 * valueEfficiency);
}

function renderFunnelTables() {
  if (!state.funnelTopRows.length) {
    el.funnelBody.innerHTML = "<tr><td colspan=\"8\">No candidates for current funnel filters.</td></tr>";
    el.funnelSummaryTitle.textContent = "Run the funnel to see where the strongest investable upside is concentrated.";
    el.funnelSummaryLeague.textContent = "-";
    el.funnelSummaryCount.textContent = "No league priority yet.";
    el.funnelSummaryGap.textContent = "-";
    el.funnelSummaryConfidence.textContent = "Confidence signal unavailable.";
    el.funnelSummaryStatus.textContent = "-";
    el.funnelSummaryCopy.textContent = "Run the funnel to decide where scouting time should go next.";
  } else {
    el.funnelBody.innerHTML = state.funnelTopRows
      .map((row, idx) => {
        const score = Number(row._funnelScore);
        const decision = summarizeRecruitmentDecision(row, null, { source: "funnel" });
        const scoreContext = resolveRowScoreContext(row, state.funnelDiagnostics, "funnel");
        const badges = buildProvenanceBadges(row)
          .map((badge) => buildBadgeChipMarkup(badge.label, badge.tone))
          .join("");
        return `
          <tr data-index="${idx}" class="row-clickable">
            <td class="player-cell">
              <strong>${safeText(row.name)}</strong>
              <span class="player-cell__sub">${safeText(row.league)} | ${safeText(row.club)} | ${safeText(row.season)}</span>
              <div class="player-cell__badges">${badges}</div>
              <span class="player-cell__note">${escapeHtml(scoreContext.scoreLabel)}</span>
            </td>
            <td>
              <div class="table-status">
                ${buildDecisionPillMarkup(decision)}
                <span class="table-status__note">${escapeHtml(decision.gapNote)}</span>
              </div>
            </td>
            <td class="num">${formatNumber(row.age)}</td>
            <td class="num">${formatCurrency(row.market_value_eur)}</td>
            <td class="num positive">${formatCurrency(conservativeGapForRanking(row))}</td>
            <td class="num">${formatNumber(row.undervaluation_confidence)}</td>
            <td class="num">${Number.isFinite(score) ? formatNumber(score) : "-"}</td>
            <td><span class="action-copy">${escapeHtml(decision.nextAction)}</span></td>
          </tr>
        `;
      })
      .join("");
  }

  if (!state.funnelRows.length) {
    el.funnelLeagueBody.innerHTML = "<tr><td colspan=\"6\">No league board yet.</td></tr>";
    return;
  }

  const grouped = new Map();
  state.funnelRows.forEach((row) => {
    const league = safeText(row.league);
    if (!grouped.has(league)) {
      grouped.set(league, { league, n: 0, gapSum: 0, confSum: 0, ageSum: 0, confN: 0, ageN: 0 });
    }
    const g = grouped.get(league);
    g.n += 1;

    const gap = conservativeGapForRanking(row);
    if (Number.isFinite(gap)) g.gapSum += gap;

    const conf = Number(row.undervaluation_confidence);
    if (Number.isFinite(conf)) {
      g.confSum += conf;
      g.confN += 1;
    }

    const age = Number(row.age);
    if (Number.isFinite(age)) {
      g.ageSum += age;
      g.ageN += 1;
    }
  });

  const rows = Array.from(grouped.values()).sort((a, b) => b.n - a.n);
  el.funnelLeagueBody.innerHTML = rows
    .map((g) => {
      const avgGap = g.n > 0 ? g.gapSum / g.n : NaN;
      const avgConf = g.confN > 0 ? g.confSum / g.confN : NaN;
      const avgAge = g.ageN > 0 ? g.ageSum / g.ageN : NaN;
      const status = summarizeLeagueStatus(g.league);
      return `
        <tr>
          <td>${safeText(g.league)}</td>
          <td class="num">${formatInt(g.n)}</td>
          <td class="num">${Number.isFinite(avgGap) ? formatCurrency(avgGap) : "-"}</td>
          <td class="num">${Number.isFinite(avgConf) ? formatNumber(avgConf) : "-"}</td>
          <td class="num">${Number.isFinite(avgAge) ? formatNumber(avgAge) : "-"}</td>
          <td><span class="status status--${escapeHtml(status.tone)}">${escapeHtml(status.label)}</span></td>
        </tr>
      `;
    })
    .join("");

  if (!rows.length) {
    el.funnelSummaryTitle.textContent = "Run the funnel to see where the strongest investable upside is concentrated.";
    el.funnelSummaryLeague.textContent = "-";
    el.funnelSummaryCount.textContent = "No league priority yet.";
    el.funnelSummaryGap.textContent = "-";
    el.funnelSummaryConfidence.textContent = "Confidence signal unavailable.";
    el.funnelSummaryStatus.textContent = "-";
    el.funnelSummaryCopy.textContent = "Run the funnel to decide where scouting time should go next.";
    return;
  }

  const topLeague = rows[0];
  const topLeagueStatus = summarizeLeagueStatus(topLeague.league);
  const topGap = topLeague.n > 0 ? topLeague.gapSum / topLeague.n : NaN;
  const topConf = topLeague.confN > 0 ? topLeague.confSum / topLeague.confN : NaN;
  el.funnelSummaryTitle.textContent =
    topLeagueStatus.tone === "bad"
      ? "The largest concentration of names still sits in a league with fragile coverage."
      : `The strongest current sourcing pocket is ${safeText(topLeague.league)}.`;
  el.funnelSummaryLeague.textContent = safeText(topLeague.league);
  el.funnelSummaryCount.textContent = `${formatInt(topLeague.n)} candidates in the current funnel.`;
  el.funnelSummaryGap.textContent = Number.isFinite(topGap) ? formatCurrency(topGap) : "-";
  el.funnelSummaryConfidence.textContent = Number.isFinite(topConf)
    ? `Avg confidence ${formatNumber(topConf)}`
    : "Confidence signal unavailable.";
  el.funnelSummaryStatus.textContent = safeText(topLeagueStatus.label);
  el.funnelSummaryCopy.textContent = topLeagueStatus.note;
}

async function runFunnel() {
  if (!backendReady()) {
    state.funnelRows = [];
    state.funnelTopRows = [];
    state.funnelDiagnostics = null;
    renderFunnelTables();
    el.funnelMeta.textContent = "Target funnel unavailable until backend artifacts are ready.";
    return;
  }
  const split = el.funnelSplit.value;
  const minAge = parseNumberOr(el.funnelMinAge.value, -1);
  const maxAge = parseNumberOr(el.funnelMaxAge.value, 23);
  const minMinutes = Math.max(parseNumberOr(el.funnelMinMinutes.value, 900), 0);
  const minConfidence = Math.max(parseNumberOr(el.funnelMinConfidence.value, 0), 0);
  const minGap = Math.max(parseNumberOr(el.funnelMinGap.value, 0), 0);
  const maxBudget = parseOptionalPositive(el.funnelBudgetBand.value);
  const maxContractYearsLeft = parseOptionalPositive(el.funnelMaxContractYears.value);
  const roleNeed = el.funnelRoleNeed.value || "";
  const topN = Math.max(Math.round(parseNumberOr(el.funnelTopN.value, 50)), 1);
  const lowerOnly = el.funnelLowerOnly.checked;

  el.funnelMeta.textContent = "Running funnel...";

  try {
    const payload = await getJson("/market-value/scout-targets", {
      split,
      top_n: Math.max(topN * 4, topN),
      non_big5_only: lowerOnly,
      min_age: minAge < 0 ? null : minAge,
      max_age: maxAge < 0 ? null : maxAge,
      min_minutes: minMinutes,
      min_confidence: minConfidence > 0 ? minConfidence : null,
      min_value_gap_eur: minGap > 0 ? minGap : null,
      max_market_value_eur: maxBudget,
      max_contract_years_left: maxContractYearsLeft,
      role_keys: roleNeed || null,
    });
    const rows = (payload.items || []).map((row) => withComputedConservativeGap(row));
    state.funnelDiagnostics = payload.diagnostics || null;
    const filtered = rows.filter((row) => {
      const gap = conservativeGapForRanking(row);
      if (minGap > 0 && (!Number.isFinite(gap) || gap < minGap)) return false;
      if (!matchesRecruitmentWorkflow(row, currentFunnelWorkflowSummary())) return false;
      return true;
    });

    filtered.forEach((row) => {
      const fromApi = Number(row.scout_target_score);
      row._funnelScore = Number.isFinite(fromApi) ? fromApi : computeFunnelScore(row);
    });
    filtered.sort((a, b) => Number(b._funnelScore) - Number(a._funnelScore));
    state.funnelRows = filtered;
    state.funnelTopRows = filtered.slice(0, topN);

    const precisionRows = payload?.diagnostics?.precision_at_k?.rows || [];
    const p50 = precisionRows.find((r) => Number(r.k) === 50);
    const precisionNote =
      p50 && Number.isFinite(Number(p50.precision))
        ? ` | precision@50 ${formatPct(Number(p50.precision))}`
        : "";
    const scoreColumn =
      payload?.diagnostics?.score_column || payload?.diagnostics?.scoreColumn || "scout_target_score";
    const filterSummary = describeRecruitmentFilters(currentFunnelWorkflowSummary());
    el.funnelMeta.textContent = `${formatInt(state.funnelTopRows.length)} shown / ${formatInt(
      state.funnelRows.length
    )} total candidates | split ${split}${lowerOnly ? " | outside Big 5" : ""} | driver ${humanizeScoreColumn(
      scoreColumn
    )}${precisionNote} | filters ${filterSummary} | start with the league board, then decide which names deserve the memo`;

    renderFunnelTables();
  } catch (err) {
    state.funnelRows = [];
    state.funnelTopRows = [];
    state.funnelDiagnostics = null;
    renderFunnelTables();
    el.funnelMeta.textContent = err instanceof Error ? err.message : String(err);
  }
}

async function loadHealthAndMetrics() {
  state.reportCache = new Map();
  state.profileCache = new Map();
  state.selectedProfile = null;
  state.selectedReport = null;
  state.selectedHistory = null;
  state.health = await getJson("/market-value/health");
  state.connected = true;
  if (state.health?.status !== "ok") {
    state.metrics = null;
    state.modelManifest = null;
    state.benchmark = null;
    state.activeArtifacts = null;
    setStatus("error", "Artifacts missing");
    renderTrustCard();
    renderMetrics();
    renderSegmentTable();
    renderBenchmarkCards();
    renderOverviewReadiness();
    return false;
  }
  state.metrics = (await getJson("/market-value/metrics")).payload || null;
  try {
    state.modelManifest = (await getJson("/market-value/model-manifest")).payload || null;
  } catch {
    state.modelManifest = null;
  }
  try {
    state.benchmark = (await getJson("/market-value/benchmarks")).payload || null;
  } catch {
    state.benchmark = null;
  }
  try {
    state.activeArtifacts = (await getJson("/market-value/active-artifacts")).payload || null;
  } catch {
    state.activeArtifacts = null;
  }

  const artifacts = state.health?.artifacts || {};
  const ok = Boolean(artifacts.test_predictions_exists && artifacts.metrics_exists);
  setStatus(ok ? "ok" : "error", ok ? "Artifacts ready" : "Artifacts missing");

  renderTrustCard();
  renderMetrics();
  renderSegmentTable();
  renderBenchmarkCards();
  renderOverviewReadiness();
  return true;
}

function resetWorkbenchControls() {
  state.mode = "shortlist";
  state.split = "test";
  state.season = "";
  state.league = "";
  state.position = "";
  state.roleNeed = "";
  state.search = "";
  state.minMinutes = 900;
  state.minAge = 18;
  state.maxAge = 23;
  state.budgetBand = "10000000";
  state.maxContractYearsLeft = 2;
  state.minConfidence = 0.5;
  state.minGapEur = 1_000_000;
  state.shortlistTopN = 100;
  state.sortBy = "value_gap_conservative_eur";
  state.sortOrder = "desc";
  state.limit = 50;
  state.offset = 0;
  state.nonBig5Only = true;
  state.undervaluedOnly = true;
  syncWorkbenchControlsFromState();
  renderModeAffordances();
}

function bindEvents() {
  let searchTimer = null;

  el.tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view || "overview"));
  });
  el.detailTabButtons.forEach((btn) => {
    btn.addEventListener("click", () => setDetailTab(btn.dataset.detailTab || "overview"));
  });

  el.connectBtn.addEventListener("click", async () => {
    readWorkbenchControlsToState();
    localStorage.setItem("scoutml_api_base", state.apiBase);
    try {
      const ready = await loadHealthAndMetrics();
      if (!ready) return;
      await refreshCoverageAndOptions();
      await runQuery();
      await refreshWatchlist();
    } catch (err) {
      setStatus("error", err instanceof Error ? err.message : String(err));
    }
  });

  el.refresh.addEventListener("click", async () => {
    state.offset = 0;
    await runQuery();
  });

  el.reset.addEventListener("click", async () => {
    resetWorkbenchControls();
    await refreshCoverageAndOptions();
    await runQuery();
  });

  el.prevBtn.addEventListener("click", async () => {
    state.offset = Math.max(state.offset - state.limit, 0);
    await runQuery();
  });

  el.nextBtn.addEventListener("click", async () => {
    state.offset = state.offset + state.limit;
    await runQuery();
  });

  el.search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(async () => {
      state.offset = 0;
      await runQuery();
    }, 300);
  });

  el.split.addEventListener("change", async () => {
    readWorkbenchControlsToState();
    state.offset = 0;
    await refreshCoverageAndOptions();
    await runQuery();
    await refreshWatchlist();
    renderTrustCard();
  });

  [
    el.mode,
    el.season,
    el.league,
    el.position,
    el.roleNeed,
    el.minMinutes,
    el.minAge,
    el.maxAge,
    el.budgetBand,
    el.maxContractYears,
    el.minConfidence,
    el.minGap,
    el.topN,
    el.sort,
    el.sortDir,
    el.limit,
    el.outsideBig5Only,
    el.undervaluedOnly,
  ].forEach((control) => {
    control.addEventListener("change", async () => {
      state.offset = 0;
      await runQuery();
    });
  });

  el.exportBtn.addEventListener("click", () => {
    const diagnostics = state.queryDiagnostics || {};
    const rows = state.rows.map((row, idx) =>
      buildRecruitmentExportRow(row, {
        source: `workbench_${state.mode}`,
        split: state.split,
        rank: state.offset + idx + 1,
        rankingDriver: diagnostics.score_column || state.sortBy,
        rankingBasis: diagnostics.ranking_basis || (state.mode === "shortlist" ? "shortlist" : "manual_sort"),
      })
    );
    downloadCsv(rows, `scoutml_recruitment_board_${state.mode}_${state.split}.csv`);
  });
  if (el.exportPackBtn) {
    el.exportPackBtn.addEventListener("click", () => {
      const diagnostics = state.queryDiagnostics || {};
      const pack = buildWindowPack(state.rows, {
        source: `workbench_${state.mode}`,
        split: state.split,
        filters: {
          ...currentWorkbenchWorkflowSummary(),
          season: state.season || null,
          league: state.league || null,
          position: state.position || null,
          search: state.search || null,
          minMinutes: state.minMinutes,
          minConfidence: state.minConfidence,
          minGapEur: state.minGapEur,
        },
        diagnostics,
        rankingDriver: diagnostics.score_column || state.sortBy,
        rankingBasis: diagnostics.ranking_basis || (state.mode === "shortlist" ? "shortlist" : "manual_sort"),
      });
      downloadJson(pack, `scoutml_window_pack_${state.mode}_${state.split}.json`);
    });
  }

  el.tbody.addEventListener("click", async (event) => {
    const tr = event.target.closest("tr");
    if (!tr || !tr.dataset.index) return;
    const idx = Number(tr.dataset.index);
    if (!Number.isFinite(idx)) return;
    await loadDetailWithReport(state.rows[idx]);
  });

  el.funnelBody.addEventListener("click", async (event) => {
    const tr = event.target.closest("tr");
    if (!tr || !tr.dataset.index) return;
    const idx = Number(tr.dataset.index);
    if (!Number.isFinite(idx)) return;
    const row = state.funnelTopRows[idx];
    if (!row) return;
    setView("workbench");
    await loadDetailWithReport(row);
  });

  if (el.detailOpenProfile) {
    el.detailOpenProfile.addEventListener("click", async () => {
      if (!state.selectedRow) return;
      await ensureSelectedReport();
      openProfileModal();
    });
  }

  el.detailExportJson.addEventListener("click", async () => {
    if (!state.selectedRow) return;
    const report = await ensureSelectedReport();
    const memo = buildPlayerMemoPayload(state.selectedRow, report);
    const playerToken = sanitizeFileToken(state.selectedRow.player_id || state.selectedRow.name);
    const splitToken = sanitizeFileToken(state.split, "split");
    downloadJson(memo, `scoutml_player_memo_${playerToken}_${splitToken}.json`);
  });

  el.detailExportCsv.addEventListener("click", async () => {
    if (!state.selectedRow) return;
    const report = await ensureSelectedReport();
    const row = buildPlayerMemoCsvRow(state.selectedRow, report);
    const playerToken = sanitizeFileToken(state.selectedRow.player_id || state.selectedRow.name);
    const splitToken = sanitizeFileToken(state.split, "split");
    downloadCsv([row], `scoutml_player_memo_${playerToken}_${splitToken}.csv`);
  });

  if (el.profileModalExportJson) {
    el.profileModalExportJson.addEventListener("click", async () => {
      if (!state.selectedRow) return;
      const report = await ensureSelectedReport();
      const memo = buildPlayerMemoPayload(state.selectedRow, report);
      const playerToken = sanitizeFileToken(state.selectedRow.player_id || state.selectedRow.name);
      const splitToken = sanitizeFileToken(state.split, "split");
      downloadJson(memo, `scoutml_player_profile_${playerToken}_${splitToken}.json`);
    });
  }
  if (el.profileModalExportCsv) {
    el.profileModalExportCsv.addEventListener("click", async () => {
      if (!state.selectedRow) return;
      const report = await ensureSelectedReport();
      const row = buildPlayerMemoCsvRow(state.selectedRow, report);
      const playerToken = sanitizeFileToken(state.selectedRow.player_id || state.selectedRow.name);
      const splitToken = sanitizeFileToken(state.split, "split");
      downloadCsv([row], `scoutml_player_profile_${playerToken}_${splitToken}.csv`);
    });
  }
  if (el.profileModalCloseBtn) {
    el.profileModalCloseBtn.addEventListener("click", closeProfileModal);
  }
  if (el.profileModalScrim) {
    el.profileModalScrim.addEventListener("click", closeProfileModal);
  }

  el.funnelRunBtn.addEventListener("click", runFunnel);
  el.funnelExportBtn.addEventListener("click", () => {
    const diagnostics = state.funnelDiagnostics || {};
    const rows = state.funnelTopRows.map((row, idx) =>
      buildRecruitmentExportRow(row, {
        source: "target_funnel",
        split: el.funnelSplit.value,
        rank: idx + 1,
        rankingDriver: diagnostics.score_column || "scout_target_score",
        rankingBasis: diagnostics.ranking_basis || "funnel_rank",
      })
    );
    downloadCsv(rows, `scoutml_talent_funnel_${el.funnelSplit.value}.csv`);
  });
  if (el.watchlistAddBtn) {
    el.watchlistAddBtn.addEventListener("click", addSelectedToWatchlist);
  }
  if (el.watchlistRefreshBtn) {
    el.watchlistRefreshBtn.addEventListener("click", refreshWatchlist);
  }
  if (el.watchlistExportBtn) {
    el.watchlistExportBtn.addEventListener("click", () => {
      const rows = state.watchlistRows.map((row, idx) =>
        buildRecruitmentExportRow(row, {
          source: "watchlist",
          split: state.split,
          rank: idx + 1,
          rankingDriver: "watchlist",
          rankingBasis: "manual_watchlist",
        })
      );
      downloadCsv(rows, `scoutml_watchlist_${state.split}.csv`);
    });
  }
  if (el.watchlistExportJsonBtn) {
    el.watchlistExportJsonBtn.addEventListener("click", () => {
      const pack = buildWindowPack(state.watchlistRows, {
        source: "watchlist",
        split: state.split,
        filters: { watchlist: true },
        diagnostics: null,
        rankingDriver: "watchlist",
        rankingBasis: "manual_watchlist",
      });
      downloadJson(pack, `scoutml_watchlist_pack_${state.split}.json`);
    });
  }
  if (el.watchlistBody) {
    el.watchlistBody.addEventListener("click", async (event) => {
      const btn = event.target.closest(".watchlist-delete");
      if (!btn) return;
      await deleteWatchlistItem(btn.dataset.watchId);
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.profileModalOpen) {
      closeProfileModal();
    }
  });
}

async function boot() {
  syncWorkbenchControlsFromState();
  renderModeAffordances();
  renderDetailTab();
  bindEvents();

  try {
    const ready = await loadHealthAndMetrics();
    if (!ready) return;
    await refreshCoverageAndOptions();
    await runQuery();
    await refreshWatchlist();
  } catch {
    el.tbody.innerHTML = `<tr><td colspan=\"10\">Connect API to start the recruitment board. Expected backend: ${state.apiBase}</td></tr>`;
    el.funnelMeta.textContent = "Connect API before building the recruitment funnel.";
    if (el.watchlistMeta) {
      el.watchlistMeta.textContent = "Connect API before using the recruitment watchlist.";
    }
  }
}

document.addEventListener("DOMContentLoaded", boot);
