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
const shortDateFmt = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});
const RESULTS_TABLE_COLSPAN = 6;
const DEFAULT_PLAYSTYLE = localStorage.getItem("scoutml_playstyle_lens") || "";
const DEFAULT_ROLE_LENS = localStorage.getItem("scoutml_role_lens") || "";
const DEFAULT_SYSTEM_FIT_TEMPLATE = "high_press_433";
const DEFAULT_SYSTEM_FIT_LANE = "valuation";

const INJECTED_API_BASE =
  typeof window !== "undefined" && typeof window.SCOUTING_API_BASE === "string"
    ? window.SCOUTING_API_BASE.trim()
    : "";
const DEFAULT_API = INJECTED_API_BASE || localStorage.getItem("scoutml_api_base") || "http://127.0.0.1:8000";
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
  initializing: true,

  view: "workbench",

  mode: "shortlist",
  split: "test",
  season: "",
  league: "",
  position: "",
  roleNeed: "",
  systemFitTemplate: DEFAULT_SYSTEM_FIT_TEMPLATE,
  systemFitActiveLane: DEFAULT_SYSTEM_FIT_LANE,
  systemFitSelectedSlot: "",
  playstyle: DEFAULT_PLAYSTYLE,
  roleLens: DEFAULT_ROLE_LENS,
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
  topPicks: [],
  total: 0,
  count: 0,
  selectedRow: null,
  selectedProfile: null,
  selectedReport: null,
  selectedHistory: null,
  selectedSimilar: null,
  selectedTrajectory: null,
  selectedLatestDecision: null,
  detailDecisionSourceSurface: "",
  decisionDraftAction: "",
  decisionDraftReasons: [],
  decisionDraftNote: "",
  activeDetailTab: "overview",
  profileModalOpen: false,
  detailRequestId: 0,
  reportCache: new Map(),
  profileCache: new Map(),
  similarCache: new Map(),
  trajectoryCache: new Map(),

  health: null,
  metrics: null,
  modelManifest: null,
  benchmark: null,
  activeArtifacts: null,
  operatorHealth: null,
  coverageRows: [],
  queryDiagnostics: null,
  funnelDiagnostics: null,
  systemFitTemplates: [],
  systemFitSlots: [],
  systemFitLanePosture: null,
  systemFitFiltersApplied: null,

  funnelRows: [],
  funnelTopRows: [],
  watchlistRows: [],
  watchlistTotal: 0,
  teamEnabled: false,
  teamAuthenticated: false,
  teamUser: null,
  teamWorkspaces: [],
  teamActiveWorkspace: null,
  teamAssignments: [],
  teamComments: [],
  teamActivity: [],
  teamCompareLists: [],
  teamCompareTray: [],
  teamPreferenceProfile: null,
  teamApplyPreferences: true,
};

const el = {
  apiBase: document.getElementById("api-base"),
  connectBtn: document.getElementById("connect-btn"),
  apiStatus: document.getElementById("api-status"),
  heroExploreBtn: document.getElementById("hero-explore-btn"),
  teamStatus: document.getElementById("team-status"),
  teamAuthMeta: document.getElementById("team-auth-meta"),
  teamWorkspaceBanner: document.getElementById("team-workspace-banner"),
  teamCurrentWorkspace: document.getElementById("team-current-workspace"),
  teamCurrentUser: document.getElementById("team-current-user"),
  teamEmail: document.getElementById("team-email"),
  teamPassword: document.getElementById("team-password"),
  teamFullName: document.getElementById("team-full-name"),
  teamWorkspaceName: document.getElementById("team-workspace-name"),
  teamInviteToken: document.getElementById("team-invite-token"),
  teamWorkspaceSelect: document.getElementById("team-workspace-select"),
  teamLoginBtn: document.getElementById("team-login-btn"),
  teamBootstrapBtn: document.getElementById("team-bootstrap-btn"),
  teamAcceptInviteBtn: document.getElementById("team-accept-invite-btn"),
  teamLogoutBtn: document.getElementById("team-logout-btn"),
  teamNewWorkspaceName: document.getElementById("team-new-workspace-name"),
  teamInviteEmail: document.getElementById("team-invite-email"),
  teamInviteRole: document.getElementById("team-invite-role"),
  teamCreateWorkspaceBtn: document.getElementById("team-create-workspace-btn"),
  teamCreateInviteBtn: document.getElementById("team-create-invite-btn"),
  teamInviteOutput: document.getElementById("team-invite-output"),

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
  operatorIngestionStatus: document.getElementById("operator-ingestion-status"),
  operatorIngestionCopy: document.getElementById("operator-ingestion-copy"),
  operatorValuationLane: document.getElementById("operator-valuation-lane"),
  operatorValuationCopy: document.getElementById("operator-valuation-copy"),
  operatorFutureLane: document.getElementById("operator-future-lane"),
  operatorFutureCopy: document.getElementById("operator-future-copy"),
  operatorPromotionStatus: document.getElementById("operator-promotion-status"),
  operatorPromotionCopy: document.getElementById("operator-promotion-copy"),
  operatorStaleStatus: document.getElementById("operator-stale-status"),
  operatorStaleCopy: document.getElementById("operator-stale-copy"),
  operatorLiveStatus: document.getElementById("operator-live-status"),
  operatorLiveCopy: document.getElementById("operator-live-copy"),
  operatorBlockedList: document.getElementById("operator-blocked-list"),
  operatorLaneList: document.getElementById("operator-lane-list"),

  mode: document.getElementById("mode-select"),
  split: document.getElementById("split-select"),
  systemFitTemplate: document.getElementById("system-fit-template-select"),
  systemFitTemplateControl: document.getElementById("system-fit-template-control"),
  systemFitActiveLane: document.getElementById("system-fit-active-lane-select"),
  systemFitLaneControl: document.getElementById("system-fit-lane-control"),
  season: document.getElementById("season-select"),
  league: document.getElementById("league-select"),
  position: document.getElementById("position-select"),
  roleNeed: document.getElementById("role-need-select"),
  roleNeedControl: document.getElementById("role-need-control"),
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
  teamPreferencesPanel: document.getElementById("team-preferences-panel"),
  teamPrefName: document.getElementById("team-pref-name"),
  teamPrefAgeMin: document.getElementById("team-pref-age-min"),
  teamPrefAgeMax: document.getElementById("team-pref-age-max"),
  teamPrefBudgetPosture: document.getElementById("team-pref-budget-posture"),
  teamPrefTrustPosture: document.getElementById("team-pref-trust-posture"),
  teamPrefRisk: document.getElementById("team-pref-risk"),
  teamPrefLane: document.getElementById("team-pref-lane"),
  teamPrefSystemTemplate: document.getElementById("team-pref-system-template"),
  teamPrefRolePriorities: document.getElementById("team-pref-role-priorities"),
  teamPrefMustHaveTags: document.getElementById("team-pref-must-have-tags"),
  teamPrefAvoidTags: document.getElementById("team-pref-avoid-tags"),
  teamPrefApply: document.getElementById("team-pref-apply"),
  teamPrefSaveBtn: document.getElementById("team-pref-save-btn"),
  teamPrefMeta: document.getElementById("team-pref-meta"),
  playstyle: document.getElementById("playstyle-select"),
  roleLens: document.getElementById("role-lens-select"),
  topPicksTitle: document.getElementById("top-picks-title"),
  topPicksMeta: document.getElementById("top-picks-meta"),
  boardLaneStatus: document.getElementById("board-lane-status"),
  topPicksGrid: document.getElementById("top-picks-grid"),
  teamCompareSection: document.getElementById("team-compare-section"),
  teamCompareMeta: document.getElementById("team-compare-meta"),
  teamCompareName: document.getElementById("team-compare-name"),
  teamCompareSelect: document.getElementById("team-compare-select"),
  teamCompareCreateBtn: document.getElementById("team-compare-create-btn"),
  teamCompareRefreshBtn: document.getElementById("team-compare-refresh-btn"),
  teamCompareSaveBtn: document.getElementById("team-compare-save-btn"),
  teamCompareTray: document.getElementById("team-compare-tray"),
  teamCompareLists: document.getElementById("team-compare-lists"),
  boardAnchor: document.getElementById("board-anchor"),
  boardHighlightTopPick: document.getElementById("board-highlight-top-pick"),
  boardHighlightTopPickNote: document.getElementById("board-highlight-top-pick-note"),
  boardHighlightGap: document.getElementById("board-highlight-gap"),
  boardHighlightGapNote: document.getElementById("board-highlight-gap-note"),
  boardHighlightMix: document.getElementById("board-highlight-mix"),
  boardHighlightMixNote: document.getElementById("board-highlight-mix-note"),

  title: document.getElementById("results-title"),
  resultCount: document.getElementById("result-count"),
  resultRange: document.getElementById("result-range"),
  resultsNote: document.getElementById("results-note"),
  systemFitSlotWrap: document.getElementById("system-fit-slot-wrap"),
  systemFitSlotMeta: document.getElementById("system-fit-slot-meta"),
  systemFitSlotBar: document.getElementById("system-fit-slot-bar"),
  resultsColTarget: document.getElementById("results-col-target"),
  resultsColDecision: document.getElementById("results-col-decision"),
  resultsColMarket: document.getElementById("results-col-market"),
  resultsColExpected: document.getElementById("results-col-expected"),
  resultsColGap: document.getElementById("results-col-gap"),
  resultsColConfidence: document.getElementById("results-col-confidence"),
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
  detailLatestDecision: document.getElementById("detail-latest-decision"),
  detailLatestDecisionPill: document.getElementById("detail-latest-decision-pill"),
  detailLatestDecisionMeta: document.getElementById("detail-latest-decision-meta"),
  detailLatestDecisionSummary: document.getElementById("detail-latest-decision-summary"),
  detailLatestDecisionNote: document.getElementById("detail-latest-decision-note"),
  detailDecisionActionButtons: Array.from(document.querySelectorAll(".scout-decision-action")),
  detailDecisionReasons: document.getElementById("detail-decision-reasons"),
  detailDecisionNoteInput: document.getElementById("detail-decision-note-input"),
  detailDecisionSaveBtn: document.getElementById("detail-decision-save-btn"),
  detailDecisionClearBtn: document.getElementById("detail-decision-clear-btn"),
  detailDecisionMeta: document.getElementById("detail-decision-meta"),
  teamCollaborationSection: document.getElementById("team-collaboration-section"),
  teamAssigneeSelect: document.getElementById("team-assignee-select"),
  teamAssignmentStatus: document.getElementById("team-assignment-status"),
  teamAssignmentDue: document.getElementById("team-assignment-due"),
  teamAssignmentNote: document.getElementById("team-assignment-note"),
  teamAssignmentSaveBtn: document.getElementById("team-assignment-save-btn"),
  teamAssignmentMeta: document.getElementById("team-assignment-meta"),
  teamAssignmentList: document.getElementById("team-assignment-list"),
  teamCommentsList: document.getElementById("team-comments-list"),
  teamCommentInput: document.getElementById("team-comment-input"),
  teamCommentSaveBtn: document.getElementById("team-comment-save-btn"),
  teamCommentMeta: document.getElementById("team-comment-meta"),
  teamActivityList: document.getElementById("team-activity-list"),
  teamCompareAddBtn: document.getElementById("team-compare-add-btn"),
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
  detailFitCard: document.getElementById("detail-fit-card"),
  detailFitSummary: document.getElementById("detail-fit-summary"),
  detailFitDrivers: document.getElementById("detail-fit-drivers"),
  detailFreshnessCard: document.getElementById("detail-freshness-card"),
  detailFreshnessSummary: document.getElementById("detail-freshness-summary"),
  detailFreshnessMeta: document.getElementById("detail-freshness-meta"),
  detailTalentCard: document.getElementById("detail-talent-card"),
  detailTalentSummary: document.getElementById("detail-talent-summary"),
  detailTalentScores: document.getElementById("detail-talent-scores"),
  detailTalentDrivers: document.getElementById("detail-talent-drivers"),
  detailContextGlance: document.getElementById("detail-context-glance"),
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
  detailProxySection: document.getElementById("detail-proxy-section"),
  detailProxySummary: document.getElementById("detail-proxy-summary"),
  detailProxyList: document.getElementById("detail-proxy-list"),
  detailTrajectoryBadge: document.getElementById("detail-trajectory-badge"),
  detailTrajectorySummary: document.getElementById("detail-trajectory-summary"),
  detailTrajectoryProject: document.getElementById("detail-trajectory-project"),
  detailTrajectoryChart: document.getElementById("detail-trajectory-chart"),
  detailTrajectoryTableBody: document.getElementById("detail-trajectory-table-body"),
  detailRadar: document.getElementById("detail-radar"),
  detailRadarMeta: document.getElementById("detail-radar-meta"),
  detailConfidence: document.getElementById("detail-confidence"),
  detailRisks: document.getElementById("detail-risks"),
  detailAvailabilitySummary: document.getElementById("detail-availability-summary"),
  detailAvailabilityList: document.getElementById("detail-availability-list"),
  detailMarketContextSummary: document.getElementById("detail-market-context-summary"),
  detailMarketContextList: document.getElementById("detail-market-context-list"),
  detailExportPdf: document.getElementById("detail-export-pdf"),
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
  profileModalExportPdf: document.getElementById("profile-modal-export-pdf"),
  profileModalExportJson: document.getElementById("profile-modal-export-json"),
  profileModalExportCsv: document.getElementById("profile-modal-export-csv"),
};

function formatCurrency(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return currencyFmt.format(n);
}

function formatSignedCurrency(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  if (n === 0) return currencyFmt.format(0);
  return `${n > 0 ? "+" : "-"}${currencyFmt.format(Math.abs(n))}`;
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

function humanizeScoutDecisionAction(action) {
  return SCOUT_DECISION_LABELS[String(action || "").trim()] || "Decision";
}

function humanizeScoutDecisionTag(tag) {
  const lookup = [...SCOUT_DECISION_REASON_OPTIONS.positive, ...SCOUT_DECISION_REASON_OPTIONS.pass].find(
    (item) => item.key === String(tag || "").trim()
  );
  return lookup?.label || safeText(String(tag || "").replace(/_/g, " "));
}

function scoutDecisionTone(action) {
  if (action === "shortlist") return "pursue";
  if (action === "watch_live") return "watch";
  if (action === "request_report") return "price";
  if (action === "pass") return "pass";
  return "neutral";
}

function actionRequiresReason(action) {
  return action === "shortlist" || action === "pass";
}

function decisionReasonOptions(action) {
  return action === "pass" ? SCOUT_DECISION_REASON_OPTIONS.pass : SCOUT_DECISION_REASON_OPTIONS.positive;
}

function formatDecisionTimestamp(value) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return safeText(value);
  return shortDateFmt.format(parsed);
}

function currentPlaystyleConfig() {
  return PLAYSTYLE_CONFIG[state.playstyle] || null;
}

function playstyleLabel(key = state.playstyle) {
  return PLAYSTYLE_CONFIG[key]?.label || "No style lens";
}

function currentRoleRankingConfig() {
  return ROLE_RANKING_CONFIG[state.roleLens] || null;
}

function roleLensLabel(key = state.roleLens) {
  return ROLE_RANKING_CONFIG[key]?.label || "No role lens";
}

function roleLensPluralLabel(key = state.roleLens) {
  return ROLE_RANKING_CONFIG[key]?.pluralLabel || "No role lens";
}

function roleLensSummaryLabel(key = state.roleLens) {
  const config = ROLE_RANKING_CONFIG[key];
  if (!config?.profileLabel) return roleLensLabel(key).toLowerCase();
  return config.profileLabel.replace(/\s+profile$/i, "");
}

function isSystemFitMode() {
  return state.mode === "system_fit";
}

function systemFitTemplateMap() {
  return new Map((state.systemFitTemplates || []).map((template) => [String(template.template_key || ""), template]));
}

function currentSystemFitTemplate() {
  return systemFitTemplateMap().get(String(state.systemFitTemplate || "")) || null;
}

function currentSystemFitSlot() {
  return (state.systemFitSlots || []).find((slot) => String(slot.slot_key || "") === String(state.systemFitSelectedSlot || "")) || null;
}

function systemFitLaneLabel(lane = state.systemFitActiveLane) {
  return lane === "future_shortlist" ? "Future Potential / Advisory" : "Current Level / Pricing";
}

function currentWorkbenchSourceMode() {
  if (isSystemFitMode()) return "system_fit";
  if (state.mode === "shortlist") return "shortlist";
  if (state.mode === "predictions") return "predictions";
  return "workbench";
}

function currentWorkbenchDecisionSource() {
  if (state.mode === "predictions") return "predictions";
  if (isSystemFitMode()) return "system_fit";
  return "workbench";
}

const SCOUT_DECISION_LABELS = {
  shortlist: "Shortlist",
  watch_live: "Watch Live",
  request_report: "Request Report",
  pass: "Pass",
};

const SCOUT_DECISION_REASON_OPTIONS = {
  positive: [
    { key: "system_fit", label: "System Fit" },
    { key: "price_gap", label: "Price Gap" },
    { key: "trajectory", label: "Trajectory" },
    { key: "role_need", label: "Role Need" },
    { key: "high_confidence", label: "High Confidence" },
    { key: "availability", label: "Availability" },
    { key: "market_opportunity", label: "Market Opportunity" },
  ],
  pass: [
    { key: "too_expensive", label: "Too Expensive" },
    { key: "data_too_thin", label: "Data Too Thin" },
    { key: "league_risk", label: "League Risk" },
    { key: "not_system_fit", label: "Not System Fit" },
    { key: "athletic_concern", label: "Athletic Concern" },
    { key: "technical_ceiling", label: "Technical Ceiling" },
    { key: "injury_risk", label: "Injury Risk" },
    { key: "contract_blocked", label: "Contract Blocked" },
  ],
};

function hasActiveLens() {
  if (isSystemFitMode()) return false;
  return Boolean(state.playstyle || state.roleLens);
}

function activeLensDisplayLabel() {
  if (state.playstyle && state.roleLens) {
    return `${playstyleLabel()} + ${roleLensLabel()}`;
  }
  if (state.playstyle) return playstyleLabel();
  if (state.roleLens) return roleLensLabel();
  return "";
}

function activeLensTitleLabel() {
  if (state.playstyle && state.roleLens) {
    return `${playstyleLabel()} ${roleLensPluralLabel()}`;
  }
  if (state.playstyle) return `${playstyleLabel()} Teams`;
  if (state.roleLens) return roleLensPluralLabel();
  return "";
}

function activeLensProfileLabel() {
  const roleConfig = currentRoleRankingConfig();
  if (state.playstyle && roleConfig) {
    return `${playstyleLabel().toLowerCase()} ${roleConfig.profileLabel}`;
  }
  if (roleConfig) return roleConfig.profileLabel;
  if (state.playstyle) return `${playstyleLabel().toLowerCase()} profile`;
  return "";
}

function activeLensSummaryLabel() {
  if (state.playstyle && state.roleLens) {
    return `${playstyleLabel().toLowerCase()} ${roleLensSummaryLabel()}`;
  }
  if (state.playstyle) return playstyleLabel().toLowerCase();
  if (state.roleLens) return roleLensSummaryLabel();
  return "active";
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
  current_level_score: "Current level score",
  future_potential_score: "Future potential score",
  future_scout_blend_score: "Future scout blend",
  future_scout_blend_score_adjusted: "Future scout blend (trust-adjusted)",
  future_growth_probability: "Future growth probability",
  future_growth_probability_adjusted: "Future growth probability (trust-adjusted)",
  scout_target_score: "Scout target score",
  scout_target_score_adjusted: "Scout target score (trust-adjusted)",
  shortlist_score: "Shortlist score",
  shortlist_score_adjusted: "Shortlist score (trust-adjusted)",
  system_fit_score: "System fit score",
  undervaluation_score: "Undervaluation score",
  value_gap_capped_eur: "Capped conservative gap",
  value_gap_conservative_eur: "Conservative gap",
};

const RANKING_BASIS_LABELS = {
  current_level_pricing_lane: "Current level / pricing lane",
  future_potential_advisory_lane: "Future potential / advisory lane",
  future_target_tuned_blend: "Future-tuned blend of growth signal and current undervaluation",
  future_target_probability: "Future growth probability only",
  guardrailed_gap_confidence_history: "Guardrailed gap x confidence x history",
  guardrailed_gap_confidence_history_efficiency: "Guardrailed gap x confidence x history x value efficiency",
  system_fit_slot_rank: "Backend slot-level system-fit rank",
  manual_sort: "Manual workbench sort",
  funnel_rank: "Talent funnel rank",
};

const PLAYSTYLE_CONFIG = {
  possession: {
    label: "Possession",
    weights: {
      sofa_accuratePassesPercentage: 1.0,
      sb_progressive_passes_per90: 1.0,
      sb_progressive_carries_per90: 0.7,
      sofa_keyPasses_per90: 0.7,
      sb_passes_into_box_per90: 0.8,
      sofa_successfulDribbles_per90: 0.4,
    },
  },
  counter_attacking: {
    label: "Counter-Attacking",
    weights: {
      sb_progressive_carries_per90: 1.0,
      sofa_successfulDribbles_per90: 0.9,
      sofa_expectedGoals_per90: 0.8,
      sofa_totalShots_per90: 0.6,
      sb_passes_into_box_per90: 0.7,
      sb_progressive_passes_per90: 0.5,
    },
  },
  high_press: {
    label: "High Press",
    weights: {
      sb_pressures_per90: 1.0,
      sofa_tackles_per90: 0.8,
      sofa_interceptions_per90: 0.8,
      sofa_totalDuelsWonPercentage: 0.7,
      sb_duel_win_rate: 0.7,
      sofa_keyPasses_per90: 0.3,
    },
  },
  defensive: {
    label: "Defensive",
    weights: {
      sofa_interceptions_per90: 1.0,
      sofa_tackles_per90: 0.9,
      sofa_clearances_per90: 0.8,
      sofa_totalDuelsWonPercentage: 0.9,
      sb_duel_win_rate: 0.9,
      sofa_accuratePassesPercentage: 0.4,
    },
  },
};

const PLAYSTYLE_METRIC_META = {
  sofa_accuratePassesPercentage: { label: "Passing" },
  sb_progressive_passes_per90: { label: "Progression" },
  sb_progressive_carries_per90: { label: "Ball carrying" },
  sofa_keyPasses_per90: { label: "Chance creation" },
  sb_passes_into_box_per90: { label: "Box access" },
  sofa_successfulDribbles_per90: { label: "Dribbling" },
  sb_pressures_per90: { label: "Pressing activity" },
  sofa_tackles_per90: { label: "Tackling" },
  sofa_interceptions_per90: { label: "Interceptions" },
  sofa_totalDuelsWonPercentage: { label: "Duel strength" },
  sb_duel_win_rate: { label: "Duel strength" },
  sofa_clearances_per90: { label: "Clearances" },
  sofa_expectedGoals_per90: { label: "Goal threat" },
  sofa_totalShots_per90: { label: "Shot volume" },
  sofa_shotsOnTarget_per90: { label: "Shot accuracy" },
};

const ROLE_RANKING_CONFIG = {
  ball_playing_cb: {
    label: "Ball-playing CB",
    pluralLabel: "Ball-playing CBs",
    profileLabel: "ball-playing centre-back profile",
    eligibleRoleKeys: ["CB", "DF"],
    weights: {
      sb_progressive_passes_per90: 1.0,
      sofa_accuratePassesPercentage: 0.9,
      sofa_interceptions_per90: 0.6,
      sb_duel_win_rate: 0.6,
      sofa_clearances_per90: 0.4,
    },
  },
  defensive_cb: {
    label: "Defensive CB",
    pluralLabel: "Defensive CBs",
    profileLabel: "defensive centre-back profile",
    eligibleRoleKeys: ["CB", "DF"],
    weights: {
      sofa_clearances_per90: 1.0,
      sofa_interceptions_per90: 0.9,
      sb_duel_win_rate: 0.9,
      sofa_tackles_per90: 0.7,
      sofa_accuratePassesPercentage: 0.3,
    },
  },
  possession_midfielder: {
    label: "Possession Midfielder",
    pluralLabel: "Possession Midfielders",
    profileLabel: "possession midfielder profile",
    eligibleRoleKeys: ["CM", "DM", "MF", "AM"],
    weights: {
      sb_progressive_passes_per90: 1.0,
      sofa_accuratePassesPercentage: 1.0,
      sofa_keyPasses_per90: 0.7,
      sb_progressive_carries_per90: 0.6,
      sofa_interceptions_per90: 0.3,
    },
  },
  ball_winning_midfielder: {
    label: "Ball-winning Midfielder",
    pluralLabel: "Ball-winning Midfielders",
    profileLabel: "ball-winning midfielder profile",
    eligibleRoleKeys: ["DM", "CM", "MF"],
    weights: {
      sofa_tackles_per90: 1.0,
      sofa_interceptions_per90: 1.0,
      sb_duel_win_rate: 0.9,
      sb_pressures_per90: 0.8,
      sb_progressive_passes_per90: 0.3,
    },
  },
  winger: {
    label: "Winger",
    pluralLabel: "Wingers",
    profileLabel: "winger profile",
    eligibleRoleKeys: ["W", "FW"],
    weights: {
      sofa_successfulDribbles_per90: 1.0,
      sb_progressive_carries_per90: 0.9,
      sb_passes_into_box_per90: 0.8,
      sofa_keyPasses_per90: 0.6,
      sofa_expectedGoals_per90: 0.4,
    },
  },
  striker: {
    label: "Striker",
    pluralLabel: "Strikers",
    profileLabel: "striker profile",
    eligibleRoleKeys: ["ST", "SS", "FW"],
    weights: {
      sofa_expectedGoals_per90: 1.0,
      sofa_totalShots_per90: 0.9,
      sofa_shotsOnTarget_per90: 0.8,
      sb_duel_win_rate: 0.4,
      sofa_successfulDribbles_per90: 0.3,
    },
  },
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

function parseFreshnessDate(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  const isoLike = /^\d{4}-\d{2}-\d{2}$/;
  const parsed = new Date(isoLike.test(raw) ? `${raw}T00:00:00Z` : raw);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatFreshnessDate(value) {
  const parsed = parseFreshnessDate(value);
  return parsed ? shortDateFmt.format(parsed) : "";
}

function latestFreshnessValue(values) {
  let bestRaw = null;
  let bestTime = -Infinity;
  values.forEach((value) => {
    const parsed = parseFreshnessDate(value);
    if (!parsed) return;
    const time = parsed.getTime();
    if (time > bestTime) {
      bestTime = time;
      bestRaw = String(value);
    }
  });
  return bestRaw;
}

function rowUsesFutureOverlay(row) {
  return hasAnyFiniteSignal(row, ["future_scout_blend_score", "future_growth_probability", "future_scout_score"]);
}

function buildFreshnessState(row, profileFreshness = null) {
  if (!row) {
    return {
      status: "limited",
      compactLine: "Freshness limited",
      summaryText: "Freshness is partially known because provider snapshot metadata is incomplete.",
      metaLine: "Freshness limited",
      partialSeason: false,
      updatedAt: null,
    };
  }

  const valuationEntry = manifestRoleEntry("valuation");
  const futureEntry = manifestRoleEntry("future_shortlist");
  const rowSeason = String(row?.season || "").trim();
  const liveLane =
    rowUsesFutureOverlay(row) &&
    rowSeason &&
    rowSeason === String(futureEntry?.config?.test_season || "").trim();
  const coverage = rowSignalCoverage(row);
  const hasProviderSignals = Boolean(coverage.statsbomb || coverage.availability || coverage.market);
  const providerUpdatedAt = latestFreshnessValue([
    row?.sb_snapshot_date,
    row?.sb_retrieved_at,
    row?.avail_snapshot_date,
    row?.avail_retrieved_at,
    row?.fixture_snapshot_date,
    row?.fixture_retrieved_at,
    row?.odds_snapshot_date,
    row?.odds_retrieved_at,
    profileFreshness?.latest_snapshot_date,
    profileFreshness?.latest_retrieved_at,
  ]);
  const artifactEntry = liveLane ? futureEntry : valuationEntry;
  const artifactUpdatedAt =
    profileFreshness?.artifact_generated_at_utc ||
    artifactEntry?.generated_at_utc ||
    artifactEntry?.artifacts?.metrics?.mtime_utc ||
    artifactEntry?.artifacts?.test_predictions?.mtime_utc ||
    null;
  const updatedAt = providerUpdatedAt || artifactUpdatedAt || null;
  const providerMetaKnown = Boolean(providerUpdatedAt);

  let status = String(profileFreshness?.status || "").trim().toLowerCase();
  if (!status) {
    if (!hasProviderSignals || !providerMetaKnown) {
      status = "limited";
    } else if (liveLane) {
      status = "live";
    } else {
      status = "stable";
    }
  }

  const partialSeason = Boolean(profileFreshness?.partial_season ?? (status === "live"));
  const baseSummary =
    profileFreshness?.message ||
    (status === "live"
      ? "Live current-season overlay. Fresh performance context is available, but season outcomes are still in progress."
      : status === "stable"
      ? "Stable valuation artifact. Use for benchmarked pricing and ranking."
      : "Freshness is partially known because provider snapshot metadata is incomplete.");
  const summaryText =
    partialSeason && !/still in progress/i.test(baseSummary)
      ? `${baseSummary} The current season is still in progress.`
      : baseSummary;
  const statusLabel =
    status === "live" ? "Live season" : status === "stable" ? "Stable artifact" : "Freshness limited";
  const compactLine =
    status === "limited" ? "Freshness limited" : `${statusLabel} | updated ${formatFreshnessDate(updatedAt) || "-"}`;
  const metaBits = [statusLabel];
  if (updatedAt) metaBits.push(`updated ${formatFreshnessDate(updatedAt)}`);
  const artifactLabel = String(profileFreshness?.artifact_label || artifactEntry?.label || "").trim();
  if (artifactLabel) metaBits.push(artifactLabel);
  return {
    status,
    compactLine,
    summaryText,
    metaLine: metaBits.join(" | "),
    partialSeason,
    updatedAt,
  };
}

function buildBadgeChipMarkup(label, tone = "neutral") {
  return `<span class="badge-chip badge-chip--${escapeHtml(tone)}">${escapeHtml(label)}</span>`;
}

function classifyTalentConfidence(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) {
    return { label: "Unknown", tone: "neutral", compact: "confidence unknown" };
  }
  if (n >= 70) return { label: "High", tone: "pursue", compact: "high confidence" };
  if (n >= 45) return { label: "Medium", tone: "watch", compact: "medium confidence" };
  return { label: "Low", tone: "price", compact: "low confidence" };
}

function getTalentView(row, profile = null) {
  const talent = profile?.talent_view && typeof profile.talent_view === "object" ? profile.talent_view : {};
  const scoreFamilies =
    talent.score_families && typeof talent.score_families === "object"
      ? talent.score_families
      : row?.score_families && typeof row.score_families === "object"
      ? row.score_families
      : {};
  const scoreExplanations =
    talent.score_explanations && typeof talent.score_explanations === "object"
      ? talent.score_explanations
      : row?.score_explanations && typeof row.score_explanations === "object"
      ? row.score_explanations
      : {};
  return {
    talent_position_family: safeText(
      talent.talent_position_family || row?.talent_position_family || inferRoleKey(row) || getPosition(row) || "-"
    ),
    current_level_score: firstFiniteNumber(talent.current_level_score, row?.current_level_score),
    future_potential_score: firstFiniteNumber(talent.future_potential_score, row?.future_potential_score),
    current_level_confidence: firstFiniteNumber(talent.current_level_confidence, row?.current_level_confidence),
    future_potential_confidence: firstFiniteNumber(talent.future_potential_confidence, row?.future_potential_confidence),
    current_level_confidence_reasons: Array.isArray(talent.current_level_confidence_reasons)
      ? talent.current_level_confidence_reasons
      : Array.isArray(row?.current_level_confidence_reasons)
      ? row.current_level_confidence_reasons
      : [],
    future_potential_confidence_reasons: Array.isArray(talent.future_potential_confidence_reasons)
      ? talent.future_potential_confidence_reasons
      : Array.isArray(row?.future_potential_confidence_reasons)
      ? row.future_potential_confidence_reasons
      : [],
    score_families: scoreFamilies,
    score_explanations: scoreExplanations,
  };
}

function activeTalentLaneKey(row) {
  if (isSystemFitMode()) {
    return state.systemFitActiveLane === "future_shortlist" ? "future_potential" : "current_level";
  }
  const freshness = buildFreshnessState(row);
  return freshness.status === "live" || state.mode === "shortlist" ? "future_potential" : "current_level";
}

function topTalentDrivers(row, profile = null, laneKey = activeTalentLaneKey(row)) {
  const talent = getTalentView(row, profile);
  const entries = Array.isArray(talent.score_explanations?.[laneKey]) ? talent.score_explanations[laneKey] : [];
  return entries.slice(0, 3);
}

function buildTalentCompactLine(row, profile = null) {
  const talent = getTalentView(row, profile);
  const laneKey = activeTalentLaneKey(row);
  const confidenceScore =
    laneKey === "future_potential" ? talent.future_potential_confidence : talent.current_level_confidence;
  const confidence = classifyTalentConfidence(confidenceScore);
  const currentLevel = Number(talent.current_level_score);
  const futurePotential = Number(talent.future_potential_score);
  return [
    Number.isFinite(currentLevel) ? `Current ${formatNumber(currentLevel)}` : null,
    Number.isFinite(futurePotential) ? `Future ${formatNumber(futurePotential)}` : null,
    confidence.compact,
  ]
    .filter(Boolean)
    .join(" | ");
}

function buildTalentDriverLine(row, profile = null, laneKey = activeTalentLaneKey(row)) {
  const entries = topTalentDrivers(row, profile, laneKey);
  if (!entries.length) return "";
  return entries
    .slice(0, 2)
    .map((entry) => `${safeText(entry.label)} ${formatNumber(entry.score)}`)
    .join(" | ");
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
  const adjustment = leagueAdjustmentMeta(row);
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
  if (adjustment.needsWarning) badges.push({ label: adjustment.label, tone: adjustment.tone });
  return badges;
}

function scoreValueForRow(row, scoreColumn) {
  const n = Number(row?.[scoreColumn]);
  if (!Number.isFinite(n)) return NaN;
  if (
    [
      "value_gap_capped_eur",
      "value_gap_conservative_eur",
      "value_gap_eur",
      "undervaluation_score",
      "fair_value_eur",
      "expected_value_eur",
    ].includes(String(scoreColumn || ""))
  ) {
    const reliability = firstFiniteNumber(row?.discovery_reliability_weight);
    if (Number.isFinite(reliability)) return n * reliability;
  }
  return n;
}

function resolveRowScoreContext(row, diagnostics = null, source = "workbench") {
  const systemFitSource = source === "system_fit" || isSystemFitMode();
  const scoreColumn = systemFitSource
    ? "system_fit_score"
    : diagnostics?.score_column ||
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
  const rankingBasis = systemFitSource
    ? "system_fit_slot_rank"
    : diagnostics?.ranking_basis ||
      diagnostics?.rankingBasis ||
      (source === "predictions" ? "manual_sort" : source === "funnel" ? "funnel_rank" : null) ||
      (scoreColumn === "current_level_score"
        ? "current_level_pricing_lane"
        : scoreColumn === "future_potential_score"
        ? "future_potential_advisory_lane"
        : scoreColumn === "future_scout_blend_score"
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

function firstFiniteMetric(row, keys) {
  const list = Array.isArray(keys) ? keys : [keys];
  for (const key of list) {
    const n = Number(row?.[key]);
    if (Number.isFinite(n)) return n;
  }
  return NaN;
}

function collectMetricValues(rows, metricKey) {
  if (!Array.isArray(rows) || !rows.length) return [];
  return rows
    .map((row) => firstFiniteMetric(row, metricKey))
    .filter((value) => Number.isFinite(value));
}

function metricLabel(metricKey) {
  return PLAYSTYLE_METRIC_META[String(metricKey || "").trim()]?.label || humanizeKey(metricKey || "metric");
}

function mergeWeightMaps(...maps) {
  return maps.reduce((acc, map) => {
    if (!map) return acc;
    Object.entries(map).forEach(([metricKey, weight]) => {
      const numericWeight = Number(weight);
      if (!Number.isFinite(numericWeight) || numericWeight <= 0) return;
      acc[metricKey] = (acc[metricKey] || 0) + numericWeight;
    });
    return acc;
  }, {});
}

function rowEligibleForRoleLens(row, roleConfig) {
  if (!roleConfig) return true;
  const allowed = Array.isArray(roleConfig.eligibleRoleKeys) ? roleConfig.eligibleRoleKeys : [];
  if (!allowed.length) return true;
  return allowed.includes(inferRoleKey(row));
}

function normalizeMetricAgainstRows(value, values) {
  if (!Number.isFinite(value)) return 0;
  const finiteValues = Array.isArray(values) ? values.filter((item) => Number.isFinite(item)) : [];
  if (!finiteValues.length) return 0;
  let min = finiteValues[0];
  let max = finiteValues[0];
  for (const item of finiteValues) {
    if (item < min) min = item;
    if (item > max) max = item;
  }
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function computeStyleScore(player, weights, rows, metricValuesByKey = null) {
  if (!player || !weights || !rows?.length) return 0;
  let weightedSum = 0;
  let totalWeight = 0;
  for (const [metricKey, weight] of Object.entries(weights)) {
    const numericWeight = Number(weight);
    if (!Number.isFinite(numericWeight) || numericWeight <= 0) continue;
    const playerValue = firstFiniteMetric(player, metricKey);
    if (!Number.isFinite(playerValue)) continue;
    const values = metricValuesByKey?.[metricKey] || collectMetricValues(rows, metricKey);
    const normalized = normalizeMetricAgainstRows(playerValue, values);
    weightedSum += normalized * numericWeight;
    totalWeight += numericWeight;
  }
  if (totalWeight <= 0) return 0;
  return weightedSum / totalWeight;
}

function computeLensScore(styleScore, roleScore) {
  const safeStyle = Number.isFinite(styleScore) ? styleScore : 0;
  const safeRole = Number.isFinite(roleScore) ? roleScore : 0;
  if (state.playstyle && state.roleLens) return (2 * safeStyle + safeRole) / 3;
  if (state.playstyle) return safeStyle;
  if (state.roleLens) return safeRole;
  return 0;
}

function uniqueDriverLabels(drivers, limit = 3) {
  const seen = new Set();
  const labels = [];
  for (const driver of drivers || []) {
    const label = String(driver?.label || "").trim();
    const key = label.toLowerCase();
    if (!label || seen.has(key)) continue;
    seen.add(key);
    labels.push(label);
    if (labels.length >= limit) break;
  }
  return labels;
}

function joinNaturalList(labels) {
  const items = (labels || []).filter(Boolean).map((label) => String(label).toLowerCase());
  if (!items.length) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function computeStyleFitDrivers(player, weights, rows, metricValuesByKey = null) {
  if (!player || !weights || !rows?.length) {
    return {
      drivers: [],
      availableMetricCount: 0,
      availableWeight: 0,
      totalConfiguredWeight: 0,
      totalWeightedContribution: 0,
    };
  }

  const drivers = [];
  let availableMetricCount = 0;
  let availableWeight = 0;
  let totalConfiguredWeight = 0;
  let totalWeightedContribution = 0;

  for (const [metricKey, weight] of Object.entries(weights)) {
    const numericWeight = Number(weight);
    if (!Number.isFinite(numericWeight) || numericWeight <= 0) continue;
    totalConfiguredWeight += numericWeight;
    const playerValue = firstFiniteMetric(player, metricKey);
    if (!Number.isFinite(playerValue)) continue;
    const values = metricValuesByKey?.[metricKey] || collectMetricValues(rows, metricKey);
    const normalizedValue = normalizeMetricAgainstRows(playerValue, values);
    const contribution = normalizedValue * numericWeight;
    drivers.push({
      metricKey,
      label: metricLabel(metricKey),
      contribution,
      value: playerValue,
      normalizedValue,
    });
    availableMetricCount += 1;
    availableWeight += numericWeight;
    totalWeightedContribution += contribution;
  }

  drivers.sort((a, b) => {
    if (b.contribution !== a.contribution) return b.contribution - a.contribution;
    if (b.normalizedValue !== a.normalizedValue) return b.normalizedValue - a.normalizedValue;
    return a.label.localeCompare(b.label);
  });

  return {
    drivers: drivers.slice(0, 3),
    availableMetricCount,
    availableWeight,
    totalConfiguredWeight,
    totalWeightedContribution,
  };
}

function buildLensFitExplanation(
  player,
  weights,
  rows,
  metricValuesByKey = null,
  lensScore = NaN,
  { roleEligible = true } = {}
) {
  if (!hasActiveLens() || !player || !weights || !rows?.length) {
    return {
      _styleFitDrivers: [],
      _styleFitReasonLine: "",
      _styleFitSummary: "",
      _styleFitDataQuality: "",
    };
  }

  const summaryLabel = activeLensSummaryLabel();
  const profileLabel = activeLensProfileLabel();
  const fitInfo = computeStyleFitDrivers(player, weights, rows, metricValuesByKey);
  const availableWeightShare =
    fitInfo.totalConfiguredWeight > 0 ? fitInfo.availableWeight / fitInfo.totalConfiguredWeight : 0;
  const limited =
    !roleEligible ||
    fitInfo.availableMetricCount < 2 ||
    availableWeightShare < 0.45 ||
    fitInfo.totalWeightedContribution <= 0;
  const labels = uniqueDriverLabels(fitInfo.drivers, 3);
  const driverPhrase = joinNaturalList(labels.slice(0, 2));
  const limitedScope = state.playstyle && state.roleLens ? "style and role" : state.roleLens ? "role" : "style";

  if (!fitInfo.drivers.length || limited) {
    return {
      _styleFitDrivers: fitInfo.drivers,
      _styleFitReasonLine: `Fit assessment is based on limited available ${limitedScope} data.`,
      _styleFitSummary: `Current ${summaryLabel} fit is directionally estimated from limited available ${limitedScope} data.`,
      _styleFitDataQuality: "limited",
    };
  }

  let summary = `Profile aligns with ${summaryLabel} needs through ${driverPhrase}.`;
  if (Number.isFinite(lensScore) && lensScore >= 0.67) {
    summary = `Strong ${summaryLabel} fit driven by ${driverPhrase}.`;
  } else if (Number.isFinite(lensScore) && lensScore < 0.4) {
    summary = `Some ${summaryLabel}-fit signals come from ${driverPhrase}, but the picture is mixed.`;
  }

  const reasonLine =
    state.roleLens && profileLabel
      ? `Fit drivers: ${labels.slice(0, 3).join(", ")} (${profileLabel})`
      : `Fit drivers: ${labels.slice(0, 3).join(", ")}`;

  return {
    _styleFitDrivers: fitInfo.drivers,
    _styleFitReasonLine: reasonLine,
    _styleFitSummary: summary,
    _styleFitDataQuality: "ok",
  };
}

function computeExistingScore(row, diagnostics, sourceMode) {
  const context = resolveRowScoreContext(row, diagnostics, sourceMode);
  if (Number.isFinite(context.scoreValue)) {
    const sortOrder = diagnostics?.sort_order || diagnostics?.sortOrder || state.sortOrder;
    return sourceMode === "predictions" && sortOrder === "asc" ? -context.scoreValue : context.scoreValue;
  }
  const gap = conservativeGapForRanking(row);
  if (Number.isFinite(gap)) return gap;
  if (sourceMode === "shortlist") {
    const confidence = Number(row?.undervaluation_confidence);
    if (Number.isFinite(confidence)) return confidence;
    const undervaluation = Number(row?.undervaluation_score);
    if (Number.isFinite(undervaluation)) return undervaluation;
  }
  return 0;
}

function styleFitLabel(score) {
  if (!hasActiveLens() || !Number.isFinite(score)) return "";
  if (score >= 0.67) return "Strong fit";
  if (score >= 0.4) return "Moderate fit";
  return "Weak fit";
}

function styleFitTone(score) {
  if (!Number.isFinite(score)) return "neutral";
  if (score >= 0.67) return "good";
  if (score >= 0.4) return "warn";
  return "neutral";
}

function buildStyleFitMarkup(row) {
  if (!hasActiveLens() || !row?._styleFitLabel) return "";
  return `<span class="fit-chip fit-chip--${escapeHtml(row._styleFitTone || "neutral")}">${escapeHtml(
    row._styleFitLabel
  )}</span>`;
}

function buildLensFitLine(row) {
  if (!hasActiveLens()) return "";
  if (state.playstyle && state.roleLens) {
    return `Lens fit | ${activeLensDisplayLabel()} | style ${formatNumber(row?._styleScore)} | role ${formatNumber(
      row?._roleScore
    )}`;
  }
  if (state.playstyle) {
    return `Playstyle fit | ${playstyleLabel()} | ${formatNumber(row?._styleScore)}`;
  }
  if (state.roleLens) {
    return `Role fit | ${roleLensLabel()} | ${formatNumber(row?._roleScore)}`;
  }
  return "";
}

function renderDetailFit(row, { loading = false } = {}) {
  if (!el.detailFitCard || !el.detailFitSummary || !el.detailFitDrivers) return;

  if (isSystemFitMode()) {
    if (!row) {
      el.detailFitCard.hidden = true;
      el.detailFitSummary.classList.remove("fit-driver-summary--muted");
      el.detailFitSummary.textContent = "Select a player to inspect slot-level system fit.";
      el.detailFitDrivers.innerHTML = "";
      return;
    }

    if (loading) {
      el.detailFitCard.hidden = false;
      el.detailFitSummary.classList.remove("fit-driver-summary--muted");
      el.detailFitSummary.textContent = "Loading system-fit summary...";
      el.detailFitDrivers.innerHTML = '<span class="fit-driver-pill">Loading fit reasons...</span>';
      return;
    }

    const reasons = Array.isArray(row.fit_reasons) ? row.fit_reasons : [];
    const score = Number(row.system_fit_score);
    const confidence = Number(row.system_fit_confidence);
    const budgetStatus = safeText(row.budget_status || "unbounded").replace(/_/g, " ");
    const slotLabel = safeText(row.slot_label || row.slot_key || "slot");
    const roleLabel = safeText(row.role_template_label || row.role_template_key || "role");
    const scoreText = Number.isFinite(score) ? formatNumber(score) : "-";
    const confidenceText = Number.isFinite(confidence) ? formatNumber(confidence) : "-";
    el.detailFitCard.hidden = false;
    el.detailFitSummary.classList.remove("fit-driver-summary--muted");
    el.detailFitSummary.textContent = `${slotLabel} | ${roleLabel} | system fit ${scoreText} | confidence ${confidenceText} | budget ${budgetStatus}.`;
    el.detailFitDrivers.innerHTML = reasons.length
      ? reasons.map((reason) => `<span class="fit-driver-pill">${escapeHtml(reason)}</span>`).join("")
      : '<span class="fit-driver-pill">No fit reasons returned for this slot.</span>';
    return;
  }

  if (!hasActiveLens()) {
    el.detailFitCard.hidden = true;
    el.detailFitSummary.classList.remove("fit-driver-summary--muted");
    el.detailFitSummary.textContent = "Select a player to see lens fit drivers.";
    el.detailFitDrivers.innerHTML = "";
    return;
  }

  if (loading) {
    el.detailFitCard.hidden = false;
    el.detailFitSummary.classList.remove("fit-driver-summary--muted");
    el.detailFitSummary.textContent = "Loading lens fit...";
    el.detailFitDrivers.innerHTML = '<span class="fit-driver-pill">Loading fit drivers...</span>';
    return;
  }

  if (!row?._styleFitSummary) {
    el.detailFitCard.hidden = true;
    el.detailFitSummary.classList.remove("fit-driver-summary--muted");
    el.detailFitSummary.textContent = "Select a player to see lens fit drivers.";
    el.detailFitDrivers.innerHTML = "";
    return;
  }

  const labels = uniqueDriverLabels(row._styleFitDrivers, 3);
  el.detailFitCard.hidden = false;
  el.detailFitSummary.textContent = row._styleFitSummary;
  el.detailFitSummary.classList.toggle("fit-driver-summary--muted", row._styleFitDataQuality === "limited");
  el.detailFitDrivers.innerHTML = labels.length
    ? labels.map((label) => `<span class="fit-driver-pill">${escapeHtml(label)}</span>`).join("")
    : '<span class="fit-driver-pill">Limited style evidence</span>';
}

function renderDetailFreshness(row, profile = null, { loading = false } = {}) {
  if (!el.detailFreshnessCard || !el.detailFreshnessSummary || !el.detailFreshnessMeta) return;

  if (!row) {
    el.detailFreshnessCard.hidden = true;
    el.detailFreshnessSummary.textContent = "Select a player to inspect freshness context.";
    el.detailFreshnessMeta.textContent = "";
    return;
  }

  if (loading) {
    el.detailFreshnessCard.hidden = false;
    el.detailFreshnessSummary.textContent = "Loading data freshness...";
    el.detailFreshnessMeta.textContent = "Checking artifact lane and provider snapshot metadata.";
    return;
  }

  const freshness = buildFreshnessState(row, profile?.data_freshness || null);
  el.detailFreshnessCard.hidden = false;
  el.detailFreshnessSummary.textContent = freshness.summaryText;
  el.detailFreshnessMeta.textContent = freshness.metaLine;
}

function renderDetailTalentView(row, profile = null, { loading = false, error = "" } = {}) {
  if (!el.detailTalentCard || !el.detailTalentSummary || !el.detailTalentScores || !el.detailTalentDrivers) return;

  if (!row) {
    el.detailTalentCard.hidden = true;
    el.detailTalentSummary.textContent = "Select a player to inspect current level, future potential, and confidence.";
    el.detailTalentScores.innerHTML = "";
    el.detailTalentDrivers.innerHTML = '<li class="risk-item risk-item--none">Select a player to load talent drivers.</li>';
    return;
  }

  if (loading) {
    el.detailTalentCard.hidden = false;
    el.detailTalentSummary.textContent = "Loading talent view...";
    el.detailTalentScores.innerHTML =
      '<div class="talent-score-metric"><span>Talent view</span><strong>Loading</strong><small>Scoring current level, future potential, and confidence.</small></div>';
    el.detailTalentDrivers.innerHTML = '<li class="risk-item risk-item--none">Loading talent drivers...</li>';
    return;
  }

  if (error) {
    el.detailTalentCard.hidden = false;
    el.detailTalentSummary.textContent = `Talent view unavailable: ${error}`;
    el.detailTalentScores.innerHTML = "";
    el.detailTalentDrivers.innerHTML = '<li class="risk-item risk-item--none">Talent drivers unavailable.</li>';
    return;
  }

  const talent = getTalentView(row, profile);
  const laneKey = activeTalentLaneKey(row);
  const currentConf = classifyTalentConfidence(talent.current_level_confidence);
  const futureConf = classifyTalentConfidence(talent.future_potential_confidence);
  const activeConf = laneKey === "future_potential" ? futureConf : currentConf;
  const drivers = topTalentDrivers(row, profile, laneKey);
  const family = safeText(talent.talent_position_family || inferRoleKey(row) || getPosition(row));
  const laneCopy =
    laneKey === "future_potential"
      ? "Future Potential / Advisory is the active lens, so treat this as next-step upside rather than a stable valuation truth."
      : "Current Level / Pricing is the active lens, so this view is anchored to the stable valuation lane.";
  const confidenceReasons =
    laneKey === "future_potential" ? talent.future_potential_confidence_reasons : talent.current_level_confidence_reasons;
  const confidenceReason = Array.isArray(confidenceReasons) && confidenceReasons.length ? confidenceReasons[0] : "";
  el.detailTalentCard.hidden = false;
  el.detailTalentSummary.textContent = `${family} profile. ${laneCopy}${confidenceReason ? ` ${confidenceReason}` : ""}`;
  el.detailTalentScores.innerHTML = `
    <div class="talent-score-metric">
      <span>Current level</span>
      <strong>${Number.isFinite(Number(talent.current_level_score)) ? formatNumber(talent.current_level_score) : "-"}</strong>
      <small>Stable valuation / pricing posture</small>
    </div>
    <div class="talent-score-metric">
      <span>Future potential</span>
      <strong>${Number.isFinite(Number(talent.future_potential_score)) ? formatNumber(talent.future_potential_score) : "-"}</strong>
      <small>Next-step growth probability view</small>
    </div>
    <div class="talent-score-metric">
      <span>Confidence</span>
      <strong>${activeConf.label}</strong>
      <small>${Number.isFinite(Number(laneKey === "future_potential" ? talent.future_potential_confidence : talent.current_level_confidence)) ? formatNumber(laneKey === "future_potential" ? talent.future_potential_confidence : talent.current_level_confidence) : "-"} / 100</small>
    </div>
  `;
  el.detailTalentDrivers.innerHTML = buildNarrativeListMarkup(
    drivers.map((entry) => ({
      label: entry.label,
      message: entry.message,
      tone: entry.tone,
    })),
    "No family-level drivers were available for this player."
  );
}

function applyLensRanking(rows, diagnostics, sourceMode) {
  if (!hasActiveLens() || !Array.isArray(rows) || !rows.length) return [...rows];
  const playstyle = currentPlaystyleConfig();
  const roleConfig = currentRoleRankingConfig();
  const finalWeights = mergeWeightMaps(playstyle?.weights, roleConfig?.weights);
  const metricValuesByKey = Object.keys(finalWeights).reduce((acc, metricKey) => {
    acc[metricKey] = collectMetricValues(rows, metricKey);
    return acc;
  }, {});

  const ranked = rows.map((row) => {
    const existingScore = computeExistingScore(row, diagnostics, sourceMode);
    const roleEligible = rowEligibleForRoleLens(row, roleConfig);
    const styleScore = playstyle ? computeStyleScore(row, playstyle.weights, rows, metricValuesByKey) : 0;
    const roleScore = roleConfig && roleEligible ? computeStyleScore(row, roleConfig.weights, rows, metricValuesByKey) : 0;
    const lensScore = computeLensScore(styleScore, roleScore);
    const finalScore = existingScore + 0.2 * styleScore + 0.1 * roleScore;
    const fitExplanation = buildLensFitExplanation(row, finalWeights, rows, metricValuesByKey, lensScore, {
      roleEligible,
    });
    return {
      ...row,
      _existingScore: existingScore,
      _styleScore: styleScore,
      _roleScore: roleScore,
      _lensScore: lensScore,
      _finalScore: finalScore,
      _styleFitLabel: styleFitLabel(lensScore),
      _styleFitTone: styleFitTone(lensScore),
      ...fitExplanation,
    };
  });

  ranked.sort((a, b) => {
    const finalDiff = (Number(b._finalScore) || 0) - (Number(a._finalScore) || 0);
    if (finalDiff !== 0) return finalDiff;
    const lensDiff = (Number(b._lensScore) || 0) - (Number(a._lensScore) || 0);
    if (lensDiff !== 0) return lensDiff;
    const gapDiff = (conservativeGapForRanking(b) || 0) - (conservativeGapForRanking(a) || 0);
    if (gapDiff !== 0) return gapDiff;
    const confDiff = (Number(b.undervaluation_confidence) || 0) - (Number(a.undervaluation_confidence) || 0);
    if (confDiff !== 0) return confDiff;
    return String(a.name || "").localeCompare(String(b.name || ""));
  });

  return ranked;
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
  const freshness = buildFreshnessState(row, profile?.data_freshness || null);
  const talent = getTalentView(row, profile);
  const laneKey = activeTalentLaneKey(row);
  const talentDrivers = topTalentDrivers(row, profile, laneKey);
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
  items.push({
    label: "Artifact lane",
    message:
      freshness.status === "live"
        ? "This ranking is leaning on the live future_shortlist overlay, so it should be read as fresh but still advisory while the season is in progress."
        : freshness.status === "limited"
        ? "Freshness metadata is limited, so this ranking should be treated more cautiously than a fully benchmarked lane."
        : "This ranking is grounded in the stable valuation lane and can be used as benchmarked pricing/ranking support.",
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
  if (talentDrivers.length) {
    items.push({
      label: laneKey === "future_potential" ? "Future family drivers" : "Current-level drivers",
      message: talentDrivers.map((entry) => entry.message).join(" "),
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
  const confidenceReasons =
    laneKey === "future_potential"
      ? talent.future_potential_confidence_reasons
      : talent.current_level_confidence_reasons;
  if (Array.isArray(confidenceReasons) && confidenceReasons.length) {
    items.push({
      label: "Confidence posture",
      message: confidenceReasons.join(" "),
    });
  }
  return items.slice(0, 6);
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

function currentScoutDecisionSource(row = state.selectedRow) {
  if (state.detailDecisionSourceSurface) return state.detailDecisionSourceSurface;
  if (Number.isFinite(Number(row?._funnelScore))) return "funnel";
  return currentWorkbenchDecisionSource();
}

function currentDecisionRankingContext(row = state.selectedRow) {
  if (!row) return {};
  let sourceRows = [];
  if (currentScoutDecisionSource(row) === "watchlist") {
    sourceRows = Array.isArray(state.watchlistRows) ? state.watchlistRows : [];
  } else if (Number.isFinite(Number(row?._funnelScore))) {
    sourceRows = Array.isArray(state.funnelRows) ? state.funnelRows : [];
  } else {
    sourceRows = Array.isArray(state.rows) ? state.rows : [];
  }
  const rank = sourceRows.findIndex((item) => rowKey(item) === rowKey(row));
  const diagnostics =
    Number.isFinite(Number(row?._funnelScore)) && state.funnelDiagnostics ? state.funnelDiagnostics : state.queryDiagnostics;
  const sourceSurface = currentScoutDecisionSource(row);
  return {
    mode: sourceSurface === "watchlist" ? "watchlist" : currentWorkbenchSourceMode(),
    sort_by: diagnostics?.ranking_score_column || diagnostics?.score_column || state.sortBy,
    rank: rank >= 0 ? rank + 1 : null,
    active_lane: isSystemFitMode()
      ? state.systemFitActiveLane
      : state.mode === "shortlist"
      ? "future_shortlist"
      : "valuation",
    system_template: isSystemFitMode() ? state.systemFitTemplate : null,
    system_slot: isSystemFitMode() ? state.systemFitSelectedSlot : null,
    discovery_reliability_weight: firstFiniteNumber(row?.discovery_reliability_weight),
  };
}

function resetDecisionDraft() {
  state.decisionDraftAction = "";
  state.decisionDraftReasons = [];
  state.decisionDraftNote = "";
  if (el.detailDecisionNoteInput) {
    el.detailDecisionNoteInput.value = "";
  }
}

function setDecisionDraftAction(action) {
  state.decisionDraftAction = String(action || "").trim();
  state.decisionDraftReasons = [];
  renderScoutDecisionComposer();
}

function toggleDecisionReason(tag) {
  const key = String(tag || "").trim();
  if (!key) return;
  if (state.decisionDraftReasons.includes(key)) {
    state.decisionDraftReasons = state.decisionDraftReasons.filter((item) => item !== key);
  } else {
    state.decisionDraftReasons = [...state.decisionDraftReasons, key];
  }
  renderScoutDecisionComposer();
}

function buildScoutDecisionSummary(decision) {
  if (!decision || typeof decision !== "object") {
    return {
      available: false,
      tone: "neutral",
      actionLabel: "No decision logged",
      meta: "No scout decision saved yet.",
      summary: "Save a scout decision to capture why this player should move forward or drop out.",
      note: "",
    };
  }
  const action = String(decision.action || "").trim();
  const reasons = Array.isArray(decision.reason_tags) ? decision.reason_tags.map(humanizeScoutDecisionTag) : [];
  const source = safeText(decision.source_surface || "detail");
  const note = String(decision.note || "").trim();
  return {
    available: true,
    tone: scoutDecisionTone(action),
    actionLabel: humanizeScoutDecisionAction(action),
    meta: `${formatDecisionTimestamp(decision.created_at_utc)} | ${source.replace(/_/g, " ")}`,
    summary: reasons.length
      ? `Reasons: ${reasons.join(" | ")}`
      : "No explicit reason tags were saved for this scout decision.",
    note,
  };
}

function renderScoutDecisionSummary(decision) {
  if (
    !el.detailLatestDecision ||
    !el.detailLatestDecisionPill ||
    !el.detailLatestDecisionMeta ||
    !el.detailLatestDecisionSummary ||
    !el.detailLatestDecisionNote
  ) {
    return;
  }
  const summary = buildScoutDecisionSummary(decision);
  el.detailLatestDecision.hidden = !summary.available;
  el.detailLatestDecisionPill.className = `decision-pill decision-pill--${summary.tone}`;
  el.detailLatestDecisionPill.textContent = summary.actionLabel;
  el.detailLatestDecisionMeta.textContent = summary.meta;
  el.detailLatestDecisionSummary.textContent = summary.summary;
  el.detailLatestDecisionNote.hidden = !summary.note;
  el.detailLatestDecisionNote.textContent = summary.note ? `Note: ${summary.note}` : "";
}

function renderScoutDecisionComposer(message = "") {
  if (!el.detailDecisionReasons || !el.detailDecisionMeta) return;
  const action = state.decisionDraftAction;
  const buttons = Array.isArray(el.detailDecisionActionButtons) ? el.detailDecisionActionButtons : [];
  buttons.forEach((btn) => {
    const selected = String(btn?.dataset?.scoutAction || "") === action;
    btn.classList.toggle("is-selected", selected);
  });
  if (!action) {
    el.detailDecisionReasons.innerHTML = '<span class="metrics-meta">Choose an action to see relevant reason tags.</span>';
    el.detailDecisionMeta.textContent = message || "Decision log is local-first and analytics-only in v1.";
    if (el.detailDecisionSaveBtn) el.detailDecisionSaveBtn.disabled = true;
    return;
  }
  const options = decisionReasonOptions(action);
  el.detailDecisionReasons.innerHTML = options
    .map((option) => {
      const selected = state.decisionDraftReasons.includes(option.key);
      return `
        <button
          type="button"
          class="decision-reason-chip${selected ? " is-selected" : ""}"
          data-decision-reason="${escapeHtml(option.key)}"
        >${escapeHtml(option.label)}</button>
      `;
    })
    .join("");
  const requirement = actionRequiresReason(action)
    ? `Reason tag required for ${humanizeScoutDecisionAction(action).toLowerCase()}.`
    : "Reason tags optional for this action.";
  el.detailDecisionMeta.textContent = message || requirement;
  if (el.detailDecisionSaveBtn) el.detailDecisionSaveBtn.disabled = !state.selectedRow;
}

async function saveScoutDecision() {
  if (!state.selectedRow) {
    renderScoutDecisionComposer("Select a player first.");
    return;
  }
  const action = String(state.decisionDraftAction || "").trim();
  if (!action) {
    renderScoutDecisionComposer("Choose a scout decision action first.");
    return;
  }
  if (actionRequiresReason(action) && !state.decisionDraftReasons.length) {
    renderScoutDecisionComposer(`Select at least one reason tag for ${humanizeScoutDecisionAction(action).toLowerCase()}.`);
    return;
  }
  const note = (el.detailDecisionNoteInput?.value || "").trim();
  state.decisionDraftNote = note;
  if (el.detailDecisionSaveBtn) {
    el.detailDecisionSaveBtn.disabled = true;
    el.detailDecisionSaveBtn.textContent = "Saving...";
  }
  renderScoutDecisionComposer("Saving scout decision...");
  try {
    const payload = await requestJson(isTeamAuthenticated() ? "/team/decisions" : "/market-value/decisions", {
      method: "POST",
      body: {
        player_id: state.selectedRow.player_id,
        split: state.split,
        season: state.selectedRow.season || null,
        action,
        reason_tags: state.decisionDraftReasons,
        note,
        actor: isTeamAuthenticated() ? safeText(state.teamUser?.full_name || state.teamUser?.email || "team_scout") : "local",
        source_surface: currentScoutDecisionSource(state.selectedRow),
        ranking_context: currentDecisionRankingContext(state.selectedRow),
      },
    });
    const latestDecision = payload?.latest_decision || payload?.decision || null;
    state.selectedLatestDecision = latestDecision;
    if (state.selectedProfile && typeof state.selectedProfile === "object") {
      state.selectedProfile.latest_decision = latestDecision;
    }
    if (state.selectedReport && typeof state.selectedReport === "object") {
      state.selectedReport.latest_decision = latestDecision;
    }
    renderScoutDecisionSummary(latestDecision);
    resetDecisionDraft();
    renderScoutDecisionComposer(
      `${humanizeScoutDecisionAction(action)} saved${isTeamAuthenticated() ? " to the shared workspace" : ""}.`
    );
    if (action !== "pass") {
      await refreshWatchlist();
    }
    if (isTeamAuthenticated()) {
      await Promise.allSettled([refreshTeamActivity(), loadTeamCollaborationForSelectedRow()]);
    }
  } catch (err) {
    renderScoutDecisionComposer(err instanceof Error ? err.message : String(err));
  } finally {
    if (el.detailDecisionSaveBtn) {
      el.detailDecisionSaveBtn.disabled = !state.selectedRow || !state.decisionDraftAction;
      el.detailDecisionSaveBtn.textContent = "Save Decision";
    }
  }
}

function focusScoutDecisionComposer() {
  setDetailTab("risk");
  window.setTimeout(() => {
    el.detailDecisionActionButtons?.[0]?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, 40);
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

function leagueAdjustmentMeta(row, report = null) {
  const guardrails = report?.valuation_guardrails || null;
  const bucket = String(
    guardrails?.league_adjustment_bucket || row?.league_adjustment_bucket || ""
  )
    .trim()
    .toLowerCase();
  const reason = safeText(guardrails?.league_adjustment_reason || row?.league_adjustment_reason || "");
  const alpha = firstFiniteNumber(guardrails?.league_adjustment_alpha, row?.league_adjustment_alpha);
  const holdoutR2 = firstFiniteNumber(guardrails?.league_holdout_r2, row?.league_holdout_r2);
  const holdoutWmape = firstFiniteNumber(guardrails?.league_holdout_wmape, row?.league_holdout_wmape);
  const holdoutCoverage = firstFiniteNumber(
    guardrails?.league_holdout_interval_coverage,
    row?.league_holdout_interval_coverage
  );
  let tone = "neutral";
  let label = "Standard pricing";
  if (bucket === "weak") {
    tone = "warn";
    label = "Weak-league pricing adj.";
  } else if (bucket === "failed") {
    tone = "bad";
    label = "Failed-league pricing adj.";
  } else if (bucket === "severe_failed") {
    tone = "bad";
    label = "Heavy pricing adjustment";
  } else if (bucket === "unknown") {
    tone = "warn";
    label = "Unknown-league pricing";
  }
  return {
    bucket,
    reason,
    alpha,
    holdoutR2,
    holdoutWmape,
    holdoutCoverage,
    tone,
    label,
    needsWarning: bucket === "weak" || bucket === "failed" || bucket === "severe_failed",
    isFailed: bucket === "failed" || bucket === "severe_failed",
  };
}

function deriveGapValues(row, report = null) {
  const guardrails = report?.valuation_guardrails || null;
  const market = firstFiniteNumber(guardrails?.market_value_eur, row.market_value_eur);
  const adjustedFair = firstFiniteNumber(
    guardrails?.league_adjusted_fair_value_eur,
    guardrails?.fair_value_eur,
    row.league_adjusted_fair_value_eur,
    row.fair_value_eur,
    row.expected_value_eur
  );
  const rawFair = firstFiniteNumber(
    guardrails?.raw_fair_value_eur,
    row.raw_fair_value_eur,
    row.expected_value_raw_eur,
    adjustedFair
  );
  const impliedRaw = Number.isFinite(market) && Number.isFinite(rawFair) ? rawFair - market : NaN;
  const raw = firstFiniteNumber(guardrails?.value_gap_raw_eur, row.value_gap_raw_eur, row.value_gap_eur, impliedRaw);
  const conservative = firstFiniteNumber(
    guardrails?.value_gap_conservative_eur,
    row.value_gap_conservative_eur,
    raw
  );
  const adjustedGap = firstFiniteNumber(
    guardrails?.league_adjusted_gap_eur,
    row.league_adjusted_gap_eur,
    Number.isFinite(market) && Number.isFinite(adjustedFair) ? adjustedFair - market : NaN
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
  const adjustment = leagueAdjustmentMeta(row, report);
  return {
    raw,
    conservative,
    capped,
    capThreshold,
    capApplied,
    rawFair,
    adjustedFair,
    adjustedGap,
    adjustmentBucket: adjustment.bucket,
    adjustmentReason: adjustment.reason,
    adjustmentAlpha: adjustment.alpha,
  };
}

function withComputedConservativeGap(row) {
  const gaps = deriveGapValues(row);
  return {
    ...row,
    raw_fair_value_eur: Number.isFinite(gaps.rawFair) ? gaps.rawFair : row.raw_fair_value_eur,
    league_adjusted_fair_value_eur: Number.isFinite(gaps.adjustedFair)
      ? gaps.adjustedFair
      : row.league_adjusted_fair_value_eur,
    league_adjusted_gap_eur: Number.isFinite(gaps.adjustedGap) ? gaps.adjustedGap : row.league_adjusted_gap_eur,
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
  if (role === "valuation" && manifest.valuation_champion) return manifest.valuation_champion;
  if (role === "future_shortlist" && manifest.future_shortlist_champion) return manifest.future_shortlist_champion;
  const legacyRole = String(manifest.legacy_default_role || "valuation").trim();
  if (legacyRole !== role || !manifest.artifacts) return null;
  return {
    role,
    label: manifest.label || role,
    generated_at_utc: manifest.generated_at_utc || null,
    artifacts: manifest.artifacts || {},
    config: manifest.config || {},
    summary: manifest.summary || {},
  };
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

function operatorLaneEntry(role) {
  const lane = state.operatorHealth?.active_lanes?.[role] || state.activeArtifacts?.[role] || {};
  return {
    role,
    laneState: safeText(lane.lane_state || lane.laneState || (role === "future_shortlist" ? "live" : "stable")),
    promotionState: safeText(lane.promotion_state || lane.promotionState || "advisory_only"),
    promotionReasons: Array.isArray(lane.promotion_reasons || lane.promotionReasons)
      ? lane.promotion_reasons || lane.promotionReasons
      : [],
    label: safeText(lane.artifact_label || lane.label || role),
    testSeason: safeText(lane.artifact_test_season || lane.test_season || lane.testSeason),
  };
}

function laneStateLabel(stateValue) {
  const lane = String(stateValue || "").trim().toLowerCase();
  if (lane === "stable") return "Stable";
  if (lane === "live") return "Live";
  if (lane === "limited") return "Limited";
  return "Unknown";
}

function boardLaneSummary() {
  const champion = activeChampionRoutingSummary();
  const valuationLane = operatorLaneEntry("valuation");
  const futureLane = operatorLaneEntry("future_shortlist");
  if (isSystemFitMode()) {
    const laneIsFuture = state.systemFitActiveLane === "future_shortlist";
    const runtimeLane = laneIsFuture ? futureLane : valuationLane;
    const systemTemplate = currentSystemFitTemplate();
    const label = `${systemFitLaneLabel()} | ${safeText(systemTemplate?.label || state.systemFitTemplate)} | ${laneStateLabel(runtimeLane.laneState)} ${laneIsFuture ? "advisory lane" : "valuation lane"}`;
    const copy = laneIsFuture
      ? "System-fit ranking is using the live future_shortlist lane. Treat it as advisory, growth-oriented role fit."
      : "System-fit ranking is using the stable valuation lane. Treat it as the pricing-safe current-level view."
    return { label, copy, runtime: `${champion.predictionBaseRole} -> ${champion.shortlistOverlayRole}` };
  }
  if (state.mode === "predictions") {
    const label = `Current Level / Pricing | ${laneStateLabel(valuationLane.laneState)} valuation lane`;
    const copy =
      valuationLane.promotionState === "promotable"
        ? "Use this lane for benchmarked pricing and current-level comparisons."
        : "Use this lane for pricing guidance, but the current valuation artifact is still advisory-only.";
    return { label, copy };
  }
  const label = `Future Potential / Advisory | ${laneStateLabel(futureLane.laneState)} shortlist overlay on ${laneStateLabel(valuationLane.laneState).toLowerCase()} valuation base`;
  const copy =
    futureLane.laneState === "live"
      ? "Ordering is leaning on the live future_shortlist overlay. Treat it as advisory upside while the season is still in progress."
      : "Ordering is leaning on the stable valuation base because no live future overlay is active.";
  return { label, copy, runtime: `${champion.predictionBaseRole} -> ${champion.shortlistOverlayRole}` };
}

function laneAwareDecisionNote(row, baseText) {
  const freshness = buildFreshnessState(row);
  if (freshness.status === "live") {
    return `${baseText} Live overlay guidance while the season is still in progress.`;
  }
  if (freshness.status === "limited") {
    return `${baseText} Freshness is limited, so treat this as an advisory call.`;
  }
  return baseText;
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
    system_fit_score: row.system_fit_score ?? null,
    system_fit_confidence: row.system_fit_confidence ?? null,
    slot_key: row.slot_key ?? null,
    slot_label: row.slot_label ?? null,
    role_template_key: row.role_template_key ?? null,
    budget_status: row.budget_status ?? null,
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

function activeTeamWorkspaceId() {
  return String(state.teamActiveWorkspace?.workspace_id || "").trim();
}

function isTeamAuthenticated() {
  return Boolean(state.teamEnabled && state.teamAuthenticated && activeTeamWorkspaceId());
}

function requestHeaders(baseHeaders = {}) {
  const headers = { ...baseHeaders };
  if (isTeamAuthenticated()) {
    headers["X-ScoutML-Workspace"] = activeTeamWorkspaceId();
  }
  return headers;
}

async function getJson(path, params = {}) {
  const response = await fetch(buildUrl(path, params), {
    credentials: "include",
    headers: requestHeaders(),
  });
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
  const options = {
    method,
    credentials: "include",
    headers: requestHeaders(),
  };
  if (body !== null && body !== undefined) {
    options.headers = requestHeaders({ "Content-Type": "application/json" });
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

async function getBlob(path, params = {}) {
  const response = await fetch(buildUrl(path, params), {
    credentials: "include",
    headers: requestHeaders(),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const filename = String(response.headers.get("content-disposition") || "")
    .match(/filename=\"?([^\";]+)\"?/i)?.[1];
  return {
    blob: await response.blob(),
    filename: filename || null,
  };
}

function teamPreferenceRequestParams() {
  if (!isTeamAuthenticated() || !state.teamApplyPreferences) {
    return {};
  }
  const profileId = String(state.teamPreferenceProfile?.preference_profile_id || "").trim();
  return {
    apply_preferences: true,
    preference_profile_id: profileId || null,
  };
}

function parseTagList(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseRolePriorityMap(value) {
  const out = {};
  String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .forEach((item) => {
      const [key, rawWeight] = item.split(":").map((part) => part.trim());
      const weight = Number(rawWeight);
      if (!key || !Number.isFinite(weight)) return;
      out[key.toUpperCase()] = weight;
    });
  return out;
}

function syncSystemFitTemplateOptions() {
  if (!el.systemFitTemplate) return;
  const templates = Array.isArray(state.systemFitTemplates) ? state.systemFitTemplates : [];
  if (!templates.length) {
    el.systemFitTemplate.value = state.systemFitTemplate;
    return;
  }
  el.systemFitTemplate.innerHTML = templates
    .map(
      (template) =>
        `<option value="${escapeHtml(safeText(template.template_key))}">${escapeHtml(safeText(template.label))}</option>`
    )
    .join("");
  const availableKeys = new Set(templates.map((template) => String(template.template_key || "")));
  if (!availableKeys.has(state.systemFitTemplate)) {
    state.systemFitTemplate = String(templates[0].template_key || DEFAULT_SYSTEM_FIT_TEMPLATE);
  }
  el.systemFitTemplate.value = state.systemFitTemplate;
}

async function loadSystemFitTemplates({ force = false } = {}) {
  if (!force && Array.isArray(state.systemFitTemplates) && state.systemFitTemplates.length) {
    syncSystemFitTemplateOptions();
    return state.systemFitTemplates;
  }
  const payload = await getJson("/market-value/system-fit/templates");
  state.systemFitTemplates = Array.isArray(payload.templates) ? payload.templates : [];
  if (!state.systemFitTemplate && payload.default_template_key) {
    state.systemFitTemplate = String(payload.default_template_key);
  }
  syncSystemFitTemplateOptions();
  return state.systemFitTemplates;
}

function currentSystemFitRows() {
  return Array.isArray(currentSystemFitSlot()?.items) ? currentSystemFitSlot().items : [];
}

function applyUiBootstrapPayload(payload = null) {
  const seasons = Array.isArray(payload?.seasons)
    ? payload.seasons.map((value) => String(value || "").trim()).filter(Boolean)
    : [];
  const leagues = Array.isArray(payload?.leagues)
    ? payload.leagues.map((value) => String(value || "").trim()).filter(Boolean)
    : [];
  state.coverageRows = Array.isArray(payload?.coverage_rows) ? payload.coverage_rows : [];
  renderCoverageTable();
  renderOverviewReadiness();

  const prevSeason = state.season;
  const prevLeague = state.league;
  updateSelectOptions(el.season, seasons, prevSeason);
  updateSelectOptions(el.league, leagues, prevLeague);
  if (prevSeason && !seasons.includes(prevSeason)) state.season = "";
  if (prevLeague && !leagues.includes(prevLeague)) state.league = "";
  if (!state.season) el.season.value = "";
  if (!state.league) el.league.value = "";
}

function renderSystemFitSlots() {
  if (!el.systemFitSlotWrap || !el.systemFitSlotBar || !el.systemFitSlotMeta) return;
  const active = isSystemFitMode();
  el.systemFitSlotWrap.hidden = !active;
  if (!active) {
    el.systemFitSlotMeta.textContent = "Select a slot to inspect backend-ranked candidates for this system.";
    el.systemFitSlotBar.innerHTML = "";
    return;
  }
  const slots = Array.isArray(state.systemFitSlots) ? state.systemFitSlots : [];
  if (!slots.length) {
    el.systemFitSlotMeta.textContent = "Run a system-fit query to load slot candidates.";
    el.systemFitSlotBar.innerHTML = '<span class="fit-driver-pill">No slots loaded yet.</span>';
    return;
  }
  if (!slots.some((slot) => String(slot.slot_key || "") === String(state.systemFitSelectedSlot || ""))) {
    state.systemFitSelectedSlot = String(slots[0].slot_key || "");
  }
  const selectedSlot = currentSystemFitSlot();
  const laneLabel = state.systemFitLanePosture?.label || systemFitLaneLabel();
  const selectedSummary = selectedSlot
    ? `${safeText(selectedSlot.slot_label)} | ${safeText(selectedSlot.role_template_label)} | ${formatInt(
        selectedSlot.result_count
      )} candidates`
    : "Select a slot to inspect candidates.";
  el.systemFitSlotMeta.textContent = `${laneLabel}. ${selectedSummary}.`;
  el.systemFitSlotBar.innerHTML = slots
    .map((slot) => {
      const selected = String(slot.slot_key || "") === String(state.systemFitSelectedSlot || "");
      return `
        <button
          type="button"
          class="system-fit-slot-chip${selected ? " is-active" : ""}"
          data-slot-key="${escapeHtml(safeText(slot.slot_key))}"
        >
          <span>${escapeHtml(safeText(slot.slot_label))}</span>
          <strong>${escapeHtml(safeText(slot.role_template_label))}</strong>
        </button>
      `;
    })
    .join("");
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
    el.tbody.innerHTML = `<tr><td colspan="${RESULTS_TABLE_COLSPAN}">Loading players for the current recruitment brief...</td></tr>`;
  }
  renderTopPicks();
  renderBoardHighlights();
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
  if (el.systemFitTemplate) el.systemFitTemplate.value = state.systemFitTemplate;
  if (el.systemFitActiveLane) el.systemFitActiveLane.value = state.systemFitActiveLane;
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
  if (el.playstyle) el.playstyle.value = state.playstyle;
  if (el.roleLens) el.roleLens.value = state.roleLens;
  if (el.teamPrefApply) el.teamPrefApply.checked = Boolean(state.teamApplyPreferences);
  el.funnelSplit.value = state.split;
}

function readWorkbenchControlsToState() {
  state.apiBase = el.apiBase.value.trim() || DEFAULT_API;
  state.mode = el.mode.value;
  state.split = el.split.value;
  if (el.systemFitTemplate) state.systemFitTemplate = el.systemFitTemplate.value || DEFAULT_SYSTEM_FIT_TEMPLATE;
  if (el.systemFitActiveLane) state.systemFitActiveLane = el.systemFitActiveLane.value || DEFAULT_SYSTEM_FIT_LANE;
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
  if (el.playstyle) state.playstyle = el.playstyle.value;
  if (el.roleLens) state.roleLens = el.roleLens.value;
  if (el.teamPrefApply) state.teamApplyPreferences = el.teamPrefApply.checked;
}

function renderTrustCard() {
  const health = state.health || {};
  const metrics = state.metrics || {};
  const artifacts = health.artifacts || {};
  const active = state.activeArtifacts || {};
  const champion = activeChampionRoutingSummary();
  const valuationLane = operatorLaneEntry("valuation");
  const futureLane = operatorLaneEntry("future_shortlist");
  const promotion = state.operatorHealth?.promotion_gate || {};
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
    el.championValuation.textContent = `${champion.valuationLabel} | ${laneStateLabel(valuationLane.laneState)} | ${valuationLane.promotionState}`;
  }
  if (el.championShortlist) {
    el.championShortlist.textContent = `${champion.futureShortlistLabel} | ${laneStateLabel(futureLane.laneState)} | ${futureLane.promotionState}`;
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
  if (promotion && promotion.promotable === false && Array.isArray(promotion.failed_checks) && promotion.failed_checks.length) {
    el.trustNote.textContent += ` Promotion gate remains advisory-only: ${promotion.failed_checks[0]}`;
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
        if (
          state.coverageRows.every(
            (row) =>
              row &&
              typeof row === "object" &&
              Object.prototype.hasOwnProperty.call(row, "rows") &&
              Object.prototype.hasOwnProperty.call(row, "undervalued_share")
          )
        ) {
          return [...state.coverageRows];
        }
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

function renderBoardLaneStatus() {
  if (!el.boardLaneStatus) return;
  if (!state.connected) {
    el.boardLaneStatus.textContent = "Connect API to load active artifact lane context.";
    return;
  }
  if (state.health?.status !== "ok") {
    el.boardLaneStatus.textContent = "Active lane unavailable while backend artifacts are missing.";
    return;
  }
  const summary = boardLaneSummary();
  el.boardLaneStatus.textContent = `${summary.label}. ${summary.copy}`;
}

function renderOperatorDashboard() {
  if (!el.operatorIngestionStatus) return;
  const payload = state.operatorHealth || null;
  if (!payload) {
    el.operatorIngestionStatus.textContent = "-";
    el.operatorIngestionCopy.textContent = "Connect API to load ingestion health.";
    el.operatorValuationLane.textContent = "-";
    el.operatorValuationCopy.textContent = "Connect API to load valuation-lane posture.";
    el.operatorFutureLane.textContent = "-";
    el.operatorFutureCopy.textContent = "Connect API to load future-shortlist posture.";
    el.operatorPromotionStatus.textContent = "-";
    el.operatorPromotionCopy.textContent = "Connect API to load promotability.";
    el.operatorStaleStatus.textContent = "-";
    el.operatorStaleCopy.textContent = "Connect API to load provider freshness risk.";
    el.operatorLiveStatus.textContent = "-";
    el.operatorLiveCopy.textContent = "Connect API to load live current-season footprint.";
    el.operatorBlockedList.innerHTML = '<li class="risk-item risk-item--none">No operator health loaded.</li>';
    el.operatorLaneList.innerHTML = '<li class="risk-item risk-item--none">No operator health loaded.</li>';
    return;
  }

  const ingestion = payload.ingestion_health || {};
  const ingestionSummary = ingestion.summary || {};
  const statusCounts = ingestionSummary.status_counts || {};
  const valuationLane = operatorLaneEntry("valuation");
  const futureLane = operatorLaneEntry("future_shortlist");
  const promotion = payload.promotion_gate || {};
  const stale = payload.stale_provider_snapshots || {};
  const live = payload.live_partial_footprint || {};
  const holdouts = payload.holdout_coverage || {};

  el.operatorIngestionStatus.textContent = `${formatInt(statusCounts.healthy)} healthy | ${formatInt(
    statusCounts.watch
  )} watch | ${formatInt(statusCounts.blocked)} blocked`;
  el.operatorIngestionCopy.textContent = `${formatInt(ingestionSummary.total)} configured league-season rows in the current ingestion report.`;

  el.operatorValuationLane.textContent = `${laneStateLabel(valuationLane.laneState)} | ${valuationLane.promotionState}`;
  el.operatorValuationCopy.textContent = `${
    valuationLane.promotionState === "promotable" ? "Stable valuation lane is promotable." : "Stable valuation lane is still advisory-only."
  } ${valuationLane.label}.`;

  el.operatorFutureLane.textContent = `${laneStateLabel(futureLane.laneState)} | ${futureLane.promotionState}`;
  el.operatorFutureCopy.textContent = "future_shortlist stays live and advisory by design for current-season scouting.";

  el.operatorPromotionStatus.textContent = promotion.promotable ? "Promotable" : "Advisory only";
  el.operatorPromotionCopy.textContent = promotion.promotable
    ? "Valuation artifact meets the current soft promotion gate."
    : `${formatInt((promotion.failed_checks || []).length)} promotion checks still failing.`;

  el.operatorStaleStatus.textContent = `${formatInt(stale.stale_count)} stale`;
  el.operatorStaleCopy.textContent = stale.latest_snapshot_date
    ? `Latest provider snapshot ${safeText(stale.latest_snapshot_date)}. Threshold: ${formatInt(stale.threshold_days)} days.`
    : "Provider snapshot recency is still incomplete.";

  el.operatorLiveStatus.textContent = `${formatInt(live.live_rows)} live rows`;
  el.operatorLiveCopy.textContent = live.live_test_season
    ? `${formatPct(Number(live.live_share) || 0)} of current test rows are in the live ${safeText(live.live_test_season)} overlay.`
    : "No live current-season footprint is active right now.";

  const blockedItems = []
    .concat(
      Array.isArray(ingestion.blocked_items)
        ? ingestion.blocked_items.map((row) => ({
            tone: "high",
            label: `${safeText(row.league_name)} | ${safeText(row.season)}`,
            message: `${safeText(row.status)} | ${(row.status_reasons || []).join(", ") || "blocked"}.`,
          }))
        : []
    )
    .concat(
      Array.isArray(ingestion.watch_items)
        ? ingestion.watch_items.slice(0, 4).map((row) => ({
            tone: "medium",
            label: `${safeText(row.league_name)} | ${safeText(row.season)}`,
            message: `${safeText(row.status)} | ${(row.status_reasons || []).join(", ") || "watch"}.`,
          }))
        : []
    );
  el.operatorBlockedList.innerHTML = buildNarrativeListMarkup(
    blockedItems,
    "No blocked or watch league-seasons in the ingestion report."
  );

  const laneNotes = [
    {
      tone: valuationLane.promotionState === "promotable" ? "low" : "medium",
      label: "Valuation lane",
      message: `${valuationLane.label} | ${laneStateLabel(valuationLane.laneState)} | ${valuationLane.promotionState}. ${
        (valuationLane.promotionReasons || [])[0] || "Valuation lane promotion posture loaded."
      }`,
    },
    {
      tone: futureLane.laneState === "live" ? "medium" : "low",
      label: "future_shortlist lane",
      message: `${futureLane.label} | ${laneStateLabel(futureLane.laneState)} | ${futureLane.promotionState}. ${
        (futureLane.promotionReasons || [])[0] || "Live shortlist posture loaded."
      }`,
    },
    {
      tone: Number(holdouts.successful_count) >= 6 ? "low" : "medium",
      label: "Holdout coverage",
      message: `${formatInt(holdouts.successful_count)} successful holdouts out of ${formatInt(
        holdouts.requested_count
      )} requested.`,
    },
  ].concat(
    (promotion.failed_checks || []).slice(0, 3).map((message) => ({
      tone: "high",
      label: "Promotion check",
      message,
    }))
  );
  el.operatorLaneList.innerHTML = buildNarrativeListMarkup(
    laneNotes,
    "Lane and holdout notes will appear here once operator health loads."
  );
}

function renderOverviewReadiness() {
  if (!el.overviewPosturePill) return;
  renderOperatorDashboard();

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
  const promotionGate = state.operatorHealth?.promotion_gate || {};
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
  if (promotionGate.promotable === false) {
    pricingTone = pricingTone === "price" ? "price" : "watch";
    pricingStatus = "Advisory valuation lane";
    pricingCopy = "Pipeline completed, but the valuation promotion gate is still advisory-only. Keep pricing posture conservative.";
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
  const futureLane = operatorLaneEntry("future_shortlist");
  if (futureLane.laneState === "live") {
    rankingTone = rankingTone === "pursue" ? "watch" : rankingTone;
    rankingStatus = "Live overlay active";
    rankingCopy = "Shortlist routing is fresh and useful, but the current season is still live. Treat ordering as advisory rather than final.";
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
      const decisionAction = String(row.decision_action || "").trim();
      const decisionLabel = decisionAction ? humanizeScoutDecisionAction(decisionAction) : "";
      const decisionReasons = Array.isArray(row.decision_reason_tags)
        ? row.decision_reason_tags.map(humanizeScoutDecisionTag).slice(0, 2).join(" | ")
        : "";
      return `
        <tr>
          <td>${safeText(row.name)}</td>
          <td>${safeText(row.league)}</td>
          <td class="num">${formatCurrency(row.value_gap_capped_eur)}</td>
          <td class="num">${formatNumber(row.undervaluation_confidence)}</td>
          <td>
            <div class="watchlist-tag-cell">
              <span>${safeText(row.tag)}</span>
              ${
                decisionLabel
                  ? `<small>${escapeHtml(`Decision: ${decisionLabel}${decisionReasons ? ` | ${decisionReasons}` : ""}`)}</small>`
                  : ""
              }
            </div>
          </td>
          <td>${safeText(row.last_decision_at_utc || row.created_at_utc)}</td>
          <td>
            <div class="watchlist-actions">
              <button type="button" class="btn-ghost watchlist-open-decision" data-watch-id="${watchId}" data-player-id="${escapeHtml(
                safeText(row.player_id)
              )}" data-season="${escapeHtml(safeText(row.season))}">Update decision</button>
              <button type="button" class="btn-ghost watchlist-delete" data-watch-id="${watchId}">Remove</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
  el.watchlistMeta.textContent = `${formatInt(rows.length)} shown / ${formatInt(state.watchlistTotal)} saved entries`;
}

function focusTagForIndex(index) {
  if (index === 0) return { label: "Top pick", tone: "top" };
  if (index > 0 && index < 3) return { label: "High upside", tone: "upside" };
  return null;
}

function buildFocusTagMarkup(index) {
  const tag = focusTagForIndex(index);
  if (!tag) return "";
  return `<span class="row-tag row-tag--${escapeHtml(tag.tone)}">${escapeHtml(tag.label)}</span>`;
}

function deriveTopPicks(rows) {
  if (!Array.isArray(rows) || !rows.length) return [];
  if (isSystemFitMode()) return rows.slice(0, 5);

  const targetCount = 5;
  const picks = [];
  const leagueCounts = new Map();
  let failedLeagueCount = 0;
  const seenKeys = new Set();

  const tryAddRow = (row, { enforceLeagueCap = true, enforceFailedCap = true } = {}) => {
    const key = rowKey(row);
    if (seenKeys.has(key)) return false;
    const leagueKey = getLeague(row).trim().toLowerCase() || "__unknown__";
    const adjustment = leagueAdjustmentMeta(row);
    if (enforceLeagueCap && (leagueCounts.get(leagueKey) || 0) >= 2) return false;
    if (enforceFailedCap && adjustment.isFailed && failedLeagueCount >= 1) return false;
    picks.push(row);
    seenKeys.add(key);
    leagueCounts.set(leagueKey, (leagueCounts.get(leagueKey) || 0) + 1);
    if (adjustment.isFailed) failedLeagueCount += 1;
    return true;
  };

  rows.forEach((row) => {
    if (picks.length < targetCount) tryAddRow(row, { enforceLeagueCap: true, enforceFailedCap: true });
  });
  rows.forEach((row) => {
    if (picks.length < targetCount) tryAddRow(row, { enforceLeagueCap: true, enforceFailedCap: false });
  });
  rows.forEach((row) => {
    if (picks.length < targetCount) tryAddRow(row, { enforceLeagueCap: false, enforceFailedCap: false });
  });
  return picks;
}

function scrollToBoard() {
  const target = el.boardAnchor || document.getElementById("workbench-entry");
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderTopPicks() {
  if (!el.topPicksGrid || !el.topPicksMeta) return;
  renderBoardLaneStatus();
  if (el.topPicksTitle) {
    if (isSystemFitMode()) {
      const slot = currentSystemFitSlot();
      el.topPicksTitle.textContent = slot ? `${safeText(slot.slot_label)} Candidates` : "System Fit";
    } else {
      el.topPicksTitle.textContent = hasActiveLens() ? `Top Picks for ${activeLensTitleLabel()}` : "Top Picks";
    }
  }

  if (state.initializing || state.loading) {
    el.topPicksMeta.textContent = "Loading players from the current recruitment brief.";
    el.topPicksGrid.innerHTML = Array.from({ length: 3 })
      .map(
        () => `
          <article class="top-pick-card top-pick-card--placeholder">
            <p>Loading players...</p>
          </article>
        `
      )
      .join("");
    return;
  }

  if (state.connected && state.health?.status !== "ok") {
    el.topPicksMeta.textContent =
      "Backend artifacts are not ready. Review Platform Readiness for the operational details.";
    el.topPicksGrid.innerHTML = `
      <article class="top-pick-card top-pick-card--empty">
        <h3>Top picks unavailable</h3>
        <p>ScoutML could not load board artifacts from the current API target, so there is no shortlist to preview yet.</p>
      </article>
    `;
    return;
  }

  if (!state.connected) {
    el.topPicksMeta.textContent = `Connect API to load top picks from ${state.apiBase}.`;
    el.topPicksGrid.innerHTML = `
      <article class="top-pick-card top-pick-card--empty">
        <h3>Connect the API</h3>
        <p>Point the UI at a live backend to populate the recruitment board and preview cards.</p>
      </article>
    `;
    return;
  }

  if (!state.topPicks.length) {
    const emptyMessage =
      isSystemFitMode()
        ? "No candidates match the current system-fit slot. Relax the filters or switch slots."
        : state.mode === "shortlist"
        ? "No players match the current recruitment brief. Relax the brief to surface new targets."
        : "No players match the current valuation filters. Widen the board to inspect more names.";
    el.topPicksMeta.textContent = emptyMessage;
    el.topPicksGrid.innerHTML = `
      <article class="top-pick-card top-pick-card--empty">
        <h3>No top picks right now</h3>
        <p>${escapeHtml(emptyMessage)}</p>
      </article>
    `;
    return;
  }

  el.topPicksMeta.textContent =
    isSystemFitMode()
      ? `Backend-ranked candidates for ${safeText(currentSystemFitSlot()?.slot_label || "the selected slot")} in ${safeText(
          currentSystemFitTemplate()?.label || state.systemFitTemplate
        )}.`
      : state.mode === "shortlist"
      ? `Best starting points from the current Future Potential / Advisory brief${
          hasActiveLens() ? ` under the ${activeLensDisplayLabel()} lens` : ""
        }.`
      : `Current highest-priority names from the Current Level / Pricing board${
          hasActiveLens() ? ` under the ${activeLensDisplayLabel()} lens` : ""
        }.`;

  el.topPicksGrid.innerHTML = state.topPicks
    .map((row, idx) => {
      if (isSystemFitMode()) {
        const systemFit = Number(row.system_fit_score);
        const confidence = Number(row.system_fit_confidence);
        const currentLevel = Number(row.current_level_score);
        const futurePotential = Number(row.future_potential_score);
        const budgetStatus = safeText(row.budget_status || "unbounded").replace(/_/g, " ");
        const reasons = Array.isArray(row.fit_reasons) ? row.fit_reasons : [];
        return `
          <button type="button" class="top-pick-card" data-index="${idx}">
            <div class="top-pick-card__topline">
              ${buildFocusTagMarkup(idx)}
              <span class="fit-chip fit-chip--good">${escapeHtml(
                `System fit ${Number.isFinite(systemFit) ? formatNumber(systemFit) : "-"}`
              )}</span>
            </div>
            <strong class="top-pick-card__name">${escapeHtml(safeText(row.name))}</strong>
            <span class="top-pick-card__meta">${escapeHtml(
              `${safeText(row.club)} | ${safeText(row.league)}`
            )}</span>
            <span class="top-pick-card__meta top-pick-card__meta--secondary">${escapeHtml(
              `${safeText(row.slot_label)} | ${safeText(row.role_template_label)}`
            )}</span>
            <div class="top-pick-card__metrics">
              <div>
                <span>Market</span>
                <strong>${formatCurrency(row.market_value_eur)}</strong>
              </div>
              <div>
                <span>Current</span>
                <strong>${Number.isFinite(currentLevel) ? formatNumber(currentLevel) : "-"}</strong>
              </div>
              <div>
                <span>Future</span>
                <strong>${Number.isFinite(futurePotential) ? formatNumber(futurePotential) : "-"}</strong>
              </div>
            </div>
            <p class="top-pick-card__note">${escapeHtml(
              `Confidence ${Number.isFinite(confidence) ? formatNumber(confidence) : "-"} | ${budgetStatus}`
            )}</p>
            <p class="top-pick-card__fit">${escapeHtml(reasons.slice(0, 2).join(" | ") || "Backend-ranked slot fit.")}</p>
          </button>
        `;
      }
      const decision = summarizeRecruitmentDecision(row, null, {
        source: currentWorkbenchDecisionSource(),
      });
      const laneDecision = laneAwareDecisionNote(row, decision.nextAction);
      const gaps = deriveGapValues(row, null);
      const expected = Number.isFinite(gaps.adjustedFair)
        ? gaps.adjustedFair
        : firstFiniteNumber(row.fair_value_eur, row.expected_value_eur);
      const freshness = buildFreshnessState(row);
      const talentLine = buildTalentCompactLine(row);
      const talentDrivers = buildTalentDriverLine(row);
      const adjustment = leagueAdjustmentMeta(row);
      return `
        <button type="button" class="top-pick-card" data-index="${idx}">
          <div class="top-pick-card__topline">
            ${buildFocusTagMarkup(idx)}
            ${adjustment.needsWarning ? buildBadgeChipMarkup(adjustment.label, adjustment.tone) : ""}
            ${buildStyleFitMarkup(row)}
            ${buildDecisionPillMarkup(decision)}
          </div>
          <strong class="top-pick-card__name">${escapeHtml(safeText(row.name))}</strong>
          <span class="top-pick-card__meta">${escapeHtml(
            `${safeText(row.club)} | ${safeText(row.league)}`
          )}</span>
          <span class="top-pick-card__meta top-pick-card__meta--secondary">${escapeHtml(
            `${safeText(getPosition(row))} | age ${formatNumber(row.age)}`
          )}</span>
          <div class="top-pick-card__metrics">
            <div>
              <span>Market</span>
              <strong>${formatCurrency(row.market_value_eur)}</strong>
            </div>
            <div>
              <span>Adj. fair</span>
              <strong>${formatCurrency(expected)}</strong>
            </div>
            <div>
              <span>Gap</span>
              <strong class="${gaps.capped >= 0 ? "positive" : "negative"}">${formatCurrency(gaps.capped)}</strong>
            </div>
          </div>
          <p class="top-pick-card__note">${escapeHtml(
            adjustment.needsWarning && adjustment.reason ? `${laneDecision} ${adjustment.reason}` : laneDecision
          )}</p>
          <p class="top-pick-card__fit">${escapeHtml(talentLine)}</p>
          ${
            hasActiveLens() && (Number.isFinite(row._styleScore) || Number.isFinite(row._roleScore))
              ? `<p class="top-pick-card__drivers${
                  row._styleFitDataQuality === "limited" ? " top-pick-card__drivers--muted" : ""
                }">${escapeHtml(buildLensFitLine(row))}</p>`
              : ""
          }
          ${
            talentDrivers
              ? `<p class="top-pick-card__drivers">${escapeHtml(talentDrivers)}</p>`
              : ""
          }
          ${
            hasActiveLens() && row._styleFitReasonLine
              ? `<p class="top-pick-card__drivers${
                  row._styleFitDataQuality === "limited" ? " top-pick-card__drivers--muted" : ""
                }">${escapeHtml(row._styleFitReasonLine)}</p>`
              : ""
          }
          <p class="top-pick-card__freshness${
            freshness.status === "limited" ? " top-pick-card__drivers--muted" : ""
          }">${escapeHtml(freshness.compactLine)}</p>
        </button>
      `;
    })
    .join("");
}

function renderBoardHighlights() {
  if (
    !el.boardHighlightTopPick ||
    !el.boardHighlightTopPickNote ||
    !el.boardHighlightGap ||
    !el.boardHighlightGapNote ||
    !el.boardHighlightMix ||
    !el.boardHighlightMixNote
  ) {
    return;
  }

  if (state.initializing || state.loading) {
    el.boardHighlightTopPick.textContent = "-";
    el.boardHighlightTopPickNote.textContent = "Loading board...";
    el.boardHighlightGap.textContent = "-";
    el.boardHighlightGapNote.textContent = "Waiting for current page.";
    el.boardHighlightMix.textContent = "-";
    el.boardHighlightMixNote.textContent = "Waiting for current page.";
    return;
  }

  if (state.connected && state.health?.status !== "ok") {
    el.boardHighlightTopPick.textContent = "Unavailable";
    el.boardHighlightTopPickNote.textContent = "Backend artifacts are not ready.";
    el.boardHighlightGap.textContent = "-";
    el.boardHighlightGapNote.textContent = "Review Platform Readiness.";
    el.boardHighlightMix.textContent = "-";
    el.boardHighlightMixNote.textContent = "Board diagnostics unavailable.";
    return;
  }

  if (!state.connected) {
    el.boardHighlightTopPick.textContent = "Connect API";
    el.boardHighlightTopPickNote.textContent = "Point the UI at a live backend to load the board.";
    el.boardHighlightGap.textContent = "-";
    el.boardHighlightGapNote.textContent = "No current board.";
    el.boardHighlightMix.textContent = "-";
    el.boardHighlightMixNote.textContent = "No current board.";
    return;
  }

  if (!state.rows.length) {
    el.boardHighlightTopPick.textContent = "No current board";
    el.boardHighlightTopPickNote.textContent = "Widen the brief to surface candidates.";
    el.boardHighlightGap.textContent = "-";
    el.boardHighlightGapNote.textContent = "No current rows.";
    el.boardHighlightMix.textContent = "-";
    el.boardHighlightMixNote.textContent = "No current rows.";
    return;
  }

  if (isSystemFitMode()) {
    const topRow = state.rows[0];
    const bestSystemFit = state.rows.reduce((best, row) => {
      const score = Number(row.system_fit_score);
      return !Number.isFinite(best) || score > best ? score : best;
    }, NaN);
    let withinBudgetCount = 0;
    let stretchCount = 0;
    state.rows.forEach((row) => {
      const status = String(row.budget_status || "").toLowerCase();
      if (status === "within_budget") withinBudgetCount += 1;
      if (status === "stretch") stretchCount += 1;
    });
    el.boardHighlightTopPick.textContent = safeText(topRow.name);
    el.boardHighlightTopPickNote.textContent = `${safeText(topRow.slot_label)} | ${safeText(
      topRow.role_template_label
    )} | ${safeText(topRow.club)}`;
    el.boardHighlightGap.textContent = Number.isFinite(bestSystemFit) ? formatNumber(bestSystemFit) : "-";
    el.boardHighlightGapNote.textContent = "Best backend system-fit score on the current slot page.";
    el.boardHighlightMix.textContent = `${formatInt(withinBudgetCount)} budget / ${formatInt(stretchCount)} stretch`;
    el.boardHighlightMixNote.textContent = "Budget posture mix across the current slot page.";
    return;
  }

  const topRow = state.rows[0];
  const topDecision = summarizeRecruitmentDecision(topRow, null, {
    source: currentWorkbenchDecisionSource(),
  });
  const bestGap = state.rows.reduce((best, row) => {
    const gap = conservativeGapForRanking(row);
    return !Number.isFinite(best) || gap > best ? gap : best;
  }, NaN);
  let pursueCount = 0;
  let watchCount = 0;
  state.rows.forEach((row) => {
    const decision = summarizeRecruitmentDecision(row, null, {
      source: currentWorkbenchDecisionSource(),
    });
    if (decision.label === "Pursue") pursueCount += 1;
    if (decision.label === "Watch") watchCount += 1;
  });

  el.boardHighlightTopPick.textContent = safeText(topRow.name);
  el.boardHighlightTopPickNote.textContent = `${topDecision.label} | ${safeText(getPosition(topRow))} | ${safeText(
    topRow.club
  )}${hasActiveLens() && topRow._styleFitLabel ? ` | ${topRow._styleFitLabel}` : ""}`;
  el.boardHighlightGap.textContent = formatCurrency(bestGap);
  el.boardHighlightGapNote.textContent = "Largest capped conservative gap on the current page.";
  el.boardHighlightMix.textContent = `${formatInt(pursueCount)} pursue / ${formatInt(watchCount)} watch`;
  el.boardHighlightMixNote.textContent = hasActiveLens()
    ? `Current page decision mix under the ${activeLensDisplayLabel()} lens.`
    : "Current page decision mix.";
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
    const payload = await requestJson(isTeamAuthenticated() ? "/team/watchlist" : "/market-value/watchlist", {
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
    await requestJson(isTeamAuthenticated() ? "/team/watchlist/items" : "/market-value/watchlist/items", {
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
    if (isTeamAuthenticated()) {
      void refreshTeamActivity();
    }
  } catch (err) {
    if (el.watchlistMeta) el.watchlistMeta.textContent = err instanceof Error ? err.message : String(err);
  }
}

async function deleteWatchlistItem(watchId) {
  const id = String(watchId || "").trim();
  if (!id) return;
  try {
    await requestJson(
      `${isTeamAuthenticated() ? "/team/watchlist/items" : "/market-value/watchlist/items"}/${encodeURIComponent(id)}`,
      { method: "DELETE" }
    );
    await refreshWatchlist();
    if (isTeamAuthenticated()) {
      void refreshTeamActivity();
    }
  } catch (err) {
    if (el.watchlistMeta) el.watchlistMeta.textContent = err instanceof Error ? err.message : String(err);
  }
}

async function openWatchlistDecision(row) {
  if (!row) return;
  setView("workbench");
  await loadDetailWithReport({ ...row, _decisionSourceSurface: "watchlist" });
  focusScoutDecisionComposer();
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

function applySystemFitSlotRows() {
  const slot = currentSystemFitSlot();
  let rows = Array.isArray(slot?.items) ? slot.items.map((row) => withComputedConservativeGap(row)) : [];
  if (state.search) {
    const q = state.search.toLowerCase();
    rows = rows.filter((row) => {
      const name = String(row.name || "").toLowerCase();
      const club = String(row.club || "").toLowerCase();
      return name.includes(q) || club.includes(q);
    });
  }
  state.rows = rows;
  state.total = Number(slot?.result_count) || rows.length;
  state.count = rows.length;
  state.topPicks = deriveTopPicks(rows);
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
      isSystemFitMode()
        ? "No slot candidates match the current system-fit brief. Relax the filters or switch to another slot."
        : state.mode === "shortlist"
        ? "No recruitment targets match the current brief. Relax the brief or widen the filters to surface more names."
        : "No valuation rows match the current filters. Widen the board to inspect more names.";
    el.tbody.innerHTML = `<tr><td colspan="${RESULTS_TABLE_COLSPAN}">${emptyMessage}</td></tr>`;
    el.resultCount.textContent = "0 rows";
    el.resultRange.textContent = "offset 0";
    renderBoardHighlights();
    return;
  }

  el.tbody.innerHTML = state.rows
    .map((row, idx) => {
      if (isSystemFitMode()) {
        const selected = state.selectedRow && rowKey(state.selectedRow) === rowKey(row) ? " selected-row" : "";
        const currentLevel = Number(row.current_level_score);
        const futurePotential = Number(row.future_potential_score);
        const confidence = Number(row.system_fit_confidence);
        const systemFit = Number(row.system_fit_score);
        const budgetStatus = safeText(row.budget_status || "unbounded").replace(/_/g, " ");
        const roleLabel = safeText(row.role_template_label || row.role_template_key || "-");
        const reasons = Array.isArray(row.fit_reasons) ? row.fit_reasons : [];
        const talentLine = buildTalentCompactLine(row);
        const freshness = buildFreshnessState(row);
        return `
          <tr data-index="${idx}" class="${selected.trim()}">
            <td class="player-cell">
              <div class="player-cell__topline">
                <strong>${safeText(row.name)}</strong>
                ${buildFocusTagMarkup(idx)}
              </div>
              <span class="player-cell__sub">${safeText(row.club)} | ${safeText(row.league)} | ${safeText(row.season)}</span>
              <span class="player-cell__meta">${safeText(row.slot_label)} | ${roleLabel} | ${safeText(
                row.talent_position_family
              )}</span>
              <span class="player-cell__note">${escapeHtml(
                `System fit ${Number.isFinite(systemFit) ? formatNumber(systemFit) : "-"} | ${budgetStatus}`
              )}</span>
              <span class="player-cell__note player-cell__note--style">${escapeHtml(talentLine)}</span>
              ${
                reasons.length
                  ? `<span class="player-cell__note player-cell__note--fit-driver">${escapeHtml(reasons.slice(0, 2).join(" | "))}</span>`
                  : ""
              }
              <span class="player-cell__note player-cell__note--freshness${
                freshness.status === "limited" ? " fit-driver-summary--muted" : ""
              }">${escapeHtml(freshness.compactLine)}</span>
            </td>
            <td>
              <div class="table-status">
                <div class="table-status__chips">
                  <span class="decision-pill decision-pill--watch">System fit</span>
                </div>
                <span class="table-status__note">${escapeHtml(
                  `${safeText(row.slot_label)} | ${roleLabel} | ${budgetStatus}`
                )}</span>
                <span class="table-status__sub">${escapeHtml(
                  reasons.length ? reasons[0] : "Backend-ranked slot fit."
                )}</span>
              </div>
            </td>
            <td class="num">
              <div class="value-stack">
                <strong>${formatCurrency(row.market_value_eur)}</strong>
                <span>Market value</span>
              </div>
            </td>
            <td class="num">
              <div class="value-stack">
                <strong>${Number.isFinite(currentLevel) ? formatNumber(currentLevel) : "-"}</strong>
                <span>Current level</span>
              </div>
            </td>
            <td class="num">
              <div class="value-stack">
                <strong>${Number.isFinite(futurePotential) ? formatNumber(futurePotential) : "-"}</strong>
                <span>Future potential</span>
              </div>
            </td>
            <td class="num">
              <div class="value-stack">
                <strong>${Number.isFinite(confidence) ? formatNumber(confidence) : "-"}</strong>
                <span>${escapeHtml(budgetStatus)}</span>
              </div>
            </td>
          </tr>
        `;
      }
      const gaps = deriveGapValues(row, null);
      const consGap = conservativeGapForRanking(row);
      const expected = Number.isFinite(gaps.adjustedFair)
        ? gaps.adjustedFair
        : firstFiniteNumber(row.fair_value_eur, row.expected_value_eur);
      const selected = state.selectedRow && rowKey(state.selectedRow) === rowKey(row) ? " selected-row" : "";
      const decision = summarizeRecruitmentDecision(row, null, {
        source: currentWorkbenchDecisionSource(),
      });
      const laneDecision = laneAwareDecisionNote(row, decision.nextAction);
      const scoreContext = resolveRowScoreContext(
        row,
        state.mode === "shortlist" || isSystemFitMode() ? state.queryDiagnostics : null,
        currentWorkbenchSourceMode()
      );
      const badges = buildProvenanceBadges(row)
        .map((badge) => buildBadgeChipMarkup(badge.label, badge.tone))
        .join("");
      const freshness = buildFreshnessState(row);
      const talentLine = buildTalentCompactLine(row);
      const talentDrivers = buildTalentDriverLine(row);
      return `
        <tr data-index="${idx}" class="${selected.trim()}">
          <td class="player-cell">
            <div class="player-cell__topline">
              <strong>${safeText(row.name)}</strong>
              ${buildFocusTagMarkup(idx)}
            </div>
            <span class="player-cell__sub">${safeText(row.club)} | ${safeText(row.league)} | ${safeText(row.season)}</span>
            <span class="player-cell__meta">${safeText(getPosition(row))} | age ${formatNumber(row.age)} | ${formatInt(
              getMinutes(row)
            )} minutes</span>
            <div class="player-cell__badges">${badges}</div>
            <span class="player-cell__note">${escapeHtml(scoreContext.scoreLabel)}${
              Number.isFinite(scoreContext.scoreValue)
                ? ` | ${scoreContext.scoreColumn.endsWith("_eur") ? formatCurrency(scoreContext.scoreValue) : formatNumber(scoreContext.scoreValue)}`
                : ""
            }</span>
            ${
              hasActiveLens() && (Number.isFinite(row._styleScore) || Number.isFinite(row._roleScore))
                ? `<span class="player-cell__note player-cell__note--style">${escapeHtml(buildLensFitLine(row))}</span>`
                : ""
            }
            ${
              hasActiveLens() && row._styleFitReasonLine
                ? `<span class="player-cell__note player-cell__note--fit-driver${
                    row._styleFitDataQuality === "limited" ? " fit-driver-summary--muted" : ""
                  }">${escapeHtml(row._styleFitReasonLine)}</span>`
                : ""
            }
            <span class="player-cell__note player-cell__note--style">${escapeHtml(talentLine)}</span>
            ${
              talentDrivers
                ? `<span class="player-cell__note player-cell__note--fit-driver">${escapeHtml(talentDrivers)}</span>`
                : ""
            }
            <span class="player-cell__note player-cell__note--freshness${
              freshness.status === "limited" ? " fit-driver-summary--muted" : ""
            }">${escapeHtml(freshness.compactLine)}</span>
          </td>
          <td>
            <div class="table-status">
              <div class="table-status__chips">
                ${buildDecisionPillMarkup(decision)}
                ${buildStyleFitMarkup(row)}
              </div>
              <span class="table-status__note">${escapeHtml(laneDecision)}</span>
              <span class="table-status__sub">${escapeHtml(decision.gapNote)} | ${escapeHtml(talentLine)}</span>
            </div>
          </td>
          <td class="num">
            <div class="value-stack">
              <strong>${formatCurrency(row.market_value_eur)}</strong>
              <span>Current market</span>
            </div>
          </td>
            <td class="num">
              <div class="value-stack">
                <strong>${formatCurrency(expected)}</strong>
                <span>Adjusted fair value</span>
              </div>
            </td>
          <td class="num">
            <div class="value-stack value-stack--gap">
              <strong class="${consGap >= 0 ? "positive" : "negative"}">${formatCurrency(consGap)}</strong>
              <span>Conservative gap</span>
            </div>
          </td>
          <td class="num">
            <div class="value-stack">
              <strong>${formatNumber(row.undervaluation_confidence)}</strong>
              <span>${escapeHtml(decision.confidenceLabel)} confidence</span>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");

  const start = state.total === 0 ? 0 : isSystemFitMode() ? 1 : state.offset + 1;
  const end = isSystemFitMode() ? state.count : Math.min(state.offset + state.count, state.total);
  el.resultCount.textContent = `${formatInt(state.count)} / ${formatInt(state.total)} rows`;
  el.resultRange.textContent = isSystemFitMode() ? `showing top ${end}` : `showing ${start}-${end}`;
  renderBoardHighlights();
}

function renderPager() {
  if (isSystemFitMode()) {
    el.prevBtn.disabled = true;
    el.nextBtn.disabled = true;
    return;
  }
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

async function activateDetailTab(tab) {
  setDetailTab(tab);
  if (tab === "trajectory") {
    await ensureSelectedTrajectoryLoaded();
    return;
  }
  if (tab === "tactical") {
    await ensureSelectedSimilarLoaded();
  }
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

function buildLoadingSkeletonMarkup(count = 3, { listOnly = false } = {}) {
  const items = Array.from({ length: count })
    .map(
      () => `
        <li class="skeleton-card">
          <span class="skeleton-line skeleton-line--short"></span>
          <span class="skeleton-line skeleton-line--long"></span>
          <span class="skeleton-line skeleton-line--mid"></span>
        </li>
      `
    )
    .join("");
  return listOnly ? items : `<ul class="loading-skeleton-list">${items}</ul>`;
}

function buildSimilarPlayerCardsMarkup(items, emptyMessage = "No similar-player matches returned.") {
  if (!Array.isArray(items) || !items.length) {
    return `<li class="risk-item risk-item--none">${escapeHtml(emptyMessage)}</li>`;
  }
  return items
    .map((item) => {
      const playerId = safeText(item?.player_id);
      const season = safeText(item?.season);
      const meta = [item?.club, item?.league, item?.position, item?.season]
        .filter(Boolean)
        .map((value) => safeText(value))
        .join(" | ");
      const similarity = firstFiniteNumber(item?.similarity_score, item?.score);
      const predicted = firstFiniteNumber(item?.predicted_value);
      const market = firstFiniteNumber(item?.market_value_eur);
      const delta = Number.isFinite(predicted) && Number.isFinite(market) ? predicted - market : NaN;
      return `
        <li class="risk-item">
          <button
            type="button"
            class="similar-card"
            data-similar-player-id="${escapeHtml(playerId)}"
            data-similar-player-season="${escapeHtml(season)}"
          >
            <div class="similar-card__header">
              <span class="similar-card__name">${escapeHtml(safeText(item?.name || playerId))}</span>
              <span class="similarity-badge">${Number.isFinite(similarity) ? formatPct(similarity) : "-"}</span>
            </div>
            <p class="similar-card__meta">${escapeHtml(meta || playerId)}</p>
            <div class="similar-card__footer">
              <span class="similar-card__delta">Position-aware match | Predicted vs market: ${Number.isFinite(delta) ? formatSignedCurrency(delta) : "-"}</span>
              <span class="similar-card__values">${formatCurrency(predicted)} vs ${formatCurrency(market)}</span>
            </div>
          </button>
        </li>
      `;
    })
    .join("");
}

function buildProxyEstimateListMarkup(proxyPayload, emptyMessage = "No advisory proxy estimates available.") {
  const metrics = Array.isArray(proxyPayload?.metrics) ? proxyPayload.metrics : [];
  if (!metrics.length) {
    return `<li class="risk-item risk-item--none">${escapeHtml(emptyMessage)}</li>`;
  }
  return metrics
    .map(
      (item) => `
        <li class="risk-item">
          <div class="risk-head">
            <span class="risk-code">${escapeHtml(safeText(item?.label || item?.metric_key))}</span>
            <span class="risk-severity risk-severity--medium">${escapeHtml(safeText(item?.support_label || "advisory"))}</span>
          </div>
          <p class="risk-message">
            Estimated ${escapeHtml(formatNumber(item?.estimated_value))} | neighbors ${escapeHtml(
        formatInt(item?.neighbor_count)
      )} | mean similarity ${escapeHtml(formatPct(item?.mean_similarity))}
          </p>
        </li>
      `
    )
    .join("");
}

function trajectoryLabelCopy(label) {
  if (label === "ascending") return "↑ Ascending";
  if (label === "declining") return "↓ Declining";
  return "→ Stable";
}

function trajectoryTone(label) {
  if (label === "ascending") return "ascending";
  if (label === "declining") return "declining";
  return "stable";
}

function buildTrajectoryChartMarkup(trajectory = null) {
  const seasons = Array.isArray(trajectory?.seasons) ? trajectory.seasons : [];
  const values = seasons
    .map((item) => ({
      season: safeText(item?.season),
      value: firstFiniteNumber(item?.predicted_value),
    }))
    .filter((item) => Number.isFinite(item.value));
  if (!values.length) {
    return '<p class="details-placeholder">No trajectory chart available.</p>';
  }

  const projected = firstFiniteNumber(trajectory?.projected_next_value);
  const chartValues = values.map((item) => item.value);
  if (Number.isFinite(projected)) chartValues.push(projected);
  const min = Math.min(...chartValues);
  const max = Math.max(...chartValues);
  const span = Math.max(max - min, 1);
  const width = 760;
  const height = 220;
  const padX = 30;
  const padY = 24;
  const usableWidth = width - padX * 2;
  const usableHeight = height - padY * 2;
  const xForIndex = (idx, total) => padX + (usableWidth * idx) / Math.max(total - 1, 1);
  const yForValue = (value) => padY + usableHeight - ((value - min) / span) * usableHeight;
  const points = values.map((item, idx) => `${xForIndex(idx, values.length).toFixed(1)},${yForValue(item.value).toFixed(1)}`);
  const guides = [0.25, 0.5, 0.75].map((fraction) => {
    const y = padY + usableHeight * fraction;
    return `<line x1="${padX}" y1="${y.toFixed(1)}" x2="${(width - padX).toFixed(1)}" y2="${y.toFixed(1)}" class="trajectory-chart__guide"></line>`;
  });
  const labels = values
    .map(
      (item, idx) => `
        <text x="${xForIndex(idx, values.length).toFixed(1)}" y="${(height - 8).toFixed(1)}" text-anchor="middle" class="trajectory-chart__label">
          ${escapeHtml(item.season)}
        </text>
      `
    )
    .join("");
  const circles = values
    .map(
      (item, idx) => `
        <circle
          cx="${xForIndex(idx, values.length).toFixed(1)}"
          cy="${yForValue(item.value).toFixed(1)}"
          r="4"
          class="trajectory-chart__point"
        />
      `
    )
    .join("");
  let projectionMarkup = "";
  if (Number.isFinite(projected)) {
    const last = values[values.length - 1];
    const startX = xForIndex(values.length - 1, values.length);
    const startY = yForValue(last.value);
    const endX = width - padX;
    const endY = yForValue(projected);
    projectionMarkup = `
      <line x1="${startX.toFixed(1)}" y1="${startY.toFixed(1)}" x2="${endX.toFixed(1)}" y2="${endY.toFixed(1)}" class="trajectory-chart__projection"></line>
      <circle cx="${endX.toFixed(1)}" cy="${endY.toFixed(1)}" r="4" class="trajectory-chart__point"></circle>
      <text x="${endX.toFixed(1)}" y="${(height - 8).toFixed(1)}" text-anchor="end" class="trajectory-chart__label">Proj.</text>
    `;
  }
  return `
    <svg class="trajectory-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Player value trajectory">
      <rect x="0" y="0" width="${width}" height="${height}" rx="16" class="trajectory-chart__surface"></rect>
      ${guides.join("")}
      <polyline points="${points.join(" ")}" class="trajectory-chart__line"></polyline>
      ${projectionMarkup}
      ${circles}
      ${labels}
    </svg>
  `;
}

function buildTrajectoryTableRows(trajectory = null) {
  const seasons = Array.isArray(trajectory?.seasons) ? trajectory.seasons.slice().reverse() : [];
  if (!seasons.length) {
    return '<tr><td colspan="8">No trajectory history loaded.</td></tr>';
  }
  return seasons
    .map(
      (item) => `
        <tr>
          <td>${escapeHtml(safeText(item?.season))}</td>
          <td class="num">${formatCurrency(item?.predicted_value)}</td>
          <td class="num">${formatSignedCurrency(item?.delta_predicted_value)}</td>
          <td class="num">${formatCurrency(item?.market_value_eur)}</td>
          <td class="num">${formatInt(item?.minutes)}</td>
          <td class="num">${formatNumber(item?.goals)}</td>
          <td class="num">${formatNumber(item?.assists)}</td>
          <td class="num">${formatNumber(item?.xg)}</td>
        </tr>
      `
    )
    .join("");
}

function renderSimilarPlayers(similarPayload = null, { loading = false, error = "" } = {}) {
  if (!el.detailSimilar || !el.detailSimilarSummary) return;
  if (loading) {
    el.detailSimilarSummary.textContent = "Loading similar players...";
    el.detailSimilar.innerHTML = buildLoadingSkeletonMarkup(3, { listOnly: true });
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
    ? `Top ${items.length} position-aware matches from the similarity index using ${formatInt(
        similarPayload?.feature_count_used
      )} active metrics for ${safeText(similarPayload?.position_group || "this role")}.`
    : "No similar-player matches returned.";
  el.detailSimilar.innerHTML = buildSimilarPlayerCardsMarkup(items);
}

function renderProxyEstimates(proxyPayload = null, { loading = false, error = "" } = {}) {
  if (!el.detailProxySummary || !el.detailProxyList || !el.detailProxySection) return;
  if (loading) {
    el.detailProxySection.hidden = false;
    el.detailProxySummary.textContent = "Loading advisory proxy estimates...";
    el.detailProxyList.innerHTML = buildLoadingSkeletonMarkup(3, { listOnly: true });
    return;
  }
  if (error) {
    el.detailProxySection.hidden = true;
    return;
  }
  const available = proxyPayload?.available === true;
  const metrics = Array.isArray(proxyPayload?.metrics) ? proxyPayload.metrics : [];
  if (!available || !metrics.length) {
    el.detailProxySection.hidden = true;
    return;
  }
  el.detailProxySection.hidden = false;
  el.detailProxySummary.textContent = `${safeText(
    proxyPayload?.summary
  )} Derived from comparable-player neighbors; do not treat as observed data.`;
  el.detailProxyList.innerHTML = buildProxyEstimateListMarkup(proxyPayload);
}

function renderTrajectoryPanel(trajectory = null, { loading = false, error = "", deferred = false } = {}) {
  if (
    !el.detailTrajectoryBadge ||
    !el.detailTrajectorySummary ||
    !el.detailTrajectoryProject ||
    !el.detailTrajectoryChart ||
    !el.detailTrajectoryTableBody
  ) {
    return;
  }
  if (loading) {
    el.detailTrajectoryBadge.className = "trajectory-badge trajectory-badge--stable";
    el.detailTrajectoryBadge.textContent = "Loading trajectory...";
    el.detailTrajectorySummary.textContent = "Building the multi-season value view...";
    el.detailTrajectoryProject.textContent = "Projected next value: -";
    el.detailTrajectoryChart.innerHTML = buildLoadingSkeletonMarkup(2);
    el.detailTrajectoryTableBody.innerHTML = '<tr><td colspan="8">Loading trajectory history...</td></tr>';
    return;
  }
  if (error) {
    el.detailTrajectoryBadge.className = "trajectory-badge trajectory-badge--stable";
    el.detailTrajectoryBadge.textContent = "Trajectory unavailable";
    el.detailTrajectorySummary.textContent = `Trajectory unavailable: ${error}`;
    el.detailTrajectoryProject.textContent = "Projected next value: -";
    el.detailTrajectoryChart.innerHTML = '<p class="details-placeholder">Trajectory chart unavailable.</p>';
    el.detailTrajectoryTableBody.innerHTML = '<tr><td colspan="8">Trajectory history unavailable.</td></tr>';
    return;
  }
  if (deferred) {
    el.detailTrajectoryBadge.className = "trajectory-badge trajectory-badge--stable";
    el.detailTrajectoryBadge.textContent = "Trajectory on demand";
    el.detailTrajectorySummary.textContent = "Open the Trajectory tab to load the multi-season value view.";
    el.detailTrajectoryProject.textContent = "Projected next value: -";
    el.detailTrajectoryChart.innerHTML = '<p class="details-placeholder">No trajectory chart loaded yet.</p>';
    el.detailTrajectoryTableBody.innerHTML = '<tr><td colspan="8">Open the Trajectory tab to load history.</td></tr>';
    return;
  }
  const seasonCount = Array.isArray(trajectory?.seasons) ? trajectory.seasons.length : 0;
  if (!seasonCount) {
    el.detailTrajectoryBadge.className = "trajectory-badge trajectory-badge--stable";
    el.detailTrajectoryBadge.textContent = "Trajectory unavailable";
    el.detailTrajectorySummary.textContent = "No multi-season history was available for this player.";
    el.detailTrajectoryProject.textContent = "Projected next value: -";
    el.detailTrajectoryChart.innerHTML = '<p class="details-placeholder">No trajectory chart available.</p>';
    el.detailTrajectoryTableBody.innerHTML = '<tr><td colspan="8">No trajectory history loaded.</td></tr>';
    return;
  }
  const label = String(trajectory?.trajectory_label || "stable");
  const tone = trajectoryTone(label);
  const slope = firstFiniteNumber(trajectory?.slope_pct);
  const r2 = firstFiniteNumber(trajectory?.r2);
  const projected = firstFiniteNumber(trajectory?.projected_next_value);
  const peakSeason = safeText(trajectory?.peak_season || "-");
  el.detailTrajectoryBadge.className = `trajectory-badge trajectory-badge--${tone}`;
  el.detailTrajectoryBadge.textContent = trajectoryLabelCopy(label);
  el.detailTrajectorySummary.textContent = [
    Number.isFinite(slope) ? `Value slope ${slope > 0 ? "+" : ""}${formatNumber(slope)}%.` : null,
    Number.isFinite(r2) ? `Consistency ${formatNumber(r2)} R².` : null,
    peakSeason !== "-" ? `Peak season ${peakSeason}.` : null,
  ]
    .filter(Boolean)
    .join(" ");
  el.detailTrajectoryProject.textContent = `Projected next value: ${formatCurrency(projected)}`;
  el.detailTrajectoryChart.innerHTML = buildTrajectoryChartMarkup(trajectory);
  el.detailTrajectoryTableBody.innerHTML = buildTrajectoryTableRows(trajectory);
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

function renderProfileModal(row, { profile = null, trajectory = null, reportError = "" } = {}) {
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
  const similar =
    state.selectedSimilar && typeof state.selectedSimilar === "object"
      ? state.selectedSimilar
      : profile?.similar_players && typeof profile.similar_players === "object"
      ? profile.similar_players
      : null;
  const proxyEstimates =
    profile?.proxy_estimates && typeof profile.proxy_estimates === "object" ? profile.proxy_estimates : null;
  const trajectoryPayload =
    trajectory && typeof trajectory === "object"
      ? trajectory
      : state.selectedTrajectory && typeof state.selectedTrajectory === "object"
      ? state.selectedTrajectory
      : null;
  const tacticalContext = profile?.external_tactical_context && typeof profile.external_tactical_context === "object" ? profile.external_tactical_context : null;
  const availabilityContext = profile?.availability_context && typeof profile.availability_context === "object" ? profile.availability_context : null;
  const marketContext = profile?.market_context && typeof profile.market_context === "object" ? profile.market_context : null;
  const rankingDiagnostics =
    Number.isFinite(Number(row?._funnelScore)) && state.funnelDiagnostics ? state.funnelDiagnostics : state.queryDiagnostics;
  const scoreContext = resolveRowScoreContext(
    mergedRow,
    rankingDiagnostics,
    Number.isFinite(Number(row?._funnelScore)) ? "funnel" : currentWorkbenchSourceMode()
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
        <p class="detail-summary">${escapeHtml(
          similar?.available
            ? `Position-aware similarity using ${formatInt(similar?.feature_count_used)} active metrics for ${safeText(
                similar?.position_group || "this role"
              )}.`
            : "No similar-player matches available."
        )}</p>
        <ul class="risk-list">${buildSimilarPlayerCardsMarkup(
          similar?.available === true && Array.isArray(similar?.items) ? similar.items : [],
          "No similar-player matches available."
        )}</ul>
      </article>
      ${
        proxyEstimates?.available === true && Array.isArray(proxyEstimates?.metrics) && proxyEstimates.metrics.length
          ? `
      <article class="profile-card profile-grid__span-4">
        <h3>Estimated From Comparable Players</h3>
        <p class="detail-summary">${escapeHtml(safeText(proxyEstimates?.summary || ""))}</p>
        <ul class="risk-list">${buildProxyEstimateListMarkup(proxyEstimates, "No advisory proxy estimates available.")}</ul>
      </article>
      `
          : ""
      }
      <article class="profile-card profile-grid__span-4">
        <h3>Schedule + Market Context</h3>
        <p class="detail-summary">${escapeHtml(providerContextFallbackSummary("market", mergedRow, marketContext))}</p>
        <ul class="risk-list">${buildSignalListMarkup(marketContext?.signals, "No schedule or market context signals.")}</ul>
      </article>
      <article class="profile-card profile-grid__span-12">
        <h3>Trajectory</h3>
        <p class="detail-summary">${
          trajectoryPayload
            ? `${escapeHtml(trajectoryLabelCopy(trajectoryPayload.trajectory_label))} | slope ${
                Number.isFinite(firstFiniteNumber(trajectoryPayload.slope_pct))
                  ? `${firstFiniteNumber(trajectoryPayload.slope_pct) > 0 ? "+" : ""}${formatNumber(trajectoryPayload.slope_pct)}%`
                  : "-"
              } | projected next value ${formatCurrency(trajectoryPayload.projected_next_value)}.`
            : "Trajectory history unavailable."
        }</p>
        <div class="trajectory-chart-wrap">${buildTrajectoryChartMarkup(trajectoryPayload)}</div>
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
  state.selectedSimilar = null;
  state.selectedTrajectory = null;
  state.selectedLatestDecision = null;
  state.detailDecisionSourceSurface = "";
  resetDecisionDraft();
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
  el.detailSummary.textContent = "Select a player to load the player brief summary.";
  renderDetailFit(null);
  renderDetailFreshness(null);
  renderDetailTalentView(null);
  if (el.detailContextGlance) {
    el.detailContextGlance.innerHTML =
      '<li class="risk-item risk-item--none">Select a player to load key signals and context.</li>';
  }
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
  if (el.detailProxySummary) {
    el.detailProxySummary.textContent = "Select a player to review advisory proxy estimates for missing metrics.";
  }
  if (el.detailProxyList) {
    el.detailProxyList.innerHTML = '<li class="risk-item risk-item--none">No advisory proxy estimates loaded.</li>';
  }
  if (el.detailProxySection) {
    el.detailProxySection.hidden = true;
  }
  renderScoutDecisionSummary(null);
  renderScoutDecisionComposer();
  if (el.detailTrajectoryBadge) {
    el.detailTrajectoryBadge.className = "trajectory-badge trajectory-badge--stable";
    el.detailTrajectoryBadge.textContent = "No trajectory loaded";
  }
  if (el.detailTrajectorySummary) {
    el.detailTrajectorySummary.textContent = "Select a player to inspect the multi-season value trajectory.";
  }
  if (el.detailTrajectoryProject) {
    el.detailTrajectoryProject.textContent = "Projected next value: -";
  }
  if (el.detailTrajectoryChart) {
    el.detailTrajectoryChart.innerHTML = '<p class="details-placeholder">No trajectory chart loaded yet.</p>';
  }
  if (el.detailTrajectoryTableBody) {
    el.detailTrajectoryTableBody.innerHTML = '<tr><td colspan="8">No trajectory history loaded.</td></tr>';
  }
  el.detailAvailabilitySummary.textContent = "Select a player to load provider availability context.";
  el.detailAvailabilityList.innerHTML = '<li class="risk-item risk-item--none">No provider availability signals loaded.</li>';
  el.detailMarketContextSummary.textContent = "Select a player to load schedule and market context.";
  el.detailMarketContextList.innerHTML = '<li class="risk-item risk-item--none">No schedule or market context loaded.</li>';
  renderArchetypeProfile(null);
  renderFormationFitProfile(null);
  renderSimilarPlayers(null);
  renderTrajectoryPanel(null);
  renderRadarProfile(null);
  if (el.detailExportPdf) el.detailExportPdf.disabled = true;
  el.detailExportJson.disabled = true;
  el.detailExportCsv.disabled = true;
  if (el.profileModalExportPdf) el.profileModalExportPdf.disabled = true;
  setDetailTab("overview");
  renderProfileModal(null);
  closeProfileModal();
  renderRows();
}

function buildContextGlanceItems(
  row,
  decision,
  scoreContext,
  historyPayload,
  tacticalContext,
  availabilityContext,
  marketContext
) {
  const items = [];
  if (decision?.priceContext) {
    items.push({
      label: "Valuation stance",
      message: decision.priceContext,
      tone: decision.tone === "pass" ? "high" : "medium",
    });
  }
  if (scoreContext?.scoreLabel) {
    const valueText = Number.isFinite(scoreContext.scoreValue)
      ? scoreContext.scoreColumn.endsWith("_eur")
        ? formatCurrency(scoreContext.scoreValue)
        : formatNumber(scoreContext.scoreValue)
      : "signal only";
    items.push({
      label: "Ranking driver",
      message: `${scoreContext.scoreLabel} | ${valueText} | ${scoreContext.rankingBasisLabel}.`,
      tone: "medium",
    });
  }
  if (historyPayload?.summary_text) {
    items.push({
      label: "Evidence depth",
      message: historyPayload.summary_text,
      tone: "medium",
    });
  }
  [
    { label: "Tactical context", key: "tactical", payload: tacticalContext },
    { label: "Availability context", key: "availability", payload: availabilityContext },
    { label: "Market context", key: "market", payload: marketContext },
  ].forEach((item) => {
    const summary = providerContextFallbackSummary(item.key, row, item.payload);
    if (
      summary &&
      !/select a player/i.test(summary) &&
      !/no .* loaded/i.test(summary) &&
      !/^no /i.test(summary)
    ) {
      items.push({
        label: item.label,
        message: summary,
        tone: "medium",
      });
    }
  });
  return items.slice(0, 5);
}

function renderDetail(
  row,
  {
    profile = null,
    similarPayload = null,
    trajectory = null,
    trajectoryError = "",
    trajectoryDeferred = false,
    reportLoading = false,
    reportError = "",
  } = {}
) {
  if (!row) return clearDetail();
  const reportPlayer = profile?.player && typeof profile.player === "object" ? profile.player : {};
  const historyPayload = profile?.history_strength && typeof profile.history_strength === "object" ? profile.history_strength : null;
  const tacticalContext = profile?.external_tactical_context && typeof profile.external_tactical_context === "object" ? profile.external_tactical_context : null;
  const availabilityContext = profile?.availability_context && typeof profile.availability_context === "object" ? profile.availability_context : null;
  const marketContext = profile?.market_context && typeof profile.market_context === "object" ? profile.market_context : null;
  const mergedRow = { ...row, ...reportPlayer };
  const talent = getTalentView(mergedRow, profile);
  const statGroups = Array.isArray(profile?.stat_groups) ? profile.stat_groups : buildStatGroups(mergedRow, profile, null);
  const gaps = deriveGapValues(mergedRow, profile);
  const rankingDiagnostics =
    Number.isFinite(Number(row?._funnelScore)) && state.funnelDiagnostics ? state.funnelDiagnostics : state.queryDiagnostics;
  const scoreContext = resolveRowScoreContext(
    mergedRow,
    rankingDiagnostics,
    Number.isFinite(Number(row?._funnelScore)) ? "funnel" : currentWorkbenchSourceMode()
  );
  const decision = summarizeRecruitmentDecision(mergedRow, profile, {
    source: Number.isFinite(Number(row?._funnelScore)) ? "funnel" : currentWorkbenchDecisionSource(),
  });
  const laneDecision = laneAwareDecisionNote(mergedRow, decision.nextAction);
  const roleLens = buildRoleLens(mergedRow, profile);
  const leagueStatus = summarizeLeagueStatus(mergedRow.league);
  const leagueAdjustment = leagueAdjustmentMeta(mergedRow, profile);
  const coverageWarnings = detailCoverageWarnings(mergedRow, profile);
  const provenanceBadges = buildProvenanceBadges(mergedRow)
    .concat([{ label: leagueStatus.label, tone: leagueStatus.tone }])
    .map((badge) => buildBadgeChipMarkup(badge.label, badge.tone))
    .join("");
  const whyRanked = whyRankedItems(mergedRow, profile, scoreContext);
  const contextGlanceItems = buildContextGlanceItems(
    mergedRow,
    decision,
    scoreContext,
    historyPayload,
    tacticalContext,
    availabilityContext,
    marketContext
  );
  state.selectedRow = mergedRow;
  state.selectedProfile = profile || null;
  state.selectedReport = profile || null;
  state.selectedHistory = profile?.history_strength ? { history_strength: profile.history_strength } : null;
  state.selectedSimilar = similarPayload || null;
  state.selectedTrajectory = trajectory || null;
  state.selectedLatestDecision = profile?.latest_decision || null;
  state.detailDecisionSourceSurface = String(row?._decisionSourceSurface || "").trim();
  el.detailPlaceholder.hidden = true;
  el.detailContent.hidden = false;
  renderRows();
  renderScoutDecisionSummary(state.selectedLatestDecision);
  renderScoutDecisionComposer();

  const market = firstFiniteNumber(profile?.valuation_guardrails?.market_value_eur, mergedRow.market_value_eur, 0);
  const expected = firstFiniteNumber(
    profile?.valuation_guardrails?.league_adjusted_fair_value_eur,
    profile?.valuation_guardrails?.fair_value_eur,
    mergedRow.league_adjusted_fair_value_eur,
    mergedRow.fair_value_eur,
    mergedRow.expected_value_eur,
    0
  );
  const rawFair = firstFiniteNumber(
    profile?.valuation_guardrails?.raw_fair_value_eur,
    mergedRow.raw_fair_value_eur,
    mergedRow.expected_value_raw_eur
  );
  const lower = firstFiniteNumber(mergedRow.expected_value_low_eur, 0);
  const scaleMax = Math.max(market, expected, lower, 1);

  el.detailName.textContent = safeText(mergedRow.name);
  el.detailMeta.textContent = `${safeText(mergedRow.club)} | ${safeText(mergedRow.league)} | ${safeText(
    getPosition(mergedRow)
  )} | age ${formatNumber(mergedRow.age)} | ${formatInt(getMinutes(mergedRow))} minutes`;
  el.detailDecisionPill.className = `decision-pill decision-pill--${decision.tone}`;
  el.detailDecisionPill.textContent = decision.label;
  el.detailDecisionNext.textContent = laneDecision;
  el.detailDecisionReason.textContent = decision.reason;
  el.detailGap.textContent = Number.isFinite(decision.gap) ? formatCurrency(decision.gap) : "-";
  el.detailGapNote.textContent = decision.gapNote;
  el.detailConfidenceScore.textContent = Number.isFinite(decision.confidenceScore)
    ? `${decision.confidenceLabel} | ${formatNumber(decision.confidenceScore)}`
    : decision.confidenceLabel;
  el.detailConfidenceNote.textContent = decision.confidenceNote;
  el.detailPriceStance.textContent = decision.priceStance;
  el.detailPriceContext.textContent =
    leagueAdjustment.needsWarning && leagueAdjustment.reason
      ? `${decision.priceContext} ${leagueAdjustment.reason}`
      : decision.priceContext;
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
    ["League-adjusted fair value", formatCurrency(expected)],
    ["Raw model fair value", formatCurrency(rawFair)],
    ["Current Level", Number.isFinite(Number(talent.current_level_score)) ? formatNumber(talent.current_level_score) : "-"],
    ["Future Potential", Number.isFinite(Number(talent.future_potential_score)) ? formatNumber(talent.future_potential_score) : "-"],
    ["Age", formatNumber(mergedRow.age)],
    ["Minutes", formatInt(getMinutes(mergedRow))],
    ["Talent Family", safeText(talent.talent_position_family)],
    ["Role Lens", safeText(roleLens.label)],
    ["Conservative Gap (capped)", formatCurrency(decision.gap)],
    ["Cap Threshold", formatCurrency(gaps.capThreshold)],
    [
      "League pricing adjustment",
      leagueAdjustment.needsWarning
        ? `${safeText(leagueAdjustment.bucket).replace(/_/g, " ")} | alpha ${Number.isFinite(leagueAdjustment.alpha) ? formatNumber(leagueAdjustment.alpha) : "-"}`
        : "standard",
    ],
    [
      "Holdout reliability",
      `${Number.isFinite(leagueAdjustment.holdoutR2) ? formatPct(leagueAdjustment.holdoutR2) : "n/a"} R² | ${
        Number.isFinite(leagueAdjustment.holdoutWmape) ? formatPct(leagueAdjustment.holdoutWmape) : "n/a"
      } WMAPE`,
    ],
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
    if (el.detailExportPdf) el.detailExportPdf.disabled = true;
    el.detailExportJson.disabled = true;
    el.detailExportCsv.disabled = true;
    if (el.profileModalExportPdf) el.profileModalExportPdf.disabled = true;
    el.detailSummary.textContent = "Loading player brief...";
    renderDetailFit(mergedRow, { loading: true });
    renderDetailFreshness(mergedRow, profile, { loading: true });
    renderDetailTalentView(mergedRow, profile, { loading: true });
    if (el.detailContextGlance) {
      el.detailContextGlance.innerHTML =
        '<li class="risk-item risk-item--none">Loading key signals and context...</li>';
    }
    el.detailRoleSummary.textContent = "Loading position-specific metric lens...";
    el.detailRoleMetrics.innerHTML = '<li class="risk-item risk-item--none">Loading role evidence...</li>';
    el.detailHistory.textContent = "Loading evidence depth...";
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
    renderProxyEstimates(null, { loading: true });
    renderTrajectoryPanel(null, { deferred: true });
    renderRadarProfile(null, { loading: true });
    if (state.profileModalOpen) {
      renderProfileModal(mergedRow, { profile, trajectory, reportError });
    }
    return;
  }
  if (reportError) {
    if (el.detailExportPdf) el.detailExportPdf.disabled = true;
    el.detailExportJson.disabled = true;
    el.detailExportCsv.disabled = true;
    if (el.profileModalExportPdf) el.profileModalExportPdf.disabled = true;
    el.detailSummary.textContent = `Player brief unavailable: ${reportError}`;
    renderDetailFit(mergedRow);
    renderDetailFreshness(mergedRow, profile);
    renderDetailTalentView(mergedRow, profile, { error: reportError });
    if (el.detailContextGlance) {
      el.detailContextGlance.innerHTML =
        '<li class="risk-item risk-item--none">Signals and context are unavailable for this player.</li>';
    }
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
    renderProxyEstimates(null, { error: reportError });
    renderTrajectoryPanel(null, { error: reportError });
    renderRadarProfile(null, { error: reportError });
    if (state.profileModalOpen) {
      renderProfileModal(mergedRow, { profile, trajectory, reportError });
    }
    return;
  }

  if (profile?.summary_text) {
    el.detailSummary.textContent = profile.summary_text;
  } else {
    el.detailSummary.textContent =
      "No player brief summary loaded yet. Use the export actions if you need the full memo payload.";
  }
  if (el.detailExportPdf) el.detailExportPdf.disabled = false;
  el.detailExportJson.disabled = false;
  el.detailExportCsv.disabled = false;
  if (el.profileModalExportPdf) el.profileModalExportPdf.disabled = false;
  renderDetailFit(mergedRow);
  renderDetailFreshness(mergedRow, profile);
  renderDetailTalentView(mergedRow, profile);
  if (el.detailContextGlance) {
    el.detailContextGlance.innerHTML = buildNarrativeListMarkup(
      contextGlanceItems,
      "No additional signals or context were available for this player."
    );
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
  renderSimilarPlayers(
    similarPayload || (profile?.similar_players && typeof profile.similar_players === "object" ? profile.similar_players : null)
  );
  renderProxyEstimates(profile?.proxy_estimates && typeof profile.proxy_estimates === "object" ? profile.proxy_estimates : null);
  renderTrajectoryPanel(trajectory, { error: trajectoryError, deferred: trajectoryDeferred && !trajectory && !trajectoryError });
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
    renderProfileModal(mergedRow, { profile, trajectory, reportError });
  }
}

function renderModeAffordances() {
  const shortlist = state.mode === "shortlist";
  const systemFit = isSystemFitMode();
  const selectedSlot = currentSystemFitSlot();
  el.title.textContent = systemFit
    ? safeText(selectedSlot?.slot_label || "System Fit Board")
    : shortlist
    ? "Recruitment Board"
    : "Valuation Board";
  el.sort.disabled = shortlist || systemFit;
  el.sortDir.disabled = shortlist || systemFit;
  el.minConfidence.disabled = shortlist ? true : false;
  if (systemFit) el.minConfidence.disabled = false;
  el.minGap.disabled = shortlist || systemFit;
  el.topN.disabled = !shortlist;
  el.position.disabled = systemFit;
  el.undervaluedOnly.disabled = shortlist || systemFit;
  if (el.playstyle) el.playstyle.disabled = systemFit;
  if (el.roleLens) el.roleLens.disabled = systemFit;
  if (el.roleNeed) el.roleNeed.disabled = systemFit;
  if (el.systemFitTemplateControl) el.systemFitTemplateControl.hidden = !systemFit;
  if (el.systemFitLaneControl) el.systemFitLaneControl.hidden = !systemFit;
  if (el.roleNeedControl) el.roleNeedControl.hidden = systemFit;
  if (el.resultsColTarget) el.resultsColTarget.textContent = "Target";
  if (el.resultsColDecision) el.resultsColDecision.textContent = systemFit ? "System fit" : "Decision";
  if (el.resultsColMarket) el.resultsColMarket.textContent = "Market";
  if (el.resultsColExpected) el.resultsColExpected.textContent = systemFit ? "Current" : "Adj. fair";
  if (el.resultsColGap) el.resultsColGap.textContent = systemFit ? "Future" : "Gap";
  if (el.resultsColConfidence) el.resultsColConfidence.textContent = systemFit ? "Confidence" : "Confidence";
  renderSystemFitSlots();
  renderResultsNote();
}

function renderResultsNote() {
  if (!el.resultsNote) return;
  const filterSummary = describeRecruitmentFilters(currentWorkbenchWorkflowSummary());
  const laneSummary = boardLaneSummary();
  const laneNote = ` Active lane: ${laneSummary.label}. ${laneSummary.copy}`;
  if (isSystemFitMode()) {
    const slot = currentSystemFitSlot();
    const template = currentSystemFitTemplate();
    const slotText = slot
      ? `${safeText(slot.slot_label)} | ${safeText(slot.role_template_label)}`
      : "select a slot";
    const budget = parseOptionalPositive(state.budgetBand);
    const budgetText = budget ? ` Budget posture: ${budgetLabel(budget)}.` : "";
    el.resultsNote.textContent = `System fit mode is backend-ranked for ${safeText(
      template?.label || state.systemFitTemplate
    )}. Current slot: ${slotText}. This view is canonical slot-level ranking, so frontend playstyle and role reranking are disabled.${budgetText}${laneNote}`;
    return;
  }
  const lensNote = state.playstyle && state.roleLens
    ? ` Active lenses: ${activeLensDisplayLabel()}. Style fit adds 20%; role fit adds 10% on top of the current ranking driver.`
    : state.roleLens
    ? ` Active role lens: ${roleLensLabel()}. Role fit adds 10% on top of the current ranking driver.`
    : state.playstyle
    ? ` Active playstyle lens: ${playstyleLabel()}. Style fit is a 20% frontend overlay on top of the current ranking driver.`
    : "";
  if (state.mode === "predictions") {
    el.resultsNote.textContent = `This is the Current Level / Pricing view, not a live pursuit order. Use it for price discipline under the active brief (${filterSummary}), then switch back to Recruitment Board or Target Funnel when you want advisory future-potential ranking.${laneNote}${lensNote}`;
    return;
  }
  const diagnostics = state.queryDiagnostics || {};
  const scoreColumn = diagnostics.score_column || diagnostics.scoreColumn || "shortlist_score";
  const rankingBasis = diagnostics.ranking_basis || diagnostics.rankingBasis || "guardrailed_gap_confidence_history";
  const precisionRows = diagnostics?.precision_at_k?.rows || [];
  const p25 = precisionRows.find((row) => Number(row.k) === 25);
  const precisionText =
    p25 && Number.isFinite(Number(p25.precision)) ? ` Precision@25 ${formatPct(Number(p25.precision))}.` : "";
  el.resultsNote.textContent = `Recruitment brief: ${filterSummary}. This board is ordered for Future Potential / Advisory review, so scan pursue and watch calls first, then open one memo to decide the next action. Ranking driver: ${humanizeScoreColumn(
    scoreColumn
  )} | ${rankingBasisLabel(rankingBasis)}.${precisionText}${laneNote}${lensNote}`;
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
  const payload = await getJson("/market-value/ui-bootstrap", {
    split: state.split,
  });
  applyUiBootstrapPayload(payload);
}

async function fetchPredictionsPage() {
  const maxBudget = parseOptionalPositive(state.budgetBand);
  const diagnostics = {
    score_column: state.sortBy,
    ranking_basis: "manual_sort",
    sort_order: state.sortOrder,
  };
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
    ...teamPreferenceRequestParams(),
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

  if (hasActiveLens()) {
    rows = applyLensRanking(rows, diagnostics, "predictions");
  }

  state.rows = rows;
  state.total = Number(payload.total) || rows.length;
  state.count = rows.length;
  state.queryDiagnostics = diagnostics;
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
    ...teamPreferenceRequestParams(),
  });

  const filtered = applyClientFilters((payload.items || []).map((r) => withComputedConservativeGap(r)));
  const sorted = hasActiveLens()
    ? applyLensRanking(filtered, payload.diagnostics || null, "shortlist")
    : sortShortlistRows(filtered);
  const page = sorted.slice(state.offset, state.offset + state.limit);

  state.rows = page;
  state.total = sorted.length;
  state.count = page.length;
  state.queryDiagnostics = payload.diagnostics || null;
  renderResultsNote();
}

async function fetchSystemFitPage() {
  await loadSystemFitTemplates();
  const maxBudget = parseOptionalPositive(state.budgetBand);
  const payload = await requestJson("/market-value/system-fit/query", {
    method: "POST",
    params: teamPreferenceRequestParams(),
    body: {
      template_key: state.systemFitTemplate || DEFAULT_SYSTEM_FIT_TEMPLATE,
      split: state.split,
      active_lane: state.systemFitActiveLane || DEFAULT_SYSTEM_FIT_LANE,
      top_n_per_slot: state.limit,
      trust_scope: "trusted_and_watch",
      filters: {
        season: state.season || null,
        include_leagues: state.league ? [state.league] : null,
        exclude_leagues: null,
        min_age: state.minAge < 0 ? null : state.minAge,
        max_age: state.maxAge < 0 ? null : state.maxAge,
        min_minutes: state.minMinutes,
        max_market_value_eur: maxBudget,
        max_contract_years_left: state.maxContractYearsLeft,
        min_confidence: state.minConfidence > 0 ? state.minConfidence : null,
        non_big5_only: state.nonBig5Only,
        budget_eur: maxBudget,
      },
    },
  });

  state.systemFitSlots = Array.isArray(payload.slots) ? payload.slots : [];
  state.systemFitLanePosture = payload.lane_posture || null;
  state.systemFitFiltersApplied = payload.filters_applied || null;
  if (!state.systemFitSlots.some((slot) => String(slot.slot_key || "") === String(state.systemFitSelectedSlot || ""))) {
    state.systemFitSelectedSlot = state.systemFitSlots[0]?.slot_key || "";
  }
  state.queryDiagnostics = {
    score_column: "system_fit_score",
    ranking_basis: "system_fit_slot_rank",
    active_lane: payload.active_lane || state.systemFitActiveLane,
    lane_posture: payload.lane_posture || null,
    template_key: payload.system_profile?.template_key || state.systemFitTemplate,
    trust_scope: payload.trust_scope || "trusted_and_watch",
  };
  renderSystemFitSlots();
  applySystemFitSlotRows();
  renderResultsNote();
}

async function runQuery() {
  readWorkbenchControlsToState();
  localStorage.setItem("scoutml_api_base", state.apiBase);
  localStorage.setItem("scoutml_playstyle_lens", state.playstyle);
  localStorage.setItem("scoutml_role_lens", state.roleLens);
  renderModeAffordances();
  if (!backendReady()) {
    state.rows = [];
    state.topPicks = [];
    state.total = 0;
    state.count = 0;
    state.queryDiagnostics = null;
    el.tbody.innerHTML = `<tr><td colspan="${RESULTS_TABLE_COLSPAN}">Backend artifacts are not ready. Review Platform Readiness for the operational details.</td></tr>`;
    renderResultsNote();
    renderPager();
    renderTopPicks();
    renderBoardHighlights();
    clearDetail();
    return;
  }
  setLoading(true);
  try {
    if (isSystemFitMode()) {
      await fetchSystemFitPage();
    } else if (state.mode === "shortlist") {
      await fetchShortlistPage();
    } else {
      await fetchPredictionsPage();
    }
    if (!isSystemFitMode()) {
      state.topPicks = deriveTopPicks(state.rows);
    }
    renderRows();
    renderSystemFitSlots();
    renderTopPicks();
    renderPager();
    clearDetail();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    el.tbody.innerHTML = `<tr><td colspan="${RESULTS_TABLE_COLSPAN}">${msg}</td></tr>`;
    state.rows = [];
    state.topPicks = [];
    state.total = 0;
    state.count = 0;
    state.queryDiagnostics = null;
    state.systemFitSlots = [];
    state.systemFitLanePosture = null;
    state.systemFitFiltersApplied = null;
    renderResultsNote();
    renderSystemFitSlots();
    renderTopPicks();
    renderBoardHighlights();
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

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function setButtonLoading(button, isLoading, loadingLabel) {
  if (!button) return;
  if (!button.dataset.defaultLabel) {
    button.dataset.defaultLabel = button.textContent || "";
  }
  button.disabled = isLoading;
  button.classList.toggle("is-loading", Boolean(isLoading));
  button.textContent = isLoading ? loadingLabel : button.dataset.defaultLabel;
}

function renderTeamActivity() {
  if (!el.teamActivityList) return;
  const items = Array.isArray(state.teamActivity) ? state.teamActivity : [];
  if (!isTeamAuthenticated()) {
    el.teamActivityList.innerHTML = '<li class="risk-item risk-item--none">Team activity appears when a shared workspace session is active.</li>';
    return;
  }
  if (!items.length) {
    el.teamActivityList.innerHTML = '<li class="risk-item risk-item--none">No shared activity logged yet.</li>';
    return;
  }
  el.teamActivityList.innerHTML = items
    .slice(0, 8)
    .map(
      (item) => `<li class="risk-item"><strong>${escapeHtml(safeText(item.summary || "Team update"))}</strong><span>${escapeHtml(
        safeText(item.actor_name || item.actor_email || "")
      )} | ${escapeHtml(formatDecisionTimestamp(item.created_at_utc))}</span></li>`
    )
    .join("");
}

function renderTeamAssignments() {
  if (!el.teamAssignmentList || !el.teamAssigneeSelect) return;
  const members = Array.isArray(state.teamActiveWorkspace?.members) ? state.teamActiveWorkspace.members : [];
  el.teamAssigneeSelect.innerHTML = [
    '<option value="">Assign to scout</option>',
    ...members.map(
      (member) =>
        `<option value="${escapeHtml(member.user_id)}">${escapeHtml(
          safeText(member.full_name || member.email)
        )} (${escapeHtml(safeText(member.role))})</option>`
    ),
  ].join("");
  const items = Array.isArray(state.teamAssignments) ? state.teamAssignments : [];
  if (!isTeamAuthenticated()) {
    el.teamAssignmentList.innerHTML = '<li class="risk-item risk-item--none">Assignments are available in team mode.</li>';
    return;
  }
  if (!items.length) {
    el.teamAssignmentList.innerHTML = '<li class="risk-item risk-item--none">No assignment for this player yet.</li>';
    return;
  }
  el.teamAssignmentList.innerHTML = items
    .map(
      (item) => `<li class="risk-item"><strong>${escapeHtml(humanizeKey(item.status))}</strong><span>${escapeHtml(
        safeText(item.assignee_name || item.assignee_email || "Unassigned")
      )}${item.due_date ? ` | due ${escapeHtml(formatDecisionTimestamp(item.due_date))}` : ""}</span></li>`
    )
    .join("");
}

function renderTeamComments() {
  if (!el.teamCommentsList) return;
  const items = Array.isArray(state.teamComments) ? state.teamComments : [];
  if (!isTeamAuthenticated()) {
    el.teamCommentsList.innerHTML = '<li class="risk-item risk-item--none">Shared comments are available in team mode.</li>';
    return;
  }
  if (!items.length) {
    el.teamCommentsList.innerHTML = '<li class="risk-item risk-item--none">No shared comments saved yet.</li>';
    return;
  }
  el.teamCommentsList.innerHTML = items
    .map(
      (item) => `<li class="risk-item"><strong>${escapeHtml(safeText(item.author_name || item.author_email || "Scout"))}</strong><span>${escapeHtml(
        formatDecisionTimestamp(item.created_at_utc)
      )}</span><p>${escapeHtml(safeText(item.body))}</p></li>`
    )
    .join("");
}

function renderTeamCompareWorkspace() {
  if (el.teamCompareSection) {
    el.teamCompareSection.hidden = !isTeamAuthenticated();
  }
  if (!el.teamCompareTray || !el.teamCompareLists || !el.teamCompareSelect || !el.teamCompareMeta) return;
  if (!isTeamAuthenticated()) {
    el.teamCompareMeta.textContent = "Shared compare lists appear when a workspace session is active.";
    el.teamCompareTray.innerHTML = '<p class="details-placeholder">Team compare tray unavailable in local mode.</p>';
    el.teamCompareLists.innerHTML = '<p class="details-placeholder">No compare lists loaded.</p>';
    el.teamCompareSelect.innerHTML = '<option value="">Choose saved compare list</option>';
    return;
  }
  const tray = Array.isArray(state.teamCompareTray) ? state.teamCompareTray : [];
  el.teamCompareTray.innerHTML = tray.length
    ? tray
        .map(
          (row) => `<article class="compare-tray-card">
            <strong>${escapeHtml(safeText(row.name || row.player_id))}</strong>
            <span>${escapeHtml(safeText(row.club))} | ${escapeHtml(safeText(row.league))}</span>
            <small>${escapeHtml(formatCurrency(row.market_value_eur))} market | ${escapeHtml(formatCurrency(row.fair_value_eur || row.expected_value_eur))} fair</small>
            <button type="button" class="btn-ghost team-compare-remove" data-player-id="${escapeHtml(safeText(row.player_id))}">Remove</button>
          </article>`
        )
        .join("")
    : '<p class="details-placeholder">Add players from the detail rail to build a compare tray.</p>';
  const lists = Array.isArray(state.teamCompareLists) ? state.teamCompareLists : [];
  el.teamCompareSelect.innerHTML = [
    '<option value="">Choose saved compare list</option>',
    ...lists.map((item) => `<option value="${escapeHtml(item.compare_id)}">${escapeHtml(safeText(item.name))}</option>`),
  ].join("");
  el.teamCompareLists.innerHTML = lists.length
    ? lists
        .map((item) => {
          const players = Array.isArray(item.players) ? item.players : [];
          return `<article class="compare-list-card">
            <div class="compare-list-card__header">
              <strong>${escapeHtml(safeText(item.name))}</strong>
              <span>${escapeHtml(safeText(item.owner_name || ""))}</span>
            </div>
            <p>${escapeHtml(safeText(item.notes || "")) || "No compare notes yet."}</p>
            <div class="compare-list-card__players">
              ${
                players.length
                  ? players
                      .map(
                        (player) => `<div class="compare-list-player">
                          <strong>${escapeHtml(safeText(player.snapshot?.name || player.player_id))}</strong>
                          <span>${escapeHtml(safeText(player.snapshot?.club || ""))} | ${escapeHtml(
                            safeText(player.snapshot?.league || "")
                          )}</span>
                          <small>${escapeHtml(
                            safeText(player.comparison?.trajectory_label || "No trajectory")
                          )} | ${escapeHtml(formatCurrency(player.comparison?.market_value_eur))}</small>
                        </div>`
                      )
                      .join("")
                  : '<p class="details-placeholder">No players saved in this compare list yet.</p>'
              }
            </div>
          </article>`;
        })
        .join("")
    : '<p class="details-placeholder">No compare lists loaded.</p>';
  el.teamCompareMeta.textContent = `${formatInt(tray.length)} in tray | ${formatInt(lists.length)} shared compare list${
    lists.length === 1 ? "" : "s"
  }.`;
}

function renderTeamPreferenceControls() {
  if (el.teamPreferencesPanel) {
    el.teamPreferencesPanel.hidden = !isTeamAuthenticated();
  }
  if (!el.teamPrefMeta) return;
  if (!isTeamAuthenticated()) {
    el.teamPrefMeta.textContent = "Scout preference profiles are available in team mode.";
    return;
  }
  const profile = state.teamPreferenceProfile || {};
  if (el.teamPrefName) el.teamPrefName.value = safeText(profile.name || "Primary");
  if (el.teamPrefAgeMin) el.teamPrefAgeMin.value = profile.target_age_min ?? "";
  if (el.teamPrefAgeMax) el.teamPrefAgeMax.value = profile.target_age_max ?? "";
  if (el.teamPrefBudgetPosture) el.teamPrefBudgetPosture.value = safeText(profile.budget_posture || "balanced");
  if (el.teamPrefTrustPosture) el.teamPrefTrustPosture.value = safeText(profile.trusted_league_posture || "balanced");
  if (el.teamPrefRisk) el.teamPrefRisk.value = safeText(profile.risk_tolerance || "balanced");
  if (el.teamPrefLane) el.teamPrefLane.value = safeText(profile.active_lane_preference || "valuation");
  if (el.teamPrefSystemTemplate) el.teamPrefSystemTemplate.value = safeText(profile.system_template_default || "");
  if (el.teamPrefRolePriorities) {
    const rolePriorities = profile.role_priorities && typeof profile.role_priorities === "object" ? profile.role_priorities : {};
    el.teamPrefRolePriorities.value = Object.entries(rolePriorities)
      .map(([key, value]) => `${key}:${value}`)
      .join(", ");
  }
  if (el.teamPrefMustHaveTags) el.teamPrefMustHaveTags.value = Array.isArray(profile.must_have_tags) ? profile.must_have_tags.join(", ") : "";
  if (el.teamPrefAvoidTags) el.teamPrefAvoidTags.value = Array.isArray(profile.avoid_tags) ? profile.avoid_tags.join(", ") : "";
  if (el.teamPrefApply) el.teamPrefApply.checked = Boolean(state.teamApplyPreferences);
  el.teamPrefMeta.textContent = `Preference-aware reranking is ${state.teamApplyPreferences ? "active" : "off"} for this workspace session.`;
}

function renderTeamSessionState() {
  if (el.teamStatus) {
    if (!state.teamEnabled) {
      el.teamStatus.className = "status status--neutral";
      el.teamStatus.textContent = "Local mode";
    } else if (!state.teamAuthenticated) {
      el.teamStatus.className = "status status--warn";
      el.teamStatus.textContent = "Team ready";
    } else {
      el.teamStatus.className = "status status--ok";
      el.teamStatus.textContent = safeText(state.teamActiveWorkspace?.name || "Workspace active");
    }
  }
  if (el.teamAuthMeta) {
    el.teamAuthMeta.textContent = !state.teamEnabled
      ? "Team mode is off until the backend is configured with a database."
      : state.teamAuthenticated
      ? "Shared workspace state is live. Decisions, watchlist, assignments, comments, and compare lists are now team-scoped."
      : "Team mode is available. Login, bootstrap an admin, or accept an invite to enter the shared workspace.";
  }
  if (el.teamWorkspaceBanner) {
    el.teamWorkspaceBanner.hidden = !state.teamAuthenticated;
  }
  if (el.teamCurrentWorkspace) {
    el.teamCurrentWorkspace.textContent = safeText(state.teamActiveWorkspace?.name || "No workspace selected");
  }
  if (el.teamCurrentUser) {
    el.teamCurrentUser.textContent = safeText(state.teamUser?.full_name || state.teamUser?.email || "Not signed in");
  }
  if (el.teamWorkspaceSelect) {
    const workspaces = Array.isArray(state.teamWorkspaces) ? state.teamWorkspaces : [];
    el.teamWorkspaceSelect.innerHTML = [
      '<option value="">No workspace loaded</option>',
      ...workspaces.map(
        (workspace) =>
          `<option value="${escapeHtml(workspace.workspace_id)}"${workspace.workspace_id === activeTeamWorkspaceId() ? " selected" : ""}>${escapeHtml(
            safeText(workspace.name)
          )} (${escapeHtml(safeText(workspace.role))})</option>`
      ),
    ].join("");
  }
  if (el.teamCollaborationSection) {
    el.teamCollaborationSection.hidden = !isTeamAuthenticated();
  }
  renderTeamAssignments();
  renderTeamComments();
  renderTeamActivity();
  renderTeamCompareWorkspace();
  renderTeamPreferenceControls();
}

async function refreshTeamPreferences() {
  if (!isTeamAuthenticated()) {
    state.teamPreferenceProfile = null;
    state.teamApplyPreferences = true;
    renderTeamPreferenceControls();
    return;
  }
  try {
    state.teamPreferenceProfile = await getJson("/team/preferences/me");
    state.teamApplyPreferences = state.teamPreferenceProfile?.apply_by_default !== false;
  } catch {
    state.teamPreferenceProfile = null;
  }
  renderTeamPreferenceControls();
}

async function refreshTeamCompareLists() {
  if (!isTeamAuthenticated()) {
    state.teamCompareLists = [];
    renderTeamCompareWorkspace();
    return;
  }
  try {
    const payload = await getJson("/team/compare-lists");
    state.teamCompareLists = Array.isArray(payload.items) ? payload.items : [];
  } catch {
    state.teamCompareLists = [];
  }
  renderTeamCompareWorkspace();
}

async function refreshTeamActivity() {
  if (!isTeamAuthenticated()) {
    state.teamActivity = [];
    renderTeamActivity();
    return;
  }
  try {
    const payload = await getJson("/team/activity", { limit: 8 });
    state.teamActivity = Array.isArray(payload.items) ? payload.items : [];
  } catch {
    state.teamActivity = [];
  }
  renderTeamActivity();
}

async function loadTeamCollaborationForSelectedRow() {
  if (!isTeamAuthenticated() || !state.selectedRow) {
    state.teamAssignments = [];
    state.teamComments = [];
    renderTeamAssignments();
    renderTeamComments();
    return;
  }
  const playerId = String(state.selectedRow.player_id || "").trim();
  const season = String(state.selectedRow.season || state.season || "").trim();
  try {
    const [assignmentsPayload, commentsPayload] = await Promise.all([
      getJson("/team/assignments", { player_id: playerId, limit: 20 }),
      getJson(`/team/player/${encodeURIComponent(playerId)}/comments`, {
        split: state.split,
        season: season || null,
        limit: 20,
      }),
    ]);
    state.teamAssignments = Array.isArray(assignmentsPayload.items) ? assignmentsPayload.items : [];
    state.teamComments = Array.isArray(commentsPayload.items) ? commentsPayload.items : [];
  } catch (err) {
    state.teamAssignments = [];
    state.teamComments = [];
    if (el.teamCommentMeta) {
      el.teamCommentMeta.textContent = err instanceof Error ? err.message : String(err);
    }
  }
  renderTeamAssignments();
  renderTeamComments();
}

async function refreshTeamSession() {
  const inviteToken = new URLSearchParams(window.location.search).get("invite_token") || "";
  if (el.teamInviteToken && !el.teamInviteToken.value && inviteToken) {
    el.teamInviteToken.value = inviteToken;
  }
  try {
    const preferredWorkspaceId = localStorage.getItem("scoutml_team_workspace_id") || "";
    const payload = await getJson("/auth/me", preferredWorkspaceId ? { workspace_id: preferredWorkspaceId } : {});
    state.teamEnabled = Boolean(payload.team_mode);
    state.teamAuthenticated = Boolean(payload.authenticated);
    state.teamUser = payload.user || null;
    state.teamWorkspaces = Array.isArray(payload.workspaces) ? payload.workspaces : [];
    state.teamActiveWorkspace = payload.active_workspace || null;
    if (state.teamAuthenticated && state.teamActiveWorkspace?.workspace_id) {
      localStorage.setItem("scoutml_team_workspace_id", state.teamActiveWorkspace.workspace_id);
      await Promise.allSettled([refreshTeamPreferences(), refreshTeamCompareLists(), refreshTeamActivity()]);
      if (state.selectedRow) {
        void loadTeamCollaborationForSelectedRow();
      }
    } else {
      localStorage.removeItem("scoutml_team_workspace_id");
      state.teamCompareLists = [];
      state.teamPreferenceProfile = null;
      state.teamAssignments = [];
      state.teamComments = [];
      state.teamActivity = [];
    }
  } catch (err) {
    localStorage.removeItem("scoutml_team_workspace_id");
    state.teamEnabled = false;
    state.teamAuthenticated = false;
    state.teamUser = null;
    state.teamWorkspaces = [];
    state.teamActiveWorkspace = null;
    state.teamCompareLists = [];
    state.teamPreferenceProfile = null;
    state.teamAssignments = [];
    state.teamComments = [];
    state.teamActivity = [];
    if (el.teamAuthMeta) {
      el.teamAuthMeta.textContent = err instanceof Error ? err.message : String(err);
    }
  }
  renderTeamSessionState();
}

function collectTeamPreferencePayload() {
  return {
    name: el.teamPrefName?.value?.trim() || "Primary",
    target_age_min: el.teamPrefAgeMin?.value ? Number(el.teamPrefAgeMin.value) : null,
    target_age_max: el.teamPrefAgeMax?.value ? Number(el.teamPrefAgeMax.value) : null,
    budget_posture: el.teamPrefBudgetPosture?.value || "balanced",
    trusted_league_posture: el.teamPrefTrustPosture?.value || "balanced",
    role_priorities: parseRolePriorityMap(el.teamPrefRolePriorities?.value || ""),
    system_template_default: el.teamPrefSystemTemplate?.value || "",
    must_have_tags: parseTagList(el.teamPrefMustHaveTags?.value || ""),
    avoid_tags: parseTagList(el.teamPrefAvoidTags?.value || ""),
    risk_tolerance: el.teamPrefRisk?.value || "balanced",
    active_lane_preference: el.teamPrefLane?.value || "valuation",
    apply_by_default: Boolean(el.teamPrefApply?.checked),
  };
}

async function handleTeamLogin() {
  if (!el.teamEmail || !el.teamPassword) return;
  setButtonLoading(el.teamLoginBtn, true, "Logging in...");
  try {
    await requestJson("/auth/login", {
      method: "POST",
      body: {
        email: el.teamEmail.value.trim(),
        password: el.teamPassword.value,
        workspace_id: el.teamWorkspaceSelect?.value || null,
      },
    });
    await refreshTeamSession();
    await refreshWatchlist();
    if (backendReady()) {
      await runQuery();
    }
  } catch (err) {
    if (el.teamAuthMeta) el.teamAuthMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamLoginBtn, false, "Login");
  }
}

async function handleTeamBootstrap() {
  if (!el.teamEmail || !el.teamPassword) return;
  setButtonLoading(el.teamBootstrapBtn, true, "Bootstrapping...");
  try {
    await requestJson("/auth/bootstrap-admin", {
      method: "POST",
      body: {
        email: el.teamEmail.value.trim(),
        password: el.teamPassword.value,
        full_name: el.teamFullName?.value?.trim() || null,
        workspace_name: el.teamWorkspaceName?.value?.trim() || null,
      },
    });
    await refreshTeamSession();
    await refreshWatchlist();
    if (backendReady()) {
      await runQuery();
    }
  } catch (err) {
    if (el.teamAuthMeta) el.teamAuthMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamBootstrapBtn, false, "Bootstrap Admin");
  }
}

async function handleTeamAcceptInvite() {
  if (!el.teamInviteToken || !el.teamEmail || !el.teamPassword) return;
  const token = el.teamInviteToken.value.trim();
  if (!token) {
    if (el.teamAuthMeta) el.teamAuthMeta.textContent = "Paste an invite token first.";
    return;
  }
  setButtonLoading(el.teamAcceptInviteBtn, true, "Accepting...");
  try {
    await requestJson(`/invites/${encodeURIComponent(token)}/accept`, {
      method: "POST",
      body: {
        email: el.teamEmail.value.trim(),
        password: el.teamPassword.value,
        full_name: el.teamFullName?.value?.trim() || null,
      },
    });
    await refreshTeamSession();
    await refreshWatchlist();
    if (backendReady()) {
      await runQuery();
    }
  } catch (err) {
    if (el.teamAuthMeta) el.teamAuthMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamAcceptInviteBtn, false, "Accept Invite");
  }
}

async function handleTeamLogout() {
  setButtonLoading(el.teamLogoutBtn, true, "Logging out...");
  try {
    await requestJson("/auth/logout", { method: "POST" });
    state.teamCompareTray = [];
    await refreshTeamSession();
    await refreshWatchlist();
    if (backendReady()) {
      await runQuery();
    }
  } catch (err) {
    if (el.teamAuthMeta) el.teamAuthMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamLogoutBtn, false, "Logout");
  }
}

async function handleTeamCreateWorkspace() {
  const name = el.teamNewWorkspaceName?.value?.trim() || el.teamWorkspaceName?.value?.trim() || "";
  if (!name) {
    if (el.teamAuthMeta) el.teamAuthMeta.textContent = "Enter a workspace name first.";
    return;
  }
  setButtonLoading(el.teamCreateWorkspaceBtn, true, "Creating...");
  try {
    await requestJson("/workspaces", {
      method: "POST",
      body: { name },
    });
    await refreshTeamSession();
  } catch (err) {
    if (el.teamAuthMeta) el.teamAuthMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamCreateWorkspaceBtn, false, "Create Workspace");
  }
}

async function handleTeamCreateInvite() {
  const workspaceId = activeTeamWorkspaceId();
  if (!workspaceId) {
    if (el.teamInviteOutput) el.teamInviteOutput.textContent = "Choose an active workspace first.";
    return;
  }
  setButtonLoading(el.teamCreateInviteBtn, true, "Creating invite...");
  try {
    const payload = await requestJson(`/workspaces/${encodeURIComponent(workspaceId)}/invites`, {
      method: "POST",
      body: {
        email: el.teamInviteEmail?.value?.trim() || null,
        role: el.teamInviteRole?.value || "scout",
      },
    });
    if (el.teamInviteOutput) {
      el.teamInviteOutput.textContent = payload.invite_url
        ? `${payload.invite_url} | token ${payload.token}`
        : payload.token || "Invite created.";
    }
    await refreshTeamSession();
  } catch (err) {
    if (el.teamInviteOutput) el.teamInviteOutput.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamCreateInviteBtn, false, "Create Invite");
  }
}

async function handleWorkspaceSwitch() {
  if (!el.teamWorkspaceSelect) return;
  const workspaceId = String(el.teamWorkspaceSelect.value || "").trim();
  if (!workspaceId) return;
  localStorage.setItem("scoutml_team_workspace_id", workspaceId);
  await refreshTeamSession();
  if (backendReady()) {
    await refreshWatchlist();
    await runQuery();
  }
}

async function saveTeamPreferences() {
  if (!isTeamAuthenticated()) return;
  setButtonLoading(el.teamPrefSaveBtn, true, "Saving...");
  try {
    state.teamPreferenceProfile = await requestJson("/team/preferences/me", {
      method: "PUT",
      body: collectTeamPreferencePayload(),
    });
    state.teamApplyPreferences = state.teamPreferenceProfile?.apply_by_default !== false;
    renderTeamPreferenceControls();
    if (backendReady()) {
      await runQuery();
    }
  } catch (err) {
    if (el.teamPrefMeta) el.teamPrefMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamPrefSaveBtn, false, "Save Profile");
  }
}

async function saveTeamAssignment() {
  if (!isTeamAuthenticated() || !state.selectedRow) return;
  setButtonLoading(el.teamAssignmentSaveBtn, true, "Saving...");
  try {
    await requestJson("/team/assignments", {
      method: "POST",
      body: {
        player_id: state.selectedRow.player_id,
        split: state.split,
        season: state.selectedRow.season || null,
        assignee_user_id: el.teamAssigneeSelect?.value || null,
        status: el.teamAssignmentStatus?.value || "to_watch",
        due_date: el.teamAssignmentDue?.value || null,
        note: el.teamAssignmentNote?.value?.trim() || null,
      },
    });
    if (el.teamAssignmentMeta) el.teamAssignmentMeta.textContent = "Shared assignment saved.";
    await Promise.allSettled([loadTeamCollaborationForSelectedRow(), refreshTeamActivity()]);
  } catch (err) {
    if (el.teamAssignmentMeta) el.teamAssignmentMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamAssignmentSaveBtn, false, "Save Assignment");
  }
}

async function saveTeamComment() {
  if (!isTeamAuthenticated() || !state.selectedRow || !el.teamCommentInput) return;
  const body = el.teamCommentInput.value.trim();
  if (!body) {
    if (el.teamCommentMeta) el.teamCommentMeta.textContent = "Write a shared note first.";
    return;
  }
  setButtonLoading(el.teamCommentSaveBtn, true, "Saving...");
  try {
    await requestJson(`/team/player/${encodeURIComponent(state.selectedRow.player_id)}/comments`, {
      method: "POST",
      body: {
        split: state.split,
        season: state.selectedRow.season || null,
        body,
      },
    });
    el.teamCommentInput.value = "";
    if (el.teamCommentMeta) el.teamCommentMeta.textContent = "Shared comment saved.";
    await Promise.allSettled([loadTeamCollaborationForSelectedRow(), refreshTeamActivity()]);
  } catch (err) {
    if (el.teamCommentMeta) el.teamCommentMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamCommentSaveBtn, false, "Save Comment");
  }
}

function addSelectedToCompareTray() {
  if (!isTeamAuthenticated() || !state.selectedRow) return;
  const playerId = String(state.selectedRow.player_id || "").trim();
  if (!playerId) return;
  if (state.teamCompareTray.some((row) => String(row.player_id || "").trim() === playerId)) {
    renderTeamCompareWorkspace();
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = "Player already in compare tray.";
    return;
  }
  if (state.teamCompareTray.length >= 4) {
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = "Compare tray supports at most 4 players.";
    return;
  }
  state.teamCompareTray = [...state.teamCompareTray, { ...state.selectedRow }];
  renderTeamCompareWorkspace();
}

function removePlayerFromCompareTray(playerId) {
  state.teamCompareTray = state.teamCompareTray.filter((row) => String(row.player_id || "").trim() !== String(playerId || "").trim());
  renderTeamCompareWorkspace();
}

async function createTeamCompareListFromInput() {
  if (!isTeamAuthenticated() || !el.teamCompareName) return;
  const name = el.teamCompareName.value.trim();
  if (!name) {
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = "Name the compare list first.";
    return;
  }
  setButtonLoading(el.teamCompareCreateBtn, true, "Creating...");
  try {
    await requestJson("/team/compare-lists", {
      method: "POST",
      body: { name, notes: "" },
    });
    await refreshTeamCompareLists();
  } catch (err) {
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamCompareCreateBtn, false, "Create Compare List");
  }
}

async function saveCompareTrayToList() {
  if (!isTeamAuthenticated() || !el.teamCompareSelect) return;
  const compareId = String(el.teamCompareSelect.value || "").trim();
  if (!compareId) {
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = "Choose a compare list first.";
    return;
  }
  if (state.teamCompareTray.length < 2) {
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = "Add at least 2 players to the compare tray first.";
    return;
  }
  setButtonLoading(el.teamCompareSaveBtn, true, "Saving...");
  try {
    await Promise.all(
      state.teamCompareTray.map((row) =>
        requestJson(`/team/compare-lists/${encodeURIComponent(compareId)}/players`, {
          method: "POST",
          body: {
            player_id: row.player_id,
            split: state.split,
            season: row.season || null,
            pinned: false,
            notes: "",
          },
        })
      )
    );
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = "Compare tray saved to the shared compare list.";
    await Promise.allSettled([refreshTeamCompareLists(), refreshTeamActivity()]);
  } catch (err) {
    if (el.teamCompareMeta) el.teamCompareMeta.textContent = err instanceof Error ? err.message : String(err);
  } finally {
    setButtonLoading(el.teamCompareSaveBtn, false, "Save Tray to Selected List");
  }
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

async function fetchPlayerSimilarForRow(row) {
  const playerId = String(row?.player_id || "").trim();
  if (!playerId) {
    throw new Error("Selected row has no player_id.");
  }
  const season = String(row?.season || state.season || "").trim();
  const key = `${reportCacheKey(row)}|similar`;
  if (state.similarCache.has(key)) {
    return state.similarCache.get(key);
  }
  const payload = await getJson(`/market-value/player/${encodeURIComponent(playerId)}/similar`, {
    split: state.split,
    season: season || null,
    n: 5,
    same_position: true,
    exclude_big5: false,
  });
  const similarPayload = {
    player_id: payload.player_id || playerId,
    available: true,
    reason: null,
    position_group: payload.position_group || null,
    feature_count_used: payload.feature_count_used,
    feature_columns_used: Array.isArray(payload.feature_columns_used) ? payload.feature_columns_used : [],
    items: Array.isArray(payload.comparisons) ? payload.comparisons : [],
  };
  state.similarCache.set(key, similarPayload);
  return similarPayload;
}

async function fetchPlayerTrajectoryForRow(row) {
  const playerId = String(row?.player_id || "").trim();
  if (!playerId) {
    throw new Error("Selected row has no player_id.");
  }
  const season = String(row?.season || state.season || "").trim();
  const key = `${reportCacheKey(row)}|trajectory`;
  if (state.trajectoryCache.has(key)) {
    return state.trajectoryCache.get(key);
  }
  const payload = await getJson(`/market-value/player/${encodeURIComponent(playerId)}/trajectory`, {
    split: state.split,
    season: season || null,
  });
  state.trajectoryCache.set(key, payload || null);
  return payload || null;
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

function embeddedSimilarPayloadFromProfile(profile = null) {
  if (!profile?.similar_players || typeof profile.similar_players !== "object") {
    return null;
  }
  return profile.similar_players;
}

function cachedTrajectoryForRow(row) {
  const key = `${reportCacheKey(row)}|trajectory`;
  if (!state.trajectoryCache.has(key)) {
    return { available: false, payload: null };
  }
  return { available: true, payload: state.trajectoryCache.get(key) || null };
}

async function ensureSelectedSimilarLoaded() {
  if (!state.selectedRow) return;
  if (state.selectedSimilar || embeddedSimilarPayloadFromProfile(state.selectedProfile)) return;
  const requestId = state.detailRequestId;
  const row = state.selectedRow;
  renderSimilarPlayers(null, { loading: true });
  try {
    const payload = await fetchPlayerSimilarForRow(row);
    if (requestId !== state.detailRequestId || !state.selectedRow || rowKey(state.selectedRow) !== rowKey(row)) return;
    state.selectedSimilar = payload || null;
    renderSimilarPlayers(payload);
    if (state.profileModalOpen) {
      renderProfileModal(state.selectedRow, { profile: state.selectedProfile, trajectory: state.selectedTrajectory });
    }
  } catch (err) {
    if (requestId !== state.detailRequestId || !state.selectedRow || rowKey(state.selectedRow) !== rowKey(row)) return;
    const message = err instanceof Error ? err.message : String(err);
    renderSimilarPlayers(null, { error: message });
  }
}

async function ensureSelectedTrajectoryLoaded() {
  if (!state.selectedRow) return;
  const cached = cachedTrajectoryForRow(state.selectedRow);
  if (cached.available) {
    state.selectedTrajectory = cached.payload;
    renderTrajectoryPanel(cached.payload);
    return;
  }
  const requestId = state.detailRequestId;
  const row = state.selectedRow;
  renderTrajectoryPanel(null, { loading: true });
  try {
    const trajectory = await fetchPlayerTrajectoryForRow(row);
    if (requestId !== state.detailRequestId || !state.selectedRow || rowKey(state.selectedRow) !== rowKey(row)) return;
    state.selectedTrajectory = trajectory || null;
    renderTrajectoryPanel(trajectory);
    if (state.profileModalOpen) {
      renderProfileModal(state.selectedRow, { profile: state.selectedProfile, trajectory: trajectory || null });
    }
  } catch (err) {
    if (requestId !== state.detailRequestId || !state.selectedRow || rowKey(state.selectedRow) !== rowKey(row)) return;
    const message = err instanceof Error ? err.message : String(err);
    renderTrajectoryPanel(null, { error: message });
  }
}

async function loadDetailWithReport(row) {
  if (!row) return clearDetail();
  const requestId = ++state.detailRequestId;
  resetDecisionDraft();
  setDetailTab("overview");
  renderDetail(row, { reportLoading: true, trajectoryDeferred: true });
  if (isTeamAuthenticated()) {
    void loadTeamCollaborationForSelectedRow();
  }
  try {
    const profile = await fetchPlayerProfileForRow(row);
    if (requestId !== state.detailRequestId) return;
    if (!profile) {
      const msg = "Player brief unavailable.";
      renderDetail(row, { reportError: msg });
      return;
    }
    const similarPayload = embeddedSimilarPayloadFromProfile(profile);
    const cachedTrajectory = cachedTrajectoryForRow(row);
    renderDetail(row, {
      profile,
      similarPayload,
      trajectory: cachedTrajectory.available ? cachedTrajectory.payload : null,
      trajectoryDeferred: !cachedTrajectory.available,
    });
    if (isTeamAuthenticated()) {
      void loadTeamCollaborationForSelectedRow();
    }
  } catch (err) {
    if (requestId !== state.detailRequestId) return;
    const msg = err instanceof Error ? err.message : String(err);
    renderDetail(row, { reportError: msg, trajectoryDeferred: true });
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
    trajectory: state.selectedTrajectory || null,
    player_type: report?.player_type || null,
    formation_fit: report?.formation_fit || null,
    radar_profile: report?.radar_profile || null,
    stat_groups: Array.isArray(report?.stat_groups) ? report.stat_groups : [],
    similar_players: state.selectedSimilar || report?.similar_players || null,
    proxy_estimates: report?.proxy_estimates || null,
    latest_decision: report?.latest_decision || state.selectedLatestDecision || null,
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

async function openSimilarPlayerReference(playerId, season = "") {
  const row = {
    player_id: String(playerId || "").trim(),
    season: String(season || "").trim(),
  };
  if (!row.player_id) return;
  setView("workbench");
  try {
    const profile = await fetchPlayerProfileForRow(row);
    const nextRow = profile?.player && typeof profile.player === "object" ? profile.player : row;
    await loadDetailWithReport(nextRow);
  } catch {
    await loadDetailWithReport(row);
  }
}

async function downloadPlayerMemoPdf(row, button) {
  const playerId = String(row?.player_id || "").trim();
  if (!playerId) return;
  const season = String(row?.season || state.season || "").trim();
  setButtonLoading(button, true, "Generating...");
  try {
    const payload = await getBlob(`/market-value/player/${encodeURIComponent(playerId)}/memo.pdf`, {
      split: state.split,
      season: season || null,
      include_trajectory: true,
      include_similar: true,
    });
    const fallbackName = `scoutml_player_memo_${sanitizeFileToken(playerId)}_${sanitizeFileToken(state.split, "split")}.pdf`;
    downloadBlob(payload.blob, payload.filename || fallbackName);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setStatus("error", message || "Failed to generate player memo PDF.");
  } finally {
    setButtonLoading(button, false, "Generating...");
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
      ...teamPreferenceRequestParams(),
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

function clearBackendState() {
  state.connected = false;
  state.health = null;
  state.metrics = null;
  state.modelManifest = null;
  state.benchmark = null;
  state.activeArtifacts = null;
  state.operatorHealth = null;
  state.coverageRows = [];
  state.systemFitTemplates = [];
  state.systemFitSlots = [];
  state.systemFitLanePosture = null;
  state.systemFitFiltersApplied = null;
}

async function loadHealthAndMetrics() {
  state.reportCache = new Map();
  state.profileCache = new Map();
  state.selectedProfile = null;
  state.selectedReport = null;
  state.selectedHistory = null;
  state.selectedSimilar = null;
  state.selectedTrajectory = null;
  state.metrics = null;
  state.modelManifest = null;
  state.benchmark = null;
  state.activeArtifacts = null;
  state.operatorHealth = null;
  state.systemFitTemplates = [];
  state.connected = false;
  state.health = await getJson("/market-value/health");
  state.connected = true;
  if (state.health?.status !== "ok") {
    setStatus("error", "Artifacts missing");
    renderTrustCard();
    renderMetrics();
    renderSegmentTable();
    renderBenchmarkCards();
    renderOverviewReadiness();
    return false;
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

async function loadDeferredWorkbenchMetadata() {
  if (!backendReady()) return;
  const tasks = [
    async () => {
      try {
        state.metrics = (await getJson("/market-value/metrics")).payload || null;
      } catch {
        state.metrics = null;
      }
      renderMetrics();
      renderSegmentTable();
      renderOverviewReadiness();
    },
    async () => {
      try {
        state.modelManifest = (await getJson("/market-value/model-manifest")).payload || null;
      } catch {
        state.modelManifest = null;
      }
      renderTrustCard();
      renderOverviewReadiness();
    },
    async () => {
      try {
        state.benchmark = (await getJson("/market-value/benchmarks")).payload || null;
      } catch {
        state.benchmark = null;
      }
      renderBenchmarkCards();
      renderCoverageTable();
      renderOverviewReadiness();
    },
    async () => {
      try {
        state.activeArtifacts = (await getJson("/market-value/active-artifacts")).payload || null;
      } catch {
        state.activeArtifacts = null;
      }
      renderTrustCard();
    },
    async () => {
      try {
        state.operatorHealth = (await getJson("/market-value/operator-health")).payload || null;
      } catch {
        state.operatorHealth = null;
      }
      renderOverviewReadiness();
    },
  ];
  await Promise.allSettled(tasks.map((task) => task()));
}

async function connectWorkbench() {
  readWorkbenchControlsToState();
  localStorage.setItem("scoutml_api_base", state.apiBase);
  state.initializing = true;
  renderTopPicks();
  renderBoardHighlights();
  try {
    const ready = await loadHealthAndMetrics();
    if (!ready) {
      return false;
    }
    await refreshTeamSession();
    await refreshCoverageAndOptions();
    await runQuery();
    void refreshWatchlist();
    void loadDeferredWorkbenchMetadata();
    return true;
  } catch (err) {
    clearBackendState();
    setStatus("error", err instanceof Error ? err.message : String(err));
    renderTrustCard();
    renderMetrics();
    renderSegmentTable();
    renderBenchmarkCards();
    renderOverviewReadiness();
    return false;
  } finally {
    state.initializing = false;
    renderTopPicks();
    renderBoardHighlights();
  }
}

function resetWorkbenchControls() {
  state.mode = "shortlist";
  state.split = "test";
  state.systemFitTemplate = DEFAULT_SYSTEM_FIT_TEMPLATE;
  state.systemFitActiveLane = DEFAULT_SYSTEM_FIT_LANE;
  state.systemFitSelectedSlot = "";
  state.systemFitSlots = [];
  state.systemFitLanePosture = null;
  state.systemFitFiltersApplied = null;
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
  if (el.heroExploreBtn) {
    el.heroExploreBtn.addEventListener("click", async () => {
      setView("workbench");
      scrollToBoard();
      if (!state.rows.length && backendReady() && !state.loading) {
        state.offset = 0;
        await runQuery();
      }
    });
  }
  el.detailTabButtons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      await activateDetailTab(btn.dataset.detailTab || "overview");
    });
  });

  el.connectBtn.addEventListener("click", async () => {
    await connectWorkbench();
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

  if (el.playstyle) {
    el.playstyle.addEventListener("change", async () => {
      state.playstyle = el.playstyle.value;
      localStorage.setItem("scoutml_playstyle_lens", state.playstyle);
      state.offset = 0;
      await runQuery();
    });
  }

  if (el.roleLens) {
    el.roleLens.addEventListener("change", async () => {
      state.roleLens = el.roleLens.value;
      localStorage.setItem("scoutml_role_lens", state.roleLens);
      state.offset = 0;
      await runQuery();
    });
  }

  [
    el.mode,
    el.systemFitTemplate,
    el.systemFitActiveLane,
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

  if (el.systemFitSlotBar) {
    el.systemFitSlotBar.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-slot-key]");
      if (!button) return;
      const nextSlot = String(button.dataset.slotKey || "").trim();
      if (!nextSlot || nextSlot === state.systemFitSelectedSlot) return;
      state.systemFitSelectedSlot = nextSlot;
      renderModeAffordances();
      applySystemFitSlotRows();
      renderRows();
      renderTopPicks();
      renderPager();
      clearDetail();
    });
  }

  el.exportBtn.addEventListener("click", () => {
    const diagnostics = state.queryDiagnostics || {};
    const rows = state.rows.map((row, idx) =>
      buildRecruitmentExportRow(row, {
        source: `workbench_${state.mode}`,
        split: state.split,
        rank: state.offset + idx + 1,
        rankingDriver: diagnostics.score_column || state.sortBy,
        rankingBasis:
          diagnostics.ranking_basis ||
          (isSystemFitMode() ? "system_fit_slot_rank" : state.mode === "shortlist" ? "shortlist" : "manual_sort"),
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
          systemTemplate: isSystemFitMode() ? state.systemFitTemplate : null,
          systemSlot: isSystemFitMode() ? state.systemFitSelectedSlot : null,
          systemLane: isSystemFitMode() ? state.systemFitActiveLane : null,
        },
        diagnostics,
        rankingDriver: diagnostics.score_column || state.sortBy,
        rankingBasis:
          diagnostics.ranking_basis ||
          (isSystemFitMode() ? "system_fit_slot_rank" : state.mode === "shortlist" ? "shortlist" : "manual_sort"),
      });
      downloadJson(pack, `scoutml_window_pack_${state.mode}_${state.split}.json`);
    });
  }

  if (Array.isArray(el.detailDecisionActionButtons)) {
    el.detailDecisionActionButtons.forEach((button) => {
      button.addEventListener("click", () => {
        setDecisionDraftAction(button.dataset.scoutAction || "");
      });
    });
  }
  if (el.detailDecisionReasons) {
    el.detailDecisionReasons.addEventListener("click", (event) => {
      const button = event.target.closest("[data-decision-reason]");
      if (!button) return;
      toggleDecisionReason(button.dataset.decisionReason || "");
    });
  }
  if (el.detailDecisionNoteInput) {
    el.detailDecisionNoteInput.addEventListener("input", () => {
      state.decisionDraftNote = el.detailDecisionNoteInput.value || "";
    });
  }
  if (el.detailDecisionSaveBtn) {
    el.detailDecisionSaveBtn.addEventListener("click", async () => {
      await saveScoutDecision();
    });
  }
  if (el.detailDecisionClearBtn) {
    el.detailDecisionClearBtn.addEventListener("click", () => {
      resetDecisionDraft();
      renderScoutDecisionComposer("Decision draft cleared.");
    });
  }
  if (el.teamLoginBtn) {
    el.teamLoginBtn.addEventListener("click", handleTeamLogin);
  }
  if (el.teamBootstrapBtn) {
    el.teamBootstrapBtn.addEventListener("click", handleTeamBootstrap);
  }
  if (el.teamAcceptInviteBtn) {
    el.teamAcceptInviteBtn.addEventListener("click", handleTeamAcceptInvite);
  }
  if (el.teamLogoutBtn) {
    el.teamLogoutBtn.addEventListener("click", handleTeamLogout);
  }
  if (el.teamCreateWorkspaceBtn) {
    el.teamCreateWorkspaceBtn.addEventListener("click", handleTeamCreateWorkspace);
  }
  if (el.teamCreateInviteBtn) {
    el.teamCreateInviteBtn.addEventListener("click", handleTeamCreateInvite);
  }
  if (el.teamWorkspaceSelect) {
    el.teamWorkspaceSelect.addEventListener("change", async () => {
      await handleWorkspaceSwitch();
    });
  }
  if (el.teamPrefApply) {
    el.teamPrefApply.addEventListener("change", async () => {
      state.teamApplyPreferences = Boolean(el.teamPrefApply.checked);
      renderTeamPreferenceControls();
      if (backendReady()) {
        state.offset = 0;
        await runQuery();
      }
    });
  }
  if (el.teamPrefSaveBtn) {
    el.teamPrefSaveBtn.addEventListener("click", async () => {
      await saveTeamPreferences();
    });
  }
  if (el.teamAssignmentSaveBtn) {
    el.teamAssignmentSaveBtn.addEventListener("click", async () => {
      await saveTeamAssignment();
    });
  }
  if (el.teamCommentSaveBtn) {
    el.teamCommentSaveBtn.addEventListener("click", async () => {
      await saveTeamComment();
    });
  }
  if (el.teamCompareAddBtn) {
    el.teamCompareAddBtn.addEventListener("click", addSelectedToCompareTray);
  }
  if (el.teamCompareCreateBtn) {
    el.teamCompareCreateBtn.addEventListener("click", async () => {
      await createTeamCompareListFromInput();
    });
  }
  if (el.teamCompareRefreshBtn) {
    el.teamCompareRefreshBtn.addEventListener("click", async () => {
      await Promise.allSettled([refreshTeamCompareLists(), refreshTeamActivity()]);
    });
  }
  if (el.teamCompareSaveBtn) {
    el.teamCompareSaveBtn.addEventListener("click", async () => {
      await saveCompareTrayToList();
    });
  }
  if (el.teamCompareTray) {
    el.teamCompareTray.addEventListener("click", (event) => {
      const button = event.target.closest(".team-compare-remove");
      if (!button) return;
      removePlayerFromCompareTray(button.dataset.playerId);
    });
  }

  el.tbody.addEventListener("click", async (event) => {
    const tr = event.target.closest("tr");
    if (!tr || !tr.dataset.index) return;
    const idx = Number(tr.dataset.index);
    if (!Number.isFinite(idx)) return;
    await loadDetailWithReport(state.rows[idx]);
  });
  if (el.topPicksGrid) {
    el.topPicksGrid.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-index]");
      if (!button) return;
      const idx = Number(button.dataset.index);
      if (!Number.isFinite(idx)) return;
      const row = state.topPicks[idx];
      if (!row) return;
      setView("workbench");
      await loadDetailWithReport(row);
    });
  }

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

  if (el.detailSimilar) {
    el.detailSimilar.addEventListener("click", async (event) => {
      const trigger = event.target.closest("[data-similar-player-id]");
      if (!trigger) return;
      await openSimilarPlayerReference(trigger.dataset.similarPlayerId, trigger.dataset.similarPlayerSeason);
    });
  }

  if (el.profileModalBody) {
    el.profileModalBody.addEventListener("click", async (event) => {
      const trigger = event.target.closest("[data-similar-player-id]");
      if (!trigger) return;
      await openSimilarPlayerReference(trigger.dataset.similarPlayerId, trigger.dataset.similarPlayerSeason);
    });
  }

  if (el.detailExportPdf) {
    el.detailExportPdf.addEventListener("click", async () => {
      if (!state.selectedRow) return;
      await downloadPlayerMemoPdf(state.selectedRow, el.detailExportPdf);
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

  if (el.profileModalExportPdf) {
    el.profileModalExportPdf.addEventListener("click", async () => {
      if (!state.selectedRow) return;
      await downloadPlayerMemoPdf(state.selectedRow, el.profileModalExportPdf);
    });
  }
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
      const decisionBtn = event.target.closest(".watchlist-open-decision");
      if (decisionBtn) {
        const playerId = String(decisionBtn.dataset.playerId || "").trim();
        const season = String(decisionBtn.dataset.season || "").trim();
        const row = state.watchlistRows.find(
          (item) => String(item.player_id || "").trim() === playerId && String(item.season || "").trim() === season
        );
        if (!row) return;
        await openWatchlistDecision(row);
        return;
      }
      const deleteBtn = event.target.closest(".watchlist-delete");
      if (!deleteBtn) return;
      await deleteWatchlistItem(deleteBtn.dataset.watchId);
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
  renderTeamSessionState();
  renderModeAffordances();
  renderDetailTab();
  renderTopPicks();
  renderBoardHighlights();
  bindEvents();

  try {
    await connectWorkbench();
  } catch {
    clearBackendState();
    el.tbody.innerHTML = `<tr><td colspan="${RESULTS_TABLE_COLSPAN}">Connect API to start the recruitment board. Expected backend: ${state.apiBase}</td></tr>`;
    el.funnelMeta.textContent = "Connect API before building the recruitment funnel.";
    if (el.watchlistMeta) {
      el.watchlistMeta.textContent = "Connect API before using the recruitment watchlist.";
    }
    renderTrustCard();
    renderMetrics();
    renderSegmentTable();
    renderBenchmarkCards();
    renderOverviewReadiness();
  }
}

document.addEventListener("DOMContentLoaded", boot);
