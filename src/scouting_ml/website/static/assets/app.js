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

  mode: "predictions",
  split: "test",
  season: "",
  league: "",
  position: "",
  search: "",
  minMinutes: 900,
  maxAge: 23,
  minConfidence: 0.5,
  minGapEur: 1_000_000,
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
  selectedReport: null,
  selectedHistory: null,
  detailRequestId: 0,
  reportCache: new Map(),

  health: null,
  metrics: null,
  activeArtifacts: null,
  coverageRows: [],

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

  trustModelVersion: document.getElementById("trust-model-version"),
  trustUpdated: document.getElementById("trust-updated"),
  trustDataset: document.getElementById("trust-dataset"),
  trustSplits: document.getElementById("trust-splits"),
  trustRows: document.getElementById("trust-rows"),
  artifactTest: document.getElementById("artifact-test"),
  artifactVal: document.getElementById("artifact-val"),
  artifactMetrics: document.getElementById("artifact-metrics"),
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

  mode: document.getElementById("mode-select"),
  split: document.getElementById("split-select"),
  season: document.getElementById("season-select"),
  league: document.getElementById("league-select"),
  position: document.getElementById("position-select"),
  search: document.getElementById("search-input"),
  minMinutes: document.getElementById("min-minutes"),
  maxAge: document.getElementById("max-age"),
  minConfidence: document.getElementById("min-confidence"),
  minGap: document.getElementById("min-gap"),
  topN: document.getElementById("top-n"),
  sort: document.getElementById("sort-select"),
  sortDir: document.getElementById("sort-direction"),
  limit: document.getElementById("limit-input"),
  undervaluedOnly: document.getElementById("undervalued-only"),
  refresh: document.getElementById("refresh-btn"),
  reset: document.getElementById("reset-btn"),
  exportBtn: document.getElementById("export-btn"),

  title: document.getElementById("results-title"),
  resultCount: document.getElementById("result-count"),
  resultRange: document.getElementById("result-range"),
  tbody: document.getElementById("results-body"),
  prevBtn: document.getElementById("prev-btn"),
  nextBtn: document.getElementById("next-btn"),

  detailPlaceholder: document.getElementById("detail-placeholder"),
  detailContent: document.getElementById("detail-content"),
  detailName: document.getElementById("detail-name"),
  detailMeta: document.getElementById("detail-meta"),
  detailMarket: document.getElementById("detail-market"),
  detailExpected: document.getElementById("detail-expected"),
  detailLower: document.getElementById("detail-lower"),
  detailList: document.getElementById("detail-list"),
  detailSummary: document.getElementById("detail-summary"),
  detailStrengths: document.getElementById("detail-strengths"),
  detailLevers: document.getElementById("detail-levers"),
  detailHistory: document.getElementById("detail-history"),
  detailRisks: document.getElementById("detail-risks"),
  detailExportJson: document.getElementById("detail-export-json"),
  detailExportCsv: document.getElementById("detail-export-csv"),
  watchlistTag: document.getElementById("watchlist-tag"),
  watchlistNotes: document.getElementById("watchlist-notes"),
  watchlistAddBtn: document.getElementById("watchlist-add-btn"),
  watchlistRefreshBtn: document.getElementById("watchlist-refresh-btn"),
  watchlistExportBtn: document.getElementById("watchlist-export-btn"),
  watchlistMeta: document.getElementById("watchlist-meta"),
  watchlistBody: document.getElementById("watchlist-body"),
  barMarket: document.getElementById("bar-market"),
  barExpected: document.getElementById("bar-expected"),
  barLower: document.getElementById("bar-lower"),

  funnelSplit: document.getElementById("funnel-split"),
  funnelMaxAge: document.getElementById("funnel-max-age"),
  funnelMinMinutes: document.getElementById("funnel-min-minutes"),
  funnelMinConfidence: document.getElementById("funnel-min-confidence"),
  funnelMinGap: document.getElementById("funnel-min-gap"),
  funnelTopN: document.getElementById("funnel-top-n"),
  funnelLowerOnly: document.getElementById("funnel-lower-only"),
  funnelRunBtn: document.getElementById("funnel-run-btn"),
  funnelExportBtn: document.getElementById("funnel-export-btn"),
  funnelMeta: document.getElementById("funnel-meta"),
  funnelBody: document.getElementById("funnel-body"),
  funnelLeagueBody: document.getElementById("funnel-league-body"),
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

function safeText(value) {
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
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
    el.tbody.innerHTML = "<tr><td colspan=\"10\">Loading data...</td></tr>";
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
  el.search.value = state.search;
  el.minMinutes.value = String(state.minMinutes);
  el.maxAge.value = String(state.maxAge);
  el.minConfidence.value = String(state.minConfidence);
  el.minGap.value = String(state.minGapEur);
  el.topN.value = String(state.shortlistTopN);
  el.sort.value = state.sortBy;
  el.sortDir.value = state.sortOrder;
  el.limit.value = String(state.limit);
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
  state.search = el.search.value.trim();
  state.minMinutes = Math.max(parseNumberOr(el.minMinutes.value, 0), 0);
  state.maxAge = parseNumberOr(el.maxAge.value, -1);
  state.minConfidence = Math.max(parseNumberOr(el.minConfidence.value, 0), 0);
  state.minGapEur = Math.max(parseNumberOr(el.minGap.value, 0), 0);
  state.shortlistTopN = Math.max(Math.round(parseNumberOr(el.topN.value, 100)), 1);
  state.sortBy = el.sort.value;
  state.sortOrder = el.sortDir.value;
  state.limit = Math.max(Math.round(parseNumberOr(el.limit.value, 50)), 1);
  state.undervaluedOnly = el.undervaluedOnly.checked;
}

function renderTrustCard() {
  const health = state.health || {};
  const metrics = state.metrics || {};
  const artifacts = health.artifacts || {};
  const active = state.activeArtifacts || {};

  const versionBits = [
    `test:${safeText(metrics.test_season)}`,
    `val:${safeText(metrics.val_season)}`,
  ];
  if (Number.isFinite(Number(metrics.trials_per_position))) {
    versionBits.push(`trials:${formatInt(metrics.trials_per_position)}`);
  }
  el.trustModelVersion.textContent = versionBits.join(" | ");

  const updated = artifacts.metrics_mtime_utc || artifacts.test_predictions_mtime_utc || "-";
  el.trustUpdated.textContent = safeText(updated);
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

  const testSegments = metrics?.segments?.test || [];
  const under5 = testSegments.find((s) => s.segment === "under_5m");
  if (under5 && Number.isFinite(Number(under5.mape))) {
    const mape = Number(under5.mape);
    if (mape > 0.45) {
      el.trustNote.textContent = "Low-value (<€5m) predictions are noisy. Use confidence + conservative gap, not point estimate alone.";
    } else {
      el.trustNote.textContent = "Segment reliability is acceptable for shortlist decisions with confidence filtering.";
    }
  } else {
    el.trustNote.textContent = "Connect API to load reliability diagnostics.";
  }
}

function renderMetrics() {
  const test = state.metrics?.overall?.test || {};
  const val = state.metrics?.overall?.val || {};

  el.metricTestR2.textContent = Number.isFinite(test.r2) ? formatPct(test.r2) : "-";
  el.metricTestMae.textContent = Number.isFinite(test.mae_eur) ? formatCurrency(test.mae_eur) : "-";
  el.metricTestMape.textContent = Number.isFinite(test.mape) ? formatPct(test.mape) : "-";

  el.metricValR2.textContent = Number.isFinite(val.r2) ? formatPct(val.r2) : "-";
  el.metricValMae.textContent = Number.isFinite(val.mae_eur) ? formatCurrency(val.mae_eur) : "-";
  el.metricValMape.textContent = Number.isFinite(val.mape) ? formatPct(val.mape) : "-";

  const dataset = state.metrics?.dataset || "-";
  const valSeason = state.metrics?.val_season || "-";
  const testSeason = state.metrics?.test_season || "-";
  el.metricsMeta.textContent = `Dataset: ${dataset} | Val: ${valSeason} | Test: ${testSeason}`;
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
    el.segmentWarning.textContent = "Warning: under_5m segment error is high. For scouting, prioritize relative rank and conservative gap over exact price.";
  } else {
    el.segmentWarning.textContent = "Segment diagnostics acceptable for ranking-based scouting workflows.";
  }
}

function renderCoverageTable() {
  if (!state.coverageRows.length) {
    el.coverageBody.innerHTML = "<tr><td colspan=\"4\">No coverage rows loaded.</td></tr>";
    return;
  }

  const grouped = new Map();
  state.coverageRows.forEach((row) => {
    const league = getLeague(row) || "Unknown";
    const key = league;
    if (!grouped.has(key)) {
      grouped.set(key, { league, n: 0, undervalued: 0, confSum: 0, confN: 0 });
    }
    const g = grouped.get(key);
    g.n += 1;

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

  const rows = Array.from(grouped.values()).sort((a, b) => b.n - a.n);
  el.coverageBody.innerHTML = rows
    .map((g) => {
      const pct = g.n > 0 ? g.undervalued / g.n : NaN;
      const avgConf = g.confN > 0 ? g.confSum / g.confN : NaN;
      return `
        <tr>
          <td>${safeText(g.league)}</td>
          <td class="num">${formatInt(g.n)}</td>
          <td class="num">${Number.isFinite(pct) ? formatPct(pct) : "-"}</td>
          <td class="num">${Number.isFinite(avgConf) ? formatNumber(avgConf) : "-"}</td>
        </tr>
      `;
    })
    .join("");
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
  if (!state.connected) return;
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
    if (el.watchlistMeta) el.watchlistMeta.textContent = "Watchlist entry saved.";
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
    el.tbody.innerHTML = "<tr><td colspan=\"10\">No rows found for current filters.</td></tr>";
    el.resultCount.textContent = "0 rows";
    el.resultRange.textContent = "offset 0";
    return;
  }

  el.tbody.innerHTML = state.rows
    .map((row, idx) => {
      const consGap = conservativeGapForRanking(row);
      return `
        <tr data-index="${idx}">
          <td class="player-cell"><strong>${safeText(row.name)}</strong><span>${safeText(row.season)}</span></td>
          <td>${safeText(row.club)}</td>
          <td>${safeText(row.league)}</td>
          <td>${safeText(getPosition(row))}</td>
          <td class="num">${formatNumber(row.age)}</td>
          <td class="num">${formatCurrency(row.market_value_eur)}</td>
          <td class="num">${formatCurrency(row.expected_value_eur)}</td>
          <td class="num ${consGap >= 0 ? "positive" : "negative"}">${formatCurrency(consGap)}</td>
          <td class="num">${formatNumber(row.undervaluation_confidence)}</td>
          <td class="num">${formatInt(getMinutes(row))}</td>
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

function clearDetail() {
  state.detailRequestId += 1;
  state.selectedRow = null;
  state.selectedReport = null;
  state.selectedHistory = null;
  el.detailContent.hidden = true;
  el.detailPlaceholder.hidden = false;
  el.detailSummary.textContent = "Select a player to load scouting summary.";
  el.detailHistory.textContent = "Select a player to load history-strength breakdown.";
  el.detailStrengths.innerHTML = '<li class="risk-item risk-item--none">Select a player to load strengths.</li>';
  el.detailLevers.innerHTML = '<li class="risk-item risk-item--none">Select a player to load development levers.</li>';
  el.detailRisks.innerHTML = '<li class="risk-item risk-item--none">Select a player to load risk flags.</li>';
  el.detailExportJson.disabled = true;
  el.detailExportCsv.disabled = true;
}

function renderDetail(row, { report = null, history = null, reportLoading = false, reportError = "" } = {}) {
  if (!row) return clearDetail();
  const reportPlayer = report?.player && typeof report.player === "object" ? report.player : {};
  const historyPayload = history?.history_strength && typeof history.history_strength === "object"
    ? history.history_strength
    : null;
  const mergedRow = { ...row, ...reportPlayer };
  const gaps = deriveGapValues(mergedRow, report);
  state.selectedRow = mergedRow;
  state.selectedReport = report || null;
  state.selectedHistory = history || null;
  el.detailPlaceholder.hidden = true;
  el.detailContent.hidden = false;
  el.detailExportJson.disabled = false;
  el.detailExportCsv.disabled = false;

  const market = firstFiniteNumber(report?.valuation_guardrails?.market_value_eur, mergedRow.market_value_eur, 0);
  const expected = firstFiniteNumber(report?.valuation_guardrails?.fair_value_eur, mergedRow.expected_value_eur, 0);
  const lower = firstFiniteNumber(mergedRow.expected_value_low_eur, 0);
  const scaleMax = Math.max(market, expected, lower, 1);

  el.detailName.textContent = safeText(mergedRow.name);
  el.detailMeta.textContent = `${safeText(mergedRow.club)} | ${safeText(mergedRow.league)} | ${safeText(
    getPosition(mergedRow)
  )} | ${safeText(mergedRow.season)}`;
  el.detailMarket.textContent = formatCurrency(market);
  el.detailExpected.textContent = formatCurrency(expected);
  el.detailLower.textContent = formatCurrency(lower);

  el.barMarket.style.width = `${Math.max((market / scaleMax) * 100, 1)}%`;
  el.barExpected.style.width = `${Math.max((expected / scaleMax) * 100, 1)}%`;
  el.barLower.style.width = `${Math.max((lower / scaleMax) * 100, 1)}%`;

  const rows = [
    ["League", safeText(mergedRow.league)],
    ["Age", formatNumber(mergedRow.age)],
    ["Minutes", formatInt(getMinutes(mergedRow))],
    ["Value Gap (raw)", formatCurrency(gaps.raw)],
    ["Conservative Gap (raw)", formatCurrency(gaps.conservative)],
    ["Conservative Gap (capped)", formatCurrency(gaps.capped)],
    ["Cap Threshold", formatCurrency(gaps.capThreshold)],
    ["Confidence", formatNumber(mergedRow.undervaluation_confidence)],
    ["Segment", safeText(mergedRow.value_segment)],
    ["Position Model", safeText(mergedRow.model_position)],
    ["Pred Low", formatCurrency(mergedRow.expected_value_low_eur)],
    ["Pred High", formatCurrency(mergedRow.expected_value_high_eur)],
  ];

  el.detailList.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  if (reportLoading) {
    el.detailSummary.textContent = "Loading scouting memo...";
    el.detailHistory.textContent = "Loading history-strength breakdown...";
    el.detailStrengths.innerHTML = '<li class="risk-item risk-item--none">Loading strengths...</li>';
    el.detailLevers.innerHTML = '<li class="risk-item risk-item--none">Loading development levers...</li>';
    el.detailRisks.innerHTML = '<li class="risk-item risk-item--none">Loading risk flags...</li>';
    return;
  }
  if (reportError) {
    el.detailSummary.textContent = `Scouting memo unavailable: ${reportError}`;
    el.detailHistory.textContent = "History-strength breakdown unavailable for this player.";
    el.detailStrengths.innerHTML = '<li class="risk-item risk-item--none">Strengths unavailable for this player.</li>';
    el.detailLevers.innerHTML = '<li class="risk-item risk-item--none">Development levers unavailable for this player.</li>';
    el.detailRisks.innerHTML = '<li class="risk-item risk-item--none">Risk flags unavailable for this player.</li>';
    return;
  }

  if (report?.summary_text) {
    el.detailSummary.textContent = report.summary_text;
  } else {
    el.detailSummary.textContent =
      "No memo summary loaded. Use the export buttons to fetch a complete player report on demand.";
  }

  const strengths = Array.isArray(report?.strengths) ? report.strengths : [];
  if (!strengths.length) {
    el.detailStrengths.innerHTML = '<li class="risk-item risk-item--none">No clear metric strengths for this cohort.</li>';
  } else {
    el.detailStrengths.innerHTML = strengths
      .slice(0, 5)
      .map((item) => `<li class="risk-item">${safeText(item.label)} (${formatNumber(item.quality_percentile)})</li>`)
      .join("");
  }

  const levers = Array.isArray(report?.development_levers) ? report.development_levers : [];
  if (!levers.length) {
    el.detailLevers.innerHTML = '<li class="risk-item risk-item--none">No high-impact development lever detected.</li>';
  } else {
    el.detailLevers.innerHTML = levers
      .slice(0, 5)
      .map(
        (item) =>
          `<li class="risk-item">${safeText(item.label)} (impact ${formatNumber(item.impact_score)})</li>`
      )
      .join("");
  }

  if (historyPayload?.summary_text) {
    const score = Number(historyPayload?.score_0_to_100);
    const cov = Number(historyPayload?.coverage_0_to_1);
    const scoreText = Number.isFinite(score) ? `${formatNumber(score)}/100` : "-";
    const covText = Number.isFinite(cov) ? formatPct(cov) : "-";
    el.detailHistory.textContent = `${historyPayload.summary_text} Coverage: ${covText}. Score: ${scoreText}.`;
  } else {
    el.detailHistory.textContent = "History-strength breakdown unavailable for this player.";
  }

  const riskFlags = Array.isArray(report?.risk_flags) ? report.risk_flags : [];
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
}

function renderModeAffordances() {
  const shortlist = state.mode === "shortlist";
  el.title.textContent = shortlist ? "Shortlist" : "Predictions";
  el.sort.disabled = shortlist;
  el.sortDir.disabled = shortlist;
  el.minConfidence.disabled = shortlist;
  el.minGap.disabled = shortlist;
  el.topN.disabled = !shortlist;
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

async function refreshCoverageAndOptions() {
  const rows = await fetchAllPredictions({
    split: state.split,
    columns:
      "season,league,undervalued_flag,undervaluation_confidence,value_gap_conservative_eur",
  });
  state.coverageRows = rows;
  renderCoverageTable();

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
  const payload = await getJson("/market-value/predictions", {
    split: state.split,
    season: state.season || null,
    league: state.league || null,
    position: state.position || null,
    min_minutes: state.minMinutes,
    max_age: state.maxAge < 0 ? null : state.maxAge,
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
  const payload = await getJson("/market-value/shortlist", {
    split: state.split,
    top_n: state.shortlistTopN,
    min_minutes: state.minMinutes,
    max_age: state.maxAge < 0 ? -1 : state.maxAge,
    positions: state.position || null,
  });

  const filtered = applyClientFilters((payload.items || []).map((r) => withComputedConservativeGap(r)));
  const sorted = sortShortlistRows(filtered);
  const page = sorted.slice(state.offset, state.offset + state.limit);

  state.rows = page;
  state.total = sorted.length;
  state.count = page.length;
}

async function runQuery() {
  readWorkbenchControlsToState();
  localStorage.setItem("scoutml_api_base", state.apiBase);
  renderModeAffordances();
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
  renderDetail(row, { reportLoading: true });
  try {
    const [reportResult, historyResult] = await Promise.allSettled([
      fetchPlayerReportForRow(row),
      fetchPlayerHistoryForRow(row),
    ]);
    if (reportResult.status !== "fulfilled") {
      throw reportResult.reason || new Error("Failed to load scouting report.");
    }
    const report = reportResult.value;
    const history = historyResult.status === "fulfilled" ? historyResult.value : null;
    if (requestId !== state.detailRequestId) return;
    renderDetail(row, { report, history });
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
    state.selectedHistory?.history_strength && typeof state.selectedHistory.history_strength === "object"
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
  };
}

function buildPlayerMemoCsvRow(row, report = null) {
  const memo = buildPlayerMemoPayload(row, report);
  const topStrengths = memo.strengths.slice(0, 3).map((m) => m.label).join("|");
  const topLevers = memo.development_levers.slice(0, 3).map((m) => m.label).join("|");
  const riskCodes = memo.risk_flags.map((r) => r.code).join("|");
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
    risk_codes: riskCodes,
    top_strengths: topStrengths,
    development_levers: topLevers,
    summary_text: memo.summary_text,
  };
}

async function ensureSelectedReport() {
  if (!state.selectedRow) return null;
  if (state.selectedReport && state.selectedHistory) return state.selectedReport;
  try {
    const [reportResult, historyResult] = await Promise.allSettled([
      fetchPlayerReportForRow(state.selectedRow),
      fetchPlayerHistoryForRow(state.selectedRow),
    ]);
    if (reportResult.status !== "fulfilled") {
      return null;
    }
    state.selectedReport = reportResult.value || null;
    state.selectedHistory = historyResult.status === "fulfilled" ? historyResult.value || null : null;
    return state.selectedReport;
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
  } else {
    el.funnelBody.innerHTML = state.funnelTopRows
      .map((row, idx) => {
        const score = Number(row._funnelScore);
        return `
          <tr data-index="${idx}" class="row-clickable">
            <td>${safeText(row.name)}</td>
            <td>${safeText(row.league)}</td>
            <td class="num">${formatNumber(row.age)}</td>
            <td class="num">${formatCurrency(row.market_value_eur)}</td>
            <td class="num">${formatCurrency(row.expected_value_eur)}</td>
            <td class="num positive">${formatCurrency(conservativeGapForRanking(row))}</td>
            <td class="num">${formatNumber(row.undervaluation_confidence)}</td>
            <td class="num">${Number.isFinite(score) ? formatNumber(score) : "-"}</td>
          </tr>
        `;
      })
      .join("");
  }

  if (!state.funnelRows.length) {
    el.funnelLeagueBody.innerHTML = "<tr><td colspan=\"5\">No league board yet.</td></tr>";
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
      return `
        <tr>
          <td>${safeText(g.league)}</td>
          <td class="num">${formatInt(g.n)}</td>
          <td class="num">${Number.isFinite(avgGap) ? formatCurrency(avgGap) : "-"}</td>
          <td class="num">${Number.isFinite(avgConf) ? formatNumber(avgConf) : "-"}</td>
          <td class="num">${Number.isFinite(avgAge) ? formatNumber(avgAge) : "-"}</td>
        </tr>
      `;
    })
    .join("");
}

async function runFunnel() {
  const split = el.funnelSplit.value;
  const maxAge = parseNumberOr(el.funnelMaxAge.value, 23);
  const minMinutes = Math.max(parseNumberOr(el.funnelMinMinutes.value, 900), 0);
  const minConfidence = Math.max(parseNumberOr(el.funnelMinConfidence.value, 0), 0);
  const minGap = Math.max(parseNumberOr(el.funnelMinGap.value, 0), 0);
  const topN = Math.max(Math.round(parseNumberOr(el.funnelTopN.value, 50)), 1);
  const lowerOnly = el.funnelLowerOnly.checked;

  el.funnelMeta.textContent = "Running funnel...";

  try {
    const payload = await getJson("/market-value/scout-targets", {
      split,
      top_n: Math.max(topN * 4, topN),
      non_big5_only: lowerOnly,
      max_age: maxAge < 0 ? null : maxAge,
      min_minutes: minMinutes,
      min_confidence: minConfidence > 0 ? minConfidence : null,
      min_value_gap_eur: minGap > 0 ? minGap : null,
    });
    const rows = (payload.items || []).map((row) => withComputedConservativeGap(row));
    const filtered = rows.filter((row) => {
      const gap = conservativeGapForRanking(row);
      if (minGap > 0 && (!Number.isFinite(gap) || gap < minGap)) return false;
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
        ? ` | precision@50=${formatPct(Number(p50.precision))}`
        : "";
    el.funnelMeta.textContent = `${formatInt(state.funnelTopRows.length)} shown / ${formatInt(
      state.funnelRows.length
    )} total candidates | split=${split}${lowerOnly ? " | lower-league-only" : ""}${precisionNote} | click a row for full scout report`;

    renderFunnelTables();
  } catch (err) {
    state.funnelRows = [];
    state.funnelTopRows = [];
    renderFunnelTables();
    el.funnelMeta.textContent = err instanceof Error ? err.message : String(err);
  }
}

async function loadHealthAndMetrics() {
  state.reportCache = new Map();
  state.selectedReport = null;
  state.selectedHistory = null;
  state.health = await getJson("/market-value/health");
  state.metrics = (await getJson("/market-value/metrics")).payload || null;
  try {
    state.activeArtifacts = (await getJson("/market-value/active-artifacts")).payload || null;
  } catch {
    state.activeArtifacts = null;
  }

  const artifacts = state.health?.artifacts || {};
  const ok = Boolean(artifacts.test_predictions_exists && artifacts.metrics_exists);
  setStatus(ok ? "ok" : "error", ok ? "Artifacts ready" : "Artifacts missing");
  state.connected = true;

  renderTrustCard();
  renderMetrics();
  renderSegmentTable();
}

function resetWorkbenchControls() {
  state.mode = "predictions";
  state.split = "test";
  state.season = "";
  state.league = "";
  state.position = "";
  state.search = "";
  state.minMinutes = 900;
  state.maxAge = 23;
  state.minConfidence = 0.5;
  state.minGapEur = 1_000_000;
  state.shortlistTopN = 100;
  state.sortBy = "value_gap_conservative_eur";
  state.sortOrder = "desc";
  state.limit = 50;
  state.offset = 0;
  state.undervaluedOnly = true;
  syncWorkbenchControlsFromState();
  renderModeAffordances();
}

function bindEvents() {
  let searchTimer = null;

  el.tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view || "overview"));
  });

  el.connectBtn.addEventListener("click", async () => {
    readWorkbenchControlsToState();
    localStorage.setItem("scoutml_api_base", state.apiBase);
    try {
      await loadHealthAndMetrics();
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
    el.minMinutes,
    el.maxAge,
    el.minConfidence,
    el.minGap,
    el.topN,
    el.sort,
    el.sortDir,
    el.limit,
    el.undervaluedOnly,
  ].forEach((control) => {
    control.addEventListener("change", async () => {
      state.offset = 0;
      await runQuery();
    });
  });

  el.exportBtn.addEventListener("click", () => {
    downloadCsv(state.rows, `scoutml_workbench_${state.mode}_${state.split}.csv`);
  });

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

  el.funnelRunBtn.addEventListener("click", runFunnel);
  el.funnelExportBtn.addEventListener("click", () => {
    downloadCsv(state.funnelTopRows, `scoutml_talent_funnel_${el.funnelSplit.value}.csv`);
  });
  if (el.watchlistAddBtn) {
    el.watchlistAddBtn.addEventListener("click", addSelectedToWatchlist);
  }
  if (el.watchlistRefreshBtn) {
    el.watchlistRefreshBtn.addEventListener("click", refreshWatchlist);
  }
  if (el.watchlistExportBtn) {
    el.watchlistExportBtn.addEventListener("click", () => {
      downloadCsv(state.watchlistRows, `scoutml_watchlist_${state.split}.csv`);
    });
  }
  if (el.watchlistBody) {
    el.watchlistBody.addEventListener("click", async (event) => {
      const btn = event.target.closest(".watchlist-delete");
      if (!btn) return;
      await deleteWatchlistItem(btn.dataset.watchId);
    });
  }
}

async function boot() {
  syncWorkbenchControlsFromState();
  renderModeAffordances();
  bindEvents();

  try {
    await loadHealthAndMetrics();
    await refreshCoverageAndOptions();
    await runQuery();
    await refreshWatchlist();
  } catch {
    el.tbody.innerHTML = `<tr><td colspan=\"10\">Connect API to start. Expected backend: ${state.apiBase}</td></tr>`;
    el.funnelMeta.textContent = "Connect API before building funnel.";
    if (el.watchlistMeta) {
      el.watchlistMeta.textContent = "Connect API before using watchlist.";
    }
  }
}

document.addEventListener("DOMContentLoaded", boot);
