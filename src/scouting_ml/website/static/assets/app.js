const DATASETS = window.SCOUTING_DATA ?? {};
const LEAGUES = Array.isArray(window.SCOUTING_LEAGUES)
  ? window.SCOUTING_LEAGUES
  : [];
const DEFAULT_LEAGUE = window.SCOUTING_DEFAULT_LEAGUE;

const state = {
  league: "",
  search: "",
  club: "",
  position: "",
  ageBand: "",
  sortKey: "market_value_eur",
  sortDirection: "desc",
};

const elements = {
  leagueSelect: document.getElementById("league-select"),
  leagueSubtitle: document.getElementById("league-subtitle"),
  searchInput: document.getElementById("search-input"),
  clubSelect: document.getElementById("club-select"),
  positionSelect: document.getElementById("position-select"),
  ageSelect: document.getElementById("age-select"),
  sortSelect: document.getElementById("sort-select"),
  sortButton: document.getElementById("sort-direction"),
  tableBody: document.getElementById("players-body"),
  emptyState: document.getElementById("empty-state"),
  summaryCount: document.getElementById("summary-count"),
  summaryMarket: document.getElementById("summary-market"),
  summaryXg: document.getElementById("summary-xg"),
  summaryMinutes: document.getElementById("summary-minutes"),
};

const numberFormatter = new Intl.NumberFormat("en-GB");
const currencyFormatter = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 0,
});
const ageFormatter = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 1,
});
const decimalFormatter = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function formatCurrency(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return currencyFormatter.format(value);
}

function formatInteger(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return numberFormatter.format(Math.round(value));
}

function formatAge(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return ageFormatter.format(value);
}

function formatDecimal(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return decimalFormatter.format(value);
}

function formatPercent(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return decimalFormatter.format(value) + "%";
}

function normaliseAgeBand(value, ageValue) {
  if (value) return value;
  if (ageValue == null || Number.isNaN(ageValue)) return "";
  if (ageValue < 18) return "u18";
  if (ageValue <= 22) return "18-22";
  if (ageValue <= 26) return "23-26";
  if (ageValue <= 30) return "27-30";
  return "30+";
}

function getDataset(slug) {
  return DATASETS[slug];
}

function getActiveDataset() {
  return getDataset(state.league);
}

function getActivePlayers() {
  const dataset = getActiveDataset();
  if (!dataset || !Array.isArray(dataset.players)) {
    return [];
  }
  return dataset.players;
}

function resetSelect(select, placeholder) {
  if (!select) return;
  select.innerHTML = "";
  const option = document.createElement("option");
  option.value = "";
  option.textContent = placeholder;
  select.appendChild(option);
}

function populateFilters(players) {
  resetSelect(elements.clubSelect, "All clubs");
  resetSelect(elements.positionSelect, "All positions");

  const clubs = new Set();
  const positions = new Set();
  players.forEach((player) => {
    if (player.club) clubs.add(player.club);
    const positionValue = player.position_group || player.position_main;
    if (positionValue) positions.add(positionValue);
  });

  Array.from(clubs)
    .sort()
    .forEach((club) => {
      const option = document.createElement("option");
      option.value = club;
      option.textContent = club;
      elements.clubSelect.appendChild(option);
    });

  Array.from(positions)
    .sort()
    .forEach((position) => {
      const option = document.createElement("option");
      option.value = position;
      option.textContent = position;
      elements.positionSelect.appendChild(option);
    });
}

function populateLeagueOptions(leagues) {
  resetSelect(elements.leagueSelect, "Select a league");
  leagues.forEach((league) => {
    const option = document.createElement("option");
    option.value = league.slug;
    option.textContent = league.season
      ? `${league.name} (${league.season})`
      : league.name;
    elements.leagueSelect.appendChild(option);
  });
}

function filterPlayers(players) {
  const query = state.search.trim().toLowerCase();
  return players.filter((player) => {
    const matchesSearch =
      !query ||
      (player.name && player.name.toLowerCase().includes(query)) ||
      (player.club && player.club.toLowerCase().includes(query));

    const matchesClub = !state.club || player.club === state.club;

    const positionValue = player.position_group || player.position_main || "";
    const matchesPosition = !state.position || positionValue === state.position;

    const band = normaliseAgeBand(player.age_band, player.age);
    const matchesAge = !state.ageBand || band === state.ageBand;

    return matchesSearch && matchesClub && matchesPosition && matchesAge;
  });
}

function sortPlayers(players) {
  const sorted = [...players];
  const { sortKey, sortDirection } = state;
  const factor = sortDirection === "asc" ? 1 : -1;
  sorted.sort((a, b) => {
    const aValue = a[sortKey] ?? -Infinity;
    const bValue = b[sortKey] ?? -Infinity;
    if (Number.isNaN(aValue) && Number.isNaN(bValue)) return 0;
    if (Number.isNaN(aValue)) return 1;
    if (Number.isNaN(bValue)) return -1;
    if (aValue === bValue) return 0;
    return aValue > bValue ? factor : -factor;
  });
  return sorted;
}

function updateLeagueSubtitle(meta, totalPlayers) {
  if (!elements.leagueSubtitle) return;
  if (!meta) {
    elements.leagueSubtitle.textContent = "Choose a league to get started.";
    return;
  }
  const seasonText = meta.season ? ` — ${meta.season}` : "";
  const playerText = Number.isFinite(totalPlayers)
    ? ` · ${numberFormatter.format(totalPlayers)} players`
    : "";
  elements.leagueSubtitle.textContent = `${meta.name}${seasonText}${playerText}`;
}

function updateSummary(players, dataset) {
  const count = players.length;
  let valueSum = 0;
  let xgSum = 0;
  let minutesSum = 0;

  players.forEach((player) => {
    valueSum += Number.isFinite(player.market_value_eur)
      ? player.market_value_eur
      : 0;
    xgSum += Number.isFinite(player.sofa_expectedGoals)
      ? player.sofa_expectedGoals
      : 0;
    minutesSum += Number.isFinite(player.sofa_minutesPlayed)
      ? player.sofa_minutesPlayed
      : 0;
  });

  elements.summaryCount.textContent = numberFormatter.format(count);
  elements.summaryMarket.textContent =
    count > 0 ? formatCurrency(valueSum / count) : "—";
  elements.summaryXg.textContent =
    count > 0 ? formatDecimal(xgSum / count) : "—";
  elements.summaryMinutes.textContent =
    count > 0 ? formatInteger(minutesSum / count) : "—";

  const totalPlayers = dataset ? dataset.players?.length : 0;
  updateLeagueSubtitle(dataset?.meta, totalPlayers);
}

function createRow(player) {
  const tr = document.createElement("tr");
  const positionValue = player.position_group || player.position_main || "—";
  const band = normaliseAgeBand(player.age_band, player.age);

  tr.innerHTML = `
    <td>
      <div class="player-name">
        <a href="${player.link ?? "#"}" target="_blank" rel="noopener">
          ${player.name ?? "Unknown"}
        </a>
        <span class="player-meta">${band ? band.toUpperCase() : ""}</span>
      </div>
    </td>
    <td>${player.club ?? "—"}</td>
    <td>${positionValue}</td>
    <td>${player.age != null ? formatAge(player.age) : "—"}</td>
    <td class="numeric">${formatCurrency(player.market_value_eur)}</td>
    <td class="numeric">${formatInteger(player.sofa_minutesPlayed)}</td>
    <td class="numeric">${formatInteger(player.sofa_goals)}</td>
    <td class="numeric">${formatInteger(player.sofa_assists)}</td>
    <td class="numeric">${formatDecimal(player.sofa_expectedGoals)}</td>
    <td class="numeric">${formatPercent(player.sofa_totalDuelsWonPercentage)}</td>
    <td class="numeric">${formatPercent(player.sofa_accuratePassesPercentage)}</td>
  `;

  return tr;
}

function render() {
  const dataset = getActiveDataset();
  const players = dataset ? dataset.players ?? [] : [];
  const filtered = filterPlayers(players);
  const sorted = sortPlayers(filtered);

  elements.tableBody.innerHTML = "";

  if (!dataset) {
    elements.emptyState.hidden = false;
    elements.emptyState.textContent =
      "Select a league above. No dataset is currently loaded.";
    updateSummary([], null);
    return;
  }

  if (sorted.length === 0) {
    elements.emptyState.hidden = false;
  } else {
    elements.emptyState.hidden = true;
    sorted.forEach((player) => {
      elements.tableBody.appendChild(createRow(player));
    });
  }

  updateSummary(filtered, dataset);
}

function resetFilters() {
  state.search = "";
  state.club = "";
  state.position = "";
  state.ageBand = "";
  if (elements.searchInput) elements.searchInput.value = "";
  if (elements.clubSelect) elements.clubSelect.value = "";
  if (elements.positionSelect) elements.positionSelect.value = "";
  if (elements.ageSelect) elements.ageSelect.value = "";
}

function attachListeners() {
  if (elements.leagueSelect) {
    elements.leagueSelect.addEventListener("change", (event) => {
      const next = event.target.value;
      if (!next) {
        return;
      }
      state.league = next;
      resetFilters();
      populateFilters(getActivePlayers());
      render();
    });
  }

  elements.searchInput.addEventListener("input", (event) => {
    state.search = event.target.value;
    render();
  });

  elements.clubSelect.addEventListener("change", (event) => {
    state.club = event.target.value;
    render();
  });

  elements.positionSelect.addEventListener("change", (event) => {
    state.position = event.target.value;
    render();
  });

  elements.ageSelect.addEventListener("change", (event) => {
    state.ageBand = event.target.value;
    render();
  });

  elements.sortSelect.addEventListener("change", (event) => {
    state.sortKey = event.target.value;
    render();
  });

  elements.sortButton.addEventListener("click", () => {
    state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
    elements.sortButton.textContent =
      state.sortDirection === "asc" ? "↑ Asc" : "↓ Desc";
    render();
  });
}

function resolveDefaultLeague() {
  if (DEFAULT_LEAGUE && DATASETS[DEFAULT_LEAGUE]) {
    return DEFAULT_LEAGUE;
  }
  if (LEAGUES.length > 0) {
    const first = LEAGUES[0].slug;
    if (DATASETS[first]) {
      return first;
    }
  }
  const keys = Object.keys(DATASETS);
  return keys.length > 0 ? keys[0] : "";
}

function boot() {
  if (!Object.keys(DATASETS).length) {
    elements.tableBody.innerHTML =
      "<tr><td colspan='11'>No player data bundle found. Rebuild via scouting_ml.website.build.</td></tr>";
    updateLeagueSubtitle(null, 0);
    return;
  }

  populateLeagueOptions(LEAGUES);

  const defaultLeague = resolveDefaultLeague();
  if (!defaultLeague) {
    elements.tableBody.innerHTML =
      "<tr><td colspan='11'>No league datasets are available.</td></tr>";
    updateLeagueSubtitle(null, 0);
    return;
  }

  state.league = defaultLeague;
  if (elements.leagueSelect) {
    elements.leagueSelect.value = defaultLeague;
  }

  resetFilters();
  populateFilters(getActivePlayers());
  attachListeners();
  render();
}

document.addEventListener("DOMContentLoaded", boot);

