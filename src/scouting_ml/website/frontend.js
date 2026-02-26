/* =========================================================
   FRONTEND CONTROLLER
   Connects everything together
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {

  // =======================================================
  // GLOBAL STATE
  // =======================================================
  window.SCOUT = {
    leagues: window.SCOUTING_LEAGUES,
    data: window.SCOUTING_DATA,

    currentLeague: null,
    currentSeason: null,
    currentPlayers: [],
    filteredPlayers: [],

    sortColumn: null,
    sortDirection: null,

    currentPage: 1,
    rowsPerPage: 50,

    comparisonList: [],
  };


  // =======================================================
  // ELEMENT REFERENCES
  // =======================================================
  const leagueSelect = document.getElementById("leagueSelect");
  const seasonSelect = document.getElementById("seasonSelect");
  const searchInput = document.getElementById("searchInput");
  const filterToggleBtn = document.getElementById("filterToggleBtn");
  const filterPanel = document.getElementById("filterPanel");
  const closeFilterBtn = document.getElementById("closeFilterBtn");

  const compareToggleBtn = document.getElementById("compareToggleBtn");
  const comparisonPanel = document.getElementById("comparisonPanel");
  const closeComparisonBtn = document.getElementById("closeComparisonBtn");

  const exportBtn = document.getElementById("exportBtn");

  const tableBody = document.querySelector("#playersTable tbody");
  const pagination = document.getElementById("pagination");

  const popup = document.getElementById("playerPopup");
  const closePopup = document.getElementById("closePlayerPopup");


  // =======================================================
  // INIT FUNCTIONS
  // =======================================================

  function initLeagues() {
    const leagues = SCOUT.leagues;

    leagues.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l.slug;
      opt.textContent = l.name;
      leagueSelect.appendChild(opt);
    });

    SCOUT.currentLeague = window.SCOUTING_DEFAULT_LEAGUE;
    leagueSelect.value = SCOUT.currentLeague;

    loadSeasons(SCOUT.currentLeague);
  }


  function loadSeasons(leagueSlug) {
    seasonSelect.innerHTML = "";

    const seasons = Object.keys(SCOUT.data[leagueSlug].seasons)
      .sort()
      .reverse();

    seasons.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      seasonSelect.appendChild(opt);
    });

    SCOUT.currentSeason = seasons[0];
    seasonSelect.value = SCOUT.currentSeason;

    loadPlayers(leagueSlug, SCOUT.currentSeason);
  }


  function loadPlayers(leagueSlug, season) {
    SCOUT.currentPlayers = SCOUT.data[leagueSlug].seasons[season].players;
    SCOUT.currentLeague = leagueSlug;
    SCOUT.currentSeason = season;

    // Populate dropdowns in filter panel
    Filters.populateClubFilter();
    Filters.populatePositionFilter();

    applyFilters();
  }


  // =======================================================
  // FILTERING PIPELINE
  // =======================================================
  function applyFilters() {
    const query = searchInput.value.toLowerCase();

    SCOUT.filteredPlayers = SCOUT.currentPlayers.filter(p => {
      const matchesSearch =
        p.name.toLowerCase().includes(query) ||
        p.club.toLowerCase().includes(query) ||
        (p.position_main || "").toLowerCase().includes(query);

      const matchesAllFilters = Filters.applyFilterRules(p);

      return matchesSearch && matchesAllFilters;
    });

    applySorting();
  }


  // =======================================================
  // SORTING
  // =======================================================
  function applySorting() {
    if (SCOUT.sortColumn) {
      const col = SCOUT.sortColumn;
      const dir = SCOUT.sortDirection;

      SCOUT.filteredPlayers.sort((a, b) => {
        let x = a[col], y = b[col];

        // Numeric sorting when possible
        if (!isNaN(x) && !isNaN(y)) {
          x = Number(x);
          y = Number(y);
        } else {
          x = String(x).toLowerCase();
          y = String(y).toLowerCase();
        }

        if (dir === "asc") return x > y ? 1 : -1;
        return x < y ? 1 : -1;
      });
    }

    SCOUT.currentPage = 1;
    Table.renderTable();
    Table.renderPagination();
  }


  // =======================================================
  // EVENT ROUTING
  // =======================================================

  leagueSelect.addEventListener("change", () =>
    loadSeasons(leagueSelect.value)
  );

  seasonSelect.addEventListener("change", () =>
    loadPlayers(leagueSelect.value, seasonSelect.value)
  );

  searchInput.addEventListener("input", () => applyFilters());

  document.querySelectorAll("th[data-col]").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;

      if (SCOUT.sortColumn === col) {
        SCOUT.sortDirection = SCOUT.sortDirection === "asc" ? "desc" : "asc";
      } else {
        SCOUT.sortColumn = col;
        SCOUT.sortDirection = "asc";
      }

      document.querySelectorAll("th")
        .forEach(th => th.classList.remove("sorted-asc", "sorted-desc"));

      th.classList.add(
        SCOUT.sortDirection === "asc" ? "sorted-asc" : "sorted-desc"
      );

      applySorting();
    });
  });


  // =======================================================
  // FILTER PANEL HANDLERS
  // =======================================================
  filterToggleBtn.addEventListener("click", () =>
    filterPanel.classList.add("open")
  );

  closeFilterBtn.addEventListener("click", () =>
    filterPanel.classList.remove("open")
  );


  // =======================================================
  // COMPARISON PANEL HANDLERS
  // =======================================================
  compareToggleBtn.addEventListener("click", () =>
    comparisonPanel.classList.add("open")
  );

  closeComparisonBtn.addEventListener("click", () =>
    comparisonPanel.classList.remove("open")
  );


  // =======================================================
  // EXPORT
  // =======================================================
  exportBtn.addEventListener("click", () =>
    ExportTools.exportFilteredCSV()
  );


  // =======================================================
  // PLAYER POPUP
  // =======================================================
  document.addEventListener("click", e => {
    if (e.target.dataset.action === "show-player") {
      Popup.openPlayerPopup(e.target.dataset.playerId);
    }

    if (e.target.dataset.action === "add-compare") {
      Comparison.addPlayerToComparison(e.target.dataset.playerId);
    }
  });

  closePopup.addEventListener("click", () => Popup.closePopup());


  // =======================================================
  // INITIALIZE
  // =======================================================
  initLeagues();
});
