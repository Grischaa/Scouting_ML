/* =========================================================
   FILTER CONTROL MODULE
   Handles all advanced filtering
   ========================================================= */

window.Filters = (() => {

  const filterClub = document.getElementById("filterClub");
  const filterPosition = document.getElementById("filterPosition");
  const filterAgeBand = document.getElementById("filterAgeBand");

  const valueSlider = document.getElementById("valueSlider");
  const valueSliderLabel = document.getElementById("valueSliderLabel");

  const minutesSlider = document.getElementById("minutesSlider");
  const minutesSliderLabel = document.getElementById("minutesSliderLabel");

  const resetBtn = document.getElementById("resetFiltersBtn");


  /* -------------------------------------------------------
     POPULATE CLUB FILTER
     ------------------------------------------------------- */
  function populateClubFilter() {
    const players = SCOUT.currentPlayers;

    const clubs = [...new Set(players.map(p => p.club))].sort();

    filterClub.innerHTML = `<option value="">All</option>`;

    clubs.forEach(club => {
      const opt = document.createElement("option");
      opt.value = club;
      opt.textContent = club;
      filterClub.appendChild(opt);
    });
  }


  /* -------------------------------------------------------
     POPULATE POSITION FILTER
     ------------------------------------------------------- */
  function populatePositionFilter() {
    const players = SCOUT.currentPlayers;

    const positions = [...new Set(players.map(p => p.position_main))].sort();

    filterPosition.innerHTML = `<option value="">All</option>`;

    positions.forEach(pos => {
      if (!pos) return;
      const opt = document.createElement("option");
      opt.value = pos;
      opt.textContent = pos;
      filterPosition.appendChild(opt);
    });
  }


  /* -------------------------------------------------------
     APPLY FILTER RULES
     (called by frontend.js → applyFilters())
     ------------------------------------------------------- */
  function applyFilterRules(p) {

    // CLUB
    if (filterClub.value && p.club !== filterClub.value) {
      return false;
    }

    // POSITION
    if (filterPosition.value && p.position_main !== filterPosition.value) {
      return false;
    }

    // AGE BAND
    if (filterAgeBand.value && p.age_band !== filterAgeBand.value) {
      return false;
    }

    // MARKET VALUE
    const maxValue = Number(valueSlider.value) * 1_000_000;
    if (p.market_value_eur && Number(p.market_value_eur) > maxValue) {
      return false;
    }

    // MINUTES PLAYED
    if (p.sofa_minutesPlayed && Number(p.sofa_minutesPlayed) < minutesSlider.value) {
      return false;
    }

    return true;
  }


  /* -------------------------------------------------------
     SLIDER HANDLING
     ------------------------------------------------------- */
  valueSlider.addEventListener("input", () => {
    valueSliderLabel.textContent = `≤ €${valueSlider.value}M`;
    SCOUT.currentPage = 1;
    window.applyFilters();
  });

  minutesSlider.addEventListener("input", () => {
    minutesSliderLabel.textContent = `≥ ${minutesSlider.value} min`;
    SCOUT.currentPage = 1;
    window.applyFilters();
  });


  /* -------------------------------------------------------
     RESET FILTERS
     ------------------------------------------------------- */
  resetBtn.addEventListener("click", () => {
    filterClub.value = "";
    filterPosition.value = "";
    filterAgeBand.value = "";
    valueSlider.value = 100;
    minutesSlider.value = 0;

    valueSliderLabel.textContent = "";
    minutesSliderLabel.textContent = "";

    window.applyFilters();
  });


  /* -------------------------------------------------------
     EXPORTED API
     ------------------------------------------------------- */
  return {
    populateClubFilter,
    populatePositionFilter,
    applyFilterRules,
  };

})();
