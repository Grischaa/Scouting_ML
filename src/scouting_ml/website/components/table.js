/* =========================================================
   TABLE RENDERING MODULE
   Handles pagination, row rendering, action buttons
   ========================================================= */

window.Table = (() => {

  const tableBody = document.querySelector("#playersTable tbody");
  const pagination = document.getElementById("pagination");

  /* -------------------------------------------------------
     RENDER TABLE
     ------------------------------------------------------- */
  function renderTable() {
    tableBody.innerHTML = "";

    const players = SCOUT.filteredPlayers;
    const start = (SCOUT.currentPage - 1) * SCOUT.rowsPerPage;
    const end = start + SCOUT.rowsPerPage;

    const pageRows = players.slice(start, end);

    pageRows.forEach(p => {
      const tr = document.createElement("tr");

      tr.innerHTML = `
        <td>${p.name}</td>
        <td>${p.club}</td>
        <td>${p.position_main || ""}</td>
        <td>${p.age || ""}</td>
        <td>${formatValue(p.market_value_eur)}</td>
        <td>${p.sofa_minutesPlayed || 0}</td>
        <td>${p.sofa_goals || 0}</td>
        <td>${p.sofa_assists || 0}</td>
        <td>${p.sofa_expectedGoals || 0}</td>
        <td>
          <button data-action="show-player" data-player-id="${p.player_id}" class="mini-btn">Info</button>
          <button data-action="add-compare" data-player-id="${p.player_id}" class="mini-btn-compare">+Compare</button>
        </td>
      `;

      tableBody.appendChild(tr);
    });
  }


  /* -------------------------------------------------------
     PAGINATION
     ------------------------------------------------------- */
  function renderPagination() {
    const total = SCOUT.filteredPlayers.length;
    const totalPages = Math.ceil(total / SCOUT.rowsPerPage);

    pagination.innerHTML = "";

    for (let i = 1; i <= totalPages; i++) {
      const btn = document.createElement("button");
      btn.className = "page-btn" + (i === SCOUT.currentPage ? " active" : "");
      btn.textContent = i;

      btn.addEventListener("click", () => {
        SCOUT.currentPage = i;
        renderTable();
        renderPagination();
      });

      pagination.appendChild(btn);
    }
  }


  /* -------------------------------------------------------
     HELPERS
     ------------------------------------------------------- */
  function formatValue(v) {
    v = Number(v);
    if (!v) return "";
    if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + "M €";
    if (v >= 1_000) return (v / 1_000).toFixed(1) + "k €";
    return v + " €";
  }


  /* -------------------------------------------------------
     EXPORT API
     ------------------------------------------------------- */
  return {
    renderTable,
    renderPagination,
  };

})();
