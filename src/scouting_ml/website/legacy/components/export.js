/* =========================================================
   EXPORT MODULE
   CSV / Excel-compatible export
   ========================================================= */

window.ExportTools = (() => {

  /* -------------------------------------------------------
     EXPORT FILTERED TABLE TO CSV
     ------------------------------------------------------- */
  function exportFilteredCSV() {
    const players = SCOUT.filteredPlayers;

    if (players.length === 0) {
      alert("No players to export.");
      return;
    }

    const header = Object.keys(players[0]);
    const rows = players.map(p =>
      header.map(h => JSON.stringify(p[h] ?? "")).join(",")
    );

    const csv = header.join(",") + "\n" + rows.join("\n");

    downloadFile(csv, `players_${SCOUT.currentLeague}_${SCOUT.currentSeason}.csv`);
  }


  /* -------------------------------------------------------
     DOWNLOAD FILE
     ------------------------------------------------------- */
  function downloadFile(content, filename) {
    const blob = new Blob([content], { type: "text/csv" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();

    URL.revokeObjectURL(url);
  }


  /* -------------------------------------------------------
     EXPORT API
     ------------------------------------------------------- */
  return {
    exportFilteredCSV,
  };

})();
