/* =========================================================
   PLAYER POPUP MODULE
   Shows modal with radar chart + basic info
   ========================================================= */

window.Popup = (() => {

  const popup = document.getElementById("playerPopup");
  const popupName = document.getElementById("popupName");
  const popupMeta = document.getElementById("popupMeta");

  const radarCanvas = document.getElementById("popupRadar");
  const exportBtn = document.getElementById("exportPlayerBtn");


  /* -------------------------------------------------------
     OPEN POPUP
     ------------------------------------------------------- */
  function openPlayerPopup(playerId) {
    const player = SCOUT.currentPlayers.find(p => String(p.player_id) === String(playerId));
    if (!player) return;

    popupName.textContent = player.name;

    popupMeta.innerHTML = `
      <div><strong>Club:</strong> ${player.club}</div>
      <div><strong>Position:</strong> ${player.position_main}</div>
      <div><strong>Age:</strong> ${player.age}</div>
      <div><strong>Market Value:</strong> €${player.market_value_eur}</div>
      <div><strong>Minutes:</strong> ${player.sofa_minutesPlayed}</div>
      <div><strong>Goals:</strong> ${player.sofa_goals}</div>
      <div><strong>Assists:</strong> ${player.sofa_assists}</div>
      <div><strong>xG:</strong> ${player.sofa_expectedGoals}</div>
    `;

    RadarChart.drawRadar(radarCanvas, player);

    exportBtn.onclick = () => exportPlayerJSON(player);

    popup.classList.add("active");
  }


  /* -------------------------------------------------------
     EXPORT PLAYER
     ------------------------------------------------------- */
  function exportPlayerJSON(player) {
    const blob = new Blob([JSON.stringify(player, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `${player.name.replace(/ /g, "_")}.json`;
    a.click();

    URL.revokeObjectURL(url);
  }


  /* -------------------------------------------------------
     CLOSE POPUP
     ------------------------------------------------------- */
  function closePopup() {
    popup.classList.remove("active");
  }


  /* -------------------------------------------------------
     EXPORT API
     ------------------------------------------------------- */
  return {
    openPlayerPopup,
    closePopup,
  };

})();
