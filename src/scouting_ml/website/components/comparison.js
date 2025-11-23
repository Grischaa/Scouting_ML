/* =========================================================
   PLAYER COMPARISON MODULE
   Allows 2–4 players + multi radar chart
   ========================================================= */

window.Comparison = (() => {

  const comparisonPanel = document.getElementById("comparisonPanel");
  const comparisonList = document.getElementById("comparisonList");
  const clearBtn = document.getElementById("clearComparisonBtn");
  const radarCanvas = document.getElementById("comparisonRadar");

  /* -------------------------------------------------------
     ADD PLAYER TO COMPARISON LIST
     ------------------------------------------------------- */
  function addPlayerToComparison(playerId) {
    const player = SCOUT.currentPlayers.find(p => String(p.player_id) === String(playerId));
    if (!player) return;

    // Limit to 4 players
    if (SCOUT.comparisonList.length >= 4) {
      alert("Max 4 players in comparison.");
      return;
    }

    // Avoid duplicates
    if (SCOUT.comparisonList.find(p => p.player_id === player.player_id)) {
      return;
    }

    SCOUT.comparisonList.push(player);

    renderComparisonList();
    drawComparisonRadar();
  }


  /* -------------------------------------------------------
     RENDER COMPARISON LIST
     ------------------------------------------------------- */
  function renderComparisonList() {
    comparisonList.innerHTML = "";

    SCOUT.comparisonList.forEach(p => {
      const div = document.createElement("div");
      div.className = "compare-entry";

      div.innerHTML = `
        <span><strong>${p.name}</strong></span>
        <span>${p.club} — ${p.position_main}</span>
        <button data-id="${p.player_id}" class="remove-compare-btn">Remove</button>
      `;

      comparisonList.appendChild(div);
    });

    // Remove handler
    document.querySelectorAll(".remove-compare-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        removePlayer(btn.dataset.id);
      });
    });
  }


  /* -------------------------------------------------------
     REMOVE PLAYER
     ------------------------------------------------------- */
  function removePlayer(playerId) {
    SCOUT.comparisonList = SCOUT.comparisonList.filter(p => String(p.player_id) !== String(playerId));
    renderComparisonList();
    drawComparisonRadar();
  }


  /* -------------------------------------------------------
     CLEAR ALL
     ------------------------------------------------------- */
  clearBtn.addEventListener("click", () => {
    SCOUT.comparisonList = [];
    renderComparisonList();
    clearCanvas();
  });


  /* -------------------------------------------------------
     MULTI-PLAYER RADAR CHART
     ------------------------------------------------------- */
  function drawComparisonRadar() {
    const list = SCOUT.comparisonList;
    if (list.length === 0) {
      clearCanvas();
      return;
    }

    const ctx = radarCanvas.getContext("2d");
    const w = radarCanvas.width;
    const h = radarCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Re-use radar background layers
    drawRadarBackground(ctx, w, h);

    // Multiple colors
    const colors = [
      "#00bcd4",
      "#57ff9b",
      "#ff5f5f",
      "#ffc107",
    ];

    // Draw players
    list.forEach((player, idx) => {
      drawRadarPlayer(ctx, w, h, player, colors[idx % colors.length]);
    });
  }


  /* -------------------------------------------------------
     HELPERS — RADAR BACKGROUND
     ------------------------------------------------------- */
  function drawRadarBackground(ctx, w, h) {
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) / 2 - 40;
    const metrics = 5;

    ctx.strokeStyle = "#2f3742";
    ctx.lineWidth = 1;

    for (let r = radius; r > 0; r -= radius / 5) {
      ctx.beginPath();
      for (let i = 0; i < metrics; i++) {
        const angle = (Math.PI * 2 / metrics) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }


  /* -------------------------------------------------------
     HELPERS — PLOT ONE PLAYER
     ------------------------------------------------------- */
  function drawRadarPlayer(ctx, w, h, player, color) {
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) / 2 - 40;

    const metrics = [
      ["sofa_goals", 30],
      ["sofa_assists", 20],
      ["sofa_expectedGoals", 20],
      ["sofa_minutesPlayed", 3500],
      ["market_value_eur", 100_000_000]
    ];

    const points = [];

    metrics.forEach(([field, max], i) => {
      const value = Number(player[field]) || 0;
      const normalized = Math.min(value / max, 1);
      const angle = (Math.PI * 2 / metrics.length) * i - Math.PI / 2;

      const px = cx + Math.cos(angle) * (radius * normalized);
      const py = cy + Math.sin(angle) * (radius * normalized);

      points.push({ x: px, y: py });
    });

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.fillStyle = color + "55"; // 30% opacity

    points.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });

    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }


  function clearCanvas() {
    const ctx = radarCanvas.getContext("2d");
    ctx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);
  }


  /* -------------------------------------------------------
     EXPORT MODULE API
     ------------------------------------------------------- */
  return {
    addPlayerToComparison,
  };

})();
