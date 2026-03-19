/* =========================================================
   RADAR CHART ENGINE
   Lightweight, custom canvas charts
   ========================================================= */

window.RadarChart = (() => {

  const metrics = [
    ["sofa_goals", "Goals"],
    ["sofa_assists", "Assists"],
    ["sofa_expectedGoals", "xG"],
    ["sofa_minutesPlayed", "Minutes"],
    ["market_value_eur", "Value (€M)"]
  ];


  /* -------------------------------------------------------
     DRAW RADAR
     ------------------------------------------------------- */
  function drawRadar(canvas, player) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) / 2 - 40;

    ctx.clearRect(0, 0, w, h);

    // Radar background
    ctx.strokeStyle = "#2f3742";
    ctx.lineWidth = 1;

    for (let r = radius; r > 0; r -= radius / 5) {
      ctx.beginPath();
      for (let i = 0; i < metrics.length; i++) {
        const angle = (Math.PI * 2 / metrics.length) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Radar labels + max values
    ctx.fillStyle = "#e6e6e6";
    ctx.font = "14px Arial";

    const maxValues = {
      sofa_goals: 30,
      sofa_assists: 20,
      sofa_expectedGoals: 20,
      sofa_minutesPlayed: 3500,
      market_value_eur: 100_000_000,
    };

    const points = [];

    metrics.forEach(([field, label], i) => {
      const angle = (Math.PI * 2 / metrics.length) * i - Math.PI / 2;
      const labelX = cx + Math.cos(angle) * (radius + 20);
      const labelY = cy + Math.sin(angle) * (radius + 20);

      // Label
      ctx.fillText(label, labelX - 20, labelY);

      const value = Number(player[field]) || 0;
      const max = maxValues[field];
      const normalized = Math.min(value / max, 1);

      const px = cx + Math.cos(angle) * (radius * normalized);
      const py = cy + Math.sin(angle) * (radius * normalized);

      points.push({ x: px, y: py });
    });

    // Radar polygon (filled)
    ctx.beginPath();
    ctx.fillStyle = "rgba(0, 188, 212, 0.35)";
    ctx.strokeStyle = "#00bcd4";
    ctx.lineWidth = 2;

    points.forEach((pt, i) => {
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });

    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }


  /* -------------------------------------------------------
     EXPORT API
     ------------------------------------------------------- */
  return {
    drawRadar,
  };

})();
