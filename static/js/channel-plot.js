const CHANNEL_PLOT = {
  widths: [128, 256, 512],
  ymax: 28,
  series: [
    { key: "flat", label: "DC-AE", color: "#9ca4af", dash: [7, 5], ys: [3.22, 9.19, 26.1] },
    { key: "dcae15", label: "DC-AE 1.5", color: "#d9a631", dash: [], ys: [3.15, 7.11, 14.76] },
    { key: "freqwarm", label: "FreqWarm", color: "#a276da", dash: [], ys: [3.53, 7.54, 20.8] },
    { key: "spectrum", label: "Spectrum Matching", color: "#4ea49b", dash: [], ys: [3.41, 8.28, 26.4] },
  ],
  apex: { label: "APEX", color: "#1B8A4A", dash: [], ys: [2.85, 3.15, 4.9] },
};
const MATCHED_RECONSTRUCTION_FID = [0.22, 0.18, 0.13];

function plotXLog(ch, left, width) {
  const a = Math.log2(CHANNEL_PLOT.widths[0]);
  const b = Math.log2(CHANNEL_PLOT.widths[CHANNEL_PLOT.widths.length - 1]);
  return left + ((Math.log2(ch) - a) / (b - a)) * width;
}

function drawProblemTradeoff(ctx, cssW, cssH, state) {
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = PAGE;
  ctx.fillRect(0, 0, cssW, cssH);

  const compact = cssW < 420;
  const left = compact ? 42 : 52;
  const right = cssW - 14;
  const plotW = right - left;
  const chartTop = compact ? 62 : 66;
  const chartBottom = cssH - (compact ? 58 : 54);
  const laneGap = compact ? 58 : 64;
  const laneHeight = (chartBottom - chartTop - laneGap) / 2;
  const reconstructionTop = chartTop;
  const generationTop = chartTop + laneHeight + laneGap;
  const xOf = (channels) => plotXLog(channels, left, plotW);
  const reconstructionY = (value) => reconstructionTop + 10 + ((0.23 - value) / (0.23 - 0.12)) * (laneHeight - 20);
  const generationY = (value) => generationTop + 10 + ((28 - value) / 28) * (laneHeight - 20);
  const fade = state.baseFade == null ? 1 : state.baseFade;

  ctx.save();
  ctx.globalAlpha = fade;
  ctx.fillStyle = MUTED;
  ctx.font = "600 12px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Standard DC-AE · matched compute", left, 18);

  for (const channels of CHANNEL_PLOT.widths) {
    const x = xOf(channels);
    ctx.strokeStyle = HAIR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, reconstructionTop);
    ctx.lineTo(x, reconstructionTop + laneHeight);
    ctx.moveTo(x, generationTop);
    ctx.lineTo(x, generationTop + laneHeight);
    ctx.stroke();
  }

  function drawLaneHeading(label, status, y, color, alpha = 1) {
    ctx.fillStyle = INK;
    ctx.font = compact ? "600 12px Inter, -apple-system, sans-serif" : "600 13px Inter, -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, left, y);
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = color;
    ctx.textAlign = "right";
    ctx.fillText(status, right, y);
    ctx.restore();
  }

  drawLaneHeading("Reconstruction · rFID ↓", "improves", reconstructionTop - 13, "#2563A6");
  drawLaneHeading("Generation · gFID ↓", "collapses", generationTop - 13, "#d70015", state.collapse);

  function drawTrack(values, yOf, color, formatValue) {
    const points = CHANNEL_PLOT.widths.map((channels, index) => ({
      x: xOf(channels),
      y: yOf(values[index]),
    }));
    const visiblePoints = points.slice(0, state.basePts);
    if (state.baseHead > 0 && state.basePts < points.length) {
      const previous = visiblePoints[visiblePoints.length - 1];
      const next = points[state.basePts];
      visiblePoints.push({
        x: lerp(previous.x, next.x, state.baseHead),
        y: lerp(previous.y, next.y, state.baseHead),
      });
    }

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.4;
    if (visiblePoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(visiblePoints[0].x, visiblePoints[0].y);
      for (let index = 1; index < visiblePoints.length; index++) {
        ctx.lineTo(visiblePoints[index].x, visiblePoints[index].y);
      }
      ctx.stroke();
    }
    for (let index = 0; index < state.basePts; index++) {
      const point = points[index];
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = compact ? "600 10px Inter, -apple-system, sans-serif" : "600 11px Inter, -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = point.y < yOf(values[1]) ? "top" : "bottom";
      const labelOffset = ctx.textBaseline === "top" ? 9 : -8;
      ctx.fillText(formatValue(values[index]), point.x, point.y + labelOffset);
    }
  }

  drawTrack(MATCHED_RECONSTRUCTION_FID, reconstructionY, "#2563A6", (value) => value.toFixed(2));
  drawTrack(CHANNEL_PLOT.series[0].ys, generationY, "#d70015", (value) => String(value));

  ctx.strokeStyle = INK;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, generationTop + laneHeight);
  ctx.lineTo(right, generationTop + laneHeight);
  ctx.stroke();

  ctx.fillStyle = INK;
  ctx.font = "12px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  for (const channels of CHANNEL_PLOT.widths) {
    ctx.fillText(String(channels), xOf(channels), generationTop + laneHeight + 18);
  }
  ctx.font = "13px Inter, -apple-system, sans-serif";
  ctx.fillText("Channels", left + plotW / 2, generationTop + laneHeight + 40);
  ctx.restore();
}

function drawChannelPlot(ctx, cssW, cssH, state) {
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = PAGE;
  ctx.fillRect(0, 0, cssW, cssH);

  const compact = cssW < 620;
  const showReconstruction = Boolean(state.reconstruction);
  const topPadding = showReconstruction ? 100 : state.collapseLabel ? 58 : 34;
  const pad = { l: 52, r: 14, t: topPadding, b: compact ? 116 : 74 };
  const left = pad.l;
  const top = pad.t;
  const plotW = cssW - pad.l - pad.r;
  const plotH = cssH - pad.t - pad.b;
  const yMax = CHANNEL_PLOT.ymax;
  const yOf = (gfid) => top + plotH * (1 - gfid / yMax);
  const xOf = (ch) => plotXLog(ch, left, plotW);
  const fade = state.baseFade == null ? 1 : state.baseFade;

  if (showReconstruction) {
    const insetTop = 24;
    const insetBottom = 52;
    const reconstructionY = (value) => insetTop + ((0.23 - value) / (0.23 - 0.12)) * (insetBottom - insetTop);
    const points = CHANNEL_PLOT.widths.map((channels, index) => ({
      x: xOf(channels),
      y: reconstructionY(MATCHED_RECONSTRUCTION_FID[index]),
    }));
    const visiblePoints = points.slice(0, state.basePts);
    if (state.baseHead > 0 && state.basePts < points.length) {
      const previous = visiblePoints[visiblePoints.length - 1];
      const next = points[state.basePts];
      visiblePoints.push({
        x: lerp(previous.x, next.x, state.baseHead),
        y: lerp(previous.y, next.y, state.baseHead),
      });
    }

    ctx.save();
    ctx.globalAlpha = fade;
    ctx.fillStyle = INK;
    ctx.font = "600 12px Inter, -apple-system, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Reconstruction FID ↓", left, 14);
    ctx.textAlign = "right";
    ctx.fillStyle = "#2563A6";
    ctx.fillText("0.22 → 0.18 → 0.13", left + plotW, 14);
    ctx.strokeStyle = "#2563A6";
    ctx.fillStyle = "#2563A6";
    ctx.lineWidth = 2.2;
    if (visiblePoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(visiblePoints[0].x, visiblePoints[0].y);
      for (let index = 1; index < visiblePoints.length; index++) {
        ctx.lineTo(visiblePoints[index].x, visiblePoints[index].y);
      }
      ctx.stroke();
    }
    for (let index = 0; index < state.basePts; index++) {
      ctx.beginPath();
      ctx.arc(points[index].x, points[index].y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = HAIR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, 60);
    ctx.lineTo(left + plotW, 60);
    ctx.stroke();
    ctx.restore();
  }

  ctx.font = "600 14px Inter, -apple-system, sans-serif";
  ctx.fillStyle = INK;
  ctx.textAlign = "left";
  ctx.fillText("Generation quality", left, showReconstruction ? 84 : 20);

  ctx.strokeStyle = HAIR;
  ctx.lineWidth = 1;
  for (let g = 5; g <= 25; g += 5) {
    const y = yOf(g);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = MUTED;
    ctx.font = "12px Inter, -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(String(g), left - 8, y + 4);
  }

  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, top + plotH);
  ctx.lineTo(left + plotW, top + plotH);
  ctx.stroke();

  ctx.fillStyle = INK;
  ctx.font = "13px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Channels", left + plotW / 2, top + plotH + 36);
  ctx.save();
  ctx.translate(16, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("gFID ↓", 0, 0);
  ctx.restore();

  CHANNEL_PLOT.widths.forEach((w) => {
    ctx.fillStyle = INK;
    ctx.font = "12px Inter, -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(w), xOf(w), top + plotH + 16);
  });

  function strokeSeries(series, nPts, headT, alpha, lineW) {
    if (nPts <= 0) return;
    const pts = [];
    for (let i = 0; i < nPts; i++) {
      pts.push({ x: xOf(CHANNEL_PLOT.widths[i]), y: yOf(series.ys[i]) });
    }
    if (headT > 0 && nPts < series.ys.length) {
      const a = pts[pts.length - 1];
      pts.push({
        x: lerp(a.x, xOf(CHANNEL_PLOT.widths[nPts]), headT),
        y: lerp(a.y, yOf(series.ys[nPts]), headT),
      });
    }
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = series.color;
    ctx.fillStyle = series.color;
    ctx.lineWidth = lineW || 2.4;
    ctx.setLineDash(series.dash || []);
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.stroke();
    ctx.setLineDash([]);
    for (let i = 0; i < nPts; i++) {
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  CHANNEL_PLOT.series.forEach((s) => {
    strokeSeries(s, state.basePts, state.baseHead, fade, 2.2);
  });

  if (state.collapse > 0) {
    const targetX = xOf(CHANNEL_PLOT.widths[2]);
    const targetY = yOf(Math.max(...CHANNEL_PLOT.series.map((series) => series.ys[2])));
    const labelW = compact ? 135 : 158;
    const labelH = 25;
    const labelX = left + plotW - labelW;
    const labelY = showReconstruction ? 62 : 8;
    ctx.save();
    ctx.globalAlpha = state.collapse;
    roundRect(ctx, labelX, labelY, labelW, labelH, 7);
    ctx.fillStyle = PAGE;
    ctx.fill();
    ctx.strokeStyle = "#d70015";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#d70015";
    ctx.font = compact ? "600 12px Inter, -apple-system, sans-serif" : "600 13px Inter, -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Generation collapses", labelX + labelW / 2, labelY + labelH / 2 + 0.5);
    drawArrow(ctx, labelX + labelW * 0.72, labelY + labelH, targetX - 4, targetY + 4, "#d70015");
    ctx.restore();
  }

  if (state.apexPts > 0) {
    strokeSeries(CHANNEL_PLOT.apex, state.apexPts, state.apexHead, 1, 2.8);
  }

  const legend = [...CHANNEL_PLOT.series];
  if (state.apexPts > 0) legend.push(CHANNEL_PLOT.apex);
  const legendFont = compact ? 12 : 13;

  function drawLegendItem(series, x, y) {
    const alpha = series.key ? fade : 1;
    ctx.font = legendFont + "px Inter, -apple-system, sans-serif";
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = series.color;
    ctx.fillStyle = series.color;
    ctx.lineWidth = 2.2;
    ctx.setLineDash(series.dash || []);
    ctx.beginPath();
    ctx.moveTo(x, y - 4);
    ctx.lineTo(x + 18, y - 4);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(x + 9, y - 4, 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = INK;
    ctx.textAlign = "left";
    ctx.fillText(series.label, x + 24, y);
    ctx.restore();
  }

  if (compact) {
    const columnWidth = plotW / 2;
    const legendTop = top + plotH + 62;
    legend.forEach((series, index) => {
      drawLegendItem(series, left + (index % 2) * columnWidth, legendTop + Math.floor(index / 2) * 20);
    });
    return;
  }

  let legendX = left;
  const legendY = cssH - 10;
  legend.forEach((series) => {
    ctx.font = legendFont + "px Inter, -apple-system, sans-serif";
    const itemWidth = 24 + ctx.measureText(series.label).width + 18;
    drawLegendItem(series, legendX, legendY);
    legendX += itemWidth;
  });
}

function sizePlotCanvas(canvas, reconstruction = false) {
  const wrap = canvas.parentElement;
  const cssW = Math.max(280, wrap.clientWidth);
  const baseHeight = cssW < 620 ? Math.max(320, Math.round(cssW * 0.82)) : Math.round(cssW * 0.6);
  const cssH = baseHeight + (reconstruction ? 30 : 0);
  return fitCanvas(canvas, cssW, cssH);
}
