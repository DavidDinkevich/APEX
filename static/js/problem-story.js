/** Viewport 1: higher latent compression requires a deeper channel dimension. */
const COMPRESS_STAGES = [
  { grid: 16, tokens: 256, channels: 32, front: 146, depth: 20 },
  { grid: 8, tokens: 64, channels: 128, front: 94, depth: 58 },
  { grid: 4, tokens: 16, channels: 512, front: 58, depth: 108 },
];

function drawChannelBrace(ctx, x, y, depth, skew, compact) {
  const angle = Math.atan2(-skew, depth);
  const outwardX = Math.sin(angle);
  const outwardY = -Math.cos(angle);
  const offset = compact ? 10 : 14;
  const x0 = x + outwardX * offset;
  const y0 = y + outwardY * offset;
  const x1 = x + depth + outwardX * offset;
  const y1 = y - skew + outwardY * offset;
  const tick = compact ? 3 : 4;

  ctx.save();
  ctx.strokeStyle = MUTED;
  ctx.fillStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.moveTo(x0 - outwardX * tick, y0 - outwardY * tick);
  ctx.lineTo(x0 + outwardX * tick, y0 + outwardY * tick);
  ctx.moveTo(x1 - outwardX * tick, y1 - outwardY * tick);
  ctx.lineTo(x1 + outwardX * tick, y1 + outwardY * tick);
  ctx.stroke();
  const labelGap = compact ? 7 : 9;
  ctx.translate((x0 + x1) / 2 + outwardX * labelGap, (y0 + y1) / 2 + outwardY * labelGap);
  ctx.rotate(angle);
  ctx.font = compact ? "600 9px Inter, -apple-system, sans-serif" : "600 11px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("channels", 0, 0);
  ctx.restore();
}

function drawCompressionVolume(ctx, stage, x, baseline, scale, alpha, compact = false, showBrace = false) {
  const front = stage.front * scale;
  const depth = stage.depth * scale;
  const skew = depth * 0.42;
  const y = baseline - front;
  const centerX = x + (front + depth) / 2;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(0, (1 - alpha) * 12);
  drawAxonometricCuboid(ctx, {
    x,
    y,
    frontW: front,
    frontH: front,
    depth,
    skew,
    rows: stage.grid,
    cols: stage.grid,
    slices: [{ fraction: 1, color: "#2563A6" }],
    frontFill: "#f0f4f9",
  });
  if (showBrace) drawChannelBrace(ctx, x, y, depth, skew, compact);

  ctx.fillStyle = INK;
  ctx.font = compact ? "600 12px Inter, -apple-system, sans-serif" : "600 16px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(stage.tokens + " tokens", centerX, baseline + (compact ? 19 : 25));
  ctx.fillStyle = MUTED;
  ctx.font = compact ? "9px Inter, -apple-system, sans-serif" : "12px Inter, -apple-system, sans-serif";
  if (compact) {
    ctx.fillText(stage.grid + "×" + stage.grid + " grid", centerX, baseline + 34);
    ctx.fillText(stage.channels + " channels", centerX, baseline + 47);
  } else {
    ctx.fillText(stage.grid + "×" + stage.grid + " grid · " + stage.channels + " channels", centerX, baseline + 44);
  }
  ctx.restore();
}

function drawCompressionSequence(ctx, cssW, cssH, reveals) {
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = PAGE;
  ctx.fillRect(0, 0, cssW, cssH);

  const mobile = cssW < 500;
  ctx.fillStyle = MUTED;
  ctx.font = "600 14px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  const heading = "Higher latent compression";
  const headingX = 18;
  ctx.fillText(heading, headingX, 24);
  const arrowStart = headingX + ctx.measureText(heading).width + 14;
  if (cssW - 20 - arrowStart > 18) drawArrow(ctx, arrowStart, 19, cssW - 20, 19, MUTED);

  const horizontalPadding = mobile ? 10 : 18;
  const scale = clip(cssW / 620, mobile ? 0.55 : 0.82, 1);
  const slotW = (cssW - horizontalPadding * 2) / 3;
  const baseline = Math.min(220, cssH - 130);
  const positions = COMPRESS_STAGES.map((stage, index) => ({
    x: horizontalPadding + index * slotW + (slotW - (stage.front + stage.depth) * scale) / 2,
    baseline,
  }));
  for (let i = 1; i < positions.length; i++) {
    const previousStage = COMPRESS_STAGES[i - 1];
    const previous = positions[i - 1];
    const current = positions[i];
    drawArrow(
      ctx,
      previous.x + (previousStage.front + previousStage.depth) * scale + 4,
      baseline - (mobile ? 28 : 44),
      current.x - 5,
      baseline - (mobile ? 28 : 44),
      `rgba(110,110,115,${reveals[i]})`
    );
  }
  COMPRESS_STAGES.forEach((stage, index) => {
    drawCompressionVolume(ctx, stage, positions[index].x, baseline, scale, reveals[index], mobile, index === 2);
  });
}

function startProblemStory() {
  const compress = document.getElementById("compress-anim");
  const plot = document.getElementById("channel-anim");
  const section = document.getElementById("problem");
  if (!compress || !plot || !section) return;

  function layout() {
    const cWrap = compress.parentElement;
    const cW = Math.max(280, cWrap.clientWidth);
    const p = sizePlotCanvas(plot, true);
    const c = fitCanvas(compress, cW, p.cssH);
    return { c, p };
  }

  function paint(reveals, plotState) {
    const { c, p } = layout();
    drawCompressionSequence(c.ctx, c.cssW, c.cssH, reveals);
    drawProblemTradeoff(p.ctx, p.cssW, p.cssH, plotState);
  }

  function stateAt(ms) {
    const first = ease(ms / 400);
    const second = ease((ms - 700) / 1000);
    const third = ease((ms - 2200) / 1200);
    const basePts = third >= 1 ? 3 : second >= 1 ? 2 : 1;
    const baseHead = third > 0 && third < 1 ? third : second > 0 && second < 1 ? second : 0;
    return {
      reveals: [first, second, third],
      plot: {
        basePts,
        baseHead,
        baseFade: first,
        collapse: ease((ms - 2900) / 500),
        apexPts: 0,
        apexHead: 0,
      },
    };
  }

  const player = createOneShotPlayer({
    section,
    replayButton: document.getElementById("problem-replay"),
    durationMs: 4300,
    draw(ms) {
      const state = stateAt(ms);
      paint(state.reveals, state.plot);
    },
  });
  window.addEventListener("resize", () => player.repaint());
}

document.addEventListener("DOMContentLoaded", () => {
  startProblemStory();
});
