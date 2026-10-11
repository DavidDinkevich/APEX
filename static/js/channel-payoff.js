/**
 * Viewport 4: same f64 channel plot, baselines already known, then APEX.
 */
function payoffState(ms) {
  const base = { basePts: 3, baseHead: 0, baseFade: 0.38, collapse: 0 };
  if (ms < 900) return { ...base, apexPts: 0, apexHead: 0 };
  if (ms < 1500) return { ...base, apexPts: 1, apexHead: 0 };
  if (ms < 3200) {
    return { ...base, apexPts: 1, apexHead: ease((ms - 1500) / 1700) };
  }
  if (ms < 5000) {
    return { ...base, apexPts: 2, apexHead: ease((ms - 3200) / 1800) };
  }
  return { ...base, apexPts: 3, apexHead: 0 };
}

async function startChannelPayoff() {
  const canvas = document.getElementById("channel-payoff");
  const section = document.getElementById("payoff");
  if (!canvas || !section) return;

  function paint(state) {
    const { ctx, cssW, cssH } = sizePlotCanvas(canvas);
    drawChannelPlot(ctx, cssW, cssH, state);
  }

  const player = createOneShotPlayer({
    section,
    replayButton: document.getElementById("payoff-replay"),
    durationMs: 6000,
    draw(ms) {
      paint(payoffState(ms));
    },
  });
  window.addEventListener("resize", () => player.repaint());
}

document.addEventListener("DOMContentLoaded", () => {
  startChannelPayoff().catch(console.error);
});
