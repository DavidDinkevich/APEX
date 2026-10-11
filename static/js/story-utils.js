const APEX_LEVEL = {
  labels: ["Level 0", "Level 1", "Level 2"],
  colors: ["#2563A6", "#2A9D8F", "#E69F00"],
  g: 0.25,
};

const PAGE = "#fbfbfd";
const INK = "#1d1d1f";
const MUTED = "#6e6e73";
const HAIR = "#e2e2e7";

function clip(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function mixHexColor(a, b, t) {
  const av = parseInt(a.slice(1), 16);
  const bv = parseInt(b.slice(1), 16);
  const channels = [16, 8, 0].map((shift) => Math.round(lerp((av >> shift) & 255, (bv >> shift) & 255, clip(t, 0, 1))));
  return "#" + channels.map((value) => value.toString(16).padStart(2, "0")).join("");
}

function ease(t) {
  const x = clip(t, 0, 1);
  return x * x * (3 - 2 * x);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load " + src));
    img.src = src;
  });
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, Math.max(0, w / 2), Math.max(0, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function fitCanvas(canvas, cssW, cssH) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, dpr, cssW, cssH };
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function drawArrow(ctx, x0, y0, x1, y1, color = INK) {
  const angle = Math.atan2(y1 - y0, x1 - x0);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - 8 * Math.cos(angle - 0.45), y1 - 8 * Math.sin(angle - 0.45));
  ctx.lineTo(x1 - 8 * Math.cos(angle + 0.45), y1 - 8 * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawGridOverlay(ctx, x, y, w, h, rows, cols, color = "rgba(29,29,31,0.35)") {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = rows > 10 || cols > 10 ? 0.45 : 0.75;
  for (let col = 1; col < cols; col++) {
    const px = x + (col / cols) * w;
    ctx.beginPath();
    ctx.moveTo(px, y);
    ctx.lineTo(px, y + h);
    ctx.stroke();
  }
  for (let row = 1; row < rows; row++) {
    const py = y + (row / rows) * h;
    ctx.beginPath();
    ctx.moveTo(x, py);
    ctx.lineTo(x + w, py);
    ctx.stroke();
  }
  ctx.restore();
}

function drawStaticNoiseBlock(ctx, x, y, width, height, color, cleanAmount, seed) {
  const colorValue = parseInt(color.slice(1), 16);
  const target = {
    r: (colorValue >> 16) & 255,
    g: (colorValue >> 8) & 255,
    b: colorValue & 255,
  };
  const image = ctx.createImageData(Math.max(1, Math.round(width)), Math.max(1, Math.round(height)));
  const pixels = image.data;
  const clean = clip(cleanAmount, 0, 1);
  for (let index = 0; index < pixels.length; index += 4) {
    const unitNoise = (((Math.sin((index + seed) * 12.9898) * 43758.5453) % 1) + 1) % 1;
    const noise = unitNoise * 255;
    pixels[index] = Math.round(lerp(noise, target.r, clean));
    pixels[index + 1] = Math.round(lerp(noise, target.g, clean));
    pixels[index + 2] = Math.round(lerp(noise, target.b, clean));
    pixels[index + 3] = 255;
  }
  const noiseCanvas = document.createElement("canvas");
  noiseCanvas.width = image.width;
  noiseCanvas.height = image.height;
  noiseCanvas.getContext("2d").putImageData(image, 0, 0);
  ctx.drawImage(noiseCanvas, x, y, width, height);
}

/**
 * Draw an axonometric latent volume. The front face is the latent token grid;
 * the extrusion is channels. Slices partition only the channel dimension.
 */
function drawAxonometricCuboid(
  ctx,
  {
    x,
    y,
    frontW,
    frontH,
    depth,
    skew = depth * 0.42,
    rows = 1,
    cols = 1,
    frontFill = "#f5f5f7",
    frontImage = null,
    slices = [{ fraction: 1, color: "#2563A6", label: "" }],
    showSliceLabels = false,
    outline = INK,
  }
) {
  const dx = depth;
  const dy = -skew;
  const inheritedAlpha = ctx.globalAlpha;
  let fractionStart = 0;

  for (const slice of slices) {
    const fractionEnd = fractionStart + slice.fraction;
    const x0 = x + dx * fractionStart;
    const y0 = y + dy * fractionStart;
    const x1 = x + dx * fractionEnd;
    const y1 = y + dy * fractionEnd;
    const alpha = slice.alpha == null ? 1 : slice.alpha;

    ctx.save();
    ctx.globalAlpha = inheritedAlpha * alpha;
    ctx.fillStyle = slice.color;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + frontW, y0);
    ctx.lineTo(x1 + frontW, y1);
    ctx.lineTo(x1, y1);
    ctx.closePath();
    ctx.fill();

    ctx.globalAlpha = inheritedAlpha * alpha * 0.86;
    ctx.beginPath();
    ctx.moveTo(x0 + frontW, y0);
    ctx.lineTo(x1 + frontW, y1);
    ctx.lineTo(x1 + frontW, y1 + frontH);
    ctx.lineTo(x0 + frontW, y0 + frontH);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    ctx.strokeStyle = "rgba(29,29,31,0.55)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 + frontW, y1);
    ctx.lineTo(x1 + frontW, y1 + frontH);
    ctx.stroke();

    if (showSliceLabels && slice.label) {
      const middle = (fractionStart + fractionEnd) / 2;
      ctx.fillStyle = slice.labelColor || slice.color;
      ctx.font = "600 12px Inter, -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(slice.label, x + frontW + dx * middle, y + dy * middle - 7);
    }
    fractionStart = fractionEnd;
  }

  ctx.save();
  ctx.fillStyle = frontFill;
  ctx.fillRect(x, y, frontW, frontH);
  if (frontImage) {
    ctx.beginPath();
    ctx.rect(x, y, frontW, frontH);
    ctx.clip();
    ctx.drawImage(frontImage, x, y, frontW, frontH);
  }
  ctx.restore();
  drawGridOverlay(ctx, x, y, frontW, frontH, rows, cols);

  ctx.strokeStyle = outline;
  ctx.lineWidth = 1.2;
  ctx.strokeRect(x, y, frontW, frontH);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + dx, y + dy);
  ctx.lineTo(x + frontW + dx, y + dy);
  ctx.lineTo(x + frontW + dx, y + frontH + dy);
  ctx.lineTo(x + frontW, y + frontH);
  ctx.stroke();
}

function drawNetworkBlock(ctx, x, y, w, h, direction, label) {
  const inset = h * 0.18;
  ctx.beginPath();
  if (direction === "encode") {
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y + inset);
    ctx.lineTo(x + w, y + h - inset);
    ctx.lineTo(x, y + h);
  } else {
    ctx.moveTo(x, y + inset);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x, y + h - inset);
  }
  ctx.closePath();
  ctx.fillStyle = "#f5f5f7";
  ctx.fill();
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.fillStyle = INK;
  ctx.font = "600 15px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + w / 2, y + h / 2);
}

/**
 * Start once on first viewport entry, then freeze on the final frame. Leaving
 * the viewport never resets. Replay is explicit and deterministic.
 */
function createOneShotPlayer({ section, replayButton, durationMs, draw, threshold = 0.28 }) {
  let started = false;
  let currentMs = 0;
  let raf = 0;
  let generation = 0;

  function paint(ms) {
    currentMs = clip(ms, 0, durationMs);
    draw(currentMs);
  }

  function play() {
    cancelAnimationFrame(raf);
    generation += 1;
    const run = generation;
    started = true;
    if (reducedMotion()) {
      paint(durationMs);
      return;
    }
    paint(0);
    const start = performance.now();
    function tick(now) {
      if (run !== generation) return;
      const elapsed = now - start;
      paint(elapsed);
      if (elapsed < durationMs) {
        raf = requestAnimationFrame(tick);
      } else {
        paint(durationMs);
      }
    }
    raf = requestAnimationFrame(tick);
  }

  paint(reducedMotion() ? durationMs : 0);
  if (reducedMotion() && replayButton) {
    replayButton.disabled = true;
    replayButton.title = "Animation disabled by reduced-motion preference";
  }

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!started && e.isIntersecting && e.intersectionRatio >= threshold) {
          play();
          io.disconnect();
        }
      }
    },
    { threshold: [threshold, 0.5, 0.75] }
  );
  if (!reducedMotion()) io.observe(section);
  if (replayButton) replayButton.addEventListener("click", play);

  return {
    play,
    seek(ms) {
      cancelAnimationFrame(raf);
      generation += 1;
      started = true;
      paint(ms);
    },
    repaint() {
      paint(currentMs);
    },
    get currentMs() {
      return currentMs;
    },
  };
}

function trapezoid(ctx, x, y, w, h, flare) {
  ctx.beginPath();
  ctx.moveTo(x + flare, y);
  ctx.lineTo(x + w - flare, y);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.closePath();
}
