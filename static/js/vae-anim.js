/** Viewport 2: the resolution-supervised, cumulative-prefix autoencoder. */
const VAE_STILLS = [
  "static/images/hero/l0.png?v=real-decodes-1",
  "static/images/hero/l1.png?v=real-decodes-1",
  "static/images/hero/l2.png?v=real-decodes-1",
];

const VAE_PREFIXES = [
  { name: "Level 0", resolution: "¼ resolution", thumbSize: 46 },
  { name: "Level 1", resolution: "½ resolution", thumbSize: 64 },
  { name: "Level 2", resolution: "full resolution", thumbSize: 82 },
];

function retainedPrefixAt(ms) {
  if (ms < 1500) return 0;
  if (ms < 3100) return 1;
  return 2;
}

function vaeCaption(retainedLevel) {
  if (retainedLevel === 0) return "Levels 1–2 are zeroed.";
  if (retainedLevel === 1) return "Level 2 is zeroed.";
  return "All three levels are kept.";
}

function drawImageCube(ctx, image, x, y, size, depth, alpha, label) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  drawAxonometricCuboid(ctx, {
    x,
    y,
    frontW: size,
    frontH: size,
    depth,
    skew: depth * 0.42,
    rows: 8,
    cols: 8,
    frontImage: image,
    frontFill: "#f5f5f7",
    slices: [{ fraction: 1, color: "#d8dde4" }],
  });
  ctx.fillStyle = MUTED;
  ctx.font = "600 11px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, x + size / 2, y + size + 19);
  ctx.restore();
}

function drawElbowArrow(ctx, x0, y0, x1, y1, label, alpha) {
  const direction = Math.sign(x1 - x0) || 1;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = MUTED;
  ctx.fillStyle = MUTED;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0, y1);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - direction * 8, y1 - 4);
  ctx.lineTo(x1 - direction * 8, y1 + 4);
  ctx.closePath();
  ctx.fill();
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, (x0 + x1) / 2, y1 - 7);
  ctx.restore();
}

function drawLossPair(ctx, target, recon, xTarget, xRecon, y, size, resolution, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;
  roundRect(ctx, xTarget, y, size, size, 6);
  ctx.save();
  ctx.clip();
  ctx.drawImage(target, xTarget, y, size, size);
  ctx.restore();
  ctx.strokeStyle = HAIR;
  ctx.stroke();

  roundRect(ctx, xRecon, y, size, size, 6);
  ctx.save();
  ctx.clip();
  ctx.drawImage(recon, xRecon, y, size, size);
  ctx.restore();
  ctx.strokeStyle = HAIR;
  ctx.stroke();

  ctx.fillStyle = MUTED;
  ctx.font = "10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(resolution + " target", xTarget + size / 2, y + size + 15);
  ctx.fillText(resolution + " recon", xRecon + size / 2, y + size + 15);
  ctx.fillStyle = INK;
  ctx.font = "600 17px Inter, -apple-system, sans-serif";
  ctx.fillText("Loss", (xTarget + size + xRecon) / 2, y + size / 2);
  ctx.restore();
}

function latentSlices(retainedLevel, zeroProgress) {
  const fractions = [0.18, 0.28, 0.54];
  return fractions.map((fraction, level) => ({
    fraction,
    color: level <= retainedLevel ? APEX_LEVEL.colors[level] : mixHexColor(APEX_LEVEL.colors[level], "#d2d3d7", zeroProgress),
    label: String(level),
    labelColor: level > retainedLevel && zeroProgress > 0.5 ? MUTED : "#fff",
  }));
}

function drawLevelLegend(ctx, x, y, compact = false) {
  let legendX = x;
  const swatch = compact ? 9 : 12;
  ctx.font = compact ? "600 10px Inter, -apple-system, sans-serif" : "600 13px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (let level = 0; level < APEX_LEVEL.labels.length; level++) {
    ctx.fillStyle = APEX_LEVEL.colors[level];
    roundRect(ctx, legendX, y - swatch / 2, swatch, swatch, 2);
    ctx.fill();
    ctx.fillStyle = INK;
    ctx.fillText(APEX_LEVEL.labels[level], legendX + swatch + 5, y);
    legendX += swatch + 5 + ctx.measureText(APEX_LEVEL.labels[level]).width + (compact ? 10 : 16);
  }
}

function drawVaeDesktop(ctx, w, h, stills, retainedLevel) {
  const drawingScale = clip(w / 1000, 0.78, 1.2);
  ctx.save();
  ctx.scale(drawingScale, drawingScale);
  w /= drawingScale;
  h /= drawingScale;

  const imageSize = 132;
  const latentFront = 136;
  const latentDepth = Math.min(245, Math.max(175, w - 690));
  const blockWidth = 54;
  const contentWidth = imageSize * 2 + latentFront + latentDepth + blockWidth * 2 + 122;
  const origin = Math.max(14, (w - contentWidth) / 2);
  const imageX = origin;
  const imageY = 122;
  const encoderX = imageX + imageSize + 30;
  const encoderY = 132;
  const latentX = encoderX + blockWidth + 30;
  const latentY = 126;
  const decoderX = latentX + latentFront + latentDepth + 32;
  const decoderY = 132;
  const reconX = decoderX + blockWidth + 30;
  const reconY = imageY;

  drawLevelLegend(ctx, latentX, 20);
  drawImageCube(ctx, stills[2], imageX, imageY, imageSize, 16, 1, "image");
  drawNetworkBlock(ctx, encoderX, encoderY, blockWidth, 98, "encode", "E");

  drawAxonometricCuboid(ctx, {
    x: latentX,
    y: latentY,
    frontW: latentFront,
    frontH: latentFront,
    depth: latentDepth,
    skew: latentDepth * 0.38,
    rows: 8,
    cols: 8,
    frontFill: "#f5f7fa",
    showSliceLabels: false,
    slices: latentSlices(retainedLevel, 1),
  });

  drawNetworkBlock(ctx, decoderX, decoderY, blockWidth, 98, "decode", "D");
  drawImageCube(ctx, stills[retainedLevel], reconX, reconY, imageSize, 16, 1, "full-size decode");

  const prefix = VAE_PREFIXES[retainedLevel];
  const thumb = prefix.thumbSize;
  const lossY = h - thumb - 28;
  const innerGap = 110;
  const targetX = w / 2 - innerGap / 2 - thumb;
  const reconThumbX = w / 2 + innerGap / 2;
  const routeY = lossY + thumb / 2;
  drawElbowArrow(ctx, imageX + imageSize / 2, imageY + imageSize + 24, targetX - 8, routeY, "resize", 1);
  drawElbowArrow(ctx, reconX + imageSize / 2, reconY + imageSize + 24, reconThumbX + thumb + 8, routeY, "resize", 1);
  drawLossPair(ctx, stills[2], stills[retainedLevel], targetX, reconThumbX, lossY, thumb, prefix.resolution, 1);
  ctx.restore();
}

function drawVaeMobile(ctx, w, h, stills, retainedLevel) {
  const imageSize = 78;
  const imageX = 12;
  const imageY = 70;
  const encoderX = 100;
  const latentX = 154;
  const latentFront = 70;
  const latentDepth = Math.max(54, w - latentX - latentFront - 12);
  const latentY = 82;
  drawLevelLegend(ctx, latentX, 48, true);
  drawImageCube(ctx, stills[2], imageX, imageY, imageSize, 10, 1, "image");

  drawNetworkBlock(ctx, encoderX, imageY + 9, 36, 60, "encode", "E");
  drawAxonometricCuboid(ctx, {
    x: latentX,
    y: latentY,
    frontW: latentFront,
    frontH: latentFront,
    depth: latentDepth,
    rows: 8,
    cols: 8,
    showSliceLabels: false,
    slices: latentSlices(retainedLevel, 1),
  });

  const decoderX = w - imageSize - 74;
  const decoderY = 218;
  const reconX = w - imageSize - 12;
  const reconY = 204;
  drawNetworkBlock(ctx, decoderX, decoderY, 40, 62, "decode", "D");
  drawImageCube(ctx, stills[retainedLevel], reconX, reconY, imageSize, 10, 1, "full-size decode");

  const prefix = VAE_PREFIXES[retainedLevel];
  const thumb = Math.min(prefix.thumbSize, 64);
  const lossY = h - thumb - 28;
  const innerGap = 58;
  const targetX = w / 2 - innerGap / 2 - thumb;
  const reconThumbX = w / 2 + innerGap / 2;
  const routeY = lossY + thumb / 2;
  drawElbowArrow(ctx, imageX + imageSize / 2, imageY + imageSize + 24, targetX - 8, routeY, "resize", 1);
  drawElbowArrow(ctx, reconX + imageSize / 2, reconY + imageSize + 24, reconThumbX + thumb + 8, routeY, "resize", 1);
  drawLossPair(ctx, stills[2], stills[retainedLevel], targetX, reconThumbX, lossY, thumb, prefix.resolution, 1);
}

function drawVaeFrame(ctx, w, h, stills, retainedLevel) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = PAGE;
  ctx.fillRect(0, 0, w, h);
  if (w < 680) {
    drawVaeMobile(ctx, w, h, stills, retainedLevel);
  } else {
    drawVaeDesktop(ctx, w, h, stills, retainedLevel);
  }
}

async function startVaeAnim() {
  const canvas = document.getElementById("vae-anim");
  const section = document.getElementById("representation");
  const status = document.getElementById("representation-status");
  const prefixButtons = Array.from(document.querySelectorAll("[data-prefix-level]"));
  if (!canvas || !section) return;
  const stills = await Promise.all(VAE_STILLS.map(loadImage));
  const replayButton = document.getElementById("representation-replay");
  let manualLevel = null;
  let retainedLevel = 0;

  function layout() {
    const wrap = canvas.parentElement;
    const cssW = Math.max(320, wrap.clientWidth);
    const drawingScale = clip(cssW / 1000, 0.78, 1.2);
    const cssH = cssW < 680 ? 500 : 460 * drawingScale;
    return fitCanvas(canvas, cssW, cssH);
  }

  function updateButtons() {
    for (const button of prefixButtons) {
      button.setAttribute("aria-pressed", String(Number(button.dataset.prefixLevel) === retainedLevel));
    }
  }

  function paint(ms) {
    retainedLevel = manualLevel == null ? retainedPrefixAt(ms) : manualLevel;
    updateButtons();
    const { ctx, cssW, cssH } = layout();
    drawVaeFrame(ctx, cssW, cssH, stills, retainedLevel);
    if (status) status.textContent = vaeCaption(retainedLevel);
  }

  if (replayButton)
    replayButton.addEventListener("click", () => {
      manualLevel = null;
    });

  const player = createOneShotPlayer({
    section,
    replayButton,
    durationMs: 4600,
    draw: paint,
    threshold: 0.55,
  });

  for (const button of prefixButtons) {
    button.addEventListener("click", () => {
      const nextLevel = Number(button.dataset.prefixLevel);
      if (!Number.isInteger(nextLevel) || nextLevel < 0 || nextLevel > 2) {
        throw new Error("Invalid prefix level: " + button.dataset.prefixLevel);
      }
      manualLevel = nextLevel;
      retainedLevel = nextLevel;
      player.seek(4600);
    });
  }
  window.addEventListener("resize", () => player.repaint());
}

document.addEventListener("DOMContentLoaded", () => {
  const section = document.getElementById("representation");
  if (!section) return;
  const observer = new IntersectionObserver(
    (entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      observer.disconnect();
      startVaeAnim().catch(console.error);
    },
    { rootMargin: "600px 0px", threshold: 0 }
  );
  observer.observe(section);
});
