/** Viewport 3: one sampled global time drives an asynchronous training step. */
const TRAINING_EXAMPLE = {
  globalProgress: 0.3,
  globalTau: 0.45,
  localTimes: [0.55, 0.8, null],
  displayNoiseAmounts: [0.18, 0.9, null],
  durationMs: 7400,
};
const LATENT_LEVEL_FRACTIONS = [0.18, 0.28, 0.54];

function trainingState(ms) {
  return {
    sample: ease((ms - 250) / 1200),
    map: ease((ms - 1850) / 1250),
    noise: ease((ms - 3550) / 1250),
    predict: ease((ms - 5250) / 1400),
  };
}

function trainingStatus(state) {
  if (state.sample < 0.98) return "Sample one global time.";
  if (state.map < 0.98) return "Map it to one local noise time per level.";
  if (state.noise < 0.98) return "Noise each active level by its local amount.";
  if (state.predict < 0.98) return "Send the noised latent and global time into one shared Transformer.";
  return "";
}

function drawStepHeading(ctx, text, x, y) {
  ctx.fillStyle = INK;
  ctx.font = "600 11px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(text, x, y);
}

function drawTimeAxis(ctx, x, y, width) {
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + width, y);
  ctx.stroke();

  ctx.fillStyle = MUTED;
  ctx.font = "10px Inter, -apple-system, sans-serif";
  ctx.textBaseline = "alphabetic";
  [0, 0.5, 1].forEach((progress) => {
    const tickX = x + progress * width;
    ctx.beginPath();
    ctx.moveTo(tickX, y - 3);
    ctx.lineTo(tickX, y + 3);
    ctx.stroke();
    ctx.textAlign = progress === 0 ? "left" : progress === 1 ? "right" : "center";
    ctx.fillText(Math.round(progress * 100) + "%", tickX, y + 17);
  });
}

function abstractDensity(progress) {
  // Hand-shaped silhouette for the visual, not the training distribution.
  const t = clip(progress, 0, 1);
  const inverse = 1 - t;
  return inverse ** 3 * 0.96 + 3 * inverse ** 2 * t * 0.82 + 3 * inverse * t ** 2 * 0.32 + t ** 3 * 0.14;
}

function drawGlobalTimeSample(ctx, x, y, width, state) {
  drawStepHeading(ctx, "1  SAMPLE GLOBAL TIME", x, y);
  const trackX = x + 18;
  const trackWidth = width - 36;
  const axisY = y + 128;
  const densityHeight = 58;

  ctx.save();
  ctx.globalAlpha = 0.16;
  ctx.beginPath();
  ctx.moveTo(trackX, axisY);
  for (let index = 0; index <= 48; index++) {
    const progress = index / 48;
    ctx.lineTo(trackX + progress * trackWidth, axisY - densityHeight * abstractDensity(progress));
  }
  ctx.lineTo(trackX + trackWidth, axisY);
  ctx.closePath();
  ctx.fillStyle = APEX_LEVEL.colors[0];
  ctx.fill();
  ctx.strokeStyle = APEX_LEVEL.colors[0];
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();

  drawTimeAxis(ctx, trackX, axisY, trackWidth);

  const sampleX = trackX + TRAINING_EXAMPLE.globalProgress * trackWidth;
  const targetY = axisY - densityHeight * abstractDensity(TRAINING_EXAMPLE.globalProgress);
  const dotY = lerp(y + 18, targetY, state.sample);
  ctx.save();
  ctx.globalAlpha = state.sample;
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = INK;
  ctx.beginPath();
  ctx.moveTo(sampleX, targetY);
  ctx.lineTo(sampleX, axisY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = INK;
  ctx.beginPath();
  ctx.arc(sampleX, dotY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("sample = 30%", sampleX, axisY + 35);
  ctx.restore();
  return { sampleX, axisY };
}

function drawLocalTimeMap(ctx, x, y, width, state, mobile) {
  drawStepHeading(ctx, "2  MAP TO PER-LEVEL NOISE TIMES", x, y);
  const labelWidth = mobile ? 58 : 64;
  const valueWidth = mobile ? 82 : 94;
  const barX = x + labelWidth;
  const barWidth = width - labelWidth - valueWidth;
  const axisY = y + (mobile ? 54 : 128);
  const rowStart = axisY + 43;
  const rowGap = mobile ? 40 : 42;
  const starts = [0, 1 / 6, 1 / 3];
  const ends = [2 / 3, 5 / 6, 1];
  const sampleX = barX + TRAINING_EXAMPLE.globalProgress * barWidth;

  drawTimeAxis(ctx, barX, axisY, barWidth);

  for (let level = 0; level < 3; level++) {
    const rowY = rowStart + level * rowGap;
    const color = APEX_LEVEL.colors[level];
    const intervalX = barX + starts[level] * barWidth;
    const intervalWidth = (ends[level] - starts[level]) * barWidth;
    ctx.fillStyle = color;
    ctx.font = "600 10px Inter, -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(APEX_LEVEL.labels[level], barX - 8, rowY);

    roundRect(ctx, barX, rowY - 9, barWidth, 18, 9);
    ctx.fillStyle = "#ececef";
    ctx.fill();
    roundRect(ctx, intervalX, rowY - 9, intervalWidth, 18, 9);
    ctx.fillStyle = mixHexColor(color, "#ffffff", 0.42);
    ctx.fill();

    ctx.save();
    ctx.globalAlpha = state.map;
    const localTime = TRAINING_EXAMPLE.localTimes[level];
    ctx.fillStyle = localTime == null ? MUTED : color;
    ctx.font = "600 9px Inter, -apple-system, sans-serif";
    ctx.textAlign = "left";
    if (localTime == null) ctx.fillText("held out", barX + barWidth + 9, rowY);
    ctx.restore();
  }

  ctx.save();
  ctx.globalAlpha = state.map;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.7;
  ctx.beginPath();
  ctx.moveTo(sampleX, axisY - 5);
  ctx.lineTo(sampleX, rowStart + 2 * rowGap + 13);
  ctx.stroke();
  ctx.fillStyle = INK;
  ctx.beginPath();
  ctx.arc(sampleX, axisY, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("30%", sampleX, axisY - 10);
  ctx.restore();
  return { sampleX, axisY };
}

function drawSampleTransfer(ctx, from, to, progress) {
  const amount = ease(progress);
  ctx.save();
  ctx.globalAlpha = amount;
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1.2;
  ctx.setLineDash([3, 5]);
  ctx.beginPath();
  ctx.moveTo(from.sampleX, from.axisY);
  ctx.lineTo(to.sampleX, to.axisY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = INK;
  ctx.beginPath();
  ctx.arc(lerp(from.sampleX, to.sampleX, amount), lerp(from.axisY, to.axisY, amount), 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawNoiseDensityBlock(ctx, x, y, width, height, density, seed) {
  const imageWidth = Math.max(1, Math.round(width));
  const imageHeight = Math.max(1, Math.round(height));
  const image = ctx.createImageData(imageWidth, imageHeight);
  for (let index = 0; index < image.data.length; index += 4) {
    const pixel = index / 4;
    const sample = (((Math.sin((pixel + seed) * 12.9898) * 43758.5453) % 1) + 1) % 1;
    const shadeSample = (((Math.sin((pixel + seed) * 78.233) * 12345.6789) % 1) + 1) % 1;
    const shade = sample < density ? Math.round(shadeSample * 190) : 245;
    image.data[index] = shade;
    image.data[index + 1] = shade;
    image.data[index + 2] = shade;
    image.data[index + 3] = 255;
  }
  const noiseCanvas = document.createElement("canvas");
  noiseCanvas.width = imageWidth;
  noiseCanvas.height = imageHeight;
  noiseCanvas.getContext("2d").putImageData(image, 0, 0);
  ctx.drawImage(noiseCanvas, x, y, width, height);
}

function drawLatentSegment(ctx, x, y, width, height, level, noiseAmount, heldOut, alpha, neutralNoise) {
  const color = APEX_LEVEL.colors[level];
  ctx.save();
  ctx.globalAlpha = alpha;
  if (heldOut) {
    ctx.fillStyle = "#ececef";
    ctx.fillRect(x, y, width, height);
  } else if (noiseAmount > 0) {
    if (neutralNoise) {
      drawNoiseDensityBlock(ctx, x, y, width, height, noiseAmount, level * 211 + 17);
    } else {
      drawStaticNoiseBlock(ctx, x, y, width, height, color, 1 - noiseAmount, level * 211 + 17);
    }
  } else {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, width, height);
  }
  ctx.globalAlpha = alpha;
  const useLightText = !heldOut && (!neutralNoise || level === 1);
  ctx.fillStyle = useLightText ? "#fff" : INK;
  ctx.shadowColor = heldOut ? "transparent" : useLightText ? "rgba(0,0,0,0.75)" : "rgba(255,255,255,0.9)";
  ctx.shadowBlur = heldOut ? 0 : 2;
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const label = heldOut ? APEX_LEVEL.labels[level] + " · held out" : APEX_LEVEL.labels[level];
  ctx.fillText(label, x + width / 2, y + height / 2);
  ctx.restore();
}

function drawLatentBar(ctx, x, y, width, height, mode, alpha) {
  let segmentX = x;
  for (let level = 0; level < 3; level++) {
    const segmentWidth = width * LATENT_LEVEL_FRACTIONS[level];
    const heldOut = mode !== "clean" && level === 2;
    let noiseAmount = 0;
    if (mode === "levelNoise" && level < 2) noiseAmount = TRAINING_EXAMPLE.displayNoiseAmounts[level];
    drawLatentSegment(ctx, segmentX, y, segmentWidth, height, level, noiseAmount, heldOut, alpha, mode === "levelNoise");
    segmentX += segmentWidth;
  }
  ctx.save();
  ctx.globalAlpha = alpha;
  roundRect(ctx, x, y, width, height, 7);
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();
}

function drawPipelineArrow(ctx, x0, y0, x1, y1, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;
  drawArrow(ctx, x0, y0, x1, y1, MUTED);
  ctx.restore();
}

function drawLevelWiseAddition(ctx, x, cleanBottom, noiseTop, width, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = INK;
  ctx.fillStyle = PAGE;
  ctx.lineWidth = 1.2;
  ctx.font = "500 15px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const operatorY = (cleanBottom + noiseTop) / 2;
  let fractionStart = 0;
  for (let level = 0; level < 2; level++) {
    const fraction = LATENT_LEVEL_FRACTIONS[level];
    const centerX = x + width * (fractionStart + fraction / 2);
    ctx.beginPath();
    ctx.moveTo(centerX, cleanBottom + 3);
    ctx.lineTo(centerX, operatorY - 11);
    ctx.moveTo(centerX, operatorY + 11);
    ctx.lineTo(centerX, noiseTop - 3);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, operatorY, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = INK;
    ctx.fillText("+", centerX, operatorY - 1);
    ctx.fillStyle = PAGE;
    fractionStart += fraction;
  }
  ctx.restore();
}

function drawTrainingPipelineDesktop(ctx, width, y, state) {
  drawStepHeading(ctx, "3  NOISE THE LATENT AND PREDICT VELOCITIES", 18, y);
  const tokenWidth = Math.min(280, width * 0.29);
  const tokenHeight = 46;
  const leftX = 24;
  const cleanY = y + 42;
  const noiseY = y + 142;
  const mixY = (cleanY + tokenHeight + noiseY) / 2;
  const outputX = width - 24 - tokenWidth;
  const outputY = mixY - tokenHeight / 2;
  const ditWidth = 160;
  const ditHeight = 66;
  const ditX = (width - ditWidth) / 2;
  const ditY = mixY - ditHeight / 2;
  const mergeX = leftX + tokenWidth + 24;

  ctx.fillStyle = MUTED;
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("clean latent  z", leftX, cleanY - 10);
  drawLatentBar(ctx, leftX, cleanY, tokenWidth, tokenHeight, "clean", 1);

  ctx.save();
  ctx.globalAlpha = state.noise;
  ctx.fillStyle = MUTED;
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("level-wise Gaussian noise  ε", leftX + tokenWidth, noiseY - 10);
  ctx.restore();
  drawLatentBar(ctx, leftX, noiseY, tokenWidth, tokenHeight, "levelNoise", state.noise);
  drawLevelWiseAddition(ctx, leftX, cleanY + tokenHeight, noiseY, tokenWidth, state.noise);

  ctx.save();
  ctx.globalAlpha = state.predict;
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(leftX + tokenWidth + 8, cleanY + tokenHeight / 2);
  ctx.lineTo(mergeX, cleanY + tokenHeight / 2);
  ctx.lineTo(mergeX, mixY);
  ctx.moveTo(leftX + tokenWidth + 8, noiseY + tokenHeight / 2);
  ctx.lineTo(mergeX, noiseY + tokenHeight / 2);
  ctx.lineTo(mergeX, mixY);
  ctx.stroke();
  ctx.restore();
  drawPipelineArrow(ctx, mergeX, mixY, ditX - 10, mixY, state.predict);

  ctx.save();
  ctx.globalAlpha = state.predict;
  roundRect(ctx, ditX, ditY, ditWidth, ditHeight, 10);
  ctx.fillStyle = INK;
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "600 13px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("ONE SHARED DiT", ditX + ditWidth / 2, ditY + ditHeight / 2);
  ctx.restore();

  const timeChipY = cleanY - 14;
  ctx.save();
  ctx.globalAlpha = state.map;
  roundRect(ctx, ditX + 9, timeChipY, ditWidth - 18, 32, 16);
  ctx.fillStyle = "#f1f1f3";
  ctx.fill();
  ctx.strokeStyle = HAIR;
  ctx.stroke();
  ctx.fillStyle = INK;
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("global time  30%", ditX + ditWidth / 2, timeChipY + 16);
  ctx.restore();
  drawPipelineArrow(ctx, ditX + ditWidth / 2, timeChipY + 38, ditX + ditWidth / 2, ditY - 8, state.predict);
  drawPipelineArrow(ctx, ditX + ditWidth + 10, ditY + ditHeight / 2, outputX - 10, mixY, state.predict);

  ctx.save();
  ctx.globalAlpha = state.predict;
  ctx.fillStyle = MUTED;
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("predicted velocities", outputX, outputY - 10);
  ctx.restore();
  drawLatentBar(ctx, outputX, outputY, tokenWidth, tokenHeight, "velocity", state.predict);
}

function drawTrainingPipelineMobile(ctx, width, y, state) {
  drawStepHeading(ctx, "3  NOISE LATENT → SHARED DiT → VELOCITIES", 10, y);
  const tokenWidth = Math.min(280, width - 70);
  const tokenHeight = 42;
  const tokenX = (width - tokenWidth) / 2;
  const cleanY = y + 38;
  const noiseY = y + 132;
  const ditWidth = 190;
  const ditHeight = 54;
  const ditX = (width - ditWidth) / 2;
  const ditY = y + 252;
  const outputY = y + 335;

  ctx.fillStyle = MUTED;
  ctx.font = "600 9px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("clean latent  z", tokenX, cleanY - 8);
  drawLatentBar(ctx, tokenX, cleanY, tokenWidth, tokenHeight, "clean", 1);

  ctx.save();
  ctx.globalAlpha = state.noise;
  ctx.fillStyle = MUTED;
  ctx.textAlign = "right";
  ctx.fillText("level-wise Gaussian noise  ε", tokenX + tokenWidth, noiseY - 8);
  ctx.restore();
  drawLatentBar(ctx, tokenX, noiseY, tokenWidth, tokenHeight, "levelNoise", state.noise);
  drawLevelWiseAddition(ctx, tokenX, cleanY + tokenHeight, noiseY, tokenWidth, state.noise);

  ctx.save();
  ctx.globalAlpha = state.map;
  const timeChipX = 14;
  roundRect(ctx, timeChipX, y + 203, 144, 30, 15);
  ctx.fillStyle = "#f1f1f3";
  ctx.fill();
  ctx.fillStyle = INK;
  ctx.font = "600 10px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("global time  30%", timeChipX + 72, y + 218);
  ctx.restore();

  ctx.save();
  ctx.globalAlpha = state.predict;
  roundRect(ctx, ditX, ditY, ditWidth, ditHeight, 10);
  ctx.fillStyle = INK;
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "600 12px Inter, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("ONE SHARED DiT", width / 2, ditY + ditHeight / 2);
  ctx.restore();
  drawPipelineArrow(ctx, width / 2, noiseY + tokenHeight + 28, width / 2, ditY - 8, state.predict);
  drawPipelineArrow(ctx, timeChipX + 72, y + 239, ditX + 35, ditY - 8, state.predict);
  drawPipelineArrow(ctx, width / 2, ditY + ditHeight + 8, width / 2, outputY - 10, state.predict);

  drawLatentBar(ctx, tokenX, outputY, tokenWidth, tokenHeight, "velocity", state.predict);
  ctx.save();
  ctx.globalAlpha = state.predict;
  ctx.fillStyle = MUTED;
  ctx.font = "600 9px Inter, -apple-system, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("predicted velocities", tokenX, outputY - 8);
  ctx.restore();
}

function drawTrainingFrame(ctx, width, height, state) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = PAGE;
  ctx.fillRect(0, 0, width, height);
  const mobile = width < 700;
  if (mobile) {
    const globalSample = drawGlobalTimeSample(ctx, 8, 24, width - 16, state);
    const localSample = drawLocalTimeMap(ctx, 8, 220, width - 16, state, true);
    drawSampleTransfer(ctx, globalSample, localSample, state.map);
    drawTrainingPipelineMobile(ctx, width, 430, state);
    return;
  }

  const drawingScale = clip(width / 1000, 0.78, 1.15);
  ctx.save();
  ctx.scale(drawingScale, drawingScale);
  width /= drawingScale;
  height /= drawingScale;
  const firstWidth = Math.min(320, width * 0.34);
  const mapX = 18 + firstWidth + 56;
  const globalSample = drawGlobalTimeSample(ctx, 18, 28, firstWidth, state);
  const localSample = drawLocalTimeMap(ctx, mapX, 28, width - mapX - 18, state, false);
  drawSampleTransfer(ctx, globalSample, localSample, state.map);
  ctx.strokeStyle = HAIR;
  ctx.beginPath();
  ctx.moveTo(18, 320);
  ctx.lineTo(width - 18, 320);
  ctx.stroke();
  drawTrainingPipelineDesktop(ctx, width, 350, state);
  ctx.restore();
}

async function startTrainingAnim() {
  const canvas = document.getElementById("infer-anim");
  const section = document.getElementById("generation");
  const status = document.getElementById("generation-status");
  if (!canvas || !section) return;

  function layout() {
    const wrap = canvas.parentElement;
    const width = Math.max(320, wrap.clientWidth);
    const drawingScale = clip(width / 1000, 0.78, 1.15);
    return fitCanvas(canvas, width, width < 700 ? 880 : 590 * drawingScale);
  }

  function paint(ms) {
    const state = trainingState(ms);
    const { ctx, cssW, cssH } = layout();
    drawTrainingFrame(ctx, cssW, cssH, state);
    if (status) status.textContent = trainingStatus(state);
  }

  const player = createOneShotPlayer({
    section,
    replayButton: document.getElementById("generation-replay"),
    durationMs: TRAINING_EXAMPLE.durationMs,
    draw: paint,
    threshold: 0.55,
  });
  window.addEventListener("resize", () => player.repaint());
}

document.addEventListener("DOMContentLoaded", () => {
  startTrainingAnim().catch(console.error);
});
