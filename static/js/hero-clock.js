/**
 * APEX hero: percent clock, channel levels, and their cumulative decodes.
 *
 *   t_ℓ(τ) = clip(1 + ℓ·g − τ, 0, 1)
 *   τ ∈ [0, 1 + (L−1)g]
 *
 * Playback pauses when Levels 1 and 2 initialize from noise.
 */
const HERO = {
  g: 0.25,
  nLevels: 3,
  imageWidth: 256,
  imageHeight: 256,
  imageCount: 51,
  trajectories: [
    "static/images/hero/trajectories/iguana/frames/",
    "static/images/hero/trajectories/cock/frames/",
    "static/images/hero/trajectories/lionfish/frames/",
    "static/images/hero/trajectories/goldfish/frames/",
    "static/images/hero/trajectories/great-grey-owl/frames/",
    "static/images/hero/trajectories/peacock/frames/",
    "static/images/hero/trajectories/macaw/frames/",
    "static/images/hero/trajectories/monarch/frames/",
    "static/images/hero/trajectories/zebra/frames/",
  ],
  labels: ["Level 0", "Level 1", "Level 2"],
  colors: ["#2563A6", "#2A9D8F", "#E69F00"],
  pauseMs: 1600,
  holdMs: 1600,
};
const HERO_FRAME_BATCH_SIZE = 6;

function clip(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

function smoothstep(p) {
  const x = clip(p, 0, 1);
  return x * x * (3 - 2 * x);
}

function localT(level, tau, g) {
  return clip(1 + level * g - tau, 0, 1);
}

function levelState(level, tau, g) {
  if (tau <= level * g) return "unstarted";
  if (tau >= 1 + level * g) return "done";
  return "denoising";
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load " + src));
    img.src = src;
  });
}

async function loadHeroFrame(frameRootUrl, index, tauMax) {
  const source = new URL(String(index).padStart(3, "0") + ".webp", frameRootUrl);
  const image = await loadImage(source.href);
  if (image.naturalWidth !== HERO.imageWidth * HERO.nLevels || image.naturalHeight !== HERO.imageHeight) {
    throw new Error("Invalid hero image dimensions: " + source);
  }
  return {
    tau: (index / (HERO.imageCount - 1)) * tauMax,
    image,
  };
}

async function loadHeroFrames(frameRoot, onBatchLoaded) {
  const frameRootUrl = new URL(frameRoot, window.location.href);
  const tauMax = 1 + (HERO.nLevels - 1) * HERO.g;
  const trajectory = {
    frameWidth: HERO.imageWidth,
    frameHeight: HERO.imageHeight,
    frames: [await loadHeroFrame(frameRootUrl, 0, tauMax)],
  };
  trajectory.complete = (async () => {
    if (document.readyState !== "complete") {
      await new Promise((resolve) => window.addEventListener("load", resolve, { once: true }));
    }
    for (let firstIndex = 1; firstIndex < HERO.imageCount; firstIndex += HERO_FRAME_BATCH_SIZE) {
      const endIndex = Math.min(firstIndex + HERO_FRAME_BATCH_SIZE, HERO.imageCount);
      const batch = await Promise.all(
        Array.from({ length: endIndex - firstIndex }, (_, offset) => loadHeroFrame(frameRootUrl, firstIndex + offset, tauMax))
      );
      trajectory.frames.push(...batch);
      onBatchLoaded(trajectory);
    }
    return trajectory;
  })();
  return trajectory;
}

function nearestHeroFrame(frames, tau) {
  let nearest = frames[0];
  let nearestDistance = Math.abs(tau - nearest.tau);
  for (let index = 1; index < frames.length; index++) {
    const distance = Math.abs(tau - frames[index].tau);
    if (distance < nearestDistance) {
      nearest = frames[index];
      nearestDistance = distance;
    }
  }
  return nearest;
}

function createGaussianNoise(size, patternKey) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(size, size);
  let state = patternKey >>> 0;

  function randomUnit() {
    state = (1664525 * state + 1013904223) >>> 0;
    return (state + 1) / 4294967297;
  }

  for (let index = 0; index < image.data.length; index += 4) {
    const radius = Math.sqrt(-2 * Math.log(randomUnit()));
    const gaussian = radius * Math.cos(2 * Math.PI * randomUnit());
    const value = Math.round(clip(128 + gaussian * 47, 0, 255));
    image.data[index] = value;
    image.data[index + 1] = value;
    image.data[index + 2] = value;
    image.data[index + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
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

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function mixRgb(a, b, t) {
  return {
    r: Math.round(lerp(a.r, b.r, t)),
    g: Math.round(lerp(a.g, b.g, t)),
    b: Math.round(lerp(a.b, b.b, t)),
  };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function rgbStr(c) {
  return "rgb(" + c.r + "," + c.g + "," + c.b + ")";
}

function clockGeometry(cssW, cssH) {
  const scale = clip(cssH / 188, 1, 2);
  const left = 70 * scale;
  const right = cssW - 12 * scale;
  return {
    scale,
    left,
    right,
    trackW: right - left,
    top: 22 * scale,
  };
}

function drawClock(ctx, cssW, cssH, tau, g, n, labels, colors, handlePulse) {
  ctx.clearRect(0, 0, cssW, cssH);
  const tauMax = 1 + (n - 1) * g;
  const pct = clip(tau / tauMax, 0, 1);
  const { scale, left, right, trackW, top } = clockGeometry(cssW, cssH);
  const rowH = (cssH - top - 6 * scale) / n;
  const barH = 16 * scale;

  ctx.fillStyle = "#6e6e73";
  ctx.font = "600 " + 12 * scale + "px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("0%", left, 14 * scale);
  ctx.textAlign = "center";
  ctx.fillText("50%", left + trackW / 2, 14 * scale);
  ctx.textAlign = "right";
  ctx.fillText("100%", right, 14 * scale);

  ctx.strokeStyle = "#e2e2e7";
  ctx.lineWidth = scale;
  [0, 0.5, 1].forEach((p) => {
    const x = left + p * trackW;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, cssH - 18 * scale);
    ctx.stroke();
  });

  const px = left + pct * trackW;
  for (let l = 0; l < n; l++) {
    const y = top + rowH * l + rowH / 2;
    ctx.font = "600 " + 13 * scale + "px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
    ctx.fillStyle = colors[l];
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(labels[l], left - 10 * scale, y);

    const start = l * g;
    const end = 1 + l * g;
    const xStart = left + (start / tauMax) * trackW;
    const xEnd = left + (Math.min(end, tauMax) / tauMax) * trackW;

    roundRect(ctx, left, y - barH / 2, trackW, barH, 8 * scale);
    ctx.fillStyle = "#ececef";
    ctx.fill();

    const denW = Math.max(0, xEnd - xStart);
    if (denW > 0) {
      const rgb = hexToRgb(colors[l]);
      ctx.fillStyle = rgbStr(mixRgb({ r: 255, g: 255, b: 255 }, rgb, 0.24));
      roundRect(ctx, xStart, y - barH / 2, denW, barH, 8 * scale);
      ctx.fill();

      const fillEnd = clip(px, xStart, xEnd);
      const fillWidth = fillEnd - xStart;
      if (fillWidth <= 0) continue;
      const grad = ctx.createLinearGradient(xStart, 0, Math.max(xStart + 1, fillEnd), 0);
      grad.addColorStop(0, rgbStr(mixRgb({ r: 255, g: 255, b: 255 }, rgb, 0.5)));
      grad.addColorStop(1, colors[l]);
      ctx.fillStyle = grad;
      roundRect(ctx, xStart, y - barH / 2, fillWidth, barH, 8 * scale);
      ctx.fill();
    }
  }

  const knobWidth = 14 * scale;
  const knobHeight = 13 * scale;
  const knobBottom = cssH - 2 * scale;
  const knobTop = knobBottom - knobHeight;
  const pointerTipY = knobTop - 7 * scale;

  ctx.strokeStyle = "#1d1d1f";
  ctx.lineWidth = 2 * scale;
  ctx.beginPath();
  ctx.moveTo(px, top - 2 * scale);
  ctx.lineTo(px, pointerTipY);
  ctx.stroke();

  ctx.save();
  ctx.globalAlpha = 0.68 + 0.32 * handlePulse;
  ctx.fillStyle = "#6e6e73";
  ctx.beginPath();
  ctx.moveTo(px, pointerTipY);
  ctx.lineTo(px - 5 * scale, knobTop + scale);
  ctx.lineTo(px + 5 * scale, knobTop + scale);
  ctx.closePath();
  ctx.fill();
  roundRect(ctx, px - knobWidth / 2, knobTop, knobWidth, knobHeight, 2.5 * scale);
  ctx.fill();
  ctx.restore();
}

function drawToken(ctx, cssW, cssH, tau, g, n, labels, colors) {
  ctx.clearRect(0, 0, cssW, cssH);
  const scale = clip(cssH / 188, 1, 2);
  const barH = 26 * scale;
  const outer = 8 * scale;
  const indexWidth = 30 * scale;
  const tokenW = Math.min(300 * scale, cssW - indexWidth - outer * 2);
  const tokenX = (cssW - tokenW - indexWidth) / 2 + indexWidth;
  const tokenYs = [10, 80, 150].map((value) => value * scale);
  const widths = [0.18, 0.28, 0.54];

  function drawTokenIndex(y, label) {
    ctx.fillStyle = "#6e6e73";
    ctx.font = "600 " + 9 * scale + "px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(label, tokenX - 7 * scale, y + barH / 2);
  }

  function drawOneToken(y, index) {
    ctx.save();
    roundRect(ctx, tokenX, y, tokenW, barH, 5 * scale);
    ctx.strokeStyle = "#8e8e93";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    let bandX = tokenX;
    for (let l = 0; l < n; l++) {
      const bandW = tokenW * widths[l];
      const state = levelState(l, tau, g);
      const t = localT(l, tau, g);
      const clean = state === "done" ? 1 : state === "unstarted" ? 0 : smoothstep(1 - t);
      ctx.save();
      ctx.beginPath();
      ctx.rect(bandX + 1, y + 1, bandW - 2, barH - 2);
      ctx.clip();
      if (state === "unstarted") {
        ctx.fillStyle = "#ececef";
        ctx.fillRect(bandX, y, bandW, barH);
      } else {
        drawStaticNoiseBlock(ctx, bandX + 1, y + 1, bandW - 2, barH - 2, colors[l], clean, tau * 800 + l * 17 + index * 31);
      }
      ctx.restore();
      ctx.fillStyle = state === "unstarted" ? colors[l] : "#fff";
      ctx.font = "600 " + 8 * scale + "px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(labels[l], bandX + bandW / 2, y + barH / 2);
      bandX += bandW;
    }
    ctx.restore();
  }

  drawOneToken(tokenYs[0], 0);
  drawTokenIndex(tokenYs[0], "1");
  const bracketY = tokenYs[0] + barH + 7 * scale;
  ctx.strokeStyle = "#6e6e73";
  ctx.lineWidth = scale;
  ctx.beginPath();
  ctx.moveTo(tokenX, bracketY - 4 * scale);
  ctx.lineTo(tokenX, bracketY);
  ctx.lineTo(tokenX + tokenW, bracketY);
  ctx.lineTo(tokenX + tokenW, bracketY - 4 * scale);
  ctx.stroke();
  ctx.fillStyle = "#6e6e73";
  ctx.font = "600 " + 9 * scale + "px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("channels", tokenX + tokenW / 2, bracketY + 3 * scale);

  drawOneToken(tokenYs[1], 1);
  drawTokenIndex(tokenYs[1], "2");
  ctx.fillStyle = "#8e8e93";
  ctx.font = "600 " + 20 * scale + "px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("⋮", tokenX + tokenW / 2, 130 * scale);
  drawOneToken(tokenYs[2], 2);
  drawTokenIndex(tokenYs[2], "N");
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

function buildTimeline(g, n, pauseMs, holdMs) {
  const tauMax = 1 + (n - 1) * g;
  const motionMs = 7800;
  const segs = [];
  let wall = 0;
  const knots = [0];
  for (let l = 1; l < n; l++) knots.push(l * g);
  knots.push(tauMax);

  for (let i = 0; i < knots.length - 1; i++) {
    const span = knots[i + 1] - knots[i];
    const dur = motionMs * (span / tauMax);
    segs.push({ kind: "play", tau0: knots[i], tau1: knots[i + 1], t0: wall, t1: wall + dur });
    wall += dur;
    if (i < knots.length - 2) {
      const lvl = i + 1;
      segs.push({
        kind: "pause",
        tau: knots[i + 1],
        caption: "Initialize Level " + lvl + " from noise",
        t0: wall,
        t1: wall + pauseMs,
      });
      wall += pauseMs;
    }
  }
  segs.push({ kind: "hold", tau: tauMax, t0: wall, t1: wall + holdMs });
  wall += holdMs;
  return { segs, cycle: wall, tauMax };
}

function atTimeline(tl, elapsed) {
  const t = clip(elapsed, 0, tl.cycle);
  for (const s of tl.segs) {
    if (t >= s.t0 && t < s.t1) {
      if (s.kind === "play") {
        const u = (t - s.t0) / (s.t1 - s.t0);
        return { tau: lerp(s.tau0, s.tau1, u), caption: "" };
      }
      if (s.kind === "pause") return { tau: s.tau, caption: s.caption };
      return { tau: s.tau, caption: "" };
    }
  }
  return { tau: tl.tauMax, caption: "" };
}

function timelineOffsetAtTau(tl, tau) {
  const targetTau = clip(tau, 0, tl.tauMax);
  for (const segment of tl.segs) {
    if (segment.kind !== "play" || targetTau < segment.tau0 || targetTau > segment.tau1) continue;
    const progress = (targetTau - segment.tau0) / (segment.tau1 - segment.tau0);
    return lerp(segment.t0, segment.t1, progress);
  }
  return tl.cycle;
}

async function startHero() {
  const outputCanvases = HERO.labels.map((_, level) => document.getElementById("hero-level-" + level));
  const clockCanvas = document.getElementById("hero-clock");
  const tokenCanvas = document.getElementById("hero-token");
  const tauLabel = document.getElementById("hero-tau");
  const statusEl = document.getElementById("hero-status");
  const previousImageButton = document.getElementById("hero-previous-image");
  const playbackButton = document.getElementById("hero-playback");
  const nextImageButton = document.getElementById("hero-next-image");
  if (outputCanvases.some((canvas) => !canvas) || !clockCanvas || !tokenCanvas || !previousImageButton || !playbackButton || !nextImageButton) {
    return;
  }
  if (!Array.isArray(HERO.trajectories) || HERO.trajectories.length === 0) {
    throw new Error("Hero requires at least one trajectory");
  }

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const g = HERO.g;
  const n = HERO.nLevels;
  const tauMax = 1 + (n - 1) * g;
  const tl = buildTimeline(g, n, HERO.pauseMs, HERO.holdMs);
  let gaussianNoise = createGaussianNoise(256, 1);
  let trajectory = null;
  let trajectoryIndex = 0;
  let lastTau = 0;
  let lastCaption = "";
  let lastHandlePulse = 1;
  let lastTimelineMs = 0;
  let raf = 0;
  let playbackState = "loading";
  const trajectoryLoads = new Map();

  function setPlaybackState(state) {
    if (!["loading", "playing", "paused"].includes(state)) {
      throw new Error("Invalid hero playback state: " + state);
    }
    playbackState = state;
    const ready = state !== "loading";
    const canChangeImage = ready && HERO.trajectories.length > 1;
    previousImageButton.disabled = !canChangeImage;
    nextImageButton.disabled = !canChangeImage;
    playbackButton.disabled = !ready || reduced;
    playbackButton.dataset.state = state === "playing" ? "playing" : "paused";
    playbackButton.setAttribute("aria-label", state === "playing" ? "Pause animation" : "Play animation");
    playbackButton.setAttribute("aria-pressed", String(state === "playing"));
  }

  function layout() {
    const targetHeight = clip(window.innerHeight * 0.29, 260, 390);
    const tokenWrap = tokenCanvas.parentElement;
    fitCanvas(tokenCanvas, tokenWrap.clientWidth, targetHeight);
    const clockWrap = clockCanvas.parentElement;
    fitCanvas(clockCanvas, clockWrap.clientWidth, targetHeight);
    for (const canvas of outputCanvases) {
      const cardWidth = canvas.parentElement.clientWidth;
      fitCanvas(canvas, cardWidth, cardWidth);
    }
  }

  layout();

  function drawOutput(canvas, level, tau, frame) {
    const ctx = canvas.getContext("2d");
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    if (!frame || levelState(level, tau, g) === "unstarted") {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(gaussianNoise, 0, 0, width, height);
      return;
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(frame.image, level * trajectory.frameWidth, 0, trajectory.frameWidth, trajectory.frameHeight, 0, 0, width, height);
  }

  function render(tau, caption, handlePulse) {
    lastTau = tau;
    lastCaption = caption;
    lastHandlePulse = handlePulse;
    const frame = trajectory ? nearestHeroFrame(trajectory.frames, tau) : null;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    outputCanvases.forEach((canvas, level) => drawOutput(canvas, level, tau, frame));

    const clock = clockCanvas.getContext("2d");
    clock.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawClock(clock, clockCanvas.clientWidth, clockCanvas.clientHeight, tau, g, n, HERO.labels, HERO.colors, handlePulse);

    const tok = tokenCanvas.getContext("2d");
    tok.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawToken(tok, tokenCanvas.clientWidth, tokenCanvas.clientHeight, tau, g, n, HERO.labels, HERO.colors);

    const pct = Math.round(clip(tau / tauMax, 0, 1) * 100);
    if (tauLabel) tauLabel.textContent = pct + "%";
    if (statusEl) statusEl.textContent = caption || "Global progress";
    clockCanvas.setAttribute("aria-valuenow", String(pct));
    clockCanvas.setAttribute("aria-valuetext", pct + "% global progress");
  }

  function loadTrajectory(index) {
    if (!trajectoryLoads.has(index)) {
      trajectoryLoads.set(
        index,
        loadHeroFrames(HERO.trajectories[index], (loadedTrajectory) => {
          if (trajectory === loadedTrajectory) render(lastTau, lastCaption, lastHandlePulse);
        })
      );
    }
    return trajectoryLoads.get(index);
  }

  function scrubCaption(tau) {
    if (tau <= 0.001) return "Level 0 is ready to start";
    if (tau < g) return "Level 0 is denoising";
    if (tau < 2 * g) return "Level 1 starts from noise";
    if (tau < tauMax) return "Level 2 starts from noise";
    return "All levels are clean";
  }

  function pausePlayback() {
    if (playbackState !== "playing") return;
    cancelAnimationFrame(raf);
    setPlaybackState("paused");
  }

  function scrubAt(clientX) {
    const bounds = clockCanvas.getBoundingClientRect();
    const canvasX = ((clientX - bounds.left) / bounds.width) * clockCanvas.clientWidth;
    const geometry = clockGeometry(clockCanvas.clientWidth, clockCanvas.clientHeight);
    const progress = clip((canvasX - geometry.left) / geometry.trackW, 0, 1);
    const tau = progress * tauMax;
    lastTimelineMs = timelineOffsetAtTau(tl, tau);
    render(tau, scrubCaption(tau), 1);
  }

  let dragging = false;
  clockCanvas.addEventListener("pointerdown", (event) => {
    if (playbackState === "loading") return;
    pausePlayback();
    dragging = true;
    clockCanvas.setPointerCapture(event.pointerId);
    scrubAt(event.clientX);
  });
  clockCanvas.addEventListener("pointermove", (event) => {
    if (dragging) scrubAt(event.clientX);
  });
  clockCanvas.addEventListener("pointerup", (event) => {
    dragging = false;
    clockCanvas.releasePointerCapture(event.pointerId);
  });
  clockCanvas.addEventListener("pointercancel", () => {
    dragging = false;
  });
  clockCanvas.addEventListener("keydown", (event) => {
    const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    if (playbackState === "loading") return;
    pausePlayback();
    let tau = lastTau;
    if (event.key === "ArrowLeft") tau -= tauMax * 0.02;
    if (event.key === "ArrowRight") tau += tauMax * 0.02;
    if (event.key === "Home") tau = 0;
    if (event.key === "End") tau = tauMax;
    tau = clip(tau, 0, tauMax);
    lastTimelineMs = timelineOffsetAtTau(tl, tau);
    render(tau, scrubCaption(tau), 1);
  });

  window.addEventListener("resize", () => {
    layout();
    render(lastTau, lastCaption, lastHandlePulse);
  });

  function startPlayback(timelineMs) {
    if (!trajectory) throw new Error("Cannot play before the hero trajectory is loaded");
    if (reduced) return;
    cancelAnimationFrame(raf);
    lastTimelineMs = clip(timelineMs, 0, tl.cycle);
    const playbackStart = performance.now() - lastTimelineMs;
    setPlaybackState("playing");

    function tick(now) {
      if (playbackState !== "playing") return;
      lastTimelineMs = clip(now - playbackStart, 0, tl.cycle);
      if (lastTimelineMs >= tl.cycle) {
        render(tauMax, "", 1);
        const nextIndex = (trajectoryIndex + 1) % HERO.trajectories.length;
        showTrajectory(nextIndex).catch(console.error);
        return;
      }
      const { tau, caption } = atTimeline(tl, lastTimelineMs);
      const attention = clip(1 - lastTimelineMs / 7000, 0, 1);
      const wave = 0.5 + 0.5 * Math.sin((lastTimelineMs / 1800) * Math.PI * 2 - Math.PI / 2);
      const handlePulse = 1 - attention * (1 - wave);
      render(tau, caption, handlePulse);
      raf = requestAnimationFrame(tick);
    }

    raf = requestAnimationFrame(tick);
  }

  async function showTrajectory(index) {
    if (!Number.isInteger(index) || index < 0 || index >= HERO.trajectories.length) {
      throw new Error("Invalid hero trajectory index: " + index);
    }
    cancelAnimationFrame(raf);
    setPlaybackState("loading");
    trajectoryIndex = index;
    trajectory = null;
    lastTimelineMs = 0;
    gaussianNoise = createGaussianNoise(256, index + 1);
    render(0, "Loading generated trajectory", 1);
    trajectory = await loadTrajectory(index);

    if (reduced) {
      await trajectory.complete;
      lastTimelineMs = tl.cycle;
      render(tauMax, "", 1);
      setPlaybackState("paused");
    } else {
      if (index === 0 && HERO.trajectories.length > 1) {
        trajectory.complete
          .then(() => loadTrajectory(1))
          .then((nextTrajectory) => nextTrajectory.complete)
          .catch(console.error);
      } else {
        trajectory.complete.catch(console.error);
      }
      render(0, "", 1);
      startPlayback(0);
    }
  }

  previousImageButton.addEventListener("click", () => {
    const previousIndex = (trajectoryIndex - 1 + HERO.trajectories.length) % HERO.trajectories.length;
    showTrajectory(previousIndex).catch(console.error);
  });
  playbackButton.addEventListener("click", () => {
    if (playbackState === "playing") {
      pausePlayback();
      return;
    }
    if (lastTimelineMs >= tl.cycle) {
      const nextIndex = (trajectoryIndex + 1) % HERO.trajectories.length;
      showTrajectory(nextIndex).catch(console.error);
      return;
    }
    startPlayback(lastTimelineMs);
  });
  nextImageButton.addEventListener("click", () => {
    const nextIndex = (trajectoryIndex + 1) % HERO.trajectories.length;
    showTrajectory(nextIndex).catch(console.error);
  });

  await showTrajectory(0);
}

document.addEventListener("DOMContentLoaded", () => {
  startHero().catch((err) => {
    console.error(err);
  });
});
