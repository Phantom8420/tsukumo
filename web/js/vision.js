// Camera (or any video) -> frames, so the tool needs no strapped-on phone.
// Sharpening: a hand landmark model finds the hand holding the blade; its knuckle line gives
// the tilt, the wrist track gives acceleration, and the clip's own audio gives the stone's sound.
// Heat: the steel's own colour and glow, read from the part of the picture the smith points at.
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const PALM_M = 0.085;   // wrist to middle knuckle on an adult hand: the only ruler a camera has

let landmarker;
async function model() {
  if (!landmarker) {
    const files = await FilesetResolver.forVisionTasks(WASM);
    landmarker = await HandLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
      runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.3, minTrackingConfidence: 0.3,
    });
  }
  return landmarker;
}

// Exponential smoother with a time constant, so uneven frame gaps don't change its feel.
const ema = (tau) => {
  let v = null, last = 0;
  return (t, x) => {
    if (v === null) { v = x; last = t; return v; }
    const a = 1 - Math.exp(-Math.max(t - last, 0) / tau);
    last = t;
    return (v = v + a * (x - v));
  };
};

export function audioMeter(ctx, source) {
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.3;
  source.connect(analyser);
  const time = new Float32Array(analyser.fftSize), freq = new Float32Array(analyser.frequencyBinCount);
  return () => {
    analyser.getFloatTimeDomainData(time);
    let s = 0;
    for (const v of time) s += v * v;
    const db = 10 * Math.log10(s / time.length + 1e-12);
    analyser.getFloatFrequencyData(freq);
    const binHz = ctx.sampleRate / analyser.fftSize;
    let num = 0, den = 0;
    for (let i = Math.floor(150 / binHz); i < Math.min(freq.length, 8000 / binHz); i++) {
      const m = Math.pow(10, freq[i] / 20);
      num += m * i * binHz;
      den += m;
    }
    return [db, den > 0 ? num / den : 0];
  };
}

// Wrist position in metres (the palm is the ruler) -> smoothed acceleration in m/s².
function wristMotion() {
  const sx = ema(0.06), sy = ema(0.06), svx = ema(0.06), svy = ema(0.06), spalm = ema(0.5);
  let prev = null;
  return (s, lm, W, H) => {
    const wx = lm[0].x * W, wy = lm[0].y * H, mx = lm[9].x * W, my = lm[9].y * H;
    const mPerPx = PALM_M / Math.max(spalm(s, Math.hypot(mx - wx, my - wy)), 1);
    const x = sx(s, wx * mPerPx), y = sy(s, wy * mPerPx);
    let ax = 0, ay = 0;
    if (prev && s > prev.s) {
      const dt = s - prev.s, vx = svx(s, (x - prev.x) / dt), vy = svy(s, (y - prev.y) / dt);
      if (prev.vx !== undefined) { ax = (vx - prev.vx) / dt; ay = (vy - prev.vy) / dt; }
      prev = { s, x, y, vx, vy };
    } else {
      prev = { s, x, y };
    }
    return [ax, ay];
  };
}

// Stay with the hand we were following; two hands on a tool swap order between frames.
function follow() {
  let wrist = null;
  return (landmarks) => {
    const pick = landmarks
      .map((lm) => ({ lm, d: wrist ? Math.hypot(lm[0].x - wrist.x, lm[0].y - wrist.y) : -lm[0].y }))
      .sort((a, b) => a.d - b.d)[0];
    if (pick) wrist = pick.lm[0];
    return pick ? pick.lm : null;
  };
}

// Call `frame(t_ms)` once per new video frame until the returned stop function is called.
function everyFrame(video, frame) {
  let running = true, last = -1;
  const step = () => {
    if (!running) return;
    if (video.currentTime !== last && video.readyState >= 2) {
      last = video.currentTime;
      frame(last * 1000);
    }
    if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
  };
  step();
  return () => { running = false; };
}

/**
 * Track the working hand in `video` and call onFrame([t_ms, angle, ax, ay, az, loud, tone]).
 * `sound()` returns [dBFS, centroid Hz]. Returns a stop function.
 * Frames without a hand are skipped: the server resamples across short gaps.
 */
export async function trackHands(video, sound, onFrame, { onHand } = {}) {
  const hands = await model();
  const motion = wristMotion(), pickHand = follow(), sang = ema(0.1);
  return everyFrame(video, (t) => {
    const lm = pickHand(hands.detectForVideo(video, t).landmarks);
    onHand?.(lm);
    if (!lm) return;
    const W = video.videoWidth, H = video.videoHeight, s = t / 1000;
    // Knuckle line against the horizon: how far the hand, and the blade in it, is tipped.
    let tilt = Math.abs(Math.atan2((lm[17].y - lm[5].y) * H, (lm[17].x - lm[5].x) * W) * 180 / Math.PI);
    if (tilt > 90) tilt = 180 - tilt;
    const [ax, ay] = motion(s, lm, W, H);
    const [loud, tone] = sound();
    onFrame([t, sang(s, tilt), ax, ay, 0, loud, tone]);
  });
}

const GLOW_PX = 160;   // the picture is shrunk to this width before reading colour
const SPOT = 0.15;     // half-size of the box read around the steel, as a fraction of the frame

// Hue (0 red .. 60 yellow) and glow (0-100) of the steel in a box centred on `at` = {x, y} in 0..1.
export function glowAt(img, at) {
  const { width: W, height: H, data } = img;
  const x0 = Math.max(0, Math.round((at.x - SPOT) * W)), x1 = Math.min(W, Math.round((at.x + SPOT) * W));
  const y0 = Math.max(0, Math.round((at.y - SPOT) * H)), y1 = Math.min(H, Math.round((at.y + SPOT) * H));
  const vals = [];
  let hw = 0, hs = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * W + x) * 4, r = data[i], g = data[i + 1], b = data[i + 2];
      const max = Math.max(r, g, b), min = Math.min(r, g, b), v = max / 255;
      // Only glowing, coloured pixels say anything about temperature; a white wall is not hot steel.
      const lit = v >= 0.35 && max - min >= 0.2 * max;
      vals.push(lit ? v : 0);
      if (!lit) continue;
      const d = max - min;
      let h = max === r ? 60 * (((g - b) / d) % 6) : max === g ? 60 * ((b - r) / d + 2) : 60 * ((r - g) / d + 4);
      if (h > 180) h -= 360;   // deep reds just under 0° sit next to 0°, not at 350°
      hw += v * v;
      hs += v * v * h;
    }
  }
  if (!vals.length) return [0, 0];
  vals.sort((p, q) => q - p);
  const top = vals.slice(0, Math.max(1, Math.round(vals.length / 10)));
  const glow = (100 * top.reduce((p, q) => p + q, 0)) / top.length;
  return [hw ? Math.max(0, hs / hw) : 0, glow];
}

/**
 * Watch heated steel in `video` and call onFrame([t_ms, hue, glow, ax, ay, az]).
 * `spot()` says where the steel is ({x, y} in 0..1); the hand, when in view, gives movement.
 */
export async function trackGlow(video, onFrame, { spot, onHand } = {}) {
  const hands = await model();
  const motion = wristMotion(), pickHand = follow();
  const canvas = document.createElement("canvas");
  canvas.width = GLOW_PX;
  canvas.height = Math.max(1, Math.round((GLOW_PX * video.videoHeight) / video.videoWidth));
  const g = canvas.getContext("2d", { willReadFrequently: true });
  return everyFrame(video, (t) => {
    g.drawImage(video, 0, 0, canvas.width, canvas.height);
    const [hue, glow] = glowAt(g.getImageData(0, 0, canvas.width, canvas.height), spot ? spot() : { x: 0.5, y: 0.5 });
    const lm = pickHand(hands.detectForVideo(video, t).landmarks);
    onHand?.(lm);
    const [ax, ay] = lm ? motion(t / 1000, lm, video.videoWidth, video.videoHeight) : [0, 0];
    onFrame([t, hue, glow, ax, ay, 0]);
  });
}

// Auto exposure and white balance would "correct" the very colour being measured: freeze them.
export async function lockExposure(track) {
  const caps = track.getCapabilities ? track.getCapabilities() : {}, now = track.getSettings();
  const fix = {};
  if (caps.exposureMode?.includes("manual")) Object.assign(fix, { exposureMode: "manual", exposureTime: now.exposureTime });
  if (caps.whiteBalanceMode?.includes("manual")) Object.assign(fix, { whiteBalanceMode: "manual", colorTemperature: now.colorTemperature });
  if (!Object.keys(fix).length) return false;
  try { await track.applyConstraints({ advanced: [fix] }); return true; } catch { return false; }
}
