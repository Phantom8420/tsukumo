// Camera (or any video) -> sharpening frames, so the tool needs no strapped-on phone.
// A hand landmark model finds the hand holding the blade; its knuckle line gives the
// tilt, the wrist track gives acceleration, and the clip's own audio gives the stone's sound.
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

/**
 * Track the working hand in `video` and call onFrame([t_ms, angle, ax, ay, az, loud, tone]).
 * `sound()` returns [dBFS, centroid Hz]. Returns a stop function.
 * Frames without a hand are skipped: the server resamples across short gaps.
 */
export async function trackHands(video, sound, onFrame, { onHand } = {}) {
  const hands = await model();
  const sx = ema(0.06), sy = ema(0.06), svx = ema(0.06), svy = ema(0.06), sang = ema(0.1), spalm = ema(0.5);
  let prev = null, wrist = null, running = true, lastVideoT = -1;

  const step = (now) => {
    if (!running) return;
    const vt = video.currentTime;
    if (vt !== lastVideoT && video.readyState >= 2) {
      lastVideoT = vt;
      const t = vt * 1000;
      const res = hands.detectForVideo(video, t);
      // Stay with the hand we were following; two hands on a tool swap order between frames.
      const pick = res.landmarks
        .map((lm) => ({ lm, d: wrist ? Math.hypot(lm[0].x - wrist.x, lm[0].y - wrist.y) : -lm[0].y }))
        .sort((a, b) => a.d - b.d)[0];
      onHand?.(pick ? pick.lm : null);
      if (pick) {
        const lm = pick.lm, W = video.videoWidth, H = video.videoHeight;
        wrist = lm[0];
        const px = (p) => [p.x * W, p.y * H];
        const [wx, wy] = px(lm[0]), [mx, my] = px(lm[9]), [ix, iy] = px(lm[5]), [kx, ky] = px(lm[17]);
        const mPerPx = PALM_M / Math.max(spalm(t / 1000, Math.hypot(mx - wx, my - wy)), 1);
        // Knuckle line against the horizon: how far the hand, and the blade in it, is tipped.
        let tilt = Math.abs(Math.atan2(ky - iy, kx - ix) * 180 / Math.PI);
        if (tilt > 90) tilt = 180 - tilt;
        const s = t / 1000, x = sx(s, wx * mPerPx), y = sy(s, wy * mPerPx);
        let ax = 0, ay = 0;
        if (prev && s > prev.s) {
          const dt = s - prev.s, vx = svx(s, (x - prev.x) / dt), vy = svy(s, (y - prev.y) / dt);
          if (prev.vx !== undefined) { ax = (vx - prev.vx) / dt; ay = (vy - prev.vy) / dt; }
          prev = { s, x, y, vx, vy };
        } else {
          prev = { s, x, y };
        }
        const [loud, tone] = sound();
        onFrame([t, sang(s, tilt), ax, ay, 0, loud, tone]);
      }
    }
    if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(step);
    else requestAnimationFrame(step);
  };
  if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(step);
  else requestAnimationFrame(step);
  return () => { running = false; };
}
