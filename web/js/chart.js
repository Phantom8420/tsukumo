// Stacked per-channel lanes on one canvas: live trace, master ghost, highlighted bands.

const RANGE = {
  angle: [0, 40], motion: [0, 6], tempo: [0, 150], loudness: [-60, -15], tone: [0, 4500], surprise: [0, 10],
  hue: [0, 60], brightness: [0, 100], hold: [0, 30],
};
const GUTTER = 128;
const AXIS = 22;

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export class Lanes {
  constructor(canvas, { channels, meta, laneHeight = 70, extra = null, live = false, onBand, onTime } = {}) {
    this.live = live;             // live views show the latest value under each label
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.channels = channels;
    this.meta = meta;
    this.laneHeight = laneHeight;
    this.extra = extra;            // optional top lane, e.g. {key: "surprise", label: "Surprise"}
    this.onBand = onBand;
    this.onTime = onTime;
    this.opts = { t0: 0, t1: 30, series: {}, ghost: {}, bands: [], cursor: null };
    this.ranges = Object.fromEntries(Object.entries(RANGE).map(([k, v]) => [k, [...v]]));
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
    window.addEventListener("resize", () => this.draw());  // also fired on theme change
    canvas.addEventListener("click", (e) => this._click(e));
    canvas.style.cursor = onBand || onTime ? "pointer" : "default";
  }

  get lanes() { return (this.extra ? [this.extra.key] : []).concat(this.channels); }

  set(opts) {
    Object.assign(this.opts, opts);
    this._fitRanges();
    this.draw();
  }

  _fitRanges() {
    for (const key of this.lanes) {
      const r = this.ranges[key];
      for (const src of [this.opts.series[key], this.opts.ghost[key]]) {
        if (!src) continue;
        for (const v of src.v) {
          if (v == null || !isFinite(v)) continue;
          if (v < r[0]) r[0] = v - (r[1] - r[0]) * 0.05;
          if (v > r[1]) r[1] = v + (r[1] - r[0]) * 0.05;
        }
      }
    }
  }

  _x(t, w) {
    const { t0, t1 } = this.opts;
    return GUTTER + ((t - t0) / Math.max(t1 - t0, 1e-6)) * (w - GUTTER - 8);
  }

  _click(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < GUTTER) return;
    const { t0, t1, bands } = this.opts;
    const t = t0 + ((x - GUTTER) / (rect.width - GUTTER - 8)) * (t1 - t0);
    const hit = bands.find((b) => b.id && t >= b.t0 - 0.4 && t <= b.t1 + 0.4);
    if (hit && this.onBand) this.onBand(hit.id);
    else if (this.onTime) this.onTime(t);
  }

  draw() {
    const { canvas, ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth;
    const lanes = this.lanes;
    const h = lanes.length * this.laneHeight + AXIS;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.height = `${h}px`;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const line = css("--line"), muted = css("--muted"), faint = css("--faint");
    const accent = css("--accent"), ghost = css("--ghost"), gold = css("--gold");
    const { t0, t1, series, ghost: ghostSeries, bands, cursor } = this.opts;
    const plotH = lanes.length * this.laneHeight;

    for (const b of bands) {
      const x0 = Math.max(GUTTER, this._x(b.t0, w)), x1 = Math.min(w - 8, this._x(b.t1, w));
      if (x1 < GUTTER || x0 > w) continue;
      ctx.fillStyle = b.color || "rgba(143,166,224,0.14)";
      ctx.fillRect(x0, 0, Math.max(x1 - x0, 2), plotH);
      if (b.selected) {
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x0 + 0.5, 0.5, Math.max(x1 - x0, 2) - 1, plotH - 1);
      }
      if (b.label) {
        ctx.fillStyle = b.selected ? accent : gold;
        ctx.font = "600 11px system-ui";
        ctx.fillText(b.label, x0 + 3, 12);
      }
    }

    lanes.forEach((key, i) => {
      const y0 = i * this.laneHeight;
      const [lo, hi] = this.ranges[key];
      const y = (v) => y0 + this.laneHeight - 6 - ((v - lo) / (hi - lo || 1)) * (this.laneHeight - 14);

      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(GUTTER, y0 + this.laneHeight + 0.5);
      ctx.lineTo(w, y0 + this.laneHeight + 0.5);
      ctx.stroke();

      const m = key === this.extra?.key ? { label: this.extra.label, unit: "" } : this.meta[key];
      ctx.fillStyle = muted;
      ctx.font = "13px system-ui";
      ctx.fillText(m.label, 10, y0 + 24);
      ctx.fillStyle = faint;
      ctx.font = "11px system-ui";
      const last = this.live ? series[key]?.v?.at(-1) : null;
      ctx.fillText(last != null && isFinite(last) ? `${fmt(last)} ${m.unit}` : m.unit, 10, y0 + 42);

      if (key === this.extra?.key) {
        this._area(series[key], y, w, y0 + this.laneHeight - 6, "rgba(226,73,47,0.35)", accent);
        return;
      }
      this._path(ghostSeries[key], y, w, ghost, 2, [5, 4]);
      this._path(series[key], y, w, accent, 1.8);
    });

    ctx.fillStyle = faint;
    ctx.font = "11px system-ui";
    const span = t1 - t0;
    const step = span > 120 ? 30 : span > 50 ? 10 : 5;
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      const x = this._x(t, w);
      ctx.fillText(clock(t), x - 12, h - 6);
      ctx.strokeStyle = line;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, plotH);
      ctx.lineTo(x + 0.5, plotH + 4);
      ctx.stroke();
    }

    if (cursor != null) {
      const x = this._x(cursor, w);
      ctx.strokeStyle = gold;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, plotH);
      ctx.stroke();
    }
  }

  _path(s, y, w, color, width, dash = []) {
    if (!s || !s.t.length) return;
    const { ctx } = this;
    const { t0, t1 } = this.opts;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.t.length; i++) {
      const t = s.t[i], v = s.v[i];
      if (t < t0 - 1 || t > t1 + 1 || v == null || !isFinite(v)) { pen = false; continue; }
      const x = this._x(t, w), yy = y(v);
      pen ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy);
      pen = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  _area(s, y, w, base, fill, stroke) {
    if (!s || !s.t.length) return;
    const { ctx } = this;
    ctx.beginPath();
    ctx.moveTo(this._x(s.t[0], w), base);
    s.t.forEach((t, i) => ctx.lineTo(this._x(t, w), y(s.v[i])));
    ctx.lineTo(this._x(s.t.at(-1), w), base);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// Rolling buffer for live views: keeps the last `span` seconds per channel.
export class Rolling {
  constructor(channels, span = 25) {
    this.span = span;
    this.data = Object.fromEntries(channels.map((c) => [c, { t: [], v: [] }]));
  }
  push(t, values) {
    if (!values) return;
    for (const [c, s] of Object.entries(this.data)) {
      s.t.push(t);
      s.v.push(values[c]);
      while (s.t.length && s.t[0] < t - this.span - 2) { s.t.shift(); s.v.shift(); }
    }
  }
  window(now) { return { t0: Math.max(0, now - this.span), t1: Math.max(this.span, now) }; }
}

export function fmt(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

export function clock(t) {
  const s = Math.max(0, Math.round(t));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function series(features, key) {
  return { t: features.t, v: features.X[key] };
}
