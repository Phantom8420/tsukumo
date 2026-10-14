"""Find the moments where the master stopped doing what they were just doing.

A written SOP says "keep a steady angle and rhythm". Tacit knowledge lives
exactly where an expert breaks that expectation: pauses, angle changes,
pressure changes. We model "what they were just doing" as the robust
median/MAD of each channel over the previous few seconds, score how far the
next half-second departs from it, and keep the strongest few peaks. Those
are the only clips the master is asked about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import CHANNEL_META, CHANNELS, FS, NOISE_FLOOR
from .signals import Features, smooth

BASELINE_S = 4.0
CURRENT_S = 0.7
THRESHOLD = 3.0
MIN_GAP_S = 4.0
MAX_EVENTS = 6
TREND_S = 15.0
Z_CAP = 10.0


@dataclass
class Evidence:
    channel: str
    before: float
    during: float
    delta: float
    unit: str
    z: float
    trend: bool = False        # a slow build-up beforehand rather than the change itself


@dataclass
class Surprise:
    id: str
    kind: str                  # "change" or "pause"
    t_start: float
    t_peak: float
    t_end: float
    score: float
    stroke_index: int          # strokes completed when it happened; aligns apprentices later
    progress: float            # fraction of the session
    evidence: list[Evidence] = field(default_factory=list)
    summary: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("t_start", "t_peak", "t_end", "score", "progress"):
            d[k] = round(d[k], 2)
        for e in d["evidence"]:
            for k in ("before", "during", "delta", "z"):
                e[k] = round(e[k], 2)
        return d


def _rolling_median_mad(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD of the n samples strictly before each index (NaN where undefined)."""
    med = np.full(len(x), np.nan)
    mad = np.full(len(x), np.nan)
    if len(x) <= n:
        return med, mad
    win = sliding_window_view(x, n)[:-1]          # win[k] covers x[k : k+n], ends before k+n
    m = np.median(win, axis=1)
    med[n:] = m
    mad[n:] = np.median(np.abs(win - m[:, None]), axis=1)
    return med, mad


def channel_z(feat: Features) -> dict[str, np.ndarray]:
    """Per-channel z of the upcoming CURRENT_S window against the preceding BASELINE_S."""
    nb, nc = int(BASELINE_S * FS), int(CURRENT_S * FS)
    out = {}
    for c in CHANNELS:
        x = feat.X[c]
        med, mad = _rolling_median_mad(x, nb)
        ahead = np.full(len(x), np.nan)
        if len(x) > nc:
            ahead[: len(x) - nc + 1] = sliding_window_view(x, nc).mean(axis=1)
        scale = np.maximum(1.4826 * mad, NOISE_FLOOR[c])
        z = (ahead - med) / scale
        out[c] = np.clip(np.nan_to_num(z, nan=0.0), -Z_CAP, Z_CAP)
    return out


def surprise_score(z: dict[str, np.ndarray]) -> np.ndarray:
    Z = np.abs(np.stack([z[c] for c in CHANNELS], axis=1))
    rms = np.sqrt((Z ** 2).mean(axis=1))
    return smooth(0.6 * Z.max(axis=1) + 0.4 * rms, 0.3)


def _pauses(feat: Features, min_s: float = 0.6) -> list[tuple[int, int]]:
    """Index ranges where the tool stopped moving mid-session."""
    still = ~_active(feat)
    edge = int(2.5 * FS)
    still[:edge] = False
    still[-edge:] = False
    runs, start = [], None
    for i, s in enumerate(np.append(still, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) / FS >= min_s:
                runs.append((start, i))
            start = None
    return runs


def _describe(kind: str, evidence: list[Evidence], duration: float) -> str:
    parts = []
    if kind == "pause":
        parts.append(f"stopped stroking for {duration:.1f}s")
    for e in [e for e in evidence if not e.trend][:3]:
        if kind == "pause" and e.channel in ("motion", "tempo", "loudness", "tone"):
            continue
        label = CHANNEL_META[e.channel]["label"].lower()
        verb = "rose" if e.delta > 0 else "dropped"
        parts.append(f"{label} {verb} {abs(e.delta):.1f}{e.unit} ({e.before:.1f}→{e.during:.1f})")
    for e in [e for e in evidence if e.trend][:1]:
        label = CHANNEL_META[e.channel]["label"].lower()
        verb = "climbed" if e.delta > 0 else "fell"
        parts.append(f"beforehand, {label} {verb} {abs(e.delta):.0f}{e.unit} over {TREND_S:.0f}s")
    text = "; ".join(parts)
    return text[0].upper() + text[1:] if text else "Behaviour changed"


def _evidence(feat: Features, z: dict[str, np.ndarray], i0: int, i1: int, peak: int) -> list[Evidence]:
    nb = int(BASELINE_S * FS)
    b0 = max(0, i0 - nb)
    ev = []
    for c in CHANNELS:
        zc = float(z[c][peak])
        if abs(zc) < 2.0:
            continue
        before = float(np.median(feat.X[c][b0:i0])) if i0 > b0 else float(feat.X[c][i0])
        during = float(np.mean(feat.X[c][i0:max(i1, i0 + 1)]))
        ev.append(Evidence(c, before, during, during - before, CHANNEL_META[c]["unit"], zc))
    ev.sort(key=lambda e: -abs(e.z))
    return ev + _trends(feat, i0)


def _active(feat: Features) -> np.ndarray:
    m = feat.X["motion"]
    moving = m[m > np.percentile(m, 20)]
    return m > 0.3 * np.median(moving) if len(moving) else np.ones(len(m), bool)


def _trends(feat: Features, i0: int) -> list[Evidence]:
    """Slow build-ups leading into an event, e.g. the stone's pitch creeping up as it dries.

    Only samples where the tool was moving count, so an earlier pause doesn't
    read as a trend.
    """
    old0, old1 = i0 - int(TREND_S * FS), i0 - int((TREND_S - 4) * FS)
    new0 = i0 - int(2 * FS)
    if old0 < 0:
        return []
    active = _active(feat)
    old_mask, new_mask = active[old0:old1], active[new0:i0]
    if old_mask.sum() < FS or new_mask.sum() < FS / 2:
        return []
    out = []
    for c in ("tone", "loudness", "angle", "tempo"):
        x = feat.X[c]
        before = float(np.median(x[old0:old1][old_mask]))
        during = float(np.median(x[new0:i0][new_mask]))
        z = (during - before) / NOISE_FLOOR[c]
        if abs(z) >= 3.0:
            out.append(Evidence(c, before, during, during - before, CHANNEL_META[c]["unit"], z, trend=True))
    out.sort(key=lambda e: -abs(e.z))
    return out


def detect(feat: Features, max_events: int = MAX_EVENTS) -> tuple[list[Surprise], np.ndarray]:
    """Return the strongest surprises (time-ordered) and the per-sample surprise score."""
    n = len(feat.t)
    z = channel_z(feat)
    score = surprise_score(z)
    # Nothing is surprising until the master has settled into a rhythm.
    settled = feat.stroke_times[2] + BASELINE_S if len(feat.stroke_times) > 2 else feat.duration
    score[: int(settled * FS)] = 0
    score[-int(2.5 * FS):] = 0

    # Pauses are high-value cues even when the score is modest: boost them.
    pauses = _pauses(feat)
    for a, b in pauses:
        boost = 3.5 + (b - a) / FS
        lo = max(0, a - int(0.5 * FS))
        score[lo:b] = np.maximum(score[lo:b], boost)

    picked: list[int] = []
    work = score.copy()
    gap = int(MIN_GAP_S * FS)
    while len(picked) < max_events:
        k = int(np.argmax(work))
        if work[k] < THRESHOLD:
            break
        picked.append(k)
        work[max(0, k - gap): k + gap] = 0

    events = []
    for k in sorted(picked):
        pause = next(((a, b) for a, b in pauses if a - FS <= k <= b), None)
        if pause:
            i0, i1, kind = pause[0], pause[1], "pause"
        else:
            half = score[k] * 0.5
            i0 = k
            while i0 > 0 and score[i0 - 1] > half and k - i0 < 4 * FS:
                i0 -= 1
            i1 = k
            while i1 < n - 1 and score[i1 + 1] > half and i1 - k < 4 * FS:
                i1 += 1
            kind = "change"
        ev = _evidence(feat, z, i0, i1, k)
        dur = (i1 - i0) / FS
        events.append(Surprise(
            id=f"s{len(events) + 1}",
            kind=kind,
            t_start=float(feat.t[i0]),
            t_peak=float(feat.t[k]),
            t_end=float(feat.t[min(i1, n - 1)]),
            score=float(score[k]),
            stroke_index=int(feat.strokes[i0]),
            progress=float(feat.t[i0] / max(feat.duration, 1e-6)),
            evidence=ev,
            summary=_describe(kind, ev, dur),
        ))
    return events, score
