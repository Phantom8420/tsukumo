"""Raw phone frames -> the five channels everything else reasons about.

The same StrokeCounter and smoothing constants are used offline (whole
session) and live (frame by frame), so a stroke means the same thing in the
review screen and in the apprentice's buzz.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import CHANNELS, FS


def resample(raw: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Resample (N, 7) raw frames onto a uniform grid.

    Returns seconds-from-start and an (M, 6) array of angle, ax, ay, az, loud, tone.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"expected (N, 7) frames, got {raw.shape}")
    raw = raw[np.argsort(raw[:, 0], kind="stable")]
    t = raw[:, 0] / 1000.0
    keep = np.concatenate([[True], np.diff(t) > 0])
    raw, t = raw[keep], t[keep]
    if len(t) < 2:
        raise ValueError("need at least two frames")
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    cols = [np.interp(grid, t, raw[:, i]) for i in range(1, 7)]
    return grid - grid[0], np.stack(cols, axis=1)


def smooth(x: np.ndarray, seconds: float, fs: float = FS) -> np.ndarray:
    """Centered moving average with edge padding; output has the input's length."""
    n = max(1, int(round(seconds * fs)))
    if n == 1 or len(x) == 0:
        return np.asarray(x, dtype=float).copy()
    pad = np.pad(np.asarray(x, dtype=float), (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


class StrokeCounter:
    """Counts back-and-forth strokes from linear acceleration.

    Acceleration is projected onto its dominant axis over a short sliding
    window (so it works whichever way the phone is strapped on), then sign
    changes are counted with hysteresis. One push plus one pull is one stroke.
    """

    def __init__(self, window_s: float = 2.5, hysteresis: float = 0.5,
                 min_std: float = 0.25, refresh: int = 10):
        self.window_s = window_s
        self.hysteresis = hysteresis
        self.min_std = min_std
        self.refresh = refresh
        self._t: deque[float] = deque()
        self._a: deque[tuple[float, float, float]] = deque()
        self._axis: np.ndarray | None = None
        self._state = 0
        self._half = 0
        self._n = 0
        self.times: list[float] = []

    @property
    def count(self) -> int:
        return len(self.times)

    def update(self, t: float, a) -> int:
        self._t.append(t)
        self._a.append((float(a[0]), float(a[1]), float(a[2])))
        while t - self._t[0] > self.window_s:
            self._t.popleft()
            self._a.popleft()
        if len(self._t) < 8:
            return self.count

        A = np.asarray(self._a)
        A = A - A.mean(axis=0)
        if self._axis is None or self._n % self.refresh == 0:
            axis = np.linalg.svd(A, full_matrices=False)[2][0]
            if self._axis is not None and axis @ self._axis < 0:
                axis = -axis
            self._axis = axis
        self._n += 1

        proj = A @ self._axis
        std = proj.std()
        if std < self.min_std:
            return self.count
        p, thr = proj[-1], self.hysteresis * std
        if p > thr and self._state != 1:
            self._state = 1
            self._flip(t)
        elif p < -thr and self._state != -1:
            self._state = -1
            self._flip(t)
        return self.count

    def _flip(self, t: float) -> None:
        self._half += 1
        if self._half % 2 == 0:
            self.times.append(t)


def tempo_series(stroke_times: np.ndarray, t: np.ndarray, idle_s: float = 2.0) -> np.ndarray:
    """Strokes per minute at each sample; zero once strokes stop for idle_s."""
    out = np.zeros_like(t, dtype=float)
    st = np.asarray(stroke_times, dtype=float)
    if len(st) < 2:
        return out
    rates = 60.0 / np.maximum(np.diff(st), 1e-3)
    idx = np.searchsorted(st, t, side="right") - 1
    valid = idx >= 1
    valid[valid] &= (t[valid] - st[idx[valid]]) <= idle_s
    out[valid] = rates[idx[valid] - 1]
    return smooth(out, 1.0)


@dataclass
class Features:
    t: np.ndarray                 # seconds from start, uniform at FS
    X: dict[str, np.ndarray]      # channel -> values, same length as t
    stroke_times: np.ndarray      # seconds at which each stroke completed
    strokes: np.ndarray           # cumulative stroke count at each sample
    side: np.ndarray | None = None  # +1/-1: which face of the blade is down

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def matrix(self, channels=CHANNELS) -> np.ndarray:
        return np.stack([self.X[c] for c in channels], axis=1)

    def index_at(self, seconds: float) -> int:
        return int(np.clip(round(seconds * FS), 0, len(self.t) - 1))

    def time_at_stroke(self, k: float) -> float:
        """Session time at which the k-th stroke completed (fractional k interpolates)."""
        if len(self.stroke_times) == 0:
            return 0.0
        ks = np.arange(1, len(self.stroke_times) + 1)
        return float(np.interp(k, np.concatenate([[0], ks]),
                               np.concatenate([[0.0], self.stroke_times])))

    def to_json(self, fs_out: float = 10.0) -> dict:
        step = max(1, int(round(FS / fs_out)))
        return {
            "t": np.round(self.t[::step], 2).tolist(),
            "X": {c: np.round(v[::step], 2).tolist() for c, v in self.X.items()},
            "strokes": self.strokes[::step].astype(int).tolist(),
            "stroke_times": np.round(self.stroke_times, 2).tolist(),
        }


def extract(raw: np.ndarray) -> Features:
    t, V = resample(raw)
    accel = V[:, 1:4]

    counter = StrokeCounter()
    strokes = np.empty(len(t), dtype=int)
    for i in range(len(t)):
        strokes[i] = counter.update(t[i], accel[i])
    stroke_times = np.asarray(counter.times)

    # Which face is on the stone is the sign; the angle itself is the magnitude.
    signed = smooth(V[:, 0], 0.3)
    X = {
        "angle": np.abs(signed),
        "motion": np.sqrt(smooth((accel ** 2).sum(axis=1), 0.6)),
        "tempo": tempo_series(stroke_times, t),
        "loudness": smooth(V[:, 4], 0.5),
        "tone": smooth(V[:, 5], 0.5),
    }
    return Features(t=t, X=X, stroke_times=stroke_times, strokes=strokes, side=np.sign(signed))
