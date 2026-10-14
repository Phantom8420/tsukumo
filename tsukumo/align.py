"""Score an apprentice session against the master's reference ("ghost").

Both sessions are expressed in units of tolerance around the master's own
levels, aligned with banded DTW (so a slower apprentice isn't punished for
being slower, only for doing it differently), then compared channel by
channel. A difference equal to the tolerance earns half credit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import FS, channel
from .signals import Features

ALIGN_FS = 5.0


def _downsample(M: np.ndarray, factor: int) -> np.ndarray:
    n = (len(M) // factor) * factor
    return M[:n].reshape(-1, factor, M.shape[1]).mean(axis=1)


def _spec(feat: Features, key: str) -> np.ndarray:
    return np.array([channel(feat.kind, c)[key] for c in feat.channels])


def normalise(feat: Features, center: np.ndarray) -> np.ndarray:
    return (feat.matrix() - center) / _spec(feat, "tol")


def dtw(A: np.ndarray, B: np.ndarray, band: float = 0.3) -> tuple[list[tuple[int, int]], float]:
    """Banded DTW between (n, d) and (m, d). Returns the warping path and mean step cost."""
    n, m = len(A), len(B)
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    cost = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_center = int(round(i * m / n))
        lo, hi = max(1, j_center - w), min(m, j_center + w)
        prev = D[i - 1]
        for j in range(lo, hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(prev[j - 1], prev[j], D[i, j - 1])
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        steps = ((D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1))
        _, i, j = min(steps, key=lambda s: s[0])
        i, j = max(i, 1), max(j, 1)
        path.append((i - 1, j - 1))
    path.reverse()
    return path, float(D[n, m] / len(path))


@dataclass
class WeakSpot:
    channel: str
    t_start: float
    t_end: float
    score: float
    message: str


@dataclass
class ScoreReport:
    overall: float
    channels: dict[str, float]
    weak_spots: list[WeakSpot] = field(default_factory=list)
    ghost: dict[str, list[float]] = field(default_factory=dict)   # master values on apprentice timeline
    ghost_t: list[float] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "overall": round(self.overall, 1),
            "channels": {c: round(v, 1) for c, v in self.channels.items()},
            "weak_spots": [{**ws.__dict__, "score": round(ws.score, 1),
                            "t_start": round(ws.t_start, 1), "t_end": round(ws.t_end, 1)}
                           for ws in self.weak_spots],
            "ghost": self.ghost,
            "ghost_t": self.ghost_t,
        }


def _advice(kind: str, name: str, signed_diff: float) -> str:
    unit = channel(kind, name)["unit"]
    amount = f"{abs(signed_diff):.1f}{unit}"
    high = signed_diff > 0
    return {
        "angle": f"Blade {amount} {'steeper' if high else 'flatter'} than the master",
        "motion": f"{'Pressing harder' if high else 'Lighter'} than the master by {amount}",
        "tempo": f"{'Faster' if high else 'Slower'} than the master by {amount}",
        "loudness": f"Stone {amount} {'louder' if high else 'quieter'} - "
                    f"{'too much pressure or a drying stone' if high else 'too little contact'}",
        "tone": f"Stone pitch {amount} {'higher' if high else 'lower'} - "
                f"{'the stone may be drying out' if high else 'check contact'}",
        "hue": f"Steel colour {amount} {'yellower - running hot' if high else 'redder - not yet at heat'}",
        "brightness": f"Glow {amount} {'brighter' if high else 'dimmer'} than the master",
        "hold": f"{'Held' if high else 'Soaked'} {amount} {'longer' if high else 'shorter'} at heat than the master",
    }[name]


def score(master: Features, apprentice: Features) -> ScoreReport:
    if master.kind != apprentice.kind:
        raise ValueError(f"can't score {apprentice.kind} work against a {master.kind} reference")
    chans = master.channels
    factor = int(FS / ALIGN_FS)
    center = np.median(master.matrix(), axis=0)
    M = _downsample(normalise(master, center), factor)
    A = _downsample(normalise(apprentice, center), factor)
    path, _ = dtw(A, M)
    ia = np.array([p[0] for p in path])
    im = np.array([p[1] for p in path])

    diff = A[ia] - M[im]                                  # in tolerance units
    credit = np.exp(-np.log(2) * diff ** 2)               # 1 at perfect, 0.5 at one tolerance
    channels = {c: float(100 * credit[:, k].mean()) for k, c in enumerate(chans)}
    overall = float(_spec(master, "weight") @ [channels[c] for c in chans])

    # Per apprentice sample: the master sample it was matched to (last match wins).
    match = np.zeros(len(A), dtype=int)
    match[ia] = im
    weak = _weak_spots(master, credit, diff, ia)

    ghost_native = M[match] * _spec(master, "tol") + center
    return ScoreReport(
        overall=overall,
        channels=channels,
        weak_spots=weak,
        ghost={c: np.round(ghost_native[:, k], 2).tolist() for k, c in enumerate(chans)},
        ghost_t=np.round(np.arange(len(A)) / ALIGN_FS, 2).tolist(),
    )


def _weak_spots(master: Features, credit: np.ndarray, diff: np.ndarray, ia: np.ndarray, top: int = 3,
                window_s: float = 4.0) -> list[WeakSpot]:
    """The worst few windows, each tied to the channel that cost the most there."""
    chans = master.channels
    n = int(ia.max()) + 1
    per_sample = np.ones((n, len(chans)))
    signed = np.zeros((n, len(chans)))
    np.minimum.at(per_sample, ia, credit)
    signed[ia] = diff

    w = max(1, int(window_s * ALIGN_FS))
    if n < w:
        return []
    kernel = np.ones(w) / w
    rolling = np.stack([np.convolve(per_sample[:, k], kernel, mode="valid") for k in range(len(chans))], 1)
    badness = ((1 - rolling) * _spec(master, "weight")).sum(axis=1)

    spots: list[WeakSpot] = []
    taken = np.zeros(len(badness), dtype=bool)
    for start in np.argsort(-badness):
        if len(spots) >= top:
            break
        if taken[start]:
            continue
        # Prefer a channel not already called out, so the advice isn't three of the same.
        used = {s.channel for s in spots}
        order = [k for k in np.argsort(rolling[start]) if rolling[start, k] <= 0.6]
        if not order:
            continue
        k = next((k for k in order if chans[k] not in used), order[0])
        c = chans[k]
        sd = float(np.mean(signed[start:start + w, k]) * channel(master.kind, c)["tol"])
        spots.append(WeakSpot(c, start / ALIGN_FS, (start + w) / ALIGN_FS,
                              100 * float(rolling[start, k]), _advice(master.kind, c, sd)))
        taken[max(0, start - w): start + w] = True
    return sorted(spots, key=lambda s: s.t_start)
