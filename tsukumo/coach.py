"""Live coaching: compare the apprentice, frame by frame, with the master's ghost.

The apprentice is aligned to the master by work done (strokes), not by clock
time: at stroke 40 the apprentice is compared with what the master was doing
at their stroke 40. A slow apprentice is therefore never told to hurry
because the master was "further along", and kotsu cards fire at the same
point in the work where the master's habit happened.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .config import CHANNELS, FS
from .signals import Features, StrokeCounter

# Live thresholds are a bit looser than scoring tolerances: a buzz has to be worth it.
LIVE_TOL = {"angle": 2.5, "loudness": 5.0, "tempo": 22.0}
SUSTAIN_S = 0.6
COOLDOWN_S = 3.0
STEADY_MARGIN_S = 1.5
RESUME_GRACE_S = 1.0
GLOBAL_COOLDOWN_S = 2.0
TICK_S = 0.1

PATTERNS = {"angle": [90, 60, 90], "loudness": [250], "tempo": [40, 40, 40, 40, 40]}

MESSAGES = {
    ("angle", 1): {"en": "Lower the blade ~{d}°", "ja": "刃を約{d}°寝かせて",
                   "vi": "Hạ lưỡi dao xuống ~{d}°", "id": "Turunkan sudut pisau ~{d}°"},
    ("angle", -1): {"en": "Raise the blade ~{d}°", "ja": "刃を約{d}°起こして",
                    "vi": "Nâng lưỡi dao lên ~{d}°", "id": "Naikkan sudut pisau ~{d}°"},
    ("loudness", 1): {"en": "Ease the pressure", "ja": "力を抜いて",
                      "vi": "Nhẹ tay hơn", "id": "Kurangi tekanan"},
    ("loudness", -1): {"en": "A little more pressure", "ja": "もう少し押して",
                       "vi": "Ấn mạnh hơn một chút", "id": "Tambah sedikit tekanan"},
    ("tempo", 1): {"en": "Slow down", "ja": "ゆっくり",
                   "vi": "Chậm lại", "id": "Pelan-pelan"},
    ("tempo", -1): {"en": "A little faster", "ja": "もう少し速く",
                    "vi": "Nhanh hơn một chút", "id": "Sedikit lebih cepat"},
}


class _Ema:
    def __init__(self, tau: float):
        self.tau = tau
        self.value: float | None = None
        self.t: float | None = None

    def update(self, t: float, x: float) -> float:
        if self.value is None:
            self.value, self.t = x, t
            return x
        alpha = 1 - math.exp(-max(t - self.t, 0.0) / self.tau)
        self.value += alpha * (x - self.value)
        self.t = t
        return self.value


class LiveTracker:
    """Online version of signals.extract, fed one raw frame at a time."""

    def __init__(self):
        self.angle = _Ema(0.15)
        self.energy = _Ema(0.3)
        self.loud = _Ema(0.25)
        self.tone = _Ema(0.25)
        self.strokes = StrokeCounter()
        self.t = 0.0
        self.t0: float | None = None
        self._recent: deque[tuple[float, float]] = deque()
        self.peak = 0.0     # largest |accel| over the last half second: drops the instant strokes stop

    def update(self, frame) -> dict[str, float]:
        t_ms, angle, ax, ay, az, loud, tone = frame
        if self.t0 is None:
            self.t0 = t_ms / 1000.0
        t = t_ms / 1000.0 - self.t0
        self.t = t
        self.strokes.update(t, (ax, ay, az))
        self._recent.append((t, math.sqrt(ax * ax + ay * ay + az * az)))
        while t - self._recent[0][0] > 0.5:
            self._recent.popleft()
        self.peak = max(a for _, a in self._recent)
        st = self.strokes.times
        tempo = 0.0
        # Two intervals that span a pause say nothing about rhythm.
        if len(st) >= 3 and t - st[-1] <= 2.0 and st[-1] - st[-3] <= 3.0:
            tempo = 60.0 / max((st[-1] - st[-3]) / 2, 1e-3)
        return {
            "angle": abs(self.angle.update(t, angle)),
            "motion": math.sqrt(max(self.energy.update(t, ax * ax + ay * ay + az * az), 0.0)),
            "tempo": tempo,
            "loudness": self.loud.update(t, loud),
            "tone": self.tone.update(t, tone),
        }

    @property
    def progress(self) -> float:
        """Strokes done, with the current stroke counted fractionally so the ghost moves smoothly."""
        st = self.strokes.times
        if len(st) < 2:
            return float(len(st))
        interval = st[-1] - st[-2]
        return len(st) + min(1.0, (self.t - st[-1]) / max(interval, 1e-3)) * 0.999


class LiveCoach:
    def __init__(self, reference: Features | None, cards: list[dict], lang: str = "en"):
        self.ref = reference
        self.cards = sorted(cards, key=lambda c: c["stroke_index"])
        self.lang = lang
        self.tracker = LiveTracker()
        self.shown: set[str] = set()
        self._off_since: dict[str, float] = {}
        self._last_alert: dict[str, float] = {}
        self._last_tick = -1.0
        self._still_at = -1e9
        self._moving_thr = 0.8
        self._steady = None
        if reference is not None:
            m = reference.X["motion"]
            moving = m[m > np.percentile(m, 20)]
            self._moving_thr = 0.3 * float(np.median(moving)) if len(moving) else 0.8
            # Master samples well clear of any pause: the only places worth coaching against.
            active = (m > self._moving_thr).astype(float)
            n = int(STEADY_MARGIN_S * FS) * 2 + 1
            self._steady = np.convolve(active, np.ones(n), mode="same") >= n - 0.5

    def _ghost_index(self, k: float) -> int | None:
        if self.ref is None or len(self.ref.stroke_times) == 0:
            return None
        k = min(k, float(len(self.ref.stroke_times)))
        return self.ref.index_at(self.ref.time_at_stroke(k))

    def ghost(self, k: float) -> dict[str, float] | None:
        i = self._ghost_index(k)
        return None if i is None else {c: float(self.ref.X[c][i]) for c in CHANNELS}

    def update(self, frame) -> list[dict]:
        live = self.tracker.update(frame)
        t, k = self.tracker.t, self.tracker.progress
        ghost = self.ghost(k)
        events: list[dict] = []

        if ghost is not None and self._steady[self._ghost_index(k)]:
            events += self._check(t, live, ghost)
        else:
            self._off_since.clear()
        for card in self.cards:
            if card["id"] not in self.shown and k >= card["stroke_index"] - 0.5:
                self.shown.add(card["id"])
                events.append({"type": "card", "t": round(t, 2), "card": card})

        if t - self._last_tick >= TICK_S:
            self._last_tick = t
            events.append({
                "type": "tick", "t": round(t, 2), "k": round(k, 2),
                "live": {c: round(v, 2) for c, v in live.items()},
                "ghost": {c: round(v, 2) for c, v in ghost.items()} if ghost else None,
            })
        return events

    def _check(self, t: float, live: dict, ghost: dict) -> list[dict]:
        # Only coach while both are actually stroking: a pause on either side isn't a deviation.
        if self.tracker.peak < self._moving_thr * 1.5:
            self._still_at = t
        if (self.tracker.strokes.count < 3 or live["tempo"] == 0
                or t - self._still_at < RESUME_GRACE_S):
            self._off_since.clear()
            return []
        out = []
        for c, tol in LIVE_TOL.items():
            d = live[c] - ghost[c]
            if abs(d) <= tol:
                self._off_since.pop(c, None)
                continue
            since = self._off_since.setdefault(c, t)
            if t - since < SUSTAIN_S or t - self._last_alert.get(c, -1e9) < COOLDOWN_S:
                continue
            # One cue at a time; a phone buzzing twice at once is just noise.
            if t - max(self._last_alert.values(), default=-1e9) < GLOBAL_COOLDOWN_S:
                continue
            self._last_alert[c] = t
            direction = 1 if d > 0 else -1
            text = {lang: msg.format(d=round(abs(d))) for lang, msg in MESSAGES[(c, direction)].items()}
            out.append({"type": "buzz", "t": round(t, 2), "channel": c, "direction": direction,
                        "delta": round(d, 2), "pattern": PATTERNS[c], "text": text,
                        "message": text.get(self.lang, text["en"])})
        return out
