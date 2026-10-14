"""Synthetic whetstone-sharpening sessions.

The master has four tacit habits the written SOP doesn't mention:
  water  - splashes the stone when its sound turns dry and high
  burr   - stops to feel for the burr with a thumb before switching sides
  flip   - switches to the other face once the burr runs the full edge
  micro  - finishes with a slightly steeper, lighter, slower micro-bevel
Apprentices lack some of them and are noisier on angle and rhythm. This is
what lets the whole pipeline be demoed and tested without a phone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STROKE_AXIS = np.array([0.9, 0.3, 0.1]) / np.linalg.norm([0.9, 0.3, 0.1])


@dataclass(frozen=True)
class Profile:
    angle: float = 15.0
    angle_jitter: float = 0.3
    drift_amp: float = 0.0          # slow wandering of the angle, degrees
    drift_period: float = 11.0
    tempo: float = 90.0             # strokes per minute
    tempo_jitter: float = 0.03      # fractional
    pressure: float = 1.0
    water_at_hz: float | None = 700.0   # splash when the stone's pitch rises this much
    burr_at: float | None = 50.0
    flip_at: float | None = 53.0
    micro_at: float | None = 74.0
    micro_angle: float = 2.5
    duration: float = 90.0
    habits: tuple[str, ...] = field(default=("water", "burr", "flip", "micro"))


PROFILES = {
    "master": Profile(),
    "skilled": Profile(angle_jitter=0.6, drift_amp=1.0, tempo=95, tempo_jitter=0.06,
                       pressure=1.1, water_at_hz=950.0, burr_at=55.0, flip_at=58.0,
                       micro_at=80.0, micro_angle=1.5, duration=92.0),
    "novice": Profile(angle_jitter=1.0, drift_amp=3.0, tempo=112, tempo_jitter=0.15,
                      pressure=1.45, water_at_hz=None, burr_at=None, flip_at=50.0,
                      micro_at=None, duration=95.0, habits=("flip",)),
}

DRY_HZ_PER_S = 35.0
DRY_DB_PER_S = 0.12


def generate(profile: str | Profile = "master", fs: float = 50.0, seed: int = 0
             ) -> tuple[np.ndarray, list[dict]]:
    """Return (N, 7) raw frames like the phone sends, plus the habit events performed."""
    p = PROFILES[profile] if isinstance(profile, str) else profile
    rng = np.random.default_rng(seed)
    n = int(p.duration * fs)
    t = np.arange(n) / fs + rng.uniform(-0.003, 0.003, n)
    t[0] = 0.0
    dt = 1.0 / fs

    frames = np.zeros((n, 7))
    events: list[dict] = []
    phase = 0.0
    last_water = 0.0
    pause_until = -1.0
    lift = 0.0
    side = 1.0
    micro = False
    tempo_wobble = 0.0
    done = {"burr": False, "flip": False, "micro": False}

    for i in range(n):
        ti = i * dt
        dryness = ti - last_water

        if p.water_at_hz and pause_until < ti and dryness * DRY_HZ_PER_S >= p.water_at_hz:
            pause_until = ti + 1.6
            last_water = ti + 1.6
            events.append({"habit": "water", "t": ti, "duration": 1.6})
        if p.burr_at and not done["burr"] and ti >= p.burr_at:
            done["burr"] = True
            pause_until = ti + 2.2
            events.append({"habit": "burr", "t": ti, "duration": 2.2})
        if p.flip_at and not done["flip"] and ti >= p.flip_at:
            done["flip"] = True
            side = -1.0
            events.append({"habit": "flip", "t": ti})
        if p.micro_at and not done["micro"] and ti >= p.micro_at:
            done["micro"] = True
            micro = True
            events.append({"habit": "micro", "t": ti})

        moving = 1.5 <= ti <= p.duration - 1.5 and ti >= pause_until
        paused = not moving

        tempo_wobble += rng.normal(0, p.tempo_jitter * 0.4) * np.sqrt(dt)
        tempo_wobble *= 0.995
        spm = p.tempo * (0.78 if micro else 1.0) * (1 + tempo_wobble)
        pressure = p.pressure * (0.6 if micro else 1.0)
        if moving:
            phase += 2 * np.pi * spm / 60.0 * dt

        # Lift the blade off the stone to look at it during a pause.
        lift += ((25.0 if paused and ti > 1.5 else 0.0) - lift) * min(1.0, dt / 0.25)

        base = p.angle + (p.micro_angle if micro else 0.0)
        drift = p.drift_amp * np.sin(2 * np.pi * ti / p.drift_period + 0.7)
        angle = side * (base + drift + lift) + 0.3 * np.sin(phase) + rng.normal(0, p.angle_jitter)

        if moving:
            amp = 3.5 * (spm / 90.0) ** 2 * (0.7 + 0.3 * pressure)
            a = STROKE_AXIS * amp * np.sin(phase) + rng.normal(0, 0.15, 3)
            dry_hz = min(dryness, 60) * DRY_HZ_PER_S
            dry_db = min(dryness, 60) * DRY_DB_PER_S
            loud = -32 + 8 * np.log(pressure) + dry_db + 1.5 * abs(np.sin(phase)) + rng.normal(0, 0.8)
            tone = 1800 + dry_hz + 250 * (pressure - 1) + rng.normal(0, 60)
        else:
            a = rng.normal(0, 0.05, 3)
            loud = -55 + rng.normal(0, 1.5)
            tone = 400 + rng.normal(0, 150)

        frames[i] = (t[i] * 1000.0, angle, a[0], a[1], a[2], loud, tone)

    return frames, events
