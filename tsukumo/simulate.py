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


# -- heat treatment (焼入れ) ------------------------------------------------------------
#
# The master's tacit habits at the forge:
#   soak  - holds the blade at an even bright-cherry colour well past "it looks red"
#   turn  - rolls the blade over in the fire halfway through the soak so the spine catches up
#   check - pulls it out for a second at the end to read the colour away from the fire's glare
#   quench - plunges and agitates up and down in the water, never side to side
# Apprentices overshoot the colour, cut the soak short and skip the turn and the check.


@dataclass(frozen=True)
class HeatProfile:
    hue: float = 18.0               # bright cherry
    glow: float = 68.0
    jitter: float = 0.6
    heat_up: float = 22.0           # seconds to reach colour
    soak: float = 14.0
    habits: tuple[str, ...] = ("soak", "turn", "check", "quench")


HEAT_PROFILES = {
    "master": HeatProfile(),
    "skilled": HeatProfile(hue=20.0, glow=71.0, jitter=1.0, heat_up=19.0, soak=10.0,
                           habits=("soak", "turn", "quench")),
    "novice": HeatProfile(hue=27.0, glow=82.0, jitter=1.8, heat_up=15.0, soak=4.0, habits=("quench",)),
}


def generate_heat(profile: str | HeatProfile = "master", fs: float = 30.0, seed: int = 0
                  ) -> tuple[np.ndarray, list[dict]]:
    """Return (N, 6) raw frames [t_ms, hue, bright, ax, ay, az] plus the habit events performed."""
    p = HEAT_PROFILES[profile] if isinstance(profile, str) else profile
    rng = np.random.default_rng(seed)
    start = 3.0                                   # blade goes into the fire
    soak_at = start + p.heat_up
    turn_at = soak_at + p.soak / 2 if "turn" in p.habits else None
    check_at = soak_at + p.soak - 1.2 if "check" in p.habits else None
    quench_at = soak_at + p.soak
    n = int((quench_at + 12.0) * fs)
    events = [{"habit": "soak", "t": soak_at, "duration": p.soak}] if "soak" in p.habits else []
    events += [{"habit": h, "t": at} for h, at in (("turn", turn_at), ("check", check_at), ("quench", quench_at)) if at]

    frames = np.zeros((n, 6))
    hue = glow = 0.0
    for i in range(n):
        ti = i / fs
        if ti < start:
            target_h, target_g = 0.0, 0.0
        elif ti < quench_at:
            # Steel heats fast at first and slowly as it nears the fire's temperature.
            k = 1 - np.exp(-3 * min(ti - start, p.heat_up) / p.heat_up)
            target_h, target_g = p.hue * k / 0.95, p.glow * k / 0.95
        else:
            target_h = target_g = 0.0
        if check_at and check_at <= ti < check_at + 1.0:
            target_h, target_g = target_h - 3.0, target_g - 9.0     # out of the fire's glare: reads darker
        # Colour follows heat with a lag; quenching kills the glow in about a second.
        rate = 1.2 if ti >= quench_at else 0.25
        hue += (target_h - hue) * min(1.0, (1 / fs) / rate)
        glow += (target_g - glow) * min(1.0, (1 / fs) / rate)
        lit = glow > 15
        h = max(0.0, hue + (rng.normal(0, p.jitter) if lit else rng.normal(0, 4)))
        g = max(0.0, glow + rng.normal(0, p.jitter * 1.5))

        a = rng.normal(0, 0.08, 3)
        if turn_at and turn_at <= ti < turn_at + 1.2:
            a += np.array([0.0, 0.0, 2.2 * np.sin(2 * np.pi * (ti - turn_at) / 1.2)])
        if quench_at <= ti < quench_at + 6.0:
            # Plunge, then up-and-down agitation.
            a += np.array([0.3, 3.0 * np.sin(2 * np.pi * 1.5 * (ti - quench_at)), 0.0]) * np.exp(-(ti - quench_at) / 4)
        frames[i] = (ti * 1000.0, h, g, *a)
    return frames, events
