"""How well does the surprise model find habits we know are there?

The simulator plants each tacit habit at a known time. Over many seeds we
count a habit as found when a surprise starts within TOLERANCE_S of it, and
a surprise as a false alarm when no planted habit is that close. Soak isn't a
moment to ask about; it is scored through the soak-time channel, so it is left out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from . import surprise
from .signals import extract
from .simulate import HEAT_PROFILES, PROFILES, generate, generate_heat

TOLERANCE_S = 2.5
SIMULATORS = {"sharpen": generate, "heat": generate_heat}
NOT_MOMENTS = {"soak"}
# The same master habits, performed with the steadiness of each profile's hands.
NOISE = {
    "master": (PROFILES["master"], HEAT_PROFILES["master"]),
    "skilled hands": (replace(PROFILES["master"], angle_jitter=0.6, drift_amp=1.0, tempo_jitter=0.06),
                      replace(HEAT_PROFILES["master"], jitter=1.0)),
    "novice hands": (replace(PROFILES["master"], angle_jitter=1.0, drift_amp=3.0, tempo_jitter=0.15),
                     replace(HEAT_PROFILES["master"], jitter=1.8)),
}


def evaluate(kind: str, seeds: range = range(20), profile=None) -> dict:
    planted, found, alarms, total, lag = Counter(), Counter(), 0, 0, []
    for seed in seeds:
        raw, events = SIMULATORS[kind](profile or "master", seed=seed)
        events = [e for e in events if e["habit"] not in NOT_MOMENTS]
        found_now, _ = surprise.detect(extract(raw, kind))
        total += len(found_now)
        for e in events:
            planted[e["habit"]] += 1
            near = [s.t_start - e["t"] for s in found_now if abs(s.t_start - e["t"]) < TOLERANCE_S]
            if near:
                found[e["habit"]] += 1
                lag.append(min(near, key=abs))
        alarms += sum(all(abs(s.t_start - e["t"]) >= TOLERANCE_S for e in events) for s in found_now)
    return {
        "kind": kind, "sessions": len(seeds),
        "recall": {h: found[h] / n for h, n in planted.items()},
        "precision": 1 - alarms / total if total else 0.0,
        "questions_per_session": total / len(seeds),
        "median_lag_s": sorted(lag)[len(lag) // 2] if lag else None,
    }


def report(seeds: range = range(20)) -> str:
    lines = []
    for noise, profiles in NOISE.items():
        for kind, profile in zip(SIMULATORS, profiles):
            r = evaluate(kind, seeds, profile)
            habits = ", ".join(f"{h} {v:.0%}" for h, v in r["recall"].items())
            lines.append(f"{noise:>13}  {kind:<7}  recall {habits} | precision {r['precision']:.0%} | "
                         f"{r['questions_per_session']:.1f} questions/session | median lag {r['median_lag_s']:+.1f}s")
    return f"{len(seeds)} simulated sessions each\n" + "\n".join(lines)
