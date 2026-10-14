import numpy as np
import pytest

from tsukumo import align, surprise
from tsukumo.coach import LiveCoach
from tsukumo.signals import extract, resample
from tsukumo.simulate import generate


@pytest.fixture(scope="module")
def master():
    frames, events = generate("master", seed=0)
    return extract(frames), events


def test_resample_handles_unsorted_and_duplicate_timestamps():
    frames, _ = generate("master", seed=1)
    shuffled = np.concatenate([frames[100:], frames[:100], frames[:5]])
    t, V = resample(shuffled)
    assert np.allclose(np.diff(t), t[1] - t[0])
    assert V.shape[1] == 6


def test_stroke_count_matches_rhythm(master):
    feat, _ = master
    # ~90 spm for ~85 moving seconds, minus pauses and the slower micro-bevel.
    assert 100 <= len(feat.stroke_times) <= 130
    assert np.all(np.diff(feat.strokes) >= 0)


def test_surprises_land_on_the_masters_habits(master):
    feat, events = master
    found, score = surprise.detect(feat)
    assert len(score) == len(feat.t)
    for habit in ("burr", "micro"):
        t = next(e["t"] for e in events if e["habit"] == habit)
        assert any(abs(s.t_start - t) < 2.5 for s in found), habit
    first_water = next(e["t"] for e in events if e["habit"] == "water")
    water = next(s for s in found if abs(s.t_start - first_water) < 2.5)
    assert water.kind == "pause"
    assert any(e.trend and e.channel == "tone" and e.delta > 0 for e in water.evidence)


def test_steady_session_has_no_surprises():
    from tsukumo.simulate import Profile
    frames, _ = generate(Profile(water_at_hz=None, burr_at=None, flip_at=None, micro_at=None), seed=3)
    found, _ = surprise.detect(extract(frames))
    assert found == []


def test_scores_rank_skill(master):
    feat, _ = master
    scores = {p: align.score(feat, extract(generate(p, seed=5)[0])).overall
              for p in ("master", "skilled", "novice")}
    assert scores["master"] > 90
    assert scores["master"] > scores["skilled"] > scores["novice"]
    assert scores["novice"] < 60


def test_dtw_identity_path():
    A = np.random.default_rng(0).normal(size=(40, 3))
    path, cost = align.dtw(A, A)
    assert cost == pytest.approx(0)
    assert path == [(i, i) for i in range(40)]


def _run(coach, profile, seed):
    frames, _ = generate(profile, seed=seed)
    events = []
    for f in frames:
        events += coach.update(f)
    return events


def test_live_coach_is_quiet_for_master_and_busy_for_novice(master):
    feat, _ = master
    buzzes = lambda evs: [e for e in evs if e["type"] == "buzz"]
    assert len(buzzes(_run(LiveCoach(feat, [], "en"), "master", 7))) <= 2
    novice = buzzes(_run(LiveCoach(feat, [], "vi"), "novice", 7))
    assert len(novice) >= 10
    assert all(e["message"] == e["text"]["vi"] for e in novice)


def test_live_cards_fire_once_in_stroke_order(master):
    feat, _ = master
    cards = [{"id": "late", "stroke_index": 60}, {"id": "early", "stroke_index": 20}]
    shown = [e["card"]["id"] for e in _run(LiveCoach(feat, cards, "en"), "skilled", 2) if e["type"] == "card"]
    assert shown == ["early", "late"]


def test_live_ticks_carry_ghost(master):
    feat, _ = master
    ticks = [e for e in _run(LiveCoach(feat, [], "en"), "skilled", 4) if e["type"] == "tick"]
    assert len(ticks) > 500
    assert ticks[-1]["ghost"] is not None and ticks[-1]["k"] > 100
