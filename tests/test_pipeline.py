import numpy as np
import pytest

from tsukumo import align, surprise
from tsukumo.coach import LiveCoach
from tsukumo.signals import extract, resample
from tsukumo.simulate import generate, generate_heat


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


def _heat(peak_hue=30.0, hold_s=20.0, fs=30.0):
    """Hand-rolled heat run: glow rises for 20s, soaks at peak_hue, then quenches to dark."""
    t = np.arange(0, 20 + hold_s + 10, 1 / fs)
    ramp = np.clip(t / 20, 0, 1)
    lit = t < 20 + hold_s
    bright = np.where(lit, 80 * ramp, 0.0)
    hue = np.where(lit, peak_hue * ramp, 0.0)
    a = np.zeros((len(t), 3))
    return np.column_stack([t * 1000, hue, bright, a])


def test_heat_tracks_soak_time_and_scores_by_colour():
    master = extract(_heat(), "heat")
    assert master.channels == ("hue", "brightness", "hold", "motion")
    assert master.X["hold"].max() == pytest.approx(20, abs=1)
    assert align.score(master, extract(_heat(), "heat")).overall > 95
    assert align.score(master, extract(_heat(peak_hue=45, hold_s=8), "heat")).overall < 75
    with pytest.raises(ValueError):
        align.score(master, extract(generate("master", seed=0)[0]))


def test_heat_coach_says_too_hot():
    coach = LiveCoach(extract(_heat(), "heat"), [], "en", "heat")
    buzzes = [e for f in _heat(peak_hue=45) for e in coach.update(f) if e["type"] == "buzz"]
    assert buzzes and buzzes[0]["channel"] == "hue" and buzzes[0]["direction"] == 1


def test_real_footage_gives_strokes_and_habits():
    """Frames the camera tracker took from real Commons footage, not the simulator."""
    from tsukumo.seed import FOOTAGE

    feat = extract(np.load(FOOTAGE), "sharpen")
    assert 20 <= len(feat.stroke_times) <= 80
    assert all(np.isfinite(feat.X[c]).all() for c in feat.channels)
    # Labelled by eye from the clip: the blade lifts off the wheel at these times, and the
    # camera is moved from 33 to 36 s. Every lift should be asked about; the camera move should not.
    found = surprise.detect(feat)[0]
    for lift in (14.6, 24.5, 42.9):
        assert any(s.t_start - 1 <= lift <= s.t_end + 1 for s in found), lift
    assert not any(33 <= s.t_start <= 36 for s in found)


def test_heat_simulator_ranks_profiles_and_catches_the_turn_and_check():
    master = extract(generate_heat("master", seed=0)[0], "heat")
    scores = {p: align.score(master, extract(generate_heat(p, seed=5)[0], "heat")).overall
              for p in ("master", "skilled", "novice")}
    assert scores["master"] > 95 > scores["skilled"] > 80 > 75 > scores["novice"]
    frames, events = generate_heat("master", seed=1)
    found = [s.t_start for s in surprise.detect(extract(frames, "heat"))[0]]
    for habit in ("turn", "check"):
        at = next(e["t"] for e in events if e["habit"] == habit)
        assert any(abs(t - at) < 2.5 for t in found), habit


def test_detection_finds_planted_habits_without_false_alarms():
    from tsukumo.evaluate import evaluate
    sharpen, heat = evaluate("sharpen", range(5)), evaluate("heat", range(5))
    for r in (sharpen, heat):
        assert r["precision"] == 1.0
    assert all(sharpen["recall"][h] == 1.0 for h in ("water", "burr", "micro"))
    assert sharpen["recall"]["flip"] == 0.0     # angle is unsigned on purpose: a side change looks like nothing
    assert all(heat["recall"][h] == 1.0 for h in ("turn", "check", "quench"))
