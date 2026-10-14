import shutil

import pytest
from fastapi.testclient import TestClient

from tsukumo import config
from tsukumo.app import create_app
from tsukumo.seed import seed
from tsukumo.services import Tsukumo
from tsukumo.simulate import generate
from tsukumo.store import Store


@pytest.fixture(scope="session")
def seeded(tmp_path_factory):
    """Seed once; each test gets its own copy."""
    root = tmp_path_factory.mktemp("seeded")
    store = Store(root / "t.db", root / "frames")
    seed(Tsukumo(store))
    store.close()
    return root


@pytest.fixture()
def client(seeded, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFFLINE", True)
    monkeypatch.setattr("tsukumo.kotsu.OFFLINE", True)
    shutil.copytree(seeded, tmp_path, dirs_exist_ok=True)
    app = create_app(tmp_path / "t.db", tmp_path / "frames")
    with TestClient(app) as c:
        yield c


def test_reference_data(client):
    assert {p["id"] for p in client.get("/api/people").json()} >= {"tanaka", "nguyen"}
    info = client.get("/api/info").json()
    assert info["channel_order"] == list(config.CHANNELS)
    assert client.get("/api/qr", params={"data": "https://x"}).text.startswith("<?xml")


def test_master_review_flow(client):
    master = client.get("/api/sessions", params={"role": "master"}).json()[0]
    detail = client.get(f"/api/sessions/{master['id']}").json()
    assert detail["is_reference"]
    assert len(detail["features"]["t"]) == len(detail["surprise_score"])
    open_ = [s for s in detail["surprises"] if s["status"] == "open"]
    assert open_, "the micro-bevel surprise is left open for the demo"

    r = client.post(f"/api/sessions/{master['id']}/surprises/{open_[0]['id']}/answer",
                    json={"transcript": "最後は少し刃を起こして、力を抜いてゆっくり。", "lang": "ja"})
    card = r.json()
    assert card["source"] == "offline"
    assert set(card["content"]) == set(config.LANGS)
    after = client.get(f"/api/sessions/{master['id']}").json()
    assert next(s for s in after["surprises"] if s["id"] == open_[0]["id"])["status"] == "answered"

    client.delete(f"/api/cards/{card['id']}")
    again = client.get(f"/api/sessions/{master['id']}").json()
    assert next(s for s in again["surprises"] if s["id"] == open_[0]["id"])["status"] == "open"


def test_live_apprentice_session_over_websocket(client):
    ses = client.post("/api/sessions", json={"skill_id": "honbazuke", "person_id": "nguyen"}).json()
    frames, _ = generate("novice", seed=99)
    got = []
    with client.websocket_connect(f"/ws/watch/{ses['id']}") as watch:
        assert watch.receive_json()["type"] == "device"
        with client.websocket_connect(f"/ws/device/{ses['id']}") as dev:
            ready = dev.receive_json()
            assert ready["type"] == "ready" and ready["lang"] == "vi"
            for i in range(0, len(frames), 50):
                dev.send_json({"type": "frames", "frames": frames[i:i + 50].tolist()})
            dev.send_json({"type": "end"})
            while True:
                msg = dev.receive_json()
                got.append(msg["type"])
                if msg["type"] in ("summary", "error"):
                    break
    assert "buzz" in got and "card" in got
    assert msg["type"] == "summary" and msg["report"]["overall"] < 60

    passport = client.get("/api/passport/nguyen").json()
    assert passport["skills"][0]["sessions"] == 4


def test_replay_runs_a_whole_session_over_http(client):
    s = client.post("/api/sessions", json={"skill_id": "honbazuke", "person_id": "nguyen"}).json()
    r = client.post(f"/api/sessions/{s['id']}/replay", json={"profile": "novice", "seed": 3}).json()
    kinds = {e["type"] for e in r["events"]}
    assert {"tick", "buzz", "card"} <= kinds
    assert [e["t"] for e in r["events"]] == sorted(e["t"] for e in r["events"])
    assert r["summary"]["report"]["overall"] < 75
    assert client.get(f"/api/sessions/{s['id']}").json()["session"]["status"] == "done"
    assert client.post(f"/api/sessions/{s['id']}/replay", json={}).status_code == 400


def test_short_recording_is_discarded(client):
    ses = client.post("/api/sessions", json={"skill_id": "honbazuke", "person_id": "sato"}).json()
    with client.websocket_connect(f"/ws/device/{ses['id']}") as dev:
        dev.receive_json()
        dev.send_json({"type": "frames", "frames": [[0, 15, 0, 0, 0, -30, 1800], [100, 15, 0, 0, 0, -30, 1800]]})
        dev.send_json({"type": "end"})
        while (msg := dev.receive_json())["type"] not in ("summary", "error"):
            pass
    assert msg["type"] == "error"
    assert client.get(f"/api/sessions/{ses['id']}").status_code == 404


def test_radar_and_ledger(client):
    radar = client.get("/api/radar").json()
    assert radar[0]["status"] == "critical"
    sharpening = next(r for r in radar if r["skill"]["id"] == "honbazuke")
    assert sharpening["cards"] == 2 and any(s["measured"] for s in sharpening["successors"])
    ledger = client.get("/api/ledger").json()
    assert ledger[0]["master"]["id"] == "tanaka" and ledger[0]["credits_yen"] > 0


def test_errors(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions", json={"skill_id": "honbazuke", "person_id": "ghost"}).status_code == 404
    master = client.get("/api/sessions", params={"role": "apprentice"}).json()[0]
    assert client.post("/api/skills/honbazuke/reference", json={"session_id": master["id"]}).status_code == 400
