"""HTTP + WebSocket API, and the static web app."""

from __future__ import annotations

import io
import os
import secrets
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import qrcode
import qrcode.image.svg
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .live import Hub
from .services import NotFound, Tsukumo, by_camera
from .simulate import PROFILES
from .store import Store


class NewSession(BaseModel):
    skill_id: str
    person_id: str


class Simulate(BaseModel):
    profile: str = "novice"
    speed: float = Field(1.0, gt=0, le=20)
    seed: int | None = None


class Answer(BaseModel):
    transcript: str = Field(min_length=1, max_length=4000)
    lang: str = "ja"


class Dismiss(BaseModel):
    note: str | None = None


class Reference(BaseModel):
    session_id: str


def lan_ip() -> str:
    """The address phones on the same Wi-Fi should use. No packet is actually sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def create_app(db_path: Path | None = None, frames_dir: Path | None = None) -> FastAPI:
    store = Store(db_path or config.DB_PATH, frames_dir or config.FRAMES_DIR)
    # Recordings interrupted by a restart never received their frames; drop them.
    for stale in store.sessions(status="live"):
        store.delete_session(stale["id"])
    service = Tsukumo(store)
    hub = Hub(service)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await hub.shutdown()
        store.close()

    app = FastAPI(title="TSUKUMO", lifespan=lifespan)
    app.state.service = service
    app.state.hub = hub

    @app.exception_handler(NotFound)
    async def not_found(_: Request, exc: NotFound):
        return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)

    @app.exception_handler(PermissionError)
    async def forbidden(_: Request, exc: PermissionError):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    def guard(request: Request, source: str | None) -> None:
        """Deleting or re-pointing the master's ghost can't be undone.

        With TSUKUMO_KEY set, those need the workshop key. Without it (a public demo),
        visitors can undo their own work but not the seeded workshop or the real footage.
        """
        key = os.environ.get("TSUKUMO_KEY")
        if key:
            if not secrets.compare_digest(request.headers.get("x-tsukumo-key", "").encode(), key.encode()):
                raise PermissionError("this needs the workshop key")
        elif source and (source == "seed" or source.startswith("video:")):
            raise PermissionError("the demo workshop's own records can't be changed; run TSUKUMO locally to edit them")

    @app.exception_handler(ValueError)
    async def bad_request(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # -- reference data -------------------------------------------------------------

    @app.get("/api/info")
    def info(request: Request):
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
        return {
            "phone_base": f"{request.url.scheme}://{lan_ip()}:{port}",
            "kinds": {k: {"raw": v["raw"], "progress": v["progress"], "channel_order": list(v["channels"]),
                          "channels": {c: {f: m[f] for f in ("label", "ja", "unit")} for c, m in v["channels"].items()},
                          "tolerance": {c: m["tol"] for c, m in v["channels"].items()},
                          "weights": {c: m["weight"] for c, m in v["channels"].items()}}
                      for k, v in config.KINDS.items()},
            "langs": config.LANG_NAMES,
            "profiles": list(PROFILES),
            # Serverless hosts can't hold WebSockets: no phone, and the simulator replays over HTTP.
            "realtime": not os.environ.get("VERCEL"),
            "certify": {"score": config.CERTIFY_SCORE, "sessions": config.CERTIFY_SESSIONS, "camera_margin": config.CAMERA_NOISE},
            "ai": "offline" if config.OFFLINE or not (os.environ.get("ANTHROPIC_API_KEY")
                                                       or os.environ.get("ANTHROPIC_AUTH_TOKEN")) else "claude",
        }

    @app.get("/api/qr")
    def qr(data: str = Query(max_length=500)):
        img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
        buf = io.BytesIO()
        img.save(buf)
        return Response(buf.getvalue(), media_type="image/svg+xml")

    @app.get("/api/people")
    def people(role: str | None = None):
        return store.people(role)

    @app.get("/api/skills")
    def skills():
        out = store.skills()
        for k in out:
            ref = k["reference_session_id"] and store.session(k["reference_session_id"])
            k["reference_filmed"] = bool(ref) and by_camera(ref["source"])
        return out

    @app.post("/api/skills/{skill_id}/reference")
    def set_reference(skill_id: str, body: Reference, request: Request):
        skill = store.skill(skill_id)
        current = skill and skill["reference_session_id"] and store.session(skill["reference_session_id"])
        guard(request, current["source"] if current else None)
        service.set_reference(skill_id, body.session_id)
        return {"ok": True}

    # -- sessions ---------------------------------------------------------------------

    @app.post("/api/sessions")
    def create_session(body: NewSession):
        return service.create_session(body.skill_id, body.person_id)

    @app.get("/api/sessions")
    def list_sessions(skill_id: str | None = None, person_id: str | None = None,
                      role: str | None = None, status: str | None = None):
        return store.sessions(skill_id=skill_id, person_id=person_id, role=role, status=status)

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str):
        return service.detail(session_id)

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str, request: Request):
        s = store.session(session_id)
        guard(request, s["source"] if s else None)
        if session_id in hub.live:
            raise ValueError("session is still recording")
        service.delete_session(session_id)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/simulate")
    async def simulate(session_id: str, body: Simulate):
        if body.profile not in PROFILES:
            raise ValueError(f"unknown profile {body.profile}")
        try:
            hub.simulate(session_id, body.profile, body.speed, body.seed)
        except LookupError as exc:
            raise NotFound(str(exc))
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/replay")
    def replay(session_id: str, body: Simulate):
        if body.profile not in PROFILES:
            raise ValueError(f"unknown profile {body.profile}")
        try:
            return hub.replay(session_id, body.profile, body.seed)
        except LookupError as exc:
            raise NotFound(str(exc))

    @app.post("/api/sessions/{session_id}/end")
    async def end_session(session_id: str):
        msg = await hub.end(session_id)
        if msg is None:
            raise HTTPException(409, "session is not recording")
        return msg

    @app.post("/api/sessions/{session_id}/surprises/{surprise_id}/answer")
    def answer(session_id: str, surprise_id: str, body: Answer):
        return service.answer(session_id, surprise_id, body.transcript, body.lang)

    @app.post("/api/sessions/{session_id}/surprises/{surprise_id}/dismiss")
    def dismiss(session_id: str, surprise_id: str, body: Dismiss):
        service.dismiss(session_id, surprise_id, body.note)
        return {"ok": True}

    # -- cards and people views ---------------------------------------------------------

    @app.get("/api/cards")
    def cards(skill_id: str | None = None):
        return store.cards(skill_id=skill_id)

    @app.delete("/api/cards/{card_id}")
    def delete_card(card_id: str, request: Request):
        card = store.card(card_id)
        guard(request, card["source"] if card else None)
        service.delete_card(card_id)
        return {"ok": True}

    @app.get("/api/passport/{person_id}")
    def passport(person_id: str):
        return service.passport(person_id)

    @app.get("/api/radar")
    def radar():
        return service.radar()

    @app.get("/api/ledger")
    def ledger():
        return service.ledger()

    # -- live -------------------------------------------------------------------------

    @app.websocket("/ws/device/{session_id}")
    async def device_ws(ws: WebSocket, session_id: str):
        await ws.accept()
        try:
            ls = hub.open(session_id)
        except (LookupError, ValueError) as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()
            return
        if ls.device is not None or ls.sim_task is not None:
            await ws.send_json({"type": "error", "message": "another device is already recording"})
            await ws.close()
            return
        ls.device = ws
        person = store.person(ls.session["person_id"])
        await ws.send_json({"type": "ready", "session": ls.session, "lang": person["lang"]})
        await hub.device_status(ls, True, "phone")
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "frames":
                    await hub.ingest(session_id, msg.get("frames", []), msg.get("source"))
                elif msg.get("type") == "end":
                    await hub.end(session_id)
                    break
        except WebSocketDisconnect:
            pass
        finally:
            if ls.device is ws:
                ls.device = None
                if not ls.ending:
                    await hub.device_status(ls, False, "phone")

    @app.websocket("/ws/watch/{session_id}")
    async def watch_ws(ws: WebSocket, session_id: str):
        await ws.accept()
        try:
            ls = hub.open(session_id)
        except (LookupError, ValueError) as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close()
            return
        ls.watchers.add(ws)
        await ws.send_json({"type": "device", "connected": ls.device is not None or ls.sim_task is not None,
                            "source": "simulator" if ls.sim_task else "phone"})
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            ls.watchers.discard(ws)

    app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
    return app
