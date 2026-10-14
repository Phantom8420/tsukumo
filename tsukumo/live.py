"""In-memory hub for sessions that are being recorded right now.

One device (the phone on the tool, or the simulator) feeds frames in; any
number of watchers (the big screen) get ticks, buzzes and cards out. When
the device says it's done, the recording is handed to the service layer for
analysis and the summary is broadcast.
"""

from __future__ import annotations

import asyncio
import logging
import math

import numpy as np
from fastapi import WebSocket
from starlette.concurrency import run_in_threadpool

from .coach import LiveCoach
from .config import KINDS
from .services import Tsukumo
from .simulate import generate

log = logging.getLogger(__name__)


def _simulated(session: dict, profile: str, seed: int | None) -> np.ndarray:
    if session["kind"] != "sharpen":
        raise ValueError(f"no simulator for {session['kind']} work yet")
    frames, _ = generate(profile, seed=seed if seed is not None else int(np.random.default_rng().integers(1e6)))
    return frames


class LiveSession:
    def __init__(self, session: dict, coach: LiveCoach):
        self.session = session
        self.coach = coach
        self.frames: list[tuple[float, ...]] = []
        self.watchers: set[WebSocket] = set()
        self.device: WebSocket | None = None
        self.ending = False
        self.sim_task: asyncio.Task | None = None


class Hub:
    def __init__(self, service: Tsukumo):
        self.service = service
        self.live: dict[str, LiveSession] = {}

    def open(self, session_id: str) -> LiveSession:
        if session_id in self.live:
            return self.live[session_id]
        s = self.service.store.session(session_id)
        if not s:
            raise LookupError(f"session {session_id}")
        if s["status"] != "live":
            raise ValueError("session already finished")
        coach = self.service.coach_for(session_id)
        ls = self.live[session_id] = LiveSession(s, coach)
        return ls

    async def send(self, ws: WebSocket, msg: dict) -> bool:
        try:
            await ws.send_json(msg)
            return True
        except Exception:   # a closed socket can fail in several ways; dropping it is the answer to all
            return False

    async def broadcast(self, ls: LiveSession, msg: dict) -> None:
        dead = [ws for ws in list(ls.watchers) if not await self.send(ws, msg)]
        ls.watchers.difference_update(dead)

    async def device_status(self, ls: LiveSession, connected: bool, source: str) -> None:
        await self.broadcast(ls, {"type": "device", "connected": connected, "source": source})

    async def ingest(self, session_id: str, frames: list) -> None:
        ls = self.live.get(session_id)
        if ls is None or ls.ending:
            return
        out: list[dict] = []
        width = len(KINDS[ls.session["kind"]]["raw"])
        for f in frames:
            if len(f) != width:
                continue
            try:
                frame = tuple(float(v) for v in f)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in frame):
                continue
            if ls.frames and frame[0] <= ls.frames[-1][0]:
                continue
            ls.frames.append(frame)
            out += ls.coach.update(frame)

        ticks = [e for e in out if e["type"] == "tick"]
        if ticks:
            await self.broadcast(ls, ticks[-1])
        for e in out:
            if e["type"] == "tick":
                continue
            if e["type"] == "card" and ls.session["role"] == "apprentice":
                self.service.store.log_card_view(e["card"]["id"], ls.session["person_id"], session_id)
            await self.broadcast(ls, e)
            if ls.device:
                await self.send(ls.device, e)

    async def end(self, session_id: str) -> dict | None:
        ls = self.live.get(session_id)
        if ls is None or ls.ending:
            return None
        ls.ending = True
        await self.broadcast(ls, {"type": "analysing"})
        try:
            summary = await run_in_threadpool(self.service.finalize, session_id, np.asarray(ls.frames))
            msg = {"type": "summary", **summary}
        except ValueError as exc:
            msg = {"type": "error", "message": str(exc)}
        except Exception:
            log.exception("finalising %s failed", session_id)
            msg = {"type": "error", "message": "analysis failed"}
        await self.broadcast(ls, msg)
        if ls.device:
            await self.send(ls.device, msg)
        self.live.pop(session_id, None)
        return msg

    def simulate(self, session_id: str, profile: str, speed: float = 1.0, seed: int | None = None) -> None:
        ls = self.open(session_id)
        if ls.device or ls.sim_task:
            raise ValueError("a device is already streaming into this session")
        frames = _simulated(ls.session, profile, seed)
        ls.sim_task = asyncio.create_task(self._play(session_id, frames, speed))

    def replay(self, session_id: str, profile: str, seed: int | None = None) -> dict:
        """Run a simulated session start to finish in one call; the client plays the events back.

        For hosts without WebSockets (serverless), where the streaming path can't work.
        """
        s = self.service.store.session(session_id)
        if not s:
            raise LookupError(f"session {session_id}")
        if s["status"] != "live" or session_id in self.live:
            raise ValueError("session is already recording or finished")
        coach = self.service.coach_for(session_id)
        frames = _simulated(s, profile, seed)
        events: list[dict] = []
        for frame in frames.tolist():
            for e in coach.update(tuple(frame)):
                if e["type"] == "card" and s["role"] == "apprentice":
                    self.service.store.log_card_view(e["card"]["id"], s["person_id"], session_id)
                events.append(e)
        return {"events": events, "summary": {"type": "summary", **self.service.finalize(session_id, frames)}}

    async def _play(self, session_id: str, frames: np.ndarray, speed: float) -> None:
        ls = self.live[session_id]
        await self.device_status(ls, True, "simulator")
        batch = 5
        step = (frames[batch, 0] - frames[0, 0]) / 1000 / max(speed, 0.1)
        try:
            for i in range(0, len(frames), batch):
                if ls.ending:
                    return
                await self.ingest(session_id, frames[i:i + batch].tolist())
                await asyncio.sleep(step)
            await self.end(session_id)
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        for ls in list(self.live.values()):
            if ls.sim_task:
                ls.sim_task.cancel()
