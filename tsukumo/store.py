"""SQLite for metadata, one .npy file per session for raw frames."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_native TEXT,
    role TEXT NOT NULL CHECK (role IN ('master', 'apprentice')),
    lang TEXT NOT NULL DEFAULT 'ja',
    retire_year INTEGER,
    origin TEXT,
    bio TEXT
);
CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_ja TEXT,
    description TEXT,
    reference_session_id TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL REFERENCES skills(id),
    person_id TEXT NOT NULL REFERENCES people(id),
    role TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'live',
    source TEXT NOT NULL DEFAULT 'phone',
    created_at REAL NOT NULL,
    duration REAL,
    strokes INTEGER,
    score REAL,
    report TEXT
);
CREATE TABLE IF NOT EXISTS surprises (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    note TEXT,
    data TEXT NOT NULL,
    PRIMARY KEY (session_id, id)
);
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL REFERENCES skills(id),
    master_id TEXT NOT NULL REFERENCES people(id),
    session_id TEXT,
    surprise_id TEXT,
    stroke_index INTEGER NOT NULL,
    transcript TEXT,
    transcript_lang TEXT,
    content TEXT NOT NULL,
    evidence TEXT,
    source TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS card_views (
    card_id TEXT NOT NULL REFERENCES cards(id),
    apprentice_id TEXT NOT NULL REFERENCES people(id),
    session_id TEXT,
    at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS proficiency (
    person_id TEXT NOT NULL REFERENCES people(id),
    skill_id TEXT NOT NULL REFERENCES skills(id),
    level REAL NOT NULL,
    PRIMARY KEY (person_id, skill_id)
);
"""

JSON_COLUMNS = {"report", "data", "content", "evidence"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Store:
    def __init__(self, db_path: Path, frames_dir: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = frames_dir
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)

    # -- plumbing -----------------------------------------------------------------

    def _rows(self, sql: str, args=()) -> list[dict]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in JSON_COLUMNS & d.keys():
                if d[k] is not None:
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    def _row(self, sql: str, args=()) -> dict | None:
        rows = self._rows(sql, args)
        return rows[0] if rows else None

    def _exec(self, sql: str, args=()) -> None:
        with self._lock:
            self._db.execute(sql, args)
            self._db.commit()

    def close(self) -> None:
        self._db.close()

    # -- people & skills --------------------------------------------------------------

    def upsert_person(self, **p) -> None:
        self._exec(
            "INSERT OR REPLACE INTO people (id, name, name_native, role, lang, retire_year, origin, bio) "
            "VALUES (:id, :name, :name_native, :role, :lang, :retire_year, :origin, :bio)",
            {"name_native": None, "retire_year": None, "origin": None, "bio": None, "lang": "ja", **p},
        )

    def upsert_skill(self, **s) -> None:
        self._exec(
            "INSERT OR REPLACE INTO skills (id, name, name_ja, description, reference_session_id) "
            "VALUES (:id, :name, :name_ja, :description, :reference_session_id)",
            {"name_ja": None, "description": None, "reference_session_id": None, **s},
        )

    def people(self, role: str | None = None) -> list[dict]:
        if role:
            return self._rows("SELECT * FROM people WHERE role = ? ORDER BY name", (role,))
        return self._rows("SELECT * FROM people ORDER BY role DESC, name")

    def person(self, person_id: str) -> dict | None:
        return self._row("SELECT * FROM people WHERE id = ?", (person_id,))

    def skills(self) -> list[dict]:
        return self._rows("SELECT * FROM skills ORDER BY name")

    def skill(self, skill_id: str) -> dict | None:
        return self._row("SELECT * FROM skills WHERE id = ?", (skill_id,))

    def set_reference(self, skill_id: str, session_id: str) -> None:
        self._exec("UPDATE skills SET reference_session_id = ? WHERE id = ?", (session_id, skill_id))

    def set_proficiency(self, person_id: str, skill_id: str, level: float) -> None:
        self._exec("INSERT OR REPLACE INTO proficiency VALUES (?, ?, ?)", (person_id, skill_id, level))

    def proficiency(self) -> list[dict]:
        return self._rows("SELECT * FROM proficiency")

    # -- sessions ---------------------------------------------------------------------

    def create_session(self, skill_id: str, person_id: str, role: str, source: str = "phone",
                       created_at: float | None = None) -> dict:
        sid = new_id("ses")
        self._exec(
            "INSERT INTO sessions (id, skill_id, person_id, role, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, skill_id, person_id, role, source, created_at or time.time()),
        )
        return self.session(sid)

    def session(self, session_id: str) -> dict | None:
        return self._row(
            "SELECT s.*, p.name AS person_name, k.name AS skill_name FROM sessions s "
            "JOIN people p ON p.id = s.person_id JOIN skills k ON k.id = s.skill_id WHERE s.id = ?",
            (session_id,),
        )

    def sessions(self, skill_id: str | None = None, person_id: str | None = None,
                 role: str | None = None, status: str | None = None) -> list[dict]:
        sql = ("SELECT s.id, s.skill_id, s.person_id, s.role, s.status, s.source, s.created_at, "
               "s.duration, s.strokes, s.score, p.name AS person_name, k.name AS skill_name "
               "FROM sessions s JOIN people p ON p.id = s.person_id JOIN skills k ON k.id = s.skill_id")
        where, args = [], []
        for col, val in (("s.skill_id", skill_id), ("s.person_id", person_id), ("s.role", role), ("s.status", status)):
            if val:
                where.append(f"{col} = ?")
                args.append(val)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self._rows(sql + " ORDER BY s.created_at DESC", args)

    def finish_session(self, session_id: str, duration: float, strokes: int,
                       score: float | None = None, report: dict | None = None) -> None:
        self._exec(
            "UPDATE sessions SET status = 'done', duration = ?, strokes = ?, score = ?, report = ? WHERE id = ?",
            (duration, strokes, score, json.dumps(report) if report is not None else None, session_id),
        )

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM card_views WHERE session_id = ?", (session_id,))
            self._db.execute("DELETE FROM surprises WHERE session_id = ?", (session_id,))
            self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._db.commit()
        self.frames_path(session_id).unlink(missing_ok=True)

    def frames_path(self, session_id: str) -> Path:
        return self.frames_dir / f"{session_id}.npy"

    def save_frames(self, session_id: str, frames: np.ndarray) -> None:
        np.save(self.frames_path(session_id), np.asarray(frames, dtype=np.float32))

    def load_frames(self, session_id: str) -> np.ndarray:
        return np.load(self.frames_path(session_id)).astype(float)

    # -- surprises --------------------------------------------------------------------

    def save_surprises(self, session_id: str, surprises: list[dict]) -> None:
        with self._lock:
            self._db.execute("DELETE FROM surprises WHERE session_id = ?", (session_id,))
            self._db.executemany(
                "INSERT INTO surprises (session_id, id, data) VALUES (?, ?, ?)",
                [(session_id, s["id"], json.dumps(s)) for s in surprises],
            )
            self._db.commit()

    def surprises(self, session_id: str) -> list[dict]:
        rows = self._rows("SELECT * FROM surprises WHERE session_id = ?", (session_id,))
        out = [{**r["data"], "status": r["status"], "note": r["note"]} for r in rows]
        return sorted(out, key=lambda s: s["t_start"])

    def surprise(self, session_id: str, surprise_id: str) -> dict | None:
        r = self._row("SELECT * FROM surprises WHERE session_id = ? AND id = ?", (session_id, surprise_id))
        return {**r["data"], "status": r["status"], "note": r["note"]} if r else None

    def set_surprise_status(self, session_id: str, surprise_id: str, status: str, note: str | None = None) -> None:
        self._exec("UPDATE surprises SET status = ?, note = ? WHERE session_id = ? AND id = ?",
                   (status, note, session_id, surprise_id))

    # -- cards ------------------------------------------------------------------------

    def add_card(self, *, skill_id: str, master_id: str, stroke_index: int, content: dict, source: str,
                 session_id: str | None = None, surprise_id: str | None = None, transcript: str | None = None,
                 transcript_lang: str | None = None, evidence: dict | None = None) -> dict:
        cid = new_id("card")
        self._exec(
            "INSERT INTO cards (id, skill_id, master_id, session_id, surprise_id, stroke_index, transcript, "
            "transcript_lang, content, evidence, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, skill_id, master_id, session_id, surprise_id, stroke_index, transcript, transcript_lang,
             json.dumps(content, ensure_ascii=False), json.dumps(evidence, ensure_ascii=False) if evidence else None,
             source, time.time()),
        )
        return self.card(cid)

    def card(self, card_id: str) -> dict | None:
        return self._row("SELECT * FROM cards WHERE id = ?", (card_id,))

    def cards(self, skill_id: str | None = None, session_id: str | None = None) -> list[dict]:
        if session_id:
            return self._rows("SELECT * FROM cards WHERE session_id = ? ORDER BY stroke_index", (session_id,))
        if skill_id:
            return self._rows("SELECT * FROM cards WHERE skill_id = ? ORDER BY stroke_index", (skill_id,))
        return self._rows("SELECT * FROM cards ORDER BY created_at DESC")

    def delete_card(self, card_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM card_views WHERE card_id = ?", (card_id,))
            self._db.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            self._db.commit()

    def log_card_view(self, card_id: str, apprentice_id: str, session_id: str | None,
                      at: float | None = None) -> None:
        self._exec("INSERT INTO card_views VALUES (?, ?, ?, ?)", (card_id, apprentice_id, session_id, at or time.time()))

    def views_by(self, apprentice_id: str) -> int:
        return self._row("SELECT COUNT(*) AS n FROM card_views WHERE apprentice_id = ?", (apprentice_id,))["n"]

    def card_view_stats(self) -> list[dict]:
        return self._rows(
            "SELECT c.master_id, COUNT(v.card_id) AS views, COUNT(DISTINCT v.apprentice_id) AS apprentices "
            "FROM cards c LEFT JOIN card_views v ON v.card_id = c.id GROUP BY c.master_id"
        )

    def card_counts(self) -> list[dict]:
        return self._rows("SELECT skill_id, master_id, COUNT(*) AS cards FROM cards GROUP BY skill_id, master_id")
