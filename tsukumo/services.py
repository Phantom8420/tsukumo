"""Everything the API does, independent of HTTP: sessions, review, cards, passports, radar."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np

from . import align, kotsu, surprise
from .coach import LiveCoach
from .config import CERTIFY_SCORE, CERTIFY_SESSIONS, FS, ROYALTY_YEN_PER_VIEW
from .signals import Features, extract
from .store import Store

MIN_SESSION_S = 5.0
FEATURE_STEP = 3   # samples per point sent to the browser (FS / 3 = 10 Hz)


class NotFound(LookupError):
    pass


class Tsukumo:
    def __init__(self, store: Store):
        self.store = store
        self._features: dict[str, Features] = {}

    # -- sessions ---------------------------------------------------------------------

    def _session(self, session_id: str) -> dict:
        s = self.store.session(session_id)
        if not s:
            raise NotFound(f"session {session_id}")
        return s

    def features(self, session_id: str) -> Features:
        if session_id not in self._features:
            kind = self._session(session_id)["kind"]
            self._features[session_id] = extract(self.store.load_frames(session_id), kind)
        return self._features[session_id]

    def create_session(self, skill_id: str, person_id: str, source: str = "phone",
                       created_at: float | None = None) -> dict:
        person = self.store.person(person_id)
        if not person:
            raise NotFound(f"person {person_id}")
        if not self.store.skill(skill_id):
            raise NotFound(f"skill {skill_id}")
        return self.store.create_session(skill_id, person_id, person["role"], source, created_at)

    def finalize(self, session_id: str, frames: np.ndarray) -> dict:
        """Persist a finished recording and run the role-appropriate analysis."""
        s = self._session(session_id)
        frames = np.asarray(frames, dtype=float)
        if len(frames) < 2 or (frames[-1, 0] - frames[0, 0]) / 1000 < MIN_SESSION_S:
            self.store.delete_session(session_id)
            raise ValueError(f"recording shorter than {MIN_SESSION_S:.0f}s was discarded")

        self.store.save_frames(session_id, frames)
        self._features.pop(session_id, None)
        feat = self.features(session_id)
        summary = {"session_id": session_id, "role": s["role"], "duration": round(feat.duration, 1),
                   "strokes": len(feat.stroke_times)}

        if s["role"] == "master":
            events, _ = surprise.detect(feat)
            self.store.save_surprises(session_id, [e.to_json() for e in events])
            self.store.finish_session(session_id, feat.duration, len(feat.stroke_times))
            skill = self.store.skill(s["skill_id"])
            if not skill["reference_session_id"]:
                self.store.set_reference(s["skill_id"], session_id)
            summary["surprises"] = len(events)
        else:
            report = self._score(s["skill_id"], feat)
            self.store.finish_session(session_id, feat.duration, len(feat.stroke_times),
                                      report["overall"] if report else None, report)
            summary["report"] = report
        return summary

    def _reference(self, skill_id: str) -> Features | None:
        skill = self.store.skill(skill_id)
        ref = skill and skill["reference_session_id"]
        return self.features(ref) if ref else None

    def _score(self, skill_id: str, feat: Features) -> dict | None:
        ref = self._reference(skill_id)
        return align.score(ref, feat).to_json() if ref is not None else None

    def detail(self, session_id: str) -> dict:
        s = self._session(session_id)
        skill = self.store.skill(s["skill_id"])
        out = {"session": s, "is_reference": skill["reference_session_id"] == session_id}
        if s["status"] == "done":
            feat = self.features(session_id)
            out["features"] = feat.to_json(FS / FEATURE_STEP)
            if s["role"] == "master":
                _, score = surprise.detect(feat)
                out["surprise_score"] = np.round(score[::FEATURE_STEP], 2).tolist()
                out["surprises"] = self.store.surprises(session_id)
                out["cards"] = self.store.cards(session_id=session_id)
        return out

    def delete_session(self, session_id: str) -> None:
        s = self._session(session_id)
        skill = self.store.skill(s["skill_id"])
        if skill["reference_session_id"] == session_id:
            self.store.set_reference(s["skill_id"], None)
        self.store.delete_session(session_id)
        self._features.pop(session_id, None)

    def set_reference(self, skill_id: str, session_id: str) -> None:
        s = self._session(session_id)
        if s["role"] != "master" or s["skill_id"] != skill_id:
            raise ValueError("the reference must be a master session of the same skill")
        self.store.set_reference(skill_id, session_id)

    # -- review -----------------------------------------------------------------------

    def answer(self, session_id: str, surprise_id: str, transcript: str, lang: str = "ja") -> dict:
        s = self._session(session_id)
        ev = self.store.surprise(session_id, surprise_id)
        if not ev:
            raise NotFound(f"surprise {surprise_id}")
        skill = self.store.skill(s["skill_id"])
        content, source = kotsu.generate_card(skill["name"], ev, transcript, lang)
        card = self.store.add_card(
            skill_id=s["skill_id"], master_id=s["person_id"], stroke_index=ev["stroke_index"],
            content=content, source=source, session_id=session_id, surprise_id=surprise_id,
            transcript=transcript, transcript_lang=lang,
            evidence={"kind": ev["kind"], "summary": ev["summary"], "t_start": ev["t_start"]},
        )
        self.store.set_surprise_status(session_id, surprise_id, "answered", card["id"])
        return card

    def dismiss(self, session_id: str, surprise_id: str, note: str | None = None) -> None:
        self._session(session_id)
        if not self.store.surprise(session_id, surprise_id):
            raise NotFound(f"surprise {surprise_id}")
        self.store.set_surprise_status(session_id, surprise_id, "dismissed", note)

    def delete_card(self, card_id: str) -> None:
        card = self.store.card(card_id)
        if not card:
            raise NotFound(f"card {card_id}")
        if card["session_id"] and card["surprise_id"]:
            self.store.set_surprise_status(card["session_id"], card["surprise_id"], "open")
        self.store.delete_card(card_id)

    # -- live -------------------------------------------------------------------------

    def coach_for(self, session_id: str) -> LiveCoach:
        """The apprentice's coach; masters get one with no ghost, which only tracks."""
        s = self._session(session_id)
        if s["role"] != "apprentice":
            return LiveCoach(None, [], "ja", s["kind"])
        person = self.store.person(s["person_id"])
        return LiveCoach(self._reference(s["skill_id"]), self.store.cards(skill_id=s["skill_id"]),
                         person["lang"], s["kind"])

    # -- people-level views -----------------------------------------------------------

    def passport(self, person_id: str) -> dict:
        person = self.store.person(person_id)
        if not person:
            raise NotFound(f"person {person_id}")
        by_skill = defaultdict(list)
        for s in reversed(self.store.sessions(person_id=person_id, status="done")):
            if s["score"] is not None:
                by_skill[s["skill_id"]].append(s)
        skills = []
        for skill_id, sessions in by_skill.items():
            scores = [s["score"] for s in sessions]
            passed = sum(sc >= CERTIFY_SCORE for sc in scores)
            skills.append({
                "skill_id": skill_id,
                "skill_name": sessions[0]["skill_name"],
                "sessions": len(sessions),
                "best": round(max(scores), 1),
                "latest": round(scores[-1], 1),
                "history": [{"session_id": s["id"], "score": round(s["score"], 1), "at": s["created_at"]}
                            for s in sessions],
                "certified": passed >= CERTIFY_SESSIONS,
                "to_certify": max(0, CERTIFY_SESSIONS - passed),
            })
        return {"person": person, "skills": skills, "cards_learned": self.store.views_by(person_id),
                "certify_rule": {"score": CERTIFY_SCORE, "sessions": CERTIFY_SESSIONS}}

    def _certified(self) -> dict[str, set[str]]:
        """skill_id -> apprentice ids who have measurably taken the skill over."""
        passes: dict[tuple[str, str], int] = defaultdict(int)
        for s in self.store.sessions(role="apprentice", status="done"):
            if s["score"] is not None and s["score"] >= CERTIFY_SCORE:
                passes[(s["skill_id"], s["person_id"])] += 1
        certified: dict[str, set[str]] = defaultdict(set)
        for (skill_id, person_id), n in passes.items():
            if n >= CERTIFY_SESSIONS:
                certified[skill_id].add(person_id)
        return certified

    def radar(self) -> list[dict]:
        """Which skills die with whose retirement, and how much of them is captured."""
        year = dt.date.today().year
        people = {p["id"]: p for p in self.store.people()}
        prof = defaultdict(dict)
        for row in self.store.proficiency():
            prof[row["skill_id"]][row["person_id"]] = row["level"]
        certified = self._certified()
        cards = defaultdict(int)
        for row in self.store.card_counts():
            cards[row["skill_id"]] += row["cards"]

        rows = []
        for skill in self.store.skills():
            sid = skill["id"]
            holders = {pid for pid, lvl in prof[sid].items() if lvl >= 0.75}
            ref = skill["reference_session_id"] and self.store.session(skill["reference_session_id"])
            if ref:
                holders.add(ref["person_id"])
            holders |= certified[sid]
            masters = [people[p] for p in holders if people[p]["role"] == "master"]
            successors = [people[p] for p in holders if people[p]["role"] == "apprentice"]
            retiring = [m["retire_year"] for m in masters if m["retire_year"]]
            years_left = (min(retiring) - year) if retiring else None

            base = 100 if not successors else 55 if len(successors) == 1 else 25
            urgency = 1.0 if years_left is None else float(np.clip((5 - years_left) / 5, 0, 1))
            coverage = min(cards[sid] / 4, 1.0)
            risk = base * (0.55 + 0.45 * urgency) * (1 - 0.35 * coverage)
            rows.append({
                "skill": skill,
                "masters": [{"id": m["id"], "name": m["name"], "retire_year": m["retire_year"]} for m in masters],
                "successors": [{"id": a["id"], "name": a["name"],
                                "measured": a["id"] in certified[sid]} for a in successors],
                "bus_factor": len(holders),
                "years_left": years_left,
                "cards": cards[sid],
                "risk": round(risk),
                "status": "critical" if risk >= 70 else "at_risk" if risk >= 40 else "covered",
            })
        return sorted(rows, key=lambda r: -r["risk"])

    def ledger(self) -> list[dict]:
        stats = {r["master_id"]: r for r in self.store.card_view_stats()}
        cards = defaultdict(int)
        for row in self.store.card_counts():
            cards[row["master_id"]] += row["cards"]
        out = []
        for m in self.store.people(role="master"):
            st = stats.get(m["id"], {"views": 0, "apprentices": 0})
            out.append({"master": m, "cards": cards[m["id"]], "views": st["views"],
                        "apprentices": st["apprentices"], "credits_yen": st["views"] * ROYALTY_YEN_PER_VIEW})
        return sorted(out, key=lambda r: -r["views"])
