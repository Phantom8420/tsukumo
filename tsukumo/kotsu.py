"""Turn a master's spoken answer about one surprise into a multilingual kotsu card.

A kotsu card is the unit of transferred knowledge: when (trigger), what to do
(action), and why (reason), in every apprentice language, pinned to the
sensor signature and stroke count where it happened.

Claude does the structuring and translation. Without credentials (or with
TSUKUMO_OFFLINE=1) a rule-based card is built from the sensor evidence so the
system never blocks on the network during a demo.
"""

from __future__ import annotations

import json
import logging

import anthropic

from .config import CLAUDE_MODEL, LANG_NAMES, LANGS, OFFLINE

log = logging.getLogger(__name__)

FIELDS = ("title", "trigger", "action", "reason")

SYSTEM = """You help a Japanese manufacturing SME preserve a retiring master's tacit skill.

Sensors on the master's tool flagged a moment where they departed from steady \
work, and the master was asked "why did you do that?". You will get the sensor \
evidence and the master's spoken answer (often Japanese, transcribed by a \
phone, so expect transcription slips).

Write one kotsu card (コツ, the knack) for apprentices:
- title: the knack in a few words, concrete and memorable
- trigger: the cue that tells the apprentice it is time, phrased as something \
they can see, hear or feel ("when the sound turns scratchy"), not a timestamp
- action: exactly what to do, imperative, one or two short sentences
- reason: the master's why, in plain words; keep their reasoning, add nothing \
they didn't say

Stay faithful to the master. If the answer is vague, keep the card modest \
rather than inventing technique. Use the sensor evidence only to sharpen the \
trigger. Write the card in every requested language, natural for a young \
factory worker who is a native speaker, keeping trade terms (e.g. カエリ/burr) \
recognisable."""


def _schema() -> dict:
    card = {
        "type": "object",
        "properties": {f: {"type": "string"} for f in FIELDS},
        "required": list(FIELDS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {lang: card for lang in LANGS},
        "required": list(LANGS),
        "additionalProperties": False,
    }


def _prompt(skill: str, surprise: dict, transcript: str) -> str:
    evidence = "\n".join(
        f"- {'build-up before: ' if e.get('trend') else ''}{e['channel']}: "
        f"{e['before']} -> {e['during']} {e['unit']}"
        for e in surprise.get("evidence", [])
    ) or "- (no strong sensor change)"
    langs = ", ".join(f"{code} ({LANG_NAMES[code]})" for code in LANGS)
    return (
        f"Skill: {skill}\n"
        f"Moment: {surprise.get('kind')} at stroke {surprise.get('stroke_index')}, "
        f"{surprise.get('t_start')}s into the session\n"
        f"Sensor summary: {surprise.get('summary')}\n"
        f"Sensor evidence:\n{evidence}\n\n"
        f"Master's answer:\n\"\"\"{transcript}\"\"\"\n\n"
        f"Languages: {langs}"
    )


def _from_claude(skill: str, surprise: dict, transcript: str) -> dict:
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema()}},
        system=SYSTEM,
        messages=[{"role": "user", "content": _prompt(skill, surprise, transcript)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("card generation was declined")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("card generation was cut off")
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return {lang: {f: data[lang][f].strip() for f in FIELDS} for lang in LANGS}


def _trigger_from_evidence(surprise: dict) -> str:
    trends = [e for e in surprise.get("evidence", []) if e.get("trend")]
    for e in trends:
        if e["channel"] == "tone" and e["delta"] > 0:
            return "When the stone's sound climbs higher and drier"
        if e["channel"] == "loudness" and e["delta"] > 0:
            return "When the stone gets louder"
    return f"Around stroke {surprise.get('stroke_index')}"


def _offline(surprise: dict, transcript: str, lang: str) -> dict:
    """Best-effort card without a model: the master's own words, framed by the sensors."""
    text = transcript.strip()
    first = text.replace("。", ".").split(".")[0].strip() or "Master's note"
    title = first if len(first) <= 40 else first[:38] + "…"
    action = ("Stop stroking. " if surprise.get("kind") == "pause" else "") + surprise.get("summary", "")
    base = {"title": title, "trigger": _trigger_from_evidence(surprise), "action": action, "reason": text}
    cards = {code: dict(base) for code in LANGS}
    cards[lang if lang in LANGS else "ja"]["action"] = text
    return cards


def generate_card(skill: str, surprise: dict, transcript: str, lang: str = "ja") -> tuple[dict, str]:
    """Return ({lang: {title, trigger, action, reason}}, source) for a master's answer."""
    if not transcript.strip():
        raise ValueError("empty answer")
    if not OFFLINE:
        try:
            return _from_claude(skill, surprise, transcript), "claude"
        except (anthropic.APIError, RuntimeError, ValueError, KeyError, StopIteration, TypeError) as exc:
            # TypeError: the SDK raises it when no credentials can be resolved.
            log.warning("falling back to offline card: %s", exc)
    return _offline(surprise, transcript, lang), "offline"
