"""Paths and tunables shared across the backend."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = Path(os.environ.get("TSUKUMO_DATA", ROOT / "data"))
DB_PATH = DATA_DIR / "tsukumo.db"
FRAMES_DIR = DATA_DIR / "frames"
CERT_DIR = ROOT / "certs"

# Uniform analysis rate. Phones deliver 30-60 Hz; everything is resampled to this.
FS = 30.0

# Each skill is measured as one kind of work. A kind fixes the raw frame the
# phone sends and the channels everything downstream reasons about.
#   tol    - how far off counts as "half credit" when scoring, in native units
#   weight - share of the overall score
#   floor  - smallest spread trusted when z-scoring, so a perfectly steady
#            master doesn't turn sensor noise into "surprises"
# progress is what lines an apprentice up with the master: completed strokes
# for rhythmic work, elapsed seconds where the clock itself is the skill.
KINDS = {
    "sharpen": {
        # t_ms, angle_deg, ax, ay, az (linear accel, m/s^2), loud_dbfs, tone_hz
        "raw": ("t", "angle", "ax", "ay", "az", "loud", "tone"),
        "progress": "strokes",
        "channels": {
            "angle": {"label": "Blade angle", "ja": "刃の角度", "unit": "°", "tol": 3.0, "weight": 0.35, "floor": 0.6},
            "motion": {"label": "Stroke force", "ja": "力加減", "unit": "m/s²", "tol": 1.5, "weight": 0.15, "floor": 0.35},
            "tempo": {"label": "Tempo", "ja": "リズム", "unit": "spm", "tol": 20.0, "weight": 0.2, "floor": 6.0},
            "loudness": {"label": "Stone sound", "ja": "砥石の音", "unit": "dB", "tol": 6.0, "weight": 0.15, "floor": 1.5},
            "tone": {"label": "Sound pitch", "ja": "音の高さ", "unit": "Hz", "tol": 400.0, "weight": 0.15, "floor": 90.0},
        },
    },
    "heat": {
        # t_ms, hue_deg (0 red .. 60 yellow), brightness (0-100), ax, ay, az (hand)
        "raw": ("t", "hue", "bright", "ax", "ay", "az"),
        "progress": "seconds",
        "channels": {
            "hue": {"label": "Steel colour", "ja": "鋼の色", "unit": "°", "tol": 4.0, "weight": 0.35, "floor": 1.0},
            "brightness": {"label": "Glow", "ja": "明るさ", "unit": "%", "tol": 8.0, "weight": 0.25, "floor": 2.0},
            "hold": {"label": "Soak time", "ja": "保持時間", "unit": "s", "tol": 4.0, "weight": 0.25, "floor": 1.0},
            "motion": {"label": "Hand movement", "ja": "手の動き", "unit": "m/s²", "tol": 1.5, "weight": 0.15, "floor": 0.35},
        },
    },
}
DEFAULT_KIND = "sharpen"


def channels(kind: str) -> tuple[str, ...]:
    return tuple(KINDS[kind]["channels"])


def channel(kind: str, name: str) -> dict:
    return KINDS[kind]["channels"][name]


# Card languages. Apprentices see their own; masters review in Japanese.
LANGS = ("ja", "en", "vi", "id")
LANG_NAMES = {"ja": "日本語", "en": "English", "vi": "Tiếng Việt", "id": "Bahasa Indonesia"}

# A skill counts as handed over once an apprentice has this many sessions at this score.
CERTIFY_SCORE = 75.0
CERTIFY_SESSIONS = 2
# Live, the camera tracker scored the same clip 87% against itself: a camera pass must clear the mark by that much.
CAMERA_NOISE = 13.0
# A recorded clip is stepped frame by frame and scored 96.5-98.3% against itself over three runs.
FILE_NOISE = 4.0

CLAUDE_MODEL = os.environ.get("TSUKUMO_MODEL", "claude-opus-5-5")
OFFLINE = os.environ.get("TSUKUMO_OFFLINE") == "1"

# Illustrative royalty: what the master earns each time one of their cards coaches someone.
ROYALTY_YEN_PER_VIEW = 120
