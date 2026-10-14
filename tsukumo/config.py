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

# Raw frame layout sent by the phone (and the simulator):
#   t_ms, angle_deg, ax, ay, az (linear accel, m/s^2), loud_dbfs, tone_hz
RAW_FIELDS = ("t", "angle", "ax", "ay", "az", "loud", "tone")

# Derived channels, in display order.
CHANNELS = ("angle", "motion", "tempo", "loudness", "tone")

CHANNEL_META = {
    "angle": {"label": "Blade angle", "ja": "刃の角度", "unit": "°"},
    "motion": {"label": "Stroke force", "ja": "力加減", "unit": "m/s²"},
    "tempo": {"label": "Tempo", "ja": "リズム", "unit": "spm"},
    "loudness": {"label": "Stone sound", "ja": "砥石の音", "unit": "dB"},
    "tone": {"label": "Sound pitch", "ja": "音の高さ", "unit": "Hz"},
}

# How far off counts as "half credit" when scoring an apprentice, in native units.
TOLERANCE = {"angle": 3.0, "motion": 1.5, "tempo": 20.0, "loudness": 6.0, "tone": 400.0}
SCORE_WEIGHTS = {"angle": 0.35, "motion": 0.15, "tempo": 0.2, "loudness": 0.15, "tone": 0.15}

# Smallest spread we trust when z-scoring a channel, so a perfectly steady
# master doesn't turn sensor noise into "surprises".
NOISE_FLOOR = {"angle": 0.6, "motion": 0.35, "tempo": 6.0, "loudness": 1.5, "tone": 90.0}

# Card languages. Apprentices see their own; masters review in Japanese.
LANGS = ("ja", "en", "vi", "id")
LANG_NAMES = {"ja": "日本語", "en": "English", "vi": "Tiếng Việt", "id": "Bahasa Indonesia"}

# A skill counts as handed over once an apprentice has this many sessions at this score.
CERTIFY_SCORE = 75.0
CERTIFY_SESSIONS = 2

CLAUDE_MODEL = os.environ.get("TSUKUMO_MODEL", "claude-opus-5-5")
OFFLINE = os.environ.get("TSUKUMO_OFFLINE") == "1"

# Illustrative royalty: what the master earns each time one of their cards coaches someone.
ROYALTY_YEN_PER_VIEW = 120
