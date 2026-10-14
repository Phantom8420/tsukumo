"""Vercel entry point. The filesystem is read-only apart from /tmp, so the demo
workshop is seeded there on each cold start."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TSUKUMO_DATA", "/tmp/tsukumo")

from tsukumo import config  # noqa: E402
from tsukumo.app import create_app  # noqa: E402
from tsukumo.seed import seed  # noqa: E402
from tsukumo.services import Tsukumo  # noqa: E402
from tsukumo.store import Store  # noqa: E402
from urllib.parse import parse_qsl, urlencode  # noqa: E402

_store = Store(config.DB_PATH, config.FRAMES_DIR)
seed(Tsukumo(_store))
_store.close()

_app = create_app()


async def app(scope, receive, send):
    """Vercel hands the function the rewrite target (/api/index) as its path; vercel.json
    passes the real one along as ?__path=, which is put back before FastAPI routes it."""
    if scope["type"] == "http":
        query = parse_qsl(scope["query_string"].decode(), keep_blank_values=True)
        path = next((v for k, v in query if k == "__path"), None)
        if path is not None:
            path = "/" + path.lstrip("/")
            rest = urlencode([(k, v) for k, v in query if k != "__path"])
            scope = {**scope, "path": path, "raw_path": path.encode(), "query_string": rest.encode()}
    await _app(scope, receive, send)

