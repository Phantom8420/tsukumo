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

_store = Store(config.DB_PATH, config.FRAMES_DIR)
seed(Tsukumo(_store))
_store.close()

app = create_app()


@app.middleware("http")
async def _debug_path(request, call_next):
    # temporary: find out what path Vercel hands the function
    resp = await call_next(request)
    if resp.status_code == 404:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not Found", "path": request.url.path, "root": request.scope.get("root_path"),
                             "raw": request.scope.get("raw_path", b"").decode()}, status_code=404)
    return resp
