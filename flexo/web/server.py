"""FastAPI + websocket overlay server.

Runs uvicorn in a background thread. The orchestrator calls `update(payload)`
from its own thread; connected browsers (OBS browser source) poll a version
counter over the websocket and re-render whenever it bumps. No cross-thread event
loop juggling — the shared state is a plain dict guarded by the GIL.
"""

import json
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

_OVERLAY_HTML = (Path(__file__).parent / "overlay.html").read_text(encoding="utf-8")


class OverlayServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 7337):
        self.host = host
        self.port = port
        self._state: dict = _empty_state()
        self._version = 0
        self._app = self._build_app()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        config = uvicorn.Config(self._app, host=self.host, port=self.port,
                                log_level="warning")
        server = uvicorn.Server(config)
        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()
        print(f"[web] overlay at http://{self.host}:{self.port}  "
              f"(add as OBS Browser Source)", flush=True)

    def update(self, **fields) -> None:
        """Merge fields into the live overlay state and bump the version."""
        self._state.update(fields)
        self._version += 1

    # ------------------------------------------------------------------ app
    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/")
        async def index():
            return HTMLResponse(_OVERLAY_HTML)

        @app.websocket("/ws")
        async def ws(sock: WebSocket):
            import asyncio
            await sock.accept()
            last = -1
            try:
                while True:
                    if self._version != last:
                        last = self._version
                        await sock.send_text(json.dumps(self._state))
                    await asyncio.sleep(0.1)
            except WebSocketDisconnect:
                pass
            except Exception:
                pass

        return app


def _empty_state() -> dict:
    return {
        "set": "",
        "pack": 0,
        "pick": 0,
        "confidence": "",
        "spoken": "Flexo is booting up...",
        "plan": "undeclared (open)",
        "picks": [],                 # [{name, gih_wr}]
        "human": 0,
        "ai": 0,
    }
