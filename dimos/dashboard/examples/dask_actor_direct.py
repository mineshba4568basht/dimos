# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone Dask actor example that replays YAML logs into Rerun."""

from __future__ import annotations

import dataclasses
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import webbrowser

from distributed import Client
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import yaml


# ------------------------ Minimal dashboard plumbing ----------------------- #
@dataclasses.dataclass
class RerunInfo:
    logging_id: str = os.environ.get("RERUN_ID", "dask_actor_demo")
    grpc_port: int = int(os.environ.get("RERUN_GRPC_PORT", "9876"))
    server_memory_limit: str = os.environ.get("RERUN_SERVER_MEMORY_LIMIT", "25%")
    url: str | None = None

    def __post_init__(self) -> None:
        if self.url is None:
            self.url = f"rerun+http://127.0.0.1:{self.grpc_port}/proxy"


class RerunConnection:
    """One connection per process/thread that knows how to log into Rerun."""

    def __init__(self, info: RerunInfo) -> None:
        self.info = info
        self.init_pid = os.getpid()
        self.stream = rr.RecordingStream(
            info.logging_id,
            recording_id=info.logging_id,
        )
        self.stream.connect_grpc(info.url)  # type: ignore[arg-type]

    def log(self, path: str, value: Any, **kwargs: Any) -> None:
        if self.init_pid != os.getpid():
            raise RuntimeError(
                "RerunConnection objects must be created and used in the same process."
            )
        self.stream.log(path, value, **kwargs)


class DashboardActor:
    """Tiny inline copy of the Dashboard module that can run as a Dask actor."""

    def __init__(
        self,
        info: RerunInfo,
        *,
        auto_open: bool = True,
        host: str = "127.0.0.1",
        port: int = 4000,
    ) -> None:
        self.info = info
        self.auto_open = auto_open
        self.host = host
        self.port = port
        self._started = False
        self._server: HTTPServer | None = None

    def start(self) -> str:
        if self._started:
            return self.info.url or ""

        rr.init(self.info.logging_id, spawn=False, recording_id=self.info.logging_id)
        default_blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Spatial3DView(
                    name="Spatial3D",
                    origin="/",
                    line_grid=rrb.LineGrid3D(spacing=1.0, stroke_width=1.0),
                ),
                rrb.TextDocumentView(name="Logs", origin="/logs"),
            )
        )
        rr.send_blueprint(default_blueprint)
        rr.serve_grpc(
            grpc_port=self.info.grpc_port,
            default_blueprint=default_blueprint,
            server_memory_limit=self.info.server_memory_limit,
        )
        self._server, _ = start_dashboard_server_thread(
            rerun_url=self.info.url or "",
            host=self.host,
            port=self.port,
        )
        if self.auto_open and self.info.url:
            # Best-effort browser launch; ignore failures.
            try:
                webbrowser.open(f"http://{self.host}:{self.port}")
            except Exception:
                pass

        self._started = True
        return self.info.url or ""

    def stop(self) -> bool:
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
        # There is no explicit shutdown hook for rr.serve_grpc; rely on process exit.
        self._started = False
        return True


def start_dashboard_server_thread(
    rerun_url: str,
    host: str = "127.0.0.1",
    port: int = 4000,
) -> tuple[HTTPServer, threading.Thread]:
    """Spin up a tiny HTTP server that hosts a web viewer pointing at rerun_url."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dask Actor Dashboard</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0;
      background: #0d1011;
      color: #f0f4f8;
      font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
      display: flex;
      align-items: stretch;
      min-height: 100vh;
    }}
    #viewer {{
      flex: 1;
      min-height: 100vh;
    }}
    header {{
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      padding: 0.75rem 1rem;
      background: linear-gradient(90deg, #0d1011 0%, #1b2024 100%);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 2;
    }}
    header .pill {{
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      font-size: 0.9rem;
      letter-spacing: 0.01em;
    }}
    main {{
      flex: 1;
      margin-top: 3.5rem;
    }}
  </style>
</head>
<body>
  <header>
    <div>Rerun Stream</div>
    <div class="pill">{rerun_url}</div>
  </header>
  <main id="viewer"></main>
  <script type="module">
    import {{ WebViewer }} from "https://esm.sh/@rerun-io/web-viewer@0.27.2";
    const viewer = new WebViewer();
    viewer.start("{rerun_url}", document.getElementById("viewer"));
  </script>
</body>
</html>"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            elif self.path == "/health":
                body = json.dumps({"status": "ok", "rerun_url": rerun_url}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="dashboard-server", daemon=True)
    thread.start()
    return server, thread


# ------------------------- Data replay as an actor ------------------------- #
DEFAULT_REPLAY_PATHS: dict[str, str] = {
    "lidar": "./dimos/dashboard/support/lidar.yaml",
    "color_image": "./dimos/dashboard/support/color_image.yaml",
}
# Fallback demo data if the real YAML files are missing.
DEFAULT_REPLAY_PATH = Path(__file__).with_name("sample_replay.yaml")
DEFAULT_YAML = """\
- kind: text
  path: /logs
  payload: "Hello from the Dask actor!"
- kind: points3d
  path: /points
  positions:
    - [0.0, 0.0, 0.0]
    - [1.0, 0.0, 0.0]
    - [1.0, 1.0, 0.0]
"""


def ensure_default_yaml() -> None:
    if DEFAULT_REPLAY_PATH.exists():
        return
    DEFAULT_REPLAY_PATH.write_text(DEFAULT_YAML, encoding="utf-8")


class DataReplayActor:
    """Reads YAML messages and publishes them to Rerun from a Dask worker."""

    def __init__(
        self,
        rerun_info: RerunInfo,
        *,
        replay_paths: dict[str, str] | None = None,
        interval_sec: float = 0.25,
        loop: bool = True,
    ) -> None:
        self.rerun_info = rerun_info
        self.replay_paths = replay_paths or DEFAULT_REPLAY_PATHS
        self.interval_sec = interval_sec
        self.loop = loop
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def _iter_messages(self, path: str):
        file_path = Path(path)
        if not file_path.exists():
            print(f"[DataReplayActor] file {path} does not exist, using fallback demo data")
            yield from yaml.safe_load(DEFAULT_YAML)
            return

        with file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    parsed = yaml.unsafe_load(line) or []
                except Exception as error:
                    print(f"[DataReplayActor] line {line_number} could not be parsed: {error}")
                    continue
                if isinstance(parsed, list):
                    yield from parsed
                else:
                    yield parsed

    def _to_rerun_payload(self, msg: Any, output_name: str) -> tuple[str, Any]:
        path = f"/{output_name}"
        if hasattr(msg, "to_rerun"):
            payload = msg.to_rerun()  # type: ignore[call-arg]
        elif isinstance(msg, dict):
            path = msg.get("path", path)
            kind = msg.get("kind", "text")
            if kind == "points3d":
                positions = msg.get("positions") or msg.get("points") or []
                payload = rr.Points3D(positions=positions)
            else:
                payload = rr.TextLog(str(msg.get("payload", msg)))
        else:
            payload = rr.TextLog(str(msg))
        return path, payload

    def _publish_stream(self, output_name: str, path: str) -> None:
        rc = RerunConnection(self.rerun_info)
        while not self._stop_event.is_set():
            any_sent = False
            for _i, msg in enumerate(self._iter_messages(path)):
                if self._stop_event.is_set():
                    break
                try:
                    log_path, payload = self._to_rerun_payload(msg, output_name)
                    rc.log(log_path, payload, strict=True)
                    any_sent = True
                    time.sleep(self.interval_sec)
                except Exception as error:
                    print(f"[DataReplayActor] error while publishing {output_name}: {error}")
            if not self.loop or not any_sent:
                break

    def start(self) -> bool:
        if self._threads:
            return True
        for output_name, path in self.replay_paths.items():
            thread = threading.Thread(
                target=self._publish_stream,
                args=(output_name, path),
                name=f"{output_name}-replay",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
            time.sleep(0.1)
        return True

    def stop(self) -> bool:
        self._stop_event.set()
        for thread in self._threads:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        return True


# ------------------------------ Entrypoint --------------------------------- #
def main() -> None:
    ensure_default_yaml()
    rerun_info = RerunInfo()

    client = Client(
        n_workers=1,
        threads_per_worker=4,
    )
    dashboard = client.submit(
        DashboardActor,
        rerun_info,
        auto_open=True,
        host="127.0.0.1",
        port=4000,
        actor=True,
    ).result()
    dashboard.start().result()

    replayer = client.submit(
        DataReplayActor,
        rerun_info,
        replay_paths=DEFAULT_REPLAY_PATHS,
        interval_sec=0.25,
        loop=True,
        actor=True,
    ).result()
    replayer.start().result()

    print(f"Dashboard running at {rerun_info.url} (Rerun gRPC on port {rerun_info.grpc_port})")
    print("Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        replayer.stop().result()
        dashboard.stop().result()
        client.close()


if __name__ == "__main__":
    main()
