# Copyright 2025-2026 Dimensional Inc.
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

"""End-to-end tests for daemon lifecycle using a lightweight PingPong blueprint.

These tests exercise real forkserver workers and module deployment — no mocks.
They use a minimal 2-module blueprint (PingModule + PongModule) that needs
no hardware, no LFS data, and no replay files.
"""

from __future__ import annotations

import json
import os
import signal
import time

import pytest

# Skip sysctl interactive prompt in CI / headless
os.environ["CI"] = "1"

from datetime import datetime, timezone

from dimos.core.blueprints import autoconnect
from dimos.core.daemon import health_check
from dimos.core.global_config import global_config
from dimos.core.module import Module
from dimos.core.run_registry import (
    REGISTRY_DIR,
    RunEntry,
    cleanup_stale,
    get_most_recent,
    list_runs,
)
from dimos.core.stream import Out

# ---------------------------------------------------------------------------
# Lightweight test modules
# ---------------------------------------------------------------------------


class PingModule(Module):
    data: Out[str]

    def start(self):
        super().start()


class PongModule(Module):
    data: Out[str]

    def start(self):
        super().start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(n_workers: int = 1):
    """Build a 2-module blueprint with *n_workers* forkserver workers."""
    global_config.update(viewer_backend="none", n_workers=n_workers)
    bp = autoconnect(PingModule.blueprint(), PongModule.blueprint())
    coord = bp.build(cli_config_overrides={"viewer_backend": "none", "n_workers": n_workers})
    return coord


def _make_entry(coord, run_id: str | None = None) -> RunEntry:
    """Create and save a registry entry for the current process."""
    if run_id is None:
        run_id = f"test-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
    entry = RunEntry(
        run_id=run_id,
        pid=os.getpid(),
        blueprint="ping-pong-test",
        started_at=datetime.now(timezone.utc).isoformat(),
        log_dir="/tmp/dimos-e2e-test",
        cli_args=["ping-pong"],
        config_overrides={"n_workers": 1},
    )
    entry.save()
    return entry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Remove any leftover registry entries before/after each test."""
    for f in REGISTRY_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
    yield
    for f in REGISTRY_DIR.glob("*.json"):
        f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDaemonE2E:
    """End-to-end daemon lifecycle with real workers."""

    def test_single_worker_lifecycle(self):
        """Build → health check → registry → status → stop (1 worker)."""
        coord = _build(n_workers=1)

        workers = coord._client.workers
        assert len(workers) == 1
        assert len(coord._deployed_modules) == 2  # PingModule + PongModule

        ok = health_check(coord, timeout=2)
        assert ok, "Health check should pass with healthy worker"

        entry = _make_entry(coord)
        runs = list_runs(alive_only=True)
        assert len(runs) == 1
        assert runs[0].run_id == entry.run_id

        latest = get_most_recent(alive_only=True)
        assert latest is not None
        assert latest.run_id == entry.run_id

        coord.stop()
        entry.remove()
        assert len(list_runs(alive_only=True)) == 0

    def test_multiple_workers(self):
        """Build with 2 workers — both should be alive after health check."""
        coord = _build(n_workers=2)

        workers = coord._client.workers
        assert len(workers) == 2
        for w in workers:
            assert w.pid is not None, f"Worker {w.worker_id} has no PID"

        ok = health_check(coord, timeout=2)
        assert ok, "Health check should pass with 2 healthy workers"

        coord.stop()

    def test_health_check_detects_dead_worker(self):
        """Kill a worker process — health check should detect and fail."""
        coord = _build(n_workers=1)

        worker = coord._client.workers[0]
        worker_pid = worker.pid
        assert worker_pid is not None

        # Kill the worker
        os.kill(worker_pid, signal.SIGKILL)
        time.sleep(0.5)  # let it die

        ok = health_check(coord, timeout=2)
        assert not ok, "Health check should FAIL after worker killed"

        coord.stop()

    def test_registry_entry_details(self):
        """Verify all fields are correctly persisted in the JSON registry."""
        coord = _build(n_workers=1)

        run_id = "detail-test-001"
        entry = RunEntry(
            run_id=run_id,
            pid=os.getpid(),
            blueprint="ping-pong-detail",
            started_at="2026-03-06T12:00:00+00:00",
            log_dir="/tmp/dimos-detail-test",
            cli_args=["--replay", "ping-pong"],
            config_overrides={"n_workers": 1, "viewer_backend": "none"},
        )
        entry.save()

        # Read raw JSON
        raw = json.loads(entry.registry_path.read_text())
        assert raw["run_id"] == run_id
        assert raw["pid"] == os.getpid()
        assert raw["blueprint"] == "ping-pong-detail"
        assert raw["started_at"] == "2026-03-06T12:00:00+00:00"
        assert raw["log_dir"] == "/tmp/dimos-detail-test"
        assert raw["cli_args"] == ["--replay", "ping-pong"]
        assert raw["config_overrides"] == {"n_workers": 1, "viewer_backend": "none"}

        # Roundtrip through list_runs
        runs = list_runs()
        assert len(runs) == 1
        loaded = runs[0]
        assert loaded.run_id == run_id
        assert loaded.blueprint == "ping-pong-detail"
        assert loaded.cli_args == ["--replay", "ping-pong"]

        coord.stop()
        entry.remove()

    def test_stale_cleanup(self):
        """Stale entries (dead PIDs) should be removed by cleanup_stale."""
        coord = _build(n_workers=1)

        # Create an entry with a bogus PID that doesn't exist
        stale = RunEntry(
            run_id="stale-dead-pid",
            pid=99999999,  # definitely not running
            blueprint="ghost",
            started_at=datetime.now(timezone.utc).isoformat(),
            log_dir="/tmp/ghost",
            cli_args=[],
            config_overrides={},
        )
        stale.save()

        # Also create a live entry
        live = _make_entry(coord)

        # alive_only=False returns ALL entries (no auto-clean)
        assert len(list_runs(alive_only=False)) == 2

        removed = cleanup_stale()
        assert removed == 1  # only the dead PID entry removed

        remaining = list_runs(alive_only=False)
        assert len(remaining) == 1
        assert remaining[0].run_id == live.run_id

        coord.stop()
        live.remove()
