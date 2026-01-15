# Copyright 2026 Dimensional Inc.
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

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

from dimos.constants import DIMOS_LOG_DIR
from dimos.protocol.pubsub.lcmpubsub import LCMPubSubBase
from dimos.utils.cli.lcmspy.lcmspy import GraphLCMSpy
from dimos.utils.logging_config import setup_logger
from dimos.utils.monitoring import get_worker_pids

if TYPE_CHECKING:
    from dimos.core.global_config import GlobalConfig

try:
    from dimos_lcm.std_msgs import Float32  # type: ignore[import-untyped]
except Exception:
    Float32 = None  # type: ignore[assignment]

logger = setup_logger()


@dataclass(frozen=True)
class _Clock:
    t0_wall: float
    t0_mono: float

    def now(self) -> tuple[float, float, float]:
        ts_wall = time.time()
        ts_mono = time.monotonic()
        return ts_wall, ts_mono, ts_mono - self.t0_mono


class _CpuSampler:
    """Compute total CPU utilization from /proc/stat deltas."""

    _prev_total: int | None = None
    _prev_idle: int | None = None

    def sample_percent(self) -> float:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            if not line.startswith("cpu "):
                return -1.0
            parts = line.split()
            values = [int(x) for x in parts[1:]]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
            if self._prev_total is None or self._prev_idle is None:
                self._prev_total, self._prev_idle = total, idle
                return -1.0
            dt = total - self._prev_total
            didle = idle - self._prev_idle
            self._prev_total, self._prev_idle = total, idle
            if dt <= 0:
                return -1.0
            return max(0.0, min(100.0, (1.0 - (didle / dt)) * 100.0))
        except Exception:
            return -1.0


def _read_meminfo_kb() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                # e.g. "MemAvailable:   123456 kB"
                m = re.match(r"^(\w+):\s+(\d+)\s+kB$", line.strip())
                if m:
                    out[m.group(1)] = int(m.group(2))
    except Exception:
        pass
    return out


def _read_loadavg() -> tuple[float, float, float]:
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().strip().split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return -1.0, -1.0, -1.0


def _read_net_dev() -> dict[str, dict[str, int]]:
    """Parse /proc/net/dev into per-interface counters."""
    stats: dict[str, dict[str, int]] = {}
    try:
        with open("/proc/net/dev") as f:
            lines = f.readlines()
        for line in lines[2:]:
            if ":" not in line:
                continue
            iface, rest = line.split(":", 1)
            iface = iface.strip()
            fields = rest.split()
            # rx: bytes packets errs drop fifo frame compressed multicast
            # tx: bytes packets errs drop fifo colls carrier compressed
            if len(fields) < 16:
                continue
            stats[iface] = {
                "rx_bytes": int(fields[0]),
                "rx_packets": int(fields[1]),
                "rx_errs": int(fields[2]),
                "rx_drop": int(fields[3]),
                "tx_bytes": int(fields[8]),
                "tx_packets": int(fields[9]),
                "tx_errs": int(fields[10]),
                "tx_drop": int(fields[11]),
            }
    except Exception:
        pass
    return stats


def _read_proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").strip()
        return raw.decode(errors="replace")
    except Exception:
        return ""


def _find_pids_with_cmd_substr(substr: str) -> list[int]:
    pids: list[int] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        cmd = _read_proc_cmdline(pid)
        if substr in cmd:
            pids.append(pid)
    return pids


def _ps_sample_pid(pid: int) -> tuple[float, int] | None:
    """Return (%cpu, rss_kb) via ps; easiest cross-kernel method."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu=", "-o", "rss="],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = result.stdout.strip().split()
        if len(parts) < 2:
            return None
        return float(parts[0]), int(parts[1])
    except Exception:
        return None


def _sample_nvidia_smi() -> list[dict[str, Any]]:
    """Best-effort GPU sample via nvidia-smi, returns one dict per GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        rows: list[dict[str, Any]] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "util_gpu_pct": float(parts[1]),
                    "util_mem_pct": float(parts[2]),
                    "mem_used_mb": float(parts[3]),
                    "mem_total_mb": float(parts[4]),
                    "temp_c": float(parts[5]),
                }
            )
        return rows
    except Exception:
        return []


def _ping_once(host: str, timeout_s: float = 1.0) -> dict[str, Any]:
    """Best-effort ping sample. Returns dict with success/loss/rtt_ms."""
    # -W: per-ping timeout seconds (linux iputils)
    cmd = ["ping", "-c", "1", "-W", str(int(max(1.0, timeout_s))), host]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        stdout = (p.stdout or "") + "\n" + (p.stderr or "")
        # Parse rtt=... ms from "time=12.3 ms"
        m = re.search(r"time=([\d.]+)\s*ms", stdout)
        rtt_ms = float(m.group(1)) if m else None
        # Parse packet loss from summary, e.g. "1 packets transmitted, 1 received, 0% packet loss"
        m2 = re.search(r"(\d+)%\s*packet loss", stdout)
        loss_pct = float(m2.group(1)) if m2 else None
        success = p.returncode == 0
        return {"success": success, "rtt_ms": rtt_ms, "loss_pct": loss_pct}
    except Exception:
        return {"success": False, "rtt_ms": None, "loss_pct": None}


class TelemetryRecorder:
    """Run-scoped telemetry recorder writing CSVs to logs/runs/<run_id>/."""

    def __init__(self, global_config: GlobalConfig) -> None:
        self._cfg = global_config
        self._clock = _Clock(t0_wall=time.time(), t0_mono=time.monotonic())
        self._stop_event = threading.Event()

        self._run_dir = self._resolve_run_dir()
        self._csv_lock = threading.Lock()

        self._cpu = _CpuSampler()
        self._prev_net: dict[str, dict[str, int]] | None = None
        self._prev_net_ts_mono: float | None = None

        self._sampler_thread: threading.Thread | None = None

        self._lcmspy: GraphLCMSpy | None = None
        self._lcmspy_thread: threading.Thread | None = None

        self._metrics_bus: LCMPubSubBase | None = None

        # File handles / writers
        self._system_f = None
        self._system_w = None
        self._process_f = None
        self._process_w = None
        self._net_f = None
        self._net_w = None
        self._ping_f = None
        self._ping_w = None
        self._gpu_f = None
        self._gpu_w = None
        self._lcm_f = None
        self._lcm_w = None
        self._app_metrics_f = None
        self._app_metrics_w = None

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def _resolve_run_dir(self) -> Path:
        if self._cfg.telemetry_run_dir:
            p = Path(self._cfg.telemetry_run_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        run_id = f"{ts}_{pid}"
        p = DIMOS_LOG_DIR / "runs" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _write_run_meta(self) -> None:
        meta: dict[str, Any] = {
            "run_dir": str(self._run_dir),
            "pid": os.getpid(),
            "start_wall": self._clock.t0_wall,
            "start_mono": self._clock.t0_mono,
            "global_config": self._cfg.model_dump(),
        }
        # Best-effort git SHA
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(DIMOS_LOG_DIR.parent),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            meta["git_sha"] = sha
        except Exception:
            pass
        try:
            (self._run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        except Exception as e:
            logger.warning("Failed to write run_meta.json", error=str(e))

    def _open_csvs(self) -> None:
        self._system_f = open(self._run_dir / "system.csv", "w", newline="")
        self._system_w = csv.writer(self._system_f)
        self._system_w.writerow(
            [
                "ts_wall",
                "ts_mono",
                "t_rel",
                "cpu_percent",
                "load1",
                "load5",
                "load15",
                "mem_total_kb",
                "mem_avail_kb",
                "swap_total_kb",
                "swap_free_kb",
            ]
        )

        self._process_f = open(self._run_dir / "process.csv", "w", newline="")
        self._process_w = csv.writer(self._process_f)
        self._process_w.writerow(
            ["ts_wall", "ts_mono", "t_rel", "pid", "kind", "cpu_percent", "rss_kb", "cmdline"]
        )

        self._net_f = open(self._run_dir / "net.csv", "w", newline="")
        self._net_w = csv.writer(self._net_f)
        self._net_w.writerow(
            [
                "ts_wall",
                "ts_mono",
                "t_rel",
                "iface",
                "rx_bytes",
                "tx_bytes",
                "rx_packets",
                "tx_packets",
                "rx_bps",
                "tx_bps",
                "rx_pps",
                "tx_pps",
                "rx_errs",
                "tx_errs",
                "rx_drop",
                "tx_drop",
            ]
        )

        self._ping_f = open(self._run_dir / "ping.csv", "w", newline="")
        self._ping_w = csv.writer(self._ping_f)
        self._ping_w.writerow(
            ["ts_wall", "ts_mono", "t_rel", "host", "success", "rtt_ms", "loss_pct"]
        )

        self._gpu_f = open(self._run_dir / "gpu.csv", "w", newline="")
        self._gpu_w = csv.writer(self._gpu_f)
        self._gpu_w.writerow(
            [
                "ts_wall",
                "ts_mono",
                "t_rel",
                "gpu_index",
                "util_gpu_pct",
                "util_mem_pct",
                "mem_used_mb",
                "mem_total_mb",
                "temp_c",
            ]
        )

        self._lcm_f = open(self._run_dir / "lcm.csv", "w", newline="")
        self._lcm_w = csv.writer(self._lcm_f)
        self._lcm_w.writerow(
            ["ts_wall", "ts_mono", "t_rel", "topic", "freq_hz", "kbps", "total_bytes"]
        )

        self._app_metrics_f = open(self._run_dir / "app_metrics.csv", "w", newline="")
        self._app_metrics_w = csv.writer(self._app_metrics_f)
        self._app_metrics_w.writerow(["ts_wall", "ts_mono", "t_rel", "metric_name", "value"])

    def _close_csvs(self) -> None:
        for f in [
            self._system_f,
            self._process_f,
            self._net_f,
            self._ping_f,
            self._gpu_f,
            self._lcm_f,
            self._app_metrics_f,
        ]:
            try:
                if f:
                    f.flush()
                    f.close()
            except Exception:
                pass

    def start(self) -> None:
        self._write_run_meta()
        self._open_csvs()

        # Start LCM spy (transport telemetry)
        try:
            self._lcmspy = GraphLCMSpy(autoconf=True, graph_log_window=0.5)
            self._lcmspy.start()
            self._lcmspy_thread = threading.Thread(target=self._lcmspy_loop, daemon=True)
            self._lcmspy_thread.start()
        except Exception as e:
            logger.warning("Telemetry: failed to start LCMSpy", error=str(e))

        # Start /metrics subscriber (typed Float32)
        try:
            if Float32 is not None:
                self._metrics_bus = LCMPubSubBase(autoconf=True)
                self._metrics_bus.start()
                assert self._metrics_bus.l is not None

                def _on_metric(channel: str, data: bytes) -> None:
                    ts_wall, ts_mono, t_rel = self._clock.now()
                    try:
                        msg = Float32.lcm_decode(data)  # type: ignore[union-attr]
                        value = float(msg.data)
                    except Exception:
                        return
                    # channel is like "/metrics/foo"
                    with self._csv_lock:
                        if self._app_metrics_w is not None:
                            self._app_metrics_w.writerow([ts_wall, ts_mono, t_rel, channel, value])

                self._metrics_bus.l.subscribe(r"^/metrics/.*", _on_metric)
        except Exception as e:
            logger.warning("Telemetry: failed to start /metrics subscriber", error=str(e))

        self._sampler_thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._sampler_thread.start()

        logger.info("Telemetry: recording started", run_dir=str(self._run_dir))

    def stop(self) -> None:
        self._stop_event.set()

        if self._sampler_thread and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=2.0)

        if self._lcmspy:
            try:
                self._lcmspy.stop()
            except Exception:
                pass
        if self._lcmspy_thread and self._lcmspy_thread.is_alive():
            self._lcmspy_thread.join(timeout=2.0)

        if self._metrics_bus:
            try:
                self._metrics_bus.stop()
            except Exception:
                pass

        self._close_csvs()
        logger.info("Telemetry: recording stopped", run_dir=str(self._run_dir))

    def _sampler_loop(self) -> None:
        rate = max(0.1, float(self._cfg.telemetry_rate_hz))
        period_s = 1.0 / rate

        next_ping = 0.0
        next_gpu = 0.0

        while not self._stop_event.is_set():
            ts_wall, ts_mono, t_rel = self._clock.now()

            # System
            cpu_pct = self._cpu.sample_percent()
            load1, load5, load15 = _read_loadavg()
            mem = _read_meminfo_kb()
            with self._csv_lock:
                if self._system_w is not None:
                    self._system_w.writerow(
                        [
                            ts_wall,
                            ts_mono,
                            t_rel,
                            cpu_pct,
                            load1,
                            load5,
                            load15,
                            mem.get("MemTotal", -1),
                            mem.get("MemAvailable", -1),
                            mem.get("SwapTotal", -1),
                            mem.get("SwapFree", -1),
                        ]
                    )

            # Network
            net = _read_net_dev()
            if self._prev_net is not None and self._prev_net_ts_mono is not None:
                dt = ts_mono - self._prev_net_ts_mono
                if dt > 0:
                    with self._csv_lock:
                        if self._net_w is not None:
                            for iface, cur in net.items():
                                prev = self._prev_net.get(iface)
                                if prev is None:
                                    continue
                                rx_bps = (cur["rx_bytes"] - prev["rx_bytes"]) / dt
                                tx_bps = (cur["tx_bytes"] - prev["tx_bytes"]) / dt
                                rx_pps = (cur["rx_packets"] - prev["rx_packets"]) / dt
                                tx_pps = (cur["tx_packets"] - prev["tx_packets"]) / dt
                                self._net_w.writerow(
                                    [
                                        ts_wall,
                                        ts_mono,
                                        t_rel,
                                        iface,
                                        cur["rx_bytes"],
                                        cur["tx_bytes"],
                                        cur["rx_packets"],
                                        cur["tx_packets"],
                                        rx_bps,
                                        tx_bps,
                                        rx_pps,
                                        tx_pps,
                                        cur["rx_errs"],
                                        cur["tx_errs"],
                                        cur["rx_drop"],
                                        cur["tx_drop"],
                                    ]
                                )
            self._prev_net = net
            self._prev_net_ts_mono = ts_mono

            # Processes
            main_pid = os.getpid()
            worker_pids = set(get_worker_pids())
            rerun_pids = set(_find_pids_with_cmd_substr("rerun"))
            pids = {main_pid} | worker_pids | rerun_pids

            with self._csv_lock:
                if self._process_w is not None:
                    for pid in sorted(pids):
                        kind = (
                            "main"
                            if pid == main_pid
                            else "dask_worker"
                            if pid in worker_pids
                            else "rerun"
                            if pid in rerun_pids
                            else "other"
                        )
                        ps_row = _ps_sample_pid(pid)
                        cpu, rss_kb = ps_row if ps_row is not None else (-1.0, -1)
                        cmd = _read_proc_cmdline(pid)
                        self._process_w.writerow(
                            [ts_wall, ts_mono, t_rel, pid, kind, cpu, rss_kb, cmd]
                        )

            # Ping (robot_ip)
            if self._cfg.robot_ip and ts_mono >= next_ping:
                next_ping = ts_mono + 1.0
                res = _ping_once(self._cfg.robot_ip, timeout_s=1.0)
                with self._csv_lock:
                    if self._ping_w is not None:
                        self._ping_w.writerow(
                            [
                                ts_wall,
                                ts_mono,
                                t_rel,
                                self._cfg.robot_ip,
                                bool(res.get("success")),
                                res.get("rtt_ms"),
                                res.get("loss_pct"),
                            ]
                        )

            # GPU (best-effort)
            if ts_mono >= next_gpu:
                next_gpu = ts_mono + 1.0
                try:
                    # Import here to avoid hard dependency at import time.
                    import shutil

                    if shutil.which("nvidia-smi"):
                        gpus = _sample_nvidia_smi()
                        with self._csv_lock:
                            if self._gpu_w is not None:
                                for row in gpus:
                                    self._gpu_w.writerow(
                                        [
                                            ts_wall,
                                            ts_mono,
                                            t_rel,
                                            row["gpu_index"],
                                            row["util_gpu_pct"],
                                            row["util_mem_pct"],
                                            row["mem_used_mb"],
                                            row["mem_total_mb"],
                                            row["temp_c"],
                                        ]
                                    )
                except Exception:
                    pass

            self._stop_event.wait(period_s)

    def _lcmspy_loop(self) -> None:
        assert self._lcmspy is not None
        rate = max(0.1, float(self._cfg.telemetry_rate_hz))
        period_s = 1.0 / rate
        window_s = max(0.1, float(self._cfg.telemetry_lcm_window_s))

        while not self._stop_event.is_set():
            ts_wall, ts_mono, t_rel = self._clock.now()
            try:
                # Snapshot topics to avoid dict-size-change errors.
                topics = list(self._lcmspy.topic.items())
            except Exception:
                topics = []

            with self._csv_lock:
                if self._lcm_w is not None:
                    # Global totals (spy itself is a Topic-like object)
                    try:
                        self._lcm_w.writerow(
                            [
                                ts_wall,
                                ts_mono,
                                t_rel,
                                "__total__",
                                self._lcmspy.freq(window_s),
                                self._lcmspy.kbps(window_s),
                                self._lcmspy.total_traffic(),
                            ]
                        )
                    except Exception:
                        pass

                    for name, topic in topics:
                        try:
                            freq = topic.freq(window_s)
                            kbps = topic.kbps(window_s)
                            if freq <= 0 and kbps <= 0:
                                continue
                            self._lcm_w.writerow(
                                [ts_wall, ts_mono, t_rel, name, freq, kbps, topic.total_traffic()]
                            )
                        except Exception:
                            continue

            self._stop_event.wait(period_s)
