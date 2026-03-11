# AGENTS.md — DimOS

Guide for AI agents working on the DimOS codebase.

## What is DimOS

An open-source universal operating system for generalist robotics. Modules communicate via typed streams over LCM, shared memory, or other transports. Blueprints compose modules into runnable robot stacks.

## Quick Start

```bash
# Install
uv sync --all-extras --no-extra dds

# Run a blueprint (Go2 robot in replay mode)
dimos --replay run unitree-go2-basic

# Run as daemon (background, with logging)
dimos --replay run unitree-go2-basic --daemon

# Check status of running instance
dimos status

# View per-run logs
dimos status   # shows log path
cat ~/.local/state/dimos/logs/<run-id>/main.jsonl

# Stop
dimos stop

# List available blueprints
dimos list
```

## Repo Structure

```
dimos/
├── core/                    # Module system, blueprints, workers, daemon, transports
│   ├── module.py            # Module base class (In/Out streams, @rpc, @skill)
│   ├── blueprints.py        # Blueprint composition (autoconnect)
│   ├── worker.py            # Forkserver worker processes
│   ├── module_coordinator.py # Module lifecycle manager
│   ├── daemon.py            # Daemon mode (daemonize, signal handling)
│   ├── run_registry.py      # Per-run tracking + log paths
│   ├── global_config.py     # GlobalConfig (env vars, CLI flags, .env)
│   ├── stream.py            # In[T]/Out[T] typed streams
│   └── transport.py         # LCMTransport, SHMTransport, pLCMTransport
├── robot/
│   ├── cli/dimos.py         # CLI entry point (typer)
│   ├── all_blueprints.py    # Auto-generated blueprint registry (DO NOT EDIT MANUALLY)
│   └── unitree/             # Unitree robot implementations (Go2, G1, B1)
├── agents/                  # AI agent system
│   ├── mcp/                 # MCP server (Model Context Protocol)
│   └── skills/              # Agent skills (navigation, manipulation, etc.)
├── navigation/              # Path planning, frontier exploration, patrolling
├── perception/              # Object detection, tracking, temporal memory
├── visualization/rerun/     # Rerun bridge (native + web viewers)
├── web/websocket_vis/       # Command center web dashboard (port 7779)
├── msgs/                    # Message types (geometry_msgs, sensor_msgs, nav_msgs)
├── protocol/                # Transport implementations (LCM, DDS, SHM)
├── memory/timeseries/       # Time series data storage + replay
├── hardware/sensors/        # Camera, lidar, ZED modules
├── mapping/                 # Voxel mapping, costmaps, occupancy grids
└── utils/                   # Logging, data loading, CLI tools
docs/                        # Human-readable documentation
├── usage/                   # Modules, blueprints, transports, visualization
│   ├── modules.md           # ← Start here for module system
│   ├── blueprints.md        # Blueprint composition guide
│   ├── visualization.md     # Viewer backends (rerun, rerun-web, foxglove)
│   └── configuration.md     # GlobalConfig + Configurable pattern
├── development/             # Testing, Docker, profiling, LFS
│   ├── testing.md           # Fast/slow tests, pytest usage
│   ├── dimos_run.md         # CLI usage, GlobalConfig, adding blueprints
│   └── large_file_management.md  # LFS + get_data()
└── agents/                  # Agent system documentation
```

## Architecture (minimum you need to know)

### Modules
Autonomous subsystems. Communicate via `In[T]`/`Out[T]` typed streams. Run in forkserver worker processes.

```python
from dimos.core.module import Module, In, Out
from dimos.core import rpc
from dimos.msgs.sensor_msgs import Image

class MyModule(Module):
    color_image: In[Image]       # input stream
    processed: Out[Image]        # output stream

    @rpc
    def start(self) -> None:
        self.color_image.subscribe(self._process)

    def _process(self, img: Image) -> None:
        self.processed.publish(do_something(img))
```

### Blueprints
Compose modules with `autoconnect()`. Streams auto-connect by `(name, type)` matching.

```python
from dimos.core.blueprints import autoconnect

my_blueprint = (
    autoconnect(module_a(), module_b(), module_c())
    .global_config(n_workers=4, robot_model="unitree_go2")
)
```

### GlobalConfig
Singleton config. Values from: defaults → `.env` → env vars → blueprint → CLI flags. Env vars prefixed with `DIMOS_`.

Key fields: `robot_ip`, `simulation`, `replay`, `viewer` (`rerun`|`rerun-web`|`foxglove`|`none`), `n_workers`.

### Transports
- **LCMTransport**: Default. Multicast UDP. Good for most messages.
- **SHMTransport/pSHMTransport**: Shared memory. Use for high-bandwidth (images, point clouds).
- **pLCMTransport**: Pickled LCM. For complex Python objects.

### Daemon Mode
`dimos run <blueprint> --daemon` daemonizes the process. Per-run logs go to `~/.local/state/dimos/logs/<run-id>/`. `dimos status` shows running instances. `dimos stop` sends SIGTERM → SIGKILL.

## Testing

```bash
# Fast tests (default — pyproject.toml addopts excludes slow, tool, mujoco)
uv run pytest

# Include slow tests (what CI runs)
./bin/pytest-slow

# Single file
uv run pytest dimos/core/test_blueprints.py -v

# Mypy
uv run mypy dimos/
```

Use `uv run` to ensure the venv and deps are correct.

**`uv run pytest` runs fast tests only** — `addopts` in `pyproject.toml` includes `-m 'not (tool or slow or mujoco)'`.

**CI runs `./bin/pytest-slow`** which uses `-m 'not (tool or mujoco)'` — includes slow tests but excludes tool and mujoco.

## Pre-commit & Code Quality

Pre-commit runs automatically on `git commit`. Includes ruff format, ruff check, license headers, LFS checks, doclinks.

**Known issue**: `doclinks` hook fails with `Executable 'python' not found`. Bypass with `SKIP=doclinks git commit -m "..."`.

**Always activate the shared venv before committing:**
```bash
source .venv/bin/activate
```

## Code Style Rules

- **Imports at top of file.** No inline imports unless there's a circular dependency.
- **Use `requests` for HTTP**, not `urllib.request`. It's an explicit dependency.
- **Use `Any` not `object`** for JSON value types.
- **Prefix non-CI manual test scripts with `demo_`** so they're excluded from pytest collection.
- **Don't hardcode ports/URLs.** Use `GlobalConfig` constants.
- **Type annotations required.** Mypy strict mode. Use `type: ignore` sparingly and only with specific error codes.

## `all_blueprints.py` is auto-generated

`dimos/robot/all_blueprints.py` is generated by `test_all_blueprints_generation.py`. After adding/renaming blueprints, run:

```bash
pytest dimos/robot/test_all_blueprints_generation.py
```

This regenerates the file locally. In CI (`CI=1`), it asserts the file is current instead of regenerating.

## Git Workflow

- **Branches**: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `chore/`, `perf/`
- **PRs target `dev`**, never push to `main` or `dev` directly
- **Don't force-push** unless after a rebase with conflicts
- **Minimize pushes to origin** — every push triggers CI (~1 hour on self-hosted runners)
- **Batch commits locally**, verify with `git diff origin/dev..HEAD`, then push once

## LFS Data

Test replay data is stored in Git LFS under `data/.lfs/`. The `get_data()` function handles extraction:

```python
from dimos.utils.data import get_data
path = get_data("go2_sf_office")  # auto-downloads from LFS if needed
```

After checking out a branch with new LFS data: `git lfs pull`.

## Common Pitfalls

1. **`all_blueprints.py` out of date** — CI will fail with `test_all_blueprints_is_current`. Run the generation test locally.
2. **`pyproject.toml` changes trigger full CI** (~1 hour). Include `uv.lock` in the same push.
3. **Forkserver workers** — modules run in separate processes. Don't rely on shared mutable state.
4. **DimOS type system** — stamped types inherit from base types. `TwistStamped` IS a `Twist`. Check class hierarchy before claiming type mismatches.
5. **`ReplayConnection.stop()` bug** — crashes with `AttributeError: stop_timer`. Known issue, pre-existing.
6. **macOS** — `kern.ipc.maxsockbuf` caps at 6291456. LCM multicast needs explicit sysctl config. Some tests excluded on macOS CI.

## Viewer Backends

- **`rerun` (default)**: Native [dimos-viewer](https://github.com/dimensionalOS/dimos-viewer) — custom Rerun fork with built-in teleop + click-to-navigate.
- **`rerun-web`**: Browser dashboard at `http://localhost:7779`. Includes command center + Rerun 3D viewer.
- **`foxglove`**: Foxglove bridge on `ws://localhost:8765`.

Set via CLI (`--viewer rerun-web`) or env var (`VIEWER=rerun-web`).

## Replay Mode

`dimos --replay run <blueprint>` uses recorded sensor data instead of a real robot. Data loops continuously. Replay data lives in `data/` (extracted from LFS tarballs in `data/.lfs/`).

## Further Reading

- **Module system**: `docs/usage/modules.md`
- **Blueprints**: `docs/usage/blueprints.md`
- **Transports**: `docs/usage/transports/`
- **Visualization**: `docs/usage/visualization.md`
- **Testing**: `docs/development/testing.md`
- **CLI / dimos run**: `docs/development/dimos_run.md`
- **LFS data**: `docs/development/large_file_management.md`
- **Agent system**: `docs/agents/`
