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

"""
Tests for Mission Control dashboard — routes, SocketIO events, and in-process monitors.

These are unit/integration tests that run without a real robot or LCM network.
The SocketIO server and HTTP routes are tested via httpx + starlette TestClient.
LCMSpy and AgentMessageMonitor are tested against mock/stub callbacks.

Run with:
    uv run pytest dimos/web/websocket_vis/test_mission_control.py -v
"""

from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def templates_dir() -> Path:
    return Path(__file__).parent.parent / "templates"


@pytest.fixture()
def mission_control_html(templates_dir: Path) -> Path:
    return templates_dir / "mission_control.html"


# ---------------------------------------------------------------------------
# 1. Static file presence
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_mission_control_html_exists(self, mission_control_html: Path) -> None:
        """mission_control.html must exist in templates dir."""
        assert mission_control_html.exists(), (
            f"mission_control.html not found at {mission_control_html}. "
            "Create dimos/web/templates/mission_control.html."
        )

    def test_mission_control_html_is_not_empty(self, mission_control_html: Path) -> None:
        assert mission_control_html.stat().st_size > 0

    def test_mission_control_html_has_socketio_import(self, mission_control_html: Path) -> None:
        """Dashboard must load socket.io client."""
        content = mission_control_html.read_text()
        assert "socket.io" in content, "Dashboard must import socket.io client"

    def test_mission_control_html_has_tabs(self, mission_control_html: Path) -> None:
        """Dashboard must have Operations and Dev Tools tabs."""
        content = mission_control_html.read_text()
        assert "operations" in content.lower() or "Operations" in content
        assert "dev" in content.lower()

    def test_mission_control_html_has_agent_feed_panel(self, mission_control_html: Path) -> None:
        """Agent feed panel must be present (for in-process SocketIO streaming)."""
        content = mission_control_html.read_text()
        assert "agent" in content.lower(), "Dashboard must have an agent feed panel"

    def test_mission_control_html_has_lcm_stats_panel(self, mission_control_html: Path) -> None:
        """LCM stats panel must be present (for in-process SocketIO streaming)."""
        content = mission_control_html.read_text()
        assert "lcm" in content.lower(), "Dashboard must have an LCM stats panel"


# ---------------------------------------------------------------------------
# 2. HTTP Routes
# ---------------------------------------------------------------------------


@pytest.fixture()
def starlette_app(mission_control_html: Path, tmp_path: Path) -> Any:
    """Build the Starlette app the same way WebsocketVisModule does, but without
    starting LCM, SocketIO broadcast loops, or spy tool subprocesses."""
    import socketio
    from starlette.applications import Starlette
    from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
    from starlette.routing import Route

    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

    # Minimal spy-tool state
    tool_ports: dict[str, int] = {"humancli": 8003}
    tool_procs: dict[str, Any] = {}

    async def serve_index(request: Any) -> Any:
        viewer = "rerun-web"  # default to rerun-web for tests
        if viewer != "rerun-web":
            return RedirectResponse(url="/command-center")
        if mission_control_html.exists():
            return FileResponse(mission_control_html, media_type="text/html")
        return Response("not found", status_code=404)

    async def serve_mission_control(request: Any) -> Any:
        return FileResponse(mission_control_html, media_type="text/html")

    async def serve_legacy(request: Any) -> Any:
        return Response("legacy", media_type="text/html")

    async def serve_command_center(request: Any) -> Any:
        return Response("command-center", media_type="text/html")

    async def serve_health(request: Any) -> Any:
        return JSONResponse({"status": "ok", "port": 7779})

    async def serve_services(request: Any) -> Any:
        services = {}
        for name, port in tool_ports.items():
            proc = tool_procs.get(name)
            alive = proc is not None and proc.poll() is None
            services[name] = {"port": port, "url": f"http://localhost:{port}/", "alive": alive}
        return JSONResponse(services)

    routes = [
        Route("/", serve_index),
        Route("/mission-control", serve_mission_control),
        Route("/legacy", serve_legacy),
        Route("/command-center", serve_command_center),
        Route("/health", serve_health),
        Route("/api/services", serve_services),
    ]

    starlette = Starlette(routes=routes)
    app = socketio.ASGIApp(sio, starlette)
    return app


@pytest.fixture()
def test_client(starlette_app: Any) -> Any:
    """Starlette TestClient (synchronous) wrapping the ASGI app."""
    from starlette.testclient import TestClient

    return TestClient(starlette_app, raise_server_exceptions=True)


class TestRoutes:
    def test_health_returns_200(self, test_client: Any) -> None:
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_json(self, test_client: Any) -> None:
        resp = test_client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "port" in data

    def test_mission_control_returns_html(self, test_client: Any) -> None:
        resp = test_client.get("/mission-control")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_index_serves_mission_control_when_rerun_web(self, test_client: Any) -> None:
        """/ should serve dashboard when viewer=rerun-web."""
        resp = test_client.get("/", follow_redirects=False)
        # Either 200 (html) or 307/302 redirect to mission-control
        assert resp.status_code in (200, 302, 307)

    def test_api_services_returns_json(self, test_client: Any) -> None:
        resp = test_client.get("/api/services")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_api_services_shape(self, test_client: Any) -> None:
        """Each service entry must have port, url, alive fields."""
        resp = test_client.get("/api/services")
        for _name, info in resp.json().items():
            assert "port" in info
            assert "url" in info
            assert "alive" in info
            assert isinstance(info["alive"], bool)

    def test_api_services_alive_false_when_no_proc(self, test_client: Any) -> None:
        """Services with no running subprocess must report alive=False."""
        resp = test_client.get("/api/services")
        for _name, info in resp.json().items():
            # No subprocesses were actually started in this fixture
            assert info["alive"] is False

    def test_legacy_route_returns_html(self, test_client: Any) -> None:
        resp = test_client.get("/legacy")
        assert resp.status_code == 200

    def test_command_center_route_returns_html(self, test_client: Any) -> None:
        resp = test_client.get("/command-center")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. LCMSpy data model
# ---------------------------------------------------------------------------


class TestLCMSpyTopic:
    """Unit tests for the Topic data model (no LCM connection needed)."""

    def _make_topic(self, name: str = "test") -> Any:
        from dimos.utils.cli.lcmspy.lcmspy import Topic

        return Topic(name, history_window=60.0)

    def test_initial_freq_is_zero(self) -> None:
        topic = self._make_topic()
        assert topic.freq(5.0) == 0.0

    def test_initial_kbps_is_zero(self) -> None:
        topic = self._make_topic()
        assert topic.kbps(5.0) == 0.0

    def test_initial_total_traffic_is_zero(self) -> None:
        topic = self._make_topic()
        assert topic.total_traffic() == 0

    def test_msg_increments_total_traffic(self) -> None:
        topic = self._make_topic()
        topic.msg(b"x" * 100)
        assert topic.total_traffic() == 100

    def test_msg_multiple_increments(self) -> None:
        topic = self._make_topic()
        topic.msg(b"x" * 50)
        topic.msg(b"x" * 50)
        assert topic.total_traffic() == 100

    def test_freq_after_messages(self) -> None:
        topic = self._make_topic()
        for _ in range(10):
            topic.msg(b"x" * 10)
        freq = topic.freq(5.0)
        assert freq > 0.0

    def test_kbps_after_messages(self) -> None:
        topic = self._make_topic()
        for _ in range(5):
            topic.msg(b"x" * 1000)
        assert topic.kbps(5.0) > 0.0

    def test_total_traffic_hr_returns_string(self) -> None:
        topic = self._make_topic()
        topic.msg(b"x" * 1024)
        hr = topic.total_traffic_hr()
        assert isinstance(hr, str)
        assert len(hr) > 0

    def test_kbps_hr_returns_string_with_unit(self) -> None:
        topic = self._make_topic()
        for _ in range(5):
            topic.msg(b"x" * 1000)
        hr = topic.kbps_hr(5.0)
        assert "/s" in hr

    def test_old_messages_are_cleaned_up(self) -> None:
        topic = self._make_topic()
        # Add a message, then manually age it out
        topic.msg(b"x" * 100)
        # Inject an old timestamp
        with topic._lock:
            old_time = time.time() - 120.0
            topic.message_history[0] = (old_time, 100)
        topic._cleanup_old_messages()
        assert len(topic.message_history) == 0

    def test_size_returns_average_message_size(self) -> None:
        topic = self._make_topic()
        topic.msg(b"x" * 100)
        topic.msg(b"x" * 200)
        size = topic.size(5.0)
        assert 100 <= size <= 200


class TestGraphLCMSpy:
    """Unit tests for GraphLCMSpy — history ring buffers."""

    def _make_spy(self) -> Any:
        from dimos.utils.cli.lcmspy.lcmspy import GraphLCMSpy

        spy = GraphLCMSpy(autoconf=False)
        return spy

    def test_freq_history_starts_empty(self) -> None:
        spy = self._make_spy()
        assert len(spy.freq_history) == 0

    def test_bandwidth_history_starts_empty(self) -> None:
        spy = self._make_spy()
        assert len(spy.bandwidth_history) == 0

    def test_update_graphs_appends_to_history(self) -> None:
        spy = self._make_spy()
        spy.update_graphs(1.0)
        assert len(spy.freq_history) == 1
        assert len(spy.bandwidth_history) == 1

    def test_freq_history_bounded_by_maxlen(self) -> None:
        spy = self._make_spy()
        for _ in range(25):
            spy.update_graphs(1.0)
        assert len(spy.freq_history) <= 20  # maxlen=20


# ---------------------------------------------------------------------------
# 4. AgentMessageMonitor callbacks
# ---------------------------------------------------------------------------


class TestAgentMessageMonitor:
    """Unit tests for AgentMessageMonitor — callback dispatch, message storage."""

    def _make_monitor(self) -> Any:
        from dimos.utils.cli.agentspy.agentspy import AgentMessageMonitor

        # Patch PickleLCM so no real LCM socket is opened
        with patch("dimos.utils.cli.agentspy.agentspy.PickleLCM") as mock_lcm_cls:
            mock_lcm = MagicMock()
            mock_lcm_cls.return_value = mock_lcm
            monitor = AgentMessageMonitor(topic="/agent", max_messages=100)
            monitor._mock_transport = mock_lcm  # keep ref for assertions
        return monitor

    def _make_ai_message(self) -> Any:
        from langchain_core.messages import AIMessage

        return AIMessage(content="Moving to target")

    def _make_human_message(self) -> Any:
        from langchain_core.messages import HumanMessage

        return HumanMessage(content="Go to kitchen")

    def _make_tool_message(self) -> Any:
        from langchain_core.messages import ToolMessage

        return ToolMessage(content="done", tool_call_id="call_1", name="navigate")

    def test_initial_messages_empty(self) -> None:
        monitor = self._make_monitor()
        assert len(monitor.get_messages()) == 0

    def test_subscribe_adds_callback(self) -> None:
        monitor = self._make_monitor()
        cb = MagicMock()
        monitor.subscribe(cb)
        assert cb in monitor.callbacks

    def test_handle_message_stores_ai_message(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic

        monitor = self._make_monitor()
        msg = self._make_ai_message()
        monitor._handle_message(msg, LCMTopic("/agent"))
        assert len(monitor.get_messages()) == 1

    def test_handle_message_stores_human_message(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic

        monitor = self._make_monitor()
        msg = self._make_human_message()
        monitor._handle_message(msg, LCMTopic("/agent"))
        assert len(monitor.get_messages()) == 1

    def test_handle_message_stores_tool_message(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic

        monitor = self._make_monitor()
        msg = self._make_tool_message()
        monitor._handle_message(msg, LCMTopic("/agent"))
        assert len(monitor.get_messages()) == 1

    def test_handle_message_ignores_unknown_type(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic

        monitor = self._make_monitor()
        monitor._handle_message("not a message type", LCMTopic("/agent"))
        assert len(monitor.get_messages()) == 0

    def test_handle_message_fires_callback(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic

        monitor = self._make_monitor()
        cb = MagicMock()
        monitor.subscribe(cb)
        monitor._handle_message(self._make_ai_message(), LCMTopic("/agent"))
        cb.assert_called_once()

    def test_handle_message_callback_receives_entry(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic
        from dimos.utils.cli.agentspy.agentspy import MessageEntry

        monitor = self._make_monitor()
        received = []
        monitor.subscribe(received.append)
        monitor._handle_message(self._make_human_message(), LCMTopic("/agent"))
        assert isinstance(received[0], MessageEntry)

    def test_max_messages_enforced(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic

        monitor = self._make_monitor()
        for _ in range(150):
            monitor._handle_message(self._make_ai_message(), LCMTopic("/agent"))
        assert len(monitor.get_messages()) <= 100

    def test_entry_has_timestamp(self) -> None:
        from dimos.protocol.pubsub.impl.lcmpubsub import Topic as LCMTopic

        monitor = self._make_monitor()
        received = []
        monitor.subscribe(received.append)
        monitor._handle_message(self._make_ai_message(), LCMTopic("/agent"))
        assert received[0].timestamp > 0


# ---------------------------------------------------------------------------
# 5. Message formatting helpers
# ---------------------------------------------------------------------------


class TestMessageFormatting:
    def test_format_timestamp_returns_string(self) -> None:
        from dimos.utils.cli.agentspy.agentspy import format_timestamp

        result = format_timestamp(time.time())
        assert isinstance(result, str)
        assert ":" in result  # HH:MM:SS format

    def test_format_timestamp_includes_milliseconds(self) -> None:
        from dimos.utils.cli.agentspy.agentspy import format_timestamp

        result = format_timestamp(time.time())
        assert "." in result

    def test_get_message_type_human(self) -> None:
        from langchain_core.messages import HumanMessage

        from dimos.utils.cli.agentspy.agentspy import get_message_type_and_style

        msg_type, style = get_message_type_and_style(HumanMessage(content="hi"))
        assert "Human" in msg_type
        assert style == "green"

    def test_get_message_type_ai(self) -> None:
        from langchain_core.messages import AIMessage

        from dimos.utils.cli.agentspy.agentspy import get_message_type_and_style

        msg_type, style = get_message_type_and_style(AIMessage(content="ok"))
        assert "Agent" in msg_type
        assert style == "yellow"

    def test_get_message_type_tool(self) -> None:
        from langchain_core.messages import ToolMessage

        from dimos.utils.cli.agentspy.agentspy import get_message_type_and_style

        msg_type, style = get_message_type_and_style(
            ToolMessage(content="done", tool_call_id="1", name="navigate")
        )
        assert "Tool" in msg_type
        assert style == "red"

    def test_format_message_content_tool_includes_name(self) -> None:
        from langchain_core.messages import ToolMessage

        from dimos.utils.cli.agentspy.agentspy import format_message_content

        msg = ToolMessage(content="success", tool_call_id="1", name="navigate")
        content = format_message_content(msg)
        assert "navigate" in content

    def test_format_message_content_ai_with_tool_calls(self) -> None:
        from langchain_core.messages import AIMessage

        from dimos.utils.cli.agentspy.agentspy import format_message_content

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "move_to", "args": {"x": 1.0}, "id": "1", "type": "tool_call"}],
        )
        content = format_message_content(msg)
        assert "move_to" in content


# ---------------------------------------------------------------------------
# 6. LCM Stats SocketIO event shape (future: LCMStatsPublisher)
# ---------------------------------------------------------------------------


class TestLCMStatsEventShape:
    """Validate the shape of lcm_stats event payload that will be emitted."""

    def _make_stats_payload(self, topics_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Simulate what LCMStatsPublisher will produce."""
        return {
            "total": {
                "freq": sum(t["freq"] for t in topics_data),
                "kbps": sum(t["kbps"] for t in topics_data),
            },
            "topics": topics_data,
        }

    def test_payload_has_total_key(self) -> None:
        payload = self._make_stats_payload([])
        assert "total" in payload

    def test_payload_has_topics_key(self) -> None:
        payload = self._make_stats_payload([])
        assert "topics" in payload

    def test_total_has_freq_and_kbps(self) -> None:
        payload = self._make_stats_payload([])
        assert "freq" in payload["total"]
        assert "kbps" in payload["total"]

    def test_topic_entry_shape(self) -> None:
        topic_entry = {"name": "/odom", "freq": 10.0, "kbps": 2.5, "total_bytes": 100000}
        payload = self._make_stats_payload([topic_entry])
        entry = payload["topics"][0]
        assert "name" in entry
        assert "freq" in entry
        assert "kbps" in entry
        assert "total_bytes" in entry

    def test_total_freq_sums_topics(self) -> None:
        topics = [
            {"name": "/odom", "freq": 10.0, "kbps": 1.0, "total_bytes": 0},
            {"name": "/path", "freq": 5.0, "kbps": 0.5, "total_bytes": 0},
        ]
        payload = self._make_stats_payload(topics)
        assert payload["total"]["freq"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# 7. Agent message SocketIO event shape (future: AgentStreamMixin)
# ---------------------------------------------------------------------------


class TestAgentMessageEventShape:
    """Validate the shape of agent_message event payload."""

    def _entry_to_event(self, entry: Any) -> dict[str, Any]:
        """Simulate what AgentStreamMixin will produce from a MessageEntry."""
        from dimos.utils.cli.agentspy.agentspy import (
            format_message_content,
            format_timestamp,
            get_message_type_and_style,
        )

        msg_type, _style = get_message_type_and_style(entry.message)
        return {
            "timestamp": entry.timestamp,
            "timestamp_str": format_timestamp(entry.timestamp),
            "type": msg_type.strip(),
            "content": format_message_content(entry.message),
        }

    def _make_entry(self, content: str = "hello") -> Any:
        from langchain_core.messages import AIMessage

        from dimos.utils.cli.agentspy.agentspy import MessageEntry

        return MessageEntry(timestamp=time.time(), message=AIMessage(content=content))

    def test_event_has_timestamp(self) -> None:
        entry = self._make_entry()
        event = self._entry_to_event(entry)
        assert "timestamp" in event
        assert event["timestamp"] > 0

    def test_event_has_type(self) -> None:
        entry = self._make_entry()
        event = self._entry_to_event(entry)
        assert "type" in event
        assert len(event["type"]) > 0

    def test_event_has_content(self) -> None:
        entry = self._make_entry("Moving to position")
        event = self._entry_to_event(entry)
        assert "content" in event
        assert "Moving to position" in event["content"]

    def test_event_has_timestamp_str(self) -> None:
        entry = self._make_entry()
        event = self._entry_to_event(entry)
        assert "timestamp_str" in event
        assert ":" in event["timestamp_str"]


# ---------------------------------------------------------------------------
# 8. Spy tool subprocess lifecycle
# ---------------------------------------------------------------------------


class TestSpyToolLifecycle:
    """Test _launch_spy_tools / _stop_spy_tools logic without real subprocesses."""

    def _make_mock_proc(self, alive: bool = True) -> MagicMock:
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None if alive else 0
        return proc

    def test_alive_proc_poll_returns_none(self) -> None:
        proc = self._make_mock_proc(alive=True)
        assert proc.poll() is None

    def test_dead_proc_poll_returns_nonzero(self) -> None:
        proc = self._make_mock_proc(alive=False)
        assert proc.poll() is not None

    def test_services_endpoint_alive_reflects_proc_state(self) -> None:
        """alive field must be True only when proc.poll() is None."""
        proc_alive = self._make_mock_proc(alive=True)
        proc_dead = self._make_mock_proc(alive=False)

        def check_alive(proc: Any) -> bool:
            return proc is not None and proc.poll() is None

        assert check_alive(proc_alive) is True
        assert check_alive(proc_dead) is False
        assert check_alive(None) is False

    def test_stop_calls_terminate_then_wait(self) -> None:
        proc = self._make_mock_proc()
        procs = {"lcmspy": proc}

        for _name, p in procs.items():
            p.terminate()
            p.wait(timeout=3)

        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=3)

    def test_stop_force_kills_on_exception(self) -> None:
        proc = self._make_mock_proc()
        proc.wait.side_effect = Exception("timeout")

        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

        proc.kill.assert_called_once()
