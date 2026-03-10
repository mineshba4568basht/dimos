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

"""Tests for Stream.save() and LiveBackend protocol split."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dimos.memory2.backend import Backend, LiveBackend
from dimos.memory2.impl.memory import ListBackend
from dimos.memory2.stream import Stream
from dimos.memory2.transform import FnTransformer
from dimos.memory2.type import Observation

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dimos.memory2.filter import StreamQuery

# ── Helpers ──────────────────────────────────────────────────────────


def make_stream(n: int = 5, start_ts: float = 0.0) -> Stream[int]:
    backend = ListBackend[int]("test")
    for i in range(n):
        backend.append(i * 10, ts=start_ts + i)
    return Stream(source=backend)


class ReadOnlyBackend:
    """A Backend that does NOT support live mode (no subscribe)."""

    def __init__(self, name: str = "<readonly>") -> None:
        self._name = name
        self._obs: list[Observation[int]] = []
        self._next_id = 0

    @property
    def name(self) -> str:
        return self._name

    def iterate(self, query: StreamQuery) -> Iterator[Observation[int]]:
        yield from self._obs

    def append(
        self,
        payload: int,
        *,
        ts: float | None = None,
        pose: Any | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Observation[int]:
        obs: Observation[int] = Observation(
            id=self._next_id, ts=ts or 0.0, pose=pose, tags=tags or {}, _data=payload
        )
        self._next_id += 1
        self._obs.append(obs)
        return obs

    def count(self, query: StreamQuery) -> int:
        return len(self._obs)


# ═══════════════════════════════════════════════════════════════════
#  Protocol checks
# ═══════════════════════════════════════════════════════════════════


class TestProtocolSplit:
    def test_list_backend_is_live(self) -> None:
        b = ListBackend[int]("x")
        assert isinstance(b, LiveBackend)

    def test_list_backend_is_backend(self) -> None:
        b = ListBackend[int]("x")
        assert isinstance(b, Backend)

    def test_readonly_is_backend(self) -> None:
        b = ReadOnlyBackend()
        assert isinstance(b, Backend)

    def test_readonly_is_not_live(self) -> None:
        b = ReadOnlyBackend()
        assert not isinstance(b, LiveBackend)


# ═══════════════════════════════════════════════════════════════════
#  .live() rejects non-LiveBackend
# ═══════════════════════════════════════════════════════════════════


class TestLiveRejectsNonLive:
    def test_live_rejects_non_live_backend(self) -> None:
        b = ReadOnlyBackend("ro")
        s = Stream(source=b)
        with pytest.raises(TypeError, match="does not support live mode"):
            s.live()


# ═══════════════════════════════════════════════════════════════════
#  .save()
# ═══════════════════════════════════════════════════════════════════


class TestSave:
    def test_save_populates_target(self) -> None:
        source = make_stream(3)
        target_backend = ListBackend[int]("target")
        target = Stream(source=target_backend)

        source.save(target)

        results = target.fetch()
        assert len(results) == 3
        assert [o.data for o in results] == [0, 10, 20]

    def test_save_returns_target_stream(self) -> None:
        source = make_stream(2)
        target_backend = ListBackend[int]("target")
        target = Stream(source=target_backend)

        result = source.save(target)

        assert result is target

    def test_save_preserves_data(self) -> None:
        backend = ListBackend[int]("src")
        backend.append(42, ts=1.0, pose=(1, 2, 3), tags={"label": "cat"})
        source = Stream(source=backend)

        target_backend = ListBackend[int]("dst")
        target = Stream(source=target_backend)
        source.save(target)

        obs = target.first()
        assert obs.data == 42
        assert obs.ts == 1.0
        assert obs.pose == (1, 2, 3)
        assert obs.tags == {"label": "cat"}

    def test_save_with_transform(self) -> None:
        source = make_stream(3)  # data: 0, 10, 20
        doubled = source.transform(FnTransformer(lambda obs: obs.derive(data=obs.data * 2)))

        target_backend = ListBackend[int]("target")
        target = Stream(source=target_backend)
        doubled.save(target)

        assert [o.data for o in target.fetch()] == [0, 20, 40]

    def test_save_rejects_transform_target(self) -> None:
        source = make_stream(2)
        base = make_stream(2)
        transform_stream = base.transform(FnTransformer(lambda obs: obs.derive(obs.data)))

        with pytest.raises(TypeError, match="Cannot save to a transform stream"):
            source.save(transform_stream)

    def test_save_target_queryable(self) -> None:
        source = make_stream(5, start_ts=0.0)  # ts: 0,1,2,3,4

        target_backend = ListBackend[int]("target")
        target = Stream(source=target_backend)
        result = source.save(target)

        after_2 = result.after(2.0).fetch()
        assert [o.data for o in after_2] == [30, 40]

    def test_save_empty_source(self) -> None:
        source = make_stream(0)
        target_backend = ListBackend[int]("target")
        target = Stream(source=target_backend)

        result = source.save(target)

        assert result.count() == 0
        assert result.fetch() == []
