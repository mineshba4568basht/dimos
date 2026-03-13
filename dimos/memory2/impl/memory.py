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

import threading
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dimos.memory2.backend import Backend
from dimos.memory2.codecs.base import Codec, codec_for
from dimos.memory2.store import Store

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dimos.memory2.type.filter import StreamQuery
    from dimos.memory2.type.observation import Observation

T = TypeVar("T")


class ListIndex(Generic[T]):
    """In-memory index for experimentation. Thread-safe."""

    def __init__(self, name: str = "<memory>") -> None:
        self._name = name
        self._observations: list[Observation[T]] = []
        self._next_id = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    def insert(self, obs: Observation[T]) -> int:
        with self._lock:
            obs.id = self._next_id
            row_id = self._next_id
            self._next_id += 1
            self._observations.append(obs)
        return row_id

    def query(self, q: StreamQuery) -> Iterator[Observation[T]]:
        with self._lock:
            snapshot = list(self._observations)

        # Text search — substring match
        if q.search_text is not None:
            needle = q.search_text.lower()
            it: Iterator[Observation[T]] = (
                obs for obs in snapshot if needle in str(obs.data).lower()
            )
            return q.apply(it)

        return q.apply(iter(snapshot))

    def count(self, q: StreamQuery) -> int:
        return sum(1 for _ in self.query(q))

    def fetch_by_ids(self, ids: list[int]) -> list[Observation[T]]:
        id_set = set(ids)
        with self._lock:
            return [obs for obs in self._observations if obs.id in id_set]


class MemoryStore(Store):
    """In-memory store for experimentation."""

    def _create_backend(
        self, name: str, payload_type: type[Any] | None = None, **config: Any
    ) -> Backend[Any]:
        index: ListIndex[Any] = ListIndex(name)

        # Resolve codec
        raw_codec = config.pop("codec", None)
        codec: Codec[Any]
        if isinstance(raw_codec, Codec):
            codec = raw_codec
        elif isinstance(raw_codec, str):
            from dimos.memory2.codecs.base import codec_from_id

            module = (
                f"{payload_type.__module__}.{payload_type.__qualname__}"
                if payload_type
                else "builtins.object"
            )
            codec = codec_from_id(raw_codec, module)
        else:
            codec = codec_for(payload_type)

        backend: Backend[Any] = Backend(
            index=index,
            codec=codec,
            blob_store=config.get("blob_store"),
            vector_store=config.get("vector_store"),
            notifier=config.get("notifier"),
            eager_blobs=config.get("eager_blobs", False),
        )
        return backend

    def list_streams(self) -> list[str]:
        return list(self._streams.keys())

    def delete_stream(self, name: str) -> None:
        self._streams.pop(name, None)
