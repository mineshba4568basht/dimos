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

"""Concrete composite Backend that orchestrates Index + BlobStore + VectorStore + Notifier."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dimos.memory2.codecs.base import Codec
from dimos.memory2.livechannel.subject import SubjectNotifier
from dimos.memory2.type.observation import _UNLOADED

if TYPE_CHECKING:
    from collections.abc import Iterator

    from reactivex.abc import DisposableBase

    from dimos.memory2.buffer import BackpressureBuffer
    from dimos.memory2.type.backend import BlobStore, Index, Notifier, VectorStore
    from dimos.memory2.type.filter import StreamQuery
    from dimos.memory2.type.observation import Observation

T = TypeVar("T")


class Backend(Generic[T]):
    """Orchestrates metadata (Index), blob, vector, and live stores for one stream.

    This is a concrete class — NOT a protocol. All shared orchestration logic
    (encode → insert → store blob → index vector → notify) lives here,
    eliminating duplication between ListIndex and SqliteIndex.
    """

    def __init__(
        self,
        *,
        index: Index[T],
        codec: Codec[Any],
        blob_store: BlobStore | None = None,
        vector_store: VectorStore | None = None,
        notifier: Notifier[T] | None = None,
        eager_blobs: bool = False,
    ) -> None:
        self._index = index
        self._codec = codec
        self._blob_store = blob_store
        self._vector_store = vector_store
        self._notifier: Notifier[T] = notifier or SubjectNotifier()
        self._eager_blobs = eager_blobs

    @property
    def name(self) -> str:
        return self._index.name

    @property
    def notifier(self) -> Notifier[T]:
        return self._notifier

    @property
    def index(self) -> Index[T]:
        return self._index

    @property
    def blob_store(self) -> BlobStore | None:
        return self._blob_store

    @property
    def vector_store(self) -> VectorStore | None:
        return self._vector_store

    # ── Write ────────────────────────────────────────────────────

    def _make_loader(self, row_id: int) -> Any:
        bs = self._blob_store
        if bs is None:
            raise RuntimeError("BlobStore required but not configured")
        name, codec = self.name, self._codec

        def loader() -> Any:
            raw = bs.get(name, row_id)
            return codec.decode(raw)

        return loader

    def append(self, obs: Observation[T]) -> Observation[T]:
        # Encode payload before any locking (avoids holding locks during IO)
        encoded: bytes | None = None
        if self._blob_store is not None:
            encoded = self._codec.encode(obs._data)

        try:
            # Insert metadata into index, get assigned id
            row_id = self._index.insert(obs)
            obs.id = row_id

            # Store blob
            if encoded is not None:
                assert self._blob_store is not None
                self._blob_store.put(self.name, row_id, encoded)
                # Replace inline data with lazy loader
                obs._data = _UNLOADED  # type: ignore[assignment]
                obs._loader = self._make_loader(row_id)

            # Store embedding vector
            if self._vector_store is not None:
                emb = getattr(obs, "embedding", None)
                if emb is not None:
                    self._vector_store.put(self.name, row_id, emb)

            # Commit if the index supports it (e.g. SqliteIndex)
            if hasattr(self._index, "commit"):
                self._index.commit()
        except BaseException:
            if hasattr(self._index, "rollback"):
                self._index.rollback()
            raise

        self._notifier.notify(obs)
        return obs

    # ── Read ─────────────────────────────────────────────────────

    def iterate(self, query: StreamQuery) -> Iterator[Observation[T]]:
        if query.search_vec is not None and query.live_buffer is not None:
            raise TypeError("Cannot combine .search() with .live() — search is a batch operation.")
        buf = query.live_buffer
        if buf is not None:
            sub = self._notifier.subscribe(buf)
            return self._iterate_live(query, buf, sub)
        return self._iterate_snapshot(query)

    def _attach_loaders(self, it: Iterator[Observation[T]]) -> Iterator[Observation[T]]:
        """Attach lazy blob loaders to observations from the index."""
        if self._blob_store is None:
            yield from it
            return
        for obs in it:
            if obs._loader is None and isinstance(obs._data, type(_UNLOADED)):
                obs._loader = self._make_loader(obs.id)
            yield obs

    def _iterate_snapshot(self, query: StreamQuery) -> Iterator[Observation[T]]:
        if query.search_vec is not None and self._vector_store is not None:
            yield from self._vector_search(query)
            return

        it: Iterator[Observation[T]] = self._attach_loaders(self._index.query(query))

        # Apply python post-filters after loaders are attached (so obs.data works)
        python_filters = getattr(self._index, "_pending_python_filters", None)
        pending_query = getattr(self._index, "_pending_query", None)
        if python_filters:
            from itertools import islice as _islice

            it = (obs for obs in it if all(f.matches(obs) for f in python_filters))
            if pending_query and pending_query.offset_val:
                it = _islice(it, pending_query.offset_val, None)
            if pending_query and pending_query.limit_val is not None:
                it = _islice(it, pending_query.limit_val)

        if self._eager_blobs and self._blob_store is not None:
            for obs in it:
                _ = obs.data  # trigger lazy loader
                yield obs
        else:
            yield from it

    def _vector_search(self, query: StreamQuery) -> Iterator[Observation[T]]:
        vs = self._vector_store
        assert vs is not None and query.search_vec is not None

        hits = vs.search(self.name, query.search_vec, query.search_k or 10)
        if not hits:
            return

        ids = [h[0] for h in hits]
        obs_list = list(self._attach_loaders(iter(self._index.fetch_by_ids(ids))))
        obs_by_id = {obs.id: obs for obs in obs_list}

        # Preserve VectorStore ranking order
        ranked: list[Observation[T]] = []
        for obs_id, sim in hits:
            match = obs_by_id.get(obs_id)
            if match is not None:
                ranked.append(
                    match.derive(data=match.data, embedding=query.search_vec, similarity=sim)
                )

        # Apply remaining query ops (skip vector search)
        rest = replace(query, search_vec=None, search_k=None)
        yield from rest.apply(iter(ranked))

    def _iterate_live(
        self,
        query: StreamQuery,
        buf: BackpressureBuffer[Observation[T]],
        sub: DisposableBase,
    ) -> Iterator[Observation[T]]:
        from dimos.memory2.buffer import ClosedError

        eager = self._eager_blobs and self._blob_store is not None

        try:
            # Backfill phase
            last_id = -1
            for obs in self._iterate_snapshot(query):
                last_id = max(last_id, obs.id)
                yield obs

            # Live tail
            filters = query.filters
            while True:
                obs = buf.take()
                if obs.id <= last_id:
                    continue
                last_id = obs.id
                if filters and not all(f.matches(obs) for f in filters):
                    continue
                if eager:
                    _ = obs.data  # trigger lazy loader
                yield obs
        except (ClosedError, StopIteration):
            pass
        finally:
            sub.dispose()

    def count(self, query: StreamQuery) -> int:
        if query.search_vec:
            return sum(1 for _ in self.iterate(query))
        return self._index.count(query)

    def stop(self) -> None:
        """Stop the index (closes per-stream connections if any)."""
        if hasattr(self._index, "stop"):
            self._index.stop()
