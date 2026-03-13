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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

from dimos.core.resource import Resource

if TYPE_CHECKING:
    from collections.abc import Iterator

    from reactivex.abc import DisposableBase

    from dimos.memory2.buffer import BackpressureBuffer
    from dimos.memory2.type.filter import StreamQuery
    from dimos.memory2.type.observation import Observation
    from dimos.models.embedding.base import Embedding

T = TypeVar("T")


@runtime_checkable
class Index(Protocol[T]):
    """Core metadata storage and query engine for observations.

    Handles only observation metadata storage, query pushdown, and count.
    Blob/vector/live orchestration is handled by the concrete Backend class.
    """

    @property
    def name(self) -> str: ...

    def insert(self, obs: Observation[T]) -> int:
        """Insert observation metadata, return assigned id."""
        ...

    def query(self, q: StreamQuery) -> Iterator[Observation[T]]:
        """Execute query against metadata. Blobs are NOT loaded here."""
        ...

    def count(self, q: StreamQuery) -> int: ...

    def fetch_by_ids(self, ids: list[int]) -> list[Observation[T]]:
        """Batch fetch by id (for vector search results)."""
        ...


# ── Live notification ─────────────────────────────────────────────


class Notifier(ABC, Generic[T]):
    """Push-notification for live observation delivery.

    Decouples the notification mechanism from storage.  The built-in
    ``SubjectNotifier`` handles same-process fan-out (thread-safe, zero
    config).  External implementations (Redis pub/sub, Postgres
    LISTEN/NOTIFY, inotify) can be injected for cross-process use.
    """

    @abstractmethod
    def subscribe(self, buf: BackpressureBuffer[Observation[T]]) -> DisposableBase:
        """Register *buf* to receive new observations. Returns a
        disposable that unsubscribes when disposed."""
        ...

    @abstractmethod
    def notify(self, obs: Observation[T]) -> None:
        """Fan out *obs* to all current subscribers."""
        ...


# ── Blob storage ──────────────────────────────────────────────────


class BlobStore(Resource):
    """Persistent storage for encoded payload blobs.

    Separates payload data from metadata indexing so that large blobs
    (images, point clouds) don't penalize metadata queries.
    """

    @abstractmethod
    def put(self, stream_name: str, key: int, data: bytes) -> None:
        """Store a blob for the given stream and observation id."""
        ...

    @abstractmethod
    def get(self, stream_name: str, key: int) -> bytes:
        """Retrieve a blob by stream name and observation id."""
        ...

    @abstractmethod
    def delete(self, stream_name: str, key: int) -> None:
        """Delete a blob by stream name and observation id."""
        ...


# ── Vector storage ───────────────────────────────────────────────


class VectorStore(Resource):
    """Pluggable storage and ANN index for embedding vectors.

    Separates vector indexing from metadata so backends can swap
    search strategies (brute-force, vec0, FAISS, Qdrant) independently.

    Same shape as BlobStore: ``put`` / ``search`` / ``delete``, keyed
    by ``(stream, observation_id)``.  Index creation is lazy — the
    first ``put`` for a stream determines dimensionality.
    """

    @abstractmethod
    def put(self, stream_name: str, key: int, embedding: Embedding) -> None:
        """Store an embedding vector for the given stream and observation id."""
        ...

    @abstractmethod
    def search(self, stream_name: str, query: Embedding, k: int) -> list[tuple[int, float]]:
        """Return top-k (observation_id, similarity) pairs, descending."""
        ...

    @abstractmethod
    def delete(self, stream_name: str, key: int) -> None:
        """Remove a vector. Silent if missing."""
        ...
