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

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from dimos.core.resource import Resource

if TYPE_CHECKING:
    from dimos.memory.stream import EmbeddingStream, Stream, TextStream
    from dimos.memory.transformer import Transformer
    from dimos.memory.type import PoseProvider
    from dimos.models.embedding.base import Embedding, EmbeddingModel

T = TypeVar("T")


class StreamNamespace:
    """Attribute-access proxy for session streams.

    Usage::

        session.streams.image_stream   # same as looking up "image_stream" from list_streams()
        session.streams["image_stream"]
        list(session.streams)          # iterate all streams
        len(session.streams)
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _load(self) -> dict[str, Stream[Any]]:
        return {s._backend.stream_name: s for s in self._session.list_streams() if s._backend}

    def __getattr__(self, name: str) -> Stream[Any]:
        if name.startswith("_"):
            raise AttributeError(name)
        streams = self._load()
        if name in streams:
            return streams[name]
        raise AttributeError(
            f"No stream named {name!r}. Available: {', '.join(streams) or '(none)'}"
        )

    def __getitem__(self, name: str) -> Stream[Any]:
        streams = self._load()
        if name in streams:
            return streams[name]
        raise KeyError(name)

    def __iter__(self):
        return iter(self._load().values())

    def __len__(self) -> int:
        return len(self._load())

    def __contains__(self, name: str) -> bool:
        return name in self._load()

    def __repr__(self) -> str:
        names = list(self._load().keys())
        return f"StreamNamespace({names})"


class Session(Resource):
    """A session against a memory store. Creates and manages streams.

    Inherits DisposableBase so sessions can be added to CompositeDisposable.
    """

    @property
    def streams(self) -> StreamNamespace:
        """Attribute-access namespace for all streams in this session."""
        return StreamNamespace(self)

    def start(self) -> None:
        pass

    @abstractmethod
    def stream(
        self,
        name: str,
        payload_type: type[T],
        *,
        pose_provider: PoseProvider | None = None,
    ) -> Stream[T]:
        """Get or create a stored stream backed by the database."""

    @abstractmethod
    def text_stream(
        self,
        name: str,
        *,
        tokenizer: str = "unicode61",
        pose_provider: PoseProvider | None = None,
    ) -> TextStream[str]:
        """Get or create a text stream with FTS index."""

    @abstractmethod
    def embedding_stream(
        self,
        name: str,
        *,
        vec_dimensions: int | None = None,
        pose_provider: PoseProvider | None = None,
        parent_table: str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> EmbeddingStream[Embedding]:
        """Get or create an embedding stream with vec0 index."""

    @abstractmethod
    def list_streams(self) -> list[Stream[Any]]: ...

    @abstractmethod
    def delete_stream(self, name: str) -> None:
        """Drop a stream and all its associated tables (payload, rtree, etc.)."""

    @abstractmethod
    def materialize_transform(
        self,
        name: str,
        source: Stream[Any],
        transformer: Transformer[Any, Any],
        *,
        payload_type: type | None = None,
        live: bool = False,
        backfill_only: bool = False,
    ) -> Stream[Any]:
        """Create a stored stream from a transform pipeline."""

    @abstractmethod
    def resolve_parent_stream(self, name: str) -> str | None:
        """Return the direct parent stream name, or None if no lineage exists."""

    @abstractmethod
    def resolve_lineage_chain(self, source: str, target: str) -> tuple[str, ...]:
        """Return intermediate tables in the parent_id chain from source to target.

        Single hop (source directly parents target) returns ``()``.
        Two hops (source → mid → target) returns ``("mid",)``.
        Raises ``ValueError`` if no lineage path exists.
        """

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self) -> Session:
        return self


class Store(Resource):
    """Top-level entry point — wraps a database file."""

    @abstractmethod
    def session(self) -> Session: ...

    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
