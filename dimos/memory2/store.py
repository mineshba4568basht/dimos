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
from typing import Any, TypeVar, cast

from dimos.core.resource import CompositeResource
from dimos.memory2.codecs.base import Codec
from dimos.memory2.stream import Stream
from dimos.memory2.type.backend import BlobStore, Notifier, VectorStore
from dimos.protocol.service.spec import BaseConfig, Configurable

T = TypeVar("T")


# ── Configuration ─────────────────────────────────────────────────


class StoreConfig(BaseConfig):
    """Store-level config. These are defaults inherited by all streams."""

    notifier: Notifier[Any] | None = None
    blob_store: BlobStore | None = None
    vector_store: VectorStore | None = None
    eager_blobs: bool = False
    codec: Codec[Any] | str | None = None


# ── Store ─────────────────────────────────────────────────────────


class Store(Configurable[StoreConfig], CompositeResource):
    """Top-level entry point — wraps a storage location (file, URL, etc.).

    Store directly manages streams. No Session layer.
    """

    default_config: type[StoreConfig] = StoreConfig

    def __init__(self, **kwargs: Any) -> None:
        Configurable.__init__(self, **kwargs)
        CompositeResource.__init__(self)
        self._streams: dict[str, Stream[Any]] = {}

    @abstractmethod
    def _create_backend(
        self, name: str, payload_type: type[Any] | None = None, **config: Any
    ) -> Any:
        """Create a Backend for the named stream. Called once per stream name."""
        ...

    def stream(self, name: str, payload_type: type[T] | None = None, **overrides: Any) -> Stream[T]:
        """Get or create a named stream. Returns the same Stream on repeated calls.

        Per-stream ``overrides`` (e.g. ``blob_store=``, ``codec=``) are merged
        on top of the store-level defaults from :class:`StoreConfig`.
        """
        if name not in self._streams:
            resolved = {k: v for k, v in vars(self.config).items() if v is not None}
            resolved.update({k: v for k, v in overrides.items() if v is not None})
            backend = self._create_backend(name, payload_type, **resolved)
            self._streams[name] = Stream(source=backend)
        return cast("Stream[T]", self._streams[name])

    @abstractmethod
    def list_streams(self) -> list[str]:
        """Return names of all streams in this store."""
        ...

    @abstractmethod
    def delete_stream(self, name: str) -> None:
        """Delete a stream by name (from cache and underlying storage)."""
        ...
