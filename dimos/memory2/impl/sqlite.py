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

import sqlite3
from typing import Any

from reactivex.disposable import Disposable

from dimos.memory2.backend import Backend
from dimos.memory2.blobstore.sqlite import SqliteBlobStore
from dimos.memory2.codecs.base import Codec, codec_for, codec_from_id, codec_id
from dimos.memory2.observationstore.sqlite import SqliteObservationStore
from dimos.memory2.registry import RegistryStore, deserialize_component, qual
from dimos.memory2.store import Store, StoreConfig
from dimos.memory2.utils import open_sqlite_connection, validate_identifier

# ── SqliteStore ──────────────────────────────────────────────────


class SqliteStoreConfig(StoreConfig):
    """Config for SQLite-backed store."""

    path: str = "memory.db"
    page_size: int = 256


class SqliteStore(Store):
    """Store backed by a SQLite database file."""

    default_config: type[SqliteStoreConfig] = SqliteStoreConfig
    config: SqliteStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._registry_conn = self._open_connection()
        self._registry = RegistryStore(self._registry_conn)

    def _open_connection(self) -> sqlite3.Connection:
        """Open a new WAL-mode connection with sqlite-vec loaded."""
        return open_sqlite_connection(self.config.path)

    # ── Backend from live objects (create path) ──────────────────

    def _build_new_backend(
        self,
        name: str,
        payload_type: type[Any],
        backend_conn: sqlite3.Connection,
        **config: Any,
    ) -> Backend[Any]:
        """Build a Backend for a new stream using live objects from config."""
        payload_module = f"{payload_type.__module__}.{payload_type.__qualname__}"

        # Resolve codec
        raw_codec = config.get("codec")
        if isinstance(raw_codec, str):
            codec = codec_from_id(raw_codec, payload_module)
        elif isinstance(raw_codec, Codec):
            codec = raw_codec
        elif raw_codec is not None:
            codec = raw_codec
        else:
            codec = codec_for(payload_type)

        # Resolve components — use overrides or create conn-shared defaults
        bs = config.get("blob_store")
        if bs is None:
            bs = SqliteBlobStore(backend_conn)
        vs = config.get("vector_store")
        if vs is None:
            from dimos.memory2.vectorstore.sqlite import SqliteVectorStore

            vs = SqliteVectorStore(backend_conn)

        blob_store_conn_match = isinstance(bs, SqliteBlobStore) and bs._conn is backend_conn
        eager_blobs = config.get("eager_blobs", False)

        metadata_store: SqliteObservationStore[Any] = SqliteObservationStore(
            backend_conn,
            name,
            codec,
            blob_store_conn_match=blob_store_conn_match and eager_blobs,
            page_size=config.get("page_size", self.config.page_size),
        )

        return Backend(
            metadata_store=metadata_store,
            codec=codec,
            blob_store=bs,
            vector_store=vs,
            notifier=config.get("notifier"),
            eager_blobs=eager_blobs,
        )

    # ── Backend from stored config (load path) ───────────────────

    def _assemble_backend(self, name: str, stored: dict[str, Any]) -> Backend[Any]:
        """Reconstruct a Backend from a stored config dict."""
        payload_module = stored["payload_module"]
        codec = codec_from_id(stored["codec_id"], payload_module)
        eager_blobs = stored.get("eager_blobs", False)
        page_size = stored.get("page_size", self.config.page_size)

        backend_conn = self._open_connection()

        # Reconstruct components from serialized config
        bs_data = stored.get("blob_store")
        if bs_data is not None:
            bs_cfg = bs_data.get("config", {})
            if bs_cfg.get("path") is None and bs_data["class"] == qual(SqliteBlobStore):
                bs: Any = SqliteBlobStore(backend_conn)
            else:
                bs = deserialize_component(bs_data)
        else:
            bs = SqliteBlobStore(backend_conn)

        vs_data = stored.get("vector_store")
        if vs_data is not None:
            from dimos.memory2.vectorstore.sqlite import SqliteVectorStore

            vs_cfg = vs_data.get("config", {})
            if vs_cfg.get("path") is None and vs_data["class"] == qual(SqliteVectorStore):
                vs: Any = SqliteVectorStore(backend_conn)
            else:
                vs = deserialize_component(vs_data)
        else:
            from dimos.memory2.vectorstore.sqlite import SqliteVectorStore

            vs = SqliteVectorStore(backend_conn)

        notifier_data = stored.get("notifier")
        if notifier_data is not None:
            notifier = deserialize_component(notifier_data)
        else:
            from dimos.memory2.livechannel.subject import SubjectNotifier

            notifier = SubjectNotifier()

        blob_store_conn_match = isinstance(bs, SqliteBlobStore) and bs._conn is backend_conn

        metadata_store: SqliteObservationStore[Any] = SqliteObservationStore(
            backend_conn,
            name,
            codec,
            blob_store_conn_match=blob_store_conn_match and eager_blobs,
            page_size=page_size,
        )

        backend: Backend[Any] = Backend(
            metadata_store=metadata_store,
            codec=codec,
            blob_store=bs,
            vector_store=vs,
            notifier=notifier,
            eager_blobs=eager_blobs,
        )
        self.register_disposables(Disposable(action=lambda: backend_conn.close()))
        return backend

    # ── Serialization helpers ────────────────────────────────────

    @staticmethod
    def _serialize_backend(
        backend: Backend[Any], payload_module: str, page_size: int
    ) -> dict[str, Any]:
        """Serialize a backend's config for registry storage."""
        cfg: dict[str, Any] = {
            "payload_module": payload_module,
            "codec_id": codec_id(backend.codec),
            "eager_blobs": backend.eager_blobs,
            "page_size": page_size,
        }
        if hasattr(backend.blob_store, "serialize"):
            cfg["blob_store"] = backend.blob_store.serialize()
        if hasattr(backend.vector_store, "serialize"):
            cfg["vector_store"] = backend.vector_store.serialize()
        if hasattr(backend.notifier, "serialize"):
            cfg["notifier"] = backend.notifier.serialize()
        return cfg

    # ── _create_backend ──────────────────────────────────────────

    def _create_backend(
        self, name: str, payload_type: type[Any] | None = None, **config: Any
    ) -> Backend[Any]:
        validate_identifier(name)

        stored = self._registry.get(name)

        if stored is not None:
            # Load path: validate type, assemble from stored config
            if payload_type is not None:
                actual_module = f"{payload_type.__module__}.{payload_type.__qualname__}"
                if actual_module != stored["payload_module"]:
                    raise ValueError(
                        f"Stream {name!r} was created with type {stored['payload_module']}, "
                        f"but opened with {actual_module}"
                    )
            backend = self._assemble_backend(name, stored)
        else:
            # Create path: build from live objects, then persist config
            if payload_type is None:
                raise TypeError(f"Stream {name!r} does not exist yet — payload_type is required")

            backend_conn = self._open_connection()
            self.register_disposables(Disposable(action=lambda: backend_conn.close()))

            page_size = config.get("page_size", self.config.page_size)
            backend = self._build_new_backend(name, payload_type, backend_conn, **config)

            payload_module = f"{payload_type.__module__}.{payload_type.__qualname__}"
            self._registry.put(
                name,
                self._serialize_backend(backend, payload_module, page_size),
            )

        return backend

    def list_streams(self) -> list[str]:
        db_names = set(self._registry.list_streams())
        return sorted(db_names | set(self._streams.keys()))

    def delete_stream(self, name: str) -> None:
        self._streams.pop(name, None)
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}_blob"')
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}_vec"')
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}_rtree"')
        self._registry.delete(name)

    def stop(self) -> None:
        super().stop()  # disposes owned metadata store connections
        self._registry_conn.close()
