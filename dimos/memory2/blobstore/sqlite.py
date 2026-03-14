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

from reactivex.disposable import Disposable

from dimos.memory2.blobstore.base import BlobStore
from dimos.memory2.utils import open_sqlite_connection, validate_identifier
from dimos.protocol.service.spec import BaseConfig


class SqliteBlobStoreConfig(BaseConfig):
    path: str | None = None


class SqliteBlobStore(BlobStore):
    """Stores blobs in a separate SQLite table per stream.

    Table layout per stream::

        CREATE TABLE "{stream}_blob" (
            id   INTEGER PRIMARY KEY,
            data BLOB NOT NULL
        );

    Supports two construction modes:

    - ``SqliteBlobStore(conn)`` — borrows an externally-managed connection.
    - ``SqliteBlobStore(path="file.db")`` — opens and owns its own connection.

    Does NOT commit; the caller (typically Backend) is responsible for commits.
    """

    def __init__(self, conn: sqlite3.Connection | None = None, *, path: str | None = None) -> None:
        super().__init__()
        if conn is not None and path is not None:
            raise ValueError("Specify either conn or path, not both")
        if conn is None and path is None:
            raise ValueError("Specify either conn or path")
        self._config = SqliteBlobStoreConfig(path=path)
        self._conn: sqlite3.Connection = conn  # type: ignore[assignment]  # set in start() if None
        self._path = path
        self._tables: set[str] = set()

    def _ensure_table(self, stream_name: str) -> None:
        if stream_name in self._tables:
            return
        validate_identifier(stream_name)
        self._conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{stream_name}_blob" '
            "(id INTEGER PRIMARY KEY, data BLOB NOT NULL)"
        )
        self._tables.add(stream_name)

    # ── Resource lifecycle ────────────────────────────────────────

    def start(self) -> None:
        if self._conn is None:
            assert self._path is not None
            self._conn = open_sqlite_connection(self._path)
            self.register_disposables(Disposable(action=lambda: self._conn.close()))

    # ── BlobStore interface ───────────────────────────────────────

    def put(self, stream_name: str, key: int, data: bytes) -> None:
        self._ensure_table(stream_name)
        self._conn.execute(
            f'INSERT OR REPLACE INTO "{stream_name}_blob" (id, data) VALUES (?, ?)',
            (key, data),
        )

    def get(self, stream_name: str, key: int) -> bytes:
        try:
            row = self._conn.execute(
                f'SELECT data FROM "{stream_name}_blob" WHERE id = ?', (key,)
            ).fetchone()
        except Exception:
            raise KeyError(f"No blob for stream={stream_name!r}, key={key}")
        if row is None:
            raise KeyError(f"No blob for stream={stream_name!r}, key={key}")
        result: bytes = row[0]
        return result

    def delete(self, stream_name: str, key: int) -> None:
        try:
            self._conn.execute(f'DELETE FROM "{stream_name}_blob" WHERE id = ?', (key,))
        except Exception:
            pass
