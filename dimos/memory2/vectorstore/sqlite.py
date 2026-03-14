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

import json
import sqlite3
from typing import TYPE_CHECKING

from reactivex.disposable import Disposable

from dimos.memory2.utils import open_sqlite_connection, validate_identifier
from dimos.memory2.vectorstore.base import VectorStore
from dimos.protocol.service.spec import BaseConfig

if TYPE_CHECKING:
    from dimos.models.embedding.base import Embedding


class SqliteVectorStoreConfig(BaseConfig):
    path: str | None = None


class SqliteVectorStore(VectorStore):
    """Vector store backed by sqlite-vec's vec0 virtual tables.

    Creates one virtual table per stream: ``"{stream}_vec"``.
    Dimensionality is determined lazily on the first ``put()``.

    Supports two construction modes:

    - ``SqliteVectorStore(conn)`` — borrows an externally-managed connection.
    - ``SqliteVectorStore(path="file.db")`` — opens and owns its own connection.
    """

    def __init__(self, conn: sqlite3.Connection | None = None, *, path: str | None = None) -> None:
        super().__init__()
        if conn is not None and path is not None:
            raise ValueError("Specify either conn or path, not both")
        if conn is None and path is None:
            raise ValueError("Specify either conn or path")
        self._config = SqliteVectorStoreConfig(path=path)
        self._conn: sqlite3.Connection = conn  # type: ignore[assignment]  # set in start() if None
        self._path = path
        self._tables: dict[str, int] = {}  # stream_name -> dimensionality

    def _ensure_table(self, stream_name: str, dim: int) -> None:
        if stream_name in self._tables:
            return
        validate_identifier(stream_name)
        self._conn.execute(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS "{stream_name}_vec" '
            f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )
        self._tables[stream_name] = dim

    def start(self) -> None:
        if self._conn is None:
            assert self._path is not None
            self._conn = open_sqlite_connection(self._path)
            self.register_disposables(Disposable(action=lambda: self._conn.close()))

    def put(self, stream_name: str, key: int, embedding: Embedding) -> None:
        vec = embedding.to_numpy().tolist()
        self._ensure_table(stream_name, len(vec))
        self._conn.execute(
            f'INSERT OR REPLACE INTO "{stream_name}_vec" (rowid, embedding) VALUES (?, ?)',
            (key, json.dumps(vec)),
        )

    def search(self, stream_name: str, query: Embedding, k: int) -> list[tuple[int, float]]:
        if stream_name not in self._tables:
            return []
        vec = query.to_numpy().tolist()
        rows = self._conn.execute(
            f'SELECT rowid, distance FROM "{stream_name}_vec" WHERE embedding MATCH ? AND k = ?',
            (json.dumps(vec), k),
        ).fetchall()
        # vec0 cosine distance = 1 - cosine_similarity
        return [(int(row[0]), max(0.0, 1.0 - row[1])) for row in rows]

    def delete(self, stream_name: str, key: int) -> None:
        if stream_name not in self._tables:
            return
        self._conn.execute(f'DELETE FROM "{stream_name}_vec" WHERE rowid = ?', (key,))
