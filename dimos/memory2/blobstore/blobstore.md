# blobstore/

Separates payload blob storage from metadata indexing. Observation payloads vary hugely in size — a `Vector3` is 24 bytes, a camera frame is megabytes. Storing everything inline penalizes metadata queries. BlobStore lets large payloads live elsewhere.

## ABC (`backend.py`)

```python
class BlobStore(Resource, ABC):
    def put(self, stream: str, key: int, data: bytes) -> None: ...
    def get(self, stream: str, key: int) -> bytes: ...    # raises KeyError if missing
    def delete(self, stream: str, key: int) -> None: ...  # silent if missing
```

- `stream` — stream name (used to organize storage: directories, tables)
- `key` — observation id
- `data` — encoded payload bytes (codec handles serialization, blob store handles persistence)
- Extends `Resource` (start/stop) but does NOT own its dependencies' lifecycle

## Implementations

### `file.py` — FileBlobStore

Stores blobs as files on disk, one directory per stream.

```
{root}/{stream}/{key}.bin
```

`__init__(root: str | os.PathLike[str])` — `start()` creates the root directory.

### `sqlite.py` — SqliteBlobStore

Stores blobs in a separate SQLite table per stream.

```sql
CREATE TABLE "{stream}_blob" (id INTEGER PRIMARY KEY, data BLOB NOT NULL)
```

`__init__(conn: sqlite3.Connection)` — does NOT own the connection.

**Internal use** (same db as metadata): `SqliteStore.session()` creates one connection, passes it to both the metadata backend and the blob store.

**External use** (separate db): user creates a separate connection and passes it. User manages that connection's lifecycle.

**JOIN optimization** (future): when `lazy=False` and the blob store shares the same connection as the metadata backend, `SqliteBackend` can optimize with a JOIN instead of separate queries:

```sql
SELECT m.id, m.ts, m.pose, m.tags, b.data
FROM "images" m JOIN "images_blob" b ON m.id = b.id
WHERE m.ts > ?
```

## Lazy loading

`lazy` is a stream-level flag, orthogonal to blob store choice. It controls WHEN data is loaded:

- `lazy=False` → backend loads payload during iteration (eager)
- `lazy=True` → backend sets `Observation._loader`, payload loaded on `.data` access

| lazy | blob store | loading strategy |
|------|-----------|-----------------|
| False | SqliteBlobStore (same conn) | JOIN — one round trip |
| False | any other | iterate meta, `blob_store.get()` per row |
| True | any | iterate meta only, `_loader = lambda: codec.decode(blob_store.get(...))` |

## Usage

```python
# Per-stream blob store choice
with store.session() as session:
    poses = session.stream("poses", PoseStamped)                           # default, eager
    images = session.stream("images", Image, lazy=True)                    # default, lazy
    images = session.stream("images", Image, blob_store=file_blobs)        # override
```

## Files

```
backend.py            BlobStore ABC (alongside Backend, LiveBackend)
blobstore/
  blobstore.md        this file
  __init__.py          re-exports BlobStore, FileBlobStore, SqliteBlobStore
  file.py             FileBlobStore
  sqlite.py           SqliteBlobStore
  test_blobstore.py   grid tests across implementations
```
