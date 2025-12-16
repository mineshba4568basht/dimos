# dimos_rust_transport (prototype)

Rust <-> Python interop layer for pub/sub transports (LCM/SHM), built with PyO3 and async Tokio. This crate currently ships an in-process bus for development and tests, and stubs for LCM and shared-memory backends.

Key goals:
- Idiomatic Rust API (`Publisher`, `Subscriber`, `Message<T>` via serde).
- Python bindings via PyO3 with async-friendly `recv_*` coroutines.
- Serialization: serde+bincode or JSON on Rust side; pickle/JSON/bincode from Python.
- Extensible for zero-copy CPU/GPU image buffers.

## Layout
- `src/transport/inproc.rs`: in-process broadcast bus used for examples/tests.
- `src/transport/lcm.rs`: scaffold for LCM backend implementation.
- `src/transport/shm.rs`: scaffold for shared-memory backend implementation.
- `src/message.rs`: generic typed message wrapper and notes for zero-copy.
- `src/lib.rs`: PyO3 module, exposes `PyPublisher`/`PySubscriber` and helpers.
- `src/runtime.rs`: placeholder for CUDA runtime configuration.

## Build (development)
Requirements: Rust toolchain, Python 3.8+, `maturin`.

```
# from repo root
python3 -m venv venv && source venv/bin/activate
pip install maturin

# build and develop-install the extension into current venv
maturin develop -m rust/dimos_rust_transport/pyproject.toml
```

Run tests:
```
pytest -q tests/test_rust_transport_roundtrip.py -s
```

## Python usage
```python
import asyncio
import dimos_rust_transport as rt

rt.start_echo("demo")
pub = rt.create_publisher("demo")
sub = rt.create_subscriber("demo.echo")

pub.publish_py({"a": 1}, serializer="json")
print(asyncio.run(sub.recv_py(serializer="json")))
```

Alternatively via `dimos.stream.rust_transport` thin wrapper:
```python
from dimos.stream import rust_transport as rt
```

## Extending: LCM backend
- Bind the LCM C library (via an existing crate or `bindgen`).
- Implement `TransportPublisher`/`TransportSubscriber` in `src/transport/lcm.rs`.
- Use a background IO thread if the API is blocking; integrate with Tokio using `spawn_blocking()` or sockets with non-blocking mode.
- Keep payload as `Vec<u8>`; Python controls serialization (`pickle`/`json`/`bincode`).

## Extending: Shared Memory (CPU)
- Design a topic->ringbuffer layout with POSIX SHM/memfd and futex/eventfd for signaling.
- Publish metadata (shape, strides, dtype) alongside offsets/lengths.
- For Python zero-copy, export a `memoryview`/NumPy array using PyO3 buffer protocol.

## Extending: CUDA (GPU)
- Under `--features cuda`, add CUDA context/stream setup in `runtime.rs`.
- For zero-copy GPU: exchange CUDA IPC mem handles or use DLPack/CUDA array interface.
- Python side can wrap device pointers via CuPy/PyTorch without copying.

## Custom message types
- Define `#[derive(Serialize, Deserialize)] struct MyMsg { ... }`.
- Use `Message<MyMsg>` for typed Rust APIs; serialize with bincode/JSON for transport.
- When exposing to Python, convert via `pyo3-serde` or provide dedicated schemas.

## Concurrency notes
- The Py API is async-friendly using `pyo3-asyncio` (Tokio runtime).
- `Publisher`/`Subscriber` are `Send + Sync` (clonable) and safe across tasks.

## Roadmap
- Swap in `lcm` and `shm` backends behind feature flags.
- Topic wildcards, QoS, backpressure metrics, and structured tracing.
- Benchmarks for throughput/latency across CPU/GPU paths.

