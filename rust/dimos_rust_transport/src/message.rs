use serde::{Deserialize, Serialize};

/// Generic message wrapper for strongly-typed messages.
/// Extend by defining your own `T: Serialize + for<'de> Deserialize<'de>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message<T> {
    pub topic: String,
    pub seq: u64,
    pub ts_nanos: u128,
    pub data: T,
}

impl<T> Message<T> {
    pub fn new(topic: String, seq: u64, ts_nanos: u128, data: T) -> Self {
        Self { topic, seq, ts_nanos, data }
    }
}

// Notes for future extensions:
// - For zero-copy raw image buffers: prefer a `Message<Bytes>`-like type that
//   references external memory (e.g., shm region or GPU device pointer) and
//   carries metadata (shape, strides, dtype). For Python interop, expose a
//   `memoryview` or NumPy array via PyO3 buffer protocol. For CUDA, use a
//   feature gate to manage device pointers and lifetime.
