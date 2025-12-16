//! Shared-memory backend (scaffold).
//!
//! Notes:
//! - Define a ringbuffer or topic->queue mapping in POSIX SHM or memfd.
//! - For zero-copy CPU images: publish (fd, offset, len, shape, strides, dtype) metadata.
//! - Expose Python-side memoryviews/NumPy arrays via PyO3 buffer protocol by
//!   mapping shared memory into the Python process address space.
//! - For CUDA, couple with a device IPC handle or CUDA array interface.

#[allow(unused_imports)]
use super::{TransportPublisher, TransportSubscriber};

/// Placeholder types for future SHM implementation.
#[allow(dead_code)]
pub struct ShmPublisher {
    _topic: String,
}

#[allow(dead_code)]
pub struct ShmSubscriber {
    _topic: String,
}
