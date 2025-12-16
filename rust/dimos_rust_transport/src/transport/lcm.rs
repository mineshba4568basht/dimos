//! LCM backend (scaffold).
//!
//! Notes:
//! - Integrate the `lcm` C library via a Rust wrapper crate or bindgen.
//! - Implement `TransportPublisher` and `TransportSubscriber` by mapping to
//!   LCM publish/subscribe. Ensure background IO thread uses `tokio::task::spawn_blocking`
//!   if necessary, or native async if a non-blocking API is available.
//! - Consider message framing: we send raw bytes that can be pickle/json/bincode.

#[allow(unused_imports)]
use super::{TransportPublisher, TransportSubscriber};

/// Placeholder types for future LCM implementation.
#[allow(dead_code)]
pub struct LcmPublisher {
    _topic: String,
}

#[allow(dead_code)]
pub struct LcmSubscriber {
    _topic: String,
}
