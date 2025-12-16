//! Runtime utilities.
//! We provide a global Tokio runtime to back background tasks (e.g., echo handlers)
//! that may be invoked from synchronous Python calls. This avoids "no reactor"
//! panics when calling `tokio::spawn` without a runtime.

use once_cell::sync::Lazy;
use tokio::runtime::{Handle, Runtime};

pub static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("dimos-rs")
        .build()
        .expect("build global tokio runtime")
});

pub fn handle() -> Handle { RUNTIME.handle().clone() }

pub fn spawn<F>(fut: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    RUNTIME.spawn(fut)
}

#[allow(dead_code)]
pub fn configure_for_cuda_if_needed() {
    // TODO(cuda): If feature "cuda" is enabled, set CUDA device/context on the
    // calling thread or ensure worker threads are initialized appropriately.
    // This is where to bind streams/contexts or integrate with cust/cudarc.
}
