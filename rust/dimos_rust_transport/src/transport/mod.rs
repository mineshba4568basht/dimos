pub mod inproc;

#[allow(dead_code)]
pub mod lcm;

#[allow(dead_code)]
pub mod shm;

use std::future::Future;

/// Common transport trait for publish/subscribe. Backends (LCM, SHM) implement this.
pub trait TransportPublisher: Send + Sync + 'static {
    fn publish(&self, bytes: Vec<u8>) -> anyhow::Result<()>;
}

pub trait TransportSubscriber: Send + Sync + 'static {
    fn try_recv(&mut self) -> anyhow::Result<Option<Vec<u8>>>;
    fn recv<'a>(&'a mut self) -> impl Future<Output = anyhow::Result<Vec<u8>>> + Send + 'a;
}
