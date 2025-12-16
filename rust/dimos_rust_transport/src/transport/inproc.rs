use std::collections::HashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use tokio::sync::broadcast::{self, Receiver, Sender};

use super::{TransportPublisher, TransportSubscriber};

static BUS: Lazy<Bus> = Lazy::new(Bus::new);

struct Bus {
    map: RwLock<HashMap<String, Sender<Vec<u8>>>>,
}

impl Bus {
    fn new() -> Self { Self { map: RwLock::new(HashMap::new()) } }

    fn get_or_create(&self, topic: &str) -> Sender<Vec<u8>> {
        if let Some(tx) = self.map.read().get(topic).cloned() {
            return tx;
        }
        let mut w = self.map.write();
        w.entry(topic.to_string())
            .or_insert_with(|| broadcast::channel::<Vec<u8>>(1024).0)
            .clone()
    }
}

pub struct InprocPublisher {
    topic: String,
    tx: Sender<Vec<u8>>,
}

impl InprocPublisher {
    pub fn new(topic: String) -> Self {
        let tx = BUS.get_or_create(&topic);
        Self { topic, tx }
    }
}

impl TransportPublisher for InprocPublisher {
    fn publish(&self, bytes: Vec<u8>) -> anyhow::Result<()> {
        let _ = self.tx.send(bytes);
        Ok(())
    }
}

pub struct InprocSubscriber {
    topic: String,
    rx: Receiver<Vec<u8>>,
}

impl InprocSubscriber {
    pub fn new(topic: String) -> Self {
        let rx = BUS.get_or_create(&topic).subscribe();
        Self { topic, rx }
    }

    pub fn cloned(&self) -> Self {
        // broadcast::Receiver is not Clone; resubscribe on the topic instead.
        let rx = BUS.get_or_create(&self.topic).subscribe();
        Self { topic: self.topic.clone(), rx }
    }
}

impl TransportSubscriber for InprocSubscriber {
    fn try_recv(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
        match self.rx.try_recv() {
            Ok(v) => Ok(Some(v)),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => Ok(None),
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {
                // Skip ahead on lag.
                Ok(None)
            }
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => Ok(None),
        }
    }

    async fn recv<'a>(&'a mut self) -> anyhow::Result<Vec<u8>> {
        loop {
            match self.rx.recv().await {
                Ok(v) => return Ok(v),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    return Err(anyhow::anyhow!("channel closed"))
                }
            }
        }
    }
}

/// Spawn an in-process echo task. Subscribes to `topic`, republishes payloads to `topic + ".echo"`.
pub fn start_echo_task(topic: String) {
    let mut sub = InprocSubscriber::new(topic.clone());
    let pub_echo = InprocPublisher::new(format!("{}.echo", topic));
    crate::runtime::spawn(async move {
        loop {
            match sub.recv().await {
                Ok(bytes) => {
                    let _ = pub_echo.publish(bytes);
                }
                Err(_) => break,
            }
        }
    });
}
