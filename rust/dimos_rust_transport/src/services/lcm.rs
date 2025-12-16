use pyo3::prelude::*;

use crate::transport::inproc::{InprocPublisher, InprocSubscriber};
use crate::transport::{TransportPublisher, TransportSubscriber};

/// Rust equivalent of dimos/protocol/service/lcmservice.py:LCMService (scaffold)
/// 1:1 API intent: start/stop, request/response pattern.
/// Divergence: Uses in-process topics `service` and `service.reply` for demo.
#[pyclass]
pub struct LCMService {
    name: String,
    running: bool,
}

#[pymethods]
impl LCMService {
    #[new]
    pub fn new(name: String) -> Self { Self { name, running: false } }

    pub fn start(&mut self) { self.running = true; }
    pub fn stop(&mut self) { self.running = false; }
    pub fn is_running(&self) -> bool { self.running }

    /// Request-response: publish to `name`, await reply on `name.reply`.
    pub fn request<'py>(&self, py: Python<'py>, payload: &[u8]) -> PyResult<&'py PyAny> {
        if !self.running { return Err(pyo3::exceptions::PyRuntimeError::new_err("service not running")); }
        let req_topic = self.name.clone();
        let rep_topic = format!("{}.reply", self.name);
        let pub_req = InprocPublisher::new(req_topic);
        let mut sub_rep = InprocSubscriber::new(rep_topic);
        let data = payload.to_vec();
        pyo3_asyncio::tokio::future_into_py::<_, pyo3::PyObject>(py, async move {
            // Send request
            pub_req.publish(data).map_err(crate::to_pyerr)?;
            // Await reply
            let bytes = sub_rep.recv().await.map_err(crate::to_pyerr)?;
            Python::with_gil(|py| Ok(pyo3::types::PyBytes::new(py, &bytes).into_py(py)))
        })
    }

    /// Start a simple echo handler: listens on `name`, replies to `name.reply`.
    /// This mimics a trivial service method handler.
    pub fn start_echo_handler(&self) {
        let mut sub = InprocSubscriber::new(self.name.clone());
        let pub_rep = InprocPublisher::new(format!("{}.reply", self.name));
        crate::runtime::spawn(async move {
            loop {
                match sub.recv().await {
                    Ok(b) => { let _ = pub_rep.publish(b); }
                    Err(_) => break,
                }
            }
        });
    }
}
