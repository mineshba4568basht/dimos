use pyo3::prelude::*;

use crate::transport::inproc::{InprocPublisher, InprocSubscriber};
use crate::transport::{TransportPublisher, TransportSubscriber};

/// Rust equivalent of Python shmtransport.py:SHMTransport (scaffold)
/// 1:1 API intent: publish/subscribe/close.
/// Divergence: Uses in-process bus; real SHM would expose zero-copy buffers.
#[pyclass]
pub struct SHMTransport {
    closed: bool,
}

#[pymethods]
impl SHMTransport {
    #[new]
    pub fn new(_name: Option<String>) -> Self { Self { closed: false } }

    pub fn publish(&self, topic: String, data: &[u8]) -> PyResult<()> {
        if self.closed { return Err(pyo3::exceptions::PyRuntimeError::new_err("transport closed")); }
        let p = InprocPublisher::new(topic);
        p.publish(data.to_vec()).map_err(crate::to_pyerr)
    }

    /// Publish a Python object with selectable serializer: "pickle" | "json" | "bincode".
    pub fn publish_py(&self, py: Python<'_>, topic: String, obj: &PyAny, serializer: Option<&str>) -> PyResult<()> {
        if self.closed { return Err(pyo3::exceptions::PyRuntimeError::new_err("transport closed")); }
        let serializer = serializer.unwrap_or("pickle");
        let bytes = match serializer {
            "pickle" => {
                let pickle = py.import("pickle")?;
                let b: pyo3::Py<pyo3::types::PyBytes> = pickle.call_method1("dumps", (obj,))?.extract()?;
                b.as_ref(py).as_bytes().to_vec()
            }
            "json" => {
                let json = py.import("json")?;
                let s: String = json.call_method1("dumps", (obj,))?.extract()?;
                s.into_bytes()
            }
            "bincode" => {
                let json = py.import("json")?;
                let s: String = json.call_method1("dumps", (obj,))?.extract()?;
                bincode::serialize(&s).map_err(crate::to_pyerr)?
            }
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown serializer: {}", other))),
        };
        let p = InprocPublisher::new(topic);
        p.publish(bytes).map_err(crate::to_pyerr)
    }

    pub fn subscribe(&self, topic: String) -> PyResult<SHMSubscription> {
        if self.closed { return Err(pyo3::exceptions::PyRuntimeError::new_err("transport closed")); }
        Ok(SHMSubscription { topic: topic.clone(), inner: InprocSubscriber::new(topic), closed: false })
    }

    pub fn close(&mut self) { self.closed = true; }
}

#[pyclass]
pub struct SHMSubscription {
    topic: String,
    inner: InprocSubscriber,
    closed: bool,
}

#[pymethods]
impl SHMSubscription {
    pub fn topic(&self) -> String { self.topic.clone() }

    pub fn try_recv_bytes(&mut self) -> PyResult<Option<Vec<u8>>> {
        if self.closed { return Ok(None); }
        self.inner.try_recv().map_err(crate::to_pyerr)
    }

    pub fn recv_bytes<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyAny> {
        if self.closed { return Err(pyo3::exceptions::PyRuntimeError::new_err("subscription closed")); }
        let mut rx = self.inner.cloned();
        pyo3_asyncio::tokio::future_into_py::<_, pyo3::PyObject>(py, async move {
            let b = rx.recv().await.map_err(crate::to_pyerr)?;
            Python::with_gil(|py| Ok(pyo3::types::PyBytes::new(py, &b).into_py(py)))
        })
    }

    /// Async receive and deserialize using serializer: "pickle" | "json" | "bincode".
    pub fn recv_py<'py>(&mut self, py: Python<'py>, serializer: Option<&str>) -> PyResult<&'py PyAny> {
        if self.closed { return Err(pyo3::exceptions::PyRuntimeError::new_err("subscription closed")); }
        let mut rx = self.inner.cloned();
        let serializer = serializer.unwrap_or("pickle").to_string();
        pyo3_asyncio::tokio::future_into_py::<_, pyo3::PyObject>(py, async move {
            let bytes = rx.recv().await.map_err(crate::to_pyerr)?;
            Python::with_gil(|py| {
                match serializer.as_str() {
                    "pickle" => {
                        let pickle = py.import("pickle")?;
                        pickle.call_method1("loads", (pyo3::types::PyBytes::new(py, &bytes),))
                            .map(|o| o.into_py(py))
                    }
                    "json" => {
                        let json = py.import("json")?;
                        let s = std::str::from_utf8(&bytes).map_err(crate::to_pyerr)?;
                        json.call_method1("loads", (s,)).map(|o| o.into_py(py))
                    }
                    "bincode" => {
                        let json = py.import("json")?;
                        let s: String = bincode::deserialize(&bytes).map_err(crate::to_pyerr)?;
                        json.call_method1("loads", (s,)).map(|o| o.into_py(py))
                    }
                    other => Err(pyo3::exceptions::PyValueError::new_err(format!("unknown serializer: {}", other))),
                }
            })
        })
    }

    pub fn unsubscribe(&mut self) { self.closed = true; }
}
