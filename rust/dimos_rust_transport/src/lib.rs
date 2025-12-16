mod message;
mod runtime;
mod transport;
mod transports;
mod services;
mod python_bindings;

use pyo3::prelude::*;

use crate::transport::inproc::{InprocPublisher, InprocSubscriber};
use crate::transport::{TransportPublisher, TransportSubscriber};

/// PyO3 module entry. We keep existing helpers (Publisher/Subscriber) and also
/// register transport/service bindings under nested submodules via
/// `python_bindings::register`.
#[pymodule]
fn dimos_rust_transport(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Initialize pyo3-asyncio with a multi-threaded Tokio runtime.
    // For pyo3-asyncio 0.20, explicit init is optional; futures created with
    // `future_into_py` will bootstrap a runtime as needed.

    // Existing simple pub/sub helpers
    m.add_class::<PyPublisher>()?;
    m.add_class::<PySubscriber>()?;

    // Helper to create a publisher
    #[pyfn(m)]
    fn create_publisher(_py: Python<'_>, topic: String) -> PyResult<PyPublisher> {
        Ok(PyPublisher::new(topic))
    }

    // Helper to create a subscriber
    #[pyfn(m)]
    fn create_subscriber(_py: Python<'_>, topic: String) -> PyResult<PySubscriber> {
        Ok(PySubscriber::new(topic))
    }

    // Start an echo task: subscribe to `topic`, publish payloads to `topic + .echo`.
    #[pyfn(m)]
    fn start_echo(_py: Python<'_>, topic: String) -> PyResult<()> {
        transport::inproc::start_echo_task(topic);
        Ok(())
    }

    // Register new transports/services API surfaces.
    python_bindings::register(py, m)?;

    Ok(())
}

#[pyclass]
struct PyPublisher {
    topic: String,
    inner: InprocPublisher,
}

#[pymethods]
impl PyPublisher {
    #[new]
    fn new(topic: String) -> Self {
        let inner = InprocPublisher::new(topic.clone());
        Self { topic, inner }
    }

    fn topic(&self) -> String { self.topic.clone() }

    /// Publish raw bytes as-is.
    fn publish_bytes(&self, data: &[u8]) -> PyResult<()> {
        self.inner.publish(data.to_vec()).map_err(to_pyerr)
    }

    /// Publish a Python object serialized via `serializer` ("pickle" | "json" | "bincode").
    /// - pickle: Python pickle.dumps(obj) bytes
    /// - json: Python json.dumps(obj) UTF-8 bytes
    /// - bincode: bincode(String) where String is Python json.dumps(obj)
    fn publish_py(&self, py: Python<'_>, obj: &PyAny, serializer: Option<&str>) -> PyResult<()> {
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
                bincode::serialize(&s).map_err(to_pyerr)?
            }
            other => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown serializer: {}", other))),
        };
        self.inner.publish(bytes).map_err(to_pyerr)
    }
}

#[pyclass]
struct PySubscriber {
    topic: String,
    inner: InprocSubscriber,
}

#[pymethods]
impl PySubscriber {
    #[new]
    fn new(topic: String) -> Self {
        let inner = InprocSubscriber::new(topic.clone());
        Self { topic, inner }
    }

    fn topic(&self) -> String { self.topic.clone() }

    /// Try to receive without blocking. Returns None if no message available.
    fn try_recv_bytes(&mut self) -> PyResult<Option<Vec<u8>>> {
        match self.inner.try_recv() {
            Ok(Some(b)) => Ok(Some(b)),
            Ok(None) => Ok(None),
            Err(e) => Err(to_pyerr(e)),
        }
    }

    /// Async receive as Python bytes.
    fn recv_bytes<'py>(&mut self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let mut rx = self.inner.cloned();
        pyo3_asyncio::tokio::future_into_py::<_, pyo3::PyObject>(py, async move {
            let bytes = rx.recv().await.map_err(to_pyerr)?;
            Python::with_gil(|py| Ok(pyo3::types::PyBytes::new(py, &bytes).into_py(py)))
        })
    }

    /// Async receive and deserialize using `serializer` ("pickle" | "json" | "bincode").
    fn recv_py<'py>(&mut self, py: Python<'py>, serializer: Option<&str>) -> PyResult<&'py PyAny> {
        let mut rx = self.inner.cloned();
        let serializer = serializer.unwrap_or("pickle").to_string();
        pyo3_asyncio::tokio::future_into_py::<_, pyo3::PyObject>(py, async move {
            let bytes = rx.recv().await.map_err(to_pyerr)?;
            Python::with_gil(|py| {
                match serializer.as_str() {
                    "pickle" => {
                        let pickle = py.import("pickle")?;
                        pickle.call_method1("loads", (pyo3::types::PyBytes::new(py, &bytes),))
                            .map(|o| o.into_py(py))
                    }
                    "json" => {
                        let json = py.import("json")?;
                        let s = std::str::from_utf8(&bytes).map_err(to_pyerr)?;
                        json.call_method1("loads", (s,)).map(|o| o.into_py(py))
                    }
                    "bincode" => {
                        let json = py.import("json")?;
                        let s: String = bincode::deserialize(&bytes).map_err(to_pyerr)?;
                        json.call_method1("loads", (s,)).map(|o| o.into_py(py))
                    }
                    other => Err(pyo3::exceptions::PyValueError::new_err(format!("unknown serializer: {}", other))),
                }
            })
        })
    }
}

pub fn to_pyerr<E: std::fmt::Display>(e: E) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}
