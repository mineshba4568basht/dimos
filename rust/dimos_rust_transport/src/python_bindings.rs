use pyo3::prelude::*;

use crate::services::lcm::LCMService;
use crate::transports::lcm::{LCMSubscription, LCMTransport};
use crate::transports::shm::{SHMSubscription, SHMTransport};

/// Register nested modules and classes to align with Python layout expectations.
/// We provide submodules: `transports.lcm`, `transports.shm`, and `services.lcm`.
/// For true drop-in replacement (e.g. dimos.protocol.pubsub.lcmpubsub), add thin
/// Python shims that re-export these classes from this extension.
pub fn register(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Create `transports` package module
    let transports = PyModule::new(py, "transports")?;

    // lcm submodule
    let lcm_mod = PyModule::new(py, "lcm")?;
    lcm_mod.add_class::<LCMTransport>()?;
    lcm_mod.add_class::<LCMSubscription>()?;
    transports.add_submodule(lcm_mod)?;

    // shm submodule
    let shm_mod = PyModule::new(py, "shm")?;
    shm_mod.add_class::<SHMTransport>()?;
    shm_mod.add_class::<SHMSubscription>()?;
    transports.add_submodule(shm_mod)?;

    m.add_submodule(transports)?;

    // services package
    let services = PyModule::new(py, "services")?;
    let lcm_service = PyModule::new(py, "lcm")?;
    lcm_service.add_class::<LCMService>()?;
    services.add_submodule(lcm_service)?;
    m.add_submodule(services)?;

    Ok(())
}

