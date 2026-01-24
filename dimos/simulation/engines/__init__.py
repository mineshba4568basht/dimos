"""Simulation engines for manipulator backends."""

from __future__ import annotations

import importlib

from dimos.simulation.engines.base import SimulationEngine

_ENGINE_REGISTRY: dict[str, str] = {
    "mujoco": "dimos.simulation.engines.mujoco_engine:MujocoEngine",
}


def get_engine(engine_name: str) -> type[SimulationEngine]:
    key = engine_name.lower()
    if key not in _ENGINE_REGISTRY:
        raise ValueError(f"Unknown simulation engine: {engine_name}")
    module_path, class_name = _ENGINE_REGISTRY[key].split(":")
    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    if not issubclass(engine_cls, SimulationEngine):
        raise TypeError(f"{engine_cls} is not a SimulationEngine")
    return engine_cls


__all__ = [
    "SimulationEngine",
    "get_engine",
]
