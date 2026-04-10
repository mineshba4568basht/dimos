# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Element types for Plot (2D charts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class Series:
    """Line connecting (t, y) points."""

    ts: list[float]
    values: list[float]
    color: str | None = None
    width: float = 1.5
    label: str | None = None
    axis: str | None = None
    opacity: float = 1.0


@dataclass
class Markers:
    """Scatter dots at (t, y) points."""

    ts: list[float]
    values: list[float]
    color: str | None = None
    radius: float = 0.5
    label: str | None = None
    axis: str | None = None
    opacity: float = 1.0


@dataclass
class HLine:
    """Horizontal reference line."""

    y: float
    color: str = "#888888"
    style: str = "dashed"
    label: str | None = None
    axis: str | None = None
    opacity: float = 1.0


PlotElement = Union[Series, Markers, HLine]
