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

"""Matplotlib-based SVG renderer for Plot."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt

from dimos.memory2.vis.color import palette_iter
from dimos.memory2.vis.plot.elements import HLine, Markers, Series

if TYPE_CHECKING:
    from dimos.memory2.vis.plot.plot import Plot

matplotlib.use("Agg")


def render(plot: Plot, width: float = 10, height: float = 3.5) -> str:
    """Render a Plot to an SVG string via matplotlib."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("#16213e")
        ax.grid(True, color="#2a2a4a", linewidth=0.5)

        # Lazily create twin y-axes for any element with axis != None.
        # All twins share the primary x-axis (matplotlib `ax.twinx()`).
        axes: dict[str | None, Any] = {None: ax}

        def axis_for(name: str | None) -> Any:
            if name not in axes:
                twin = ax.twinx()
                twin.set_facecolor("none")
                axes[name] = twin
            return axes[name]

        # Drive a single shared color cycle across all axes (primary + twins)
        # so series on a twin don't reuse the primary's first color. Excludes
        # any color the user has already pinned to a specific element so the
        # auto-cycle won't double-up on it.
        explicit_colors = {
            el.color
            for el in plot.elements
            if isinstance(el, (Series, Markers)) and el.color is not None
        }
        color_iter = palette_iter(exclude=explicit_colors)

        for el in plot.elements:
            target = axis_for(el.axis)
            color = el.color
            if color is None and isinstance(el, (Series, Markers)):
                color = next(color_iter)
            if isinstance(el, Series):
                target.plot(
                    el.ts,
                    el.values,
                    color=color,
                    linewidth=el.width,
                    label=el.label,
                    alpha=el.opacity,
                )
            elif isinstance(el, Markers):
                target.scatter(
                    el.ts,
                    el.values,
                    color=color,
                    s=el.radius**2 * 10,
                    label=el.label,
                    alpha=el.opacity,
                )
            elif isinstance(el, HLine):
                style = "--" if el.style == "dashed" else "-"
                target.axhline(
                    el.y,
                    color=el.color,
                    linestyle=style,
                    linewidth=1,
                    label=el.label,
                    alpha=el.opacity,
                )

        # Combine handles from all axes into a single legend. Attach it to the
        # *last* axes created (the most recent twin, or the primary if there
        # are no twins) so the legend paints last and isn't covered by twin
        # tick labels / spines drawn afterward in matplotlib's axes draw order.
        all_handles: list[Any] = []
        all_labels: list[str] = []
        for axes_obj in axes.values():
            h, l = axes_obj.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)
        if all_handles:
            legend_host = next(reversed(axes.values()))
            legend_host.legend(
                all_handles,
                all_labels,
                facecolor="#1a1a2e",
                edgecolor="#2a2a4a",
                framealpha=0.9,
            )

        ax.set_xlabel("time (s)")
        fig.tight_layout()

        buf = io.StringIO()
        fig.savefig(buf, format="svg")
        plt.close(fig)

        return buf.getvalue()
