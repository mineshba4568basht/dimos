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

"""Tests for Plot builder and SVG rendering."""

import pytest

from dimos.memory2.type.observation import Observation
from dimos.memory2.vis.plot.elements import HLine, Markers, Series
from dimos.memory2.vis.plot.plot import Plot


class TestPlotAdd:
    """Plot.add() smart dispatch."""

    def test_add_series(self):
        p = Plot()
        s = Series(ts=[1, 2, 3], values=[10, 20, 30], label="speed")
        p.add(s)
        assert len(p) == 1
        assert p.elements[0] is s

    def test_add_markers(self):
        p = Plot()
        m = Markers(ts=[1, 2], values=[5, 10], color="red")
        p.add(m)
        assert len(p) == 1
        assert isinstance(p.elements[0], Markers)

    def test_add_hline(self):
        p = Plot()
        p.add(HLine(y=0.5, label="threshold"))
        assert len(p) == 1
        assert p.elements[0].y == 0.5

    def test_add_from_observation_list(self):
        obs_list = [
            Observation(id=i, ts=float(i), pose=(i, 0, 0, 0, 0, 0, 1), _data=float(i * 10))
            for i in range(5)
        ]
        p = Plot()
        p.add(obs_list, label="test", color="blue")
        assert len(p) == 1
        el = p.elements[0]
        assert isinstance(el, Series)
        assert el.ts == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert el.values == [0.0, 10.0, 20.0, 30.0, 40.0]
        assert el.label == "test"
        assert el.color == "blue"

    def test_add_chaining(self):
        p = Plot().add(Series(ts=[1, 2], values=[10, 20])).add(HLine(y=15))
        assert len(p) == 2

    def test_add_unknown_type_raises(self):
        p = Plot()
        with pytest.raises(TypeError, match="does not know how to handle"):
            p.add(42)


class TestPlotSVG:
    """SVG rendering via matplotlib."""

    def test_empty_plot(self):
        svg = Plot().to_svg()
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_series_renders(self):
        p = Plot()
        p.add(Series(ts=[0, 1, 2, 3], values=[0, 1, 4, 9], label="y=x²"))
        svg = p.to_svg()
        assert "<svg" in svg

    def test_mixed_elements(self):
        p = Plot()
        p.add(Series(ts=[0, 1, 2], values=[10, 20, 30], label="speed"))
        p.add(Markers(ts=[0.5, 1.5], values=[15, 25], label="events"))
        p.add(HLine(y=20, label="limit"))
        svg = p.to_svg()
        assert "<svg" in svg

    def test_to_svg_writes_file(self, tmp_path):
        p = Plot()
        p.add(Series(ts=[0, 1], values=[0, 1]))
        out = tmp_path / "test.svg"
        p.to_svg(str(out))
        assert out.exists()
        assert "<svg" in out.read_text()

    def test_default_axis_is_shared(self):
        p = Plot()
        p.add(Series(ts=[0, 1, 2], values=[10, 20, 30], label="a"))
        p.add(Series(ts=[0, 1, 2], values=[15, 25, 35], label="b"))
        svg = p.to_svg()
        assert "<svg" in svg

    def test_named_axis_creates_twin(self):
        p = Plot()
        p.add(Series(ts=[0, 1, 2], values=[10, 20, 30], label="speed (m/s)"))
        p.add(
            Series(
                ts=[0, 1, 2],
                values=[100, 200, 300],
                label="elapsed (s)",
                axis="right",
            )
        )
        svg = p.to_svg()
        assert "<svg" in svg
        # Both labels should appear in the merged legend
        assert "speed" in svg
        assert "elapsed" in svg

    def test_auto_color_cycle_uses_theme_palette(self):
        # First two theme colors are blue then red. Without our shared cycle,
        # the twin-axis series would reuse the primary's first color.
        p = Plot()
        p.add(Series(ts=[0, 1, 2], values=[10, 20, 30]))
        p.add(Series(ts=[0, 1, 2], values=[100, 200, 300], axis="right"))
        svg = p.to_svg()
        assert "#3498db" in svg  # blue (first theme color, primary axis)
        assert "#e74c3c" in svg  # red  (second theme color, twin axis)

    def test_opacity_appears_in_svg(self):
        # opacity=0.4 should land as opacity="0.4" on the matplotlib-rendered
        # path. matplotlib emits it as either `opacity` or `stroke-opacity`
        # depending on the artist; we just need to see the value in the output.
        p = Plot()
        p.add(Series(ts=[0, 1, 2], values=[0, 1, 2], opacity=0.4))
        svg = p.to_svg()
        assert "0.4" in svg

    def test_explicit_color_excluded_from_auto_cycle(self):
        # If the user pins a Series to color.red, the auto-cycle for the next
        # series should skip red and yield yellow (the third color) instead
        # of red — otherwise we'd get two red lines.
        from dimos.memory2.vis import color

        p = Plot()
        p.add(Series(ts=[0, 1], values=[0, 1]))  # auto → blue (first)
        p.add(Series(ts=[0, 1], values=[2, 3], color=color.red))  # explicit red
        p.add(Series(ts=[0, 1], values=[4, 5]))  # auto → yellow (red is excluded)
        svg = p.to_svg()
        # Both blue and yellow should appear, plus the explicit red.
        assert color.blue in svg
        assert color.red in svg
        assert color.yellow in svg
        # Red should appear exactly once (the explicit one, not from the cycle).
        assert svg.count(color.red) == 1


class TestPlotRepr:
    def test_repr_empty(self):
        assert repr(Plot()) == "Plot()"

    def test_repr_with_elements(self):
        p = Plot()
        p.add(Series(ts=[0], values=[0]))
        p.add(Series(ts=[0], values=[0]))
        p.add(HLine(y=1))
        assert repr(p) == "Plot(HLine=1, Series=2)"


class TestPlotRerunStub:
    """Plot.to_rerun() is currently a no-op placeholder — must not raise."""

    def test_to_rerun_does_not_raise(self):
        Plot().to_rerun()


class TestPalette:
    """The named palette and palette_iter live in vis/color.py."""

    def test_named_constants_exist(self):
        from dimos.memory2.vis import color

        assert color.blue == "#3498db"
        assert color.red == "#e74c3c"
        assert color.green.startswith("#")
        assert color.amber.startswith("#")
        assert len(color.PALETTE) == 12
        assert color.PALETTE[0] == color.blue
        assert color.PALETTE[6] == color.green

    def test_palette_iter_yields_palette_first(self):
        from dimos.memory2.vis import color

        it = color.palette_iter()
        assert [next(it) for _ in range(12)] == color.PALETTE

    def test_palette_iter_continues_past_palette(self):
        from dimos.memory2.vis import color

        it = color.palette_iter()
        first_thirteen = [next(it) for _ in range(13)]
        # 13th color is generated, must be a hex string and distinct from all 12 named.
        assert first_thirteen[12].startswith("#")
        assert first_thirteen[12] not in color.PALETTE

    def test_palette_iter_excludes(self):
        from dimos.memory2.vis import color

        it = color.palette_iter(exclude={color.red, color.yellow})
        first_three = [next(it) for _ in range(3)]
        # Skipped red and yellow, so the first three are blue, teal, purple.
        assert first_three == [color.blue, color.teal, color.purple]
