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

"""Tests for Drawing builder and vis types."""

import numpy as np
import pytest

from dimos.memory2.type.observation import EmbeddedObservation, Observation
from dimos.memory2.vis.drawing.drawing import Drawing2D as Drawing
from dimos.memory2.vis.type import Arrow, Box3D, Camera, Point, Polyline, Pose, Text
from dimos.msgs.geometry_msgs.Point import Point as GeoPoint
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path as Path
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.vision_msgs.Detection3D import Detection3D


class TestVisTypes:
    """Vis types wrap msgs with rendering intent + style."""

    def test_pose_wraps_posestamped(self):
        ps = PoseStamped(3.2, 1.5, 0.0)
        p = Pose(ps, color="red", label="fridge")
        assert p.msg is ps
        assert p.color == "red"
        assert p.label == "fridge"

    def test_arrow_wraps_posestamped(self):
        ps = PoseStamped(1, 2, 0, 0, 0, 0.1, 1)
        a = Arrow(ps, color="orange", length=0.8)
        assert a.msg is ps
        assert a.length == 0.8

    def test_point_wraps_geopoint(self):
        gp = GeoPoint(7.1, 4.3, 0)
        p = Point(gp, color="green", label="bottle")
        assert p.msg is gp
        assert p.msg.x == pytest.approx(7.1)

    def test_point_wraps_posestamped(self):
        ps = PoseStamped(3, 1, 0)
        p = Point(ps, radius=0.5)
        assert p.msg.x == pytest.approx(3.0)

    def test_box3d_from_center_size(self):
        from dimos.msgs.geometry_msgs.Pose import Pose as GeoPose
        from dimos.msgs.geometry_msgs.Vector3 import Vector3

        b = Box3D(center=GeoPose(5, 3, 0), size=Vector3(2, 1, 0.5), label="table")
        assert b.center.x == pytest.approx(5.0)
        assert b.size.x == pytest.approx(2.0)
        assert b.label == "table"

    def test_camera_with_image(self):
        ps = PoseStamped(1, 2, 0)
        img = Image(np.zeros((480, 640, 3), dtype=np.uint8))
        c = Camera(pose=ps, image=img, color="purple")
        assert c.pose is ps
        assert c.image is img
        assert c.camera_info is None

    def test_text(self):
        t = Text((1, 8, 0), "exploration run #3")
        assert t.text == "exploration run #3"
        assert t.color == "#333333"


class TestDrawingExplicitVisTypes:
    """Drawing.add() with explicit vis types stores them as-is."""

    def test_add_pose(self):
        d = Drawing()
        ps = PoseStamped(3, 1, 0)
        pose = Pose(ps, color="red")
        d.add(pose)
        assert len(d) == 1
        assert d.elements[0] is pose

    def test_add_multiple_types(self):
        d = Drawing()
        ps = PoseStamped(3, 1, 0)
        d.add(Pose(ps, color="red"))
        d.add(Arrow(ps, color="orange"))
        d.add(Point(GeoPoint(1, 2, 0), label="x"))
        d.add(Text((0, 0, 0), "hello"))
        assert len(d) == 4

    def test_chaining(self):
        ps = PoseStamped(1, 1, 0)
        d = Drawing().add(Pose(ps)).add(Arrow(ps)).add(Text((0, 0, 0), "hi"))
        assert len(d) == 3


class TestDrawingAutoWrap:
    """Drawing.add() with raw dimos msgs auto-wraps into default vis type."""

    def test_posestamped_becomes_pose(self):
        d = Drawing()
        ps = PoseStamped(3.2, 1.5, 0)
        d.add(ps, color="blue", label="auto")
        assert len(d) == 1
        el = d.elements[0]
        assert isinstance(el, Pose)
        assert el.msg is ps
        assert el.color == "blue"
        assert el.label == "auto"

    def test_geopoint_becomes_point(self):
        d = Drawing()
        gp = GeoPoint(7, 4, 0)
        d.add(gp, color="yellow")
        el = d.elements[0]
        assert isinstance(el, Point)
        assert el.msg is gp
        assert el.color == "yellow"

    def test_path_becomes_polyline(self):
        d = Drawing()
        p = Path(poses=[PoseStamped(i, 0, 0) for i in range(3)])
        d.add(p, color="blue", width=0.1)
        el = d.elements[0]
        assert isinstance(el, Polyline)
        assert el.color == "blue"
        assert el.width == 0.1
        assert len(el.msg.poses) == 3

    def test_occupancy_grid_passthrough(self):
        d = Drawing()
        grid = OccupancyGrid()
        d.add(grid)
        assert d.elements[0] is grid

    def test_detection3d_becomes_box3d(self):
        det = Detection3D()
        det.bbox.center.position.x = 5.0
        det.bbox.center.position.y = 3.0
        det.bbox.size.x = 2.0
        det.bbox.size.y = 1.0
        det.bbox.size.z = 0.5

        d = Drawing()
        d.add(det, color="yellow")
        el = d.elements[0]
        assert isinstance(el, Box3D)
        assert el.center.position.x == pytest.approx(5.0)
        assert el.size.x == pytest.approx(2.0)
        assert el.color == "yellow"

    def test_unknown_type_raises(self):
        d = Drawing()
        with pytest.raises(TypeError, match="does not know how to handle"):
            d.add(42)


class TestDrawingObservations:
    """Drawing.add() smart dispatch for Observation types."""

    def test_image_observation_stored_as_observation(self):
        img = Image(np.zeros((480, 640, 3), dtype=np.uint8))
        obs = Observation(id=1, ts=1.0, pose=(3, 1, 0, 0, 0, 0, 1), _data=img)

        d = Drawing()
        d.add(obs)
        el = d.elements[0]
        assert isinstance(el, Observation)
        assert el.data is img

    def test_non_image_observation_stored_as_observation(self):
        obs = Observation(id=2, ts=2.0, pose=(5, 2, 0, 0, 0, 0, 1), _data="some_data")

        d = Drawing()
        d.add(obs)
        el = d.elements[0]
        assert isinstance(el, Observation)
        assert el.data == "some_data"

    def test_posestamped_observation_decomposed(self):
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped as PS

        obs = Observation(id=3, ts=3.0, pose=(1, 2, 0, 0, 0, 0, 1), _data=PS(5, 2, 0))

        d = Drawing()
        d.add(obs)
        el = d.elements[0]
        assert isinstance(el, Pose)
        assert el.msg.x == pytest.approx(5.0)

    def test_embedded_observation_stored_as_observation(self):
        obs = EmbeddedObservation(
            id=0,
            ts=0.0,
            pose=(1, 2, 0, 0, 0, 0, 1),
            _data="x",
            similarity=0.8,
        )

        d = Drawing()
        d.add(obs)
        assert len(d) == 1
        el = d.elements[0]
        assert isinstance(el, EmbeddedObservation)


class TestDrawingConvenience:
    """Drawing convenience methods: base_map."""

    def test_base_map(self):
        grid = OccupancyGrid()
        d = Drawing().base_map(grid)
        assert len(d) == 1
        assert isinstance(d.elements[0], OccupancyGrid)

    def test_add_list_of_msgs(self):
        poses = [PoseStamped(i, 0, 0) for i in range(3)]
        d = Drawing()
        d.add(poses, color="red")
        assert len(d) == 3
        for el in d.elements:
            assert isinstance(el, Pose)
            assert el.color == "red"


class TestDrawingRepr:
    def test_repr_empty(self):
        assert repr(Drawing()) == "Drawing2D()"

    def test_repr_with_elements(self):
        d = Drawing()
        ps = PoseStamped(0, 0, 0)
        d.add(Pose(ps))
        d.add(Pose(ps))
        d.add(Arrow(ps))
        assert repr(d) == "Drawing2D(Arrow=1, Pose=2)"


class TestSVGRender:
    """SVG rendering produces valid SVG with expected elements."""

    def test_empty_drawing(self):
        svg = Drawing().to_svg()
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")

    def test_point_renders_circle(self):
        d = Drawing()
        d.add(Point(GeoPoint(3, 4, 0), color="red", label="hi"))
        svg = d.to_svg()
        assert "<circle" in svg
        assert "red" in svg
        assert "hi" in svg

    def test_pose_renders_circle_and_arrow(self):
        d = Drawing()
        d.add(Pose(PoseStamped(1, 2, 0), color="blue"))
        svg = d.to_svg()
        assert "<circle" in svg
        assert "<line" in svg
        assert "blue" in svg

    def test_arrow_renders_line(self):
        d = Drawing()
        d.add(Arrow(PoseStamped(0, 0, 0, 0, 0, 0.38, 0.92), color="orange"))
        svg = d.to_svg()
        assert "<line" in svg
        assert "orange" in svg

    def test_polyline_renders(self):
        d = Drawing()
        d.add(
            Polyline(
                msg=Path(poses=[PoseStamped(i, i * 0.5, 0) for i in range(5)]),
                color="blue",
            )
        )
        svg = d.to_svg()
        assert "<polyline" in svg

    def test_box3d_renders_rect(self):
        from dimos.msgs.geometry_msgs.Pose import Pose as GeoPose
        from dimos.msgs.geometry_msgs.Vector3 import Vector3

        d = Drawing()
        d.add(Box3D(center=GeoPose(5, 3, 0), size=Vector3(2, 1, 0), label="table"))
        svg = d.to_svg()
        assert "<rect" in svg
        assert "table" in svg

    def test_text_renders(self):
        d = Drawing()
        d.add(Text((1, 1, 0), "hello <world>"))
        svg = d.to_svg()
        assert "<text" in svg
        assert "hello &lt;world&gt;" in svg

    def test_camera_without_info_renders_dot(self):
        d = Drawing()
        d.add(Camera(pose=PoseStamped(1, 2, 0), color="purple"))
        svg = d.to_svg()
        assert "<circle" in svg
        assert "purple" in svg

    def test_occupancy_grid_renders_image(self):
        grid = OccupancyGrid(
            grid=np.zeros((10, 10), dtype=np.int8),
            resolution=0.1,
        )
        d = Drawing().base_map(grid)
        svg = d.to_svg()
        assert "<image" in svg
        assert "data:image/png;base64," in svg

    def test_mixed_drawing(self):
        d = Drawing()
        ps = PoseStamped(3, 1, 0)
        d.add(Pose(ps, color="red", label="robot"))
        d.add(Arrow(ps, color="orange"))
        d.add(Point(GeoPoint(5, 5, 0), color="green", label="goal"))
        d.add(Text((0, 0, 0), "test"))
        svg = d.to_svg()
        assert svg.count("<circle") == 2  # pose dot + point dot
        assert "<text" in svg
        assert "<line" in svg

    def test_to_svg_writes_file(self, tmp_path):
        d = Drawing()
        d.add(Point(GeoPoint(1, 1, 0)))
        out = tmp_path / "test.svg"
        d.to_svg(str(out))
        assert out.exists()
        assert "<svg" in out.read_text()
