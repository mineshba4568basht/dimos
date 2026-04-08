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

"""Rerun renderer for Drawing2D. Logs scene elements as 3D archetypes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from dimos.memory2.type.observation import Observation
from dimos.memory2.vis.color import hex_to_rgb
from dimos.memory2.vis.type import Arrow, Box3D, Camera, Point, Polyline, Pose, Text
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

if TYPE_CHECKING:
    from dimos.memory2.vis.drawing.drawing import Drawing2D

# base_link → camera_optical extrinsics (applied at render time for image observations)
_BASE_TO_OPTICAL = Transform(
    translation=Vector3(0.3, 0.0, 0.0),
    rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
    frame_id="base_link",
    child_frame_id="camera_link",
) + Transform(
    translation=Vector3(0.0, 0.0, 0.0),
    rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
    frame_id="camera_link",
    child_frame_id="camera_optical",
)


def render(drawing: Drawing2D, app_id: str = "drawing", spawn: bool = True) -> None:
    """Render a Drawing2D to a Rerun viewer."""
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init(app_id, spawn=spawn)

    # Collect elements by type
    points: list[Point] = []
    poses: list[Pose] = []
    arrows: list[Arrow] = []
    boxes: list[Box3D] = []
    cameras: list[Camera] = []
    polylines: list[Polyline] = []
    texts: list[Text] = []
    grids: list[OccupancyGrid] = []
    pointclouds: list[PointCloud2] = []
    observations: list[Observation[Any]] = []
    panels: list[Observation[Any]] = []

    for el in drawing.elements:
        if isinstance(el, Observation):
            if _is_image(el.data) and el.pose is None:
                panels.append(el)
            else:
                observations.append(el)
        elif isinstance(el, Point):
            points.append(el)
        elif isinstance(el, Pose):
            poses.append(el)
        elif isinstance(el, Arrow):
            arrows.append(el)
        elif isinstance(el, Box3D):
            boxes.append(el)
        elif isinstance(el, Camera):
            cameras.append(el)
        elif isinstance(el, Polyline):
            polylines.append(el)
        elif isinstance(el, Text):
            texts.append(el)
        elif isinstance(el, OccupancyGrid):
            grids.append(el)
        elif isinstance(el, PointCloud2):
            pointclouds.append(el)

    # Build and send blueprint
    has_images = (
        any(c.image is not None for c in cameras)
        or any(_has_image(obs) for obs in observations)
        or bool(panels)
    )
    views: list[Any] = [
        rrb.Spatial3DView(
            origin="scene",
            name="Scene",
            background=rrb.Background(kind="SolidColor", color=[0, 0, 0]),
            line_grid=rrb.LineGrid3D(
                plane=rr.components.Plane3D.XY.with_distance(0.5),
            ),
        )
    ]
    if has_images:
        views.append(rrb.Spatial2DView(origin="scene", name="Images"))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(*views, column_shares=[2, 1]) if len(views) > 1 else views[0]
    )
    rr.send_blueprint(blueprint)

    # Log elements
    if grids:
        for i, el in enumerate(grids):
            rr.log(f"scene/map/{i}", el.to_rerun(), static=True)

    if pointclouds:
        for i, el in enumerate(pointclouds):
            rr.log(f"scene/pointcloud/{i}", el.to_rerun(), static=True)

    if points:
        rr.log(
            "scene/points",
            rr.Points3D(
                positions=[[p.msg.x, p.msg.y, p.msg.z] for p in points],
                colors=[hex_to_rgb(p.color) for p in points],
                radii=[max(p.radius, 0.05) for p in points],
                labels=[p.label or "" for p in points] if any(p.label for p in points) else None,
            ),
            static=True,
        )

    if poses:
        rr.log(
            "scene/poses",
            rr.Points3D(
                positions=[[p.msg.x, p.msg.y, 0] for p in poses],
                colors=[hex_to_rgb(p.color) for p in poses],
                radii=[p.size * 0.3 for p in poses],
                labels=[p.label or "" for p in poses] if any(p.label for p in poses) else None,
            ),
            static=True,
        )
        rr.log(
            "scene/poses/headings",
            rr.Arrows3D(
                origins=[[p.msg.x, p.msg.y, 0] for p in poses],
                vectors=[
                    [math.cos(p.msg.yaw) * p.size, math.sin(p.msg.yaw) * p.size, 0] for p in poses
                ],
                colors=[hex_to_rgb(p.color) for p in poses],
            ),
            static=True,
        )

    if arrows:
        rr.log(
            "scene/arrows",
            rr.Arrows3D(
                origins=[[a.msg.x, a.msg.y, 0] for a in arrows],
                vectors=[
                    [math.cos(a.msg.yaw) * a.length, math.sin(a.msg.yaw) * a.length, 0]
                    for a in arrows
                ],
                colors=[hex_to_rgb(a.color) for a in arrows],
            ),
            static=True,
        )

    if boxes:
        rr.log(
            "scene/boxes",
            rr.Boxes3D(
                centers=[[b.center.x, b.center.y, 0] for b in boxes],
                half_sizes=[[b.size.x / 2, b.size.y / 2, b.size.z / 2] for b in boxes],
                colors=[hex_to_rgb(b.color) for b in boxes],
                labels=[b.label or "" for b in boxes] if any(b.label for b in boxes) else None,
            ),
            static=True,
        )

    for i, el in enumerate(polylines):
        rr.log(
            f"scene/polylines/{i}",
            rr.LineStrips3D(
                strips=[[[p.x, p.y, 0] for p in el.msg.poses]],
                colors=[hex_to_rgb(el.color)],
                radii=[el.width / 2],
            ),
            static=True,
        )

    if texts:
        rr.log(
            "scene/texts",
            rr.Points3D(
                positions=[[t.position[0], t.position[1], 0] for t in texts],
                labels=[t.text for t in texts],
                colors=[hex_to_rgb(t.color) for t in texts],
                radii=[0.01] * len(texts),
            ),
            static=True,
        )

    for i, el in enumerate(cameras):
        path = f"scene/cameras/{i}"
        rr.log(path, el.pose.to_rerun(), static=True)
        if el.camera_info:
            rr.log(path, el.camera_info.to_rerun(), static=True)
        elif el.image:
            h, w = el.image.shape[:2]
            focal = max(w, h)
            rr.log(
                path,
                rr.Pinhole(focal_length=focal, principal_point=[w / 2, h / 2], resolution=[w, h]),
                static=True,
            )
        if el.image:
            rr.log(f"{path}/image", el.image.to_rerun(), static=True)

    for i, obs in enumerate(observations):
        path = f"scene/observations/{i}"
        data = obs.data
        if _is_image(data):
            # Apply base→optical extrinsics for camera frustum rendering
            world_T_optical = Transform.from_pose("world", obs.pose_stamped) + _BASE_TO_OPTICAL
            rr.log(path, world_T_optical.to_pose().to_rerun(), static=True)
            h, w = data.shape[:2]
            focal = max(w, h)
            rr.log(
                path,
                rr.Pinhole(
                    focal_length=focal,
                    principal_point=[w / 2, h / 2],
                    resolution=[w, h],
                    image_plane_distance=1.0,
                ),
                static=True,
            )
            rr.log(f"{path}/image", data.to_rerun(), static=True)
        elif isinstance(data, PointCloud2):
            rr.log(path, obs.pose_stamped.to_rerun(), static=True)
            rr.log(f"{path}/pointcloud", data.to_rerun(), static=True)
        elif isinstance(data, (int, float)):
            rr.log(
                path,
                rr.Points3D(
                    positions=[[obs.pose_stamped.x, obs.pose_stamped.y, 0]],
                    labels=[str(data)],
                    radii=[0.05],
                ),
                static=True,
            )
        elif isinstance(data, str):
            # Word-wrap for label
            words = data.split()
            lines, line = [], ""
            for w in words:
                if line and len(line) + len(w) + 1 > 40:
                    lines.append(line)
                    line = w
                else:
                    line = f"{line} {w}" if line else w
            if line:
                lines.append(line)
            label = "\n".join(lines)
            x, y = obs.pose_stamped.x, obs.pose_stamped.y
            # Pin: line from ground up, label at the tip
            rr.log(
                f"{path}/pin",
                rr.LineStrips3D(
                    strips=[[[x, y, 0], [x, y, 5.0]]],
                    colors=[(100, 100, 100)],
                    radii=[0.01],
                ),
                static=True,
            )
            rr.log(
                f"{path}/label",
                rr.Points3D(
                    positions=[[x, y, 5.0]],
                    labels=[label],
                    radii=[0.001],
                ),
                static=True,
            )
        else:
            rr.log(
                path,
                rr.Points3D(positions=[[obs.pose_stamped.x, obs.pose_stamped.y, 0]], radii=[0.05]),
                static=True,
            )

    for i, obs in enumerate(panels):
        rr.log(f"scene/panels/{i}", obs.data.to_rerun(), static=True)


def _is_image(data: Any) -> bool:
    from dimos.msgs.sensor_msgs.Image import Image

    return isinstance(data, Image)


def _has_image(obs: Observation[Any]) -> bool:
    return _is_image(obs.data)
