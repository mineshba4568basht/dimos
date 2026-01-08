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

"""Rerun tap factory (dashboard-owned).

This module defines a picklable import path entrypoint for installing Out.tap callbacks
on worker processes via Module.install_out_tap(...).
"""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any

from dimos.dashboard.rerun_init import connect_rerun

if TYPE_CHECKING:
    from collections.abc import Callable

# Module-global caches shared across all Out taps within a single worker process.
# This is important because `camera_info` and `color_image` are separate `Out.tap` callbacks.
_LAST_LOGGED_PINHOLE_RES_BY_ENTITY: dict[str, tuple[int, int]] = {}


def make_rerun_tap(
    *,
    entity_path: str,
    server_addr: str | None = None,
    recording_id: str | None = None,
    rate_limit_hz: float | None = None,
    to_rerun_kwargs: dict[str, Any] | None = None,
    force_frame_id: str | None = None,
    also_log_to: list[str] | None = None,
    static: bool = False,
    camera_model: dict[str, Any] | None = None,
) -> Callable[[Any], None]:
    """Return a callback suitable for Out.tap that logs messages to Rerun."""
    kwargs = to_rerun_kwargs or {}
    last = 0.0
    last_frame_id_by_path: dict[str, str] = {}

    # If a static camera model was provided, we can use it to log pinholes immediately
    # even if we haven't seen a CameraInfo message yet.
    if camera_model:
        try:
            K = camera_model.get("K")
            if K and len(K) == 9:
                camera_model["fx"] = float(K[0])
                camera_model["fy"] = float(K[4])
                camera_model["cx"] = float(K[2])
                camera_model["cy"] = float(K[5])
                camera_model["w"] = int(camera_model.get("width", 0))
                camera_model["h"] = int(camera_model.get("height", 0))
        except Exception:
            camera_model = None

    def _cb(msg: Any) -> None:
        nonlocal last
        if not hasattr(msg, "to_rerun"):
            return

        # Best-effort rate limiting
        if rate_limit_hz is not None and rate_limit_hz > 0:
            now = time.monotonic()
            if now - last < 1.0 / rate_limit_hz:
                return
            last = now

        # Best-effort connect (no-op if already connected).
        connect_rerun(server_addr=server_addr, recording_id=recording_id)

        rr = importlib.import_module("rerun")

        # If the message carries a frame_id, attach this entity to that transform frame.
        # Use Transform3D with parent_frame to place this entity in the named TF frame's coordinate system.
        # This avoids the implicit "tf#/entity/path" frames that CoordinateFrame can create.
        frame_id = force_frame_id or getattr(msg, "frame_id", None)
        if isinstance(frame_id, str) and frame_id:
            for p in [entity_path, *(also_log_to or [])]:
                if last_frame_id_by_path.get(p) == frame_id:
                    continue
                try:
                    rr.log(
                        p,
                        rr.Transform3D(
                            translation=[0, 0, 0],
                            rotation=rr.Quaternion(xyzw=[0, 0, 0, 1]),
                            parent_frame=frame_id,
                        ),
                        static=True,
                    )
                    last_frame_id_by_path[p] = frame_id
                except Exception:
                    pass

        # Special-case camera models: emit a Pinhole for camera intrinsics.
        # NOTE: Do *not* set child_frame/parent_frame on the Pinhole here.
        # The entity is already attached to the named TF frame via Transform3D(parent_frame=...) above.
        msg_name = getattr(msg, "msg_name", "")
        if msg_name == "sensor_msgs.CameraInfo" and isinstance(frame_id, str) and frame_id:
            try:
                K = getattr(msg, "K", None)
                w = getattr(msg, "width", None)
                h = getattr(msg, "height", None)
                if (
                    isinstance(K, list)
                    and len(K) == 9
                    and isinstance(w, int)
                    and isinstance(h, int)
                    and w > 0
                    and h > 0
                ):
                    fx = float(K[0])
                    fy = float(K[4])
                    cx = float(K[2])
                    cy = float(K[5])
                    # (No need to update global cache anymore, relying on closure or local args)

                    # Log a pinhole model. Frame association is handled via CoordinateFrame above.
                    pinhole = rr.Pinhole(
                        focal_length=[fx, fy],
                        principal_point=[cx, cy],
                        width=w,
                        height=h,
                        image_plane_distance=0.5,
                    )
                    rr.log(entity_path, pinhole, static=True)
                    for p in also_log_to or []:
                        rr.log(p, pinhole, static=True)
                    _LAST_LOGGED_PINHOLE_RES_BY_ENTITY[entity_path] = (int(w), int(h))
                    return
            except Exception:
                # Fall back to msg.to_rerun below.
                pass

        # If we're logging an Image and we have camera intrinsics, ensure the pinhole resolution matches
        # the *actual* image resolution. This prevents the frustum/image plane from appearing to "grow"
        # when the upstream video resolution changes over time (common with WebRTC/adaptive streaming).
        if msg_name == "sensor_msgs.Image" and camera_model:
            try:
                img_w = int(getattr(msg, "width", 0))
                img_h = int(getattr(msg, "height", 0))
                if img_w > 0 and img_h > 0:
                    prev = _LAST_LOGGED_PINHOLE_RES_BY_ENTITY.get(entity_path)
                    if prev != (img_w, img_h):
                        base_w = int(camera_model.get("w", img_w))
                        base_h = int(camera_model.get("h", img_h))
                        fx = float(camera_model.get("fx", 0.0))
                        fy = float(camera_model.get("fy", 0.0))
                        cx = float(camera_model.get("cx", 0.0))
                        cy = float(camera_model.get("cy", 0.0))
                        if base_w > 0 and base_h > 0 and fx > 0 and fy > 0:
                            sx = img_w / base_w
                            sy = img_h / base_h
                            pinhole = rr.Pinhole(
                                focal_length=[fx * sx, fy * sy],
                                principal_point=[cx * sx, cy * sy],
                                width=img_w,
                                height=img_h,
                                image_plane_distance=0.5,
                            )
                            rr.log(entity_path, pinhole, static=True)
                            for p in also_log_to or []:
                                rr.log(p, pinhole, static=True)
                            _LAST_LOGGED_PINHOLE_RES_BY_ENTITY[entity_path] = (img_w, img_h)
            except Exception:
                pass

        try:
            data = msg.to_rerun(**kwargs)
        except Exception:
            # Best-effort: never break the producer due to visualization conversion.
            return
        
        # A message may intentionally opt out of Rerun logging.
        if data is None:
            return
        # An empty batch should not clear/override existing entities.
        if data == []:
            return

        # Support messages that return multiple logs (e.g. [(path, archetype), ...]).
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], str):
            # Ensure custom-path logs are still attached to a TF frame if the message has one.
            if isinstance(frame_id, str) and frame_id:
                p = data[0]
                if last_frame_id_by_path.get(p) != frame_id:
                    try:
                        rr.log(
                            p,
                            rr.Transform3D(
                                translation=[0, 0, 0],
                                rotation=rr.Quaternion(xyzw=[0, 0, 0, 1]),
                                parent_frame=frame_id,
                            ),
                            static=True,
                        )
                        last_frame_id_by_path[p] = frame_id
                    except Exception:
                        pass
            rr.log(data[0], data[1], static=static)
            return
        if isinstance(data, list) and data and isinstance(data[0], tuple) and len(data[0]) == 2:
            first_path = data[0][0]
            if isinstance(first_path, str):
                for p, d in data:
                    if isinstance(p, str):
                        full_path = f"{entity_path}/{p}"
                        # Ensure per-object child entities inherit the correct TF frame.
                        if isinstance(frame_id, str) and frame_id:
                            if last_frame_id_by_path.get(full_path) != frame_id:
                                try:
                                    rr.log(
                                        full_path,
                                        rr.Transform3D(
                                            translation=[0, 0, 0],
                                            rotation=rr.Quaternion(xyzw=[0, 0, 0, 1]),
                                            parent_frame=frame_id,
                                        ),
                                        static=True,
                                    )
                                    last_frame_id_by_path[full_path] = frame_id
                                except Exception:
                                    pass
                        rr.log(full_path, d, static=static)
                return

        rr.log(entity_path, data, static=static)
        for p in also_log_to or []:
            rr.log(p, data, static=static)

    return _cb
