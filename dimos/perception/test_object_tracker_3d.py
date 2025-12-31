#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""Integration test for ObjectTracker3D with ZED camera.

Usage:
- Click and drag to draw a bounding box around an object to track
- Press 'r' to reset tracking
- Press 'q' to quit
"""

import queue
import time

import cv2
from dimos_lcm.foxglove_msgs import SceneUpdate

from dimos.core import LCMTransport
from dimos.core.blueprints import autoconnect
from dimos.hardware.camera.zed.camera import zed_module
from dimos.msgs.sensor_msgs import CameraInfo, Image
from dimos.msgs.vision_msgs import Detection3D
from dimos.perception.object_tracker_3d import ObjectTracker3D, object_tracker_3d
from dimos.robot.foxglove_bridge import foxglove_bridge
from dimos.utils.logging_config import setup_logger

logger = setup_logger(__name__)


mouse_state = {
    "start_point": None,
    "current_point": None,
    "drawing": False,
    "new_bbox": None,
}


def mouse_callback(event, x, y, _flags, _param):
    """Handle mouse events for drawing bounding boxes."""
    global mouse_state

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["start_point"] = (x, y)
        mouse_state["drawing"] = True
        mouse_state["current_point"] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_state["drawing"]:
            mouse_state["current_point"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state["drawing"] = False
        start_x, start_y = mouse_state["start_point"]

        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)

        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            mouse_state["new_bbox"] = (x1, y1, x2, y2)


def draw_overlay(image_cv, mouse_info, tracking_active=False, has_detection=False):
    """Draw tracking status and mouse interaction overlay."""
    if tracking_active:
        status_text = "Tracking Active"
        color = (0, 255, 0) if has_detection else (0, 165, 255)
    else:
        status_text = "Click and drag to start tracking"
        color = (200, 200, 200)

    cv2.putText(image_cv, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if mouse_info["drawing"] and mouse_info["start_point"]:
        start = mouse_info["start_point"]
        curr = mouse_info["current_point"]
        cv2.rectangle(image_cv, start, curr, (0, 0, 255), 2)


def visualize_detection3d(image_cv, detection: Detection3D):
    """Visualize a Detection3D message on the image."""
    if not detection or not detection.results:
        return

    bbox = detection.bbox
    center_x = bbox.center.position.x
    center_y = bbox.center.position.y
    width = bbox.size.x
    height = bbox.size.y

    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if detection.results:
        hypothesis = detection.results[0].hypothesis
        label = f"ID: {hypothesis.class_id} ({hypothesis.score:.2f})"
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if detection.results and detection.results[0].pose:
        pose = detection.results[0].pose.pose
        pos_text = f"3D: ({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
        cv2.putText(image_cv, pos_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def tracker_blueprint():
    """Create ObjectTracker3D blueprint with ZED camera and Foxglove visualization."""
    return (
        autoconnect(
            zed_module(
                camera_id=0,
                resolution="HD720",
                depth_mode="NEURAL",
                fps=15,
                enable_tracking=False,
                publish_rate=15.0,
                frame_id="zed_camera",
            ),
            object_tracker_3d(tracking_timeout=10.0),
            foxglove_bridge(),
        )
        .global_config(n_dask_workers=6)
        .transports({
            # ZED outputs
            ("color_image", Image): LCMTransport("/zed/color_image", Image),
            ("depth_image", Image): LCMTransport("/zed/depth_image", Image),
            ("camera_info", CameraInfo): LCMTransport("/zed/camera_info", CameraInfo),
            
            # Explicitly bind ObjectTracker3D inputs to ZED topics to ensure connection
            (ObjectTracker3D, "color_image"): LCMTransport("/zed/color_image", Image),
            (ObjectTracker3D, "depth_image"): LCMTransport("/zed/depth_image", Image),
            (ObjectTracker3D, "camera_info"): LCMTransport("/zed/camera_info", CameraInfo),
            
            # ObjectTracker3D outputs
            (ObjectTracker3D, "detection3d"): LCMTransport("/tracker3d/detection3d", Detection3D),
            (ObjectTracker3D, "scene_update"): LCMTransport("/tracker3d/scene_update", SceneUpdate),
        })
    )


def main():
    """Main integration test."""
    logger.info("Starting ObjectTracker3D integration test")
    from dimos.protocol import pubsub
    pubsub.lcm.autoconf()

    blueprint = tracker_blueprint()
    coordinator = blueprint.build()

    tracker = coordinator.get_instance(ObjectTracker3D)

    color_queue = queue.Queue(maxsize=2)
    detection_queue = queue.Queue(maxsize=2)

    def color_handler(msg):
        if not color_queue.full():
            color_queue.put(msg)

    def detection_handler(detection):
        if not detection_queue.full():
            detection_queue.put(detection)

    color_transport = LCMTransport("/zed/color_image", Image)
    detection_transport = LCMTransport("/tracker3d/detection3d", Detection3D)

    color_transport.subscribe(color_handler)
    detection_transport.subscribe(detection_handler)

    window_name = "ObjectTracker3D - ZED Integration Test"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    logger.info("Starting interactive visualization")
    print("\nControls:")
    print("- Click and drag to draw a bounding box around an object to track")
    print("- Press 'r' to reset tracking")
    print("- Press 'q' to quit")

    tracking_active = False
    last_detection = None
    last_image = None

    while True:
        while not color_queue.empty():
            last_image = color_queue.get_nowait()

        while not detection_queue.empty():
            last_detection = detection_queue.get_nowait()

        if last_image is None:
            time.sleep(0.01)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            logger.info("Resetting tracking")
            tracker.stop()
            tracker.start()
            tracking_active = False
            last_detection = None
            mouse_state["new_bbox"] = None
            print("Tracking reset")

        if mouse_state["new_bbox"] and not tracking_active:
            x1, y1, x2, y2 = mouse_state["new_bbox"]
            mouse_state["new_bbox"] = None
            logger.info(f"Initializing tracking with bbox: ({x1}, {y1}, {x2}, {y2})")
            bbox = (float(x1), float(y1), float(x2), float(y2))
            tracker.track(bbox)
            tracking_active = True
            print("Tracking initialized")

        if tracking_active:
            tracking_active = tracker.is_tracking()
            if not tracking_active:
                logger.warning("Tracking lost")
                print("Tracking lost - draw a new bbox to restart")

        image_cv = last_image.to_opencv()
        draw_overlay(image_cv, mouse_state, tracking_active=tracking_active, has_detection=(last_detection is not None))

        if tracking_active and last_detection:
            visualize_detection3d(image_cv, last_detection)

        cv2.imshow(window_name, image_cv)

    logger.info("Cleaning up")
    cv2.destroyAllWindows()
    coordinator.stop_all_modules()
    logger.info("Test completed")


if __name__ == "__main__":
    main()
