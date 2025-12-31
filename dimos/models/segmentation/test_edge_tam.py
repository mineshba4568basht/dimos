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

import queue
import threading
import time
import cv2
import numpy as np

from dimos.hardware.camera.webcam import Webcam, WebcamConfig
from dimos.models.segmentation.edge_tam import EdgeTAMProcessor
from dimos.msgs.sensor_msgs import Image
from dimos.perception.detection.type.detection2d.seg import Detection2DSeg
from dimos.utils.gpu_utils import is_cuda_available

# Global state for mouse interaction
mouse_state = {
    "start_point": None,
    "current_point": None,
    "drawing": False,
    "new_prompt": None,  # (type, data) -> ('point', [x, y]) or ('box', [x1, y1, x2, y2])
}


def mouse_callback(event, x, y, flags, param):
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
        
        # If distance is small, treat as point click
        dist = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
        if dist < 5:
            mouse_state["new_prompt"] = ("point", np.array([[x, y]], dtype=np.float32))
        else:
            # Treat as bounding box
            x1, y1 = min(start_x, x), min(start_y, y)
            x2, y2 = max(start_x, x), max(start_y, y)
            mouse_state["new_prompt"] = ("box", np.array([x1, y1, x2, y2], dtype=np.float32))


def draw_overlay(image_cv, detections, mouse_info):
    if detections is None:
        return

    # Draw detections
    for det in detections.detections:
        if isinstance(det, Detection2DSeg):
            # Draw mask overlay
            mask = det.mask.astype(bool)
            
            # Create colored mask (green with some transparency)
            colored_mask = np.zeros_like(image_cv)
            colored_mask[mask] = [0, 255, 0]
            
            # Blend with original image
            alpha = 0.4
            mask_indices = mask > 0
            
            # Use robust blending logic to handle squeeze issues
            if np.any(mask_indices):
                roi = image_cv[mask_indices]
                colored_roi = colored_mask[mask_indices]
                blended = cv2.addWeighted(roi, 1 - alpha, colored_roi, alpha, 0)
                
                # Squeeze only if necessary and handle dimensions
                if len(blended.shape) > len(roi.shape):
                    blended = blended.squeeze()
                
                image_cv[mask_indices] = blended
            
            # Draw contour
            contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv, contours, -1, (0, 255, 0), 2)
            
            # Draw bbox
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw current mouse action
    if mouse_info["drawing"] and mouse_info["start_point"]:
        start = mouse_info["start_point"]
        curr = mouse_info["current_point"]
        cv2.rectangle(image_cv, start, curr, (0, 0, 255), 2)


def test_edgetam_webcam_interactive():
    """
    Interactive test for EdgeTAM using Webcam.
    Usage:
    - Run the test.
    - A window will open showing the webcam feed.
    - Click to define a point prompt (adds positive click).
    - Drag to define a box prompt.
    - Press 'q' or Esc to exit.
    """
    
    # Initialize Webcam
    # Note: Adjust camera_index if needed (e.g. 0 or 1)
    webcam = Webcam(
        camera_index=0, 
        frequency=30, 
        frame_width=640, 
        frame_height=480
    )
    
    # Initialize EdgeTAM Processor
    try:
        processor = EdgeTAMProcessor(device="cuda" if is_cuda_available() else "cpu")
    except Exception as e:
        print(f"Skipping EdgeTAM test: {e}. Model might not be available.")
        return

    # Queue for passing frames from webcam thread to main thread
    frame_queue = queue.Queue(maxsize=2)
    
    def frame_handler(image: Image):
        if not frame_queue.full():
            frame_queue.put(image)

    # Start webcam stream
    dispose = webcam.image_stream().subscribe(on_next=frame_handler)
    
    # Setup OpenCV window
    window_name = "EdgeTAM Interactive Test"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\nControls:")
    print("- Click to track object at point")
    print("- Drag to track object in box")
    print("- Press 'r' to reset tracking")
    print("- Press 'q' to quit")

    tracking_active = False
    last_detections = None
    tracking_loss_start_time = None
    TRACKING_TIMEOUT = 20.0  # seconds
    
    try:
        while True:
            try:
                # Get latest frame
                image = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or Esc
                break
            elif key == ord('r'):
                processor.stop()
                tracking_active = False
                last_detections = None
                tracking_loss_start_time = None
                print("Tracking reset")

            # Check for mouse prompts
            if mouse_state["new_prompt"]:
                prompt_type, prompt_data = mouse_state["new_prompt"]
                mouse_state["new_prompt"] = None
                
                print(f"Initializing track with {prompt_type}")
                
                if prompt_type == "point":
                    last_detections = processor.init_track(
                        image=image, 
                        points=prompt_data, 
                        labels=np.array([1], dtype=np.int32)
                    )
                elif prompt_type == "box":
                    last_detections = processor.init_track(
                        image=image, 
                        box=prompt_data
                    )
                
                tracking_active = True
                tracking_loss_start_time = None
            elif tracking_active:
                # Continue tracking
                detections_found = False
                try:
                    # Use process_image which now returns empty detections if none valid
                    current_detections = processor.process_image(image)
                    
                    if current_detections and len(current_detections.detections) > 0:
                        last_detections = current_detections
                        detections_found = True
                except Exception as e:
                    print(f"Tracking error: {e}")
                    
                if detections_found:
                    tracking_loss_start_time = None
                else:
                    # Tracking lost or error
                    if tracking_loss_start_time is None:
                        tracking_loss_start_time = time.time()
                        print("Tracking lost, starting timeout...")
                    
                    if time.time() - tracking_loss_start_time > TRACKING_TIMEOUT:
                        print(f"Tracking lost for {TRACKING_TIMEOUT}s, stopping.")
                        processor.stop()
                        tracking_active = False
                        last_detections = None
                        tracking_loss_start_time = None
            
            # Visualization
            image_cv = image.to_opencv()
            
            if tracking_active:
                # Draw status
                status_text = "Tracking"
                color = (0, 255, 0)
                
                if tracking_loss_start_time is not None:
                    elapsed = time.time() - tracking_loss_start_time
                    status_text = f"Lost ({elapsed:.1f}s)"
                    color = (0, 0, 255)
                
                cv2.putText(image_cv, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw last valid detection (if any) or current detection
                if last_detections and tracking_loss_start_time is None:
                    draw_overlay(image_cv, last_detections, mouse_state)
            
            elif mouse_state["drawing"]:
                # Create dummy detection for overlay drawing of just the mouse rect
                from dimos.perception.detection.type import ImageDetections2D
                draw_overlay(image_cv, ImageDetections2D(image), mouse_state)
                
            cv2.imshow(window_name, image_cv)
            
    finally:
        if dispose:
            dispose.dispose()  # Stop webcam
        if processor:
            processor.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        test_edgetam_webcam_interactive()
    except KeyboardInterrupt:
        pass
