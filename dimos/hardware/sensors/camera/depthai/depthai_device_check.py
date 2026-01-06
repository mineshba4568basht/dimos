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

"""
Standalone DepthAI / OAK device connectivity check (no DimOS/LCM required).

This script answers: "Is my DepthAI device connected and can I get frames?"

Requirements:
- Python 3.10+
- `pip install depthai`
- optional: `pip install opencv-python` to visualize frames
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any


def _require_py310() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            f"[depthai] Python {sys.version.split()[0]} detected. Need Python >= 3.10.\n"
            "Create/activate a Python 3.10+ environment and retry."
        )


def _try_import_cv2() -> Any | None:
    try:
        import cv2 as _cv2  # type: ignore[import-not-found]

        return _cv2
    except Exception:
        return None


def main() -> None:
    _require_py310()

    parser = argparse.ArgumentParser(description="Standalone DepthAI device check (no DimOS)")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show RGB preview window if OpenCV is installed (default: true)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to grab before exiting (default: 100)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS (default: 30)")
    args = parser.parse_args()

    try:
        import depthai as dai  # type: ignore[import-not-found]
    except Exception as e:
        raise SystemExit(
            "[depthai] Missing dependency: `depthai`.\nInstall with: `pip install depthai`."
        ) from e

    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise SystemExit(
            "[depthai] No DepthAI devices found.\n"
            "- Check USB cable / power\n"
            "- On Linux you may need udev rules/permissions\n"
            "- Try `lsusb | grep -i luxonis`"
        )

    print(f"[depthai] Found {len(devices)} device(s):")
    for d in devices:
        try:
            print(f"  - name={d.name} mxid={d.mxid} state={d.state}")
        except Exception:
            print(f"  - {d}")

    try:
        pipeline = dai.Pipeline()
    except RuntimeError as e:
        msg = str(e)
        if "X_LINK_DEVICE_NOT_FOUND" in msg:
            raise SystemExit(
                "[depthai] Device was detected but could not be found after booting.\n"
                "This is commonly caused by Linux USB permissions for the *booted* device ID.\n"
                "\n"
                "Try adding udev rules for both:\n"
                "  - 03e7:2485 (unbooted Movidius MyriadX)\n"
                "  - 03e7:f63b (booted Luxonis Device)\n"
                "\n"
                "Then reload udev rules and unplug/replug the device.\n"
                "Also try a different USB3 port/cable and avoid USB hubs."
            ) from e
        raise
    # DepthAI v3 API note:
    # - `XLinkOut` and `Device.getOutputQueue(...)` are not used here.
    # - Instead, create host queues directly from node outputs with `createOutputQueue(...)`,
    #   and start/stop the pipeline via `pipeline.start()` / `pipeline.stop()`.
    cam = pipeline.create(dai.node.ColorCamera)  # deprecated but simplest for a connectivity check

    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam.setFps(int(args.fps))

    rgb_q = cam.video.createOutputQueue(maxSize=1, blocking=False)

    cv2 = _try_import_cv2() if args.show else None
    if args.show and cv2 is None:
        print("[depthai] OpenCV not installed; running headless. Install with `pip install opencv-python`.")

    t0 = time.time()
    n = 0
    pipeline.start()
    try:
        while n < int(args.frames):
            pkt = rgb_q.tryGet()
            if pkt is None:
                time.sleep(0.001)
                continue

            frame = pkt.getCvFrame()  # RGB numpy array (HxWx3)
            n += 1

            if cv2 is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("DepthAI RGB (standalone)", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    dt = max(time.time() - t0, 1e-6)
    print(f"[depthai] Grabbed {n} frames in {dt:.2f}s ({n/dt:.1f} FPS).")

    if cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


