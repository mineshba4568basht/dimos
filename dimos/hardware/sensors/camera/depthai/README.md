# DepthAI / OAK-D Pro (no ROS) camera

This folder contains a minimal DepthAI (Luxonis) acquisition layer that publishes:
- RGB frames (`ImageFormat.RGB`)
- aligned depth frames (`ImageFormat.DEPTH16`, **uint16 millimeters**)
- RGB `CameraInfo` from the device calibration

## Install

DepthAI support is optional in this repo. Install the vendor Python package:

```bash
pip install depthai
```

If the device is not detected / cannot be opened on Linux, you may need USB permissions / udev rules for OAK devices.

Common DepthAI USB IDs:
- unbooted: `03e7:2485` (Movidius MyriadX)
- booted: `03e7:f63b` (Luxonis Device)

Example udev rule (requires `sudo`):

```bash
sudo tee /etc/udev/rules.d/80-depthai.rules >/dev/null <<'EOF'
# Luxonis / DepthAI
SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", ATTR{idProduct}=="2485", MODE:="0666", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", ATTR{idProduct}=="f63b", MODE:="0666", TAG+="uaccess"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then unplug/replug the device.

## Quick test (OpenCV)

Run the built-in OpenCV viewer:

```bash
python dimos/hardware/sensors/camera/depthai/depthai_camera_test_script.py
```

(If you did not install the package with `pip install -e .`, this script will still work from a source checkout.)

## Standalone device check (no DimOS/LCM)

If you only want to verify the device is connected (independent of DimOS/LCM), run:

```bash
python dimos/hardware/sensors/camera/depthai/depthai_device_check.py
```

Enable IR emitters (OAK-D Pro only) with:

```bash
python dimos/hardware/sensors/camera/depthai/depthai_camera_test_script.py --enable-ir --ir-dot-ma 200 --ir-flood-ma 200
```


