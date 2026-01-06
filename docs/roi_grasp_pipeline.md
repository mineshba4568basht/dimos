# ROI Grasp Pipeline (BBox → SAM → ROI Pointcloud → AnyGrasp → Top‑K)

This repo already has a strong manipulation stack (planning/execution) and a perception pipeline that produces:
- segmentation masks
- per-object Open3D pointclouds + numpy arrays suitable for grasp generation
- grasp visualization utilities

This document describes the **new modular ROI grasp pipeline** added under `dimos/perception/grasp_generation/` that covers the part you asked for:

RGB‑D frame  
↓  
User draws bounding box  
↓  
MobileSAM/SAM (bbox → mask)  
↓  
Masked point cloud + 3D cleanup  
↓  
AnyGrasp (ROI only) + filtering  
↓  
Top‑K grasps (ranked)

## What was added (files)

- `dimos/perception/grasp_generation/roi_types.py`
  - `UserBBoxSelection`: a small dataclass representing the click‑drag bbox in pixel coordinates.
  - bbox helpers: clamp + int conversion.

- `dimos/perception/grasp_generation/segmenters.py`
  - `BBoxMaskSegmenter` protocol: clean interface `predict_mask(rgb, bbox) -> mask`.
  - `SamBBoxSegmenter`: optional SAM/MobileSAM adapter (tries `mobile_sam` then `segment_anything`).
  - `BBoxOnlySegmenter`: dependency‑free fallback that just returns the bbox rectangle as a mask (keeps plumbing usable even without SAM).

- `dimos/perception/grasp_generation/roi_cleanup.py`
  - `cleanup_roi_pointcloud`: downsample → outlier removal → DBSCAN → keep largest cluster.
  - This is the “mask leakage cleanup” stage for cluttered scenes.

- `dimos/perception/grasp_generation/grasp_filters.py`
  - `filter_grasps`: applies **collision proxy filtering** and optional **kinematic feasibility** callback, then returns Top‑K.
  - Collision proxy uses a simple parallel‑jaw gripper box model in the same grasp coordinate convention used by existing visualization code.

- `dimos/perception/grasp_generation/backends.py`
  - `GraspGeneratorBackend` protocol: a single method `generate_grasps(points_xyz, colors_rgb) -> list[dict]`.
  - `AnyGraspBackend`: intentionally left as a thin **adapter stub** because AnyGrasp APIs differ by fork; you wire it to your AnyGrasp install.

- `dimos/perception/grasp_generation/roi_grasp_pipeline.py`
  - `RoiGraspPipeline`: orchestrates the end‑to‑end flow on a single RGB‑D frame.
  - Uses existing Dimos pointcloud utility `create_point_cloud_and_extract_masks()` so we stay consistent with the repo’s formats.

## Key design decisions (why it matches the repo)

- **Pipeline style (like `ManipulationPipeline`)**: pure‑python class that you can call per-frame or wrap in reactive streams.
- **Camera agnostic**: only requires aligned `(rgb, depth_m, intrinsics)`. Your Oak‑D Pro, RealSense, ZED, simulation, etc. can all feed this as long as you produce those three.
- **Optional heavy deps**: SAM and AnyGrasp are behind adapters. If they are not installed, the pipeline can still import and you’ll get a clear error only when you actually execute the stage that needs them.
- **Output format compatibility**: grasps are kept in Dimos’ existing dict format (`translation`, `rotation_matrix`, `width`, `score`, …) so feeding them into the planner later is straightforward.

## How you wire user selection (UI → backend)

The web server now supports a simple endpoint:
- `POST /select_bbox` on `FastAPIServer` (optional; only active if you pass `bbox_selection_subject`).

Payload example:

```json
{ "x1": 100, "y1": 80, "x2": 240, "y2": 200, "source": "ui" }
```

Your UI can do click‑drag, then send that JSON to the server. The server emits it on the Rx subject you provide, and your grasp pipeline subscribes to update the “latest selection”.

## How you wire AnyGrasp

Edit `dimos/perception/grasp_generation/backends.py`:
- implement `AnyGraspBackend.generate_grasps()` for your AnyGrasp install
- return a list of dicts (each dict needs at least `translation`, `rotation_matrix`, `score`, `width`)

Once that adapter is wired, `RoiGraspPipeline` will:
- run AnyGrasp on the **cleaned ROI pointcloud only**
- then apply `filter_grasps()` to return collision‑filtered, Top‑K grasps

## OAK‑D Pro / DepthAI (no ROS) acquisition layer

For OAK‑D Pro specifically, we added a DepthAI hardware driver and module:
- `dimos/hardware/sensors/camera/depthai/camera.py` (`DepthAI`): provides RGB stream + aligned DEPTH16 stream
- `dimos/hardware/sensors/camera/depthai/module.py` (`DepthAICameraModule`): publishes `color_image`, `depth_image`, `camera_info`

Why this is correct for OAK‑D Pro:
- The OAK‑D Pro is an **active stereo** camera with an **IR dot projector** and **IR illumination** for low‑light, which makes depth on texture‑less surfaces much more reliable than passive stereo ([Luxonis OAK‑D Pro product page](https://shop.luxonis.com/products/oak-d-pro?variant=42455252369631)).

Depth conventions:
- `depth_image` is `DEPTH16` in **millimeters** (uint16). Convert to meters with `depth_m = depth_mm.astype(np.float32) / 1000.0`.



