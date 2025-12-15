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

import onnxruntime as ort
from PIL import Image
import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

try:
	cv2.setNumThreads(int(os.getenv("CV_THREADS", "1")))
except Exception:
	pass

try:
    import cupy as cp
    HAS_GPU = True
    mempool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(mempool.malloc)
except ImportError:
    HAS_GPU = False
    cp = None

# ---- GPU-only bilinear resize (CHW or HWC) ----
def resize_bilinear_hwc(img_cu: cp.ndarray, out_h: int, out_w: int) -> cp.ndarray:
    """Pure CuPy bilinear resize for HWC float32 images"""
    in_h, in_w, C = img_cu.shape
    out = cp.empty((out_h, out_w, C), dtype=cp.float32)

    scale_y = in_h / out_h
    scale_x = in_w / out_w

    oy = cp.arange(out_h, dtype=cp.float32)
    ox = cp.arange(out_w, dtype=cp.float32)
    yy, xx = cp.meshgrid(oy, ox, indexing="ij")
    yy = yy * scale_y
    xx = xx * scale_x

    y0 = cp.floor(yy).astype(cp.int32)
    x0 = cp.floor(xx).astype(cp.int32)
    y1 = cp.clip(y0 + 1, 0, in_h - 1)
    x1 = cp.clip(x0 + 1, 0, in_w - 1)

    wy = yy - y0
    wx = xx - x0

    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx

    for c in range(C):
        Ia = img_cu[y0, x0, c]
        Ib = img_cu[y0, x1, c]
        Ic = img_cu[y1, x0, c]
        Id = img_cu[y1, x1, c]
        out[..., c] = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out


class Metric3D:
    def __init__(self, gt_depth_scale=256.0, camera_intrinsics=None,
                 provider='auto', onnx_model_path='onnx/metric3d_vit_small.onnx'):
        self.input_size = (616, 1064)  # (H, W)
        self.gt_depth_scale = gt_depth_scale
        self.intrinsic = camera_intrinsics or [707.0493, 707.0493, 604.0814, 180.5066]
        self.intrinsic_scaled = None
        self.pad_info = None
        self.rgb_origin = None
        self.onnx_model_path = onnx_model_path

        print(f"### Using model: {onnx_model_path} ###")

        # ---- ORT provider config ----
        MIN_SHAPE = "image:1x3x616x1064"
        OPT_SHAPE = "image:1x3x616x1064"
        MAX_SHAPE = "image:1x3x616x1064"

        tensorrt_opts = {
            "device_id": 0,
            "trt_max_workspace_size": 600 * 1024**2,
            "trt_builder_optimization_level": 4,
            "trt_auxiliary_streams": 1,
            "trt_fp16_enable": True,
            "trt_int8_enable": False,
            "trt_max_partition_iterations": 1000,
            "trt_min_subgraph_size": 1,
            "trt_build_heuristics_enable": True,
            "trt_layer_norm_fp32_fallback": True,
            "trt_context_memory_sharing_enable": True,
            "trt_cuda_graph_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_engines",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": "./trt_timing",
            "trt_profile_min_shapes": MIN_SHAPE,
            "trt_profile_opt_shapes": OPT_SHAPE,
            "trt_profile_max_shapes": MAX_SHAPE,
        }

        cuda_opts = {
            'cudnn_conv_use_max_workspace': '0',
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }

        self.contains_io_binding = False
        if provider == 'auto':
            available_providers = ort.get_available_providers()
            if 'TensorrtExecutionProvider' in available_providers:
                providers = [
                    ('TensorrtExecutionProvider', tensorrt_opts),
                    ('CUDAExecutionProvider', cuda_opts)
                ]
                self.contains_io_binding = True
            elif 'CUDAExecutionProvider' in available_providers:
                providers = [('CUDAExecutionProvider', cuda_opts)]
                self.contains_io_binding = True
            else:
                providers = ['CPUExecutionProvider']
        elif provider == 'cuda':
            providers = [('CUDAExecutionProvider', cuda_opts)]
            self.contains_io_binding = True
        elif provider == 'tensorrt':
            providers = [('TensorrtExecutionProvider', tensorrt_opts)]
            self.contains_io_binding = True
        else:
            providers = ['CPUExecutionProvider']

        print(f"### Using providers: {providers} ###")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_THREADS", max(1, (os.cpu_count() or 4) // 2)))
        so.inter_op_num_threads = int(os.getenv("ORT_INTER_OP_THREADS", "1"))

        self.session = ort.InferenceSession(onnx_model_path, sess_options=so, providers=providers)
        if self.contains_io_binding:
            io = self.session.io_binding()
            io.bind_output(name="pred_depth", device_type="cuda", device_id=0)
            self.io_binding = io

    def update_intrinsic(self, intrinsic):
        if len(intrinsic) != 4:
            raise ValueError("Intrinsic must be a list or tuple with 4 values: [fx, fy, cx, cy]")
        self.intrinsic = intrinsic
        logger.info(f"Intrinsics updated to: {self.intrinsic}")

    def prepare_input(self, rgb_image):
        h, w = rgb_image.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        if HAS_GPU and isinstance(rgb_image, cp.ndarray):
            # ---- GPU path ----
            rgb = resize_bilinear_hwc(rgb_image.astype(cp.float32), new_h, new_w)

            pad_h = self.input_size[0] - new_h
            pad_w = self.input_size[1] - new_w
            pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
            padding = ((pad_h_half, pad_h - pad_h_half),
                       (pad_w_half, pad_w - pad_w_half),
                       (0, 0))

            pad_val = [123.675, 116.28, 103.53]
            pads = ((pad_h_half, pad_h - pad_h_half),
                    (pad_w_half, pad_w - pad_w_half))

            rgb_padded = []
            for c in range(3):
                ch = cp.pad(rgb[..., c], pads, mode="constant", constant_values=pad_val[c])
                rgb_padded.append(ch)
            rgb = cp.stack(rgb_padded, axis=-1)

            mean = cp.array([123.675, 116.28, 103.53], dtype=cp.float32)[:, None, None]
            std = cp.array([58.395, 57.12, 57.375], dtype=cp.float32)[:, None, None]
            chw = rgb.transpose(2, 0, 1)
            chw = (chw - mean) / std
            chw = cp.ascontiguousarray(chw[None])
            self.intrinsic_scaled = [self.intrinsic[0] * scale,
                         self.intrinsic[1] * scale,
                         self.intrinsic[2] * scale,
                         self.intrinsic[3] * scale]
            self.pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

            if self.contains_io_binding:
                self.io_binding.bind_input(
                    name="image",
                    device_type="cuda",
                    device_id=0,
                    element_type=np.float32,
                    shape=chw.shape,
                    buffer_ptr=int(chw.data.ptr),  # raw GPU pointer
                )
                return self.io_binding, rgb_image.shape[:2]
            else:
                # Fallback: copy to host
                return {"image": cp.asnumpy(chw)}, rgb_image.shape[:2]
        else:
            # ---- CPU path ----
            rgb = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_h = self.input_size[0] - new_h
            pad_w = self.input_size[1] - new_w
            pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
            rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                                     pad_w_half, pad_w - pad_w_half,
                                     cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)[:, None, None]
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)[:, None, None]
            chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
            chw = (chw - mean) / std
            x_np = np.ascontiguousarray(chw[None], dtype=np.float32)

            self.intrinsic_scaled = [self.intrinsic[0] * scale,
                                     self.intrinsic[1] * scale,
                                     self.intrinsic[2] * scale,
                                     self.intrinsic[3] * scale]
            self.pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

            if self.contains_io_binding:
                x_dev = ort.OrtValue.ortvalue_from_numpy(x_np, 'cuda', 0)
                self.io_binding.bind_input(
                    name='image', device_type='cuda', device_id=0,
                    element_type=np.float32, shape=x_dev.shape(), buffer_ptr=x_dev.data_ptr()
                )
                return self.io_binding, rgb_image.shape[:2]
            else:
                return {"image": x_np}, rgb_image.shape[:2]

    def infer_depth(self, img, debug=False):
        if debug:
            print(f"Input image: {img}")
        try:
            if isinstance(img, str):
                self.rgb_origin = cv2.imread(img)[:, :, ::-1]
            else:
                self.rgb_origin = img
        except Exception as e:
            logger.error(f"Error parsing into infer_depth: {e}")
            return np.array([])

        onnx_input, original_shape = self.prepare_input(self.rgb_origin)
        if self.contains_io_binding:
            self.session.run_with_iobinding(onnx_input)
            out_dev = self.io_binding.get_outputs()[0]
            shape = tuple(out_dev.shape())
            dtype = np.float32
            ptr = out_dev.data_ptr()
            mem = cp.cuda.UnownedMemory(ptr, np.prod(shape) * np.dtype(dtype).itemsize, self)
            depth = cp.ndarray(shape=shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(mem, 0))
        else:
            outputs = self.session.run(None, onnx_input)
            depth = outputs[0]

        depth = depth.squeeze()

        # Remove padding
        pad_info = self.pad_info
        depth = depth[pad_info[0]: self.input_size[0] - pad_info[1],
                      pad_info[2]: self.input_size[1] - pad_info[3]]

        if HAS_GPU and isinstance(depth, cp.ndarray):
            depth = resize_bilinear_hwc(depth[..., None], original_shape[0], original_shape[1])[..., 0]
            if self.intrinsic_scaled is not None:
                canonical_to_real_scale = self.intrinsic_scaled[0] / 1000.0
                depth = depth * canonical_to_real_scale
            return depth.astype(cp.float32)
        else:
            depth = cv2.resize(depth, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            if self.intrinsic_scaled is not None:
                canonical_to_real_scale = self.intrinsic_scaled[0] / 1000.0
                depth = depth * canonical_to_real_scale
            return depth.astype(np.float32)

    def save_depth(self, pred_depth):
        if isinstance(pred_depth, np.ndarray):
            pred_depth_scaled = (pred_depth * self.gt_depth_scale).astype(np.uint16)
        elif isinstance(pred_depth, Image.Image):
            pred_depth_scaled = np.array(pred_depth)
        else:
            pred_depth_scaled = pred_depth
        output_depth_file = "output_depth_map.png"
        cv2.imwrite(output_depth_file, pred_depth_scaled)
        logger.info(f"Depth map saved to {output_depth_file}")

    def eval_predicted_depth(self, depth_file, pred_depth):
        if depth_file is not None:
            gt_depth = cv2.imread(depth_file, -1)
            gt_depth = gt_depth / self.gt_depth_scale
            if isinstance(pred_depth, Image.Image):
                pred_depth = np.array(pred_depth) / self.gt_depth_scale
            else:
                pred_depth = pred_depth / self.gt_depth_scale
            assert gt_depth.shape == pred_depth.shape
            mask = gt_depth > 1e-8
            abs_rel_err = (np.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
            logger.info(f"abs_rel_err: {abs_rel_err}")

    def cleanup(self):
        if hasattr(self, 'session') and self.session:
            self.session = None

