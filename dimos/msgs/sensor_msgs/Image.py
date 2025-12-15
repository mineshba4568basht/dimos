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

import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

# Optional CuPy
try:
    import cupy as cp

    HAS_CUDA = True
except Exception:
    cp = None
    HAS_CUDA = False

# Import LCM types
from dimos_lcm.sensor_msgs.Image import Image as LCMImage
from dimos_lcm.std_msgs.Header import Header

from dimos.types.timestamped import Timestamped


class ImageFormat(Enum):
    """Supported image formats for internal representation."""

    BGR = "BGR"  # 8-bit Blue-Green-Red color
    RGB = "RGB"  # 8-bit Red-Green-Blue color
    RGBA = "RGBA"  # 8-bit RGB with Alpha
    BGRA = "BGRA"  # 8-bit BGR with Alpha
    GRAY = "GRAY"  # 8-bit Grayscale
    GRAY16 = "GRAY16"  # 16-bit Grayscale
    DEPTH = "DEPTH"  # 32-bit Float Depth


# -----------------------
# CuPy helper primitives
# -----------------------


def _is_cu(x) -> bool:
    return HAS_CUDA and isinstance(x, cp.ndarray)


def _ascontig(x):
    if _is_cu(x):
        return x if x.flags["C_CONTIGUOUS"] else cp.ascontiguousarray(x)
    return x if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x)


def _to_cpu(x):
    return cp.asnumpy(x) if _is_cu(x) else x


def _to_cu(x):
    if HAS_CUDA and isinstance(x, np.ndarray):
        return cp.asarray(x)
    return x


# --------------- GPU color ops (CuPy) ---------------


def _rgb_to_bgr_gpu(img: cp.ndarray) -> cp.ndarray:
    # HWC
    return img[..., ::-1]


def _bgr_to_rgb_gpu(img: cp.ndarray) -> cp.ndarray:
    return img[..., ::-1]


def _rgba_to_bgra_gpu(img: cp.ndarray) -> cp.ndarray:
    # RGBA -> BGRA : swap R and B, keep A
    out = img.copy()
    out[..., 0], out[..., 2] = img[..., 2], img[..., 0]
    return out


def _bgra_to_rgba_gpu(img: cp.ndarray) -> cp.ndarray:
    out = img.copy()
    out[..., 0], out[..., 2] = img[..., 2], img[..., 0]
    return out


def _gray_to_rgb_gpu(gray: cp.ndarray) -> cp.ndarray:
    # gray HxW (uint8/uint16/float) -> HxWx3
    return cp.stack([gray, gray, gray], axis=-1)


def _rgb_to_gray_gpu(rgb: cp.ndarray) -> cp.ndarray:
    # ITU-R BT.601 luma transform: Y = 0.299 R + 0.587 G + 0.114 B
    r = rgb[..., 0].astype(cp.float32)
    g = rgb[..., 1].astype(cp.float32)
    b = rgb[..., 2].astype(cp.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # keep dtype reasonable (uint8 if input was uint8 RGB)
    if rgb.dtype == cp.uint8:
        y = cp.clip(y, 0, 255).astype(cp.uint8)
    return y


# --------------- GPU resize (CuPy bilinear) ---------------


def _resize_bilinear_hwc_gpu(img: cp.ndarray, out_h: int, out_w: int) -> cp.ndarray:
    """Bilinear resize for HWC images (C=1,3,4) using CuPy; dtype preserved if uint8 else float32."""
    in_h, in_w = img.shape[:2]
    C = 1 if img.ndim == 2 else img.shape[2]

    # Work in float32 for interpolation, convert back if needed
    work = img.astype(cp.float32, copy=False)
    if C == 1 and img.ndim == 2:
        work = work[..., None]

    oy = cp.arange(out_h, dtype=cp.float32)
    ox = cp.arange(out_w, dtype=cp.float32)
    yy, xx = cp.meshgrid(oy, ox, indexing="ij")

    scale_y = (in_h - 1) / max(out_h - 1, 1)
    scale_x = (in_w - 1) / max(out_w - 1, 1)
    yy = yy * scale_y
    xx = xx * scale_x

    y0 = cp.floor(yy).astype(cp.int32)
    x0 = cp.floor(xx).astype(cp.int32)
    y1 = cp.clip(y0 + 1, 0, in_h - 1)
    x1 = cp.clip(x0 + 1, 0, in_w - 1)

    wy = yy - y0
    wx = xx - x0

    wy0 = 1.0 - wy
    wx0 = 1.0 - wx

    out = cp.empty((out_h, out_w, C), dtype=cp.float32)
    for c in range(C):
        Ia = work[y0, x0, c]
        Ib = work[y0, x1, c]
        Ic = work[y1, x0, c]
        Id = work[y1, x1, c]
        out[..., c] = (wy0 * wx0) * Ia + (wy0 * wx) * Ib + (wy * wx0) * Ic + (wy * wx) * Id

    if C == 1:
        out = out[..., 0]

    # Convert back to original dtype if 8-bit
    if img.dtype == cp.uint8:
        out = cp.clip(out, 0, 255).astype(cp.uint8)
    return out


# -----------------------
# Image class (GPU-aware)
# -----------------------


@dataclass
class Image(Timestamped):
    """Standardized image type with LCM integration (NumPy or CuPy)."""

    msg_name = "sensor_msgs.Image"
    data: np.ndarray  # or cupy.ndarray
    format: ImageFormat = field(default=ImageFormat.BGR)
    frame_id: str = field(default="")
    ts: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate image data and format."""
        if self.data is None:
            raise ValueError("Image data cannot be None")

        is_np = isinstance(self.data, np.ndarray)
        is_cp = _is_cu(self.data)
        if not (is_np or is_cp):
            raise ValueError("Image data must be a numpy or cupy array")

        if self.data.ndim < 2:
            raise ValueError("Image data must be at least 2D")

        # Ensure data is contiguous for efficient ops
        self.data = _ascontig(self.data)

    @property
    def is_cuda(self) -> bool:
        return _is_cu(self.data)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        if self.data.ndim == 2:
            return 1
        elif self.data.ndim == 3:
            return int(self.data.shape[2])
        else:
            raise ValueError("Invalid image dimensions")

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self):
        return self.data.dtype

    def copy(self) -> "Image":
        if self.is_cuda:
            return self.__class__(
                data=self.data.copy(),
                format=self.format,
                frame_id=self.frame_id,
                ts=self.ts,
            )
        else:
            return self.__class__(
                data=self.data.copy(),
                format=self.format,
                frame_id=self.frame_id,
                ts=self.ts,
            )

    # ------------- Constructors -------------

    @classmethod
    def from_opencv(
        cls, cv_image: np.ndarray, format: ImageFormat = ImageFormat.BGR, **kwargs
    ) -> "Image":
        return cls(data=cv_image, format=format, **kwargs)

    @classmethod
    def from_numpy(
        cls, np_image: np.ndarray, format: ImageFormat = ImageFormat.BGR, to_gpu: bool = False, **kwargs
    ) -> "Image":
        return cls(data=np_image, format=format, **kwargs) if to_gpu == False else cls(data=cp.asarray(np_image), format=format, **kwargs)

    @classmethod
    def from_file(
        cls, filepath: str, format: ImageFormat = ImageFormat.BGR, to_gpu: bool = False
    ) -> "Image":
        """Load image from file (CPU), optionally move to GPU."""
        cv_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if cv_image is None:
            raise ValueError(f"Could not load image from {filepath}")

        # Detect format based on channels and dtype
        if cv_image.ndim == 2:
            detected_format = (
                ImageFormat.GRAY16 if cv_image.dtype == np.uint16 else ImageFormat.GRAY
            )
        elif cv_image.shape[2] == 3:
            detected_format = ImageFormat.BGR
        elif cv_image.shape[2] == 4:
            detected_format = ImageFormat.BGRA
        else:
            detected_format = format

        if to_gpu and HAS_CUDA:
            cv_image = cp.asarray(cv_image)

        return cls(data=cv_image, format=detected_format)

    @classmethod
    def from_depth(cls, depth_data, frame_id: str = "", ts: float = None) -> "Image":
        """Create Image from depth data (float32 array; NumPy or CuPy)."""
        if _is_cu(depth_data):
            if depth_data.dtype != cp.float32:
                depth_data = depth_data.astype(cp.float32)
        else:
            if depth_data.dtype != np.float32:
                depth_data = depth_data.astype(np.float32)

        return cls(
            data=depth_data,
            format=ImageFormat.DEPTH,
            frame_id=frame_id,
            ts=ts if ts is not None else time.time(),
        )

    # ------------- Device transfers -------------

    def to_cpu(self) -> "Image":
        return self.__class__(
            data=_to_cpu(self.data),
            format=self.format,
            frame_id=self.frame_id,
            ts=self.ts,
        )

    def to_cupy(self) -> "Image":
        if not HAS_CUDA:
            return self.copy()
        return self.__class__(
            data=_to_cu(self.data),
            format=self.format,
            frame_id=self.frame_id,
            ts=self.ts,
        )

    # ------------- Color space conversions -------------

    def to_opencv(self) -> np.ndarray:
        """Return a CPU array suitable for OpenCV (BGR if color)."""
        arr = _to_cpu(self.data)
        if self.format == ImageFormat.BGR:
            return arr
        elif self.format == ImageFormat.RGB:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif self.format == ImageFormat.RGBA:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif self.format == ImageFormat.BGRA:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif self.format in (ImageFormat.GRAY, ImageFormat.GRAY16, ImageFormat.DEPTH):
            return arr
        else:
            raise ValueError(f"Unsupported format conversion: {self.format}")

    def to_rgb(self) -> "Image":
        if self.format == ImageFormat.RGB:
            return self.copy()
        if self.format == ImageFormat.BGR:
            if self.is_cuda:
                rgb = _bgr_to_rgb_gpu(self.data)
            else:
                rgb = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
            return self.__class__(
                data=rgb, format=ImageFormat.RGB, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.RGBA:
            return self.copy()  # RGBA already has RGB order + alpha
        if self.format == ImageFormat.BGRA:
            if self.is_cuda:
                rgba = _bgra_to_rgba_gpu(self.data)
                return self.__class__(
                    data=rgba, format=ImageFormat.RGBA, frame_id=self.frame_id, ts=self.ts
                )
            else:
                rgba = cv2.cvtColor(self.data, cv2.COLOR_BGRA2RGBA)
                return self.__class__(
                    data=rgba, format=ImageFormat.RGBA, frame_id=self.frame_id, ts=self.ts
                )
        if self.format == ImageFormat.GRAY:
            if self.is_cuda:
                rgb = _gray_to_rgb_gpu(self.data)
            else:
                rgb = cv2.cvtColor(self.data, cv2.COLOR_GRAY2RGB)
            return self.__class__(
                data=rgb, format=ImageFormat.RGB, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.GRAY16:
            # Convert 16-bit grayscale to 8-bit then RGB for display-like consistency
            if self.is_cuda:
                gray8 = (self.data.astype(cp.float32) / 256.0).clip(0, 255).astype(cp.uint8)
                rgb = _gray_to_rgb_gpu(gray8)
            else:
                gray8 = (self.data / 256).astype(np.uint8)
                rgb = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
            return self.__class__(
                data=rgb, format=ImageFormat.RGB, frame_id=self.frame_id, ts=self.ts
            )
        raise ValueError(f"Unsupported conversion from {self.format} to RGB")

    def to_bgr(self) -> "Image":
        if self.format == ImageFormat.BGR:
            return self.copy()
        if self.format == ImageFormat.RGB:
            if self.is_cuda:
                bgr = _rgb_to_bgr_gpu(self.data)
            else:
                bgr = cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
            return self.__class__(
                data=bgr, format=ImageFormat.BGR, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.RGBA:
            if self.is_cuda:
                bgr = _rgba_to_bgra_gpu(self.data)[..., :3]  # RGBA->BGRA then drop A
            else:
                bgr = cv2.cvtColor(self.data, cv2.COLOR_RGBA2BGR)
            return self.__class__(
                data=bgr, format=ImageFormat.BGR, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.BGRA:
            if self.is_cuda:
                bgr = self.data[..., :3]
            else:
                bgr = cv2.cvtColor(self.data, cv2.COLOR_BGRA2BGR)
            return self.__class__(
                data=bgr, format=ImageFormat.BGR, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.GRAY:
            if self.is_cuda:
                bgr = _gray_to_rgb_gpu(self.data)  # 3-ch
                bgr = _rgb_to_bgr_gpu(bgr)
            else:
                bgr = cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
            return self.__class__(
                data=bgr, format=ImageFormat.BGR, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.GRAY16:
            if self.is_cuda:
                gray8 = (self.data.astype(cp.float32) / 256.0).clip(0, 255).astype(cp.uint8)
                bgr = _rgb_to_bgr_gpu(_gray_to_rgb_gpu(gray8))
            else:
                gray8 = (self.data / 256).astype(np.uint8)
                bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
            return self.__class__(
                data=bgr, format=ImageFormat.BGR, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.DEPTH:
            return self.copy()  # no change for depth
        raise ValueError(f"Unsupported conversion from {self.format} to BGR")

    def to_grayscale(self) -> "Image":
        if self.format in (ImageFormat.GRAY, ImageFormat.GRAY16, ImageFormat.DEPTH):
            return self.copy()
        if self.format == ImageFormat.BGR:
            if self.is_cuda:
                gray = _rgb_to_gray_gpu(_bgr_to_rgb_gpu(self.data))
            else:
                gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
            return self.__class__(
                data=gray, format=ImageFormat.GRAY, frame_id=self.frame_id, ts=self.ts
            )
        if self.format == ImageFormat.RGB:
            if self.is_cuda:
                gray = _rgb_to_gray_gpu(self.data)
            else:
                gray = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
            return self.__class__(
                data=gray, format=ImageFormat.GRAY, frame_id=self.frame_id, ts=self.ts
            )
        if self.format in (ImageFormat.RGBA, ImageFormat.BGRA):
            # drop alpha, then convert
            if self.is_cuda:
                rgb = (
                    self.data[..., :3]
                    if self.format == ImageFormat.RGBA
                    else _bgra_to_rgba_gpu(self.data)[..., :3]
                )
                gray = _rgb_to_gray_gpu(rgb)
            else:
                code = (
                    cv2.COLOR_RGBA2GRAY if self.format == ImageFormat.RGBA else cv2.COLOR_BGRA2GRAY
                )
                gray = cv2.cvtColor(self.data, code)
            return self.__class__(
                data=gray, format=ImageFormat.GRAY, frame_id=self.frame_id, ts=self.ts
            )
        raise ValueError(f"Unsupported conversion from {self.format} to GRAY")

    # ------------- Geometric ops -------------

    def resize(self, width: int, height: int, interpolation: int = cv2.INTER_LINEAR) -> "Image":
        """Resize the image; GPU bilinear if on device, else cv2."""
        if self.is_cuda:
            resized = _resize_bilinear_hwc_gpu(self.data, height, width)
            return self.__class__(
                data=resized, format=self.format, frame_id=self.frame_id, ts=self.ts
            )
        else:
            resized = cv2.resize(self.data, (width, height), interpolation=interpolation)
            return self.__class__(
                data=resized, format=self.format, frame_id=self.frame_id, ts=self.ts
            )

    def crop(self, x: int, y: int, width: int, height: int) -> "Image":
        x = max(0, min(x, self.width))
        y = max(0, min(y, self.height))
        x2 = min(x + width, self.width)
        y2 = min(y + height, self.height)
        cropped = self.data[y:y2, x:x2]
        return self.__class__(data=cropped, format=self.format, frame_id=self.frame_id, ts=self.ts)

    # ------------- I/O & encoding -------------

    def save(self, filepath: str) -> bool:
        """Save image to file (CPU path)."""
        cv_image = self.to_opencv()  # ensures CPU + BGR if color
        return cv2.imwrite(filepath, cv_image)

    def lcm_encode(self, frame_id: Optional[str] = None) -> bytes:
        """Convert to LCM Image message (CPU bytes)."""
        msg = LCMImage()

        # Header
        msg.header = Header()
        msg.header.seq = 0
        msg.header.frame_id = frame_id or self.frame_id

        if self.ts is not None:
            msg.header.stamp.sec = int(self.ts)
            msg.header.stamp.nsec = int((self.ts - int(self.ts)) * 1e9)
        else:
            now = time.time()
            msg.header.stamp.sec = int(now)
            msg.header.stamp.nsec = int((now - int(now)) * 1e9)

        arr = _to_cpu(self.data)
        msg.height = arr.shape[0]
        msg.width = arr.shape[1]
        msg.encoding = self._get_lcm_encoding()
        msg.is_bigendian = False
        msg.step = arr.shape[1] * arr.dtype.itemsize * (arr.shape[2] if arr.ndim == 3 else 1)

        image_bytes = arr.tobytes()
        msg.data_length = len(image_bytes)
        msg.data = image_bytes

        return msg.lcm_encode()

    @classmethod
    def lcm_decode(cls, data: bytes, **kwargs) -> "Image":
        """Create Image from LCM Image message (CPU array)."""
        msg = LCMImage.lcm_decode(data)
        format_info = cls._parse_encoding(msg.encoding)

        arr = np.frombuffer(msg.data, dtype=format_info["dtype"])
        if format_info["channels"] == 1:
            arr = arr.reshape((msg.height, msg.width))
        else:
            arr = arr.reshape((msg.height, msg.width, format_info["channels"]))

        return cls(
            data=arr,
            format=format_info["format"],
            frame_id=msg.header.frame_id if hasattr(msg, "header") else "",
            ts=msg.header.stamp.sec + msg.header.stamp.nsec / 1e9
            if hasattr(msg, "header") and getattr(msg.header, "stamp", None)
            else time.time(),
            **kwargs,
        )

    def agent_encode(self) -> str:
        """Encode image to base64 JPEG for agent consumption (CPU path)."""
        bgr_image = self.to_bgr()  # returns host if needed
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        success, buffer = cv2.imencode(".jpg", bgr_image.data, encode_param)
        if not success:
            raise ValueError("Failed to encode image as JPEG")

        import base64

        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    # ------------- Encoding helpers -------------

    def _get_row_step(self) -> int:
        """Bytes per row."""
        bytes_per_element = self.data.dtype.itemsize
        return self.width * bytes_per_element * self.channels

    def _get_lcm_encoding(self) -> str:
        """Get LCM encoding string from internal format and dtype."""
        dt = self.data.dtype
        if self.format == ImageFormat.GRAY:
            if dt == np.uint8 or (_is_cu(self.data) and dt == cp.uint8):
                return "mono8"
            elif dt == np.uint16 or (_is_cu(self.data) and dt == cp.uint16):
                return "mono16"
        elif self.format == ImageFormat.GRAY16:
            return "mono16"
        elif self.format == ImageFormat.RGB:
            return "rgb8"
        elif self.format == ImageFormat.RGBA:
            return "rgba8"
        elif self.format == ImageFormat.BGR:
            return "bgr8"
        elif self.format == ImageFormat.BGRA:
            return "bgra8"
        elif self.format == ImageFormat.DEPTH:
            if dt == np.float32 or (_is_cu(self.data) and dt == cp.float32):
                return "32FC1"
            elif dt == np.float64 or (_is_cu(self.data) and dt == cp.float64):
                return "64FC1"

        raise ValueError(f"Cannot determine LCM encoding for format={self.format}, dtype={dt}")

    @staticmethod
    def _parse_encoding(encoding: str) -> dict:
        """Parse LCM image encoding string to determine format and data type."""
        encoding_map = {
            "mono8": {"format": ImageFormat.GRAY, "dtype": np.uint8, "channels": 1},
            "mono16": {"format": ImageFormat.GRAY16, "dtype": np.uint16, "channels": 1},
            "rgb8": {"format": ImageFormat.RGB, "dtype": np.uint8, "channels": 3},
            "rgba8": {"format": ImageFormat.RGBA, "dtype": np.uint8, "channels": 4},
            "bgr8": {"format": ImageFormat.BGR, "dtype": np.uint8, "channels": 3},
            "bgra8": {"format": ImageFormat.BGRA, "dtype": np.uint8, "channels": 4},
            "32FC1": {"format": ImageFormat.DEPTH, "dtype": np.float32, "channels": 1},
            "32FC3": {"format": ImageFormat.RGB, "dtype": np.float32, "channels": 3},
            "64FC1": {"format": ImageFormat.DEPTH, "dtype": np.float64, "channels": 1},
        }
        if encoding not in encoding_map:
            raise ValueError(f"Unsupported encoding: {encoding}")
        return encoding_map[encoding]

    def as_memoryview(self) -> memoryview:
        """Return a memoryview of the image data for CPU shared memory transport.
        If the data is CuPy, this will copy to host (NumPy) first.
        """
        if isinstance(self.data, (bytes, bytearray)):
            return memoryview(self.data)
        elif isinstance(self.data, np.ndarray):
            return memoryview(self.data.tobytes())
        elif _is_cu(self.data):
            # Falls back to host copy, since memoryview can't wrap device pointers
            return memoryview(cp.asnumpy(self.data).tobytes())
        else:
            raise TypeError(f"Unsupported data type {type(self.data)}")

    def as_cuda_ipc_handle(self):
        """Return a CUDA IPC handle for GPU-resident images (CuPy).
        This should be used instead of as_memoryview for zero-copy GPU transport.
        """
        if not _is_cu(self.data):
            raise TypeError("CUDA IPC handle requested but data is not a CuPy array")

        # Ensure contiguous device buffer
        arr = _ascontig(self.data)
        ptr = arr.data.ptr
        size = arr.nbytes

        # Export IPC handle
        handle = cp.cuda.runtime.ipcGetMemHandle(ptr)
        return handle, size, arr.shape, arr.dtype

    @classmethod
    def from_memoryview(cls, mem: memoryview, width: int, height: int, format: "ImageFormat"):
        """Reconstruct an Image from a CPU memoryview (SharedMemory buffer)."""
        return cls(bytes(mem), width=width, height=height, format=format)

    @classmethod
    def from_cuda_ipc_handle(cls, handle, size, shape, dtype, width: int, height: int, format: "ImageFormat"):
        """Reconstruct an Image from a CUDA IPC handle."""
        ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)
        arr = cp.ndarray(shape=shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size, None), 0))
        return cls(arr, width=width, height=height, format=format)

    def as_cuda_ipc_bytes(self) -> bytes:
        """Return CUDA IPC handle + metadata as a serialized bytes object."""
        handle, size, shape, dtype = self.as_cuda_ipc_handle()
        payload = {
            "handle": handle,
            "size": size,
            "shape": shape,
            "dtype": str(dtype),
        }
        return pickle.dumps(payload)

    @classmethod
    def from_cuda_ipc_bytes(cls, payload: bytes, width: int, height: int, format: "ImageFormat"):
        """Reconstruct an Image from a serialized CUDA IPC payload."""
        obj = pickle.loads(payload)
        ptr = cp.cuda.runtime.ipcOpenMemHandle(obj["handle"])
        arr = cp.ndarray(
            shape=obj["shape"],
            dtype=np.dtype(obj["dtype"]),
            memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, obj["size"], None), 0)
        )
        return cls(arr, width=width, height=height, format=format)

    # ------------- Repr / equality -------------

    def __repr__(self) -> str:
        dev = "gpu" if self.is_cuda else "cpu"
        return (
            f"Image(shape={self.shape}, format={self.format.value}, "
            f"dtype={self.dtype}, dev={dev}, frame_id='{self.frame_id}', ts={self.ts})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Image):
            return False
        a = _to_cpu(self.data)
        b = _to_cpu(other.data)
        return (
            np.array_equal(a, b)
            and self.format == other.format
            and self.frame_id == other.frame_id
            and abs(self.ts - other.ts) < 1e-6
        )

    def __len__(self) -> int:
        return int(self.height * self.width)
