"""Native TensorRT FP16 detector for the isolated Jetson deployment.

The engine remains a TensorRT engine: preprocessing and NMS run on the CPU,
while the model execution uses reusable CUDA buffers and one CUDA stream.  No
ONNX Runtime execution provider is involved.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .detection_backend import DetectionBackend
from .models import Detection, DetectionConfig, DetectionResult, Frame
from .onnx_detector import OnnxPersonDetector


_SYSTEM_TENSORRT_PYTHON = Path("/usr/lib/python3.10/dist-packages")


def _import_tensorrt():
    """Import the matching vendor binding without changing Python's environment.

    Compatibility path: append only ``/usr/lib/python3.10/dist-packages`` so the
    installed JetPack TensorRT 10.3 binding can load inside the project venv.
    Do not ``pip install tensorrt`` and do not rewrite system packages.
    """
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("TensorRT system Python binding requires Python 3.10")
    candidate = str(_SYSTEM_TENSORRT_PYTHON)
    if candidate not in sys.path and _SYSTEM_TENSORRT_PYTHON.is_dir():
        sys.path.append(candidate)
    try:
        import tensorrt as trt
    except Exception as error:
        raise RuntimeError(f"Native TensorRT Python binding is unavailable: {type(error).__name__}") from error
    return trt


def tensorrt_binding_info() -> dict[str, object]:
    """Report binding source and CUDA readiness without logging secrets."""
    trt = _import_tensorrt()
    cuda = _CudaRuntime()
    count = ctypes.c_int()
    code = cuda._lib.cudaGetDeviceCount(ctypes.byref(count))
    if code != 0 or count.value < 1:
        raise RuntimeError(f"CUDA/GPU readiness failed code={code} count={count.value}")
    return {
        "tensorrt_python_version": trt.__version__,
        "tensorrt_binding_path": str(Path(trt.__file__).resolve()),
        "cuda_device_count": int(count.value),
        "gpu_ready": True,
    }


class _CudaRuntime:
    """Small checked CUDA Runtime wrapper; prevents per-frame allocation."""
    HOST_TO_DEVICE, DEVICE_TO_HOST = 1, 2

    def __init__(self) -> None:
        try:
            self._lib = ctypes.CDLL("libcudart.so.12")
        except OSError:
            self._lib = ctypes.CDLL("libcudart.so")
        pointer = ctypes.POINTER(ctypes.c_void_p)
        self._lib.cudaMalloc.argtypes = [pointer, ctypes.c_size_t]
        self._lib.cudaMalloc.restype = ctypes.c_int
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int
        self._lib.cudaStreamCreate.argtypes = [pointer]
        self._lib.cudaStreamCreate.restype = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                              ctypes.c_int, ctypes.c_void_p]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int
        self._lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.cudaGetDeviceCount.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"CUDA {operation} failed with error {code}")

    def stream(self) -> ctypes.c_void_p:
        result = ctypes.c_void_p()
        self._check(self._lib.cudaStreamCreate(ctypes.byref(result)), "stream creation")
        return result

    def malloc(self, size: int) -> ctypes.c_void_p:
        result = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(result), size), "allocation")
        return result

    def memcpy_async(self, destination, source: int, size: int, kind: int, stream) -> None:
        self._check(self._lib.cudaMemcpyAsync(
            destination, ctypes.c_void_p(source), size, kind, stream), "asynchronous copy"
        )

    def synchronize(self, stream) -> None:
        self._check(self._lib.cudaStreamSynchronize(stream), "stream synchronization")

    def free(self, pointer) -> None:
        if pointer:
            self._check(self._lib.cudaFree(pointer), "free")

    def destroy(self, stream) -> None:
        if stream:
            self._check(self._lib.cudaStreamDestroy(stream), "stream destruction")


class TensorRTPersonDetector(DetectionBackend):
    name, device, precision = "tensorrt", "cuda", "fp16"

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.image_size, self.confidence = config.image_size, config.confidence
        self.engine_path = Path(config.trt_engine_model or f"models/{Path(config.model).stem}_{config.image_size}_fp16.engine")
        self._lock = threading.Lock()
        self._trt = self._logger = self._runtime = self._engine = self._context = None
        self._cuda: _CudaRuntime | None = None
        self._stream = self._input_device = self._output_device = None
        self._input_name = self._output_name = ""
        self._host_input: np.ndarray | None = None
        self._host_output: np.ndarray | None = None
        self._canvas: np.ndarray | None = None
        self.error_summary: str | None = None
        self._next_retry = 0.0
        self.model_initialization_ms: float | None = None
        self.warmup_ms: float | None = None

    @property
    def retry_after(self) -> float:
        return self._next_retry

    @staticmethod
    def _volume(shape) -> int:
        values = tuple(int(value) for value in shape)
        if not values or any(value <= 0 for value in values):
            raise RuntimeError(f"TensorRT engine has unsupported dynamic or empty shape: {values}")
        return int(np.prod(values))

    def _load(self) -> None:
        with self._lock:
            if self._context is not None:
                return
            try:
                if "int8" in self.engine_path.name.lower():
                    raise RuntimeError("INT8 TensorRT engines are intentionally not supported")
                if not self.engine_path.is_file():
                    raise FileNotFoundError("TensorRT engine is missing; run scripts/export_yolo_trt.sh")
                started = time.monotonic()
                trt = _import_tensorrt()
                logger = trt.Logger(trt.Logger.ERROR)
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
                if engine is None:
                    raise RuntimeError("TensorRT could not deserialize the engine")
                context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError("TensorRT could not create an execution context")
                inputs, outputs = [], []
                for index in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(index)
                    mode = engine.get_tensor_mode(name)
                    (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
                if len(inputs) != 1 or len(outputs) != 1:
                    raise RuntimeError("TensorRT engine must have exactly one input and one output")
                input_name, output_name = inputs[0], outputs[0]
                input_shape, output_shape = engine.get_tensor_shape(input_name), engine.get_tensor_shape(output_name)
                if tuple(int(value) for value in input_shape) != (1, 3, self.image_size, self.image_size):
                    raise RuntimeError(f"TensorRT engine input shape {tuple(input_shape)} does not match 1x3x{self.image_size}x{self.image_size}")
                input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
                output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output_name)))
                if input_dtype != np.dtype(np.float32) or output_dtype != np.dtype(np.float32):
                    raise RuntimeError("TensorRT engine bindings must use FP32 input/output tensors")
                host_input = np.zeros(tuple(int(value) for value in input_shape), dtype=input_dtype)
                host_output = np.empty(tuple(int(value) for value in output_shape), dtype=output_dtype)
                cuda = _CudaRuntime(); stream = cuda.stream()
                input_device = cuda.malloc(host_input.nbytes); output_device = cuda.malloc(host_output.nbytes)
                if not context.set_tensor_address(input_name, int(input_device.value)):
                    raise RuntimeError("TensorRT could not bind input CUDA memory")
                if not context.set_tensor_address(output_name, int(output_device.value)):
                    raise RuntimeError("TensorRT could not bind output CUDA memory")
                self._trt, self._logger, self._runtime, self._engine, self._context = trt, logger, runtime, engine, context
                self._cuda, self._stream = cuda, stream
                self._input_device, self._output_device = input_device, output_device
                self._input_name, self._output_name = input_name, output_name
                self._host_input, self._host_output = host_input, host_output
                self._canvas = np.empty((self.image_size, self.image_size, 3), dtype=np.uint8)
                self.model_initialization_ms = (time.monotonic() - started) * 1000
            except Exception as error:
                self.close()
                self.error_summary = f"{type(error).__name__}: {str(error)[:200]}"
                self._next_retry = time.monotonic() + 5
                raise RuntimeError(self.error_summary) from error

    def load(self) -> None:
        self._load()

    def preflight(self) -> dict[str, object]:
        """Startup-safe TensorRT binding and engine path report."""
        info = tensorrt_binding_info()
        info.update({
            "engine": str(self.engine_path),
            "engine_exists": self.engine_path.is_file(),
            "precision": self.precision,
            "backend": self.name,
        })
        return info

    def _execute(self) -> None:
        assert self._cuda and self._stream and self._host_input is not None and self._host_output is not None
        assert self._input_device and self._output_device and self._context is not None
        self._cuda.memcpy_async(self._input_device, self._host_input.ctypes.data, self._host_input.nbytes,
                                _CudaRuntime.HOST_TO_DEVICE, self._stream)
        if not self._context.execute_async_v3(int(self._stream.value)):
            raise RuntimeError("TensorRT execution failed")
        self._cuda.memcpy_async(ctypes.c_void_p(self._host_output.ctypes.data), int(self._output_device.value),
                                self._host_output.nbytes, _CudaRuntime.DEVICE_TO_HOST, self._stream)
        self._cuda.synchronize(self._stream)

    def warmup(self) -> dict[str, object]:
        self._load(); assert self._host_input is not None and self._host_output is not None
        started = time.monotonic()
        self._host_input.fill(0)
        self._execute()
        if not np.isfinite(self._host_output).all():
            raise RuntimeError("TensorRT warm-up returned non-finite output")
        self.warmup_ms = (time.monotonic() - started) * 1000
        details = {
            "backend": self.name, "device": self.device, "precision": self.precision,
            "execution_provider": "TensorRT", "engine": str(self.engine_path),
            "model_load_ms": self.model_initialization_ms, "warmup_ms": self.warmup_ms,
            "input_shape": f"1x3x{self.image_size}x{self.image_size}",
            "input_name": self._input_name, "output_name": self._output_name,
            "reuses_engine": True, "reuses_context": True, "reuses_cuda_stream": True,
            "reuses_buffers": True,
        }
        details.update(tensorrt_binding_info())
        return details

    def _prepare(self, image: np.ndarray) -> tuple[float, int, int]:
        import cv2
        assert self._host_input is not None and self._canvas is not None
        h, w = image.shape[:2]
        ratio = min(self.image_size / w, self.image_size / h)
        resized = (round(w * ratio), round(h * ratio))
        left = (self.image_size - resized[0]) // 2
        top = (self.image_size - resized[1]) // 2
        self._canvas.fill(114)
        self._canvas[top:top + resized[1], left:left + resized[0]] = cv2.resize(image, resized)
        rgb = self._canvas[:, :, ::-1]
        self._host_input[0] = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        self._host_input /= np.float32(255.0)
        return ratio, left, top

    @staticmethod
    def _decode_person_boxes(output: np.ndarray, confidence: float) -> tuple[list[tuple[float, float, float, float]], list[float]]:
        """Decode YOLO11 ``1×84×8400`` (xywh + 80 class scores); person is class 0."""
        rows = output[0] if output.ndim == 3 else output
        if rows.shape[0] in {84, 85} and rows.shape[0] < rows.shape[1]:
            rows = rows.T
        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        for row in rows:
            if row.size < 84 or not np.isfinite(row[:5]).all():
                continue
            person = float(row[4])
            if person < confidence:
                continue
            cx, cy, bw, bh = (float(value) for value in row[:4])
            if not all(np.isfinite(value) for value in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                continue
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
            scores.append(person)
        return boxes, scores

    def infer(self, frames: list[tuple[str, Frame]]) -> list[DetectionResult]:
        self._load()
        assert self._host_output is not None
        results = []
        # One context/buffer set intentionally serializes this camera-owned backend.
        for camera_id, frame in frames:
            started = time.monotonic()
            ratio, left, top = self._prepare(frame.image)
            prepared = time.monotonic()
            self._execute()
            output = self._host_output.copy()
            inferred = time.monotonic()
            boxes, scores = self._decode_person_boxes(output, self.confidence)
            detections = []
            if boxes:
                # NMS once here; the engine does not embed NMS for this export.
                for index in OnnxPersonDetector._nms(np.asarray(boxes), np.asarray(scores)):
                    x1, y1, x2, y2 = boxes[index]
                    x1, x2 = (x1 - left) / ratio, (x2 - left) / ratio
                    y1, y2 = (y1 - top) / ratio, (y2 - top) / ratio
                    h, w = frame.image.shape[:2]
                    x1, x2 = sorted((max(0.0, min(w, x1)), max(0.0, min(w, x2))))
                    y1, y2 = sorted((max(0.0, min(h, y1)), max(0.0, min(h, y2))))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    sw, sh = frame.source_width or w, frame.source_height or h
                    detections.append(Detection(
                        camera_id,
                        (x1 * sw / w, y1 * sh / h, x2 * sw / w, y2 * sh / h),
                        scores[index],
                        0,
                    ))
            completed = time.monotonic()
            h, w = frame.image.shape[:2]
            results.append(DetectionResult(
                camera_id, tuple(detections), frame.captured_at, started, completed,
                (prepared - started) * 1000, (inferred - prepared) * 1000, (completed - inferred) * 1000,
                frame.sequence, frame.source_width or w, frame.source_height or h,
            ))
        return results

    def detect(self, camera_id: str, frame: Frame) -> DetectionResult:
        return self.infer([(camera_id, frame)])[0]

    def detect_batch(self, frames):
        return self.infer(frames)

    def close(self) -> None:
        cuda, stream = self._cuda, self._stream
        if cuda is not None:
            for pointer in (self._input_device, self._output_device):
                try: cuda.free(pointer)
                except Exception: pass
            try: cuda.destroy(stream)
            except Exception: pass
        self._cuda = self._stream = self._input_device = self._output_device = None
        self._host_input = self._host_output = self._canvas = None
        self._context = self._engine = self._runtime = self._logger = self._trt = None
