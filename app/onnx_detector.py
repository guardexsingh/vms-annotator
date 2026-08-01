"""CPU ONNX Runtime person detector with project-local preprocessing/NMS."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from .detection_backend import DetectionBackend
from .models import Detection, DetectionConfig, DetectionResult, Frame


class OnnxPersonDetector(DetectionBackend):
    name, device = "onnx", "cpu"

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.precision = "int8" if config.precision == "int8" else "fp32"
        self.image_size, self.confidence = config.image_size, config.confidence
        precision_suffix = "_int8" if self.precision == "int8" else ""
        self.model_path = Path(
            config.onnx_model or f"models/{Path(config.model).stem}_{config.image_size}{precision_suffix}.onnx"
        )
        self._session = None
        self._input_name = "images"
        self._input_dtype = np.float32
        self._lock = threading.Lock()
        self.error_summary: str | None = None
        self._next_retry = 0.0
        self.model_initialization_ms: float | None = None
        self.warmup_ms: float | None = None

    @property
    def retry_after(self) -> float: return self._next_retry

    def _load(self) -> None:
        with self._lock:
            if self._session is not None: return
            try:
                if not self.model_path.is_file():
                    raise FileNotFoundError("ONNX model is missing; run scripts/export_onnx.sh")
                import onnx
                import onnxruntime as ort
                onnx.checker.check_model(str(self.model_path))
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                started = time.monotonic()
                session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                               providers=["CPUExecutionProvider"])
                if "CPUExecutionProvider" not in session.get_providers():
                    raise RuntimeError("CPUExecutionProvider is unavailable")
                input_type = session.get_inputs()[0].type
                # Dynamic INT8 quantization stores quantized weights while its
                # public image input remains float32 and is quantized in-graph.
                expected_type = "tensor(float)"
                if input_type != expected_type:
                    raise RuntimeError(
                        f"ONNX model input is {input_type}, but configured precision is {self.precision}"
                    )
                if self.precision == "int8":
                    graph = onnx.load(str(self.model_path)).graph
                    has_int8_weights = any(
                        initializer.data_type in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
                        for initializer in graph.initializer
                    )
                    has_quantized_nodes = any(
                        node.op_type in {"DynamicQuantizeLinear", "QuantizeLinear", "QLinearConv", "ConvInteger"}
                        for node in graph.node
                    )
                    if not has_int8_weights or not has_quantized_nodes:
                        raise RuntimeError("Configured INT8 ONNX model is not quantized")
                self._session, self._input_name = session, session.get_inputs()[0].name
                self._input_dtype = np.float32
                self.model_initialization_ms = (time.monotonic() - started) * 1000
            except Exception as error:
                self.error_summary = f"{type(error).__name__}: {str(error)[:200]}"
                self._next_retry = time.monotonic() + 5
                raise RuntimeError(self.error_summary) from error

    def load(self) -> None: self._load()

    def warmup(self) -> dict[str, object]:
        self._load(); assert self._session is not None
        started = time.monotonic()
        self._session.run(None, {self._input_name: np.zeros((1, 3, self.image_size, self.image_size), self._input_dtype)})
        self.warmup_ms = (time.monotonic() - started) * 1000
        return {"backend": self.name, "device": self.device, "precision": self.precision,
                "execution_provider": "CPUExecutionProvider", "model": str(self.model_path),
                "model_load_ms": self.model_initialization_ms, "warmup_ms": self.warmup_ms,
                "input_shape": f"1x3x{self.image_size}x{self.image_size}"}

    def _prepare(self, image: np.ndarray):
        import cv2
        h, w = image.shape[:2]; ratio = min(self.image_size / w, self.image_size / h)
        resized = (round(w * ratio), round(h * ratio)); left = (self.image_size-resized[0]) // 2; top = (self.image_size-resized[1]) // 2
        canvas = np.full((self.image_size, self.image_size, 3), 114, np.uint8)
        canvas[top:top+resized[1], left:left+resized[0]] = cv2.resize(image, resized)
        tensor = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1), dtype=self._input_dtype
        )[None]
        tensor /= self._input_dtype(255.0)
        return tensor, ratio, left, top

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float = .7) -> list[int]:
        order, keep = scores.argsort()[::-1], []
        while order.size:
            index = int(order[0]); keep.append(index)
            if order.size == 1: break
            rest = order[1:]; box, other = boxes[index], boxes[rest]
            x1, y1 = np.maximum(box[0], other[:, 0]), np.maximum(box[1], other[:, 1]); x2, y2 = np.minimum(box[2], other[:, 2]), np.minimum(box[3], other[:, 3])
            overlap = np.maximum(0, x2-x1)*np.maximum(0, y2-y1); union = (box[2]-box[0])*(box[3]-box[1]) + (other[:,2]-other[:,0])*(other[:,3]-other[:,1])-overlap
            order = rest[overlap / np.maximum(union, 1e-9) <= iou]
        return keep

    def infer(self, frames: list[tuple[str, Frame]]) -> list[DetectionResult]:
        self._load(); assert self._session is not None
        results = []
        for camera_id, frame in frames:
            started = time.monotonic(); tensor, ratio, left, top = self._prepare(frame.image); prepared = time.monotonic()
            output = np.asarray(self._session.run(None, {self._input_name: tensor})[0]); inferred = time.monotonic()
            rows = output[0] if output.ndim == 3 else output
            if rows.shape[0] in {84, 85} and rows.shape[0] < rows.shape[1]: rows = rows.T
            boxes, scores = [], []
            for row in rows:
                if row.size < 84 or not np.isfinite(row[:5]).all(): continue
                confidence = float(row[4])
                if confidence < self.confidence: continue  # exported YOLO11 output: xywh + 80 class scores
                cx, cy, bw, bh = (float(value) for value in row[:4]); boxes.append((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)); scores.append(confidence)
            detections = []
            if boxes:
                for index in self._nms(np.asarray(boxes), np.asarray(scores)):
                    x1, y1, x2, y2 = boxes[index]; x1, x2 = (x1-left)/ratio, (x2-left)/ratio; y1, y2 = (y1-top)/ratio, (y2-top)/ratio
                    h, w = frame.image.shape[:2]
                    x1, x2 = sorted((max(0., min(w, x1)), max(0., min(w, x2))))
                    y1, y2 = sorted((max(0., min(h, y1)), max(0., min(h, y2))))
                    if x2 <= x1 or y2 <= y1: continue
                    sw, sh = frame.source_width or w, frame.source_height or h
                    detections.append(Detection(camera_id, (x1*sw/w, y1*sh/h, x2*sw/w, y2*sh/h), scores[index], 0))
            completed = time.monotonic(); h, w = frame.image.shape[:2]
            results.append(DetectionResult(camera_id, tuple(detections), frame.captured_at, started, completed,
                (prepared-started)*1000, (inferred-prepared)*1000, (completed-inferred)*1000,
                frame.sequence, frame.source_width or w, frame.source_height or h))
        return results

    def detect(self, camera_id: str, frame: Frame) -> DetectionResult: return self.infer([(camera_id, frame)])[0]
    def detect_batch(self, frames): return self.infer(frames)
    def close(self) -> None: self._session = None
