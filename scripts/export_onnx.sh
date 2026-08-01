#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly MODEL="$ROOT/yolo11n.pt"
readonly INT8_CALIBRATION_DIR="${INT8_CALIBRATION_DIR:-$ROOT/run/int8-calibration}"
case "${1:-}" in
  "") readonly OUTPUT="$ROOT/models/yolo11n_640.onnx" ;;
  --int8) readonly OUTPUT="$ROOT/models/yolo11n_640_int8.onnx" ;;
  *) echo "Usage: $0 [--int8]" >&2; exit 2 ;;
esac
[[ -x "$VENV_PYTHON" ]] || { echo "Missing isolated Python." >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "Missing yolo11n.pt." >&2; exit 1; }
mkdir -p "$ROOT/models"
PYTHONNOUSERSITE=1 "$VENV_PYTHON" - "$MODEL" "$OUTPUT" "${1:-}" "$INT8_CALIBRATION_DIR" <<'PY'
import os
import shutil
import sys
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO

model, output = map(Path, sys.argv[1:3])
int8 = sys.argv[3] == "--int8"
calibration_dir = Path(sys.argv[4])
exported = Path(YOLO(str(model)).export(format="onnx", imgsz=640, batch=1, dynamic=False,
                                         simplify=False, opset=None, verbose=False))
output.unlink(missing_ok=True)
shutil.move(str(exported), output)
if int8:
    import cv2
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    images = sorted(path for pattern in ("*.jpg", "*.jpeg", "*.png") for path in calibration_dir.glob(pattern))
    if len(images) < 8:
        raise RuntimeError(f"INT8 export requires at least 8 calibration images in {calibration_dir}")

    class CalibrationFrames(CalibrationDataReader):
        def __init__(self, paths):
            self.paths, self.index, self.input_name = paths, 0, "images"
        def get_next(self):
            if self.index >= len(self.paths):
                return None
            image = cv2.imread(str(self.paths[self.index]))
            self.index += 1
            if image is None:
                return self.get_next()
            height, width = image.shape[:2]
            ratio = min(640 / width, 640 / height)
            resized = (round(width * ratio), round(height * ratio))
            left, top = (640 - resized[0]) // 2, (640 - resized[1]) // 2
            canvas = np.full((640, 640, 3), 114, np.uint8)
            canvas[top:top + resized[1], left:left + resized[0]] = cv2.resize(image, resized)
            tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None]
            tensor /= np.float32(255.0)
            return {self.input_name: tensor}

    quantized = output.with_name(output.stem + ".tmp.onnx")
    quantize_static(
        str(output), str(quantized), CalibrationFrames(images),
        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
    )
    quantized.replace(output)
onnx.checker.check_model(str(output))
session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
input_type = session.get_inputs()[0].type
expected_type = "tensor(float)"
if input_type != expected_type:
    raise RuntimeError(f"Exported model input is {input_type}, expected {expected_type}")
session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 640, 640), np.float32)})
if int8:
    graph = onnx.load(str(output)).graph
    has_int8_weights = any(item.data_type in {onnx.TensorProto.INT8, onnx.TensorProto.UINT8}
                           for item in graph.initializer)
    has_quantized_nodes = any(item.op_type in {"QuantizeLinear", "DequantizeLinear", "QLinearConv"}
                              for item in graph.node)
    if not has_int8_weights or not has_quantized_nodes:
        raise RuntimeError("INT8 conversion produced an unquantized graph")
graph = onnx.load(str(output)).opset_import[0].version
print(f"ONNX model: {output}")
print(f"Size: {output.stat().st_size} bytes")
print(f"Opset: {graph}")
print(f"Provider: {session.get_providers()[0]}")
print(f"Precision: {'int8' if int8 else 'fp32'}")
PY
