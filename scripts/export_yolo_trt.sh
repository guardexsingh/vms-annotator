#!/usr/bin/env bash
# Build a static-batch TensorRT FP16 engine from the validated FP32 ONNX model.
# Never builds from INT8 ONNX. Never publishes a failed/partial engine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ROOT
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly ONNX_MODEL="${TRT_ONNX_MODEL:-models/yolo11n_640.onnx}"
readonly ENGINE_MODEL="${TRT_ENGINE_MODEL:-models/yolo11n_640_fp16.engine}"
readonly WORKSPACE_MB="${TRT_WORKSPACE_MB:-2048}"
readonly WARMUP_MS="${TRT_WARMUP_MS:-1000}"
readonly BENCHMARK_ITERATIONS="${TRT_BENCHMARK_ITERATIONS:-500}"
readonly BENCHMARK_DURATION_SECONDS="${TRT_BENCHMARK_DURATION_SECONDS:-10}"
readonly LOCK_FILE="$ROOT/logs/trt_export.lock"
readonly MIN_FREE_MB="${TRT_MIN_FREE_MB:-2048}"

cd "$ROOT"

[[ -x "$VENV_PYTHON" ]] || { echo "Missing project Python: $VENV_PYTHON" >&2; exit 1; }
[[ "$WORKSPACE_MB" =~ ^[1-9][0-9]*$ ]] || { echo "TRT_WORKSPACE_MB must be a positive integer." >&2; exit 1; }
[[ "$WARMUP_MS" =~ ^[1-9][0-9]*$ ]] || { echo "TRT_WARMUP_MS must be a positive integer." >&2; exit 1; }
[[ "$BENCHMARK_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "TRT_BENCHMARK_ITERATIONS must be a positive integer." >&2; exit 1; }
[[ "$BENCHMARK_DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "TRT_BENCHMARK_DURATION_SECONDS must be a positive integer." >&2; exit 1; }

resolve_project_path() {
  local path="$1"
  [[ "$path" = /* ]] && printf '%s\n' "$path" || printf '%s/%s\n' "$ROOT" "$path"
}

readonly ONNX_PATH="$(resolve_project_path "$ONNX_MODEL")"
readonly ENGINE_PATH="$(resolve_project_path "$ENGINE_MODEL")"
readonly ENGINE_DIRECTORY="$(dirname "$ENGINE_PATH")"
readonly ENGINE_BASENAME="$(basename "$ENGINE_PATH")"

[[ -f "$ONNX_PATH" ]] || { echo "Missing FP32 ONNX model: $ONNX_PATH" >&2; exit 1; }
[[ "${ONNX_PATH,,}" != *int8* ]] || { echo "TensorRT FP16 export refuses an INT8 ONNX source." >&2; exit 1; }
[[ "$ENGINE_BASENAME" != *int8* ]] || { echo "TensorRT FP16 export refuses an INT8 engine destination." >&2; exit 1; }

mkdir -p "$ENGINE_DIRECTORY" "$ROOT/logs"

# Exclusive build lock: two engine builds must never run concurrently.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another TensorRT engine build already holds $LOCK_FILE" >&2
  exit 1
fi

find_trtexec() {
  if [[ -n "${TRTEXEC:-}" && -x "${TRTEXEC}" ]]; then
    printf '%s\n' "$TRTEXEC"
    return 0
  fi
  if command -v trtexec >/dev/null 2>&1; then
    command -v trtexec
    return 0
  fi
  for candidate in \
    /usr/src/tensorrt/bin/trtexec \
    /usr/local/bin/trtexec \
    /opt/nvidia/tensorrt/bin/trtexec
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  find /usr -type f -name trtexec -perm -u+x 2>/dev/null | head -n 1 || true
}

readonly TRTEXEC="$(find_trtexec)"
[[ -n "$TRTEXEC" && -x "$TRTEXEC" ]] || { echo "TensorRT trtexec was not found." >&2; exit 1; }

readonly TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly LOG_FILE="$ROOT/logs/trt_build_${TIMESTAMP}.log"
readonly BENCHMARK_LOG="$ROOT/logs/trt_benchmark_${TIMESTAMP}.log"
readonly TEMP_ENGINE="$ENGINE_DIRECTORY/.${ENGINE_BASENAME}.${TIMESTAMP}.$$.tmp"
readonly BUILD_STARTED="$(date +%s)"

cleanup_temp() {
  rm -f "$TEMP_ENGINE"
}
trap cleanup_temp EXIT

INPUT_NAME="$("$VENV_PYTHON" - "$ONNX_PATH" <<'PY'
from pathlib import Path
import sys
import onnx

path = Path(sys.argv[1])
model = onnx.load(str(path))
onnx.checker.check_model(model)
inputs = model.graph.input
if len(inputs) != 1:
    raise SystemExit("TensorRT export requires exactly one model input")
input_value = inputs[0]
shape = [dimension.dim_value for dimension in input_value.type.tensor_type.shape.dim]
if shape != [1, 3, 640, 640]:
    raise SystemExit(f"Expected static input 1x3x640x640, found {shape}")
if any(value <= 0 for value in shape):
    raise SystemExit(f"Dynamic ONNX input shapes are not allowed: {shape}")
if input_value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
    raise SystemExit("TensorRT FP16 engine source must have an FP32 input")
print(input_value.name)
print(f"ONNX input: {input_value.name} {shape} FP32", file=sys.stderr)
print("ONNX outputs:", ", ".join(value.name for value in model.graph.output), file=sys.stderr)
PY
)"
readonly INPUT_NAME

{
  printf '=== TensorRT FP16 engine export ===\n'
  printf 'Project root: %s\n' "$ROOT"
  printf 'Source ONNX: %s\n' "$ONNX_PATH"
  printf 'ONNX SHA256: %s\n' "$(sha256sum "$ONNX_PATH" | awk '{print $1}')"
  printf 'Engine output: %s\n' "$ENGINE_PATH"
  printf 'Workspace: workspace:%sM\n' "$WORKSPACE_MB"
  printf 'Input name: %s\n' "$INPUT_NAME"
  printf 'Static shapes: enabled (no --min/opt/maxShapes)\n'
  printf 'Precision: FP16 (+FP32 where required)\n'
  printf 'Batch: static 1\n'
  printf 'Image size: 640x640\n'
  printf 'TensorRT executable: %s\n' "$TRTEXEC"
  "$TRTEXEC" --help 2>&1 | sed -n '1,3p' || true
  printf '\n--- Platform ---\n'
  uname -a || true
  if [[ -r /etc/nv_tegra_release ]]; then
    head -n 2 /etc/nv_tegra_release || true
  fi
  if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -q || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  fi
  "$VENV_PYTHON" - <<'PY' || true
import sys
from pathlib import Path
candidate = "/usr/lib/python3.10/dist-packages"
if candidate not in sys.path:
    sys.path.append(candidate)
try:
    import tensorrt as trt
    print(f"TensorRT Python: {trt.__version__}")
    print(f"TensorRT binding: {trt.__file__}")
except Exception as error:
    print(f"TensorRT Python unavailable: {type(error).__name__}: {error}")
PY
  printf '\n--- GPU readiness ---\n'
  if ! "$VENV_PYTHON" - <<'PY'
import ctypes
import sys
try:
    lib = ctypes.CDLL("libcudart.so.12")
except OSError:
    lib = ctypes.CDLL("libcudart.so")
count = ctypes.c_int()
code = lib.cudaGetDeviceCount(ctypes.byref(count))
if code != 0 or count.value < 1:
    raise SystemExit(f"CUDA device enumeration failed code={code} count={count.value}")
print(f"CUDA devices: {count.value}")
PY
  then
    echo "GPU access verification failed." >&2
    exit 1
  fi
  printf '\n--- Disk space ---\n'
  df -h "$ENGINE_DIRECTORY"
  available_kb="$(df -Pk "$ENGINE_DIRECTORY" | awk 'NR==2 {print $4}')"
  available_mb=$((available_kb / 1024))
  if (( available_mb < MIN_FREE_MB )); then
    echo "Insufficient free disk space: ${available_mb} MiB < ${MIN_FREE_MB} MiB" >&2
    exit 1
  fi
  printf 'Free space: %s MiB\n' "$available_mb"
} | tee "$LOG_FILE"

build_command=(
  "$TRTEXEC"
  "--onnx=$ONNX_PATH"
  "--saveEngine=$TEMP_ENGINE"
  --fp16
  "--memPoolSize=workspace:${WORKSPACE_MB}M"
  "--warmUp=$WARMUP_MS"
  "--iterations=$BENCHMARK_ITERATIONS"
  "--duration=$BENCHMARK_DURATION_SECONDS"
)

{
  printf '\n=== Build command ===\n'
  printf '%q ' "${build_command[@]}"
  printf '\n\n'
} | tee -a "$LOG_FILE"

set +e
"${build_command[@]}" 2>&1 | tee -a "$LOG_FILE"
build_status=${PIPESTATUS[0]}
set -e
if [[ "$build_status" -ne 0 ]]; then
  echo "TensorRT engine build failed with status $build_status" >&2
  exit "$build_status"
fi
[[ -s "$TEMP_ENGINE" ]] || { echo "TensorRT did not create a temporary engine." >&2; exit 1; }

benchmark_command=(
  "$TRTEXEC"
  "--loadEngine=$TEMP_ENGINE"
  "--warmUp=$WARMUP_MS"
  "--iterations=$BENCHMARK_ITERATIONS"
  "--duration=$BENCHMARK_DURATION_SECONDS"
)

{
  printf '=== Benchmark command ===\n'
  printf '%q ' "${benchmark_command[@]}"
  printf '\n\n'
} | tee "$BENCHMARK_LOG"

set +e
"${benchmark_command[@]}" 2>&1 | tee -a "$BENCHMARK_LOG" | tee -a "$LOG_FILE"
benchmark_status=${PIPESTATUS[0]}
set -e
if [[ "$benchmark_status" -ne 0 ]]; then
  echo "TensorRT engine benchmark failed with status $benchmark_status" >&2
  exit "$benchmark_status"
fi
if ! grep -Eq '&&&& PASSED|Throughput:|GPU Compute Time' "$BENCHMARK_LOG"; then
  echo "TensorRT benchmark log does not show a successful measurement." >&2
  exit 1
fi

# Atomic publish only after a successful build + benchmark.
mv -f "$TEMP_ENGINE" "$ENGINE_PATH"
trap - EXIT
readonly BUILD_FINISHED="$(date +%s)"
readonly BUILD_DURATION="$((BUILD_FINISHED - BUILD_STARTED))"
{
  printf '\n=== Published engine ===\n'
  printf 'Engine: %s\n' "$ENGINE_PATH"
  printf 'Size: %s bytes\n' "$(stat -c '%s' "$ENGINE_PATH")"
  printf 'SHA256: %s\n' "$(sha256sum "$ENGINE_PATH" | awk '{print $1}')"
  printf 'Build duration: %s seconds\n' "$BUILD_DURATION"
  printf 'Build log: %s\n' "$LOG_FILE"
  printf 'Benchmark log: %s\n' "$BENCHMARK_LOG"
} | tee -a "$LOG_FILE"
