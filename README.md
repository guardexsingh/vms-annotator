# Local VMS person-detection experiment

This standalone three-camera prototype does not modify or control the Guardex
production VMS.

## Frozen video path

Normal startup preserves the validated path:

```text
HEVC camera/NVR
  → experiment MediaMTX source-on-demand pull
  → unchanged HEVC RTSP live/<camera-id>
  → unchanged HEVC WHEP
  → Chrome
```

There is no normal-mode video relay, Python raw-frame video pipe, compositor,
annotation encoder, H.264 conversion, or fallback. Detection readiness cannot
block video, and changing detection never restarts a WHEP session.

## On-demand detection path

Detection is off for every camera at application startup. A tile's top-right
checkmark starts this independent branch for that camera:

```text
local MediaMTX RTSP live/<selected-camera>
  → one AI-only FFmpeg decoder sampled to about 1 FPS
  → one replaceable latest-frame slot
  → YOLO11n correction, image size 640, COCO person class 0, confidence 0.40
  → Ultralytics 8.3.0 ByteTrack correction at 1 FPS
  → elapsed-time Kalman prediction at 5 FPS
  → normalized track metadata over one browser WebSocket
  → transparent browser canvas
```

At most one camera is selected. Selecting another tile stops and releases the
old decoder, inference scheduler, model, latest-frame slot, and ByteTrack
state before starting a fresh session. Clicking the selected tile disables
detection. Every connected browser shares this single backend state.

The controller is idempotent:

```http
POST /api/detection/active-camera
Content-Type: application/json

{"camera_id":"cam02"}
```

Use `{"camera_id":null}` to disable it. `GET
/api/detection/active-camera` returns the current state. Camera URLs and
credentials are never returned.

The WebSocket sends `active_camera`, `detector_status`, `tracks`, and
`clear_tracks` messages. Each slow client has at most one pending metadata
message per camera. Track boxes use normalized coordinates and contain
ByteTrack ID, confidence, active/lost state, and confirmation age—never frames,
images, model tensors, URLs, or credentials.

YOLO and tracking prediction are separate loops. YOLO supplies a fresh
correction about once per second; the independent prediction ticker advances
only existing tracks every 200 ms. It never passes a prior detection array
back to ByteTrack, cannot create a track, and cannot raise confidence. A
correction reports `source: yolo`, `predicted: false`; an intermediate update
reports `source: bytetrack_prediction`, `track_state: predicted`,
`predicted: true`. Browser metadata remains bounded near five Hz: a prediction
near a fresh correction is omitted rather than producing a six-Hz stream.

## ByteTrack persistence

The project uses only Ultralytics ByteTrack; it has no optical flow, CSRT, KCF,
MOSSE, MedianFlow, or other visual tracker. A small local `lap.py`
compatibility module supplies ByteTrack's assignment call through the locked
SciPy Hungarian solver because this ARM64 environment does not contain
`lapx`. The tracker itself remains Ultralytics 8.3.0.

Ultralytics' public `STrack.predict()` uses a fixed single-step transition.
The project-local adapter retains its `KalmanFilterXYAH` state but constructs a
transition matrix with the measured monotonic delta (`dt`) and scales process
noise by `sqrt(dt)`. Five 200 ms advances therefore equal one one-second
advance; the installed package is not edited. Coordinates are finite-checked,
clamped to the frame, speed-bounded, and held within a small motion dead zone.
Display stops after 1500 ms without correction and state is removed at 2000 ms.

The checked-in starting values are:

```yaml
tracking:
  enabled: true
  tracker: bytetrack
  track_high_thresh: 0.50
  track_low_thresh: 0.10
  new_track_thresh: 0.50
  match_thresh: 0.80
  track_buffer: 2
  fuse_score: true
  hold_box_ms: 1500
  remove_track_ms: 2000
  prediction_fps: 5
  prediction_deadzone_norm: 0.002
  max_prediction_displacement_norm_per_second: 0.25
```

`track_buffer` is interpreted as tracker update cycles. Since ByteTrack is
updated only after a fresh approximately one-FPS YOLO result, the value `2`
means two missed update cycles, not 30 native video frames. A lost box is shown
for at most 1500 ms and all unconfirmed state is removed by 2000 ms. Camera
switches and AI decoder reconnects clear it immediately.

The latest-frame scheduler permits at most 1500 ms from local capture to a
completed result. This accommodates the phase difference between a one-FPS
capture sample and a one-FPS inference opportunity on this CPU while remaining
bounded below the 2000 ms hard track-removal limit. It never queues a delayed
job or runs catch-up inference.

This design does not claim native-frame tracking. Stationary people should be
the strongest case. Slow movement can update in one-second steps. Fast motion,
entries, exits, occlusion, and crossing people can produce lag or ID changes;
entry delay can approach one second plus inference, while exit boxes can last
up to the configured hold/removal limit.

## UI

The accessible tile control has four states:

- `○` / Off — enable detection.
- `…` / Starting — detector and AI decoder are starting.
- `✓` / Active — click to disable.
- `!` / Failed — direct video remains live.

The tile reports video state, YOLO FPS, people count, and track age. Normal
labels are `person 91%`; add `?debug=1` to include track IDs and independent
YOLO/tracker rates, source (`yolo`/`predicted`), and confirmation age.

## Start and verify

```bash
cd /mnt/guardex-nvme/vms-annotator
cp .env.example .env
chmod 600 .env
./scripts/start.sh config/cameras.yaml
```

Open `http://HOST:18080`. The isolated listeners are:

| Service | Listener |
| --- | --- |
| Dashboard, API, WebSocket | TCP 18080 |
| RTSP | TCP 18554 |
| WHEP | TCP 18889 |
| WebRTC ICE | UDP 18189 |
| MediaMTX API | TCP 19997 |

RTMP, HLS, SRT, and MoQ are disabled.

Startup securely loads `.env`, derives required URL variables from enabled
cameras in the selected YAML, and uses only `.venv/bin/python`. Camera IDs and
environment-variable names are arbitrary. Disabled and removed cameras need no
URL. The script rejects group/world-readable `.env` files, isolates Python
import controls, checks dependencies with the exact interpreter, generates a
private `0600` MediaMTX configuration, validates direct HEVC paths, and prints
success only after both processes and application health are ready.

## ByteTrack prediction-rate override

YOLO correction remains fixed at one FPS. The independent ByteTrack prediction
and metadata target use `BYTETRACK_PREDICTION_FPS` when it is set; otherwise
they use `tracking.prediction_fps` in the selected YAML (five FPS in the
checked-in configuration), then the built-in five-FPS default. Supported values
are finite integer or decimal numbers from 1 through 25 inclusive. Invalid,
empty, zero, negative, NaN, infinite, or out-of-range values fail startup with
a safe error and never reveal camera URLs.

Use the YAML/default value:

```bash
./scripts/start.sh config/cameras.yaml
```

Test a specific rate without editing any source or YAML:

```bash
BYTETRACK_PREDICTION_FPS=5 ./scripts/start.sh config/cameras.yaml
BYTETRACK_PREDICTION_FPS=10 ./scripts/start.sh config/cameras.yaml
BYTETRACK_PREDICTION_FPS=15 ./scripts/start.sh config/cameras.yaml
```

For repeated starts:

```bash
export BYTETRACK_PREDICTION_FPS=10
./scripts/start.sh config/cameras.yaml

# Return to the YAML/default rate.
unset BYTETRACK_PREDICTION_FPS
```

Startup logs the effective FPS, source, and derived interval. `/metrics` and
safe camera configuration responses expose the YAML value, optional override,
effective value, and actual prediction/metadata rates; the browser debug view
shows the effective tracker target rather than assuming five FPS.

## Native TensorRT detection and rate overrides

The direct HEVC browser path is independent of detection and is never
re-encoded. Native `tensorrt` is the GPU backend: it loads a local FP16 engine,
uses one CUDA stream with reusable host/device buffers, and exposes
`TensorRT` as its execution provider. An explicit TensorRT request fails
clearly if engine load or warm-up fails; it never silently falls back to CPU.
`auto` first performs that native warm-up, then may use PyTorch CPU only if
fallback is explicitly allowed. The rejected ONNX INT8 model is not selectable.

`YOLO_INFERENCE_FPS` controls fresh detector results, while
`AI_CAPTURE_FPS` controls source sampling and is automatically kept at least
as high as YOLO's target. `BYTETRACK_PREDICTION_FPS` remains independent. All
controls use: process environment, `.env`, selected YAML, then the built-in
default. Restart after changing `.env`.

```bash
cd /mnt/guardex-nvme/vms-annotator
./scripts/export_yolo_trt.sh

DETECTION_BACKEND=tensorrt DETECTION_PRECISION=fp16 \
TRT_ENGINE_MODEL=models/yolo11n_640_fp16.engine \
AI_CAPTURE_FPS=3 YOLO_INFERENCE_FPS=2 \
BYTETRACK_PREDICTION_FPS=10 ./scripts/start.sh config/cameras.yaml
```

TensorRT Python uses a narrow compatibility loader that appends only
`/usr/lib/python3.10/dist-packages` before importing the installed JetPack
binding. Do not `pip install tensorrt`.
The export validates the FP32 static-batch-one 640×640 ONNX source, builds an
ignored `models/yolo11n_640_fp16.engine`, load-benchmarks that exact engine,
and atomically publishes it only after both stages pass. Engine files are tied
to the installed TensorRT/CUDA/Jetson stack and must be rebuilt after a
relevant platform upgrade. The runtime letterboxes BGR input, reverses that
transform for boxes, uses person class 0 only, and applies one local NMS pass.

Useful commands:

```bash
./scripts/status.sh
./scripts/healthcheck.sh
curl -s http://127.0.0.1:18080/api/detection/active-camera
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"camera_id":"cam01"}' \
  http://127.0.0.1:18080/api/detection/active-camera
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"camera_id":null}' \
  http://127.0.0.1:18080/api/detection/active-camera
./scripts/stop.sh
```

`/metrics` exposes the active camera, detector and AI decoder states,
requested/actual YOLO and prediction FPS, correction/prediction counts,
skipped ticks, prediction and inference p50/p95, source-frame age,
capture-to-result latency, person and track counts, application CPU, AI FFmpeg
CPU, and separate YOLO/prediction worker CPU-time samples. Video health is separate:
detector failure does not make `/healthz` fail while MediaMTX remains ready.

## Validation notes

Unit tests use synthetic detections to verify ByteTrack ID association and
expiry. Real-office ID stability, entry/exit delay, false retained boxes, and
crossing-person ID switches require a visible operator and must be recorded
from observation; they are not inferred from empty frames or unit tests.

The historical direct-HEVC and AI optimization measurements remain in
[`docs/direct-hevc-validation.md`](docs/direct-hevc-validation.md) and
[`docs/ai-optimization-report.md`](docs/ai-optimization-report.md). They
describe the previous continuous detector and are baselines, not results for
this on-demand one-FPS controller.

Diagnostic relay scripts remain explicitly separate from normal
`video.mode: direct_hevc`; they are not selected by the checkmark or detection
API.

## Independent Docker metadata-only deployment

The Docker deployment is intentionally separate from `guardex_vms` and does
not start MediaMTX. It is an on-demand metadata service: it connects to an
existing VMS MediaMTX RTSP path only while a camera detection session is
active, then publishes normalized boxes on its application WebSocket. It never
creates WHEP, RTSP, HLS, or an annotated video stream.

This deployment targets the Jetson/JetPack 6.2 TensorRT 10.3 runtime used by
the checked-in FP16 engine. It pins the R36.4 TensorRT runtime and Python
binding to `10.3.0.30-1+cuda12.5`, the build used by the supplied engine.
TensorRT plans are tied to the TensorRT/CUDA/Jetson software stack; startup
logs the effective CUDA/TensorRT values plus engine filename and checksum
prefix, then fails before HTTP/capture if deserialization is incompatible.
The service does not rebuild the engine and explicit `tensorrt` never falls
back to CPU.

### Configure and build

```bash
cd /mnt/guardex-nvme/vms-annotator
cp .env.example .env
chmod 600 .env
docker compose build
```

Edit only the placeholders in `.env`. `ANNOTATOR_CAMERA_RTSP_URL` must be the
main VMS MediaMTX RTSP path published on the Docker host, for example
`rtsp://host.docker.internal:8554/<tenant>/<nvr-id>_ch_01`. Do not commit a
real tenant, path, camera credential, token, or private IP. `cam_01` is the
default canonical camera ID and must match the future VMS camera mapping.

The Compose service mounts the existing `models/yolo11n_640_fp16.engine` and
`yolo11n.pt` read-only. Keep their filenames or update both `.env` paths and
Compose mounts together. It exposes only application port `18080`; it neither
joins nor depends on the VMS Compose project. Linux Docker resolves the host
MediaMTX through `host.docker.internal:host-gateway`.

The Docker-specific detector controls use `ANNOTATOR_*` names in `.env` and
are mapped to the application's established `DETECTION_*` names only inside
the container. This prevents old host-development environment settings from
silently changing the Docker service backend or precision.

### Run and inspect

```bash
docker compose up -d
docker compose ps
docker compose logs -f vms-annotator
curl http://localhost:18080/healthz
curl http://localhost:18080/api/cameras
```

The service starts healthy with its TensorRT backend pre-warmed and retained,
but with no active camera, FFmpeg capture, inference scheduler, or tracker.
Only one camera can be active at a time.

```bash
curl -X POST http://localhost:18080/api/detection/active-camera \
  -H 'Content-Type: application/json' \
  -d '{"camera_id":"cam_01"}'

curl -X POST http://localhost:18080/api/detection/active-camera \
  -H 'Content-Type: application/json' \
  -d '{"camera_id":null}'
```

Stopping emits `clear_tracks` for the camera, terminates FFmpeg capture, stops
the scheduler/prediction worker, and resets ByteTrack while retaining the
pre-warmed detector for a fast next activation.

### Metadata WebSocket

Connect to `ws://localhost:18080/ws/detections` and subscribe:

```json
{"type":"subscribe","camera_ids":["cam_01"]}
```

Messages include `active_camera`, `detector_status`, `tracks`, and
`clear_tracks`. A `tracks` message has `camera_id: "cam_01"`; each box uses
normalized `x`, `y`, `width`, and `height` in `[0, 1]`, with `track_id`,
confidence, source, and capture/completion timing. A client can confirm data
by receiving a `tracks` message after activation. This endpoint has no VMS
frontend integration in this phase.

### Stop or remove

```bash
docker compose stop
docker compose down
```

### Part 4A gateway-only mode

The default Dockerfile target remains the legacy TensorRT detector. The new
`gateway` target is a dependency-free metadata gateway and is validated
separately; do not replace the live Part 3 service until Part 4B connects a
producer.

```text
External analytics producer → POST /api/metadata → short-lived aggregate store
→ active overlay session → /ws/detections → VMS overlay
```

Start an isolated gateway on a different host port:

```bash
docker build --target gateway -t vms-annotator-gateway:test .
docker run --rm -p 18081:18080 \
  -e ANNOTATOR_RUNTIME_MODE=gateway_only \
  -e ANNOTATOR_ALLOWED_CAMERA_IDS=cam_03 \
  -e METADATA_TTL_MS=2000 vms-annotator-gateway:test
```

`POST /api/metadata` accepts schema version 1 `analytics_metadata` or
`clear_metadata` messages. Required fields are `source`, `camera_id`, and an
integer millisecond `timestamp_ms`; analytics messages also carry normalized
positive boxes, numeric `track_id`, and confidence in `[0,1]`. Identity is
namespaced internally as `source:camera_id:track_id`; numeric IDs are retained
for current VMS compatibility and `track_key` is supplied as an optional field.
States expire after `METADATA_TTL_MS` (default 2000). Set
`ANNOTATOR_INGEST_TOKEN` to require `Authorization: Bearer <token>`.

The compatibility `POST /api/detection/active-camera` endpoint starts or stops
an overlay-routing session only; it never starts FFmpeg or inference in gateway
mode. The gateway image uses Python slim, no NVIDIA runtime, models, engines,
RTSP, or media dependencies. The legacy internal detector remains available
for rollback and diagnostics.

### Troubleshooting

* `host.docker.internal` fails: use current Docker Compose with the included
  `extra_hosts` entry; verify `getent hosts host.docker.internal` inside the
  container.
* RTSP `8554` unavailable or wrong path: verify the main VMS stack is already
  running and test the configured path from a trusted operator environment.
  This compose project never starts or restarts VMS MediaMTX.
* GPU/TensorRT failure: inspect `docker compose logs vms-annotator`. The
  `tensorrt` backend is explicit and the container intentionally exits rather
  than silently changing backend or regenerating the engine.
* Camera ID rejected: `/api/cameras` exposes the configured
  `ANNOTATOR_CAMERA_ID`; POST exactly that value.
* No tracks: a successful health check means the service/model are ready, not
  that the RTSP path is reachable. Activate the camera and inspect FFmpeg
  capture status in `/healthz` and logs.
* Container unhealthy: check engine/model mounts, the expected JetPack/TensorRT
  compatibility, and that port `18080` is free on the host.
