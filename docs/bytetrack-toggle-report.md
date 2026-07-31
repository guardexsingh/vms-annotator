# On-demand ByteTrack live validation

Date: 2026-07-31  
Host: six-core ARM64 Jetson environment  
Configuration: `config/cameras.yaml`

## Result

The exclusive controller was exercised through:

```text
off → cam01 → cam02 → cam03 → off
```

At startup and after the final disable, the API returned `camera_id: null`,
every AI capture state was `disabled`, inference was 0 FPS, all track/person
counts were zero, and no FFmpeg process existed. Each active selection created
one local-RTSP AI decoder for only that camera. Switching marked the old camera
disabled, reset its metrics and tracks, stopped and reaped its decoder PID, and
started one new decoder. A cleanup bug found during this run (an exited FFmpeg
left as a zombie) was fixed by explicitly waiting for the child process; the
subsequent cam01-to-cam03 switch fully reaped the old PID.

Direct video remained separate and reported HEVC, non-transcoded, 25 FPS at
both source and MediaMTX for all cameras. A connected cam02 browser reported
24.74–25.15 decoded FPS during detection. Browser FPS was not available for
cam01 or cam03 during the final samples.

## Performance

Linux process CPU percentages below are percentages of one core. Divide the
sum by six to estimate whole-host capacity. RSS is the application process.
The detector-off row was measured before importing/loading YOLO; Python's
allocator retained model-era memory after later deselection even though the
session objects and workers were released.

| Test | App CPU | AI decoder CPU | App RSS | YOLO FPS | Inference p50 | Inference p95 | MediaMTX video FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Detection off | ~0.5% | 0% | 61.6 MiB | 0 | — | — | 25 each |
| cam01 active | 53.5% | 54.7% | 446.0 MiB | 1.0 | 399.1 ms | 412.1 ms | 25 |
| cam02 active | 51.2% | 56.0% | 480.7 MiB | 1.0 | 381.9 ms | 412.5 ms | 25 |
| cam03 active | 54.4% | 45.8% | 501.7 MiB | 1.0 | 389.2 ms | 423.9 ms | 25 |

cam01's longer sample recorded a 905 ms capture-to-result value and no stale
inputs or results. cam02 recorded about 881 ms and cam03 about 825 ms. The
1500 ms bounded freshness gate accommodates the phase difference between
one-FPS capture and inference; the latest slot remains depth one and does not
run catch-up jobs.

The earlier continuous configuration measured about 2.8–3.1 FPS for one
camera without persistence in the user's baseline and 34.3% average host CPU
in the previous direct-video report. The new selected-camera process samples
consume about 17–18% of six-core capacity for Python plus the AI decoder
(about 19% including the separately sampled MediaMTX load for cam01). These are
not perfectly controlled simultaneous host samples, but they show the expected
material reduction and eliminate all AI work when detection is off.

## Track observation

A 30-second cam01 metadata observation received 29 ByteTrack messages:

| Track ID | Messages present |
| ---: | ---: |
| 2 | 27 / 29 |
| 4 | 25 / 29 |
| 1 | 9 / 29 |
| 10 | 8 / 29 |
| 6 | 3 / 29 |
| 8 | 2 / 29 |
| 15 | 1 / 29 |

There were 71 active and four lost-box observations, with two empty messages
and four removed active tracks reported by the end. IDs 2 and 4 persisted
through most of the observation. Because no operator supplied ground-truth
entry, exit, occlusion, or crossing actions during this sample, the shorter
IDs cannot honestly be classified as true ID switches versus different people
entering/leaving or intermittent low-confidence detections.

Appearance delay, exit delay, false retained boxes, crossing-person switches,
and visual position lag remain operator-observation items. Their configured
bounds are approximately one second plus inference for a new appearance,
1500 ms display hold after a miss, and 2000 ms hard removal without
confirmation.

## ByteTrack configuration

The implementation is Ultralytics `8.3.0` `BYTETracker`, updated only on fresh
YOLO person results. The assignment compatibility module is project-local and
uses locked SciPy `1.15.3`; no hybrid or native-frame tracker is present.

```yaml
track_high_thresh: 0.50
track_low_thresh: 0.10
new_track_thresh: 0.50
match_thresh: 0.80
track_buffer: 2
fuse_score: true
hold_box_ms: 1500
remove_track_ms: 2000
```

`track_buffer: 2` is forced to mean two ByteTrack update cycles at the
approximately one-FPS update cadence.
