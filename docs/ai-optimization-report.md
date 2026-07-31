# AI inference optimization report

Measured on the isolated prototype on 2026-07-30 and 2026-07-31. The video
architecture remained:

```text
HEVC camera → MediaMTX direct pull → HEVC WHEP → Chrome
```

No result below changes source video resolution, FPS, codec, GOP, or bitrate.
No video encoder or annotated video path was added.

## Hardware and usable runtimes

- NVIDIA Jetson Orin Nano Developer Kit, Jetson Linux R36.4.7 / JetPack 6-era
  vendor stack, Linux 5.15.148-tegra, aarch64.
- Six physical Cortex-A78AE cores, 115.2–1510.4 MHz; 7.4 GiB RAM and 3.7 GiB
  swap; 15 W power mode.
- CUDA toolkit 12.6, TensorRT 10.3, and cuDNN 9.3 are installed system-wide.
- Project Python 3.10.12 uses PyTorch 2.5.1 CPU-only
  (`torch.cuda.is_available() == False`, `torch.version.cuda == None`).
- TensorRT was not usable: CUDA initialization returned error 999 and
  `trtexec` failed during device initialization. The accelerator device nodes
  needed by this environment were not exposed.
- ONNX Runtime, OpenVINO, TensorFlow Lite, TensorFlow, and TensorRT Python were
  not installed in the isolated environment. No vendor or generic accelerator
  package was installed or overwritten.
- FFmpeg advertises Jetson HEVC decode support, but no usable NVDEC device was
  exposed. The AI branch therefore uses software HEVC decode.

No model exports were performed because no corresponding runtime passed a real
load/inference capability test. Consequently there is no honest ONNX,
TensorRT, OpenVINO, FP16, or INT8 result or export command to report. The
working weights remain `yolo11n.pt`; image size 640, class 0, and confidence
0.40 remain locked.

## AI capture

Representative 15–20 second three-camera measurements:

| Strategy/camera | Source FPS | AI output FPS | Output | BGR pipe rate | Decoder CPU | Frame age p50/p95 | Frames replaced |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous scaled, cam01 | ~25 | 24.5 | 640×360 | 16.1 MiB/s | 68% core | 20–24 / 48–91 ms across cameras | 294–297/15s |
| Continuous scaled, cam02 | ~25 | 24.7 | 640×360 | 16.3 MiB/s | 70.8% core | same run | 294–297/15s |
| Continuous scaled, cam03 | ~25 | 24.7 | 640×360 | 16.3 MiB/s | 87.6% core | same run | 294–297/15s |
| Sampled+scaled, cam01 | 25 | 4.9 | 640×360 | 3.23 MiB/s | 58.4% core | 142–150 / 191–197 ms across cameras | 9/20s |
| Sampled+scaled, cam02 | 25 | 4.95 | 640×360 | 3.26 MiB/s | 54.6% core | same run | 9/20s |
| Sampled+scaled, cam03 | 25 | 4.9 | 640×360 | 3.23 MiB/s | 63.6% core | same run | 5/20s |

The selected `select=not(mod(n,5))` strategy continues to drain/decode the
inter-frame source, but avoids scaling, BGR conversion, raw pipe traffic, and
Python allocation for about 80% of decoded frames. Three-camera host CPU in
the capture-only benchmark fell from 42.35% to 33.93%. Full-resolution BGR was
rejected: it drove about 144 MiB/s for each 1080p camera and 219 MiB/s for the
1440p camera with 53.23% host CPU.

## Backend and thread comparison

All rows use PyTorch CPU FP32, `yolo11n.pt`, image size 640, and representative
live frames kept only in memory. Each row used 15 measured iterations after
warm-up. Accuracy comparison to a different backend is not applicable because
no other backend was usable.

| Threads | Batch | p50 | p95 | Images/s | Process CPU | Host CPU | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 405 ms | 406 ms | 2.47 | 100% | 18.5% | 386 MiB |
| 1 | 3 | 1186 ms | 1187 ms | 2.53 | 101% | 20.1% | 434 MiB |
| 2 | 1 | 287 ms | 288 ms | 3.48 | 193% | 32.8% | 405 MiB |
| 2 | 3 | 745 ms | 747 ms | 4.02 | 195% | 34.6% | 470 MiB |
| 4 | 1 | 189 ms | 190 ms | 5.29 | 373% | 61.0% | 421 MiB |
| 4 | 3 | 461 ms | 468 ms | 6.50 | 377% | 66.0% | 491 MiB |
| 6 | 1 | 188 ms | 207 ms | 5.09 | 513% | 84.5% | 431 MiB |
| 6 | 3 | 438 ms | 465 ms | 6.77 | 512% | 89.2% | 516 MiB |

Four threads were the best isolated efficiency point. Six threads gained only
0.27 images/s in batch while consuming nearly all six cores, leaving
insufficient room for MediaMTX and three software AI decoders.

## Scheduler and worker comparison

Live integrated scheduler results include all three sampled/scaled AI
decoders:

| Mode | Aggregate useful FPS | cam01 | cam02 | cam03 | Result age | Video impact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Shared serial, 4 threads | ~2.5 | 0.8 | 0.8 | 0.9 | ~462–511 ms median | Guard passed |
| Batch of 3, 4 threads | near 0 published | ~0 | ~0.05 | ~0 | ~706–745 ms for rare accepted results; 61 stale/camera | Guard passed |
| Opportunistic, 4 threads | near 0 published | 0 | 0 | 0 | ~747–749 ms for two accepted results | Guard passed |
| Three one-thread workers | ~6.8 | 2.3 | 2.2 | 2.3 | ~528–557 ms median | Guard passed |

The separate bounded worker harness confirmed the choice:

| Layout | Aggregate raw/useful FPS | Per-camera useful FPS | p95 age | Peak worker-family RSS | Host CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| One serial 4-thread | 3.17 / 3.17 | 1.04–1.09 | 520–556 ms | 669 MiB | 76.2% |
| One batch 4-thread | 4.20 / 1.28 | 0.35–0.54 | 935–946 ms | 736 MiB | 75.9% |
| Two serial 2-thread | 5.31 / 5.31 | 1.28, 1.33, 2.71 | 560–582 ms | 1040 MiB | 80.8% |
| Three serial 1-thread | 6.90 / 6.90 | 2.28–2.33 | 610–649 ms | 1404 MiB | 78.3% |
| Three serial 2-thread | 5.01 / 3.13 | 0.69–1.39 | 810–931 ms | 1412 MiB | 88.2% |

Three one-thread workers are both fair and fastest for useful results. The
integrated same-process implementation shares runtime pages and measured about
484 MiB app RSS instead of the multi-process harness's 1.4 GiB family peak.
The app is niced to level 5 below the separate MediaMTX process.

## Selected result and stability

- Backend/device/precision: PyTorch / CPU / FP32.
- Scheduler: one serial worker per camera, three model instances, one thread
  per worker, one interop thread, no batching.
- AI capture: software HEVC, approximately 5 sampled frames/s, 640×360 BGR,
  one latest slot per camera.
- Integrated 60-second result: 2.2–2.3 FPS/camera; aggregate about 6.8 FPS;
  inference p50 441 ms, p95 480 ms; median capture-to-result about 528–557 ms;
  no stale result publications; median host CPU 84.9%.
- Direct source and MediaMTX video remained HEVC at nominal 25 FPS. Every
  pre/post process/codec guard passed. Browser telemetry was not connected
  during automated benchmarks; the previously operator-validated Chrome
  result remains approximately 25 FPS with acceptable latency and was not
  re-measured by automation.

The 10-minute cam01 run completed:

- 2.8 FPS median; final inference p50/p95 356/365 ms.
- Observed capture-to-result p50/p95 448/546 ms.
- Median host CPU 33.3%; peak app RSS 424 MiB.
- Temperature range across readable zones 48.1–53.1°C; CPU clocks did not
  collapse; GPU remained idle at 306 MHz.
- Zero stale inputs, decode failures, reconnects, or process restarts.
- Pre/post direct-HEVC guards passed.

The required 30-minute three-camera run subsequently completed:

| Metric | cam01 | cam02 | cam03 |
| --- | ---: | ---: | ---: |
| Sustained result FPS | 2.365 | 2.302 | 2.291 |
| Rolling FPS median / p05 | 2.4 / 2.3 | 2.3 / 2.2 | 2.3 / 2.2 |
| Final inference p50 / p95 | 420 / 449 ms | 432 / 461 ms | 433 / 465 ms |
| Snapshot result-age p50 / p95 | 532 / 635 ms | 527 / 632 ms | 523 / 633 ms |
| Maximum observed result age | 700 ms | 702 ms | 706 ms |
| Result interval p50 / p95 | 421 / 450 ms | 432 / 461 ms | 434 / 464 ms |
| Completed results | 4,229 | 4,117 | 4,097 |
| Rejected stale inputs | 2 | 2 | 3 |
| AI capture reconnects / failures | 0 / 0 | 0 / 0 | 0 / 0 |

Aggregate inference p50/p95 was 427/456 ms with a 621 ms maximum. Median/p95
host CPU was 81.4/84.9%; peak app RSS was 493 MiB and peak host RAM use was
2.42 GiB. Readable thermal zones ranged from 45.8°C to 55.5°C. Early, middle,
and late thirds all held median rolling rates of 2.4/2.3/2.3 FPS; late median
per-camera inference times were 417/430/429 ms, showing no thermal performance
collapse. CPU clocks ranged from 883 MHz under governor control to 1.51 GHz
and were repeatedly observed at 1.51 GHz under load. The GPU stayed idle at
306 MHz because inference was CPU-only.

Both video regression guards passed. Source and MediaMTX paths remained HEVC
at nominal 25 FPS, direct and non-transcoded, with H265 tracks and no video
encoder. Browser telemetry was not connected during the automated soak, so
browser FPS and glass-to-glass latency were not re-measured. The harness
stopped both experiment processes after completion.

## Accuracy status

The selected backend, weights, precision, image size, class filter, and
confidence are unchanged from the PyTorch baseline. Scheduler and capture
changes do not alter model output for a frame, but 5 FPS sampling can change
which motion frames are observed. The automated live frames did not contain a
controlled set of people; therefore the required stationary/multiple/walking/
occlusion/edge/distant/empty office-scene matrix is **not claimed complete**.
It requires an operator to stage those scenes. No faster backend or reduced
precision was selected, so there is no cross-backend accuracy claim.

## Classification

This is **partial success**: the selected configuration provides at least
2–3 FPS per camera with bounded age while preserving direct HEVC video. It
does not achieve the 15 images/s requirement. The isolated CPU maximum was
6.50 images/s at the practical four-thread batch point (6.77 images/s only
when six cores were nearly saturated), demonstrating that 15 images/s is not
feasible with the exposed CPU-only runtime.

Reaching approximately 5 FPS on all three cameras requires an exposed,
vendor-compatible Jetson Orin CUDA/TensorRT execution path (including working
GPU device access and an appropriate JetPack PyTorch/TensorRT environment), or
another accelerator proven by real model load/inference. No generic CUDA or
TensorRT package should be installed over the vendor stack.
