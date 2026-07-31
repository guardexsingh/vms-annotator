# Direct HEVC validation record

## Architecture decision

The standalone default is:

```text
camera HEVC RTSP → MediaMTX direct source-on-demand pull → HEVC WHEP → Chrome
```

No video relay or encoder belongs in normal startup. Detection independently
decodes the credential-free local HEVC RTSP path into a one-slot latest-frame
pipeline and emits normalized bounding-box metadata over WebSocket. Video is
never decoded or re-encoded merely to add annotations.

The shipped YAML locks:

```yaml
video:
  mode: direct_hevc
  allow_transcode_fallback: false
```

An H.264 source, a non-HEVC MediaMTX path, a browser without advertised HEVC
WebRTC support, or a WHEP answer selecting another codec is an explicit failure.

## Security and lifecycle

- Enabled camera URLs are expanded dynamically from the selected YAML.
- Exact `live/<camera-id>` sources exist only in `run/mediamtx_direct.yml`.
- Runtime configuration and validation files are mode `0600`, Git-ignored, and
  removed by failed-start cleanup and `stop.sh`.
- `/api/cameras`, `/metrics`, browser diagnostics, startup output, and
  application diagnostics contain no camera URLs.
- Startup order is configuration generation, MediaMTX/API readiness, source
  and local-path HEVC validation, dashboard health, then asynchronous YOLO.

## Measurements

The previous three-camera HEVC-to-H.264 relay consumed approximately 58–61%
host CPU with detection disabled. Direct-mode CPU, negotiated browser codec/FPS,
and operator-observed start/steady-state latency are recorded after each live
run rather than inferred from unit tests.

| Test | Detection | Source codec | MediaMTX codec | Browser codec | Browser FPS | Start latency | Steady latency | Host CPU |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A: cam01 only | off | HEVC, 25 FPS | H265 Main, 1920×1080 | Headless Chromium unsupported | unmeasured | unmeasured | unmeasured | not sampled |
| B: all cameras | off | HEVC, 25 FPS each | H265 Main; 1080p/1080p/1440p | Headless Chromium unsupported | unmeasured | server ready in about 5.1 s | unmeasured | 9.9% avg, 32.3% peak |
| C: cam01 AI | cam01 | HEVC, 25 FPS | HEVC, 25 FPS | not measured | not measured | direct path unchanged | unmeasured | 34.3% avg, 58.1% peak |
| D: all AI | all | HEVC, 25 FPS | HEVC, 25 FPS | not measured | not measured | direct path unchanged | unmeasured | 55.2% avg, 76.0% peak |

The previous transcoding baseline was approximately 58–61% CPU with detection
off. Direct video-only averaged 9.9% in the 30-second live sample. With cam01 AI,
capture drained at 24 FPS and inference reached 3.1 FPS (p50 326 ms, p95 413
ms). With all AI readers, capture drained at 23–25 FPS; fair inference measured
0.6–0.7 FPS per camera (p50 544 ms, p95 655 ms). The process tree contained
three HEVC-to-rawvideo AI decoder processes and no output encoder.

An intentional missing-model test left application health, MediaMTX, and all
three H265 paths ready while detector state became `failed`. This demonstrates
that detector failure changes metadata state only.

The available headless Chromium advertised no HEVC WebRTC capability and the UI
displayed `HEVC unsupported`; it did not start a fallback. Therefore negotiated
browser codec/FPS, visual start latency, and five-minute glass-to-glass latency
remain operator-Chrome measurements. They are deliberately not inferred from
the successful H265 MediaMTX tracks.
