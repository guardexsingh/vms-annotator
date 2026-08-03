# VMS metadata gateway

This service is a lightweight, producer-driven metadata gateway. It accepts
analytics metadata and broadcasts normalized overlay tracks to the VMS browser.
It has no GPU requirement, video decoder, RTSP input, AI model, inference loop,
or internal tracker.

## Start

```bash
cd /mnt/guardex-nvme/vms-annotator
docker compose up -d --build
curl -s http://localhost:18080/healthz | python3 -m json.tool
```

The health response reports `runtime_mode: gateway_only`. Stop it with:

```bash
docker compose down
```

## Configuration

`ANNOTATOR_ALLOWED_CAMERA_IDS` is the comma-separated camera allow-list;
`METADATA_TTL_MS` bounds how long a producer update remains visible. Set
`ANNOTATOR_INGEST_TOKEN` to require `Authorization: Bearer <token>` for
`POST /api/metadata`. Do not commit a real token.

## Producer schema

Handraise is the current producer. It sends one target box per active alert:

```json
{
  "schema_version": 1,
  "type": "analytics_metadata",
  "source": "handraise",
  "camera_id": "cam_03",
  "event_id": "alert-id",
  "timestamp_ms": 0,
  "objects": [{
    "object_type": "person",
    "track_id": 7,
    "box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.3},
    "confidence": 0.9,
    "label": "Hand Raise"
  }]
}
```

Send the same `source`, `camera_id`, and `event_id` with
`"type": "clear_metadata"` to clear explicitly. Otherwise metadata clears at
the configured TTL. Tracks are namespaced as `<source>:<camera_id>:<track_id>`.

## VMS compatibility

`POST /api/detection/active-camera` is retained for the VMS. It controls an
overlay-routing session only; it never starts detection. The VMS subscribes to
`/ws/detections` with `{"type":"subscribe","camera_ids":["cam_03"]}`.
The gateway supports HTTP/1.1 WebSocket upgrade, ping/pong, clean close,
`tracks`, `clear_tracks`, and `active_camera` messages.

## Migration

The internal YOLO11/TensorRT detector was removed after the producer-driven
metadata gateway became production architecture. The previous source remains
available in Git tag `pre-gateway-only-cleanup`; the old Docker image is kept
locally for rollback until explicitly removed later.
