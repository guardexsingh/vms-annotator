from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.gateway_main import Gateway, validate, ws_accept
from app.evidence_journal import calculate_evidence_coverage


def payload(**changes):
    value = {
        "schema_version": 1, "type": "analytics_metadata", "source": "handraise",
        "camera_id": "cam_03", "event_id": "alert-1", "timestamp_ms": 0,
        "objects": [{"object_type": "person", "track_id": 7,
                     "box": {"x": .1, "y": .2, "width": .3, "height": .4},
                     "confidence": .9, "label": "Hand Raise"}],
    }
    value.update(changes)
    return value


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.environ = patch.dict(os.environ, {"ANNOTATOR_ALLOWED_CAMERA_IDS": "cam_03", "METADATA_TTL_MS": "2000", "ANNOTATOR_EVIDENCE_JOURNAL_DIR": self.tempdir.name})
        self.environ.start(); self.gateway = Gateway()

    def tearDown(self):
        self.gateway.journal.close(); self.environ.stop(); self.tempdir.cleanup()

    def test_metadata_validation_namespaces_tracks(self):
        source, camera, event, boxes = validate(payload(), self.gateway)
        self.assertEqual((source, camera, event), ("handraise", "cam_03", "alert-1"))
        self.assertEqual(boxes[0]["track_key"], "handraise:cam_03:7")

    def test_camera_allow_list_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "camera_id"):
            validate(payload(camera_id="other"), self.gateway)

    def test_clear_metadata_has_no_boxes(self):
        source, camera, event, boxes = validate(payload(type="clear_metadata", objects=[]), self.gateway)
        self.assertEqual((source, camera, event, boxes), ("handraise", "cam_03", "alert-1", []))

    def test_websocket_accept_matches_rfc_example(self):
        self.assertEqual(ws_accept("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")

    def test_gateway_does_not_import_heavy_ai_packages(self):
        source = Path(__file__).parents[1].joinpath("app/gateway_main.py").read_text(encoding="utf-8")
        for module in ("torch", "ultralytics", "tensorrt", "cv2"):
            self.assertNotIn(f"import {module}", source)
            self.assertNotIn(f"from {module}", source)

    def test_journal_exports_analytics_and_clear_as_empty_sample(self):
        now = int(time.time() * 1000)
        for body in (payload(timestamp_ms=now), payload(type="clear_metadata", objects=[], timestamp_ms=now + 100)):
            source, camera, event, boxes = validate(body, self.gateway)
            self.gateway.journal.submit({"schema_version": 1, "type": body["type"], "source": source, "camera_id": camera,
                                         "event_id": event, "event_type": body.get("event_type"), "timestamp_ms": body["timestamp_ms"],
                                         "gateway_received_timestamp_ms": now, "objects": boxes})
        deadline = time.monotonic() + 2
        while self.gateway.journal.counters["journal_records_written"] < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        exported = self.gateway.journal.export({"schema_version": 1, "source": "handraise", "camera_id": "cam_03", "event_id": "alert-1", "alert_mongo_id": "mongo-1", "clip": {"relative_path": "x.mp4", "capture_start_timestamp_ms": now - 100, "capture_end_timestamp_ms": now + 200, "duration_ms": 300, "width": 1920, "height": 1080, "alert_received_timestamp_ms": now, "timing_basis": "host_chunk_receipt_clock_v1"}})
        self.assertEqual(exported["samples"][-1]["objects"], [])
        self.assertEqual(exported["alert_mongo_id"], "mongo-1")

    def test_coverage_helper(self):
        base={"backfill_received":True,"backfill_complete":True,"backfill_sample_count":2,"last_record_receipt_timestamp_ms":2000,"earliest_record_timestamp_ms":1000,"latest_record_timestamp_ms":3000,"records_dropped":0,"journal_incomplete":False}
        ready=calculate_evidence_coverage(manifest=base,records=[],capture_start_timestamp_ms=1000,capture_end_timestamp_ms=3000,settle_ms=1,now_ms=2100)
        self.assertEqual((ready["complete"],ready["reason"]),(True,"ready"))
        self.assertEqual(calculate_evidence_coverage(manifest={},records=[],capture_start_timestamp_ms=0,capture_end_timestamp_ms=1,settle_ms=1000,now_ms=0)["reason"],"missing_backfill")

    def test_export_order_clear_and_object_deduplication(self):
        journal=self.gateway.journal; key=journal.key("handraise","cam_03","order")
        records=[
            {"schema_version":1,"type":"analytics_metadata","source":"handraise","camera_id":"cam_03","event_id":"order","timestamp_ms":20,"gateway_received_timestamp_ms":2,"objects":[{"track_id":1,"track_key":"t", "x":.1,"y":.1,"width":.2,"height":.2}]},
            {"schema_version":1,"type":"analytics_metadata","source":"handraise","camera_id":"cam_03","event_id":"order","timestamp_ms":10,"gateway_received_timestamp_ms":1,"objects":[{"track_id":1,"track_key":"t","x":.1,"y":.1,"width":.2,"height":.2},{"track_id":1,"track_key":"t","x":.2,"y":.2,"width":.2,"height":.2}]},
            {"schema_version":1,"type":"clear_metadata","source":"handraise","camera_id":"cam_03","event_id":"order","timestamp_ms":20,"gateway_received_timestamp_ms":3,"objects":[]},]
        path=journal._path(key); path.write_text("\n".join(__import__("json").dumps(x) for x in records)+"\n")
        journal.counts[key]=(3,path.stat().st_size); journal.manifests[key]={"backfill_received":False,"backfill_complete":False,"records_dropped":0,"journal_incomplete":False}
        out=journal.export({"source":"handraise","camera_id":"cam_03","event_id":"order","alert_mongo_id":"m","clip":{"capture_start_timestamp_ms":0,"capture_end_timestamp_ms":30,"duration_ms":30,"width":1,"height":1,"alert_received_timestamp_ms":0,"timing_basis":"x"}})
        self.assertEqual([s["timestamp_ms"] for s in out["samples"]],[10,20]); self.assertEqual(len(out["samples"][0]["objects"]),1); self.assertEqual(out["samples"][0]["objects"][0]["x"],.2); self.assertEqual(out["samples"][1]["objects"],[])


if __name__ == "__main__":
    unittest.main()
