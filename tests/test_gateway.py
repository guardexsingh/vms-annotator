from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.gateway_main import Gateway, validate, ws_accept


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
        self.environ = patch.dict(os.environ, {"ANNOTATOR_ALLOWED_CAMERA_IDS": "cam_03", "METADATA_TTL_MS": "2000"})
        self.environ.start(); self.gateway = Gateway()

    def tearDown(self):
        self.environ.stop()

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


if __name__ == "__main__":
    unittest.main()
