"""Bounded, stdlib-only persistence for accepted metadata gateway messages."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path

def calculate_evidence_coverage(*, manifest, records, capture_start_timestamp_ms, capture_end_timestamp_ms, settle_ms, now_ms):
    if not all(isinstance(x,int) and not isinstance(x,bool) for x in (capture_start_timestamp_ms,capture_end_timestamp_ms,now_ms)) or capture_end_timestamp_ms<capture_start_timestamp_ms: raise ValueError("invalid capture timestamps")
    settle_ms=max(100,min(10000,int(settle_ms))); received=bool(manifest.get("backfill_received",False)); complete_backfill=bool(manifest.get("backfill_complete",False)); dropped=int(manifest.get("records_dropped",0)); healthy=not bool(manifest.get("journal_incomplete",False)) and dropped==0
    last=manifest.get("last_record_receipt_timestamp_ms"); settled=isinstance(last,int) and now_ms-last>=settle_ms
    latest=manifest.get("latest_record_timestamp_ms"); end=isinstance(latest,int) and latest>=capture_end_timestamp_ms
    clears=[r for r in records if r.get("type")=="clear_metadata" and capture_start_timestamp_ms<=r.get("timestamp_ms",-1)<=capture_end_timestamp_ms]
    if clears:
        clear=max(clears,key=lambda r:r["timestamp_ms"]); end=end or not any(r.get("type")=="analytics_metadata" and r.get("objects") and clear["timestamp_ms"]<r.get("timestamp_ms",-1)<=capture_end_timestamp_ms for r in records)
    end=end or (complete_backfill and settled); start=complete_backfill; final=complete_backfill and start and end and settled and healthy
    reason="journal_incomplete" if not healthy else "missing_backfill" if not received else "backfill_incomplete" if not complete_backfill else "waiting_for_capture_end" if not end else "waiting_for_settle" if not settled else "ready"
    return {"backfill_received":received,"backfill_complete":complete_backfill,"backfill_sample_count":int(manifest.get("backfill_sample_count",0)),"earliest_record_timestamp_ms":manifest.get("earliest_record_timestamp_ms"),"latest_record_timestamp_ms":latest,"last_record_receipt_timestamp_ms":last,"capture_start_covered":start,"capture_end_reached":end,"settled":settled,"journal_healthy":healthy,"complete":final,"reason":reason}


class EvidenceJournal:
    def __init__(self):
        self.root = Path(os.getenv("ANNOTATOR_EVIDENCE_JOURNAL_DIR", "/data/evidence-journal"))
        self.retention_seconds = max(1, int(os.getenv("ANNOTATOR_EVIDENCE_JOURNAL_RETENTION_HOURS", "24"))) * 3600
        self.max_events = max(1, int(os.getenv("ANNOTATOR_EVIDENCE_JOURNAL_MAX_EVENTS", "1000")))
        self.max_records = max(1, int(os.getenv("ANNOTATOR_EVIDENCE_JOURNAL_MAX_RECORDS_PER_EVENT", "5000")))
        self.max_bytes = max(1024, int(os.getenv("ANNOTATOR_EVIDENCE_JOURNAL_MAX_BYTES_PER_EVENT", "5242880")))
        self.q: queue.Queue[dict] = queue.Queue(maxsize=max(1, int(os.getenv("ANNOTATOR_EVIDENCE_JOURNAL_QUEUE_SIZE", "2048"))))
        self.lock = threading.RLock(); self.writing: set[str] = set(); self.incomplete: set[str] = set()
        self.counts: dict[str, tuple[int, int]] = {}; self.last_warning = 0.0
        self.manifests: dict[str, dict] = {}
        self.counters = {"journal_records_queued": 0, "journal_records_written": 0, "journal_records_dropped": 0,"evidence_export_coverage_ready":0,"evidence_export_coverage_not_ready":0,
                         "journal_write_errors": 0, "journal_retention_deletes": 0, "evidence_export_requests": 0,
                         "evidence_export_successes": 0, "evidence_export_not_found": 0, "evidence_export_failures": 0,
                         "evidence_export_samples_returned": 0}
        self.stop_event = threading.Event(); self.available = True; self.thread: threading.Thread | None = None
        try: self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            # A journal volume failure must never prevent gateway-only live routing.
            self.available = False; self.counters["journal_write_errors"] += 1
            print("[WARN] Evidence journal unavailable; live routing continues.", flush=True)
            return
        self._load_existing(); self.thread = threading.Thread(target=self._run, daemon=True, name="evidence-journal")
        self.thread.start(); threading.Thread(target=self._retention_loop, daemon=True, name="evidence-journal-retention").start()

    @staticmethod
    def key(source: str, camera_id: str, event_id: str) -> str:
        return hashlib.sha256(json.dumps([source, camera_id, event_id], separators=(",", ":")).encode()).hexdigest()

    def _path(self, key: str) -> Path: return self.root / f"{key}.jsonl"
    def _manifest_path(self, key: str) -> Path: return self.root / f"{key}.manifest.json"

    def _load_existing(self):
        for path in self.root.glob("*.jsonl"):
            try:
                records = self._read(path); self.counts[path.stem] = (len(records), path.stat().st_size)
            except OSError: pass
        for path in self.root.glob("*.manifest.json"):
            try:
                key=path.name.removesuffix(".manifest.json"); self.manifests[key]=json.loads(path.read_text())
                if self.manifests[key].get("journal_incomplete",self.manifests[key].get("incomplete",False)): self.incomplete.add(key)
            except Exception: continue

    def submit(self, record: dict) -> None:
        if not self.available:
            self.counters["journal_records_dropped"] += 1
            return
        key = self.key(record["source"], record["camera_id"], record["event_id"])
        item = {"key": key, "record": record}
        try:
            self.q.put_nowait(item); self.counters["journal_records_queued"] += 1
        except queue.Full:
            # Deterministic policy: drop the newest record and mark this event incomplete.
            with self.lock: self.incomplete.add(key)
            self.counters["journal_records_dropped"] += 1
            if time.monotonic() - self.last_warning > 30:
                self.last_warning = time.monotonic(); print("[WARN] Evidence journal queue full; live routing continues.", flush=True)

    def _write_manifest(self, key: str, identity: dict, record: dict | None = None):
        body=self.manifests.get(key,{"identity":identity,"backfill_received":False,"backfill_complete":False,"backfill_sample_count":0,"earliest_backfill_timestamp_ms":None,"latest_backfill_timestamp_ms":None,"earliest_record_timestamp_ms":None,"latest_record_timestamp_ms":None,"last_record_receipt_timestamp_ms":None,"live_record_count":0,"backfill_record_count":0,"clear_record_count":0,"records_dropped":0,"journal_incomplete":False})
        body["identity"]=identity
        if record:
            ts=record["timestamp_ms"]; body["earliest_record_timestamp_ms"]=ts if body["earliest_record_timestamp_ms"] is None else min(body["earliest_record_timestamp_ms"],ts); body["latest_record_timestamp_ms"]=ts if body["latest_record_timestamp_ms"] is None else max(body["latest_record_timestamp_ms"],ts); body["last_record_receipt_timestamp_ms"]=record["gateway_received_timestamp_ms"]
            if record.get("backfill"):
                body["backfill_received"]=True; body["backfill_complete"]=body["backfill_complete"] or bool(record.get("backfill_complete")); body["backfill_sample_count"]+=1; body["backfill_record_count"]+=1; body["earliest_backfill_timestamp_ms"]=ts if body["earliest_backfill_timestamp_ms"] is None else min(body["earliest_backfill_timestamp_ms"],ts); body["latest_backfill_timestamp_ms"]=ts if body["latest_backfill_timestamp_ms"] is None else max(body["latest_backfill_timestamp_ms"],ts)
            else: body["live_record_count"]+=1
            if record.get("type")=="clear_metadata": body["clear_record_count"]+=1
        body["journal_incomplete"]=key in self.incomplete; body["incomplete"]=body["journal_incomplete"]; body["records_dropped"]=max(body["records_dropped"],int(key in self.incomplete)); body["updated_timestamp_ms"]=int(time.time()*1000); self.manifests[key]=body
        tmp = self._manifest_path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8"); os.replace(tmp, self._manifest_path(key))

    def _run(self):
        while not self.stop_event.is_set() or not self.q.empty():
            try: item = self.q.get(timeout=.25)
            except queue.Empty: continue
            key, record = item["key"], item["record"]
            with self.lock: self.writing.add(key)
            try:
                count, size = self.counts.get(key, (0, 0)); raw = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
                if count >= self.max_records or size + len(raw) > self.max_bytes:
                    with self.lock: self.incomplete.add(key)
                    self.counters["journal_records_dropped"] += 1
                else:
                    with self._path(key).open("ab") as fh: fh.write(raw); fh.flush(); os.fsync(fh.fileno())
                    self.counts[key] = (count + 1, size + len(raw)); self._write_manifest(key, {k: record[k] for k in ("source", "camera_id", "event_id")}, record)
                    self.counters["journal_records_written"] += 1
            except Exception:
                self.counters["journal_write_errors"] += 1
                with self.lock: self.incomplete.add(key)
            finally:
                with self.lock: self.writing.discard(key)

    @staticmethod
    def _read(path: Path) -> list[dict]:
        result = []
        try:
            with path.open("rb") as fh:
                for line in fh:
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict): result.append(value)
                    except (UnicodeDecodeError, json.JSONDecodeError): pass  # including a torn final line
        except OSError: raise
        return result

    def _retention_loop(self):
        while not self.stop_event.wait(300): self.retain()

    def retain(self):
        if not self.available: return
        now = time.time(); candidates = []
        for path in self.root.glob("*.jsonl"):
            try: candidates.append((path.stat().st_mtime, path))
            except OSError: continue
        candidates.sort()
        delete = [(mtime, path) for mtime, path in candidates if now - mtime > self.retention_seconds]
        delete += candidates[:max(0, len(candidates) - self.max_events)]
        for _, path in {p: (m, p) for m, p in delete}.values():
            key = path.stem
            with self.lock:
                if key in self.writing: continue
            try:
                path.unlink(missing_ok=True); self._manifest_path(key).unlink(missing_ok=True); self.counts.pop(key, None); self.incomplete.discard(key)
                self.counters["journal_retention_deletes"] += 1
            except OSError: self.counters["journal_write_errors"] += 1

    def export(self, request: dict) -> dict:
        self.counters["evidence_export_requests"] += 1
        if not self.available: self.counters["evidence_export_not_found"] += 1; raise FileNotFoundError()
        source, camera, event = request["source"], request["camera_id"], request["event_id"]
        key = self.key(source, camera, event); path = self._path(key)
        if not path.exists(): self.counters["evidence_export_not_found"] += 1; raise FileNotFoundError()
        records = [r for r in self._read(path) if r.get("source") == source and r.get("camera_id") == camera and r.get("event_id") == event]
        if not records: self.counters["evidence_export_not_found"] += 1; raise FileNotFoundError()
        clip = request["clip"]; start, end = clip["capture_start_timestamp_ms"], clip["capture_end_timestamp_ms"]
        duration = clip["duration_ms"]; span = end - start; basis = clip.get("timing_basis", "host_chunk_receipt_clock_v1")
        if span <= 0 or duration <= 0:
            alert_at = clip["alert_received_timestamp_ms"]; start = alert_at - int(request.get("pre_seconds", 5) * 1000); end = start + duration; span = duration; basis = "nominal_alert_receipt_clock_v1"
        ordered = sorted(enumerate(records), key=lambda item: (item[1].get("timestamp_ms", 0), item[1].get("gateway_received_timestamp_ms", 0), item[0]))
        # One playback state per producer timestamp.  Equal canonical states collapse;
        # conflicting states resolve to the later receipt/JSONL occurrence (clear wins
        # only when it is later, and vice versa).
        chosen: dict[int, dict] = {}
        tolerance = 500
        for _, record in ordered:
            ts = record.get("timestamp_ms")
            if not isinstance(ts, int) or ts < start - tolerance or ts > end + tolerance: continue
            objects={}
            for obj in record.get("objects",[]):
                identity=obj.get("track_key") or f"{obj.get('source',record.get('source'))}:{obj.get('track_id')}"; objects[identity]=obj
            normalized={**record,"objects":[objects[k] for k in sorted(objects)]}
            json.dumps({"source":normalized.get("source"),"camera_id":normalized.get("camera_id"),"event_id":normalized.get("event_id"),"timestamp_ms":ts,"type":normalized.get("type"),"objects":normalized["objects"]},sort_keys=True,separators=(",",":"),ensure_ascii=False)
            chosen[ts]=normalized
        samples = []
        for ts, record in sorted(chosen.items()):
            offset = (ts - start) * duration / span
            if offset < -tolerance or offset > duration + tolerance: continue
            offset = int(round(min(duration, max(0, offset))))
            samples.append({"timestamp_ms": ts, "offset_ms": offset, "objects": record.get("objects", [])})
        event_type = next((r.get("event_type") for r in records if isinstance(r.get("event_type"), str)), None)
        coverage=calculate_evidence_coverage(manifest=self.manifests.get(key,{}),records=records,capture_start_timestamp_ms=start,capture_end_timestamp_ms=end,settle_ms=request.get("settle_ms",1000),now_ms=int(time.time()*1000))
        self.counters["evidence_export_coverage_ready" if coverage["complete"] else "evidence_export_coverage_not_ready"]+=1
        result = {"schema_version": 1, "type": "evidence_annotations", "alert_id": event,
                  "alert_mongo_id": request["alert_mongo_id"], "camera_id": camera, "source": source,
                  "event_type": event_type, "timing_basis": basis,
                  "journal": {"complete": key not in self.incomplete, "records_read": len(records), "records_dropped": int(key in self.incomplete)},
                  "clip": {**clip, "capture_start_timestamp_ms": start, "capture_end_timestamp_ms": end, "duration_ms": duration,
                           "alert_offset_ms": int(round((clip["alert_received_timestamp_ms"] - start) * duration / span))},
                  "playback": {"sample_hold_ms": 500}, "samples": samples,"coverage":coverage}
        self.counters["evidence_export_successes"] += 1; self.counters["evidence_export_samples_returned"] += len(samples)
        return result

    def metrics(self) -> dict:
        try: size = sum(p.stat().st_size for p in self.root.glob("*.jsonl")) if self.available else 0
        except OSError: size = 0
        return {**self.counters, "journal_active_events": len(self.counts), "journal_files": len(self.counts), "journal_bytes": size,
                "journal_incomplete_events": len(self.incomplete)}

    def close(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)
