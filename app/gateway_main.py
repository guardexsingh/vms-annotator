"""Dependency-free metadata gateway runtime; never imports the AI stack."""
from __future__ import annotations

import base64, hashlib, hmac, json, os, socket, struct, threading, time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .evidence_journal import EvidenceJournal

MAX_BODY = 262_144
MAX_OBJECTS = 100

def ws_frame(raw: bytes, opcode: int = 0x1) -> bytes:
    n = len(raw); head = bytearray([0x80 | opcode])
    if n < 126: head.append(n)
    elif n < 65536: head.extend((126, *struct.pack("!H", n)))
    else: head.extend((127, *struct.pack("!Q", n)))
    return bytes(head) + raw

def frame(data: dict) -> bytes:
    return ws_frame(json.dumps(data, separators=(",", ":"), allow_nan=False).encode())

def ws_accept(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()

def read_ws(stream):
    header = stream.read(2)
    if len(header) != 2: raise ConnectionError()
    a,b=header; n=b&127
    if not a & 0x80: raise ValueError("fragmented frames unsupported")
    if n==126: n=struct.unpack("!H", stream.read(2))[0]
    elif n==127: n=struct.unpack("!Q", stream.read(8))[0]
    if n > 1_048_576 or not b&128 or ((a & 15) >= 8 and n > 125): raise ValueError()
    mask=stream.read(4); raw=stream.read(n)
    return a&15, bytes(x ^ mask[i%4] for i,x in enumerate(raw))

class Gateway:
    def __init__(self):
        self.cameras=frozenset(x.strip() for x in os.getenv("ANNOTATOR_ALLOWED_CAMERA_IDS","cam_03").split(",") if x.strip())
        self.ttl=max(100, int(os.getenv("METADATA_TTL_MS","2000")))/1000
        self.token=os.getenv("ANNOTATOR_INGEST_TOKEN")
        self.export_token=os.getenv("ANNOTATOR_EVIDENCE_EXPORT_TOKEN")
        self.backfill_token=os.getenv("ANNOTATOR_EVIDENCE_BACKFILL_TOKEN")
        self.journal=EvidenceJournal()
        self.active=None; self.states={}; self.clients=[]; self.lock=threading.RLock()
        self.counters={"messages_received":0,"messages_accepted":0,"messages_rejected":0,"objects_received":0,"messages_broadcast":0,"explicit_clears":0,"ttl_clears":0,"backfill_requests":0,"backfill_samples":0,"backfill_rejected":0}
        threading.Thread(target=self.expire, daemon=True).start()
    def state(self): return {"camera_id":self.active,"status":"active" if self.active else "disabled"}
    def send(self, client, message):
        if self.active and (message.get("camera_id") in (self.active, None)):
            try: client.sendall(frame(message)); self.counters["messages_broadcast"]+=1
            except OSError: pass
    def broadcast(self, message):
        for c,subs in list(self.clients):
            if message.get("camera_id") is None or message.get("camera_id") in subs: self.send(c,message)
    def aggregate(self, camera):
        now=time.monotonic(); boxes=[]
        for state in self.states.values():
            if state["camera_id"]==camera and now-state["received"]<=self.ttl: boxes.extend(state["boxes"])
        self.broadcast({"type":"tracks","camera_id":camera,"sequence":int(time.time()*1000),"completed_at_unix_ms":round(time.time()*1000,3),"boxes":boxes})
    def expire(self):
        while True:
            time.sleep(min(self.ttl/2, .5)); now=time.monotonic(); expired=[]
            with self.lock:
                for k,v in list(self.states.items()):
                    if now-v["received"]>self.ttl: expired.append((k,v)); del self.states[k]
                if expired: self.counters["ttl_clears"]+=len(expired)
                cameras={v["camera_id"] for _,v in expired}
                for camera in cameras:
                    if camera==self.active: self.aggregate(camera)

def validate(payload, gateway):
    if not isinstance(payload,dict) or payload.get("schema_version")!=1 or payload.get("type") not in ("analytics_metadata","clear_metadata"): raise ValueError("unsupported schema or type")
    source=payload.get("source"); camera=payload.get("camera_id"); ts=payload.get("timestamp_ms")
    if not isinstance(source,str) or not source or len(source)>80: raise ValueError("invalid source")
    if camera not in gateway.cameras: raise ValueError("invalid camera_id")
    if not isinstance(ts,int) or isinstance(ts,bool) or ts<0 or ts>int(time.time()*1000)+60_000: raise ValueError("invalid timestamp_ms")
    event=payload.get("event_id") or "_default"
    if not isinstance(event,str) or len(event)>160: raise ValueError("invalid event_id")
    if payload["type"]=="clear_metadata": return source,camera,event,[]
    objects=payload.get("objects")
    if not isinstance(objects,list) or len(objects)>MAX_OBJECTS: raise ValueError("invalid objects")
    boxes=[]
    for obj in objects:
        if not isinstance(obj,dict) or not isinstance(obj.get("object_type"),str) or not obj["object_type"]: raise ValueError("invalid object_type")
        tid=obj.get("track_id"); box=obj.get("box"); conf=obj.get("confidence")
        if not isinstance(tid,int) or isinstance(tid,bool) or abs(tid)>9_007_199_254_740_991: raise ValueError("invalid track_id")
        if not isinstance(conf,(int,float)) or not 0<=conf<=1 or not isinstance(box,dict): raise ValueError("invalid confidence or box")
        vals=[box.get(x) for x in ("x","y","width","height")]
        if any(not isinstance(v,(int,float)) or isinstance(v,bool) for v in vals): raise ValueError("invalid box")
        x,y,w,h=map(float,vals)
        if x < -.00001 or y < -.00001 or w<=0 or h<=0 or x>1.00001 or y>1.00001 or x+w>1.00001 or y+h>1.00001: raise ValueError("box outside normalized bounds")
        label=obj.get("label","Person")
        if not isinstance(label,str) or len(label)>160: raise ValueError("invalid label")
        boxes.append({"track_id":tid,"track_key":f"{source}:{camera}:{tid}","source":source,"event_id":event,"event_type":payload.get("event_type"),"x":max(0,x),"y":max(0,y),"width":min(w,1-max(0,x)),"height":min(h,1-max(0,y)),"confidence":float(conf),"label":label,"attributes":obj.get("attributes",{})})
    return source,camera,event,boxes

def main():
    gateway=Gateway()
    class Handler(BaseHTTPRequestHandler):
        # BaseHTTPRequestHandler defaults to HTTP/1.0, which makes a valid
        # RFC 6455 upgrade appear malformed to standards-compliant clients.
        protocol_version = "HTTP/1.1"
        def log_message(self,*_): pass
        def json(self,status,data):
            raw=json.dumps(data,separators=(",",":")).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(raw))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(raw)
        def _get(self):
            if self.path=="/healthz" or self.path=="/metrics":
                with gateway.lock: self.json(200,{"status":"ok","runtime_mode":"gateway_only","http":"ready","gateway":{"status":"ready","active_camera_id":gateway.active,"active_sessions":int(bool(gateway.active)),"known_sources":sorted({k[0] for k in gateway.states}),"fresh_metadata_streams":len(gateway.states)},"metadata":{**gateway.counters,**gateway.journal.metrics(),"websocket_clients":len(gateway.clients)},"detector":{"status":"not_applicable"},"capture":{"status":"not_applicable"}}); return
            if self.path=="/api/cameras": self.json(200,{"cameras":[{"id":c,"runtime_mode":"gateway_only","metadata_enabled":True} for c in sorted(gateway.cameras)],"metadata_path":"/ws/detections","active_camera":gateway.state()}); return
            if self.path=="/api/detection/active-camera": self.json(200,gateway.state()); return
            self.send_error(404)
        def do_POST(self):
            if int(self.headers.get("Content-Length","0"))>MAX_BODY: self.json(413,{"error":"payload_too_large"}); return
            try: payload=json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))))
            except Exception: self.json(400,{"error":"invalid_json"}); return
            if self.path=="/api/detection/active-camera":
                camera=payload.get("camera_id") if isinstance(payload,dict) else object()
                if camera is not None and camera not in gateway.cameras: self.json(400,{"error":"Invalid or unavailable camera_id"}); return
                with gateway.lock:
                    old=gateway.active; gateway.active=camera; gateway.broadcast({"type":"active_camera","camera_id":camera,"status":"active" if camera else "disabled"})
                    if old and not camera: gateway.broadcast({"type":"clear_tracks","camera_id":old})
                    if camera: gateway.aggregate(camera)
                self.json(200,gateway.state()); return
            if self.path=="/api/evidence/export":
                if gateway.export_token and not hmac.compare_digest(self.headers.get("Authorization",""),f"Bearer {gateway.export_token}"): self.json(401,{"error":"unauthorized"}); return
                try:
                    if not isinstance(payload,dict) or payload.get("schema_version") != 1: raise ValueError("unsupported schema_version")
                    for field in ("source","camera_id","event_id","alert_mongo_id"):
                        if not isinstance(payload.get(field),str) or not payload[field] or len(payload[field]) > 160: raise ValueError(f"invalid {field}")
                    if payload["camera_id"] not in gateway.cameras: raise ValueError("invalid camera_id")
                    clip=payload.get("clip")
                    if not isinstance(clip,dict): raise ValueError("invalid clip")
                    for field in ("relative_path","timing_basis"):
                        if not isinstance(clip.get(field),str) or not clip[field] or len(clip[field]) > 320: raise ValueError(f"invalid clip.{field}")
                    for field in ("capture_start_timestamp_ms","capture_end_timestamp_ms","duration_ms","width","height","alert_received_timestamp_ms"):
                        if not isinstance(clip.get(field),int) or isinstance(clip[field],bool) or clip[field] < 0: raise ValueError(f"invalid clip.{field}")
                    if clip["duration_ms"] <= 0 or clip["duration_ms"] > 600_000 or clip["width"] <= 0 or clip["height"] <= 0: raise ValueError("invalid clip dimensions or duration")
                    if "wait_for_coverage" in payload and not isinstance(payload["wait_for_coverage"],bool): raise ValueError("invalid wait_for_coverage")
                    if "settle_ms" in payload and (not isinstance(payload["settle_ms"],(int,float)) or isinstance(payload["settle_ms"],bool)): raise ValueError("invalid settle_ms")
                    self.json(200,gateway.journal.export(payload))
                except FileNotFoundError: self.json(404,{"error":"journal_not_found"})
                except ValueError as error: gateway.journal.counters["evidence_export_failures"]+=1; self.json(400,{"error":str(error)})
                except Exception: gateway.journal.counters["evidence_export_failures"]+=1; self.json(500,{"error":"evidence_export_failed"})
                return
            if self.path=="/api/evidence/backfill":
                if gateway.backfill_token and not hmac.compare_digest(self.headers.get("Authorization",""),f"Bearer {gateway.backfill_token}"): self.json(401,{"error":"unauthorized"}); return
                try:
                    if not isinstance(payload,dict) or payload.get("schema_version")!=1 or payload.get("type")!="evidence_metadata_backfill": raise ValueError("invalid backfill schema")
                    samples=payload.get("samples"); maximum=max(1,min(500,int(os.getenv("ANNOTATOR_EVIDENCE_BACKFILL_MAX_SAMPLES","500"))))
                    if not isinstance(samples,list) or not samples or len(samples)>maximum: raise ValueError("invalid backfill samples")
                    canonical=[]
                    for sample in samples:
                        if not isinstance(sample,dict): raise ValueError("invalid backfill sample")
                        live={"schema_version":1,"type":"analytics_metadata","source":payload.get("source"),"camera_id":payload.get("camera_id"),"event_id":payload.get("event_id"),"event_type":payload.get("event_type"),"timestamp_ms":sample.get("timestamp_ms"),"objects":sample.get("objects")}
                        source,camera,event,boxes=validate(live,gateway); canonical.append({"schema_version":1,"type":"analytics_metadata","source":source,"camera_id":camera,"event_id":event,"event_type":payload.get("event_type"),"timestamp_ms":live["timestamp_ms"],"gateway_received_timestamp_ms":int(time.time()*1000),"objects":boxes,"backfill":True,"backfill_complete":bool(payload.get("backfill_complete"))})
                    for record in canonical: gateway.journal.submit(record)
                    gateway.counters["backfill_requests"]+=1; gateway.counters["backfill_samples"]+=len(canonical); self.json(202,{"ok":True,"samples":len(canonical)})
                except ValueError as error: gateway.counters["backfill_rejected"]+=1; self.json(400,{"error":str(error)})
                return
            if self.path!="/api/metadata": self.send_error(404); return
            if gateway.token and not hmac.compare_digest(self.headers.get("Authorization",""),f"Bearer {gateway.token}"): self.json(401,{"error":"unauthorized"}); return
            gateway.counters["messages_received"]+=1
            try: source,camera,event,boxes=validate(payload,gateway)
            except ValueError as error: gateway.counters["messages_rejected"]+=1; self.json(400,{"error":str(error)}); return
            with gateway.lock:
                key=(source,camera,event)
                if payload["type"]=="clear_metadata": gateway.states.pop(key,None); gateway.counters["explicit_clears"]+=1
                elif key not in gateway.states or payload["timestamp_ms"] >= gateway.states[key]["timestamp"]-100:
                    gateway.states[key]={"camera_id":camera,"received":time.monotonic(),"timestamp":payload["timestamp_ms"],"boxes":boxes}; gateway.counters["objects_received"]+=len(boxes)
                gateway.counters["messages_accepted"]+=1
                if camera==gateway.active: gateway.aggregate(camera)
            # Journal after validation and independently of live state/routing.  It is intentionally
            # best-effort: queue pressure or writer failures never alter this accepted response.
            gateway.journal.submit({"schema_version":1,"type":payload["type"],"source":source,"camera_id":camera,
                                    "event_id":event,"event_type":payload.get("event_type"),"timestamp_ms":payload["timestamp_ms"],
                                    "gateway_received_timestamp_ms":int(time.time()*1000),"objects":boxes})
            self.json(202,{"ok":True})
        def do_GET_ws(self): pass
        def do_GET(self):
            if self.path=="/ws/detections" and self.headers.get("Upgrade","").lower()=="websocket":
                key=self.headers.get("Sec-WebSocket-Key",""); self.send_response(101); self.send_header("Upgrade","websocket"); self.send_header("Connection","Upgrade"); self.send_header("Sec-WebSocket-Accept",ws_accept(key)); self.end_headers()
                conn=self.connection
                try:
                    subscribed = False
                    while True:
                        opcode,data=read_ws(self.rfile)
                        if opcode == 0x9:
                            conn.sendall(ws_frame(data, 0xA))
                            continue
                        if opcode == 0xA:
                            continue
                        if opcode == 0x8:
                            conn.sendall(ws_frame(data, 0x8))
                            break
                        if opcode != 0x1 or subscribed:
                            continue
                        msg=json.loads(data); subs=frozenset(msg["camera_ids"])
                        if msg.get("type")!="subscribe" or not subs<=gateway.cameras: raise ValueError()
                        with gateway.lock: gateway.clients.append((conn,subs)); conn.sendall(frame({"type":"subscribed","camera_ids":sorted(subs)})); conn.sendall(frame({"type":"active_camera",**gateway.state()})); [gateway.aggregate(c) for c in subs if c==gateway.active]
                        subscribed = True
                except Exception: pass
                finally:
                    with gateway.lock: gateway.clients[:]=[x for x in gateway.clients if x[0] is not conn]
                return
            return self._get()
    ThreadingHTTPServer((os.getenv("ANNOTATOR_BIND_HOST","0.0.0.0"),int(os.getenv("ANNOTATOR_PORT","18080"))),Handler).serve_forever()
if __name__=="__main__": main()
