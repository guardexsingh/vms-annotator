function calculateVideoRect(containerWidth, containerHeight, videoWidth, videoHeight) {
  if (![containerWidth, containerHeight, videoWidth, videoHeight].every(value => value > 0)) return null;
  const scale = Math.min(containerWidth / videoWidth, containerHeight / videoHeight);
  const width = videoWidth * scale, height = videoHeight * scale;
  return { x:(containerWidth-width)/2, y:(containerHeight-height)/2, width, height };
}

function mapNormalizedBox(box, rect) {
  return { x:rect.x + box.x*rect.width, y:rect.y + box.y*rect.height,
    width:box.width*rect.width, height:box.height*rect.height };
}

function canvasPixelSize(width, height, devicePixelRatio) {
  const ratio = Math.max(1, devicePixelRatio || 1);
  return { width:Math.round(width*ratio), height:Math.round(height*ratio), ratio };
}

function buildSubscription(cameraIds) {
  return { type:"subscribe", camera_ids:[...new Set(cameraIds)] };
}

function metadataAge(result, nowPerformance) {
  const receivedAge = result.result_to_browser_latency_ms ?? result.metadata_age_ms ?? 0;
  return Math.max(0, Number(receivedAge) + nowPerformance - result.receivedAt);
}

function isExpired(result, ttlMs, nowPerformance) { return !result || metadataAge(result, nowPerformance) > ttlMs; }

function visibleTrackBoxes(result, holdMs, nowPerformance) {
  if (!result) return [];
  const sinceReceipt = Math.max(0, nowPerformance - result.receivedAt);
  return (result.boxes || []).filter(box =>
    box.track_state !== "lost"
    || Number(box.last_confirmed_age_ms || 0) + sinceReceipt <= holdMs
  );
}

function supportsHevcCodec(capabilities) {
  return Boolean(capabilities?.codecs?.some(codec => /^video\/(h265|hevc)$/i.test(codec.mimeType || "")));
}

function negotiatedVideoCodec(sdp) {
  const payloads=(sdp.match(/^m=video\s+\d+\s+\S+\s+(.+)$/m)?.[1] || "").trim().split(/\s+/);
  const mappings=new Map([...sdp.matchAll(/^a=rtpmap:(\d+)\s+([^/\s]+)/gm)].map(match=>[match[1],match[2].toLowerCase()]));
  const codec=payloads.map(payload=>mappings.get(payload)).find(Boolean);
  if (codec === "h265" || codec === "hevc") return "hevc";
  if (codec === "h264") return "h264";
  return codec || "unknown";
}

function nextActiveCamera(currentCameraId, clickedCameraId) {
  return currentCameraId === clickedCameraId ? null : clickedCameraId;
}

function controlState(status) {
  return ({loading:"starting", ready:"active", failed:"error"})[status] || status || "disabled";
}

if (typeof module !== "undefined") module.exports = {
  calculateVideoRect, mapNormalizedBox, canvasPixelSize, buildSubscription, metadataAge, isExpired,
  visibleTrackBoxes, supportsHevcCodec, negotiatedVideoCodec, nextActiveCamera, controlState
};

if (typeof document !== "undefined") {
  const tiles = new Map(), results = new Map(), players = new Map();
  const template = document.querySelector("#tile-template"), grid = document.querySelector("#grid");
  let cameraConfig = [], detectionTtlMs = 2000, trackHoldMs = 1500, metadataPath = "/ws/detections", metadataSocket = null, reconnectTimer = null;
  let activeCameraId = null, activeDetectionStatus = "disabled", detectionRequestPending = false, intendedCameraId = null;
  const debugLabels = new URLSearchParams(location.search).get("debug") === "1";
  const format = (value, suffix="") => value == null ? "—" : `${Number(value).toFixed(1)}${suffix}`;
  const controlLog = (event, detail={}) => console.info("[detection-control]", {event,...detail});

  class WhepPlayer {
    constructor(video, tile, url, camera) {
      this.video=video; this.tile=tile; this.url=url; this.camera=camera; this.pc=null; this.sessionUrl=null;
      this.timer=null; this.statsTimer=null; this.closed=false; this.unsupported=false; this.frameTimes=[];
      this.frameGeneration=0; this.previousStats=null;
    }
    countFrame(now, generation) {
      if (generation !== this.frameGeneration) return;
      this.frameTimes.push(now); while (this.frameTimes.length && now-this.frameTimes[0] > 1000) this.frameTimes.shift();
      if (this.pc?.connectionState === "connected") this.tile.querySelector(".video-status").textContent=`Live · ${this.frameTimes.length.toFixed(1)} FPS`;
      if (!this.closed && "requestVideoFrameCallback" in this.video) this.video.requestVideoFrameCallback(time=>this.countFrame(time,generation));
    }
    async connect() {
      if (this.closed) return;
      this.disposeConnection();
      const pc = this.pc = new RTCPeerConnection();
      pc.addTransceiver("video", {direction:"recvonly"});
      pc.ontrack = event => { this.video.srcObject=event.streams[0]; this.video.play().catch(()=>{}); this.frameTimes=[]; const generation=++this.frameGeneration; if ("requestVideoFrameCallback" in this.video) this.video.requestVideoFrameCallback(time=>this.countFrame(time,generation)); redraw(this.tile.dataset.cameraId); };
      pc.onconnectionstatechange = () => {
        if (pc !== this.pc) return;
        const live=pc.connectionState === "connected";
        setVideoStatus(this.tile, live ? "Live" : pc.connectionState, live);
        if (live) this.startStats();
        if (["failed","disconnected","closed"].includes(pc.connectionState)) this.scheduleReconnect();
      };
      try {
        const offer=await pc.createOffer(); await pc.setLocalDescription(offer);
        await new Promise(resolve => { if (pc.iceGatheringState === "complete") resolve(); else pc.onicegatheringstatechange=()=>pc.iceGatheringState === "complete" && resolve(); });
        const response=await fetch(this.url,{method:"POST",headers:{"Content-Type":"application/sdp",Accept:"application/sdp"},body:pc.localDescription.sdp});
        if (!response.ok) throw new Error(`WHEP ${response.status}`);
        this.sessionUrl=new URL(response.headers.get("location") || "",this.url).toString();
        const answer=await response.text(), codec=negotiatedVideoCodec(answer);
        if (this.camera.expected_codec === "hevc" && codec !== "hevc") {
          const error=new Error(`Unsupported negotiated codec: ${codec}`); error.unsupportedCodec=true; throw error;
        }
        this.tile.querySelector(".codec-status").textContent=`Source HEVC · WebRTC ${codec.toUpperCase()}`;
        await pc.setRemoteDescription({type:"answer",sdp:answer});
      } catch (error) {
        if (pc !== this.pc) return;
        if (error.unsupportedCodec) {
          this.unsupported=true; this.disposeConnection(); setVideoStatus(this.tile,"Unsupported codec",false);
          this.tile.querySelector(".offline").textContent="HEVC was not negotiated — no H.264 fallback";
        } else { setVideoStatus(this.tile,"Offline",false); this.scheduleReconnect(); }
      }
    }
    startStats() {
      if (this.statsTimer) return;
      this.statsTimer=setInterval(()=>this.collectStats(),1000); this.collectStats();
    }
    async collectStats() {
      if (!this.pc || this.pc.connectionState !== "connected") return;
      const reports=await this.pc.getStats(), codecs=new Map();
      reports.forEach(report=>{ if (report.type === "codec") codecs.set(report.id,report); });
      let inbound=null; reports.forEach(report=>{ if (report.type === "inbound-rtp" && (report.kind === "video" || report.mediaType === "video")) inbound=report; });
      if (!inbound) return;
      const codecName=(codecs.get(inbound.codecId)?.mimeType || "video/unknown").split("/").pop().toLowerCase();
      if (this.camera.expected_codec === "hevc" && !["hevc","h265"].includes(codecName)) {
        this.unsupported=true; this.disposeConnection(); setVideoStatus(this.tile,"Unsupported codec",false);
        this.tile.querySelector(".offline").textContent="Browser received a non-HEVC codec — fallback refused";
        return;
      }
      const previous=this.previousStats, elapsed=previous ? Math.max(.001,(inbound.timestamp-previous.timestamp)/1000) : null;
      const received=Number(inbound.framesReceived ?? inbound.framesDecoded ?? 0), decoded=Number(inbound.framesDecoded ?? 0);
      const receivedFps=elapsed ? Math.max(0,(received-previous.received)/elapsed) : null;
      const decodedFps=elapsed ? Math.max(0,(decoded-previous.decoded)/elapsed) : null;
      const jitterCount=Number(inbound.jitterBufferEmittedCount || 0);
      const jitterMs=jitterCount ? 1000*Number(inbound.jitterBufferDelay || 0)/jitterCount : null;
      this.previousStats={timestamp:inbound.timestamp,received,decoded};
      this.tile.querySelector(".browser-stats").textContent=`${format(receivedFps," recv FPS")} · ${format(decodedFps," dec FPS")} · drop ${inbound.framesDropped || 0} · jitter ${format(jitterMs," ms")}`;
      fetch("/api/browser-stats",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
        camera_id:this.camera.id,webrtc_codec:codecName,browser_received_fps:receivedFps,
        browser_decoded_fps:decodedFps,browser_frames_dropped:Number(inbound.framesDropped || 0),
        browser_jitter_buffer_delay_ms:jitterMs
      })}).catch(()=>{});
    }
    scheduleReconnect() { if (this.closed || this.unsupported || this.timer) return; this.disposeConnection(); this.timer=setTimeout(()=>{this.timer=null;this.connect();},1500); }
    disposeConnection() {
      ++this.frameGeneration; clearInterval(this.statsTimer); this.statsTimer=null; this.previousStats=null;
      const pc=this.pc; this.pc=null; if (pc) pc.close();
      if (this.sessionUrl) { fetch(this.sessionUrl,{method:"DELETE",keepalive:true}).catch(()=>{}); this.sessionUrl=null; }
    }
    close() { this.closed=true; clearTimeout(this.timer); this.disposeConnection(); }
  }

  function setVideoStatus(tile, text, live) {
    tile.classList.toggle("is-offline",!live); tile.querySelector(".state").textContent=live ? "live" : text.toLowerCase();
    tile.querySelector(".state").className=`state ${live ? "live" : "offline"}`; tile.querySelector(".video-status").textContent=text;
  }

  function createTile(camera, whepPort) {
    const tile=template.content.firstElementChild.cloneNode(true); tile.dataset.cameraId=camera.id;
    const stage=tile.querySelector(".camera-stage"), video=tile.querySelector("video");
    stage.dataset.cameraId=camera.id; tile.querySelector("strong").textContent=camera.name;
    tile.querySelector(".video-path").textContent=`${camera.video_path} · transcode ${camera.transcoding ? "yes" : "no"}`;
    tile.querySelector(".codec-status").textContent=`Expected ${camera.expected_codec.toUpperCase()}`;
    const toggle=tile.querySelector(".detection-toggle");
    toggle.disabled=!camera.detection_enabled;
    if (!camera.detection_enabled) {
      const status=tile.querySelector(".detector-status"); status.textContent="Unavailable"; status.className="detector-status disabled";
      toggle.title=toggle.ariaLabel="Person detection unavailable";
    } else {
      toggle.addEventListener("click",event=>{
        event.preventDefault(); event.stopPropagation();
        controlLog("button_clicked",{camera_id:camera.id});
        void toggleDetection(camera.id);
      });
    }
    grid.append(tile); tiles.set(camera.id,tile);
    const scheme=location.protocol === "https:" ? "https" : "http";
    const capabilities=globalThis.RTCRtpReceiver?.getCapabilities?.("video");
    if (camera.expected_codec === "hevc" && !supportsHevcCodec(capabilities)) {
      setVideoStatus(tile,"HEVC unsupported",false);
      tile.querySelector(".offline").textContent="This browser does not advertise HEVC WebRTC decode — no fallback";
      return;
    }
    const player=new WhepPlayer(video,tile,`${scheme}://${location.hostname}:${whepPort}/${camera.stream_path}/whep`,camera);
    players.set(camera.id,player); player.connect();
    const observer=new ResizeObserver(()=>redraw(camera.id)); observer.observe(stage);
    video.addEventListener("loadedmetadata",()=>redraw(camera.id)); video.addEventListener("resize",()=>redraw(camera.id));
  }

  function redraw(cameraId) {
    const tile=tiles.get(cameraId); if (!tile) return;
    const stage=tile.querySelector(".camera-stage"), video=tile.querySelector("video"), canvas=tile.querySelector("canvas");
    const cssWidth=stage.clientWidth, cssHeight=stage.clientHeight, pixels=canvasPixelSize(cssWidth,cssHeight,window.devicePixelRatio);
    if (canvas.width !== pixels.width || canvas.height !== pixels.height) { canvas.width=pixels.width; canvas.height=pixels.height; }
    const context=canvas.getContext("2d"); context.setTransform(pixels.ratio,0,0,pixels.ratio,0,0); context.clearRect(0,0,cssWidth,cssHeight);
    const result=results.get(cameraId), age=result ? metadataAge(result,performance.now()) : null;
    tile.querySelector(".age").textContent=age == null ? "—" : `${format(age," ms")} · arrival ${format(result?.result_to_browser_latency_ms," ms")}`;
    if (isExpired(result,detectionTtlMs,performance.now())) {
      tile.querySelector(".people").textContent="0";
      if (result && activeCameraId === cameraId && activeDetectionStatus === "active") {
        const status=tile.querySelector(".detector-status"); status.textContent="Active · waiting"; status.className="detector-status stale";
      }
      return;
    }
    const rect=calculateVideoRect(cssWidth,cssHeight,video.videoWidth || result.frame_width,video.videoHeight || result.frame_height);
    if (!rect) return;
    context.lineWidth=2; context.font="13px system-ui,sans-serif"; context.textBaseline="top";
    const visibleBoxes=visibleTrackBoxes(result,trackHoldMs,performance.now());
    for (const box of visibleBoxes) {
      const draw=mapNormalizedBox(box,rect), x=Math.round(draw.x)+.5, y=Math.round(draw.y)+.5;
      context.strokeStyle="#55e991"; context.strokeRect(x,y,Math.round(draw.width),Math.round(draw.height));
      const identity=debugLabels && box.track_id != null ? ` #${box.track_id}` : "";
      const label=`person${identity} ${Math.round(box.confidence*100)}%`, textWidth=context.measureText(label).width;
      context.fillStyle="#07140ddd"; context.fillRect(x,y,Math.ceil(textWidth)+8,19); context.fillStyle="#8cffb6"; context.fillText(label,x+4,y+2);
    }
    tile.querySelector(".people").textContent=String(visibleBoxes.length);
  }

  function clearCamera(cameraId) {
    results.delete(cameraId);
    const tile=tiles.get(cameraId);
    if (!tile) return;
    tile.querySelector(".people").textContent="0";
    tile.querySelector(".age").textContent="—";
    redraw(cameraId);
  }

  function applyActiveCamera(cameraId, status) {
    activeCameraId=cameraId ?? null;
    activeDetectionStatus=controlState(status || (cameraId ? "starting" : "disabled"));
    controlLog("active_camera_state",{camera_id:activeCameraId,status:activeDetectionStatus});
    for (const camera of cameraConfig) {
      const tile=tiles.get(camera.id); if (!tile || !camera.detection_enabled) continue;
      const selected=camera.id === activeCameraId, button=tile.querySelector(".detection-toggle");
      const state=selected ? activeDetectionStatus : "disabled";
      button.dataset.state=state;
      button.disabled=detectionRequestPending;
      button.textContent=state === "active" ? "✓" : state === "starting" ? "…" : state === "error" ? "!" : "○";
      button.setAttribute("aria-pressed",String(selected && state === "active"));
      const cameraName=camera.name || camera.id;
      button.title=state === "active" ? "Disable person detection" : state === "starting" ? "Detection starting"
        : state === "error" ? "Detection failed" : "Enable person detection";
      button.setAttribute("aria-label",`${button.title}${state === "active" ? " for " : " for "}${cameraName}`);
      const detector=tile.querySelector(".detector-status");
      detector.textContent=state === "active" ? "Active" : state === "starting" ? "Starting"
        : state === "stopping" ? "Stopping" : state === "error" ? "Failed" : "Off";
      detector.className=`detector-status ${state}`;
      const detail=tile.querySelector(".detector-message");
      if (state === "starting") detail.textContent="Starting detector…";
      else if (state === "active") detail.textContent="Receiving track metadata";
      else if (state === "disabled") detail.textContent="—";
      if (!selected) clearCamera(camera.id);
    }
  }

  function refreshToggleAvailability() {
    for (const camera of cameraConfig) {
      const button=tiles.get(camera.id)?.querySelector(".detection-toggle");
      if (button && camera.detection_enabled) button.disabled=detectionRequestPending;
    }
  }

  async function toggleDetection(cameraId) {
    const base=intendedCameraId === null && !detectionRequestPending ? activeCameraId : intendedCameraId;
    intendedCameraId=nextActiveCamera(base,cameraId);
    if (detectionRequestPending) {
      controlLog("selection_queued",{camera_id:intendedCameraId});
      return;
    }
    detectionRequestPending=true;
    refreshToggleAvailability();
    try {
      while (true) {
        const requested=intendedCameraId;
        applyActiveCamera(requested,requested ? "starting" : "disabled");
        controlLog("api_request",{camera_id:requested});
        const response=await fetch("/api/detection/active-camera",{
          method:"POST",headers:{"Content-Type":"application/json"},
          body:JSON.stringify({camera_id:requested})
        });
        const text=await response.text(); let state={};
        try { state=text ? JSON.parse(text) : {}; } catch (_) { state={error:"Invalid API response"}; }
        controlLog("api_response",{camera_id:requested,status:response.status,body:state});
        if (!response.ok) throw new Error(state.error || `Detection control ${response.status}`);
        applyActiveCamera(state.camera_id,state.status);
        if (requested === intendedCameraId) break;
      }
    } catch (error) {
      const message=error instanceof Error ? error.message : "Detection request failed";
      console.error("[detection-control] api_failure",{camera_id:intendedCameraId,message});
      const tile=tiles.get(intendedCameraId);
      if (tile) tile.querySelector(".detector-message").textContent=message;
      applyActiveCamera(intendedCameraId,intendedCameraId ? "error" : "disabled");
    } finally { detectionRequestPending=false; refreshToggleAvailability(); }
  }

  function connectMetadata() {
    if (metadataSocket && [WebSocket.OPEN,WebSocket.CONNECTING].includes(metadataSocket.readyState)) return;
    clearTimeout(reconnectTimer); const scheme=location.protocol === "https:" ? "wss" : "ws";
    const socket=metadataSocket=new WebSocket(`${scheme}://${location.host}${metadataPath}`); let subscribed=false;
    socket.onopen=()=>{ if (!subscribed) { socket.send(JSON.stringify(buildSubscription(cameraConfig.filter(camera=>camera.detection_enabled).map(camera=>camera.id)))); subscribed=true; } };
    socket.onmessage=event=>{
      let message; try { message=JSON.parse(event.data); } catch (_) { return; }
      if (message.type === "active_camera") {
        controlLog("websocket_active_camera",message);
        applyActiveCamera(message.camera_id,message.status);
        return;
      }
      const tile=tiles.get(message.camera_id); if (!tile) return;
      if (message.type === "tracks" && message.camera_id === activeCameraId) {
        controlLog("tracks_received",{camera_id:message.camera_id,sequence:message.sequence,boxes:message.boxes?.length || 0});
        message.receivedAt=performance.now();
        message.result_to_browser_latency_ms=Math.max(0,Date.now()-Number(message.completed_at_unix_ms || Date.now()));
        results.set(message.camera_id,message);
        tile.querySelector(".inference").textContent=`${format(message.actual_yolo_fps," FPS")} / 1.0 target`;
        if (debugLabels) {
          const source=message.prediction_only ? "predicted" : "yolo";
          tile.querySelector(".detector-message").textContent=
            `YOLO: ${format(message.actual_yolo_fps," FPS")} · Tracker target: ${format(message.configured_bytetrack_prediction_fps," FPS")} · Tracker actual: ${format(message.actual_tracker_fps," FPS")} · Source: ${source} · Confirmed age: ${format(message.last_yolo_age_ms," ms")}`;
        }
        redraw(message.camera_id);
      }
      if (message.type === "clear_tracks") clearCamera(message.camera_id);
      if (message.type === "detector_status") {
        controlLog("detector_status",message);
        if (message.message) tile.querySelector(".detector-message").textContent=message.message;
        if (message.camera_id === activeCameraId && ["starting","active","stopping","error"].includes(message.status)) {
          applyActiveCamera(message.camera_id,message.status);
        }
        if (["stale","error","offline","disabled"].includes(message.status)) clearCamera(message.camera_id);
      }
    };
    socket.onclose=()=>{ if (socket === metadataSocket) { metadataSocket=null; controlLog("websocket_closed"); const tile=tiles.get(activeCameraId); if (tile) tile.querySelector(".detector-message").textContent="Metadata connection disconnected; reconnecting"; reconnectTimer=setTimeout(connectMetadata,1500); } };
    socket.onerror=()=>socket.close();
  }

  function updateMetrics(metrics) {
    const detector=metrics.detector?.status || metrics.detector?.state || "—";
    document.querySelector("#system").textContent=`Detector ${detector} · p50 ${format(metrics.inference?.p50_ms," ms")} · p95 ${format(metrics.inference?.p95_ms," ms")} · CPU ${format(metrics.system?.cpu_percent,"%")}`;
    for (const [id,data] of Object.entries(metrics.cameras || {})) { const tile=tiles.get(id); if (!tile) continue;
      tile.querySelector(".inference").textContent=`${format(data.completed_inference_fps," FPS")} / ${format(data.requested_inference_fps," target")}`;
      tile.querySelector(".capture").textContent=`${format(data.ai_capture_fps," FPS")} · replaced ${data.ai_frames_replaced || 0}`;
      tile.querySelector(".source-stats").textContent=`${String(data.source_codec || "unknown").toUpperCase()} · ${format(data.source_fps," src FPS")} · ${format(data.mediamtx_fps," MTX FPS")}`;
      if (data.webrtc_codec && data.webrtc_codec !== "unknown") tile.querySelector(".codec-status").textContent=`Source ${String(data.source_codec).toUpperCase()} · WebRTC ${String(data.webrtc_codec).toUpperCase()}`;
      if (!results.has(id) && activeCameraId === id) tile.querySelector(".people").textContent=String(data.person_count || 0);
    }
  }

  async function poll() { try { const response=await fetch("/metrics",{cache:"no-store"}); updateMetrics(await response.json()); } catch (_) {} finally { setTimeout(poll,1000); } }
  async function initialize() {
    const response=await fetch("/api/cameras",{cache:"no-store"}), config=await response.json();
    cameraConfig=config.cameras || []; detectionTtlMs=config.detection_ttl_ms || 2000;
    trackHoldMs=config.tracking?.hold_box_ms || 1500; metadataPath=config.metadata_path || metadataPath;
    cameraConfig.forEach(camera=>createTile(camera,config.whep_port || 18889));
    applyActiveCamera(config.active_camera?.camera_id,config.active_camera?.status || "disabled");
    connectMetadata(); poll();
    setInterval(()=>cameraConfig.forEach(camera=>redraw(camera.id)),100);
  }
  addEventListener("resize",()=>cameraConfig.forEach(camera=>redraw(camera.id)));
  document.addEventListener("fullscreenchange",()=>cameraConfig.forEach(camera=>redraw(camera.id)));
  addEventListener("beforeunload",()=>{ if (metadataSocket) metadataSocket.close(); for (const player of players.values()) player.close(); },{once:true});
  initialize().catch(()=>{ document.querySelector("#system").textContent="Configuration unavailable"; });
}
