# Jetson/JetPack 6.2 (L4T R36.4) TensorRT runtime. The existing engine was
# serialized with the R36.4 TensorRT 10.3.0.30 package build, so pin the
# container-side native libraries and Python binding to that exact build.
FROM python:3.12-slim AS gateway
WORKDIR /app
COPY app/gateway_main.py ./app/gateway_main.py
EXPOSE 18080
CMD ["python", "-m", "app.gateway_main"]

FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime AS legacy-ai

ARG JETSON_L4T_RELEASE=r36.4
ARG TENSORRT_VERSION=10.3.0.30-1+cuda12.5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    ANNOTATOR_L4T_RELEASE=${JETSON_L4T_RELEASE} \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    XDG_CONFIG_HOME=/tmp/xdg-config

RUN curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
        -o /tmp/jetson-ota-public.asc \
    && echo "576f852981855e5c6cfb9b625ffb51b984ca451f1181b2e70435b005034fad55  /tmp/jetson-ota-public.asc" | sha256sum -c - \
    && gpg --dearmor --batch --yes \
        --output /usr/share/keyrings/nvidia-jetson-ota.gpg /tmp/jetson-ota-public.asc \
    && rm /tmp/jetson-ota-public.asc \
    && printf '%s\n' \
        "deb [signed-by=/usr/share/keyrings/nvidia-jetson-ota.gpg] https://repo.download.nvidia.com/jetson/common ${JETSON_L4T_RELEASE} main" \
        "deb [signed-by=/usr/share/keyrings/nvidia-jetson-ota.gpg] https://repo.download.nvidia.com/jetson/t234 ${JETSON_L4T_RELEASE} main" \
        > /etc/apt/sources.list.d/nvidia-jetson.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        python3-pip \
        libnvinfer10=${TENSORRT_VERSION} \
        libnvinfer-plugin10=${TENSORRT_VERSION} \
        libnvinfer-vc-plugin10=${TENSORRT_VERSION} \
        libnvinfer-lean10=${TENSORRT_VERSION} \
        libnvinfer-dispatch10=${TENSORRT_VERSION} \
        libnvonnxparsers10=${TENSORRT_VERSION} \
        libnvinfer-bin=${TENSORRT_VERSION} \
        python3-libnvinfer=${TENSORRT_VERSION} \
        tensorrt-libs=${TENSORRT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# This is the existing runtime-only dependency set for TensorRT plus the
# intentionally configurable diagnostic backends. It contains no tests or
# development tooling; TensorRT is supplied by the pinned NVIDIA packages.
COPY requirements-runtime.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-runtime.txt

COPY app ./app
COPY lap.py ./
COPY config/metadata-only.yaml ./config/metadata-only.yaml
COPY web ./web

# Models and engine are bind-mounted read-only by Compose. Create the target
# directory so missing mounts fail clearly in the TensorRT preflight.
RUN mkdir -p /app/models

EXPOSE 18080

# Exec form forwards SIGTERM/SIGINT directly to app.main, which owns orderly
# shutdown of the active capture/scheduler/tracker session.
CMD ["python3", "-m", "app.main", "--config", "/app/config/metadata-only.yaml"]
