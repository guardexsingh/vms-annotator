"""Inference backend contract and explicit host-capability selection."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import DetectionConfig, DetectionResult, Frame


class DetectionBackend(ABC):
    name: str
    device: str
    precision: str

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def warmup(self) -> dict[str, object]:
        ...

    @abstractmethod
    def infer(self, frames: list[tuple[str, Frame]]) -> list[DetectionResult]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


@dataclass(frozen=True)
class BackendSelection:
    requested: str
    selected: str
    device: str
    precision: str
    fallback_used: bool = False
    fallback_reason: str | None = None


def select_backend(config: DetectionConfig):
    """Select only a runtime proven usable on this host.

    TensorRT is installed system-wide but cannot initialize CUDA because the
    Jetson accelerator device nodes are not exposed. ONNX Runtime and OpenVINO
    are not installed. PyTorch CPU is therefore the sole usable backend.
    """
    from .yolo_detector import YoloPersonDetector

    requested = config.backend
    unavailable = {
        "tensorrt": "TensorRT CUDA initialization is unavailable",
        "onnxruntime": "ONNX Runtime is not installed",
        "openvino": "OpenVINO is not installed",
    }
    fallback_reason = unavailable.get(requested)
    if fallback_reason and not config.allow_backend_fallback:
        raise RuntimeError(fallback_reason)
    selected = "pytorch"
    selection = BackendSelection(
        requested=requested, selected=selected, device="cpu", precision="fp32",
        fallback_used=fallback_reason is not None,
        fallback_reason=fallback_reason,
    )
    backend = YoloPersonDetector(
        config.model, config.image_size, config.confidence,
        torch_threads=config.torch_threads,
        torch_interop_threads=config.torch_interop_threads,
    )
    return backend, selection
