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
    execution_provider: str | None = None


def _pytorch_backend(config: DetectionConfig):
    from .yolo_detector import YoloPersonDetector
    return YoloPersonDetector(
        config.model, config.image_size, config.confidence,
        torch_threads=config.torch_threads,
        torch_interop_threads=config.torch_interop_threads,
    )


def select_backend(config: DetectionConfig):
    """Select one verified backend; explicit TensorRT never falls back silently."""
    from .onnx_detector import OnnxPersonDetector
    from .tensorrt_detector import TensorRTPersonDetector

    requested = config.backend
    if requested == "tensorrt":
        if config.precision not in {"auto", "fp16"}:
            raise RuntimeError("TensorRT backend requires FP16 or auto precision")
        return TensorRTPersonDetector(config), BackendSelection(
            requested="tensorrt", selected="tensorrt", device="cuda", precision="fp16",
            execution_provider="TensorRT",
        )
    if requested == "onnx":
        if config.precision == "fp16":
            raise RuntimeError("ONNX CPU backend does not support FP16 precision")
        return OnnxPersonDetector(config), BackendSelection(
            requested="onnx", selected="onnx", device="cpu",
            precision="fp32",
            execution_provider="CPUExecutionProvider",
        )
    if requested == "auto":
        candidate = TensorRTPersonDetector(config)
        try:
            candidate.warmup()
            return candidate, BackendSelection(
                requested="auto", selected="tensorrt", device="cuda", precision="fp16",
                execution_provider="TensorRT",
            )
        except Exception as error:
            fallback_reason = getattr(candidate, "error_summary", None) or type(error).__name__
            candidate.close()
            if not config.allow_backend_fallback:
                raise RuntimeError(fallback_reason) from error
            return _pytorch_backend(config), BackendSelection(
                requested="auto", selected="pytorch", device="cpu", precision="fp32",
                fallback_used=True, fallback_reason=fallback_reason,
            )
    if requested != "pytorch":
        raise RuntimeError(f"Unsupported backend selection: {requested}")
    return _pytorch_backend(config), BackendSelection(
        requested="pytorch", selected="pytorch", device="cpu", precision="fp32",
    )
