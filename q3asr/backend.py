"""硬件后端探测与选择。优先级: --device > QASR_DEVICE > auto。"""
import os
from typing import Callable

BACKENDS = ("cuda", "vulkan", "metal", "cpu")
_ORDER = ("cuda", "vulkan", "metal", "cpu")


def _default_available(backend: str) -> bool:
    if backend == "cuda":
        try:
            import llama_cpp  # noqa: F401
            return "CUDA" in getattr(llama_cpp, "__backend__", "") or "CUDA" in os.environ.get("LLAMA_CUDA", "")
        except Exception:
            return False
    if backend == "vulkan":
        try:
            import llama_cpp  # noqa: F401
            return "Vulkan" in getattr(llama_cpp, "__backend__", "") or "Vulkan" in os.environ.get("LLAMA_VULKAN", "")
        except Exception:
            return False
    if backend == "metal":
        import platform
        return platform.system() == "Darwin"
    if backend == "cpu":
        return True
    return False


def detect_backend(device: str = "auto",
                   _available: Callable[[str], bool] | None = None) -> str:
    avail = _available or _default_available
    if device != "auto":
        if device not in BACKENDS:
            raise ValueError(f"unknown device: {device}; choose from {BACKENDS}")
        return device
    for b in _ORDER:
        if avail(b):
            return b
    return "cpu"


def onnx_providers(backend: str) -> list[str]:
    if backend == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if backend == "metal":
        return ["MPSExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def llama_backend(backend: str) -> str:
    return backend
