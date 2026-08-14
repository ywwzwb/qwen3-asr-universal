"""硬件后端探测与选择。优先级: --device > QASR_DEVICE > auto。

后端可用性由 llama_cpp 捆绑的 GGML 后端库决定: lib 目录里存在
libggml-cuda.so / libggml-vulkan.so 即认为该后端被编译进当前构建
(不再依赖 abetlen wheel 上的 __backend__ 属性, 那个属性在各版本不一致)。
"""
import ctypes
from pathlib import Path
from typing import Callable

BACKENDS = ("cuda", "vulkan", "metal", "cpu")
_ORDER = ("cuda", "vulkan", "metal", "cpu")

# GGML 后端共享库文件 → 对应后端名
_BACKEND_LIBS = {
    "cuda": "libggml-cuda.so",
    "vulkan": "libggml-vulkan.so",
}

# 各 GPU 后端在运行时需要的宿主共享库 (不含 CUDA driver libcuda, 它由驱动提供)
_RUNTIME_LIBS = {
    "cuda": ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"),
    "vulkan": ("libvulkan.so.1",),
}


def _llama_cpp_lib_dir() -> Path | None:
    try:
        import llama_cpp
        return Path(llama_cpp.__file__).resolve().parent / "lib"
    except Exception:
        return None


def _backend_compiled(backend: str) -> bool:
    """该后端是否被编译进当前 llama_cpp 构建 (检查捆绑的 GGML 后端库)。"""
    libdir = _llama_cpp_lib_dir()
    if libdir is None or not libdir.is_dir():
        return False
    for name in libdir.iterdir():
        if name.name.startswith(_BACKEND_LIBS.get(backend, "\0")):
            return True
    return False


def runtime_missing(backend: str) -> list[str]:
    """backend 运行时缺失的宿主共享库 (空列表 = 可用)。

    CUDA/Vulkan 构建即使被编译进 llama_cpp, 运行时仍需要系统提供
    cudart/cublas (cuda) 或 libvulkan (vulkan)。缺失时列出具体库名,
    供上层给出明确报错, 而不是让 llama.cpp 加载时静默失败。
    """
    if backend == "cpu" or backend == "metal":
        return []
    missing = []
    for lib in _RUNTIME_LIBS.get(backend, ()):
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append(lib)
    return missing


def _default_available(backend: str) -> bool:
    if backend == "cuda":
        return _backend_compiled("cuda")
    if backend == "vulkan":
        return _backend_compiled("vulkan")
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
    if backend == "vulkan":
        # The vulkan build bundles onnxruntime-directml on Windows; use it so the
        # encoder also runs on the GPU. On other platforms fall back to CPU.
        try:
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                return ["DmlExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return ["CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def llama_backend(backend: str) -> str:
    return backend
