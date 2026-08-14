# ci/build.py
"""PyInstaller 打包, 按 BUILD_TARGET 选后端 wheel。用法: python ci/build.py <target>
target: windows-x64-cpu|cuda|vulkan | linux-x64-cpu|cuda|vulkan | macos-arm64-metal|cpu
"""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

TARGETS = {
    "windows-x64-cpu":    {"onnx": "onnxruntime", "llama": ["llama-cpp-python", ""]},
    "windows-x64-cuda":   {"onnx": "onnxruntime-gpu", "llama": ["llama-cpp-python", "cuda"]},
    "windows-x64-vulkan": {"onnx": "onnxruntime-directml", "llama": ["llama-cpp-python", "vulkan"]},
    "linux-x64-cpu":      {"onnx": "onnxruntime", "llama": ["llama-cpp-python", ""]},
    # linux CUDA 从源码构建: abetlen 预编译 cu124 wheel 把 AVX512 编进了 libggml-cpu,
    # 在无 AVX512 的 CPU (Zen1/2 等) 上即使只跑 CUDA 也会 SIGILL。
    "linux-x64-cuda":     {"onnx": "onnxruntime-gpu", "llama": ["llama-cpp-python", "cuda"],
                           "source_build": True,
                           "llama_version": "0.3.28"},
    "linux-x64-vulkan":   {"onnx": "onnxruntime", "llama": ["llama-cpp-python", "vulkan"]},
    "macos-arm64-metal":  {"onnx": "onnxruntime", "llama": ["llama-cpp-python", "metal"]},   # onnxruntime macOS arm64 wheel 内置 MPS EP (onnxruntime-silicon 已下架)
    "macos-arm64-cpu":    {"onnx": "onnxruntime", "llama": ["llama-cpp-python", ""]},
}

HIDDEN = [
    "scipy.special._ufuncs_cxx", "scipy.special.cython_special",
    "imageio_ffmpeg",
]


LLAMA_INDEX = {"cpu": "cpu", "vulkan": "vulkan", "cuda": "cu124", "metal": "metal"}


def _pip_install(cmd, retries=3, env=None):
    """pip install with retries (transient CDN/CRC corruptions happen)."""
    for i in range(retries):
        try:
            subprocess.run(cmd, check=True, env=env)
            return
        except subprocess.CalledProcessError:
            if i == retries - 1:
                raise
            print(f"[build] pip install failed; retrying {i + 1}/{retries}", file=sys.stderr)


def main(target: str):
    if target not in TARGETS:
        sys.exit(f"unknown target {target}; choose from {sorted(TARGETS)}")
    t = TARGETS[target]
    backend = t["llama"][1] or "cpu"
    # App runtime deps + pyinstaller from PyPI. llama-cpp-python's runtime deps
    # (typing-extensions, diskcache, jinja2, tqdm, numpy) are listed explicitly
    # because the llama wheel is later force-installed with --no-deps (its index
    # has no PyPI fallback), and onnxruntime-directml does not provide them.
    _pip_install([sys.executable, "-m", "pip", "install",
                  "pyinstaller", t["onnx"],
                  "numpy", "scipy", "gguf", "pyyaml", "imageio-ffmpeg",
                  "typing-extensions", "diskcache", "jinja2", "tqdm"])
    # Force the correct llama-cpp-python backend wheel (exclusive index + force
    # reinstall + no-deps, so a pre-existing CPU/source build is replaced).
    if target.startswith("macos"):
        # abetlen's macosx arm64 wheels are currently corrupt (BadCRC on
        # libggml-base.dylib); build from the PyPI sdist instead — a macOS
        # arm64 source build enables Metal by default.
        _pip_install([sys.executable, "-m", "pip", "install", "--force-reinstall",
                      "llama-cpp-python"])
    elif t.get("source_build"):
        # Build llama-cpp-python from the PyPI sdist with explicit CMAKE_ARGS.
        # GGML_NATIVE=OFF keeps the CPU backend AVX512-free (avoids SIGILL on
        # non-AVX512 machines); GGML_CUDA=ON enables GPU offload for the
        # decoder. Requires nvcc + cmake + ninja in the build env (see
        # build_linux_container.sh for linux-x64-cuda).
        ver = t.get("llama_version")
        spec = f"llama-cpp-python=={ver}" if ver else "llama-cpp-python"
        cmake_args = os.environ.get(
            "CMAKE_ARGS",
            "-DGGML_CUDA=ON -DGGML_NATIVE=OFF "
            "-DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF "
            "-DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF")
        env = dict(os.environ, CMAKE_ARGS=cmake_args)
        _pip_install([sys.executable, "-m", "pip", "install", "--force-reinstall",
                      spec], env=env)
    else:
        _pip_install([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                      "llama-cpp-python",
                      "--index-url",
                      "https://abetlen.github.io/llama-cpp-python/whl/" + LLAMA_INDEX[backend]])

    data_sep = ";" if sys.platform.startswith("win") else ":"
    cmd = [sys.executable, "-m", "PyInstaller", "--onefile", "--name", "q3asr",
           "--collect-all", "imageio_ffmpeg",
           "--collect-all", "llama_cpp",
           f"--add-data=resources{data_sep}resources"]
    for mod in HIDDEN:
        cmd += ["--hidden-import", mod]
    cmd.append("q3asr/__main__.py")
    subprocess.run(cmd, check=True)
    exe = Path("dist/q3asr.exe" if sys.platform.startswith("win") else "dist/q3asr")
    assert exe.exists()

    os_name = target.split("-")[0]
    arch = target.split("-")[1]
    backend = target.split("-")[2]
    zip_name = f"qwen3-asr-{os_name}-{arch}-{backend}.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(exe, exe.name)
        manifest = {"os": os_name, "arch": arch, "backend": backend,
                    "cli": "q3asr.exe" if os_name == "windows" else "q3asr",
                    "version": os.environ.get("GITHUB_REF_NAME", "dev")}
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    sha = hashlib.sha256(Path(zip_name).read_bytes()).hexdigest()
    print(f"built {zip_name} sha256={sha}")


if __name__ == "__main__":
    main(sys.argv[1])
