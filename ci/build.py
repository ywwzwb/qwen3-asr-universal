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
    "linux-x64-cuda":     {"onnx": "onnxruntime-gpu", "llama": ["llama-cpp-python", "cuda"]},
    "linux-x64-vulkan":   {"onnx": "onnxruntime", "llama": ["llama-cpp-python", "vulkan"]},
    "macos-arm64-metal":  {"onnx": "onnxruntime", "llama": ["llama-cpp-python", "metal"]},   # onnxruntime macOS arm64 wheel 内置 MPS EP (onnxruntime-silicon 已下架)
    "macos-arm64-cpu":    {"onnx": "onnxruntime", "llama": ["llama-cpp-python", ""]},
}

HIDDEN = [
    "scipy.special._ufuncs_cxx", "scipy.special.cython_special",
    "imageio_ffmpeg",
]


def main(target: str):
    if target not in TARGETS:
        sys.exit(f"unknown target {target}; choose from {sorted(TARGETS)}")
    t = TARGETS[target]
    backend = t["llama"][1] or "cpu"
    # App runtime deps + pyinstaller from PyPI (no llama-cpp-python here: pip may
    # have installed a source-built CPU wheel already, and a later same-version
    # install is treated as satisfied — silently keeping the wrong backend).
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "pyinstaller", t["onnx"],
                    "numpy", "scipy", "gguf", "pyyaml", "imageio-ffmpeg"], check=True)
    # Force the correct llama-cpp-python backend wheel (exclusive index + force
    # reinstall + no-deps, so a pre-existing CPU/source build is replaced).
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                    "llama-cpp-python",
                    "--index-url", "https://abetlen.github.io/llama-cpp-python/whl/" + backend],
                   check=True)

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
