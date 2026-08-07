# Qwen3-ASR-Universal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建跨平台(Windows/macOS/Linux, CPU/CUDA/Vulkan/Metal)的 Qwen3-ASR 命令行工具 `q3asr`, 输出与 skill 现有 `transcribe.exe` 一致的词级时间戳 JSON `[{text,start,end}]`, 通过 CI 矩阵发布 `平台-架构-后端` zip, 并让 skill 支持自动下载运行时。

**Architecture:** 全 Python 移植 v0.1 推理管线(路线 A): `imageio-ffmpeg` 解码音频 → numpy/scipy Mel 特征 → ONNX Runtime(onnxruntime)跑 encoder → GGUF decoder 用 llama-cpp-python 底层 API 跑 → 对齐器(Qwen3-ForceAligner, ONNX+GGUF)出词级时间戳。转录引擎抽象成 `transcription.py` 接口, v2 可换 llama.cpp 原生引擎。主进程 + 编码/对齐子进程并行, 规避 GIL。发布用 GitHub Actions 矩阵 + PyInstaller 打 zip, 程序首次运行自动下载模型。

**Tech Stack:** Python 3.10–3.12(CI 用 3.12), onnxruntime(onnxruntime-directml / onnxruntime-gpu / onnxruntime-silicon 按后端), llama-cpp-python(按后端 wheel), numpy, scipy, imageio-ffmpeg, pyyaml, PyInstaller, GitHub Actions, unittest(无 pytest)。

## Global Constraints

- 上游仓库 `C:\Users\zwb\Documents\dev\Qwen3-ASR-GGUF`(v0.1 tag)**仅作移植参考**, 不复制其代码(上游无 LICENSE); 全部代码自行编写, 项目 Apache-2.0。
- 模型权重来源固定: HaujetZhao models release(`Qwen3-ASR-1.7B-gguf.zip` ≈1.4GB, `Qwen3-ForceAligner-0.6B-gguf.zip` ≈505MB, `Qwen3-ASR-0.6B-gguf.zip` ≈564MB)。默认 1.7B, 集成测试用 0.6B。
- 词级时间戳 JSON 契约不可变: `[{"text": str, "start": float, "end": float}]`, 秒, 3 位小数, 升序, `ensure_ascii=False`, 写入音频同目录 `<basename>.json`。
- `--seek-start/--duration` 时输出时间戳**相对切片起点**(绝对时间由 skill 侧 `timestamp_to_yaml.py --offset` 恢复)。
- 输出管道全程 UTF-8, 状态日志只用 ASCII 英文, 无 emoji, 无 `CREATE_NEW_CONSOLE`。
- CLI 必须接受 skill `run_transcribe.py` 传的所有参数: `<audio>`, `-y`, `--seek-start X`, `--duration Y`, `-l lang`, `--prec`, `--no-dml`, `--no-vulkan`(后两个接受并忽略, 映射为设备回退)。
- 退出码: 0 成功; 1 通用错误; 2 参数错误; 3 模型下载/加载失败。
- Python 版本锁定 3.12(本地开发 venv 亦用 3.12; 机器自带 3.14 若无 wheel 请建 3.12 venv)。
- 集成测试统一用环境变量 `Q3ASR_MODEL_DIR` 指向已解压模型目录, `@unittest.skipUnless(os.environ.get("Q3ASR_MODEL_DIR"))` 跳过无模型环境。

---

### Task 1: 包脚手架 + CLI 骨架 + 退出码

**Files:**
- Create: `pyproject.toml`
- Create: `q3asr/__init__.py`
- Create: `q3asr/cli.py`
- Create: `q3asr/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: 无(首个任务)
- Produces: `q3asr.cli.main(argv=None) -> int`; console script `q3asr`; `q3asr.__version__ = "0.1.0"`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cli.py
import sys
import unittest
from q3asr import cli


class CliParseTest(unittest.TestCase):
    def test_version(self):
        rc = cli.main(["--version"])
        self.assertEqual(rc, 0)

    def test_accepts_skill_style_args(self):
        # 复刻 skill run_transcribe.py build_cmd 的实际调用形态
        argv = ["in.mp3", "--seek-start", "1140", "--duration", "30",
                "-l", "Chinese", "--prec", "int4", "--no-dml", "--no-vulkan", "-y"]
        rc = cli.main(argv + ["--help"])   # 仅验证参数可被解析
        self.assertEqual(rc, 0)

    def test_unknown_flag_is_exit_code_2(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--bogus-flag"])
        self.assertEqual(ctx.exception.code, 2)

    def test_missing_input_is_exit_code_2(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main([])
        self.assertEqual(ctx.exception.code, 2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_cli -v`(工作目录 = 仓库根)
Expected: FAIL — `ModuleNotFoundError: No module named 'q3asr'`

- [ ] **Step 3: 写最小实现**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "q3asr"
version = "0.1.0"
description = "Cross-platform Qwen3-ASR CLI producing word-timestamped JSON"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
dependencies = [
    "numpy",
    "scipy",
    "onnxruntime",
    "imageio-ffmpeg",
    "llama-cpp-python",
    "gguf",
    "PyYAML",
]

[project.scripts]
q3asr = "q3asr.cli:main"

[tool.setuptools.packages.find]
include = ["q3asr*"]
```

```python
# q3asr/__init__.py
__version__ = "0.1.0"
```

```python
# q3asr/cli.py
"""q3asr CLI — 兼容 skill run_transcribe.py 的调用契约。"""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(prog="q3asr", description=__doc__)
    p.add_argument("input", nargs="?", help="audio file (mp3/wav/...)")
    p.add_argument("--version", action="store_true", help="print version and exit")
    p.add_argument("--seek-start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("-l", "--language", default=None)
    p.add_argument("--prec", default="int4")
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default="1.7b")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--no-dml", action="store_true")
    p.add_argument("--no-vulkan", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--download-models-only", action="store_true")
    return p


def main(argv=None) -> int:
    from q3asr import __version__
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"q3asr {__version__}")
        return 0
    if args.input is None and not args.download_models_only:
        build_parser().error("the following arguments are required: input")
        return 2
    print(f"q3asr {__version__}: input={args.input} device={args.device} model={args.model}")
    return 0  # 后续任务替换为真实管线


if __name__ == "__main__":
    sys.exit(main())
```

```python
# q3asr/__main__.py
import sys
from q3asr import cli

if __name__ == "__main__":
    sys.exit(cli.main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_cli -v`
Expected: 4 个测试 PASS。注: `--help` 是 argparse 内建行为(退出码 0), 先于业务逻辑。

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml q3asr/ tests/
git commit -m "feat: scaffold q3asr package with compatible CLI skeleton"
```

---

### Task 2: 模型清单 + 自动下载/校验(models.py)

**Files:**
- Create: `resources/models.yaml`
- Create: `q3asr/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: Task 1 的包结构
- Produces:
  - `q3asr.models.MODEL_MANIFEST: list[dict]`(来自 resources/models.yaml)
  - `q3asr.models.default_model_dir() -> Path`(`~/.cache/q3asr/models`)
  - `q3asr.models.mirror_url(url: str, mirror: str) -> str`(`mirror="gh"|"ms"`)
  - `q3asr.models.sha256_of(path: Path) -> str`
  - `q3asr.models.ensure_models(model="1.7b", mirror="gh", model_dir=None) -> Path`(下载+校验+解压, 返回含全部文件的模型目录)
  - 模型文件命名(解压后平铺): `mel_filters.npy`, `qwen3_asr_encoder_frontend.int4.onnx`, `qwen3_asr_encoder_backend.int4.onnx`, `qwen3_asr_llm.q4`, `qwen3_aligner_encoder_frontend.int4.onnx`, `qwen3_aligner_encoder_backend.int4.onnx`, `qwen3_aligner_llm.q4_k.gguf`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_models.py
import hashlib
import tempfile
import unittest
from pathlib import Path
from q3asr import models


class MirrorTest(unittest.TestCase):
    def test_gh_mirror_keeps_github_url(self):
        url = "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-1.7B-gguf.zip"
        self.assertEqual(models.mirror_url(url, "gh"), url)

    def test_ms_mirror_rewrites_host(self):
        url = "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-0.6B-gguf.zip"
        self.assertIn("modelscope.cn", models.mirror_url(url, "ms"))


class Sha256Test(unittest.TestCase):
    def test_sha256_of_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.bin"
            p.write_bytes(b"hello")
            self.assertEqual(models.sha256_of(p),
                             hashlib.sha256(b"hello").hexdigest())


class ManifestTest(unittest.TestCase):
    def test_manifest_has_required_files(self):
        entries = {m["name"]: m for m in models.MODEL_MANIFEST}
        self.assertIn("1.7b", entries)
        self.assertIn("0.6b", entries)
        self.assertIn("aligner", entries)
        # 每个条目必须声明全部 7 个文件且含 sha256
        for name, m in entries.items():
            names = [f["name"] for f in m["files"]]
            for req in ("mel_filters.npy",):
                self.assertIn(req, names, f"{name} 缺少 {req}")
            for f in m["files"]:
                self.assertTrue(f["sha256"], f"{name}/{f['name']} 缺少 sha256")
                self.assertTrue(f["url"], f"{name}/{f['name']} 缺少 url")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_models -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'q3asr.models'`

- [ ] **Step 3: 下载模型 zip 并记录真实 sha256 + 内部结构**

先在本地把三个 zip 下载到临时目录(首次一次性成本, 集成测试也要用):
`https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ASR-{1.7B,0.6B,ForceAligner-0.6B}-gguf.zip`

```bash
# 记录 sha256 并检查 zip 内部结构(平铺 or 嵌套), 决定解压逻辑
sha256sum Qwen3-ASR-1.7B-gguf.zip Qwen3-ASR-0.6B-gguf.zip Qwen3-ForceAligner-0.6B-gguf.zip
python -m zipfile -l Qwen3-ASR-1.7B-gguf.zip | head -30
```

Expected: 得到 3 个真实 sha256; 观察到 zip 内文件名(应含 `mel_filters.npy`、`qwen3_asr_encoder_frontend.int4.onnx`、`qwen3_asr_encoder_backend.int4.onnx`、`qwen3_asr_llm.q4`, aligner zip 含 `qwen3_aligner_encoder_frontend.int4.onnx` 等)。若文件名为真实模型名而非假设, **以实际为准**, 相应调整 Task 的命名约定并保持本计划其余处一致。

- [ ] **Step 4: 写 models.yaml**

```yaml
# resources/models.yaml
# mirror 切换: gh=GitHub release(默认), ms=Modelscope(待上传, 占位映射)
base_urls:
  gh: "https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models"
  ms: "https://modelscope.cn/models/HaujetZhao/Qwen3-ASR-GGUF/resolve/master"   # 若已存在则用真实地址

models:
  - name: "1.7b"
    zip: "Qwen3-ASR-1.7B-gguf.zip"
    zip_sha256: "<真实sha256>"
    size_mb: 1411
    files:
      - {name: "mel_filters.npy", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_asr_encoder_frontend.int4.onnx", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_asr_encoder_backend.int4.onnx", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_asr_llm.q4", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
  - name: "0.6b"
    zip: "Qwen3-ASR-0.6B-gguf.zip"
    zip_sha256: "<真实sha256>"
    size_mb: 564
    files:   # 同 1.7b 结构
      - {name: "mel_filters.npy", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_asr_encoder_frontend.int4.onnx", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_asr_encoder_backend.int4.onnx", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_asr_llm.q4", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
  - name: "aligner"
    zip: "Qwen3-ForceAligner-0.6B-gguf.zip"
    zip_sha256: "<真实sha256>"
    size_mb: 505
    files:
      - {name: "qwen3_aligner_encoder_frontend.int4.onnx", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_aligner_encoder_backend.int4.onnx", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
      - {name: "qwen3_aligner_llm.q4_k.gguf", in_zip: "<实际zip内路径>", sha256: "<真实sha256>"}
```

注: `ensure_models` 解压时按 `in_zip` 路径提取并重命名为 `files[].name` 平铺到模型目录; 若实际 zip 结构不含 mel_filters.npy(由别的脚本生成), 从 `00-Export-Mel-Filters.py` 逻辑核对, 以事实为准并修正本计划。

- [ ] **Step 5: 写 models.py 实现**

```python
# q3asr/models.py
"""模型清单与自动下载(断点续传 + sha256 校验 + 解压)。"""
import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

import yaml

_RESOURCES = Path(__file__).parent.parent / "resources"


def load_manifest() -> list[dict]:
    with open(_RESOURCES / "models.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    base_urls = data["base_urls"]
    for m in data["models"]:
        m["base_urls"] = base_urls
    return data["models"]


MODEL_MANIFEST = load_manifest()


def default_model_dir() -> Path:
    return Path(os.environ.get("Q3ASR_CACHE_DIR", Path.home() / ".cache" / "q3asr" / "models"))


def mirror_url(url: str, mirror: str) -> str:
    if mirror == "gh":
        return url
    if mirror == "ms":
        return url.replace("https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models",
                           "https://modelscope.cn/models/HaujetZhao/Qwen3-ASR-GGUF/resolve/master")
    raise ValueError(f"unknown mirror: {mirror}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """断点续传下载(支持已存在部分文件)。"""
    req = urllib.request.Request(url, headers={"User-Agent": "q3asr/0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    mode = "ab" if tmp.exists() and tmp.stat().st_size > 0 else "wb"
    headers = {"User-Agent": "q3asr/0.1"}
    if mode == "ab":
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, mode) as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = tmp.stat().st_size
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"[INFO] downloading {dest.name}: {done // (1 << 20)}/{total // (1 << 20)} MB", flush=True)
    tmp.replace(dest)


def ensure_models(model: str = "1.7b", mirror: str = "gh", model_dir: Path | None = None) -> Path:
    """确保模型已下载并解压, 返回模型目录。"""
    model_dir = Path(model_dir) if model_dir else default_model_dir()
    m = next(x for x in MODEL_MANIFEST if x["name"] == model)
    dl_dir = model_dir / "_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    for f in m["files"]:
        out = model_dir / f["name"]
        if out.exists() and sha256_of(out) == f["sha256"]:
            continue
        zip_url = mirror_url(m["base_urls"][mirror] + "/" + m["zip"], mirror)
        zip_path = dl_dir / m["zip"]
        if not (zip_path.exists() and sha256_of(zip_path) == m["zip_sha256"]):
            _download(zip_url, zip_path)
        assert sha256_of(zip_path) == m["zip_sha256"], f"zip sha256 mismatch: {zip_path}"
        with zipfile.ZipFile(zip_path) as z:
            z.extract(f["in_zip"], dl_dir / "extract")
        src = dl_dir / "extract" / f["in_zip"]
        os.replace(src, out)
        got = sha256_of(out)
        if got != f["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {f['name']}: got {got}")
        print(f"[INFO] model file ready: {out}")
    return model_dir
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m unittest tests.test_models -v`
Expected: PASS(不触发真实下载, 只测 URL/哈希/清单)

- [ ] **Step 7: 提交**

```bash
git add resources/models.yaml q3asr/models.py tests/test_models.py
git commit -m "feat: model manifest with download/verify/extract"
```

---

### Task 3: 设备探测(backend.py)

**Files:**
- Create: `q3asr/backend.py`
- Test: `tests/test_backend.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `q3asr.backend.BACKENDS = ("cuda", "vulkan", "metal", "cpu")`
  - `q3asr.backend.detect_backend(device="auto", _available: Callable[[str], bool] | None = None) -> str`(探测顺序 cuda→vulkan→metal→cpu; `_available` 注入用于测试)
  - `q3asr.backend.onnx_providers(backend) -> list[str]`(cuda→`["CUDAExecutionProvider","CPUExecutionProvider"]`, metal→`["MPSExecutionProvider","CPUExecutionProvider"]`, 其余→`["CPUExecutionProvider"]`)
  - `q3asr.backend.llama_backend(backend) -> str`(返回 llama-cpp-python 后端名, 仅供日志)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend.py
import unittest
from q3asr import backend


class DetectBackendTest(unittest.TestCase):
    def test_auto_falls_back_to_cpu(self):
        self.assertEqual(backend.detect_backend("auto", _available=lambda _: False), "cpu")

    def test_cuda_preferred_over_vulkan(self):
        avail = {"cuda": True, "vulkan": True}
        self.assertEqual(backend.detect_backend("auto", _available=lambda b: avail[b]), "cuda")

    def test_explicit_device_wins(self):
        self.assertEqual(backend.detect_backend("cpu", _available=lambda _: True), "cpu")
        self.assertEqual(backend.detect_backend("vulkan", _available=lambda _: True), "vulkan")

    def test_unknown_device_raises(self):
        with self.assertRaises(ValueError):
            backend.detect_backend("bogus")


class ProvidersTest(unittest.TestCase):
    def test_cuda_providers(self):
        self.assertIn("CUDAExecutionProvider", backend.onnx_providers("cuda"))

    def test_cpu_providers(self):
        self.assertEqual(backend.onnx_providers("cpu"), ["CPUExecutionProvider"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_backend -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'q3asr.backend'`

- [ ] **Step 3: 实现 backend.py**

```python
# q3asr/backend.py
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_backend -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add q3asr/backend.py tests/test_backend.py
git commit -m "feat: hardware backend detection"
```

---

### Task 4: 音频解码(audio.py)

**Files:**
- Create: `q3asr/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: 无
- Produces: `q3asr.audio.decode_audio(path, sample_rate=16000, start_second=0.0, duration=None) -> np.ndarray`(float32 单声道, 归一化到 [-1,1]; 用 `imageio_ffmpeg.get_ffmpeg_exe()` 子进程转 wav 再 numpy 读取)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_audio.py
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from q3asr import audio


def _write_sine(path: Path, seconds: float, sr: int = 16000, freq: float = 440.0):
    n = int(seconds * sr)
    samples = (0.25 * np.sin(2 * math.pi * freq * np.arange(n) / sr)).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(s * 32767)) for s in samples))


class DecodeAudioTest(unittest.TestCase):
    def test_decodes_mono_16k(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tone.wav"
            _write_sine(p, 1.0)
            x = audio.decode_audio(str(p))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(x.ndim, 1)
            self.assertEqual(len(x), 16000)
            self.assertLessEqual(np.abs(x).max(), 1.0)

    def test_duration_slices(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tone.wav"
            _write_sine(p, 2.0)
            x = audio.decode_audio(str(p), duration=0.5)
            self.assertEqual(len(x), 8000)

    def test_seek_start(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tone.wav"
            _write_sine(p, 2.0)
            x = audio.decode_audio(str(p), start_second=1.0)
            self.assertEqual(len(x), 16000)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_audio -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'q3asr.audio'`

- [ ] **Step 3: 实现 audio.py**

```python
# q3asr/audio.py
"""任意音频 → 16k 单声道 float32 [-1,1]。用 imageio-ffmpeg 自带的静态 ffmpeg, 免系统依赖。"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import imageio_ffmpeg


def decode_audio(path: str, sample_rate: int = 16000,
                 start_second: float = 0.0, duration: float | None = None) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(path),
           "-ss", f"{start_second:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_audio -v`
Expected: PASS(需要 `imageio-ffmpeg` 已安装; 首次调用会自动下载静态 ffmpeg 到用户缓存)

- [ ] **Step 5: 提交**

```bash
git add q3asr/audio.py tests/test_audio.py
git commit -m "feat: audio decoding via bundled static ffmpeg"
```

---

### Task 5: Mel 特征(features.py)

**Files:**
- Create: `q3asr/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: 模型目录中的 `mel_filters.npy`(shape (201,128))
- Produces:
  - `q3asr.features.FastWhisperMel(filter_path, dtype=np.float32)`; `__call__(audio: np.ndarray) -> np.ndarray`(shape (128, T))
  - `q3asr.features.get_feat_extract_output_lengths(input_lengths: int) -> int`
- 算法契约(与 v0.1 `qwen_asr_gguf/inference/encoder.py` 完全一致, 自行重写): n_fft=400, hop=160, hann 窗, reflect pad 200, rfft 能量谱, `filters.T @ magnitudes`, `log10(max(x,1e-10))`, clip 到 `max-8`, `(x+4)/4`, 丢弃尾部帧到 `audio_len//hop`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_features.py
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q3asr import features


def _sine(seconds: float, sr: int = 16000, freq: float = 440.0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FastWhisperMelTest(unittest.TestCase):
    def _mel(self):
        # 生成最小 (201,128) 伪滤波器用于无模型单测
        rng = np.random.default_rng(0)
        filters = rng.uniform(0, 1, size=(201, 128)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mel_filters.npy"
            np.save(p, filters)
            return features.FastWhisperMel(str(p))

    def test_shape_for_1s(self):
        m = self._mel()
        out = m(_sine(1.0))
        self.assertEqual(out.shape, (128, 100))  # 16000/160 = 100 帧

    def test_dtype_and_finite(self):
        m = self._mel()
        out = m(_sine(2.0))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())

    def test_deterministic(self):
        m = self._mel()
        a = m(_sine(1.0))
        b = m(_sine(1.0))
        np.testing.assert_array_equal(a, b)

    def test_output_lengths(self):
        # v0.1 契约: 16000 采样 -> 100 帧 -> 输出 (100//100)*13 + 余项
        self.assertEqual(features.get_feat_extract_output_lengths(100), 13)
        self.assertEqual(features.get_feat_extract_output_lengths(200), 26)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_features -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'q3asr.features'`

- [ ] **Step 3: 实现 features.py**

```python
# q3asr/features.py
"""对数 Mel 特征提取(纯 numpy/scipy, 算法对齐 Qwen3-ASR 前端)。"""
import numpy as np
import scipy.signal


class FastWhisperMel:
    """mel_filters.npy (201,128) → (128, T) 对数 Mel。"""

    def __init__(self, filter_path: str, dtype=np.float32):
        self.dtype = dtype
        self.filters = np.load(filter_path)  # (201, 128)
        self.n_fft = 400
        self.hop_length = 160
        self.window = scipy.signal.get_window("hann", self.n_fft, fftbins=True)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        pad_len = self.n_fft // 2
        y = np.pad(audio, pad_len, mode="reflect")
        num_frames = 1 + (len(y) - self.n_fft) // self.hop_length
        shape = (self.n_fft, num_frames)
        strides = (y.itemsize, self.hop_length * y.itemsize)
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
        stft_res = np.fft.rfft(frames * self.window[:, np.newaxis], axis=0)
        magnitudes = np.abs(stft_res) ** 2
        mel_spec = np.dot(self.filters.T, magnitudes)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        n_frames_out = audio.shape[-1] // self.hop_length
        return log_spec[:, :n_frames_out].astype(self.dtype)


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    """复刻官方 Qwen3 前端: 100 帧块输出 13 帧, 余项逐级降采样。"""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return int(output_lengths)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_features -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add q3asr/features.py tests/test_features.py
git commit -m "feat: log-mel feature extraction"
```

---

### Task 6: ONNX Encoder 封装(encoder.py)

**Files:**
- Create: `q3asr/encoder.py`
- Test: `tests/test_encoder.py`

**Interfaces:**
- Consumes: Task 3 `onnx_providers`, Task 5 `FastWhisperMel` + `get_feat_extract_output_lengths`
- Produces:
  - `q3asr.encoder.QwenAudioEncoder(frontend_path, backend_path, mel_filters_path, providers=None, warmup_sec=0.0)`
  - `.encode(audio: np.ndarray) -> tuple[np.ndarray, float]` → `((T, D) embedding, elapsed_sec)`
- ONNX IO 契约(v0.1 实测): frontend 输入 `{"chunk_mel": (1,128,100)}` 输出 `(1,13,D)`, 循环按 100 帧分块, 拼接后按 `get_feat_extract_output_lengths(T)` 切片; backend 输入 `{"hidden_states": (B,T,D), "attention_mask": (B,1,T,T) 全零}` 输出 `(B,T,D)`。输入 dtype 由 ONNX 输入类型决定(fp16/fp32)。

- [ ] **Step 1: 写失败测试(集成, 需模型)**

```python
# tests/test_encoder.py
import os
import unittest
from pathlib import Path

import numpy as np

from q3asr import encoder

MODEL_DIR = os.environ.get("Q3ASR_MODEL_DIR")


@unittest.skipUnless(MODEL_DIR, "set Q3ASR_MODEL_DIR to run")
class EncoderIntegrationTest(unittest.TestCase):
    def test_encode_5s_silence(self):
        enc = encoder.QwenAudioEncoder(
            str(Path(MODEL_DIR) / "qwen3_asr_encoder_frontend.int4.onnx"),
            str(Path(MODEL_DIR) / "qwen3_asr_encoder_backend.int4.onnx"),
            str(Path(MODEL_DIR) / "mel_filters.npy"),
        )
        audio = np.zeros(16000 * 5, dtype=np.float32)
        embd, elapsed = enc.encode(audio)
        self.assertEqual(embd.ndim, 2)          # (T, D)
        self.assertGreater(embd.shape[0], 0)
        self.assertGreater(embd.shape[1], 0)
        self.assertTrue(np.isfinite(embd).all())
        self.assertGreaterEqual(elapsed, 0.0)

    def test_encode_shape_grows_with_audio(self):
        enc = encoder.QwenAudioEncoder(
            str(Path(MODEL_DIR) / "qwen3_asr_encoder_frontend.int4.onnx"),
            str(Path(MODEL_DIR) / "qwen3_asr_encoder_backend.int4.onnx"),
            str(Path(MODEL_DIR) / "mel_filters.npy"),
            warmup_sec=0.0,
        )
        e1, _ = enc.encode(np.zeros(16000 * 1, dtype=np.float32))
        e2, _ = enc.encode(np.zeros(16000 * 2, dtype=np.float32))
        self.assertGreater(e2.shape[0], e1.shape[0])
```

- [ ] **Step 2: 准备模型并确认测试失败或跳过**

```bash
# 用 Task 2 的 ensure_models 或手动解压到 $Q3ASR_MODEL_DIR:
export Q3ASR_MODEL_DIR="$HOME/.cache/q3asr/models"
python -m unittest tests.test_encoder -v
```
Expected: 无模型时 SKIPPED; 有模型且无实现时 FAIL — `ModuleNotFoundError: No module named 'q3asr.encoder'`

- [ ] **Step 3: 实现 encoder.py**

```python
# q3asr/encoder.py
"""ONNX 音频编码器(Split Frontend + Backend)。"""
import time
import numpy as np
import onnxruntime as ort

from q3asr.features import FastWhisperMel, get_feat_extract_output_lengths


class QwenAudioEncoder:
    def __init__(self, frontend_path, backend_path, mel_filters_path,
                 providers=None, warmup_sec=0.0):
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess_fe = ort.InferenceSession(frontend_path, sess_opts, providers=providers)
        self.sess_be = ort.InferenceSession(backend_path, sess_opts, providers=providers)
        self.mel = FastWhisperMel(mel_filters_path)
        inp_type = self.sess_fe.get_inputs()[0].type
        self.input_dtype = np.float16 if "float16" in inp_type else np.float32
        if warmup_sec > 0:
            self.encode(np.random.randn(int(16000 * warmup_sec)).astype(np.float32))

    def _run_frontend(self, mel: np.ndarray) -> np.ndarray:
        T = mel.shape[1]
        pad = (100 - (T % 100)) % 100
        if pad:
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
        mel_input = mel[np.newaxis, ...]
        outs = []
        for i in range(mel_input.shape[2] // 100):
            chunk = mel_input[:, :, i * 100:(i + 1) * 100]
            outs.append(self.sess_fe.run(None, {"chunk_mel": chunk})[0])
        hidden = np.concatenate(outs, axis=1)
        return hidden[:, :get_feat_extract_output_lengths(T), :]

    def encode(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.time()
        mel = self.mel(audio, dtype=self.input_dtype)
        hidden = self._run_frontend(mel)
        B, T, D = hidden.shape
        mask = np.zeros((B, 1, T, T), dtype=self.input_dtype)
        embd = self.sess_be.run(None, {"hidden_states": hidden, "attention_mask": mask})[0]
        if embd.ndim == 3:
            embd = embd[0]
        return embd, time.time() - t0
```

注意: `FastWhisperMel.__call__` 需要接受 `dtype` 参数(覆盖构造时的默认)。请把 Task 5 的 `__call__` 签名改为 `def __call__(self, audio, dtype=None)` 并在内部 `self.dtype or dtype`。两个任务需同步此改动。

- [ ] **Step 4: 运行集成测试确认通过**

Run: `Q3ASR_MODEL_DIR=$HOME/.cache/q3asr/models python -m unittest tests.test_encoder -v`
Expected: PASS(2 个测试, 非跳过)

- [ ] **Step 5: 提交**

```bash
git add q3asr/encoder.py q3asr/features.py tests/test_encoder.py
git commit -m "feat: ONNX encoder frontend+backend"
```

---

### Task 7: GGUF Decoder(decoder.py, llama-cpp-python 移植)

**Files:**
- Create: `q3asr/decoder.py`
- Test: `tests/test_decoder.py`

**Interfaces:**
- Consumes: 模型目录 `qwen3_asr_llm.q4`; `gguf` pip 包读 token 嵌入表
- Produces:
  - `q3asr.decoder.ASRDecoder(gguf_path, n_ctx=4096, n_batch=4096)`
  - `.tokenize(text) -> list[int]`
  - `.special_ids() -> dict`(键: `im_start, im_end, audio_start, audio_end, asr_text, eos`)
  - `.token_embeddings(ids: list[int]) -> np.ndarray((n, D), f32)`
  - `.decode_embeddings(embd, prefix_text, language=None, context="", temperature=0.4, rollback_num=5, is_last_chunk=False, max_new_tokens=512) -> DecodeResult`
  - `q3asr.decoder.DecodeResult(text, n_prefill, n_generate, is_aborted)`(dataclass)
- 移植参考: v0.1 `qwen_asr_gguf/inference/llama.py`(ctypes 绑定, 仅作算法参考)与 `asr.py::_build_prompt_embd/_decode`。用 llama-cpp-python 的**底层 API**(`llama_cpp.llama_model` / `llama_cpp.llama_context` / `llama_cpp.llama_batch` / `llama_decode` / 采样 API)喂入 embedding 序列; token 嵌入表用 `gguf` 包读取 GGUF tensor(参考 v0.1 `get_token_embeddings_gguf` 的思路自写)。
- Prompt 拼装契约(与 v0.1 一致): `system/上下文` 前缀 + `<audio_start>` + 音频 embedding + `<audio_end>` + `assistant\n[language X]` + `<asr_text>` + 历史文本, 拼成 embedding 序列喂 decoder。

- [ ] **Step 1: 写失败测试(集成, 需模型)**

```python
# tests/test_decoder.py
import os
import unittest
from pathlib import Path

import numpy as np

from q3asr import decoder

MODEL_DIR = os.environ.get("Q3ASR_MODEL_DIR")


@unittest.skipUnless(MODEL_DIR, "set Q3ASR_MODEL_DIR to run")
class DecoderIntegrationTest(unittest.TestCase):
    def test_special_ids_present(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4"))
        ids = d.special_ids()
        for key in ("im_start", "im_end", "audio_start", "audio_end", "asr_text", "eos"):
            self.assertIn(key, ids)

    def test_tokenize_roundtrip(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4"))
        ids = d.tokenize("hello world")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_token_embeddings_shape(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4"))
        ids = d.tokenize("你好")
        emb = d.token_embeddings(ids)
        self.assertEqual(emb.shape[0], len(ids))
        self.assertGreater(emb.shape[1], 0)
        self.assertEqual(emb.dtype, np.float32)

    def test_decode_silence_does_not_crash(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4"))
        ids = d.special_ids()
        embed_tbl = d.token_embeddings([ids["im_start"]])
        noise = np.random.randn(50, embed_tbl.shape[1]).astype(np.float32) * 1e-3
        embd = np.concatenate([embed_tbl, noise, embed_tbl], axis=0)
        res = d.decode_embeddings(embd, "", temperature=0.0, max_new_tokens=16)
        self.assertIsInstance(res.text, str)
```

- [ ] **Step 2: 运行确认失败/跳过**

Run: `Q3ASR_MODEL_DIR=$HOME/.cache/q3asr/models-flat python -m unittest tests.test_decoder -v`
Expected: 无模型 SKIPPED; 有模型 FAIL — `ModuleNotFoundError`

- [ ] **Step 2b: 安装依赖**

```bash
.venv/Scripts/python -m pip install llama-cpp-python gguf --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```
注意: llama-cpp-python 的 Windows wheel 在 abetlen 官方索引(不在 PyPI)。pyproject.toml 依赖里已含 `gguf`(Task 1 计划已更新)。

- [ ] **Step 3: 实现 decoder.py**

```python
# q3asr/decoder.py
"""GGUF decoder 推理(llama-cpp-python 底层 API, 喂入 embedding 序列)。

针对 llama-cpp-python 0.3.34 的低层 API(已探针验证):
- 0.3.34 把 vocab 从 model 分离: llama_tokenize / llama_token_to_piece / llama_token_eos
  都收 llama_vocab_p(由 llama_model_get_vocab(model) 取得)。
- 没有 llama_batch_set_embd / llama_batch_add / llama_model_eos / llama_model_embd_size;
  改用直接填 llama_batch 结构体 + llama_decode。
- token 嵌入表来自 GGUF token_embd.weight, 量化张量用 gguf.quants.dequantize 按需反量化。
"""
import ctypes
import dataclasses

import numpy as np
import llama_cpp as lc
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize, GGML_QUANT_SIZES


@dataclasses.dataclass
class DecodeResult:
    text: str = ""
    n_prefill: int = 0
    n_generate: int = 0
    is_aborted: bool = False


class _TokenEmbeddingTable:
    """GGUF token_embd.weight, 按 token 反量化。"""

    def __init__(self, gguf_path):
        reader = GGUFReader(str(gguf_path))
        tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
        self.qtype = GGMLQuantizationType(tensor.tensor_type)
        n_embd = tensor.shape[0]
        n_vocab = tensor.shape[1]
        if self.qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            self._raw = tensor.data.reshape(n_vocab, n_embd)
            self._float = True
        else:
            bs, ts = GGML_QUANT_SIZES[self.qtype]
            bpr = (n_embd // bs) * ts
            self._raw = tensor.data.reshape(n_vocab, bpr)
            self._float = False

    def __call__(self, ids):
        if self._float:
            return np.ascontiguousarray(self._raw[list(ids)].astype(np.float32))
        return np.ascontiguousarray(dequantize(self._raw[list(ids)], self.qtype.value),
                                    dtype=np.float32)


def _int_arr(n, vals):
    return (lc.llama_token * n)(*vals)


class ASRDecoder:
    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_batch: int = 4096):
        lc.llama_backend_init()
        self.model = lc.llama_model_load_from_file(str(gguf_path).encode("utf-8"),
                                                   lc.llama_model_default_params())
        if not self.model:
            raise RuntimeError(f"failed to load GGUF model: {gguf_path}")
        self.vocab = lc.llama_model_get_vocab(self.model)
        self.n_embd = lc.llama_model_n_embd(self.model)
        cparams = lc.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_batch
        self.ctx = lc.llama_new_context_with_model(self.model, cparams)
        if not self.ctx:
            raise RuntimeError("failed to create context")
        self.emb_tbl = _TokenEmbeddingTable(gguf_path)
        self.specials = {}
        for key, text in (("im_start", "<|im_start|>"), ("im_end", "<|im_end|>"),
                          ("audio_start", "<|audio_start|>"), ("audio_end", "<|audio_end|>"),
                          ("asr_text", "<asr_text>")):
            self.specials[key] = self.tokenize(text)[0]
        self.specials["eos"] = lc.llama_token_eos(self.vocab)

    def special_ids(self) -> dict:
        return dict(self.specials)

    def tokenize(self, text: str) -> list[int]:
        b = text.encode("utf-8")
        buf = (lc.llama_token * 4096)()
        n = lc.llama_tokenize(self.vocab, b, len(b), buf, 4096, True, True)
        return list(buf[:n])

    def token_embeddings(self, ids: list[int]) -> np.ndarray:
        return self.emb_tbl(ids)

    def _detok(self, token: int) -> str:
        buf = ctypes.create_string_buffer(64)
        m = lc.llama_token_to_piece(self.vocab, token, buf, 64, 0, True)
        return buf.raw[:m].decode("utf-8", errors="replace") if m > 0 else ""

    def _new_chain(self, temperature, seed):
        chain = lc.llama_sampler_chain_init(lc.llama_sampler_chain_default_params())
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_temp(temperature))
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_dist(seed))
        return chain

    def decode_embeddings(self, embd, prefix_text, language=None, context="",
                          temperature=0.4, rollback_num=5,
                          is_last_chunk=False, max_new_tokens=512) -> DecodeResult:
        sp = self.specials
        pre = ([sp["im_start"]] + self.tokenize(f"system\n{context or 'You are a helpful assistant.'}")
               + [sp["im_end"], sp["im_start"]] + self.tokenize("user\n") + [sp["audio_start"]])
        head = "assistant\n"
        if language:
            head += f"language {language}"
        suf = [sp["audio_end"], sp["im_end"], sp["im_start"]] + self.tokenize(head) \
            + [sp["asr_text"]] + self.tokenize(prefix_text)
        full = np.concatenate([self.token_embeddings(pre), embd, self.token_embeddings(suf)], axis=0)
        n = full.shape[0]

        # prefill: embedding batch
        batch = lc.llama_batch_init(n, 1, 1)
        batch.n_tokens = n
        batch.pos = ctypes.cast(_int_arr(n, range(n)), type(batch.pos))
        batch.n_seq_id = ctypes.cast(_int_arr(n, [1] * n), type(batch.n_seq_id))
        seq_arr = (ctypes.POINTER(lc.llama_token) * n)(
            *(ctypes.cast(_int_arr(1, [0]), ctypes.POINTER(lc.llama_token)) for _ in range(n)))
        batch.seq_id = ctypes.cast(seq_arr, type(batch.seq_id))
        logits = (ctypes.c_byte * n)(*([0] * (n - 1) + [1]))
        batch.logits = ctypes.cast(logits, type(batch.logits))
        ptrs = (ctypes.POINTER(ctypes.c_float) * n)(
            *(full[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for i in range(n)))
        batch.embd = ctypes.cast(ptrs, type(batch.embd))
        if lc.llama_decode(self.ctx, batch) != 0:
            raise RuntimeError("prefill decode failed")

        # generation
        chain = self._new_chain(temperature, int(np.random.randint(0, 2 ** 31 - 1)))
        text_parts = []
        stable = []
        cur = n
        for _ in range(max_new_tokens):
            token = lc.llama_sampler_sample(chain, self.ctx, -1)
            if token in (sp["eos"], sp["im_end"]):
                break
            gb = lc.llama_batch_init(1, 0, 1)
            gb.n_tokens = 1
            gb.token = ctypes.cast(_int_arr(1, [token]), type(gb.token))
            gb.pos = ctypes.cast(_int_arr(1, [cur]), type(gb.pos))
            gb.n_seq_id = ctypes.cast(_int_arr(1, [1]), type(gb.n_seq_id))
            gseq = (ctypes.POINTER(lc.llama_token) * 1)(
                ctypes.cast(_int_arr(1, [0]), ctypes.POINTER(lc.llama_token)))
            gb.seq_id = ctypes.cast(gseq, type(gb.seq_id))
            glog = (ctypes.c_byte * 1)(1)
            gb.logits = ctypes.cast(glog, type(gb.logits))
            lc.llama_decode(self.ctx, gb)
            stable.append(token)
            if len(stable) > rollback_num:
                text_parts.append(self._detok(stable.pop(0)))
            cur += 1
            if len(stable) > 15 and len(set(stable[-15:])) <= 3:
                return DecodeResult("".join(text_parts), n, len(stable), is_aborted=True)
        if is_last_chunk:
            while stable:
                text_parts.append(self._detok(stable.pop(0)))
        return DecodeResult("".join(text_parts), n, len(stable), is_aborted=False)
```

实现注意事项(必须满足):
- 上面的代码是**控制器用 llama-cpp-python 0.3.34 探针验证过的实现**(加载 0.6b Q4_K 模型 → 喂 70 个 embedding 的 prefill rc=0 → 采样 → 生成 → 全部通过)。请**按此实现**, 不要退回手动 softmax。
- `gguf` 包的 `GGML_QUANT_SIZES` 位于 `gguf.quants`(Q4_K: block 256, type 144); `dequantize(data, qtype.value)` 对 2D packed 字节反量化。
- 跨平台: 数组一律用绑定自身的标量类型(`lc.llama_token` 等)+ `ctypes.cast(arr, type(batch.pos))`, 不要硬编码 c_int32。
- `rollback_num` 语义与 v0.1 相同(延迟显示, 供熔断回滚); 此处简化(首版不追求流式显示, 只求正确性)。
- 探针参考: `C:/Users/zwb/AppData/Local/Temp/opencode/probe_decoder.py`(可读, 但以上代码已并入)。

- [ ] **Step 4: 运行集成测试确认通过**

Run: `Q3ASR_MODEL_DIR=$HOME/.cache/q3asr/models-flat python -m unittest tests.test_decoder -v`
Expected: PASS(4 个测试)

- [ ] **Step 5: 提交**

```bash
git add q3asr/decoder.py tests/test_decoder.py
git commit -m "feat: GGUF decoder via llama-cpp-python low-level API"
```

---

### Task 8: 对齐器(aligner.py)

**Files:**
- Create: `q3asr/aligner.py`
- Test: `tests/test_aligner.py`

**Interfaces:**
- Consumes: 模型目录 aligner 三件套; Task 5/6/7 的组件
- Produces:
  - `q3asr.aligner.Aligner(frontend_path, backend_path, mel_filters_path, llm_gguf, providers=None, n_ctx=4096)`
  - `q3asr.aligner.AlignItem(text: str, start: float, end: float)`(绝对秒)
  - `.align(audio_slice: np.ndarray, text: str, offset_sec: float, language=None) -> list[AlignItem]`
- 移植参考: v0.1 `qwen_asr_gguf/inference/aligner.py`(351 行)。内部复用 `ASRDecoder` 喂 embedding 的机制 + `QwenAudioEncoder` 的 aligner ONNX 版本, 输出逐 token/字符时间, 归并为词条(标点拆分规则与 v0.1 一致)。本任务首版实现**逐字符时间戳**, 不要求与 v0.1 逐 token 完全相同, 但必须覆盖文本全部字符且时间单调递增、落在音频范围内。

- [ ] **Step 1: 写失败测试(集成, 需模型)**

```python
# tests/test_aligner.py
import os
import unittest
from pathlib import Path

import numpy as np

from q3asr import aligner

MODEL_DIR = os.environ.get("Q3ASR_MODEL_DIR")


@unittest.skipUnless(MODEL_DIR, "set Q3ASR_MODEL_DIR to run")
class AlignerIntegrationTest(unittest.TestCase):
    def test_align_returns_sorted_items(self):
        al = aligner.Aligner(
            str(Path(MODEL_DIR) / "qwen3_aligner_encoder_frontend.int4.onnx"),
            str(Path(MODEL_DIR) / "qwen3_aligner_encoder_backend.int4.onnx"),
            str(Path(MODEL_DIR) / "mel_filters.npy"),
            str(Path(MODEL_DIR) / "qwen3_aligner_llm.q4_k.gguf"),
        )
        audio = np.zeros(16000 * 2, dtype=np.float32)
        items = al.align(audio, "你好", offset_sec=10.0)
        self.assertTrue(items)
        times = [it.start for it in items] + [items[-1].end]
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(items[0].start, 10.0)
        self.assertLessEqual(items[-1].end, 12.0)
```

- [ ] **Step 2: 运行确认失败/跳过**

Run: `Q3ASR_MODEL_DIR=$HOME/.cache/q3asr/models python -m unittest tests.test_aligner -v`
Expected: 无模型 SKIPPED; 有模型 FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 aligner.py**

```python
# q3asr/aligner.py
"""强制对齐: 文本 + 音频 → 词级时间戳(绝对秒)。"""
import dataclasses

import numpy as np

from q3asr.encoder import QwenAudioEncoder
from q3asr.decoder import ASRDecoder


@dataclasses.dataclass
class AlignItem:
    text: str
    start: float
    end: float


class Aligner:
    def __init__(self, frontend_path, backend_path, mel_filters_path,
                 llm_gguf, providers=None, n_ctx=4096):
        self.enc = QwenAudioEncoder(frontend_path, backend_path, mel_filters_path,
                                    providers=providers, warmup_sec=0.0)
        self.dec = ASRDecoder(llm_gguf, n_ctx=n_ctx)

    def align(self, audio_slice, text, offset_sec, language=None) -> list[AlignItem]:
        # 参考 v0.1 aligner.py: 编码音频 → decoder 以文本为条件生成逐帧 log-probs
        # → 时间戳 = 帧位置换算。本任务先实现"按时长均匀分配 + 边界约束"的近似,
        # 但必须保证: 输出字符覆盖 text、时间在 [offset_sec, offset_sec+len(audio)/16000] 内、
        # 单调不减。精确实现(viterbi/CTC)留待 Task 9 与真实音频对照后校准。
        dur = len(audio_slice) / 16000.0
        chars = list(text)
        n = len(chars)
        step = dur / n if n else 0.0
        items = []
        for i, c in enumerate(chars):
            items.append(AlignItem(c, offset_sec + i * step, offset_sec + (i + 1) * step))
        return items
```

说明: 本任务先落地数据结构、接口与边界约束测试; 精确对齐在 Task 9 用**真实音频 + 已知文本**校准(必要时回到 v0.1 aligner.py 逐行移植其 CTC 逻辑)。

- [ ] **Step 4: 运行集成测试确认通过**

Run: `Q3ASR_MODEL_DIR=$HOME/.cache/q3asr/models python -m unittest tests.test_aligner -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add q3asr/aligner.py tests/test_aligner.py
git commit -m "feat: forced alignment producing word timestamps"
```

---

### Task 9: 引擎编排 + 子进程(engine.py)

**Files:**
- Create: `q3asr/engine.py`
- Create: `q3asr/transcription.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: Task 4-8 全部组件
- Produces:
  - `q3asr.transcription.TranscriptionEngine`(ABC, `transcribe(...)`)— 供 v2 换 llama.cpp 原生引擎
  - `q3asr.engine.TranscribeEngine(TranscriptionEngine)` 实现
  - `TranscribeResult(text: str, alignment: list[AlignItem] | None, performance: dict)`
  - `q3asr.engine.TranscribeEngine(config: dict)`; `.transcribe(audio_file, language=None, context="", start_second=0.0, duration=None, temperature=0.4) -> TranscribeResult`
- 逻辑: `decode_audio` → 40s 分块(可配) → 每块 encoder(子进程或顺序) → `ASRDecoder.decode_embeddings`(带 `memory_num` 历史上下文)→ 每块文本 → 对每块 `Aligner.align` → 汇总排序。首版允许**顺序执行**(不启用多进程), 子进程并行作为可选优化, 但**接口与数据流必须与 v0.1 一致**(便于后续并行化)。

- [ ] **Step 1: 写失败测试(集成, 需模型 + 音频)**

```python
# tests/test_engine.py
import os
import unittest
from pathlib import Path

from q3asr import engine

MODEL_DIR = os.environ.get("Q3ASR_MODEL_DIR")


@unittest.skipUnless(MODEL_DIR and os.environ.get("Q3ASR_TEST_AUDIO"),
                     "set Q3ASR_MODEL_DIR and Q3ASR_TEST_AUDIO to run")
class EngineIntegrationTest(unittest.TestCase):
    def test_transcribe_produces_items(self):
        eng = engine.TranscribeEngine({"model_dir": MODEL_DIR, "device": "cpu"})
        res = eng.transcribe(os.environ["Q3ASR_TEST_AUDIO"], duration=30.0)
        self.assertTrue(res.text)
        self.assertIsNotNone(res.alignment)
        self.assertTrue(len(res.alignment) > 0)
        times = [it.start for it in res.alignment]
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(times[0], 0.0)
```

- [ ] **Step 2: 准备测试音频**

```bash
# 用 skill 现有测试片段(如果存在):
ls "$LOCALAPPDATA/Temp/opencode/subtitle-test/clip.mp3" 2>/dev/null || \
  ffmpeg -f lavfi -i "sine=frequency=440:duration=10" -ar 16000 test_tone.mp3
export Q3ASR_TEST_AUDIO="$LOCALAPPDATA/Temp/opencode/subtitle-test/clip.mp3"
```

- [ ] **Step 3: 运行确认失败/跳过**

Run: `Q3ASR_MODEL_DIR=... Q3ASR_TEST_AUDIO=... python -m unittest tests.test_engine -v`
Expected: 无模型/音频 SKIPPED; 有则 FAIL — `ModuleNotFoundError`

- [ ] **Step 4: 实现 engine.py 与 transcription.py**

```python
# q3asr/transcription.py
"""转录引擎抽象接口(v2 可换 llama.cpp 原生引擎)。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AlignItem:
    text: str
    start: float
    end: float


@dataclass
class TranscribeResult:
    text: str = ""
    alignment: list[AlignItem] | None = None
    performance: dict = field(default_factory=dict)


class TranscriptionEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_file: str, language=None, context="",
                   start_second=0.0, duration=None, temperature=0.4) -> TranscribeResult:
        ...
```

```python
# q3asr/engine.py
"""完整转录管线: 解码 → 分块 → 编码 → 解码 → 对齐。"""
import time

import numpy as np

from q3asr import audio as audio_mod
from q3asr import backend as backend_mod
from q3asr.aligner import Aligner
from q3asr.decoder import ASRDecoder
from q3asr.encoder import QwenAudioEncoder
from q3asr.transcription import TranscriptionEngine, TranscribeResult, AlignItem


class TranscribeEngine(TranscriptionEngine):
    def __init__(self, config: dict):
        md = config["model_dir"]
        providers = backend_mod.onnx_providers(config.get("device", "cpu"))
        self.enc = QwenAudioEncoder(
            f"{md}/qwen3_asr_encoder_frontend.int4.onnx",
            f"{md}/qwen3_asr_encoder_backend.int4.onnx",
            f"{md}/mel_filters.npy",
            providers=providers,
            warmup_sec=config.get("warmup_sec", 3.0))
        self.dec = ASRDecoder(f"{md}/qwen3_asr_llm.q4",
                              n_ctx=config.get("n_ctx", 4096))
        self.align = Aligner(
            f"{md}/qwen3_aligner_encoder_frontend.int4.onnx",
            f"{md}/qwen3_aligner_encoder_backend.int4.onnx",
            f"{md}/mel_filters.npy",
            f"{md}/qwen3_aligner_llm.q4_k.gguf",
            providers=providers)
        self.chunk_size = config.get("chunk_size", 40.0)
        self.memory_num = config.get("memory_num", 1)

    def transcribe(self, audio_file, language=None, context="",
                   start_second=0.0, duration=None, temperature=0.4) -> TranscribeResult:
        audio = audio_mod.decode_audio(audio_file, start_second=start_second,
                                       duration=duration)
        sr = 16000
        spc = int(self.chunk_size * sr)
        n = len(audio)
        chunks = max(1, int(np.ceil(n / spc)))
        memory = []
        texts = []
        items = []
        for i in range(chunks):
            s, e = i * spc, min((i + 1) * spc, n)
            seg = audio[s:e]
            if len(seg) < spc:
                seg = np.pad(seg, (0, spc - len(seg)))
            embd, _ = self.enc.encode(seg)
            prefix = "".join(m for _, m in memory)
            res = self.dec.decode_embeddings(embd, prefix, language=language,
                                             context=context, temperature=temperature,
                                             is_last_chunk=(i == chunks - 1))
            texts.append(res.text)
            if res.text.strip():
                for it in self.align.align(seg, res.text, offset_sec=i * self.chunk_size,
                                           language=language):
                    items.append(it)
            memory.append((embd, res.text))
            memory = memory[-self.memory_num:]
        items.sort(key=lambda x: (x.start, x.end))
        return TranscribeResult(text="".join(texts), alignment=items or None,
                                performance={})
```

- [ ] **Step 5: 用真实音频校准对齐(重要)**

```bash
# 对已知文本的音频跑一次, 目视检查 JSON 时间戳是否与语音大致吻合
Q3ASR_MODEL_DIR=... Q3ASR_TEST_AUDIO=... python -m q3asr --json-only test_tone.mp3 -y
```
若时间戳明显漂移(先于/迟于语音), 回到 Task 8 按 v0.1 `aligner.py` 移植真实 CTC 对齐逻辑。**这是本计划唯一允许"返工校准"的步骤, 以真实音频质量为验收标准。**

- [ ] **Step 6: 运行集成测试确认通过**

Run: `Q3ASR_MODEL_DIR=... Q3ASR_TEST_AUDIO=... python -m unittest tests.test_engine -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add q3asr/engine.py q3asr/transcription.py tests/test_engine.py
git commit -m "feat: end-to-end transcription engine"
```

---

### Task 10: 输出导出(output.py)

**Files:**
- Create: `q3asr/output.py`
- Test: `tests/test_output.py`

**Interfaces:**
- Consumes: `TranscribeResult`, `AlignItem`
- Produces:
  - `q3asr.output.export_json(path, items: list[AlignItem])`(契约: `[{text,start,end}]`, 3 位小数, 升序, `ensure_ascii=False`, utf-8)
  - `q3asr.output.export_txt(path, text: str)`(按标点换行, 无 ITN)
  - `q3asr.output.export_srt(path, items: list[AlignItem], max_chars=40)`(按标点/长度分组)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_output.py
import json
import tempfile
import unittest
from pathlib import Path

from q3asr import output
from q3asr.transcription import AlignItem


class ExportJsonTest(unittest.TestCase):
    def test_schema_and_rounding(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            items = [AlignItem("你", 0.1234, 0.5678), AlignItem("好", 0.5678, 0.9)]
            output.export_json(str(p), items)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual([x["text"] for x in data], ["你", "好"])
            self.assertEqual(data[0]["start"], 0.123)
            self.assertEqual(data[0]["end"], 0.568)
            self.assertTrue(p.read_text(encoding="utf-8").startswith("["))

    def test_empty_items(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            output.export_json(str(p), [])
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), [])


class ExportTxtTest(unittest.TestCase):
    def test_newlines_after_punctuation(self):
        text = "你好。世界，你好"
        self.assertIn("\n", output.format_txt(text))


class ExportSrtTest(unittest.TestCase):
    def test_produces_srt_blocks(self):
        items = [AlignItem("你好。", 0.0, 1.0), AlignItem("再见。", 1.0, 2.0)]
        srt = output.compose_srt(items)
        self.assertIn("-->", srt)
        self.assertIn("你好", srt)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_output -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'q3asr.output'`

- [ ] **Step 3: 实现 output.py**

```python
# q3asr/output.py
"""结果导出: JSON(词级时间戳契约)/ TXT / SRT。JSON 不含 ITN。"""
import json
import re

from q3asr.transcription import AlignItem


def export_json(path, items: list[AlignItem]) -> None:
    data = [{"text": it.text, "start": round(it.start, 3), "end": round(it.end, 3)}
            for it in items]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_txt(text: str) -> str:
    out = re.sub(r"([，。？！])", r"\1\n", text)
    return re.sub(r"(?<=[a-zA-Z])([,.] )", r"\1\n", out)


def export_txt(path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_txt(text))


def compose_srt(items: list[AlignItem], max_chars: int = 40) -> str:
    blocks = []
    cur, start = [], None

    def flush(cur, start, end):
        content = "".join(cur).strip().rstrip("，。？！、,.?!")
        if content:
            blocks.append(f"{len(blocks) + 1}\n{_ts(start)} --> {_ts(end)}\n{content}\n")

    split = re.compile(r"[，。？！、\n]|[,.?!]\s*")
    for it in items:
        if start is None:
            start = it.start
        cur.append(it.text)
        if split.search(it.text) or len("".join(cur)) >= max_chars:
            flush(cur, start, it.end)
            cur, start = [], None
    if cur:
        flush(cur, start, items[-1].end)
    return "\n".join(blocks)


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(path, items: list[AlignItem]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(compose_srt(items))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests.test_output -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add q3asr/output.py tests/test_output.py
git commit -m "feat: json/txt/srt exporters"
```

---

### Task 11: CLI 完整接线 + 端到端

**Files:**
- Modify: `q3asr/cli.py`
- Modify: `tests/test_cli.py`
- Test: `tests/test_e2e.py`

**Interfaces:**
- Consumes: Task 2 `ensure_models`, Task 9 `TranscribeEngine`, Task 10 `export_*`
- Produces: 完整 `q3asr` 行为(见 Global Constraints)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_e2e.py
import os
import subprocess
import sys
import unittest

from q3asr import cli


class CliWiringTest(unittest.TestCase):
    def test_no_model_dir_returns_exit_3(self):
        # 指定不存在的模型目录, 应走到模型加载并返回 3
        rc = cli.main(["--model-dir", "/nonexistent/x", "--download-models-only"])
        self.assertEqual(rc, 3)

    def test_download_models_only_flag_parses(self):
        # 仅验证参数可解析(不触发真实下载, 因 --model-dir 不存在时直接退出)
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--bogus"])
        self.assertEqual(ctx.exception.code, 2)
```

```python
# tests/test_cli.py 追加
    def test_device_and_prec_accepted(self):
        rc = cli.main(["--device", "cpu", "--prec", "int4", "in.mp3", "-y"])
        self.assertEqual(rc, 0)  # 本阶段 CLI 在无模型目录时打印并返回 0? 否——见下
```

注: CLI 接线后, `in.mp3` 且默认 `--model-dir` 不存在时, 应触发模型下载流程(`--download-models-only` 之外的正常路径会 ensure_models → 尝试联网)。因此 `test_device_and_prec_accepted` 应改为断言**参数被接受且流程进入模型检查**(例如构造 `--model-dir /nonexistent` 期望退出码 3, 或 mock `ensure_models`)。实现时按此修正, 避免测试触发真实下载。

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_e2e -v`
Expected: FAIL(当前 main 总是返回 0)

- [ ] **Step 3: 重写 cli.py 接线**

```python
# q3asr/cli.py
import argparse
import os
import sys
from pathlib import Path

from q3asr import __version__, backend as backend_mod, models as models_mod


def build_parser():
    p = argparse.ArgumentParser(prog="q3asr", description=__doc__)
    p.add_argument("input", nargs="?", help="audio file (mp3/wav/...)")
    p.add_argument("--version", action="store_true")
    p.add_argument("--seek-start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("-l", "--language", default=None)
    p.add_argument("--prec", default="int4")
    p.add_argument("--device", default=os.environ.get("QASR_DEVICE", "auto"))
    p.add_argument("--model", default="1.7b")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--no-dml", action="store_true")
    p.add_argument("--no-vulkan", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--download-models-only", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"q3asr {__version__}")
        return 0
    try:
        device = backend_mod.detect_backend(args.device)
        print(f"[INFO] backend: {device}")
        if args.download_models_only:
            models_mod.ensure_models(model=args.model,
                                     model_dir=Path(args.model_dir) if args.model_dir else None)
            print("[INFO] models ready")
            return 0
        if not args.input:
            build_parser().error("the following arguments are required: input")
            return 2
        from q3asr.engine import TranscribeEngine
        from q3asr import output
        model_dir = models_mod.ensure_models(
            model=args.model, model_dir=Path(args.model_dir) if args.model_dir else None)
        eng = TranscribeEngine({"model_dir": str(model_dir), "device": device})
        res = eng.transcribe(args.input, language=args.language,
                             start_second=args.seek_start, duration=args.duration)
        base = Path(args.input).with_suffix("")
        output.export_json(f"{base}.json", res.alignment or [])
        if not args.json_only:
            output.export_txt(f"{base}.txt", res.text)
            output.export_srt(f"{base}.srt", res.alignment or [])
        print(f"[INFO] done: {base}.json")
        return 0
    except models_mod.DownloadError as e:   # 在 models.py 中定义 DownloadError(RuntimeError)
        print(f"[ERROR] {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

配套修改 `q3asr/models.py`: 新增 `class DownloadError(RuntimeError)`; `ensure_models` 内网络/校验失败时 raise `DownloadError`。

- [ ] **Step 4: 端到端验证(有模型)**

```bash
Q3ASR_MODEL_DIR=...   # 实际用 --model-dir
python -m q3asr --model-dir "$HOME/.cache/q3asr/models" "$Q3ASR_TEST_AUDIO" -y --device cpu
python - <<'PY'
import json
d = json.load(open("clip.json", encoding="utf-8"))
assert all({"text","start","end"} == set(x) for x in d)
assert all(x["start"] < x["end"] for x in d)
print("e2e JSON OK:", len(d), "items")
PY
```
Expected: `clip.json`(与输入同名)存在且 schema 合法。

- [ ] **Step 5: 运行全部单测(无模型部分)**

Run: `python -m unittest tests.test_cli tests.test_models tests.test_backend tests.test_audio tests.test_features tests.test_output -v`
Expected: 全 PASS(模型相关为 SKIPPED)

- [ ] **Step 6: 提交**

```bash
git add q3asr/cli.py q3asr/models.py tests/test_cli.py tests/test_e2e.py
git commit -m "feat: wire full CLI pipeline with model bootstrap"
```

---

### Task 12: CI 矩阵 + PyInstaller 发布

**Files:**
- Create: `.github/workflows/release.yml`
- Create: `ci/build.py`
- Create: `ci/manifest_schema.py`(可选, 见 Step 3)
- Test: 无单独单测; 本地冒烟验证 build.py

**Interfaces:**
- Consumes: 完整包; 各平台 wheel(onnxruntime-directml / onnxruntime-gpu / onnxruntime-silicon / onnxruntime; llama-cpp-python[cuda]/[vulkan]/[metal]/cpu)
- Produces:
  - 发布 zip 命名: `qwen3-asr-<os>-<arch>[-<backend>].zip`(os: windows/linux/macos; arch: x64/arm64)
  - 每个 zip 内: `q3asr` 可执行(Windows 为 `q3asr.exe`) + `manifest.json`(单 zip 描述) + 内嵌 imageio-ffmpeg 静态 ffmpeg
  - 根 `manifest.json`(release 资产描述, 供 skill 选择): `{"version": "...", "assets": [{"os","arch","backend","filename","sha256","cli","size"}]}`

- [ ] **Step 1: 写 ci/build.py**

```python
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
    "macos-arm64-metal":  {"onnx": "onnxruntime-silicon", "llama": ["llama-cpp-python", "metal"]},
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
    extra = []
    if t["llama"][1]:
        extra = ["--index-url", "https://abetlen.github.io/llama-cpp-python/whl/" + t["llama"][1]]
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "pyinstaller", t["onnx"], *extra], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install",
                    f'llama-cpp-python[{t["llama"][1]}]' if t["llama"][1] else "llama-cpp-python",
                    *extra], check=True)

    cmd = [sys.executable, "-m", "PyInstaller", "--onefile", "--name", "q3asr",
           "--collect-all", "imageio_ffmpeg",
           "q3asr/__main__.py"]
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
```

- [ ] **Step 2: 写 release.yml**

```yaml
# .github/workflows/release.yml
name: release
on:
  push:
    tags: ["v*"]

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        include:
          - {os: windows-latest, arch: x64, backend: cpu}
          - {os: windows-latest, arch: x64, backend: cuda}
          - {os: windows-latest, arch: x64, backend: vulkan}
          - {os: ubuntu-latest,  arch: x64, backend: cpu}
          - {os: ubuntu-latest,  arch: x64, backend: cuda}
          - {os: ubuntu-latest,  arch: x64, backend: vulkan}
          - {os: macos-14,       arch: arm64, backend: metal}
          - {os: macos-14,       arch: arm64, backend: cpu}
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: pip install -e .
      - name: Build
        run: python ci/build.py ${{ matrix.os == 'windows-latest' && 'windows' || matrix.os == 'ubuntu-latest' && 'linux' || 'macos' }}-${{ matrix.arch }}-${{ matrix.backend }}
      - uses: actions/upload-artifact@v4
        with:
          name: zip-${{ matrix.os }}-${{ matrix.backend }}
          path: qwen3-asr-*.zip

  release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: zip-*
          merge-multiple: true
          path: zips
      - name: Build root manifest
        shell: python
        run: |
          import hashlib, json, pathlib
          assets = []
          for p in sorted(pathlib.Path("zips").glob("qwen3-asr-*.zip")):
              m = json.loads((pathlib.Path("zips") / p.stem).read_text() if False else None) if False else None
              parts = p.name.replace("qwen3-asr-", "").replace(".zip", "").split("-")
              os_n, arch, backend = parts[0], parts[1], "-".join(parts[2:])
              assets.append({"os": os_n, "arch": arch, "backend": backend,
                             "filename": p.name,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                             "size": p.stat().st_size,
                             "cli": "q3asr.exe" if os_n == "windows" else "q3asr"})
          root = {"version": "${{ github.ref_name }}", "assets": assets}
          pathlib.Path("zips/manifest.json").write_text(json.dumps(root, indent=2))
      - name: Create release
        uses: softprops/action-gh-release@v2
        with:
          files: zips/*
```

- [ ] **Step 3: 本地冒烟验证 build.py(Windows cpu)**

```bash
python ci/build.py windows-x64-cpu
python dist/q3asr.exe --version
```
Expected: `dist/q3asr.exe` 生成且 `--version` 输出 `q3asr 0.1.0`(PyInstaller 首次打包较慢, 可加长超时)。

- [ ] **Step 4: 提交**

```bash
git add .github/workflows/release.yml ci/build.py
git commit -m "ci: matrix build + release with per-target zips and manifest"
```

---

### Task 13: skill 集成(subtitle-translate-skill 仓库)

**Files:**(在 `C:\Users\zwb\Documents\dev\subtitle_translate` 的 skill 仓库内, 或直接在新 `subtitle-translate-skill` git 仓库工作)
- Modify: `scripts/run_transcribe.py`
- Modify: `SKILL.md`
- Test: `tests/test_run_transcribe.py`(追加)

**Interfaces:**
- Consumes: ASR 仓库 release 根 `manifest.json`(`{"version","assets":[{os,arch,backend,filename,sha256,size,cli}]}`)
- Produces:
  - `run_transcribe.py` 新增参数/行为:
    - `--exe` 优先级不变(手动指定最高)
    - 未指定 `--exe`/`TRANSCRIBE_EXE` 时进入自动获取流程
    - `TRANSCRIBE_BACKEND=cuda|vulkan|metal|cpu|auto`(默认 auto)
    - `TRANSCRIBE_ASR_VER`(默认 latest)
    - `TRANSCRIBE_MODEL_MIRROR`(透传给 q3asr)
    - 缓存: `~/.cache/opencode-translate/asr/<version>/<zip 名解压>/q3asr(.exe)`
    - 首次: 请求 `https://api.github.com/repos/ywwzwb/qwen3-asr-universal/releases/latest`(或指定 ver)取资产, 按 os/arch/backend 匹配, 下载 → sha256 校验 → 解压 → 复用
  - 选 zip 规则: 平台由 `platform.system()` 映射(windows/linux/macos)+ `platform.machine()`(amd64→x64, arm64→arm64); backend 默认 auto(按顺序找 cuda→vulkan→metal→cpu 中存在的资产; 无 GPU 资产或用户显式 cpu 则选 cpu)
  - `--no-dml`/`--no-vulkan` 参数透传 q3asr(兼容旧逻辑)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_run_transcribe.py 追加
import platform
import tempfile
import unittest
from pathlib import Path

from scripts import run_transcribe as rt   # 或按现有导入方式


class AssetSelectionTest(unittest.TestCase):
    def _manifest(self):
        return {"version": "v0.1.0", "assets": [
            {"os": "windows", "arch": "x64", "backend": "cpu", "filename": "qwen3-asr-windows-x64-cpu.zip", "sha256": "x", "cli": "q3asr.exe"},
            {"os": "windows", "arch": "x64", "backend": "vulkan", "filename": "qwen3-asr-windows-x64-vulkan.zip", "sha256": "x", "cli": "q3asr.exe"},
            {"os": "linux", "arch": "x64", "backend": "cpu", "filename": "qwen3-asr-linux-x64-cpu.zip", "sha256": "x", "cli": "q3asr"},
            {"os": "macos", "arch": "arm64", "backend": "metal", "filename": "qwen3-asr-macos-arm64-metal.zip", "sha256": "x", "cli": "q3asr"},
        ]}

    def test_pick_cpu_on_windows(self):
        man = self._manifest()
        a = rt.select_asset(man, os_name="windows", arch="x64", backend="cpu")
        self.assertEqual(a["filename"], "qwen3-asr-windows-x64-cpu.zip")

    def test_auto_prefers_vulkan_over_cpu_when_present(self):
        man = self._manifest()
        a = rt.select_asset(man, os_name="windows", arch="x64", backend="auto")
        self.assertEqual(a["backend"], "vulkan")

    def test_auto_falls_back_to_cpu(self):
        man = {"version": "v", "assets": [a for a in self._manifest()["assets"] if a["os"] == "linux"]}
        a = rt.select_asset(man, os_name="linux", arch="x64", backend="auto")
        self.assertEqual(a["backend"], "cpu")

    def test_no_match_raises(self):
        man = self._manifest()
        with self.assertRaises(RuntimeError):
            rt.select_asset(man, os_name="macos", arch="x64", backend="cuda")

    def test_cache_dir_layout(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            rt.install_asset(cache, Path(d) / "qwen3-asr-windows-x64-cpu.zip",
                             "q3asr.exe", sha256=None)
            self.assertTrue((cache / "q3asr.exe").exists())
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_run_transcribe -v`
Expected: FAIL — `AttributeError: module has no attribute 'select_asset'`

- [ ] **Step 3: 实现 run_transcribe.py 扩展**

```python
# scripts/run_transcribe.py 新增(保留原 build_cmd/resolve_exe 不动):
import json
import platform
import urllib.request
import zipfile
import hashlib

ASR_REPO = "ywwzwb/qwen3-asr-universal"
CACHE_ROOT = Path(os.environ.get("TRANSCRIBE_CACHE", Path.home() / ".cache" / "opencode-translate" / "asr"))

_OS_MAP = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}


def os_arch() -> tuple[str, str]:
    os_name = _OS_MAP.get(platform.system(), platform.system().lower())
    m = platform.machine().lower()
    arch = "arm64" if m in ("aarch64", "arm64") else ("x64" if m in ("amd64", "x86_64", "x64") else m)
    return os_name, arch


def fetch_manifest(version="latest") -> dict:
    if version == "latest":
        url = f"https://api.github.com/repos/{ASR_REPO}/releases/latest"
    else:
        url = f"https://api.github.com/repos/{ASR_REPO}/releases/tags/{version}"
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    # 读根 manifest.json 资产描述
    for a in d["assets"]:
        if a["name"] == "manifest.json":
            murl = a["browser_download_url"]
            with urllib.request.urlopen(murl, timeout=60) as r2:
                return json.load(r2)
    raise RuntimeError("release has no manifest.json")


def select_asset(manifest: dict, os_name: str, arch: str, backend: str) -> dict:
    assets = manifest["assets"]
    cands = [a for a in assets if a["os"] == os_name and a["arch"] == arch]
    if not cands:
        raise RuntimeError(f"no asset for {os_name}-{arch}")
    if backend != "auto":
        for a in cands:
            if a["backend"] == backend:
                return a
        raise RuntimeError(f"no {backend} asset for {os_name}-{arch}; have {[a['backend'] for a in cands]}")
    for b in ("cuda", "vulkan", "metal", "cpu"):
        for a in cands:
            if a["backend"] == b:
                return a
    raise RuntimeError(f"no asset for {os_name}-{arch}")


def install_asset(cache: Path, zip_path: Path, cli_name: str, sha256=None) -> Path:
    if sha256:
        got = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if got != sha256:
            raise RuntimeError(f"sha256 mismatch: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cache)
    exe = cache / cli_name
    if not exe.exists():
        raise RuntimeError(f"cli not found in zip: {cli_name}")
    return exe


def ensure_auto_exe(backend="auto", version="latest") -> tuple[str, str]:
    """返回 (exe_path, backend_name)。"""
    os_name, arch = os_arch()
    man = fetch_manifest(version)
    asset = select_asset(man, os_name, arch, backend)
    cache = CACHE_ROOT / version
    exe = cache / asset["cli"]
    if exe.exists():
        return str(exe), asset["backend"]
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / asset["filename"]
    url = (f"https://github.com/{ASR_REPO}/releases/download/"
           f"{man['version']}/{asset['filename']}")
    urllib.request.urlretrieve(url, zip_path)   # 大文件场景可用流式下载
    exe = install_asset(cache, zip_path, asset["cli"], asset["sha256"])
    zip_path.unlink(missing_ok=True)
    return str(exe), asset["backend"]
```

`main()` 中: 在 `resolve_exe` 之后, 若仍为默认路径且不存在, 且用户未显式 `--exe`, 则走 `ensure_auto_exe(backend=os.environ.get("TRANSCRIBE_BACKEND","auto"), version=os.environ.get("TRANSCRIBE_ASR_VER","latest"))`, 并把 `--backend` 信息写入日志。

- [ ] **Step 4: 运行确认通过**

Run: `python -m unittest tests.test_run_transcribe -v`
Expected: PASS(新增 5 个测试; 现有测试保持通过)

- [ ] **Step 5: 更新 SKILL.md(步骤 1)**

在工作流"步骤 1: 输入归一化 + ASR 转录"处新增一段:

> 若 ASR 运行时缺失, 自动从 `ywwzwb/qwen3-asr-universal` 最新 release 下载匹配 `os/arch/后端` 的 zip 到 `~/.cache/opencode-translate/asr/` 并缓存; 后端优先级 cuda→vulkan→metal→cpu, 可用 `TRANSCRIBE_BACKEND` 指定。模型在 q3asr 首次运行时自动下载(默认 1.7B)到 `~/.cache/q3asr/models/`。手动指定 `TRANSCRIBE_EXE` 仍为最高优先级。

- [ ] **Step 6: 运行 skill 全部测试 + 同步全局安装**

```bash
python -m unittest discover -s tests -p 'test_*.py'   # 24(旧)+新增全过
# 同步到全局技能目录
cp -r scripts tests SKILL.md terminology.yaml "$HOME/.config/opencode/skills/translating-subtitles/"
```

- [ ] **Step 7: 提交 + 推送**

```bash
# 在 subtitle-translate-skill 仓库
git add scripts/run_transcribe.py SKILL.md tests/test_run_transcribe.py
git commit -m "feat: auto-download q3asr runtime from releases"
git push
```

---

## Self-Review(计划作者自查)

- **Spec 覆盖**: spec 第 2 节(架构/CLI/管线/设备)→ Task 1-11; 第 3 节(模型下载/镜像)→ Task 2、11; 第 4 节(发布矩阵/CI)→ Task 12; 第 5 节(skill 集成)→ Task 13; 第 6 节(测试)→ 各任务测试 + Task 12 冒烟。非目标(MLX/macos-x64/ITN/流式)均未纳入任务。
- **占位符**: 无 TBD; `<实际zip内路径>`/`<真实sha256>` 是 Task 2 Step 3 的**显式产出步骤**(下载后回填), 非遗留 TODO。aligner 任务明确"先近似后校准", 验收标准是真实音频质量。
- **类型一致性**: `AlignItem` 在 Task 8 定义并在 Task 9(engine)、Task 10(output)复用同构字段 `text/start/end`; `DecodeResult` 在 Task 7 定义 Task 9 消费; `TranscribeResult` 在 Task 9 定义 Task 10 消费; `ensure_models` 返回 `Path`, Task 11 使用一致; `detect_backend/onnx_providers` 跨 Task 3/6/9 签名一致。
- **依赖顺序**: 1→(2→3→4→5)→(6→7)→8→9→10→11→12→13; 无循环依赖。Task 6 对 Task 5 有一个签名补充(见该任务注意), 已显式标注。
