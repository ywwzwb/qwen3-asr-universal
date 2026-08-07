"""模型清单与自动下载(断点续传 + sha256 校验 + 解压)。"""
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml

_RESOURCES = Path(__file__).parent.parent / "resources"


class DownloadError(RuntimeError):
    pass


def load_manifest() -> list[dict]:
    with open(_RESOURCES / "models.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    base_urls = data["base_urls"]
    for m in data["models"]:
        m["base_urls"] = base_urls
        for f in m["files"]:
            f["url"] = base_urls["gh"] + "/" + m["zip"]
    return data["models"]


MODEL_MANIFEST = load_manifest()

_ASR_FILES = ("qwen3_asr_encoder_frontend.int4.onnx",
              "qwen3_asr_encoder_backend.int4.onnx",
              "qwen3_asr_llm.q4_k.gguf")
_ALIGN_FILES = ("qwen3_aligner_encoder_frontend.int4.onnx",
                "qwen3_aligner_encoder_backend.int4.onnx",
                "qwen3_aligner_llm.q4_k.gguf")


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
    tmp = dest.with_suffix(dest.suffix + ".part")
    mode = "ab" if tmp.exists() and tmp.stat().st_size > 0 else "wb"
    headers = {"User-Agent": "q3asr/0.1"}
    if mode == "ab":
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
    req = urllib.request.Request(url, headers=headers)
    try:
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
    except Exception as e:
        raise DownloadError(f"download failed: {url}: {e}") from e
    tmp.replace(dest)


def _ensure_entry(m: dict, f: dict, dl_dir: Path, mirror: str) -> Path:
    out = dl_dir / "extract" / f["in_zip"]
    if out.exists() and sha256_of(out) == f["sha256"]:
        return out
    zip_path = dl_dir / m["zip"]
    zip_url = mirror_url(m["base_urls"][mirror] + "/" + m["zip"], mirror)
    if not (zip_path.exists() and sha256_of(zip_path) == m["zip_sha256"]):
        _download(zip_url, zip_path)
    if sha256_of(zip_path) != m["zip_sha256"]:
        raise DownloadError(f"zip sha256 mismatch: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extract(f["in_zip"], dl_dir / "extract")
    if sha256_of(out) != f["sha256"]:
        raise DownloadError(f"sha256 mismatch for {f['name']}: got {sha256_of(out)}")
    print(f"[INFO] model file ready: {out}")
    return out


def ensure_models(model: str = "1.7b", mirror: str = "gh") -> dict[str, Path]:
    """下载+校验+解压到默认缓存布局, 返回路径表。"""
    root = default_model_dir()
    root.mkdir(parents=True, exist_ok=True)
    dl_dir = root / "_downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)

    mel = root / "mel_filters.npy"
    if not mel.exists():
        shutil.copyfile(_RESOURCES / "mel_filters.npy", mel)

    asr_m = next(x for x in MODEL_MANIFEST if x["name"] == model)
    align_m = next(x for x in MODEL_MANIFEST if x["name"] == "aligner")
    asr_dir = root / model
    align_dir = root / "aligner"
    for sub, m, names in ((asr_dir, asr_m, _ASR_FILES), (align_dir, align_m, _ALIGN_FILES)):
        sub.mkdir(parents=True, exist_ok=True)
        for fn in names:
            f = next(x for x in m["files"] if x["name"] == fn)
            src = _ensure_entry(m, f, dl_dir, mirror)
            out = sub / fn
            if not out.exists():
                os.replace(src, out)
    return build_spec(model)


def spec_from_dir(model_dir: Path, model: str = "1.7b") -> dict[str, Path]:
    """由用户提供的平铺目录构建路径表。"""
    md = Path(model_dir)
    return {
        "mel_filters": md / "mel_filters.npy",
        "asr_frontend": md / "qwen3_asr_encoder_frontend.int4.onnx",
        "asr_backend": md / "qwen3_asr_encoder_backend.int4.onnx",
        "asr_llm": md / "qwen3_asr_llm.q4_k.gguf",
        "align_frontend": md / "qwen3_aligner_encoder_frontend.int4.onnx",
        "align_backend": md / "qwen3_aligner_encoder_backend.int4.onnx",
        "align_llm": md / "qwen3_aligner_llm.q4_k.gguf",
    }


def build_spec(model: str = "1.7b") -> dict[str, Path]:
    root = default_model_dir()
    d = spec_from_dir(root / model, model)
    d["mel_filters"] = root / "mel_filters.npy"
    for align in ("align_frontend", "align_backend", "align_llm"):
        d[align] = root / "aligner" / d[align].name
    return d


def resolve_paths(model: str = "1.7b", model_dir: Path | None = None) -> dict[str, Path]:
    """model_dir 未给 → ensure_models(自动下载); 给了 → 平铺目录解释, 缺文件抛 DownloadError。"""
    if model_dir is None:
        return ensure_models(model=model)
    spec = spec_from_dir(model_dir, model)
    missing = [str(v) for v in spec.values() if not v.exists()]
    if missing:
        raise DownloadError(f"model files not found in {model_dir}: {missing}")
    return spec
