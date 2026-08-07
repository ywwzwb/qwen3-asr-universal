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
        # 1.7b/0.6b 条目必须含 encoder frontend/backend + llm, aligner 条目含三件套; 全部带 sha256 和 url
        for name, m in entries.items():
            names = [f["name"] for f in m["files"]]
            if name != "aligner":
                for req in ("qwen3_asr_encoder_frontend.int4.onnx",
                            "qwen3_asr_encoder_backend.int4.onnx",
                            "qwen3_asr_llm.q4_k.gguf"):
                    self.assertIn(req, names, f"{name} 缺少 {req}")
            for f in m["files"]:
                self.assertTrue(f["sha256"], f"{name}/{f['name']} 缺少 sha256")
                self.assertTrue(f["url"], f"{name}/{f['name']} 缺少 url")
        aligner = {f["name"] for f in entries["aligner"]["files"]}
        for req in ("qwen3_aligner_encoder_frontend.int4.onnx",
                    "qwen3_aligner_encoder_backend.int4.onnx",
                    "qwen3_aligner_llm.q4_k.gguf"):
            self.assertIn(req, aligner, f"aligner 缺少 {req}")

    def test_resolve_paths_flat_dir(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("qwen3_asr_encoder_frontend.int4.onnx",
                         "qwen3_asr_encoder_backend.int4.onnx",
                         "qwen3_asr_llm.q4_k.gguf",
                         "qwen3_aligner_encoder_frontend.int4.onnx",
                         "qwen3_aligner_encoder_backend.int4.onnx",
                         "qwen3_aligner_llm.q4_k.gguf"):
                (root / name).touch()
            spec = models.spec_from_dir(root, "0.6b")
            for key in ("asr_frontend", "asr_backend", "asr_llm",
                        "align_frontend", "align_backend", "align_llm"):
                self.assertTrue(spec[key].exists(), f"{key} -> {spec[key]}")
