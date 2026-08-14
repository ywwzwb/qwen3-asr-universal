import sys
import unittest
from q3asr import cli
from unittest import mock


class CliParseTest(unittest.TestCase):
    def test_version(self):
        rc = cli.main(["--version"])
        self.assertEqual(rc, 0)

    def test_accepts_skill_style_args(self):
        # 复刻 skill run_transcribe.py build_cmd 的实际调用形态
        argv = ["in.mp3", "--seek-start", "1140", "--duration", "30",
                "-l", "Chinese", "--prec", "int4", "--no-dml", "--no-vulkan", "-y"]
        # --help 是 argparse 内建行为，解析成功即以 SystemExit(0) 退出
        with self.assertRaises(SystemExit) as ctx:
            cli.main(argv + ["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_unknown_flag_is_exit_code_2(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--bogus-flag"])
        self.assertEqual(ctx.exception.code, 2)

    def test_missing_input_is_exit_code_2(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_device_and_prec_accepted(self):
        # --model-dir 不存在 → resolve_paths 抛 DownloadError → 退出码 3, 不触发网络下载
        rc = cli.main(["--device", "cpu", "--prec", "int4",
                       "--model-dir", "/nonexistent/x", "in.mp3", "-y"])
        self.assertEqual(rc, 3)

    def test_explicit_gpu_with_missing_runtime_libs_exits_1(self):
        # 显式 --device cuda 但运行库缺失 → 明确报错退出码 1, 不触发模型下载
        with mock.patch("q3asr.backend.runtime_missing",
                        return_value=["libcudart.so.12"]):
            rc = cli.main(["--device", "cuda", "-y", "in.mp3",
                           "--model-dir", "/nonexistent/x"])
        self.assertEqual(rc, 1)

    def test_auto_gpu_with_missing_runtime_libs_falls_back_to_cpu(self):
        with mock.patch("q3asr.backend.runtime_missing",
                        return_value=["libcudart.so.12"]), \
             mock.patch("q3asr.backend.detect_backend", return_value="cuda"):
            # 回退 CPU 后继续正常流程 → resolve_paths 抛 DownloadError → 3
            rc = cli.main(["-y", "in.mp3", "--model-dir", "/nonexistent/x"])
        self.assertEqual(rc, 3)
