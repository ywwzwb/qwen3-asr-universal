import unittest
from unittest import mock

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


class CompiledBackendTest(unittest.TestCase):
    def test_backend_compiled_from_libdir(self):
        with mock.patch.object(backend, "_llama_cpp_lib_dir",
                               return_value=backend.Path("/fake/lib")):
            with mock.patch.object(backend.Path, "is_dir", return_value=True):
                with mock.patch.object(backend.Path, "iterdir", return_value=[
                        backend.Path("libggml-cuda.so"), backend.Path("libggml-cpu.so")]):
                    self.assertTrue(backend._backend_compiled("cuda"))
                    self.assertFalse(backend._backend_compiled("vulkan"))

    def test_backend_not_compiled_when_libdir_missing(self):
        with mock.patch.object(backend, "_llama_cpp_lib_dir", return_value=None):
            self.assertFalse(backend._backend_compiled("cuda"))
            self.assertFalse(backend._backend_compiled("vulkan"))


class RuntimeMissingTest(unittest.TestCase):
    def test_cpu_metal_never_missing(self):
        self.assertEqual(backend.runtime_missing("cpu"), [])
        self.assertEqual(backend.runtime_missing("metal"), [])

    def test_cuda_missing_lists_lib_names(self):
        with mock.patch("ctypes.CDLL", side_effect=OSError):
            miss = backend.runtime_missing("cuda")
            self.assertEqual(miss, ["libcudart.so.12", "libcublas.so.12",
                                    "libcublasLt.so.12"])

    def test_cuda_ok_when_all_loadable(self):
        with mock.patch("ctypes.CDLL"):
            self.assertEqual(backend.runtime_missing("cuda"), [])

    def test_vulkan_missing(self):
        with mock.patch("ctypes.CDLL", side_effect=OSError):
            self.assertEqual(backend.runtime_missing("vulkan"), ["libvulkan.so.1"])


class ProvidersTest(unittest.TestCase):
    def test_cuda_providers(self):
        self.assertIn("CUDAExecutionProvider", backend.onnx_providers("cuda"))

    def test_cpu_providers(self):
        self.assertEqual(backend.onnx_providers("cpu"), ["CPUExecutionProvider"])

    def test_vulkan_uses_dml_when_available(self):
        with mock.patch("onnxruntime.get_available_providers",
                        return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            self.assertEqual(backend.onnx_providers("vulkan"),
                             ["DmlExecutionProvider", "CPUExecutionProvider"])

    def test_vulkan_falls_back_to_cpu_without_dml(self):
        with mock.patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            self.assertEqual(backend.onnx_providers("vulkan"), ["CPUExecutionProvider"])
