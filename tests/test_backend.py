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
