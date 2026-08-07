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
