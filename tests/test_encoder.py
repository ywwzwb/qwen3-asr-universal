import os
import unittest
from pathlib import Path

import numpy as np

from q3asr import encoder, models

MODEL_DIR = os.environ.get("Q3ASR_MODEL_DIR")
SPEC = models.resolve_paths("0.6b", Path(MODEL_DIR) if MODEL_DIR else None)


@unittest.skipUnless(SPEC["asr_frontend"].exists(), "set Q3ASR_MODEL_DIR to run")
class EncoderIntegrationTest(unittest.TestCase):
    def test_encode_5s_silence(self):
        enc = encoder.QwenAudioEncoder(
            str(SPEC["asr_frontend"]),
            str(SPEC["asr_backend"]),
            str(SPEC["mel_filters"]),
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
            str(SPEC["asr_frontend"]),
            str(SPEC["asr_backend"]),
            str(SPEC["mel_filters"]),
            warmup_sec=0.0,
        )
        e1, _ = enc.encode(np.zeros(16000 * 1, dtype=np.float32))
        e2, _ = enc.encode(np.zeros(16000 * 2, dtype=np.float32))
        self.assertGreater(e2.shape[0], e1.shape[0])
