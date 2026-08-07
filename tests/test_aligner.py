import os
import unittest
from pathlib import Path

import numpy as np

from q3asr import aligner, models

MODEL_DIR = os.environ.get("Q3ASR_MODEL_DIR")
SPEC = models.resolve_paths("0.6b", Path(MODEL_DIR)) if MODEL_DIR else None


@unittest.skipUnless(SPEC and SPEC["align_frontend"].exists(), "set Q3ASR_MODEL_DIR to run")
class AlignerIntegrationTest(unittest.TestCase):
    def test_align_returns_sorted_items(self):
        al = aligner.Aligner(
            str(SPEC["align_frontend"]),
            str(SPEC["align_backend"]),
            str(SPEC["mel_filters"]),
            str(SPEC["align_llm"]),
        )
        audio = np.zeros(16000 * 2, dtype=np.float32)
        items = al.align(audio, "你好", offset_sec=10.0, language="Chinese")
        self.assertTrue(items)
        joined = "".join(it.text for it in items)
        self.assertIn("你", joined)
        self.assertIn("好", joined)
        times = [it.start for it in items] + [items[-1].end]
        self.assertEqual(times, sorted(times))
