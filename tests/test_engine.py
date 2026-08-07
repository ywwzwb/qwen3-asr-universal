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
