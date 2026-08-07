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

    def test_seek_past_end_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tone.wav"
            _write_sine(p, 1.0)
            with self.assertRaises(ValueError):
                audio.decode_audio(str(p), start_second=99999.0)
