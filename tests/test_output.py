import json
import tempfile
import unittest
from pathlib import Path

from q3asr import output
from q3asr.transcription import AlignItem


class ExportJsonTest(unittest.TestCase):
    def test_schema_and_rounding(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            items = [AlignItem("你", 0.1234, 0.5678), AlignItem("好", 0.5678, 0.9)]
            output.export_json(str(p), items)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual([x["text"] for x in data], ["你", "好"])
            self.assertEqual(data[0]["start"], 0.123)
            self.assertEqual(data[0]["end"], 0.568)
            self.assertTrue(p.read_text(encoding="utf-8").startswith("["))

    def test_empty_items(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.json"
            output.export_json(str(p), [])
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), [])


class ExportTxtTest(unittest.TestCase):
    def test_newlines_after_punctuation(self):
        text = "你好。世界，你好"
        self.assertIn("\n", output.format_txt(text))


class ExportSrtTest(unittest.TestCase):
    def test_produces_srt_blocks(self):
        items = [AlignItem("你好。", 0.0, 1.0), AlignItem("再见。", 1.0, 2.0)]
        srt = output.compose_srt(items)
        self.assertIn("-->", srt)
        self.assertIn("你好", srt)
