import os
import unittest
from pathlib import Path

import numpy as np

from q3asr import decoder

MODEL_DIR = os.environ.get("Q3ASR_MODEL_DIR")


@unittest.skipUnless(MODEL_DIR, "set Q3ASR_MODEL_DIR to run")
class DecoderIntegrationTest(unittest.TestCase):
    def test_special_ids_present(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4_k.gguf"))
        ids = d.special_ids()
        for key in ("im_start", "im_end", "audio_start", "audio_end", "asr_text", "eos"):
            self.assertIn(key, ids)

    def test_tokenize_roundtrip(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4_k.gguf"))
        ids = d.tokenize("hello world")
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_token_embeddings_shape(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4_k.gguf"))
        ids = d.tokenize("你好")
        emb = d.token_embeddings(ids)
        self.assertEqual(emb.shape[0], len(ids))
        self.assertGreater(emb.shape[1], 0)
        self.assertEqual(emb.dtype, np.float32)

    def test_decode_silence_does_not_crash(self):
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4_k.gguf"))
        ids = d.special_ids()
        embed_tbl = d.token_embeddings([ids["im_start"]])
        noise = np.random.randn(50, embed_tbl.shape[1]).astype(np.float32) * 1e-3
        embd = np.concatenate([embed_tbl, noise, embed_tbl], axis=0)
        res = d.decode_embeddings(embd, "", temperature=0.0, max_new_tokens=16)
        self.assertIsInstance(res.text, str)

    def test_non_last_chunk_flushes_tail_tokens(self):
        if not os.environ.get("Q3ASR_TEST_AUDIO"):
            self.skipTest("set Q3ASR_TEST_AUDIO to run")
        from q3asr import audio as audio_mod
        from q3asr.encoder import QwenAudioEncoder
        d = decoder.ASRDecoder(str(Path(MODEL_DIR) / "qwen3_asr_llm.q4_k.gguf"))
        enc = QwenAudioEncoder(str(Path(MODEL_DIR) / "qwen3_asr_encoder_frontend.int4.onnx"),
                               str(Path(MODEL_DIR) / "qwen3_asr_encoder_backend.int4.onnx"),
                               str(Path(MODEL_DIR) / "mel_filters.npy"))
        x = audio_mod.decode_audio(os.environ["Q3ASR_TEST_AUDIO"], duration=15.0)
        seg = np.pad(x, (0, 16000 * 40 - len(x)))
        embd, _ = enc.encode(seg)
        final = d.decode_embeddings(embd, "", temperature=0.0, max_new_tokens=64,
                                    is_last_chunk=True)
        mid = d.decode_embeddings(embd, "", temperature=0.0, max_new_tokens=64,
                                  is_last_chunk=False)
        self.assertTrue(final.text, "expected model to generate text")
        self.assertEqual(mid.text, final.text,
                         "non-last chunk must not drop tail tokens")
