"""完整转录管线: 解码 → 分块 → 编码 → 解码 → 对齐。"""
import time
from pathlib import Path

import numpy as np

from q3asr import audio as audio_mod
from q3asr import backend as backend_mod
from q3asr.aligner import Aligner
from q3asr.decoder import ASRDecoder
from q3asr.encoder import QwenAudioEncoder
from q3asr.transcription import TranscriptionEngine, TranscribeResult, AlignItem


class TranscribeEngine(TranscriptionEngine):
    def __init__(self, config: dict):
        p = config.get("paths")
        if p is None:
            from q3asr import models
            p = models.resolve_paths(config.get("model", "1.7b"),
                                     Path(config["model_dir"]))
        providers = backend_mod.onnx_providers(config.get("device", "cpu"))
        self.enc = QwenAudioEncoder(
            str(p["asr_frontend"]), str(p["asr_backend"]), str(p["mel_filters"]),
            providers=providers,
            warmup_sec=config.get("warmup_sec", 3.0))
        self.dec = ASRDecoder(str(p["asr_llm"]), n_ctx=config.get("n_ctx", 4096))
        self.align = Aligner(
            str(p["align_frontend"]), str(p["align_backend"]), str(p["mel_filters"]),
            str(p["align_llm"]), providers=providers)
        self.chunk_size = config.get("chunk_size", 40.0)
        self.memory_num = config.get("memory_num", 1)

    def transcribe(self, audio_file, language=None, context="",
                   start_second=0.0, duration=None, temperature=0.4) -> TranscribeResult:
        audio = audio_mod.decode_audio(audio_file, start_second=start_second,
                                       duration=duration)
        sr = 16000
        spc = int(self.chunk_size * sr)
        n = len(audio)
        chunks = max(1, int(np.ceil(n / spc)))
        memory = []
        texts = []
        items = []
        for i in range(chunks):
            s, e = i * spc, min((i + 1) * spc, n)
            seg = audio[s:e]
            if len(seg) < spc:
                seg = np.pad(seg, (0, spc - len(seg)))
            embd, _ = self.enc.encode(seg)
            prefix = "".join(m for _, m in memory)
            res = self.dec.decode_embeddings(embd, prefix, language=language,
                                             context=context, temperature=temperature,
                                             is_last_chunk=(i == chunks - 1))
            texts.append(res.text)
            if res.text.strip():
                for it in self.align.align(seg, res.text, offset_sec=i * self.chunk_size,
                                           language=language):
                    items.append(it)
            memory.append((embd, res.text))
            memory = memory[-self.memory_num:]
        items.sort(key=lambda x: (x.start, x.end))
        return TranscribeResult(text="".join(texts), alignment=items or None,
                                performance={})
