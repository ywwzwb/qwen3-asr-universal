"""ONNX 音频编码器(Split Frontend + Backend)。"""
import time
import numpy as np
import onnxruntime as ort

from q3asr.features import FastWhisperMel, get_feat_extract_output_lengths


class QwenAudioEncoder:
    def __init__(self, frontend_path, backend_path, mel_filters_path,
                 providers=None, warmup_sec=0.0):
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.sess_fe = ort.InferenceSession(frontend_path, sess_opts, providers=providers)
        self.sess_be = ort.InferenceSession(backend_path, sess_opts, providers=providers)
        self.mel = FastWhisperMel(mel_filters_path)
        inp_type = self.sess_fe.get_inputs()[0].type
        self.input_dtype = np.float16 if "float16" in inp_type else np.float32
        if warmup_sec > 0:
            self.encode(np.random.randn(int(16000 * warmup_sec)).astype(np.float32))

    def _run_frontend(self, mel: np.ndarray) -> np.ndarray:
        T = mel.shape[1]
        pad = (100 - (T % 100)) % 100
        if pad:
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
        mel_input = mel[np.newaxis, ...]
        outs = []
        for i in range(mel_input.shape[2] // 100):
            chunk = mel_input[:, :, i * 100:(i + 1) * 100]
            outs.append(self.sess_fe.run(None, {"chunk_mel": chunk})[0])
        hidden = np.concatenate(outs, axis=1)
        return hidden[:, :get_feat_extract_output_lengths(T), :]

    def encode(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.time()
        mel = self.mel(audio, dtype=self.input_dtype)
        hidden = self._run_frontend(mel)
        B, T, D = hidden.shape
        mask = np.zeros((B, 1, T, T), dtype=self.input_dtype)
        embd = self.sess_be.run(None, {"hidden_states": hidden, "attention_mask": mask})[0]
        if embd.ndim == 3:
            embd = embd[0]
        return embd, time.time() - t0
