"""对数 Mel 特征提取(纯 numpy/scipy, 算法对齐 Qwen3-ASR 前端)。"""
import numpy as np
import scipy.signal


class FastWhisperMel:
    """mel_filters.npy (201,128) → (128, T) 对数 Mel。"""

    def __init__(self, filter_path: str, dtype=np.float32):
        self.dtype = dtype
        self.filters = np.load(filter_path)  # (201, 128)
        self.n_fft = 400
        self.hop_length = 160
        self.window = scipy.signal.get_window("hann", self.n_fft, fftbins=True)

    def __call__(self, audio: np.ndarray, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = self.dtype
        pad_len = self.n_fft // 2
        y = np.pad(audio, pad_len, mode="reflect")
        num_frames = 1 + (len(y) - self.n_fft) // self.hop_length
        shape = (self.n_fft, num_frames)
        strides = (y.itemsize, self.hop_length * y.itemsize)
        frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
        stft_res = np.fft.rfft(frames * self.window[:, np.newaxis], axis=0)
        magnitudes = np.abs(stft_res) ** 2
        mel_spec = np.dot(self.filters.T, magnitudes)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        n_frames_out = audio.shape[-1] // self.hop_length
        return log_spec[:, :n_frames_out].astype(dtype)


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    """复刻官方 Qwen3 前端: 100 帧块输出 13 帧, 余项逐级降采样。"""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return int(output_lengths)
