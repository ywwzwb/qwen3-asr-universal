#!/usr/bin/env python3
"""Generate resources/mel_filters.npy.

Replicates HuggingFace transformers' mel_filter_bank
(num_frequency_bins=201, num_mel_filters=128, min_frequency=0.0,
max_frequency=8000.0, sampling_rate=16000, norm="slaney", mel_scale="slaney"),
the same filters Qwen3-ASR's WhisperFeatureExtractor produces.

Run: python ci/generate_mel_filters.py
"""
import numpy as np

MIN_LOG_HERTZ = 1000.0
MIN_LOG_MEL = 15.0
H2M_LOGSTEP = 27.0 / np.log(6.4)
M2H_LOGSTEP = np.log(6.4) / 27.0


def _hertz_to_mel_slaney(freq):
    mels = 3.0 * np.asarray(freq) / 200.0
    log_region = np.asarray(freq) >= MIN_LOG_HERTZ
    mels = np.asarray(mels, dtype=np.float64).copy()
    mels[log_region] = MIN_LOG_MEL + np.log(np.asarray(freq, dtype=np.float64)[log_region] / MIN_LOG_HERTZ) * H2M_LOGSTEP
    return mels


def _mel_to_hertz_slaney(mels):
    freq = 200.0 * np.asarray(mels) / 3.0
    log_region = np.asarray(mels) >= MIN_LOG_MEL
    freq = np.asarray(freq, dtype=np.float64).copy()
    freq[log_region] = MIN_LOG_HERTZ * np.exp(M2H_LOGSTEP * (np.asarray(mels, dtype=np.float64)[log_region] - MIN_LOG_MEL))
    return freq


def _create_triangular_filter_bank(fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


def mel_filter_bank(num_frequency_bins, num_mel_filters, min_frequency,
                    max_frequency, sampling_rate, norm="slaney", mel_scale="slaney"):
    mel_min = _hertz_to_mel_slaney(np.asarray(min_frequency))
    mel_max = _hertz_to_mel_slaney(np.asarray(max_frequency))
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
    if norm == "slaney":
        enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        filters *= np.expand_dims(enorm, 0)
    return filters.astype(np.float32)


def main():
    filters = mel_filter_bank(
        num_frequency_bins=400 // 2 + 1,
        num_mel_filters=128,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    assert filters.shape == (201, 128), filters.shape
    assert filters.dtype == np.float32
    assert np.isfinite(filters).all()
    assert (filters >= 0.0).all()
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "resources" / "mel_filters.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, filters)
    print(f"wrote {out} ({filters.nbytes / 1024:.1f} KiB, shape {filters.shape})")


if __name__ == "__main__":
    main()
