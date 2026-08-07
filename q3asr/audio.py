"""任意音频 → 16k 单声道 float32 [-1,1]。用 imageio-ffmpeg 自带的静态 ffmpeg, 免系统依赖。"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import imageio_ffmpeg


def decode_audio(path: str, sample_rate: int = 16000,
                 start_second: float = 0.0, duration: float | None = None) -> np.ndarray:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(path),
           "-ss", f"{start_second:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()
