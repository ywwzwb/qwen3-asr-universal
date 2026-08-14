"""q3asr CLI — 兼容 skill run_transcribe.py 的调用契约。"""
import argparse
import os
import sys
from pathlib import Path

from q3asr import __version__, backend as backend_mod, models as models_mod


def build_parser():
    p = argparse.ArgumentParser(prog="q3asr", description=__doc__)
    p.add_argument("input", nargs="?", help="audio file (mp3/wav/...)")
    p.add_argument("--version", action="store_true")
    p.add_argument("--seek-start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("-l", "--language", default=None)
    p.add_argument("--prec", default="int4")
    p.add_argument("--device", default=os.environ.get("QASR_DEVICE", "auto"))
    p.add_argument("--n-ctx", type=int, default=2048, help="LLM context window (KV cache size; 2048 halves GPU memory vs 4096)")
    p.add_argument("--model", default="1.7b")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--no-dml", action="store_true")
    p.add_argument("--no-vulkan", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--download-models-only", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"q3asr {__version__}")
        return 0
    try:
        device = backend_mod.detect_backend(args.device)
        print(f"[INFO] backend: {device}")
        missing = backend_mod.runtime_missing(device)
        if missing and args.device != "auto":
            print(f"[ERROR] device '{device}' requested but required runtime libraries "
                  f"are missing: {', '.join(missing)}", file=sys.stderr)
            print("[ERROR] CUDA builds need NVIDIA CUDA runtime (libcudart.so.12, "
                  "libcublas.so.12); install via 'pip install nvidia-cudart-cu12 "
                  "nvidia-cublas-cu12' or the system CUDA toolkit.", file=sys.stderr)
            return 1
        if missing:
            print(f"[WARN] {device} runtime libraries missing ({', '.join(missing)}); "
                  "falling back to CPU", file=sys.stderr)
            device = "cpu"
        if args.download_models_only:
            models_mod.ensure_models(model=args.model)
            print("[INFO] models ready")
            return 0
        if not args.input:
            build_parser().error("the following arguments are required: input")
            return 2
        from q3asr.engine import TranscribeEngine
        from q3asr import output
        paths = models_mod.resolve_paths(
            model=args.model, model_dir=Path(args.model_dir) if args.model_dir else None)
        eng = TranscribeEngine({"paths": paths, "device": device, "n_ctx": args.n_ctx})
        res = eng.transcribe(args.input, language=args.language,
                             start_second=args.seek_start, duration=args.duration)
        base = Path(args.input).with_suffix("")
        output.export_json(f"{base}.json", res.alignment or [])
        if not args.json_only:
            output.export_txt(f"{base}.txt", res.text)
            output.export_srt(f"{base}.srt", res.alignment or [])
        print(f"[INFO] done: {base}.json")
        return 0
    except models_mod.DownloadError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
