"""q3asr CLI — 兼容 skill run_transcribe.py 的调用契约。"""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(prog="q3asr", description=__doc__)
    p.add_argument("input", nargs="?", help="audio file (mp3/wav/...)")
    p.add_argument("--version", action="store_true", help="print version and exit")
    p.add_argument("--seek-start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("-l", "--language", default=None)
    p.add_argument("--prec", default="int4")
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default="1.7b")
    p.add_argument("--model-dir", default=None)
    p.add_argument("--no-dml", action="store_true")
    p.add_argument("--no-vulkan", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--download-models-only", action="store_true")
    return p


def main(argv=None) -> int:
    from q3asr import __version__
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"q3asr {__version__}")
        return 0
    if args.input is None and not args.download_models_only:
        build_parser().error("the following arguments are required: input")
        return 2
    print(f"q3asr {__version__}: input={args.input} device={args.device} model={args.model}")
    return 0  # 后续任务替换为真实管线


if __name__ == "__main__":
    sys.exit(main())
