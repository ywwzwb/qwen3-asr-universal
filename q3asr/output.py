"""结果导出: JSON(词级时间戳契约)/ TXT / SRT。JSON 不含 ITN。"""
import json
import re

from q3asr.transcription import AlignItem


def export_json(path, items: list[AlignItem]) -> None:
    data = [{"text": it.text, "start": round(it.start, 3), "end": round(it.end, 3)}
            for it in items]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_txt(text: str) -> str:
    out = re.sub(r"([，。？！])", r"\1\n", text)
    return re.sub(r"(?<=[a-zA-Z])([,.] )", r"\1\n", out)


def export_txt(path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_txt(text))


def compose_srt(items: list[AlignItem], max_chars: int = 40) -> str:
    blocks = []
    cur, start = [], None

    def flush(cur, start, end):
        content = "".join(cur).strip().rstrip("，。？！、,.?!")
        if content:
            blocks.append(f"{len(blocks) + 1}\n{_ts(start)} --> {_ts(end)}\n{content}\n")

    split = re.compile(r"[，。？！、\n]|[,.?!]\s*")
    for it in items:
        if start is None:
            start = it.start
        cur.append(it.text)
        if split.search(it.text) or len("".join(cur)) >= max_chars:
            flush(cur, start, it.end)
            cur, start = [], None
    if cur:
        flush(cur, start, items[-1].end)
    return "\n".join(blocks)


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(path, items: list[AlignItem]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(compose_srt(items))
