"""强制对齐: 文本 + 音频 → 词级时间戳(绝对秒)。

移植 v0.1 QwenForcedAligner 机制(自行重写, 算法对齐 v0.1 aligner.py):
- 文本分词: CJK 逐字, 其余按空格(通用分词, 日/韩退化为逐字);
- prompt: <|audio_start|> + 音频 embd + <|audio_end|> + 词 tokens 间插 <timestamp>(每词 2 个: 起始+结束);
- 单次前向, 在 <timestamp> 位置取 logits, argmax(logits[:4000]) → 时间戳索引 × STEP_MS(80ms);
- LIS 单调修正(fix_timestamps) + reconcile 找回标点/空格。
"""
import dataclasses
import unicodedata

import numpy as np

from q3asr.encoder import QwenAudioEncoder
from q3asr.decoder import ASRDecoder


@dataclasses.dataclass
class AlignItem:
    text: str
    start: float
    end: float


class _AlignerProcessor:
    """分词 + LIS 单调修正 + 标点找回(算法对齐 v0.1 AlignerProcessor)。"""

    @staticmethod
    def _is_kept_char(ch):
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    @staticmethod
    def _is_cjk_char(ch):
        c = ord(ch)
        return (0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or
                0x20000 <= c <= 0x2A6DF or 0x2A700 <= c <= 0x2B73F or
                0x2B740 <= c <= 0x2B81F or 0x2B820 <= c <= 0x2CEAF or
                0xF900 <= c <= 0xFAFF)

    def tokenize(self, text, language=None):
        tokens = []
        for seg in text.split():
            cleaned = "".join(c for c in seg if self._is_kept_char(c))
            if not cleaned:
                continue
            buf = []
            for ch in cleaned:
                if self._is_cjk_char(ch):
                    if buf:
                        tokens.append("".join(buf)); buf = []
                    tokens.append(ch)
                else:
                    buf.append(ch)
            if buf:
                tokens.append("".join(buf))
        return tokens

    def fix_timestamps(self, data):
        """LIS 最长递增子序列单调化: 剔除/插值异常时间戳(v0.1 思路)。"""
        data = list(data)
        n = len(data)
        if n == 0:
            return []
        dp, parent = [1] * n, [-1] * n
        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1; parent[i] = j
        max_idx = dp.index(max(dp))
        lis = []
        idx = max_idx
        while idx != -1:
            lis.append(idx); idx = parent[idx]
        lis.reverse()
        normal = [False] * n
        for i in lis:
            normal[i] = True
        result = data[:]
        i = 0
        while i < n:
            if not normal[i]:
                j = i
                while j < n and not normal[j]:
                    j += 1
                cnt = j - i
                left = next((result[k] for k in range(i - 1, -1, -1) if normal[k]), None)
                right = next((result[k] for k in range(j, n) if normal[k]), None)
                if cnt <= 2:
                    for k in range(i, j):
                        if left is None:
                            result[k] = right
                        elif right is None:
                            result[k] = left
                        else:
                            result[k] = left if (k - i + 1) <= (j - k) else right
                else:
                    if left is not None and right is not None:
                        step = (right - left) / (cnt + 1)
                        for k in range(i, j):
                            result[k] = int(left + step * (k - i + 1))
                    elif left is not None:
                        result[i:j] = [left] * cnt
                    elif right is not None:
                        result[i:j] = [right] * cnt
                i = j
            else:
                i += 1
        return [int(v) for v in result]

    def reconcile(self, original_text, items):
        """把标点/空格找回, 时间戳贴近相邻词(v0.1 思路)。"""
        if not items:
            return [AlignItem(original_text, 0.0, 0.0)] if original_text else []
        out = []
        curr = 0
        last_ts = items[0].start
        for it in items:
            sp, ep = self._find(original_text, it.text, curr)
            if sp != -1:
                if sp > curr:
                    out.append(AlignItem(original_text[curr:sp], last_ts, last_ts))
                out.append(AlignItem(original_text[sp:ep], it.start, it.end))
                curr = ep
                last_ts = it.end
            else:
                out.append(it)
                last_ts = it.end
        if curr < len(original_text):
            out.append(AlignItem(original_text[curr:], last_ts, last_ts))
        return out

    def _find(self, text, target, start_index):
        if not target:
            return -1, -1
        t_ptr = 0
        first = -1
        i = start_index
        while i < len(text):
            ch = text[i]
            if ch == target[t_ptr]:
                if t_ptr == 0:
                    first = i
                t_ptr += 1
                if t_ptr == len(target):
                    return first, i + 1
            elif self._is_kept_char(ch):
                if first != -1:
                    i = first
                    first = -1
                    t_ptr = 0
            i += 1
        return -1, -1


class Aligner:
    STEP_MS = 80.0

    def __init__(self, frontend_path, backend_path, mel_filters_path,
                 llm_gguf, providers=None, n_ctx=4096, device="cpu"):
        self.enc = QwenAudioEncoder(frontend_path, backend_path, mel_filters_path,
                                    providers=providers, warmup_sec=0.0)
        self.dec = ASRDecoder(llm_gguf, n_ctx=n_ctx, device=device)
        self.proc = _AlignerProcessor()
        self._ts_id = None

    def _timestamp_id(self):
        if self._ts_id is None:
            ids = self.dec.tokenize("<timestamp>")
            if not ids:
                raise RuntimeError("<timestamp> token not found in aligner vocab")
            self._ts_id = ids[0]
        return self._ts_id

    def align(self, audio_slice, text, offset_sec, language=None) -> list[AlignItem]:
        audio_embd, _ = self.enc.encode(audio_slice)
        words = self.proc.tokenize(text, language)
        if not words:
            return [AlignItem(text, offset_sec, offset_sec)] if text else []

        sp = self.dec.special_ids()
        ts = self._timestamp_id()
        pre_ids = [sp["audio_start"]]
        post_ids = [sp["audio_end"]]
        ts_positions = []
        prefix_len = len(pre_ids) + audio_embd.shape[0] + len(post_ids)
        post_len = 0
        for word in words:
            wt = self.dec.tokenize(word)
            post_ids.extend(wt); post_len += len(wt)
            ts_positions.append(prefix_len + post_len); post_ids.append(ts); post_len += 1
            ts_positions.append(prefix_len + post_len); post_ids.append(ts); post_len += 1

        full = np.concatenate([self.dec.token_embeddings(pre_ids), audio_embd,
                               self.dec.token_embeddings(post_ids)], axis=0)
        logits = self.dec.prefill_logits(full, ts_positions)  # (K, n_vocab)
        raw = [int(np.argmax(row[:min(4000, len(row))])) for row in logits]
        fixed = self.proc.fix_timestamps(raw)
        ms = np.asarray(fixed) * self.STEP_MS / 1000.0
        items = []
        for i, w in enumerate(words):
            items.append(AlignItem(w, offset_sec + float(ms[2 * i]),
                                   offset_sec + float(ms[2 * i + 1])))
        return self.proc.reconcile(text, items)
