# q3asr/decoder.py
"""GGUF decoder 推理(llama-cpp-python 底层 API, 喂入 embedding 序列)。

针对 llama-cpp-python 0.3.34 的低层 API(已探针验证):
- 0.3.34 把 vocab 从 model 分离: llama_tokenize / llama_token_to_piece / llama_token_eos
  都收 llama_vocab_p(由 llama_model_get_vocab(model) 取得)。
- 没有 llama_batch_set_embd / llama_batch_add / llama_model_eos / llama_model_embd_size;
  改用直接填 llama_batch 结构体 + llama_decode。
- token 嵌入表来自 GGUF token_embd.weight, 量化张量用 gguf.quants.dequantize 按需反量化。
"""
import ctypes
import dataclasses

import numpy as np
import llama_cpp as lc
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize, GGML_QUANT_SIZES


@dataclasses.dataclass
class DecodeResult:
    text: str = ""
    n_prefill: int = 0
    n_generate: int = 0
    is_aborted: bool = False


class _TokenEmbeddingTable:
    """GGUF token_embd.weight, 按 token 反量化。"""

    def __init__(self, gguf_path):
        reader = GGUFReader(str(gguf_path))
        tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
        self.qtype = GGMLQuantizationType(tensor.tensor_type)
        n_embd = tensor.shape[0]
        n_vocab = tensor.shape[1]
        if self.qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            self._raw = tensor.data.reshape(n_vocab, n_embd)
            self._float = True
        else:
            bs, ts = GGML_QUANT_SIZES[self.qtype]
            bpr = (n_embd // bs) * ts
            self._raw = tensor.data.reshape(n_vocab, bpr)
            self._float = False

    def __call__(self, ids):
        if self._float:
            return np.ascontiguousarray(self._raw[list(ids)].astype(np.float32))
        return np.ascontiguousarray(dequantize(self._raw[list(ids)], self.qtype.value),
                                    dtype=np.float32)


def _int_arr(n, vals):
    return (lc.llama_token * n)(*vals)


class ASRDecoder:
    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_batch: int = 4096):
        lc.llama_backend_init()
        self.model = lc.llama_model_load_from_file(str(gguf_path).encode("utf-8"),
                                                   lc.llama_model_default_params())
        if not self.model:
            raise RuntimeError(f"failed to load GGUF model: {gguf_path}")
        self.vocab = lc.llama_model_get_vocab(self.model)
        self.n_embd = lc.llama_model_n_embd(self.model)
        cparams = lc.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_batch
        self.ctx = lc.llama_new_context_with_model(self.model, cparams)
        if not self.ctx:
            raise RuntimeError("failed to create context")
        self.emb_tbl = _TokenEmbeddingTable(gguf_path)
        self.specials = {}
        for key, text in (("im_start", "<|im_start|>"), ("im_end", "<|im_end|>"),
                          ("audio_start", "<|audio_start|>"), ("audio_end", "<|audio_end|>"),
                          ("asr_text", "<asr_text>")):
            self.specials[key] = self.tokenize(text)[0]
        self.specials["eos"] = lc.llama_token_eos(self.vocab)

    def special_ids(self) -> dict:
        return dict(self.specials)

    def tokenize(self, text: str) -> list[int]:
        b = text.encode("utf-8")
        buf = (lc.llama_token * 4096)()
        n = lc.llama_tokenize(self.vocab, b, len(b), buf, 4096, False, True)
        return list(buf[:n])

    def token_embeddings(self, ids: list[int]) -> np.ndarray:
        return self.emb_tbl(ids)

    def _detok(self, token: int) -> str:
        buf = ctypes.create_string_buffer(64)
        m = lc.llama_token_to_piece(self.vocab, token, buf, 64, 0, True)
        return buf.raw[:m].decode("utf-8", errors="replace") if m > 0 else ""

    def _new_chain(self, temperature, seed):
        chain = lc.llama_sampler_chain_init(lc.llama_sampler_chain_default_params())
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_temp(temperature))
        lc.llama_sampler_chain_add(chain, lc.llama_sampler_init_dist(seed))
        return chain

    def decode_embeddings(self, embd, prefix_text, language=None, context="",
                          temperature=0.4, rollback_num=5,
                          is_last_chunk=False, max_new_tokens=512) -> DecodeResult:
        lc.llama_memory_clear(lc.llama_get_memory(self.ctx), True)
        sp = self.specials
        pre = ([sp["im_start"]] + self.tokenize(f"system\n{context or 'You are a helpful assistant.'}")
               + [sp["im_end"], sp["im_start"]] + self.tokenize("user\n") + [sp["audio_start"]])
        head = "assistant\n"
        if language:
            head += f"language {language}"
        suf = [sp["audio_end"], sp["im_end"], sp["im_start"]] + self.tokenize(head) \
            + [sp["asr_text"]] + self.tokenize(prefix_text)
        full = np.concatenate([self.token_embeddings(pre), embd, self.token_embeddings(suf)], axis=0)
        n = full.shape[0]

        # prefill: embedding batch
        # M-RoPE (qwen3vl arch) requires 4 position ids per embedding token:
        # [pos, pos, pos, 0]. llama.cpp copies n_tokens*n_pos_per_embd positions
        # directly from batch.pos for embedding (no-token) batches, so we must
        # provide all 4n; non-mrope models read only the first n (harmless).
        batch = lc.llama_batch_init(4 * n, 1, 1)
        batch.n_tokens = n
        pos_arr = np.concatenate([np.arange(n, dtype=np.int32),
                                  np.arange(n, dtype=np.int32),
                                  np.arange(n, dtype=np.int32),
                                  np.zeros(n, dtype=np.int32)])
        batch.pos = ctypes.cast(_int_arr(4 * n, pos_arr), type(batch.pos))
        batch.n_seq_id = ctypes.cast(_int_arr(n, [1] * n), type(batch.n_seq_id))
        seq_arr = (ctypes.POINTER(lc.llama_token) * n)(
            *(ctypes.cast(_int_arr(1, [0]), ctypes.POINTER(lc.llama_token)) for _ in range(n)))
        batch.seq_id = ctypes.cast(seq_arr, type(batch.seq_id))
        logits = (ctypes.c_byte * n)(*([0] * (n - 1) + [1]))
        batch.logits = ctypes.cast(logits, type(batch.logits))
        batch.embd = ctypes.cast(full.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                 type(batch.embd))
        if lc.llama_decode(self.ctx, batch) != 0:
            raise RuntimeError("prefill decode failed")

        # generation
        chain = self._new_chain(temperature, int(np.random.randint(0, 2 ** 31 - 1)))
        text_parts = []
        stable = []
        cur = n
        for _ in range(max_new_tokens):
            token = lc.llama_sampler_sample(chain, self.ctx, -1)
            if token in (sp["eos"], sp["im_end"]):
                break
            gb = lc.llama_batch_init(1, 0, 1)
            gb.n_tokens = 1
            gb.token = ctypes.cast(_int_arr(1, [token]), type(gb.token))
            gb.pos = ctypes.cast(_int_arr(1, [cur]), type(gb.pos))
            gb.n_seq_id = ctypes.cast(_int_arr(1, [1]), type(gb.n_seq_id))
            gseq = (ctypes.POINTER(lc.llama_token) * 1)(
                ctypes.cast(_int_arr(1, [0]), ctypes.POINTER(lc.llama_token)))
            gb.seq_id = ctypes.cast(gseq, type(gb.seq_id))
            glog = (ctypes.c_byte * 1)(1)
            gb.logits = ctypes.cast(glog, type(gb.logits))
            lc.llama_decode(self.ctx, gb)
            stable.append(token)
            if len(stable) > rollback_num:
                text_parts.append(self._detok(stable.pop(0)))
            cur += 1
            if len(stable) > 15 and len(set(stable[-15:])) <= 3:
                return DecodeResult("".join(text_parts), n, len(stable), is_aborted=True)
        if is_last_chunk:
            while stable:
                text_parts.append(self._detok(stable.pop(0)))
        return DecodeResult("".join(text_parts), n, len(stable), is_aborted=False)
