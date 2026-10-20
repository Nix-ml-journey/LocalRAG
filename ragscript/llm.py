"""Ollama client with OOM -> halve num_ctx -> retry."""
import logging
import re

import ollama

from . import vram
from .chunking import tokens

log = logging.getLogger("ragscript")
# Ollama's real OOM messages; broader words ("memory", "resource") also match unrelated errors
# and would permanently halve num_ctx for nothing.
_OOM = ("out of memory", "requires more", "cudamalloc", "failed to allocate", "unable to allocate")


class LLM:
    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.c = c
        self.client = ollama.Client(host=c["host"])
        self.ctx = c["num_ctx"]
        self._think_ok = True

    def scale(self) -> float:
        """How much smaller than configured the context is now (shrink retrieval accordingly)."""
        return self.ctx / self.c["num_ctx"]

    def chat(self, prompt: str, think: bool = False, temperature=None, max_tokens=None) -> str:
        while True:
            opts = {"num_ctx": self.ctx,
                    "temperature": self.c["temperature"] if temperature is None else temperature,
                    "num_predict": max_tokens or self.ctx // 2,
                    "repeat_penalty": self.c["repeat_penalty"]}
            if (n_in := tokens(prompt)) > self.ctx * 0.75:
                log.warning("prompt ~%d tokens is close to num_ctx=%d; output may be truncated", n_in, self.ctx)
            kw = {"think": think} if self._think_ok else {}
            try:
                r = self.client.chat(model=self.c["model"], messages=[{"role": "user", "content": prompt}],
                                     options=opts, keep_alive=self.c["keep_alive"], **kw)
                vram.log_stage("llm")
                # prompt_eval_count counts only tokens NOT served from Ollama's prefix cache
                out = getattr(r, "eval_count", 0) or 0
                log.info("llm %-30r %5.1fs  in(new)=%s out=%s  %.0f tok/s", prompt[:30],
                         (getattr(r, "total_duration", 0) or 0) / 1e9, getattr(r, "prompt_eval_count", "?"), out,
                         out / max((getattr(r, "eval_duration", 0) or 1) / 1e9, 1e-9))
                return re.sub(r"<think>.*?</think>", "", r["message"]["content"], flags=re.S).strip()
            except Exception as e:
                msg = str(e).lower()
                if "think" in msg and self._think_ok:  # model has no thinking support (e.g. mistral-nemo)
                    self._think_ok = False
                    continue
                if any(k in msg for k in _OOM) and self.ctx // 2 >= self.c["min_ctx"]:
                    self.ctx //= 2
                    log.warning("LLM error (%s); retrying with num_ctx=%d", e, self.ctx)
                    continue
                raise
