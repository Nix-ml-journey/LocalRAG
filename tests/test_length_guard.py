"""Offline guard for the final length control (Pipeline._fit_length): scripts land near the target, and the
length fix can never re-introduce repeated text. Uses the same stubbed pipeline as test_pipeline_guard.py."""
import itertools
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscript import checks, config
from ragscript.pipeline import Pipeline
from test_pipeline_guard import FakeLLM, FakeRetriever

POOL = ["".join(t) for t in itertools.product(string.ascii_lowercase, repeat=3)]  # 17,576 distinct "words"


def pseudo(start, n):
    """n distinct words in sentences of 10, so no 4-word phrase ever repeats."""
    w = POOL[start:start + n]
    return " ".join(". ".join(" ".join(w[i:i + 10]) for i in range(0, n, 10)).split()) + "."


class LengthLLM(FakeLLM):
    """Sections of `section_words` distinct words; the fit prompt either honours the target or (mode='repeat')
    pads with a paragraph that is already in the script."""

    def __init__(self, section_words, mode="honest"):
        super().__init__()
        self.section_words, self.mode, self.fit_calls, self.counter = section_words, mode, 0, 0

    def chat(self, prompt, **kw):
        if prompt.startswith("You are writing one section"):
            text = pseudo(len(self.written) * 400, self.section_words)
            self.written.append(text)
            return text
        if prompt.startswith("You are a length editor"):
            self.fit_calls += 1
            target = int(re.search(r"to about (\d+) words", prompt).group(1))
            text = prompt.split("Section to rewrite:\n", 1)[1].strip()
            w = text.split()
            if self.mode == "repeat":
                return text + "\n\n" + self.written[0]  # pads with text already in the script
            if target < len(w):
                return " ".join(w[:target])
            self.counter += 1
            return text + " " + pseudo(10000 + self.counter * 400, target - len(w))
        return super().chat(prompt, **kw)


def run(tmp_path, llm):
    p = Pipeline.__new__(Pipeline)
    p.cfg = config.load()
    p.cfg["llm"]["parallel"] = 1
    p.cfg["paths"]["output_dir"] = str(tmp_path)
    p.ret, p.llm, p.banned = FakeRetriever(), llm, []
    md = p.write("a guard keeps seeing himself", "horror", 5, "hook text", progress=lambda m: None)["markdown"]
    return md[md.index("## Script") + 9: md.index("## Chapters")].strip()


def test_overlong_script_is_trimmed_to_within_10_percent(tmp_path):
    llm = LengthLLM(section_words=150)  # 7 x 150 = 1050 words against a 750 target
    script = run(tmp_path, llm)
    n = checks.spoken_count(script)
    assert 675 <= n <= 825, n
    assert 1 <= llm.fit_calls <= 4
    assert checks.dedupe(script) == script


def test_short_script_moves_toward_target_without_repeats(tmp_path):
    llm = LengthLLM(section_words=60)  # 420 words: more than a rewrite cap (1.8x) can fix
    script = run(tmp_path, llm)
    n = checks.spoken_count(script)
    assert 420 < n <= 750 * 1.1
    assert checks.dedupe(script) == script


def test_length_fix_cannot_reintroduce_repeats(tmp_path):
    llm = LengthLLM(section_words=60, mode="repeat")  # the fit step "expands" by pasting an existing paragraph
    script = run(tmp_path, llm)
    assert llm.fit_calls >= 1
    assert checks.spoken_count(script) == 420, "a rewrite that repeats text must be rejected"
    assert checks.dedupe(script) == script


def test_on_target_script_makes_no_extra_calls(tmp_path):
    llm = LengthLLM(section_words=107)  # 749 words
    run(tmp_path, llm)
    assert llm.fit_calls == 0
