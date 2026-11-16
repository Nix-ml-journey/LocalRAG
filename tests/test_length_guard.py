"""Offline guard for the final length control (Refiner.fit_length): scripts land near the target, and the length fix can
never re-introduce repeated text. Uses the same stubbed pipeline as test_pipeline_guard.py (see support.py)."""
import itertools
import re
import string

from support import FakeLLM, make_pipeline, script_of, write_script

from ragscript import checks, config

CK = config.load()["checks"]
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
        if kw.get("stage") == "section":
            text = pseudo(len(self.written) * 400, self.section_words)
            self.written.append(text)
            return text
        if kw.get("stage") == "fit":
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
    return script_of(write_script(make_pipeline(tmp_path, llm))["markdown"])


def test_overlong_script_is_trimmed_to_within_10_percent(tmp_path):
    llm = LengthLLM(section_words=150)  # 7 x 150 = 1050 words against a 750 target
    script = run(tmp_path, llm)
    n = checks.spoken_count(script)
    assert 675 <= n <= 825, n
    assert 1 <= llm.fit_calls <= 4
    assert checks.dedupe(script, CK) == script


def test_short_script_moves_toward_target_without_repeats(tmp_path):
    llm = LengthLLM(section_words=60)  # 420 words: more than a rewrite cap (1.8x) can fix
    script = run(tmp_path, llm)
    n = checks.spoken_count(script)
    assert 420 < n <= 750 * 1.1
    assert checks.dedupe(script, CK) == script


def test_length_fix_cannot_reintroduce_repeats(tmp_path):
    llm = LengthLLM(section_words=60, mode="repeat")  # the fit step "expands" by pasting an existing paragraph
    script = run(tmp_path, llm)
    assert llm.fit_calls >= 1
    assert checks.spoken_count(script) == 420, "a rewrite that repeats text must be rejected"
    assert checks.dedupe(script, CK) == script


def test_on_target_script_makes_no_extra_calls(tmp_path):
    llm = LengthLLM(section_words=107)  # 749 words
    run(tmp_path, llm)
    assert llm.fit_calls == 0


def long_words(start, n):
    """n distinct 7-letter words in sentences of 10 (content words, unlike the 3-letter pseudo words)."""
    w = [x + "ings" for x in POOL[start:start + n]]
    return " ".join(". ".join(" ".join(w[i:i + 10]) for i in range(0, n, 10)).split()) + "."


def reverse_sentences(text):
    """The same words in reverse order within each sentence: no shared 4-gram, but the same content words."""
    return ". ".join(" ".join(reversed(s.rstrip(".").split())) for s in text.split(". ")) + "."


class RetoldLLM(LengthLLM):
    """The second section retells the first in different word order, on the retry as well."""

    def chat(self, prompt, **kw):
        if kw.get("stage") == "section":
            if "Previous attempt:" in prompt:
                return self.written[-1]  # the model does not fix it
            k = len(self.written)
            text = reverse_sentences(self.written[0]) if k == 1 else long_words(k * 400, self.section_words)
            self.written.append(text)
            return text
        return super().chat(prompt, **kw)


def test_a_retold_section_is_dropped_before_the_fit_which_makes_up_the_length(tmp_path):
    from ragscript import config
    llm = RetoldLLM(section_words=110)  # 7 x 110 = 770 words; without the retelling 660, below the 675 floor
    md = write_script(make_pipeline(tmp_path, llm))["markdown"]
    script = script_of(md)
    n = checks.spoken_count(script)
    assert 675 <= n <= 825, n  # the fit ran after the repeat guard and made the loss up
    assert llm.fit_calls >= 1
    assert checks.drop_similar(script, "", CK["dedupe_similar"], CK["dedupe_similar_min"]) == script
    assert "Repeat guard removed" in md and "Final repeat guard removed" not in md  # nothing was cut after the fit
