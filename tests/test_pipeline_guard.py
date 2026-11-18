"""Offline end-to-end guard: a fake LLM that re-tells earlier scenes must not leak repeated text into the script.
Runs Pipeline.write() with stubbed retrieval and model (no GPU, no Ollama, ~1 s): python -m pytest tests"""
import re

from support import make_pipeline, write_script

from ragscript import checks, config
from ragscript.report import script_of

CK = config.load()["checks"]


def run(tmp_path):
    return script_of(write_script(make_pipeline(tmp_path))["markdown"])


def test_repeated_scenes_do_not_survive(tmp_path):
    script = run(tmp_path)
    assert checks.drop_repeats(script, "", CK["dedupe_ngram"], CK["dedupe_paragraph"]) == script, \
        "a repeated paragraph reached the final script"
    assert checks.drop_repeat_sentences(script, "", CK["dedupe_ngram"], CK["dedupe_sentence"],
                                        CK["dedupe_min_words"]) == script, "a repeated sentence reached the final script"
    assert script.lower().count("pallet out of place") <= 1
    assert checks.repeated_rate(script, 4) < 0.08
    assert len(re.findall(r"I counted \d+ slow breaths", script)) == 7  # the fresh scenes were all kept


def test_guard_has_teeth(tmp_path, monkeypatch):
    """With the dedupe switched off the same fake model must produce visible repeats; otherwise the guard proves nothing."""
    monkeypatch.setattr(checks, "drop_repeats", lambda text, *a, **k: text)
    monkeypatch.setattr(checks, "drop_similar", lambda text, *a, **k: text)
    monkeypatch.setattr(checks, "drop_repeat_sentences", lambda text, *a, **k: text)
    script = run(tmp_path)
    assert script.lower().count("pallet out of place") > 1
    assert checks.repeated_rate(script, 4) > 0.2


def test_clean_strips_invented_brackets_but_keeps_cues():
    from ragscript import refine
    t = "I sharpened my quill [recurring]. [PAUSE] Wind rose [SFX: wind] over the fort [S2] [B-ROLL: the fort]."
    out = refine.clean_draft(t)
    assert "[recurring]" not in out
    assert all(c in out for c in ("[PAUSE]", "[SFX: wind]", "[S2]", "[B-ROLL: the fort]"))
