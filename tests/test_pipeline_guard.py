"""Offline end-to-end guard: a fake LLM that re-tells earlier scenes must not leak repeated text into the script.
Runs Pipeline.write() with stubbed retrieval and model (no GPU, no Ollama, ~1 s): python -m pytest tests"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscript import checks, config
from ragscript.pipeline import Pipeline

BEATS = ["hook", "setup", "rising_tension", "reveal", "climax", "aftermath", "outro"]
FRESH = [  # seven different scenes; I counted N slow breaths is the marker the test checks for
    "The boiler room pipes began to hum at two. I counted 7 slow breaths while a long shadow slid past the gauges.",
    "A cold draft came off the loading dock. I counted 8 slow breaths and heard a radio whisper my own name.",
    "My flashlight died inside the records vault. I counted 9 slow breaths before the cabinets started breathing back.",
    "On the roof the antenna sang a single wrong note. I counted 10 slow breaths watching the city lights flicker out.",
    "The elevator opened onto a floor that does not exist. I counted 11 slow breaths and stepped into the dark carpet.",
    "Morning found me asleep in the stairwell with salt on my lips. I counted 12 slow breaths and checked every lock.",
    "In the parking deck my car sat running, empty, headlights pointing at me. I counted 13 slow breaths and walked away.",
]


class FakeRetriever:
    def search(self, *a, **k):
        return []


class FakeLLM:
    """Answers by prompt type. Each section re-tells the first scene (slightly reworded), like the real model does."""
    ctx = 16384

    def __init__(self):
        self.written, self.polished = [], 0

    def scale(self):
        return 1.0

    def chat(self, prompt, **kw):
        if kw.get("stage") == "outline":
            beats = [{"beat": b, "title": b, "words": 107, "summary": f"{b} scene"} for b in BEATS]
            return '{"narrator": "Eli, a night guard", "beats": %s}' % str(beats).replace("'", '"')
        if kw.get("stage") == "section":
            if "Previous attempt:" in prompt:  # corrective retry: the real model repeats itself again
                return self.written[-1]
            k = len(self.written)
            fresh = FRESH[k]
            retold = ("Then I saw it. A pallet out of place. Just slightly. Like someone moved it. [PAUSE] "
                      "The freezer door was cracked open, more than I remember. It was locked before.")
            text = fresh if k == 0 else f"{fresh}\n\n{retold}\n\n[SFX: faint scraping] {retold}"
            if k == 0:
                text = f"{fresh}\n\n{retold}"
            self.written.append(text)
            return text
        if kw.get("stage") == "polish":
            self.polished += 1
            return self.written[self.polished - 1]  # polish keeps the (already deduped) draft's repeats if any survive
        if kw.get("stage") == "fit":  # a model that does not change the section
            return prompt.split("Section to rewrite:\n", 1)[1].strip()
        if kw.get("stage") == "revise":  # a model that does not fix anything
            return prompt.split("Section to revise:\n", 1)[1].split("\n\nFix exactly", 1)[0].strip()
        if kw.get("stage") == "summary":
            return "summary"
        if kw.get("stage") == "package":
            return '{"titles": ["a","b","c","d","e"], "thumbnails": ["x","y","z"], "description": "d", "tags": ["t"]}'
        raise AssertionError("unexpected prompt: " + prompt[:40])


def run(tmp_path):
    cfg = config.load()
    cfg["llm"]["parallel"] = 1
    cfg["paths"]["output_dir"] = str(tmp_path)
    p = Pipeline(cfg, FakeRetriever(), FakeLLM())  # no Chroma, no embedder, no Ollama
    md = p.write("a guard keeps seeing himself", "horror", 5, "hook text", progress=lambda m: None)["markdown"]
    return md[md.index("## Script") + 9: md.index("## Chapters")].strip()


def test_repeated_scenes_do_not_survive(tmp_path):
    script = run(tmp_path)
    assert checks.drop_repeats(script) == script, "a repeated paragraph reached the final script"
    assert checks.drop_repeat_sentences(script) == script, "a repeated sentence reached the final script"
    assert script.lower().count("pallet out of place") <= 1
    assert checks.repeated_rate(script, 4) < 0.08
    assert len(re.findall(r"I counted \d+ slow breaths", script)) == 7  # the fresh scenes were all kept


def test_guard_has_teeth(tmp_path, monkeypatch):
    """With the dedupe switched off the same fake model must produce visible repeats; otherwise the guard proves nothing."""
    monkeypatch.setattr(checks, "drop_repeats", lambda text, *a, **k: text)
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
