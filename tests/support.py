"""Shared test support: a fake retriever, a fake model that answers by stage, and a factory for an offline pipeline
(no Chroma, no embedder, no Ollama). The real LLM and this fake are the two adapters at the pipeline's model seam."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscript import config
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
    """Answers by stage. Each section re-tells the first scene (slightly reworded), like the real model does."""
    ctx = 16384

    def __init__(self):
        self.written, self.polished = [], 0

    def scale(self):
        return 1.0

    def chat(self, prompt, **kw):
        stage = kw.get("stage")
        if stage == "outline":
            beats = [{"beat": b, "title": b, "words": 107, "summary": f"{b} scene", "place": f"room {i}",
                      "change": f"event number {i} {'alpha beta gamma delta'.split()[i % 4]} {'north south east west'.split()[i % 4]} {i * 7}",
                      "detail": ["hum: stalls", "smell: sharpens", "keys: rattle"][i % 3]} for i, b in enumerate(BEATS)]
            return '{"narrator": "Eli, a night guard", "beats": %s}' % str(beats).replace("'", '"')
        if stage == "section":
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
        if stage == "polish":
            self.polished += 1
            return self.written[self.polished - 1]  # polish keeps the (already deduped) draft's repeats if any survive
        if stage == "fit":  # a model that does not change the section
            return prompt.split("Section to rewrite:\n", 1)[1].strip()
        if stage == "revise":  # a model that does not fix anything
            return prompt.split("Section to revise:\n", 1)[1].split("\n\nFix exactly", 1)[0].strip()
        if stage == "critic":
            return "NONE"
        if stage == "summary":
            return "summary"
        if stage == "package":
            return '{"titles": ["a","b","c","d","e"], "thumbnails": ["x","y","z"], "description": "d", "tags": ["t"]}'
        raise AssertionError("unexpected stage: " + str(stage))


def make_pipeline(tmp_path, llm=None, retriever=None, banned=()):
    """An offline Pipeline that writes its output under tmp_path."""
    cfg = config.load()
    cfg["llm"]["parallel"] = 1
    cfg["paths"]["output_dir"] = str(tmp_path)
    return Pipeline(cfg, retriever or FakeRetriever(), llm or FakeLLM(), banned=banned)


def write_script(p, genre="horror"):
    """Run the whole pipeline on a fixed brief; returns the result dict (path, markdown, stats)."""
    return p.write("a guard keeps seeing himself", genre, 5, "hook text", progress=lambda m: None)


def script_of(markdown):
    """The ## Script section of an output file."""
    return markdown[markdown.index("## Script") + 9: markdown.index("## Chapters")].strip()
