"""Quick regression run: 3 fixed 5-minute briefs (history, cthulhu, horror) -> output/<label>/ + a stats table.
Usage: python -m eval.rerun3 <label>      (about 13 minutes with gemma3:12b)"""
import json
import shutil
import sys

from ragscript import config
from ragscript.pipeline import Pipeline

BRIEFS = [
    ("history", "The Mary Celeste: what was found aboard the abandoned ship in 1872"),
    ("cthulhu", "A deep-sea survey crew hears a sound from the trench that matches an old sailor's song"),
    ("horror", "A night-shift security guard keeps seeing himself on the monitors a few seconds before he walks into the room"),
]


def main(label):
    dest = config.ROOT / "output" / label
    dest.mkdir(parents=True, exist_ok=True)
    pl, rows = Pipeline(), []
    for genre, brief in BRIEFS:
        r = pl.write(brief, genre, 5, pl.hooks(brief, genre)[0], progress=lambda m: None)
        shutil.copy(r["path"], dest / r["path"].name)
        s = r["stats"]
        rows.append(dict(genre=genre, words=s["words"], accuracy=s["word_accuracy"], repeated=s["repeated_phrase_rate"],
                         avg_sentence=s["avg_sentence_words"], invented=len(s["invented"]), uncited=len(s["uncited_facts"])))
        print(rows[-1], flush=True)
    (dest / "stats.json").write_text(json.dumps(rows, indent=1))
    print(f"\nsaved to {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rerun")
