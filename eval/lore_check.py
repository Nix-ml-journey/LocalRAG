"""Pre-ingest check for lore notes. Usage: python -m eval.lore_check data/history/lore_*.md
Errors (exit code 1): the note would ingest badly. Warnings: look at them, they may be fine."""
import glob
import re
import string
import sys
from collections import Counter
from pathlib import Path

import yaml

from ragscript import loaders

MIN_WORDS, MAX_WORDS = 150, 1500  # ~400-token chunks: under 150 words is one thin chunk, over 1500 mixes topics
ONCE_CHECK_WORDS = 400  # the once-only name/year warning applies from this length
KEYS = ("topic", "source", "license")  # provenance, so a wrong fact can be traced back
MONTHS_DAYS = set("january february march april may june july august september october november december "
                  "monday tuesday wednesday thursday friday saturday sunday".split())
OK_CHARS = set(string.punctuation) | set("“”‘’—–…°·½¼¾§")  # anything else that is not a letter, digit or space counts as odd


def check(text: str, name: str = "lore_x.md") -> dict:
    errors, warnings = [], []
    if not name.lower().startswith("lore"):
        errors.append(f"file name '{name}' does not start with 'lore', so it would be ingested as a story, not lore")
    try:
        fm, body = loaders.split_front_matter(text)
    except yaml.YAMLError:
        fm, body = {}, text
        errors.append("front-matter is not valid YAML")
    else:
        if not fm:
            errors.append("no front-matter: add a --- block with topic, source and license")
        elif missing := [k for k in KEYS if not fm.get(k)]:
            warnings.append("front-matter lacks: " + ", ".join(missing))

    odd = [c for c in body if not (c.isalnum() or c.isspace() or c in OK_CHARS)]
    if "�" in body or re.search(r"â€|Ã.|Â ", body) or len(odd) > 0.01 * max(len(body), 1):
        errors.append(f"garbled text ({len(odd)} odd characters, e.g. {''.join(dict.fromkeys(odd))[:12]!r}): "
                      "bad OCR or encoding, re-extract it")

    n = len(body.split())
    if n < MIN_WORDS:
        errors.append(f"too short: {n} words (minimum {MIN_WORDS})")
    elif n > MAX_WORDS:
        warnings.append(f"long: {n} words (over {MAX_WORDS}); split it into one note per topic")

    # a name or year mentioned exactly once cannot be cross-checked inside the note: typo, OCR slip, or loose detail.
    # Only meaningful in longer notes: in a short one nearly every name appears once, so it would just be noise.
    if n >= ONCE_CHECK_WORDS:
        years = Counter(re.findall(r"\b(?:1\d{3}|20\d{2})\b", body))
        names = Counter()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
            for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", sentence)[1:]:  # skip the capitalised first word
                w = re.sub(r"['’]s$", "", w)
                if w[0].isupper() and len(w) > 2 and not w.isupper() and w.lower() not in MONTHS_DAYS:
                    names[w] += 1
        once = sorted(w for w, c in names.items() if c == 1) + sorted(y for y, c in years.items() if c == 1)
        if once:
            shown = ", ".join(once[:12]) + (f" (+{len(once) - 12} more)" if len(once) > 12 else "")
            warnings.append(f"names/years that appear only once (check each against a second source): {shown}")
    return {"errors": errors, "warnings": warnings, "words": n}


def main(args):
    files = [Path(f) for a in args for f in (glob.glob(a) or [a])]
    if not files:
        sys.exit("usage: python -m eval.lore_check <lore files or globs>")
    failed = 0
    for f in files:
        r = check(f.read_text(encoding="utf-8", errors="replace"), f.name)
        failed += bool(r["errors"])
        print(f"{'FAIL' if r['errors'] else 'ok  '} {f.name} ({r['words']} words)")
        for e in r["errors"]:
            print(f"   ERROR   {e}")
        for w in r["warnings"]:
            print(f"   warning {w}")
    print(f"\n{len(files) - failed}/{len(files)} notes ready to ingest")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
