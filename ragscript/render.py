"""Markdown output for one finished script."""
from dataclasses import dataclass

from .grounding import clean_narration


@dataclass
class Doc:
    """What the markdown shows besides the script and its Checks."""
    brief: str
    genre: str
    minutes: int
    narrator: str
    pkg: dict       # titles, thumbnails, description, tags from the packaging step
    chapters: list  # [(start second, title)]
    sources: dict   # {source file: [S numbers]}, style-only sources map to []


def _ts(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"


def chapters(sections, total_words, wpm):
    """[(start second, title)]: each beat's share of the finished length `total_words`, spoken at `wpm`.
    sections: [(beat, words drafted)], the drafted lengths only give the proportions."""
    scale = total_words / sum(w or 1 for _, w in sections)
    out, acc = [], 0
    for b, w in sections:
        out.append((int(acc * scale / wpm * 60), b.get("title") or b["beat"]))
        acc += w or 1
    return out


def render(doc: Doc, script, report) -> str:
    pkg, r = doc.pkg, report
    titles = pkg.get("titles", [])
    clean = clean_narration(script)
    L = [f"# {titles[0] if titles else doc.brief}", "",
         f"*Genre: {doc.genre} · Target: {doc.minutes} min · {r.words} words (target {r.target_words})*", "",
         "## Title options", *[f"{i}. {t}" for i, t in enumerate(titles, 1)], "",
         "## Thumbnail text", *[f"- {t}" for t in pkg.get("thumbnails", [])], "",
         "## Narrator", str(doc.narrator), "",
         # the main script is the clean copy (no [S#] tags, no INVENTED braces): it reads and records as is
         "## Script", "", clean, "",
         "## Chapters (estimated)", *[f"- {_ts(s)} {t}" for s, t in doc.chapters], "",
         "## Description", pkg.get("description", ""), "",
         "## Tags", ", ".join(pkg.get("tags", [])), "",
         "## Sources used", *[f"- {src}" + (" (as " + ", ".join(f"S{n}" for n in ns) + ")" if ns else " (style)") for src, ns in doc.sources.items()], "",
         "## Checks",
         f"- Word accuracy: {r.word_accuracy:.1%}",
         f"- Repeated-phrase rate: {r.repeated_phrase_rate:.1%}",
         f"- Average sentence length: {r.avg_sentence_words} words (natural narration is about {r.sentence_range})"]
    if r.overused_words:
        L.append("- Words used often (crutch words, consider varying): " + ", ".join(f"{w} ×{n}" for w, n in r.overused_words))
    if r.repeated_images:
        L.append("- Images returning through the script (consider cutting repeats): " + ", ".join(f"{p} ×{n}" for p, n in r.repeated_images))
    if r.mixed_tense:
        L.append("- **Tense switches between past and present in the narration (check before recording).**")
    if r.pause_cues > 2 * r.minutes:
        L.append(f"- {r.pause_cues} [PAUSE] cues in {r.minutes} minutes: more than 2 a minute reads as halting")
    if r.overlap_flags:
        L += ["- **>80% overlap with a source (rewrite these):**"] + [f"  - ({x}) {p}…" for p, x in r.overlap_flags]
    else:
        L.append("- No passage over the overlap threshold")
    if r.banned_hits:
        L.append("- Banned phrases still present: " + "; ".join(r.banned_hits))
    if r.invented:
        L += ["- **Invented details to fact-check:**"] + [f"  - {x}" for x in r.invented]
    if r.unsupported_cites:
        L += ["- **Cited [S#] but numbers, quotes or names not found in that lore chunk (verify or delete):**"]
        L += [f"  - {s} → missing: {', '.join(m)}" for s, m in r.unsupported_cites]
    if r.uncited_facts:
        L += ["- **Names/dates/numbers with no [S#] citation and no INVENTED mark (check each):**"]
        L += [f"  - {s}" for s in r.uncited_facts]
    L += [f"- Note: {n}" for n in r.notes]
    if clean != script:  # the version with every [S#] tag and {{INVENTED}} mark, for locating what the Checks list names
        L += ["", "## Fact-check copy (annotated script)", "", script]
    return "\n".join(L) + "\n"
