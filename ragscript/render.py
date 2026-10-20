"""Markdown output for one finished script."""
from .checks import clean_narration


def _ts(sec: int) -> str:
    return f"{sec // 60:02d}:{sec % 60:02d}"


def render(brief, genre, minutes, narrator, pkg, script, chapters, sources, stats) -> str:
    titles = pkg.get("titles", [])
    clean = clean_narration(script)
    L = [f"# {titles[0] if titles else brief}", "",
         f"*Genre: {genre} · Target: {minutes} min · {stats['words']} words (target {stats['target_words']})*", "",
         "## Title options", *[f"{i}. {t}" for i, t in enumerate(titles, 1)], "",
         "## Thumbnail text", *[f"- {t}" for t in pkg.get("thumbnails", [])], "",
         "## Narrator", str(narrator), "",
         "## Script", "", script, "",
         "## Chapters (estimated)", *[f"- {_ts(s)} {t}" for s, t in chapters], "",
         "## Clean narration (for recording)", "",
         # the annotated script above is for fact-checking; this copy has no [S#] tags and no INVENTED wrappers
         clean if clean != script else "*The script above has no source tags or invented-detail markers, so it can be recorded as is.*", "",
         "## Description", pkg.get("description", ""), "",
         "## Tags", ", ".join(pkg.get("tags", [])), "",
         "## Sources used", *[f"- {src}" + (" (as " + ", ".join(f"S{n}" for n in ns) + ")" if ns else " (style)") for src, ns in sources.items()], "",
         "## Checks",
         f"- Word accuracy: {stats['word_accuracy']:.1%}",
         f"- Repeated-phrase rate: {stats['repeated_phrase_rate']:.1%}",
         f"- Average sentence length: {stats['avg_sentence_words']} words (natural narration is about 10-15)"]
    if stats["overlap_flags"]:
        L += ["- **>80% overlap with a source (rewrite these):**"] + [f"  - ({r}) {p}…" for p, r in stats["overlap_flags"]]
    else:
        L.append("- No passage over the overlap threshold")
    if stats["banned_hits"]:
        L.append("- Banned phrases still present: " + "; ".join(stats["banned_hits"]))
    if stats["invented"]:
        L += ["- **Invented details to fact-check:**"] + [f"  - {x}" for x in stats["invented"]]
    if stats["unsupported_cites"]:
        L += ["- **Cited [S#] but numbers, quotes or names not found in that lore chunk (verify or delete):**"]
        L += [f"  - {s} → missing: {', '.join(m)}" for s, m in stats["unsupported_cites"]]
    if stats["uncited_facts"]:
        L += ["- **Names/dates/numbers with no [S#] citation and no INVENTED mark (check each):**"]
        L += [f"  - {s}" for s in stats["uncited_facts"]]
    L += [f"- Note: {n}" for n in stats["notes"]]
    return "\n".join(L) + "\n"
