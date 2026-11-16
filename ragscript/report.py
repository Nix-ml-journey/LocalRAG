"""Every measurement taken on a finished script, in one object: the Checks section of the output, and the numbers the
eval scripts compare. The Report also decides what deserves a warning (`flags`), from config values, so `render` only
prints and the eval scripts only read. Adding a check means adding a field here and a line in `render`. The same module turns
a `Doc`, the script and the Report into the output markdown."""
import json
import re
import time
from dataclasses import dataclass, field

from . import checks, config, grounding
from .grounding import clean_narration


@dataclass
class Report:
    words: int
    target_words: int
    word_accuracy: float
    repeated_phrase_rate: float
    avg_sentence_words: float
    sentence_range: str             # the configured target, e.g. "10-15", shown next to the measurement
    overlap_flags: list
    banned_hits: list
    invented: list
    unsupported_cites: list
    uncited_facts: list
    pause_cues: int
    minutes: int
    mixed_tense: bool
    overused_words: list
    repeated_wording: list          # [(phrase, times)] details that came back in the same words
    motif_limit: int                # the most times one detail may return (config checks.motif_max)
    flags: list = field(default_factory=list)            # warnings the Report itself decided on
    notes: list = field(default_factory=list)
    repeated_outside_setting: float = 0.0   # repeated-phrase rate without phrases that hold a setting word
    detail_returns: list = field(default_factory=list)   # [(detail, paragraphs)] how often each outline detail came back

    @property
    def pauses_per_minute(self) -> float:
        return self.pause_cues / self.minutes


def build(script, *, target_words, minutes, refs, cited_text, grounded, banned, cfg, notes, brief="", details=(), setting=frozenset()) -> Report:
    """Measure `script`. refs: every source chunk text shown to the model; cited_text: {S number: chunk text};
    the two fact checks only run for grounded genres."""
    ck = cfg["checks"]
    words = checks.spoken_count(script)
    r = Report(
        words=words, target_words=target_words,
        word_accuracy=round(1 - abs(words - target_words) / target_words, 3),
        repeated_phrase_rate=round(checks.repeated_rate(script, ck["repeated_ngram"]), 4),
        avg_sentence_words=round(checks.avg_sentence_words(script), 1),
        sentence_range=config.sentence_range(cfg),
        overlap_flags=checks.overlap_flags(script, refs, ck["ngram"], ck["overlap_threshold"]),
        banned_hits=checks.banned_hits(script, banned),
        invented=grounding.invented(script),
        unsupported_cites=grounding.unsupported_cites(script, cited_text) if grounded else [],
        uncited_facts=grounding.uncited_facts(script) if grounded else [],
        pause_cues=script.count("[PAUSE]"), minutes=minutes,
        mixed_tense=checks.mixed_tense(script, ck),
        overused_words=checks.overused_words(script, ck, skip=checks.topic_words(brief)),
        repeated_wording=checks.repeated_wording(script), motif_limit=ck["motif_max"], notes=notes,
        detail_returns=checks.detail_returns(script, details),
        repeated_outside_setting=round(checks.repeated_rate_outside(script, ck["repeated_ngram"], setting), 4))
    if r.mixed_tense:
        r.flags.append("**Tense switches between past and present in the narration (check before recording).**")
    if over := [p for p, n in r.repeated_wording if n > r.motif_limit]:
        r.flags.append(f"A detail returned in the same words more than {r.motif_limit} times: " + "; ".join(over))
    if over := [f"{n} ({c})" for n, c in r.detail_returns if c > r.motif_limit]:
        r.flags.append(f"A recurring detail came back in more than {r.motif_limit} paragraphs: " + "; ".join(over))
    if r.repeated_outside_setting > ck["repeat_flag"]:
        r.flags.append(f"{r.repeated_outside_setting:.1%} of the script repeats phrases that are not part of the setting "
                       "(real repetition: vary or cut them)")
    limit = cfg["style"]["max_pauses_per_minute"]
    if r.pauses_per_minute > limit:
        r.flags.append(f"{r.pause_cues} [PAUSE] cues in {minutes} minutes: more than {limit} a minute reads as halting")
    return r


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
        out.append((int(acc * scale / wpm * 60), b.title or b.beat))
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
         "## Sources used", *[f"- {src}" + (" (as " + ", ".join(f"S{n}" for n in ns) + ")" if ns else " (style)")
                              for src, ns in doc.sources.items()], "",
         "## Checks",
         f"- Word accuracy: {r.word_accuracy:.1%}",
         f"- Repeated-phrase rate: {r.repeated_phrase_rate:.1%} ({r.repeated_outside_setting:.1%} outside the setting words)",
         f"- Average sentence length: {r.avg_sentence_words} words (natural narration is about {r.sentence_range})"]
    if r.overused_words:
        L.append("- Words used often (crutch words, consider varying): " + ", ".join(f"{w} ×{n}" for w, n in r.overused_words))
    if r.repeated_wording:
        L.append("- Details returning in the same words (fine when few; vary them): "
                 + ", ".join(f"\"{p}\" ×{n}" for p, n in r.repeated_wording))
    if r.detail_returns:
        L.append("- Outline details and the paragraphs that bring them back: "
                 + ", ".join(f"{n} x{c}" for n, c in r.detail_returns))
    L += [f"- {f}" for f in r.flags]
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


def save(out_dir, slug, minutes, brief, md, trace):
    """Write the markdown and, next to it, the run's trace as JSON (so a later stage can be replayed without the
    model); returns the markdown path. The name carries a time stamp, and a counter if two runs share a second, so a
    rerun never overwrites an earlier script."""
    out_dir.mkdir(exist_ok=True)
    name = re.sub(r"[^a-z0-9]+", "-", brief.lower())[:50].strip("-") or "script"
    stem = f"{slug}-{minutes}min-{name}-{time.strftime('%m%d-%H%M%S')}"
    path, n = out_dir / f"{stem}.md", 1
    while path.exists():
        n += 1
        path = out_dir / f"{stem}-{n}.md"
    path.write_text(md, encoding="utf-8")
    path.with_suffix(".trace.json").write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")
    return path
