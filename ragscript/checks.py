"""Post-generation checks: source overlap, repeated phrases, banned phrases, invented marks, word count."""
import re
from collections import Counter

CUE = re.compile(r"\[[^\]]*\]")
QUOTE = re.compile(r"[“\"]([^”\"]{6,})[”\"]")  # a quoted phrase of 6+ characters
# sentence boundary; a [S#] tag after the full stop stays with its sentence, and the tag itself ends one
SENT = r"(?<=[.!?”\"\]])\s+(?!\[S\d)"


def words(t: str):
    return re.findall(r"[a-z0-9']+", CUE.sub(" ", t).lower().replace("’", "'"))


def spoken_count(t: str) -> int:
    return len(words(re.sub(r"\{\{INVENTED:|\}\}", " ", t)))


def _grams(w, n):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def overlap_flags(text, refs, n, thr):
    """Paragraphs where > thr of their n-grams also occur in a single source chunk."""
    ref_sets = [_grams(words(r), n) for r in refs]
    flags = []
    for para in re.split(r"\n\s*\n", text):
        sh = _grams(words(para), n)
        if len(sh) < 3:
            continue
        best = max((len(sh & rs) / len(sh) for rs in ref_sets), default=0)
        if best > thr:
            flags.append((para.strip()[:140], round(best, 2)))
    return flags


def reused_quotes(text, earlier):
    """Quoted phrases (6+ chars) in text that already appear in earlier text."""
    low = earlier.lower()
    return [q for q in QUOTE.findall(text) if q.lower() in low]


def uncited_facts(text):
    """Sentences in a fact-grounded script that name something (capitalised word after the first, or a digit)
    but carry neither a [S#] citation nor an {{INVENTED}} mark. A heuristic: proper nouns trigger it."""
    out = []
    for s in re.split(SENT, text):
        if "[S" in s or "INVENTED" in s:
            continue
        c = CUE.sub("", s).strip()
        body = c.split(None, 1)[1] if len(c.split()) > 1 else ""  # skip the sentence-initial capital
        if re.search(r"\d", c) or re.search(r"\b[A-Z][a-z]{2,}", body):
            out.append(s.strip())  # the sentence exactly as in the text, so it can be located and replaced
    return out


def drop_repeats(text, earlier="", n=4, thr=0.6):
    """Remove paragraphs that mostly repeat an earlier one or a paragraph of `earlier`
    (small models re-write the previous section)."""
    kept, seen = [], _grams(words(earlier), n)  # union of everything before: a retelling may merge several old paragraphs
    for para in re.split(r"\n\s*\n", text):
        sh = _grams(words(para), n)
        if sh and len(sh & seen) / len(sh) > thr:
            continue
        kept.append(para)
        seen |= sh
    return "\n\n".join(kept)


def drop_repeat_sentences(text, earlier="", n=4, thr=0.7, min_words=6):
    """Remove sentences whose n-grams are mostly already present in `earlier` or earlier in this text
    (re-told facts, repeated quotes). Paragraph breaks are kept."""
    seen = _grams(words(earlier), n)
    out = []
    for para in re.split(r"\n\s*\n", text):
        keep = []
        for s in re.split(SENT, para):
            w = words(s)
            sh = _grams(w, n)
            if len(w) >= min_words and sh and len(sh & seen) / len(sh) > thr:
                continue
            keep.append(s)
            seen |= sh
        if keep:
            out.append(" ".join(keep))
    return "\n\n".join(out)


def dedupe(text, earlier=""):
    """The anti-repetition pass: repeated paragraphs, then repeated sentences, against `earlier` and the text itself.
    Always call this instead of the two functions separately: each catches what the other misses."""
    return drop_repeat_sentences(drop_repeats(text, earlier), earlier)


def avg_sentence_words(text) -> float:
    """Average spoken words per sentence (cues and INVENTED markers ignored). Natural narration sits near 10-15;
    below ~7 it sounds like a string of fragments, above ~20 it reads like a book."""
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", re.sub(r"\{\{INVENTED:|\}\}", " ", CUE.sub(" ", text))) if words(s)]
    return sum(len(words(s)) for s in sentences) / len(sentences) if sentences else 0.0


def repeated_rate(text, n) -> float:
    """Share of n-gram occurrences that belong to an n-gram appearing more than once."""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v for v in c.values() if v > 1) / len(grams)


def banned_hits(text, phrases):
    low = text.lower()
    return [p for p in phrases if p.lower() in low]


def unsupported_cites(text, chunks):
    """Sentences citing [S#] whose numbers, quoted phrases or capitalised names (people, places, ships) are missing
    from every cited lore chunk. chunks: {n: chunk text}. Cheap and strict; it cannot judge paraphrased claims, and
    a capitalised word that is not a name only costs one extra Verify call."""
    bad = []
    for s in re.split(SENT, text):
        cited = [int(n) for n in re.findall(r"\[S(\d+)\]", s)]
        if not cited:
            continue
        src = " ".join(chunks.get(n, "") for n in cited).lower()
        clean = re.sub(r"\[S\d+\]", "", s)
        # names inside {{INVENTED}} and cues are already labelled, so only the plain narration is checked; the first
        # word of the sentence is skipped because it is capitalised anyway
        plain = CUE.sub(" ", re.sub(r"\{\{INVENTED:.*?\}\}", " ", clean, flags=re.S))
        names = [re.sub(r"['’]s$", "", w) for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", plain)[1:]
                 if w[0].isupper() and len(w) > 2 and not w.isupper() and not re.match(r"I['’]", w)]  # not I'll, I've
        claims = re.findall(r"\b\d[\d,.]*\d\b|\b\d\b", clean) + QUOTE.findall(clean) + names
        miss = list(dict.fromkeys(c for c in claims if c.lower().strip(".,") not in src))
        if miss:
            bad.append((s.strip(), miss))
    return bad


def mark_invented(script, sentences, is_supported, limit=12):
    """Verify step: wrap each flagged sentence that `is_supported(sentence)` rejects as {{INVENTED: ...}}.
    Returns (script, number marked). Capped at `limit` checks to bound the number of LLM calls."""
    n = 0
    for s in list(dict.fromkeys(sentences))[:limit]:
        if s in script and not is_supported(s):
            script = script.replace(s, "{{INVENTED: " + re.sub(r"\s*\[S\d+\]", "", s) + "}}", 1)
            n += 1
    return script, n


def clean_narration(script) -> str:
    """The script as it should be recorded: [S#] source tags removed, {{INVENTED: x}} unwrapped to x, cues kept."""
    text = re.sub(r"[ \t]*\[S\d+\]", "", script)
    text = re.sub(r"\{\{INVENTED:\s*(.*?)\}\}", r"\1", text, flags=re.S)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", text))


def invented(text):
    return re.findall(r"\{\{INVENTED:\s*(.*?)\}\}", text, re.S)
