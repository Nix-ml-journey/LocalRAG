"""Post-generation checks on the narration itself: repeats, source overlap, sentence length, tense, crutch words.
Fact grounding ([S#] and {{INVENTED}}) lives in grounding.py; the sentence splitter and tokenizer in text.py."""
import re
from collections import Counter

from .grounding import QUOTE, unmark
from .text import CUE, sentences, words


def spoken_count(t: str) -> int:
    return len(words(unmark(t)))


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
        for s in sentences(para):
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
    sents = [s for s in sentences(unmark(CUE.sub(" ", text))) if words(s)]
    return sum(len(words(s)) for s in sents) / len(sents) if sents else 0.0


def repeated_rate(text, n) -> float:
    """Share of n-gram occurrences that belong to an n-gram appearing more than once."""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v for v in c.values() if v > 1) / len(grams)


_PRESENT = r"am|see|hear|feel|walk|stand|look|know|think|find|kick|reach|run|take|go|watch|step|move|have|can|do|don't|cannot|stare|wipe|open|close|hold|turn|pull|push|follow|wait|listen|say|ask|tell|try|want|need|notice|realize"
_PAST = r"was|saw|heard|felt|walked|stood|looked|knew|thought|found|kicked|reached|ran|took|went|watched|stepped|moved|had|could|did|didn't|couldn't|stared|wiped|opened|closed|held|turned|pulled|pushed|followed|waited|listened|said|asked|told|tried|wanted|needed|noticed|realized"


def tense_mix(text):
    """(present, past) counts of first-person verbs ("I see" / "I saw"). A narrator who uses both in meaningful
    numbers switched tense; the heuristic only looks at "I" + verb, so it never fires on third-person scripts."""
    # quoted speech keeps its own tense ("I don't know," she said), so only the narration is counted
    plain = re.sub(r"[“\"][^”\"]*[”\"]", " ", unmark(CUE.sub(" ", text)))
    pres = len(re.findall(rf"\bI (?:{_PRESENT})\b", plain))
    past = len(re.findall(rf"\bI (?:{_PAST})\b", plain))
    return pres, past


def mixed_tense(text, floor=3, share=0.2) -> bool:
    pres, past = tense_mix(text)
    return min(pres, past) >= floor and min(pres, past) / (pres + past) >= share


_STOP = set("""about above after again against along also always another around because before behind being below between
beyond could doing during every exactly except further having himself herself itself might never often other
should since still than their theirs them themselves there these they those though through under until upon
very were what when where whether which while whose would your yours yourself""".split())


def overused_words(text, min_count=5, share=0.006, top=5):
    """Content words (5+ letters) that recur far more than natural prose does: the crutch words a small model
    leans on ("heavy", "hum"). Returns [(word, count)], most frequent first."""
    w = [x for x in words(unmark(text)) if len(x) >= 5 and x not in _STOP]
    total = max(1, len(words(text)))
    c = Counter(w)
    return [(k, v) for k, v in c.most_common(top * 3) if v >= min_count and v / total >= share][:top]


_FUNC = _STOP | set("""that with from this they them were been have into over like just when then than what your
does some only more each even also very much such back there here will""".split())


def _pairs(text):
    """Counter of pairs of consecutive content words (function words skipped, one sentence at a time):
    "the smell of ozone" and "smell like ozone" both count as ("smell", "ozone")."""
    c = Counter()
    for s in sentences(unmark(text)):
        cw = [w for w in words(s) if len(w) >= 4 and w not in _FUNC]
        c.update(zip(cw, cw[1:]))
    return c


def crowded_phrases(text, min_count=3, top=5):
    """[(phrase, count)] for the images (two content words) that return `min_count`+ times in the text."""
    return [(" ".join(k), v) for k, v in _pairs(text).most_common(top * 2) if v >= min_count][:top]


def repeated_phrases(text, earlier, min_count=3):
    """[(phrase, total)] for images in `text` that were already used in `earlier` and, counting both, appear
    `min_count`+ times: the sensory anchors a small model brings back in every section."""
    before, now = _pairs(earlier), _pairs(text)
    return [(" ".join(k), before[k] + v) for k, v in now.items() if before[k] and before[k] + v >= min_count]


def repeated_lines(text, earlier="", min_words=2):
    """Sentences of `min_words`+ words that already appear word for word in `earlier` or earlier in `text`: the
    short refrains ("It was watching me back.") the 4-gram repeat guards cannot see."""
    seen = {tuple(words(s)) for s in sentences(earlier) if len(words(s)) >= min_words}
    out = []
    for s in sentences(text):
        w = tuple(words(s))
        if len(w) >= min_words:
            if w in seen:
                out.append(s.strip())
            seen.add(w)
    return out


def crowded_words(text, limit=3):
    """[(word, count)] for content words (5+ letters) used `limit` or more times in one stretch of text."""
    c = Counter(x for x in words(unmark(text)) if len(x) >= 5 and x not in _STOP)
    return [(w, n) for w, n in c.most_common() if n >= limit]


def banned_hits(text, phrases):
    low = text.lower()
    return [p for p in phrases if p.lower() in low]
