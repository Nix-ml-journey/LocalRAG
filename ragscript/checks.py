"""Post-generation checks on the narration itself: repeats, source overlap, sentence length, tense, crutch words.
Fact grounding ([S#] and {{INVENTED}}) lives in grounding.py; the sentence splitter and tokenizer in text.py; the
story-logic checks (outline holes, details, dropped objects, the ending) in logic.py.
Every threshold comes from config.yaml `checks:` (passed in as `ck`); nothing is tuned by editing this file."""
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


def drop_repeats(text, earlier, n, thr):
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


def drop_repeat_sentences(text, earlier, n, thr, min_words):
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


def content_set(text):
    """The distinct content words (4+ letters, no function words) of a stretch of text."""
    return {w for w in words(unmark(text)) if len(w) >= 4 and w not in FUNC}


def drop_similar(text, earlier, thr, min_content):
    """Remove paragraphs that retell an earlier one in different words. The 4-gram guards miss a retelling that
    rewords every phrase, but it still reuses the same content words: a paragraph is dropped when `thr` of the smaller
    paragraph's content words occur in an earlier paragraph (of `earlier` or kept before it). Paragraphs with fewer
    than `min_content` content words are never compared."""
    seen = [s for s in (content_set(p) for p in re.split(r"\n\s*\n", earlier)) if len(s) >= min_content]
    kept = []
    for para in re.split(r"\n\s*\n", text):
        cs = content_set(para)
        if len(cs) >= min_content:
            if any(len(cs & s) / min(len(cs), len(s)) >= thr for s in seen):
                continue
            seen.append(cs)
        kept.append(para)
    return "\n\n".join(kept)


def dedupe(text, ck, earlier=""):
    """The anti-repetition pass: repeated paragraphs, reworded retellings, then repeated sentences, against `earlier`
    and the text itself. Always call this instead of the functions separately: each catches what the others miss."""
    n = ck["dedupe_ngram"]
    text = drop_repeats(text, earlier, n, ck["dedupe_paragraph"])
    text = drop_similar(text, earlier, ck["dedupe_similar"], ck["dedupe_similar_min"])
    return drop_repeat_sentences(text, earlier, n, ck["dedupe_sentence"], ck["dedupe_min_words"])


def repeated_wording(text, n=5, top=5):
    """Details that return in the same words: [(phrase, times)], most frequent first. A phrase is a run of repeated
    n-grams (with at least 2 content words, so "i was in the" never counts); times is how often its opening n-gram
    starts a run. A recurring detail is fine when it changes; this lists the ones that came back unchanged."""
    w = words(unmark(text))
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(grams)
    runs, i = {}, 0
    while i < len(grams):
        if c[grams[i]] < 2 or sum(x not in FUNC and len(x) >= 4 for x in grams[i]) < 2:
            i += 1
            continue
        j = i
        while j + 1 < len(grams) and c[grams[j + 1]] > 1:
            j += 1
        run = w[i:j + n]
        key = grams[i]
        if key not in runs or len(run) > len(runs[key][0]):
            runs[key] = (run, runs.get(key, (0, 0))[1] + 1)
        else:
            runs[key] = (runs[key][0], runs[key][1] + 1)
        i = j + 1
    out = [(" ".join(r), k) for r, k in runs.values() if k > 1]
    return sorted(out, key=lambda x: -x[1])[:top]


def tidy(text):
    """Layout clean-up of a finished script: no leading or trailing spaces on a line, a [PAUSE] glued to the end of a
    sentence goes on its own line, and a [PAUSE] directly followed by another is dropped. Words are unchanged."""
    t = "\n".join(line.strip() for line in text.split("\n"))
    t = re.sub(r"(?m)(\S)[ \t]+\[PAUSE\]$", r"\1\n\n[PAUSE]", t)
    t = re.sub(r"\[PAUSE\]\s*(?=\[PAUSE\])", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


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


_PRESENT = (
    r"am|see|hear|feel|walk|stand|look|know|think|find|kick|reach|"
    r"run|take|go|watch|step|move|have|can|do|don't|cannot|stare|"
    r"wipe|open|close|hold|turn|pull|push|follow|wait|listen|say|ask|"
    r"tell|try|want|need|notice|realize"
)
_PAST = (
    r"was|saw|heard|felt|walked|stood|looked|knew|thought|found|kicked|reached|"
    r"ran|took|went|watched|stepped|moved|had|could|did|didn't|couldn't|stared|"
    r"wiped|opened|closed|held|turned|pulled|pushed|followed|waited|listened|said|asked|"
    r"told|tried|wanted|needed|noticed|realized"
)


def tense_mix(text):
    """(present, past) counts of first-person verbs ("I see" / "I saw"). A narrator who uses both in meaningful
    numbers switched tense; the heuristic only looks at "I" + verb, so it never fires on third-person scripts."""
    # quoted speech keeps its own tense ("I don't know," she said), so only the narration is counted
    plain = re.sub(r"[“\"][^”\"]*[”\"]", " ", unmark(CUE.sub(" ", text)))
    pres = len(re.findall(rf"\bI (?:{_PRESENT})\b", plain))
    past = len(re.findall(rf"\bI (?:{_PAST})\b", plain))
    return pres, past


def mixed_tense(text, ck) -> bool:
    pres, past = tense_mix(text)
    rarer = min(pres, past)
    return rarer >= ck["tense_floor"] and rarer / (pres + past) >= ck["tense_share"]


STOP = set("""about above after again against along also always another around because before behind being below between
beyond could doing during every exactly except further having himself herself itself might never often other
should since still than their theirs them themselves there these they those though through under until upon
very were what when where whether which while whose would your yours yourself""".split())
FUNC = STOP | set("""that with from this they them were been have into over like just when then than what your
does some only more each even also very much such back there here will""".split())


_SHORT_STOP = set("and but for the its his her our was are had has not you who how why can may few all any one two out off "
                  "own too yet now per via she him air".split())


def setting_words(*texts):
    """The stems of the content words (4+ letters) in the given texts: used for the brief and the outline's place,
    sees and detail text, i.e. the story's setting and its recurring details."""
    # "key_weight" is one token to the splitter, so underscores become spaces first; 3 letters are kept for "hum", "fan"
    return {stem(w) for t in texts for w in words(unmark(t).replace("_", " "))
            if len(w) >= 3 and w not in FUNC and w not in _SHORT_STOP}


def repeated_rate_outside(text, n, setting) -> float:
    """Like `repeated_rate`, but a repeated n-gram that holds a setting word does not count: a night-shift guard story
    will say "screen" and "fan" again and again, and that is the place, not a habit. What is left is real tic."""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v for g, v in c.items() if v > 1 and not any(stem(x) in setting for x in g)) / len(grams)


def stem(w):
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


def topic_words(brief):
    """The brief's own content words (stemmed): a story about monitors says "monitor" often, which is not a crutch."""
    return {stem(w) for w in words(brief) if len(w) >= 5 and w not in STOP}


def content_counts(text, skip=frozenset()):
    """Counter of the content words (5+ letters, no function words, none of the `skip` topic words) in the text: the
    one counter behind both crutch-word checks."""
    return Counter(x for x in words(unmark(text)) if len(x) >= 5 and x not in STOP and stem(x) not in skip)


def overused_words(text, ck, top=5, skip=frozenset()):
    """Content words that recur far more than natural prose does across a whole script: the crutch words a small
    model leans on ("heavy", "hum"). Returns [(word, count)], most frequent first."""
    total = max(1, len(words(text)))
    found = [(k, v) for k, v in content_counts(text, skip).most_common(top * 3)
             if v >= ck["crowd_min"] and v / total >= ck["crowd_share"]]
    return found[:top]


def banned_hits(text, phrases):
    low = text.lower()
    return [p for p in phrases if p.lower() in low]
