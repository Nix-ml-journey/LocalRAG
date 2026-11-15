"""Fact grounding for history-style genres: [S#] citations and the {{INVENTED: ...}} mark protocol.

This module is the only place that knows how a mark is written, found, counted, removed or relabelled; every other
module goes through these functions. It also holds the Verify step, which checks the sentences the code cannot trust."""
import re

from .text import CUE, sentences, words

QUOTE = re.compile(r"[“\"]([^”\"]{6,})[”\"]")  # a quoted phrase of 6+ characters
MARK = re.compile(r"\{\{INVENTED:\s*(.*?)\}\}", re.S)


def invented(text):
    """The text of every INVENTED mark."""
    return MARK.findall(text)


def unmark(text):
    """The text with the mark delimiters turned into spaces and the marked words kept (for counting words)."""
    return re.sub(r"\{\{INVENTED:|\}\}", " ", text)


def drop_marks(text):
    """The text without the marked passages at all."""
    return MARK.sub(" ", text)


def wrap(sentence):
    """A sentence as an INVENTED mark, with its [S#] tags removed (an invented claim cites nothing)."""
    return "{{INVENTED: " + re.sub(r"\s*\[S\d+\]", "", sentence) + "}}"


def relabel(text):
    """A {{...}} the model wrote without the INVENTED: label is the mark it meant, so it gets the label."""
    return re.sub(r"\{\{\s*(?!invented\s*:)(.*?)\}\}", r"{{INVENTED: \1}}", text, flags=re.S | re.I)


def clean_narration(script) -> str:
    """The script as it should be recorded: [S#] source tags removed, {{INVENTED: x}} unwrapped to x, cues kept."""
    text = re.sub(r"[ \t]*\[S\d+\]", "", script)
    text = MARK.sub(r"\1", text)
    text = text.replace("{{", "").replace("}}", "")  # never leave stray braces in the recording copy
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", text))


def uncited_facts(text):
    """Sentences in a fact-grounded script that name something (capitalised word after the first, or a digit)
    but carry neither a [S#] citation nor an {{INVENTED}} mark. A heuristic: proper nouns trigger it."""
    out, pos = [], 0
    marks = [m.span() for m in re.finditer(r"\{\{INVENTED:.*?\}\}", text, re.S)]
    for s in sentences(text):
        pos = text.find(s, pos)
        if "[S" in s or "INVENTED" in s or any(a <= pos < b for a, b in marks):  # cited, or inside an INVENTED mark that spans sentences
            continue
        c = CUE.sub("", s).strip()
        body = c.split(None, 1)[1] if len(c.split()) > 1 else ""  # skip the sentence-initial capital
        if re.search(r"\d", c) or re.search(r"\b[A-Z][a-z]{2,}", body):
            out.append(s.strip())  # the sentence exactly as in the text, so it can be located and replaced
    return out


def unsupported_cites(text, chunks):
    """Sentences citing [S#] whose numbers, quoted phrases or capitalised names (people, places, ships) are missing
    from every cited lore chunk. chunks: {n: chunk text}. Cheap and strict; it cannot judge paraphrased claims, and
    a capitalised word that is not a name only costs one extra Verify call."""
    bad = []
    for s in sentences(text):
        cited = [int(n) for n in re.findall(r"\[S(\d+)\]", s)]
        if not cited:
            continue
        src = " ".join(chunks.get(n, "") for n in cited).lower()
        clean = re.sub(r"\[S\d+\]", "", s)
        # names inside {{INVENTED}} and cues are already labelled, so only the plain narration is checked; the first
        # word of the sentence is skipped because it is capitalised anyway
        plain = CUE.sub(" ", drop_marks(clean))
        names = [re.sub(r"['’]s$", "", w) for w in re.findall(r"[^\W\d_][\w'’-]*", plain)[1:]
                 if w[0].isupper() and len(w) > 2 and not w.isupper() and not re.match(r"I['’]", w)]  # not I'll, I've
        claims = re.findall(r"\b\d[\d,.]*\d\b|\b\d\b", clean) + QUOTE.findall(clean) + names
        miss = list(dict.fromkeys(c for c in claims if c.lower().strip(".,") not in src))
        if miss:
            bad.append((s.strip(), miss))
    return bad


def mark_invented(script, flagged, is_supported, limit=12):
    """Wrap each flagged sentence that `is_supported(sentence)` rejects as an INVENTED mark.
    Returns (script, number marked). Capped at `limit` checks to bound the number of LLM calls."""
    n = 0
    for s in list(dict.fromkeys(flagged))[:limit]:
        if "{{" in s or "}}" in s:  # already marked, or half of a mark that spans sentences: wrapping again would nest
            continue
        if s in script and not is_supported(s):
            script = script.replace(s, wrap(s), 1)
            n += 1
    return script, n


def verify(script, cited_text, ask, limit=40):
    """Wrap every unsupported sentence as {{INVENTED: ...}}. Returns (script, number marked).

    cited_text: {S number: lore chunk text}. ask(lore, sentence) -> the model's YES/NO answer, lore as sole evidence.
    Checked: sentences naming something with no citation, cited sentences whose numbers/quotes/names are missing from
    their chunk, and every other cited sentence (a [S1] tag proves nothing: the model also cites invented claims)."""
    flagged = (uncited_facts(script) + [s for s, _ in unsupported_cites(script, cited_text)]
               + [s for s in sentences(script) if re.search(r"\[S\d+\]", s)])

    def supported(s):
        sw = set(words(s))  # the 12 chunks sharing most words with the sentence, so its source is in
        near = sorted(cited_text.values(), key=lambda t: -len(sw & set(words(t))))[:12]
        return not ask("\n\n".join(near), s).strip().upper().startswith("NO")  # an unclear answer leaves it alone

    return mark_invented(script, flagged, supported, limit)
