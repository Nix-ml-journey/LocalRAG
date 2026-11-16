"""Beat chunking with a 300-500 token sliding fallback (15% overlap)."""
import re
from dataclasses import dataclass

BEATS = ["hook", "setup", "rising_tension", "reveal", "climax", "aftermath", "outro"]
ALIASES = {
    "hook": ["hook", "cold open", "opening"],
    "setup": ["setup", "set-up", "background", "introduction"],
    "rising_tension": ["rising", "tension", "escalation"],
    "reveal": ["reveal", "twist", "discovery"],
    "climax": ["climax", "confrontation"],
    "aftermath": ["aftermath", "resolution", "fallout"],
    "outro": ["outro", "ending", "conclusion", "closing"],
}
# "## Hook" markdown headings, or a line that is only "[BEAT: climax]"
_HEADER = re.compile(r"^(?:#{1,4}\s+(.+)|\[BEAT:\s*([^\]\n]+)\])\s*$", re.M | re.I)


@dataclass
class Beat:
    """One beat of the outline: the one place the beat's fields are named. `words` is the budget the drafter rolls
    forward; `orig` is the planned budget it scales from."""
    beat: str
    words: int = 0
    orig: int = 0
    title: str = ""
    summary: str = ""
    place: str = ""    # where the narrator's body is for the whole beat
    sees: str = ""     # what he looks at from there (a named screen stays that screen)
    change: str = ""   # the one new fact, event or danger the beat adds
    detail: str = ""   # "name: how it changes", the recurring detail used here

    @classmethod
    def from_json(cls, d):
        """A beat from the model's JSON. "Rising Tension" becomes rising_tension; an unknown beat or a word count
        that is not a positive number raises, so the outline attempt fails and is retried."""
        beat, words = str(d["beat"]).strip().lower().replace(" ", "_"), int(d["words"])
        if beat not in BEATS or words <= 0:
            raise ValueError(f"bad beat {d!r}")
        return cls(beat, words, **{k: str(d.get(k) or "") for k in ("title", "summary", "place", "sees", "change", "detail")})


def tokens(s: str) -> int:
    return int(len(s.split()) * 1.3)  # rough; avoids a tokenizer dependency


def _beat_of(header: str):
    h = header.lower()
    return next((b for b, al in ALIASES.items() if any(a in h for a in al)), None)


def _split_headers(text: str):
    """[(beat, body)] if >=3 recognised beat headers exist, else None."""
    marks = [(m.start(), m.end(), _beat_of(m.group(1) or m.group(2))) for m in _HEADER.finditer(text)]
    marks = [m for m in marks if m[2]]
    if len(marks) < 3:
        return None
    out = []
    for i, (_, end, beat) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[end:stop].strip()
        if body:
            out.append((beat, body))
    return out


def sliding(text: str, min_t: int, max_t: int, overlap: float):
    sents, n = [], int(max_t / 1.3 / 2)  # a "sentence" longer than half a chunk (subtitles, bare PDF text) is cut by words
    for s in re.split(r"(?<=[.!?])\s+", text):
        w = s.split()
        sents += [" ".join(w[i:i + n]) for i in range(0, len(w), n)] if len(w) > n else [s]
    chunks, cur = [], []
    for s in sents:
        if cur and tokens(" ".join(cur + [s])) > max_t and tokens(" ".join(cur)) >= min_t:
            chunks.append(" ".join(cur))
            keep, n = [], 0
            for prev in reversed(cur):  # carry the last ~15% of tokens into the next chunk
                if n >= max_t * overlap:
                    break
                keep.insert(0, prev)
                n += tokens(prev)
            cur = keep
        cur.append(s)
    if cur:
        tail = " ".join(cur)
        # fold a small stub into the previous chunk only if it still fits under max_t
        if chunks and tokens(chunks[-1] + " " + tail) <= max_t:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)
    return chunks


def chunk(text: str, cfg: dict):
    """Returns [(beat, text)]."""
    c = cfg["chunking"]
    segs = _split_headers(text)
    if segs:
        out = []
        for beat, body in segs:
            if tokens(body) > c["max_tokens"] * 1.3:
                out += [(beat, x) for x in sliding(body, c["min_tokens"], c["max_tokens"], c["overlap"])]
            else:
                out.append((beat, body))
        return out
    parts = sliding(text, c["min_tokens"], c["max_tokens"], c["overlap"])
    # no beat markers: estimate beat from position in the story
    return [(BEATS[min(int(i / len(parts) * len(BEATS)), len(BEATS) - 1)], p) for i, p in enumerate(parts)]
