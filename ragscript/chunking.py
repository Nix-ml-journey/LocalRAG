"""Beat chunking with a 300-500 token sliding fallback (15% overlap)."""
import re

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
