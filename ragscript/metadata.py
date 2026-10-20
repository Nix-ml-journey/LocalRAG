"""Per-file metadata: front-matter first, then folder rules, then a cached one-time LLM tag."""
import json
import logging
import re

log = logging.getLogger("ragscript")
TAG_KEYS = ("subgenre", "tone", "pov")


def kind_of(rel: str, fm: dict) -> str:
    parts = rel.split("/")
    if parts[0] == "hooks":
        return "hooks"
    if fm.get("kind") == "lore" or parts[-1].lower().startswith("lore"):
        return "lore"
    return "scripts"


def file_meta(rel: str, fm: dict, text: str, h: str, state: dict, llm_fn):
    """Returns (meta, tagged). llm_fn(prompt)->str is only called for files missing subgenre/tone/pov
    (cached by file hash). tagged=False means the tagger failed: the caller must not mark the file as ingested."""
    tagged = True
    parts = rel.split("/")
    meta = {"genre": "any" if parts[0] == "hooks" else parts[0], "source": rel}
    for k in TAG_KEYS + ("views", "retention"):
        if fm.get(k) is not None:
            meta[k] = fm[k]
    if parts[0] != "hooks" and any(k not in meta for k in TAG_KEYS):
        tags = state["tags"].get(h)
        if tags is None:
            try:
                m = re.search(r"\{.*\}", llm_fn(" ".join(text.split()[:1500])), re.S)
                tags = json.loads(m.group(0))
            except Exception as e:  # Ollama down or bad JSON: index with 'unknown', retry on the next ingest
                log.warning("tagging %s failed (%s); using 'unknown', will retry next ingest", rel, e)
                tags, tagged = {}, False
            else:
                state["tags"][h] = tags
        for k in TAG_KEYS:
            meta.setdefault(k, str(tags.get(k, "unknown")))
    return meta, tagged
