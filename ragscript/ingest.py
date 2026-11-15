"""Incremental ingestion: read data_dir, and index only the files whose content hash changed. The ingest state (file hashes
and the LLM's cached tags) is saved after every file, so an interrupted run resumes where it stopped."""
import hashlib
import json
import logging
import re

from . import chunking, config, loaders, metadata, vram

log = logging.getLogger("ragscript")


class Ingest:
    """Fills a Store (anything with add, remove, sources and counts) from the files in data_dir."""

    def __init__(self, store, cfg):
        self.store, self.cfg = store, cfg
        self.state_file = config.path(cfg, "state_file")

    def run(self, llm_tag_fn):
        """Index new and changed files, drop deleted ones. llm_tag_fn(text) -> JSON str, only called for files
        without subgenre/tone/pov in their front matter. Returns the collection sizes."""
        state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        state.setdefault("files", {})
        state.setdefault("tags", {})
        seen = set()
        for f, rel, h in self._files(config.path(self.cfg, "data_dir"), seen):
            if state["files"].get(rel) == h:
                continue
            log.info("ingesting %s", rel)
            self.store.remove(rel)
            if self._index_file(f, rel, h, state, llm_tag_fn):  # an untagged file stays "changed": next ingest retries
                state["files"][rel] = h
                self._save(state)  # per file: an interrupted ingest resumes here instead of redoing everything
        self._prune(seen, state)
        self._save(state)
        vram.log_stage("ingest done")
        return self.store.counts()

    def _save(self, state):
        """Write the state through a temp file, so a kill during the write cannot leave broken JSON."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state_file)

    @staticmethod
    def _files(data, seen):
        """(path, relative name, content hash) of every supported file; their names are added to `seen`."""
        for f in sorted(data.rglob("*")):
            if f.is_file() and f.suffix.lower() in loaders.SUPPORTED:
                rel = f.relative_to(data).as_posix()
                seen.add(rel)
                yield f, rel, hashlib.sha256(f.read_bytes()).hexdigest()

    def _index_file(self, f, rel, h, state, llm_tag_fn) -> bool:
        """Chunk and add one file to its collection. False when the LLM tagger failed for it."""
        if rel.startswith("hooks/"):
            self._hooks(rel, loaders.clean(loaders.read_raw(f)))
            return True
        fm, text = loaders.load(f)
        meta, tagged = metadata.file_meta(rel, fm, text, h, state, llm_tag_fn)
        items = chunking.chunk(text, self.cfg)
        kind = metadata.kind_of(rel, fm)
        self.store.add(kind, rel, items, meta)
        if kind == "scripts" and items:
            # auto-fill the hooks library from each script's opening and closing beats
            self.store.add("hooks", rel, [("hook", items[0][1]), ("ending", items[-1][1])], meta)
        return tagged

    def _prune(self, seen, state):
        """Remove deleted files from the index; also files indexed but never recorded (tagger was down)."""
        for rel in (set(state["files"]) | self.store.sources()) - seen:
            log.info("removing deleted %s", rel)
            self.store.remove(rel)
            state["files"].pop(rel, None)

    def _hooks(self, rel, text):
        """Hand-written library: blocks separated by '---', first line 'TYPE: hook|ending'."""
        items = []
        for block in re.split(r"\n\s*---\s*\n", text):
            m = re.match(r"\s*TYPE:\s*(hook|ending)\s*\n(.+)", block, re.S | re.I)
            if m:
                items.append((m.group(1).lower(), m.group(2).strip()))
        self.store.add("hooks", rel, items, {"genre": "any", "source": rel})
