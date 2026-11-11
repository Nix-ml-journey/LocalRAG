"""ChromaDB (persistent) + in-memory BM25, with incremental ingestion by file hash."""
import hashlib
import json
import logging
import re

import chromadb
from rank_bm25 import BM25Okapi

from . import chunking, config, loaders, metadata, vram

log = logging.getLogger("ragscript")
COLLECTIONS = ("scripts", "lore", "hooks")
_embedder = None


def embedder(cfg):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(cfg["embedding"]["model"], device=cfg["embedding"]["device"])
        _embedder.max_seq_length = 1024
        vram.log_stage("embedder loaded")
    return _embedder


def tok(s: str):
    return re.findall(r"\w+", s.lower())


class Store:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = chromadb.PersistentClient(path=str(config.path(cfg, "chroma_dir")))
        self.cols = {n: self.client.get_or_create_collection(n, metadata={"hnsw:space": "cosine"})
                     for n in COLLECTIONS}
        self._bm25 = {}

    def embed(self, texts):
        return embedder(self.cfg).encode(texts, normalize_embeddings=True, batch_size=8).tolist()

    def dense(self, col, query, n, where=None):
        """Nearest chunks of one Chroma collection: [{id, text, meta, distance}]. The one place a query is embedded
        and run, shared by the retriever and the HTTP service."""
        if col.count() == 0:
            return []
        r = col.query(query_embeddings=self.embed([query]), n_results=min(n, col.count()), where=where)
        return [{"id": i, "text": d, "meta": m, "distance": x}
                for i, d, m, x in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def bm25(self, name):
        """(bm25, ids, docs, metas) built lazily from Chroma, cleared after ingest."""
        if name not in self._bm25:
            d = self.cols[name].get()
            bm = BM25Okapi([tok(t) for t in d["documents"]]) if d["documents"] else None
            self._bm25[name] = (bm, d["ids"], d["documents"], d["metadatas"])
        return self._bm25[name]

    def _remove(self, rel):
        for c in self.cols.values():
            c.delete(where={"source": rel})

    def _add(self, name, rel, items, meta):
        """items: [(beat, text)]"""
        if not items:
            return
        texts = [t for _, t in items]
        metas = [{**meta, "beat": b, "words": len(t.split())} for b, t in items]
        ids = [f"{name}:{rel}:{i}" for i in range(len(items))]
        self.cols[name].add(ids=ids, documents=texts, embeddings=self.embed(texts), metadatas=metas)

    def ingest(self, llm_tag_fn):
        """Incrementally (re)index data_dir. llm_tag_fn(text)->JSON str for untagged files."""
        sf = config.path(self.cfg, "state_file")
        state = json.loads(sf.read_text()) if sf.exists() else {}
        state.setdefault("files", {})
        state.setdefault("tags", {})
        seen = set()
        for f, rel, h in self._files(config.path(self.cfg, "data_dir"), seen):
            if state["files"].get(rel) == h:
                continue
            log.info("ingesting %s", rel)
            self._remove(rel)
            if self._index_file(f, rel, h, state, llm_tag_fn):  # an untagged file stays "changed": next ingest retries
                state["files"][rel] = h
        self._prune(seen, state)
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(json.dumps(state, indent=1))
        self._bm25.clear()
        vram.log_stage("ingest done")
        return {n: c.count() for n, c in self.cols.items()}

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
            self._ingest_hooks(rel, loaders.clean(loaders.read_raw(f)))
            return True
        fm, text = loaders.load(f)
        meta, tagged = metadata.file_meta(rel, fm, text, h, state, llm_tag_fn)
        items = chunking.chunk(text, self.cfg)
        kind = metadata.kind_of(rel, fm)
        self._add(kind, rel, items, meta)
        if kind == "scripts" and items:
            # auto-fill the hooks library from each script's opening and closing beats
            self._add("hooks", rel, [("hook", items[0][1]), ("ending", items[-1][1])], meta)
        return tagged

    def _prune(self, seen, state):
        """Remove deleted files from the index; also files indexed but never recorded (tagger was down)."""
        indexed = {m["source"] for c in self.cols.values() for m in c.get(include=["metadatas"])["metadatas"]}
        for rel in (set(state["files"]) | indexed) - seen:
            log.info("removing deleted %s", rel)
            self._remove(rel)
            state["files"].pop(rel, None)

    def _ingest_hooks(self, rel, text):
        """Hand-written library: blocks separated by '---', first line 'TYPE: hook|ending'."""
        items = []
        for block in re.split(r"\n\s*---\s*\n", text):
            m = re.match(r"\s*TYPE:\s*(hook|ending)\s*\n(.+)", block, re.S | re.I)
            if m:
                items.append((m.group(1).lower(), m.group(2).strip()))
        self._add("hooks", rel, items, {"genre": "any", "source": rel})
