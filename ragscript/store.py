"""The search index: ChromaDB (persistent) for dense vectors and an in-memory BM25 index built from it.
Filling it from files is ingest.py's job; this module stores, removes and finds chunks, and `Retriever` fuses the dense
and BM25 rankings (reciprocal rank fusion)."""
import re

import chromadb
from rank_bm25 import BM25Okapi

from . import config, vram

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
    def __init__(self, cfg: dict, embed=None):
        """`embed(texts) -> vectors` defaults to bge-m3 on the CPU; a test passes a fake so search runs offline."""
        self.cfg = cfg
        self._embed = embed or (lambda texts: embedder(cfg).encode(
            texts, normalize_embeddings=True, batch_size=8).tolist())
        self.client = chromadb.PersistentClient(path=str(config.path(cfg, "chroma_dir")))
        self.cols = {n: self.client.get_or_create_collection(n, metadata={"hnsw:space": "cosine"})
                     for n in COLLECTIONS}
        self._bm25 = {}

    def embed(self, texts):
        return self._embed(texts)

    def counts(self):
        return {n: c.count() for n, c in self.cols.items()}

    def sources(self):
        """Every source file that has at least one chunk in any collection."""
        return {m["source"] for c in self.cols.values() for m in c.get(include=["metadatas"])["metadatas"]}

    # ---- writing -------------------------------------------------------
    def add(self, name, rel, items, meta):
        """Add the chunks of one source file to collection `name`. items: [(beat, text)]."""
        if not items:
            return
        texts = [t for _, t in items]
        metas = [{**meta, "beat": b, "words": len(t.split())} for b, t in items]
        ids = [f"{name}:{rel}:{i}" for i in range(len(items))]
        self.cols[name].add(ids=ids, documents=texts, embeddings=self.embed(texts), metadatas=metas)
        self._bm25.clear()  # the lexical index is rebuilt from Chroma on the next search

    def remove(self, rel):
        """Remove every chunk of one source file from all collections."""
        for c in self.cols.values():
            c.delete(where={"source": rel})
        self._bm25.clear()

    # ---- searching -----------------------------------------------------
    def dense(self, col, query, n, where=None):
        """Nearest chunks of one Chroma collection: [{id, text, meta, distance}]. The one place a query is embedded
        and run, used by the retriever."""
        if col.count() == 0:
            return []
        r = col.query(query_embeddings=self.embed([query]), n_results=min(n, col.count()), where=where)
        return [{"id": i, "text": d, "meta": m, "distance": x}
                for i, d, m, x in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    @staticmethod
    def where(genre, beat=None):
        """The Chroma filter for `genre` (plus the shared "any" genre) and optionally one beat. `matches` below is
        the same filter as a Python test; the two live side by side so they cannot drift apart."""
        w = {"genre": {"$in": [genre, "any"]}}
        return {"$and": [w, {"beat": beat}]} if beat else w

    @staticmethod
    def matches(meta, genre, beat=None):
        return meta.get("genre") in (genre, "any") and (not beat or meta.get("beat") == beat)

    def lexical(self, name, query, n, genre, beat=None):
        """BM25 matches of one collection for the same filter as `where`: [{id, text, meta}], best first."""
        if name not in self._bm25:
            d = self.cols[name].get()
            bm = BM25Okapi([tok(t) for t in d["documents"]]) if d["documents"] else None
            self._bm25[name] = (bm, d["ids"], d["documents"], d["metadatas"])
        bm, ids, texts, metas = self._bm25[name]
        if bm is None:
            return []
        scores = bm.get_scores(tok(query))
        top = sorted((j for j in range(len(ids)) if scores[j] > 0 and self.matches(metas[j], genre, beat)),
                     key=lambda j: -scores[j])[:n]
        return [{"id": ids[j], "text": texts[j], "meta": metas[j]} for j in top]


class Retriever:
    def __init__(self, store, cfg):
        self.store, self.cfg = store, cfg

    def search(self, coll, query, genre, beat=None, k=None):
        r = self.cfg["retrieval"]
        k = k or r["final_k"]
        col = self.store.cols[coll]
        if col.count() == 0:
            return []
        dense = [{"id": d["id"], "text": d["text"], "meta": d["meta"]}
                 for d in self.store.dense(col, query, r["candidates"], Store.where(genre, beat))]
        lexical = self.store.lexical(coll, query, r["candidates"], genre, beat)
        docs = {d["id"]: d for d in dense + lexical}

        fused = {}
        for ranked in (dense, lexical):
            for pos, d in enumerate(ranked):
                fused[d["id"]] = fused.get(d["id"], 0) + 1 / (r["rrf_k"] + pos + 1)
        out = [docs[i] for i in sorted(fused, key=fused.get, reverse=True)]
        return out[:k]
