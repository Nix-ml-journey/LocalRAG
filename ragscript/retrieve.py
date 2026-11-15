"""Hybrid retrieval: dense (Chroma) + BM25 -> reciprocal rank fusion -> optional cross-encoder rerank."""
from .store import Store


class Retriever:
    def __init__(self, store, cfg):
        self.store, self.cfg, self._rr = store, cfg, None

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

        if self.cfg["reranker"]["enabled"] and out:
            if self._rr is None:
                from sentence_transformers import CrossEncoder
                self._rr = CrossEncoder(self.cfg["reranker"]["model"], device=self.cfg["reranker"]["device"])
            out = out[:r["candidates"]]
            sc = self._rr.predict([(query, d["text"]) for d in out])
            out = [d for _, d in sorted(zip(sc, out), key=lambda x: -x[0])]
        return out[:k]
