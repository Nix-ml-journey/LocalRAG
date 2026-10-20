"""Hybrid retrieval: dense (Chroma) + BM25 -> reciprocal rank fusion -> optional cross-encoder rerank."""
from .store import tok


class Retriever:
    def __init__(self, store, cfg):
        self.store, self.cfg, self._rr = store, cfg, None

    def search(self, coll, query, genre, beat=None, k=None):
        r = self.cfg["retrieval"]
        k = k or r["final_k"]
        col = self.store.cols[coll]
        if col.count() == 0:
            return []
        where = {"genre": {"$in": [genre, "any"]}}
        if beat:
            where = {"$and": [where, {"beat": beat}]}

        def ok(m):
            return m.get("genre") in (genre, "any") and (not beat or m.get("beat") == beat)

        dense = col.query(query_embeddings=self.store.embed([query]), n_results=min(r["candidates"], col.count()),
                          where=where)
        docs = {}
        for i, d, m in zip(dense["ids"][0], dense["documents"][0], dense["metadatas"][0]):
            docs[i] = {"id": i, "text": d, "meta": m}
        ranks = [list(docs)]

        bm, ids, texts, metas = self.store.bm25(coll)
        scores = bm.get_scores(tok(query))
        top = sorted((j for j in range(len(ids)) if scores[j] > 0 and ok(metas[j])),
                     key=lambda j: -scores[j])[: r["candidates"]]
        for j in top:
            docs.setdefault(ids[j], {"id": ids[j], "text": texts[j], "meta": metas[j]})
        ranks.append([ids[j] for j in top])

        fused = {}
        for lst in ranks:
            for pos, i in enumerate(lst):
                fused[i] = fused.get(i, 0) + 1 / (r["rrf_k"] + pos + 1)
        out = [docs[i] for i in sorted(fused, key=fused.get, reverse=True)]

        if self.cfg["reranker"]["enabled"] and out:
            if self._rr is None:
                from sentence_transformers import CrossEncoder
                self._rr = CrossEncoder(self.cfg["reranker"]["model"], device=self.cfg["reranker"]["device"])
            out = out[:r["candidates"]]
            sc = self._rr.predict([(query, d["text"]) for d in out])
            out = [d for _, d in sorted(zip(sc, out), key=lambda x: -x[0])]
        return out[:k]
