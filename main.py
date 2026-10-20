"""HTTP service for the story pipeline: python main.py  (http://127.0.0.1:8000).

Serves the two routes the pipeline's StoryForge client calls, backed by this project's Chroma store:
POST /vector_store/query  -> nearest lore chunks plus stored episode summaries
POST /vector_store/insert -> store one episode summary so the next part can recall it
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

EPISODES = "episodes"  # not in store.COLLECTIONS, so ingest never touches or prunes it


class Query(BaseModel):
    query: str
    n_results: int = 5
    collection: str = ""  # the pipeline's collection name; this store has fixed collections
    where: dict | None = None  # Chroma metadata filter, e.g. {"series_id": "s"}; lore has no series_id, so it drops out


class Insert(BaseModel):
    ids: list[str]
    metadata: dict
    collection: str = ""


def create_app(store) -> FastAPI:
    app = FastAPI(title="Story Script RAG service")
    episodes = store.client.get_or_create_collection(EPISODES, metadata={"hnsw:space": "cosine"})

    @app.get("/")
    def root():
        return {"status": "ok", "lore": store.cols["lore"].count(), "episodes": episodes.count()}

    @app.post("/vector_store/query")
    def query(req: Query):
        # shortcut: dense search only, no genre filter or BM25; upgrade if lore gets large or multi-genre
        emb = store.embed([req.query])
        rows = []
        for col in (store.cols["lore"], episodes):  # n from each, so history can't crowd out lore or reverse
            if col.count() == 0:
                continue
            r = col.query(query_embeddings=emb, n_results=min(req.n_results, col.count()), where=req.where)
            for i, d, m, x in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
                rows.append({"id": i, "document": d, "metadata": {"Title": m.get("source", ""), **m}, "distance": x})
        rows.sort(key=lambda row: row["distance"])
        return {"success": True, "results": rows}

    @app.post("/vector_store/insert")
    def insert(req: Insert):
        meta = dict(req.metadata)
        doc = str(meta.pop("document", "")).strip()
        if len(req.ids) != 1 or not doc:
            raise HTTPException(422, "send exactly one id and metadata.document")
        meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}  # Chroma takes scalars only
        episodes.upsert(ids=req.ids, documents=[doc], embeddings=store.embed([doc]), metadatas=[meta])
        return {"success": True}

    return app


if __name__ == "__main__":
    import uvicorn

    from ragscript import config
    from ragscript.store import Store

    store = Store(config.load())
    store.embed(["warm up"])  # load bge-m3 now: the first query otherwise takes ~40s and trips the client's 30s timeout
    uvicorn.run(create_app(store), host="127.0.0.1", port=8000)
