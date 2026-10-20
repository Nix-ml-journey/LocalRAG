"""The pipeline-facing HTTP service: real Chroma in a temp dir, fake embedder (no model download)."""
import sys
from pathlib import Path

import chromadb
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import create_app


class FakeStore:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path=str(path))
        self.cols = {"lore": self.client.get_or_create_collection("lore", metadata={"hnsw:space": "cosine"})}

    def embed(self, texts):
        return [[1.0 if "castle" in t else 0.0, 1.0 if "ship" in t else 0.0, 0.1] for t in texts]


def client(tmp_path):
    store = FakeStore(tmp_path)
    store.cols["lore"].add(ids=["a", "b"], documents=["the castle fell", "the ship sank"],
                           embeddings=store.embed(["castle", "ship"]), metadatas=[{"source": "lore_castle.md"}] * 2)
    return TestClient(create_app(store))


def test_query_returns_nearest_lore_with_a_title(tmp_path):
    r = client(tmp_path).post("/vector_store/query", json={"query": "castle", "n_results": 1}).json()
    assert r["success"] and r["results"][0]["document"] == "the castle fell"
    assert r["results"][0]["metadata"]["Title"] == "lore_castle.md"


def test_inserted_episode_comes_back_with_its_metadata(tmp_path):
    c = client(tmp_path)
    meta = {"document": "ship story summary", "doc_type": "episode_summary", "series_id": "s", "part_number": 2}
    assert c.post("/vector_store/insert", json={"ids": ["ep1"], "metadata": meta}).json() == {"success": True}
    rows = c.post("/vector_store/query", json={"query": "ship", "n_results": 3}).json()["results"]
    ep = next(r for r in rows if r["id"] == "ep1")
    assert ep["document"] == "ship story summary" and ep["metadata"]["part_number"] == 2


def test_insert_rejects_a_missing_document(tmp_path):
    r = client(tmp_path).post("/vector_store/insert", json={"ids": ["x"], "metadata": {"doc_type": "t"}})
    assert r.status_code == 422


def test_empty_store_queries_return_nothing(tmp_path):
    store = FakeStore(tmp_path)
    r = TestClient(create_app(store)).post("/vector_store/query", json={"query": "castle"}).json()
    assert r == {"success": True, "results": []}


def test_history_query_can_be_limited_to_one_series(tmp_path):
    c = client(tmp_path)
    for series, i in (("a", "ep_a"), ("b", "ep_b")):
        meta = {"document": "ship story", "doc_type": "episode_summary", "series_id": series, "part_number": 1}
        c.post("/vector_store/insert", json={"ids": [i], "metadata": meta})
    body = {"query": "ship", "n_results": 5, "where": {"series_id": "a"}}
    ids = [r["id"] for r in c.post("/vector_store/query", json=body).json()["results"]]
    assert ids == ["ep_a"]
