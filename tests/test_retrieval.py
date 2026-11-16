"""Retriever on a real Chroma store with a fake embedder: filters and rank fusion run offline."""
from ragscript import config
from ragscript.store import Retriever, Store


def embed(texts):  # three topics plus a constant, so no vector is zero
    return [[float("castle" in t), float("ship" in t), float("ghost" in t), 0.1] for t in (s.lower() for s in texts)]


def retriever(tmp_path):
    cfg = config.load()
    cfg["paths"]["chroma_dir"] = str(tmp_path / "db")
    st = Store(cfg, embed=embed)
    st.add("scripts", "a.txt", [("hook", "the castle fell in the night"), ("climax", "a ship sank near the castle")],
           {"genre": "history", "source": "a.txt"})
    st.add("scripts", "b.txt", [("hook", "a ghost walked the halls"), ("hook", "a ghost climbed the stairs")],
           {"genre": "horror", "source": "b.txt"})
    return Retriever(st, cfg)


def sources(rows):
    return {r["meta"]["source"] for r in rows}


def test_search_is_limited_to_the_genre(tmp_path):
    r = retriever(tmp_path)
    assert sources(r.search("scripts", "castle", "history")) == {"a.txt"}
    assert sources(r.search("scripts", "castle", "horror")) == {"b.txt"}


def test_search_can_be_limited_to_one_beat(tmp_path):
    rows = retriever(tmp_path).search("scripts", "castle", "history", beat="climax")
    assert [x["meta"]["beat"] for x in rows] == ["climax"]


def test_a_word_match_lifts_a_chunk_that_the_vectors_cannot_tell_apart(tmp_path):
    rows = retriever(tmp_path).search("scripts", "ghost halls", "horror")
    assert "halls" in rows[0]["text"]  # both ghost chunks tie on the vectors; BM25 puts "halls" first


def test_an_empty_collection_returns_nothing(tmp_path):
    assert retriever(tmp_path).search("lore", "castle", "history") == []
