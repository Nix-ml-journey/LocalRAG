"""Offline tests of the pipeline stages: model JSON, verify, chunking, ingest, prompts, reports and budgets."""
from support import FakeLLM, FakeRetriever, ingest_cfg
from support import make_pipeline as pipeline
from support import write_script as write

from ragscript import chunking, config
from ragscript.ingest import Ingest

CFG = config.load()


def test_package_answer_that_is_not_an_object_is_retried(tmp_path):
    class L(FakeLLM):
        packs = 0

        def chat(self, prompt, **kw):
            if kw.get("stage") == "package":
                self.packs += 1
                return '["a", "b"]' if self.packs == 1 else super().chat(prompt, **kw)
            return super().chat(prompt, **kw)

    llm = L()
    md = write(pipeline(tmp_path, llm))["markdown"]
    assert llm.packs == 2 and "## Title options\n1. a" in md


def test_verify_uses_the_lore_chunk_that_matches_the_sentence(tmp_path):
    fact = "Captain Morehouse sailed from Gibraltar in November."

    class R(FakeRetriever):
        n = 0

        def search(self, coll, *a, **k):
            if coll != "lore":
                return []
            out = []
            for _ in range(4):
                R.n += 1
                text = fact if R.n == 20 else f"Filler note {R.n} about unrelated matters."
                out.append({"id": f"lore:f.md:{R.n}", "text": text, "meta": {"source": "f.md"}})
            return out

    class L(FakeLLM):
        def chat(self, prompt, **kw):
            if kw.get("stage") == "verify":
                lore, sentence = prompt.split("Sentence from the script:")
                return "YES" if "Morehouse" in sentence and "Gibraltar in November" in lore else "NO"
            text = super().chat(prompt, **kw)
            if kw.get("stage") == "section" and not self.written[1:]:
                text = fact + " " + text
                self.written[-1] = text
            return text

    md = write(pipeline(tmp_path, L(), R()), genre="history")["markdown"]
    script = md[md.index("## Script"): md.index("## Chapters")]
    assert "Captain Morehouse sailed" in script
    assert "{{INVENTED: Captain Morehouse" not in script, "the sentence's own source was left out of the check"


def test_text_without_full_stops_is_still_chunked():
    text = " ".join(f"w{i}" for i in range(3000))
    chunks = chunking.sliding(text, CFG["chunking"]["min_tokens"], CFG["chunking"]["max_tokens"], CFG["chunking"]["overlap"])
    assert len(chunks) > 5
    assert all(chunking.tokens(c) <= CFG["chunking"]["max_tokens"] for c in chunks)


def test_deleted_file_leaves_no_chunks_even_if_its_tagging_failed(tmp_path, monkeypatch):
    from ragscript.store import Store
    cfg = ingest_cfg(tmp_path, "state.json")
    (tmp_path / "data/horror").mkdir(parents=True)
    f = tmp_path / "data/horror/a.txt"
    f.write_text("The tide came in. " * 80, encoding="utf-8")
    st = Store(cfg)
    monkeypatch.setattr(st, "embed", lambda texts: [[1.0, 0.0] for _ in texts])

    def tagger_down(text):
        raise ConnectionError("ollama is down")

    assert Ingest(st, cfg).run(tagger_down)["scripts"] > 0  # indexed, but the file is not marked done
    f.unlink()
    assert Ingest(st, cfg).run(tagger_down) == {"scripts": 0, "lore": 0, "hooks": 0}


def test_every_genre_gets_the_shared_tense_and_pause_rules(tmp_path):
    from ragscript import genre
    for name in ("horror", "history", "sci-fi", "unknown-genre"):
        assert "past tense" in genre.load(CFG, name).rules and "[PAUSE]" in genre.load(CFG, name).rules


def test_store_filter_exists_once_as_chroma_where_and_python_test(tmp_path, monkeypatch):
    from ragscript.store import Store
    cfg = ingest_cfg(tmp_path)
    fm = "---\nsubgenre: s\ntone: t\npov: first\n---\n"
    for genre, name, word in (("horror", "a", "tide"), ("fantasy", "a", "tide"), ("horror", "b", "lantern"),
                              ("horror", "c", "ladder"), ("horror", "d", "cellar"), ("horror", "e", "window")):
        (tmp_path / "data" / genre).mkdir(parents=True, exist_ok=True)
        text = fm + f"The {word} stood near the old {genre} wall. " * 60
        (tmp_path / "data" / genre / f"{name}.txt").write_text(text, encoding="utf-8")
    st = Store(cfg)
    monkeypatch.setattr(st, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    Ingest(st, cfg).run(lambda text: "{}")
    hits = st.lexical("scripts", "tide", 5, "horror")
    assert hits and all(h["meta"]["genre"] == "horror" for h in hits)  # the fantasy file is filtered out
    assert Store.matches({"genre": "any"}, "horror") and not Store.matches({"genre": "fantasy"}, "horror")
    assert not Store.matches({"genre": "horror", "beat": "hook"}, "horror", "setup")
    assert Store.where("horror", "hook") == {"$and": [{"genre": {"$in": ["horror", "any"]}}, {"beat": "hook"}]}
    assert st.lexical("scripts", "tide", 5, "cthulhu") == []


def test_interrupted_ingest_keeps_the_progress_of_the_files_already_done(tmp_path, monkeypatch):
    import json as _json
    import pytest
    from ragscript.store import Store
    cfg = ingest_cfg(tmp_path)
    (tmp_path / "data/horror").mkdir(parents=True)
    fm = """---
subgenre: s
tone: t
pov: first
---
"""
    for name in ("a", "b"):
        text = fm + f"The {name} tide came in over the rocks. " * 80
        (tmp_path / "data/horror" / f"{name}.txt").write_text(text, encoding="utf-8")
    st = Store(cfg)
    calls = []

    def embed(texts):
        calls.append(1)
        if len(calls) > 2:  # each file embeds twice (scripts, hooks): the second file is "killed" half way
            raise KeyboardInterrupt
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(st, "embed", embed)
    with pytest.raises(KeyboardInterrupt):
        Ingest(st, cfg).run(lambda text: "{}")
    state = _json.loads((tmp_path / "db/s.json").read_text())
    assert list(state["files"]) == ["horror/a.txt"]  # the finished file is recorded, so the next run skips it


def test_a_value_that_looks_like_a_placeholder_is_inserted_as_it_is():
    from ragscript import prompts
    out = prompts.stage(config.load(), "revise", ["bad phrase"], text="the lore says <<banned>> here", narrator="n",
                        genre_rules="g", problems="p", target_words=5)
    assert out.count("<<banned>>") == 1 and "bad phrase" in out  # the template's own placeholder was filled


def test_report_measures_the_script_and_only_fact_checks_grounded_genres():
    from ragscript import report
    cfg = config.load()
    script = "Captain Briggs sailed in 1872. [PAUSE] {{INVENTED: A cold wind rose.}}"
    kw = dict(target_words=10, minutes=5, refs=[], cited_text={}, banned=[], cfg=cfg, notes=[])
    plain = report.build(script, grounded=False, **kw)
    assert plain.uncited_facts == [] and plain.pause_cues == 1 and plain.invented == ["A cold wind rose."]
    assert report.build(script, grounded=True, **kw).uncited_facts == ["Captain Briggs sailed in 1872."]
    assert plain.sentence_range == "12-16"


def test_report_decides_which_warnings_to_show_and_render_only_prints_them():
    from ragscript import report
    cfg = config.load()
    kw = dict(target_words=10, minutes=5, refs=[], cited_text={}, grounded=False, banned=[], cfg=cfg, notes=[])
    calm = report.build("I walked in. [PAUSE] Then I left.", **kw)
    assert calm.flags == [] and calm.pauses_per_minute == 0.2
    halting = report.build("I walked in. " + "[PAUSE] I stood. " * 12, **kw)
    pause = next(f for f in halting.flags if "[PAUSE] cues in 5 minutes" in f)
    md = report.render(report.Doc("b", "horror", 5, "n", {}, [(0, "c")], {}), "text", halting)
    assert all("- " + f in md for f in halting.flags) and "- " + pause in md


def test_rolling_budget_spreads_an_overshoot_over_the_beats_still_to_write():
    from ragscript.draft import Drafter
    beats = [chunking.Beat("hook", 100, 100) for _ in range(3)]
    sections = [(beats[0], "word " * 250)]  # 250 written for a 100-word beat: counts only up to its plan
    Drafter._rebudget(beats, 1, sections, 300)
    assert beats[1].words == beats[2].words == 100 * 200 // 200  # 100 done, 200 left for 200 planned
    sections = [(beats[0], "word " * 40)]  # an undershoot is made up by the later beats
    Drafter._rebudget(beats, 1, sections, 300)
    assert beats[1].words > 100


def test_prompt_filling_uses_config_values_and_leaves_unknown_placeholders():
    from ragscript import prompts
    cfg = config.load()
    out = prompts.fill("range <<sentence_range>>, pause <<max_pause>>, <<unknown>>, <<name>>", cfg, (), name="<<banned>>")
    assert out == "range 12-16, pause 1, <<unknown>>, <<banned>>"


def test_two_runs_of_the_same_brief_keep_both_files(tmp_path, monkeypatch):
    import time
    stamps = iter(["0101-000001", "0101-000002"])
    monkeypatch.setattr(time, "strftime", lambda fmt: next(stamps))
    a = write(pipeline(tmp_path))["path"]
    b = write(pipeline(tmp_path))["path"]
    assert a != b and a.exists() and b.exists()


def test_save_never_overwrites_an_earlier_script(tmp_path, monkeypatch):
    import time
    from ragscript import report
    monkeypatch.setattr(time, "strftime", lambda fmt: "0101-000001")
    a = report.save(tmp_path, "horror", 5, "A Brief!", "one", {"k": 1})
    b = report.save(tmp_path, "horror", 5, "A Brief!", "two", {"k": 2})
    assert a != b and a.read_text(encoding="utf-8") == "one" and b.read_text(encoding="utf-8") == "two"
    assert a.name.startswith("horror-5min-a-brief-") and a.with_suffix(".trace.json").exists()


def test_a_file_straight_in_data_is_skipped_with_a_warning_not_indexed_under_its_own_name(tmp_path, monkeypatch, caplog):
    from ragscript.store import Store
    cfg = ingest_cfg(tmp_path)
    (tmp_path / "data/horror").mkdir(parents=True)
    fm = "---\nsubgenre: s\ntone: t\npov: first\n---\n"
    (tmp_path / "data/stray.txt").write_text(fm + "The tide came in. " * 80, encoding="utf-8")
    (tmp_path / "data/horror/a.txt").write_text(fm + "The tide came in. " * 80, encoding="utf-8")
    st = Store(cfg)
    monkeypatch.setattr(st, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    Ingest(st, cfg).run(lambda text: "{}")
    assert st.sources() == {"horror/a.txt"} and "skipping stray.txt" in caplog.text
