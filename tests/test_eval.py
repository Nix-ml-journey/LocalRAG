"""Offline tests of the eval helpers: retrieval ranks, the pairwise judge and saved-script lookups."""
from ragscript import config


def test_eval_ranks_are_computed_once_and_give_both_the_hit_and_the_top_k():
    import sys
    sys.path.insert(0, str(config.ROOT))
    from eval.run_eval import ranks, top_k

    class Ret:
        def search(self, coll, query, genre, k=4):
            return [{"meta": {"source": "lore_a.md"}}, {"meta": {"source": "x.txt"}}] if coll == "lore" else []

    class PL:
        ret = Ret()

    rk = ranks(PL, {"brief": "q", "genre": "history", "expect_sources": ["lore_a", "missing"]})
    assert rk == {"lore_a": 1, "missing": 11}  # not found = k + 1
    assert top_k([rk]) == {"top1": 0.5, "top3": 0.5, "top10": 0.5}


def test_script_of_cuts_the_narration_out_of_a_saved_file():
    from ragscript.report import script_of
    assert script_of("# T\n\n## Script\n\nOne.\n\nTwo.\n\n## Chapters\n- a") == "One.\n\nTwo."


def test_compare_replies_are_parsed_and_the_two_orderings_combined():
    from ragscript.refine import combine, parse_compare
    reply = 'Premise: A | fine | "x"\nHook: tie | same | "y"\nLogic: a clear hole | no | "z"\nOverall: B | better | "w"'
    first = parse_compare(reply)
    assert first == {"Premise": "A", "Hook": "tie", "Overall": "B"}  # "a clear hole" is not a vote for A
    swapped = {"Premise": "B", "Hook": "tie", "Overall": "B"}  # asked with the scripts exchanged
    # Overall: the judge followed position
    assert combine(first, swapped) == {"Premise": "A", "Hook": "tie", "Overall": "tie"}


def test_pairs_match_the_same_brief_across_two_eval_folders(tmp_path, monkeypatch):
    from eval import run_eval
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for folder, stamp in (("a", "1119-070334"), ("b", "1119-080000")):
        (tmp_path / "output" / folder).mkdir(parents=True)
        (tmp_path / "output" / folder / f"horror-5min-a-keeper-{stamp}.md").write_text("x")
        (tmp_path / "output" / folder / f"sci-fi-5min-only-in-{folder}-{stamp}.md").write_text("x")
    found = run_eval.pairs("a", "b")
    assert [(p.name, q.name) for p, q in found] == [
        ("horror-5min-a-keeper-1119-070334.md", "horror-5min-a-keeper-1119-080000.md")]


def test_compare_prompt_fills_every_placeholder():
    from ragscript.refine import CRITERIA, parse_compare
    from ragscript import prompts
    out = prompts.stage(config.load(), "compare", (), genre="horror", brief="b", script_a="SA", script_b="SB")
    assert "<<" not in out and "\nSA\n" in out and "\nSB\n" in out
    assert len(parse_compare(out[out.index("The layout below"):])) == len(CRITERIA)  # the sample reply parses fully


def test_the_brief_of_a_saved_script_comes_from_its_trace_else_from_its_title(tmp_path, monkeypatch):
    import json as _json
    from eval import run_eval
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "output").mkdir()
    with_trace, without = tmp_path / "horror-5min-a-1119-070334.md", tmp_path / "horror-5min-b-1119-070335.md"
    with_trace.write_text("# A Title\n", encoding="utf-8")
    without.write_text("# Another Title\n\nbody", encoding="utf-8")
    (tmp_path / "output" / "horror-5min-a-1119-070334.trace.json").write_text(_json.dumps({"brief": "the real brief"}))
    assert run_eval._brief_of(with_trace) == "the real brief" and run_eval._brief_of(without) == "Another Title"
