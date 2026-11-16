"""Regression tests for the audit findings: bad model JSON, verify evidence, length fit with line breaks, long text
without full stops. Same offline stubs as test_pipeline_guard.py."""
import json

from support import BEATS, FakeLLM, FakeRetriever
from support import make_pipeline as pipeline
from support import write_script as write
from test_length_guard import LengthLLM

from ragscript import checks, chunking, config
from ragscript.ingest import Ingest
from ragscript.pipeline import Story

CFG = config.load()
CK = CFG["checks"]


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


def test_outline_with_loose_beat_names_and_bad_words_is_fixed_or_retried(tmp_path):
    class L(FakeLLM):
        outlines, section_prompts = 0, []

        def chat(self, prompt, **kw):
            if kw.get("stage") == "outline":
                self.outlines += 1
                names = [b.title().replace("_", " ") for b in BEATS]  # "Hook", "Rising Tension"...
                words = "about 100" if self.outlines == 1 else 107
                beats = [{"beat": n, "title": n, "words": words, "summary": "s", "place": f"room {i}", "change": f"new event {i * 11} {'ab cd ef gh ij kl mn'.split()[i]} xyz", "detail": ""}
                     for i, n in enumerate(names)]
                return json.dumps({"narrator": "Eli", "beats": beats})
            if kw.get("stage") == "section":
                self.section_prompts.append(prompt)
            return super().chat(prompt, **kw)

    llm = L()
    write(pipeline(tmp_path, llm))
    assert llm.outlines == 2, "words that are not a number must make the outline attempt fail"
    assert "Open with this chosen hook" in llm.section_prompts[0], "'Hook' must be read as the hook beat"


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


def test_length_fit_accepts_a_rewrite_with_a_line_break(tmp_path):
    class L(LengthLLM):
        def chat(self, prompt, **kw):
            out = super().chat(prompt, **kw)
            if kw.get("stage") == "fit":
                w = out.split()
                out = " ".join(w[:5]) + ".\n" + " ".join(w[5:])  # a break after a full stop is what dedupe flattens
            return out

    md = write(pipeline(tmp_path, L(section_words=150)))["markdown"]
    n = checks.spoken_count(md[md.index("## Script") + 9: md.index("## Chapters")])
    assert 675 <= n <= 825, n


def test_text_without_full_stops_is_still_chunked():
    text = " ".join(f"w{i}" for i in range(3000))
    chunks = chunking.sliding(text, CFG["chunking"]["min_tokens"], CFG["chunking"]["max_tokens"], CFG["chunking"]["overlap"])
    assert len(chunks) > 5
    assert all(chunking.tokens(c) <= CFG["chunking"]["max_tokens"] for c in chunks)


def test_deleted_file_leaves_no_chunks_even_if_its_tagging_failed(tmp_path, monkeypatch):
    from ragscript.store import Store
    cfg = config.load()
    cfg["paths"].update(data_dir=str(tmp_path / "data"), chroma_dir=str(tmp_path / "db"), state_file=str(tmp_path / "db/state.json"))
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


def test_outline_retries_with_thinking_off_when_thinking_returns_no_json(tmp_path):
    class L(FakeLLM):
        think_flags = []

        def chat(self, prompt, **kw):
            if kw.get("stage") == "outline":
                self.think_flags.append(kw.get("think"))
                if kw.get("think"):
                    return "I am still thinking about the story..."  # budget spent, no JSON
            return super().chat(prompt, **kw)

    llm = L()
    pl = pipeline(tmp_path, llm)
    pl.cfg["llm"]["think_outline"] = True  # off by default; this is the path for anyone who turns it on
    write(pl)
    assert llm.think_flags[0] is True and llm.think_flags[1:] == [False], "one thinking attempt, then plain ones"


def test_weakest_section_is_revised_only_when_the_rewrite_has_fewer_flaws(tmp_path):
    from ragscript import genre
    from ragscript.refine import Refiner
    cfg = config.load()
    g = genre.load(cfg, "horror")
    bad = "I walked in. [PAUSE] I see the door. [PAUSE] I knew it. [PAUSE] I find the key. I looked up."
    good = "I walked in and saw the door. I knew it was locked. [PAUSE] I found the key."

    parts, notes = ["Plain text here.", bad], []
    out = Refiner(lambda stage, vars, **kw: good, cfg, []).revise_weakest(parts, Story("b", g, "Eli", 750, 5, notes=notes))
    assert out[1] == good and out[0] == parts[0] and "Revised section 2" in notes[0]
    notes.clear()
    # a rewrite that fixes nothing is not kept
    assert Refiner(lambda stage, vars, **kw: bad, cfg, []).revise_weakest(parts, Story("b", g, "Eli", 750, 5, notes=notes)) == parts and "rejected" in notes[0]


def test_every_genre_gets_the_shared_tense_and_pause_rules(tmp_path):
    from ragscript import genre
    for name in ("horror", "history", "sci-fi", "unknown-genre"):
        assert "past tense" in genre.load(CFG, name).rules and "[PAUSE]" in genre.load(CFG, name).rules


def test_cue_tense_and_crutch_word_checks():
    assert checks.mixed_tense("I walked in. I saw it. I knew. I see the door. I hear it. I feel cold.", CK)
    assert not checks.mixed_tense("I walked in. I saw it. I knew. I found the key. She runs.", CK)
    assert not checks.mixed_tense("She walked in and saw him.", CK)  # third person never fires
    text = " ".join(["The heavy door shut."] * 6 + ["Then silence fell over everything we owned here."] * 2)
    assert checks.overused_words(text, CK)[0][0] == "heavy"


def test_store_filter_exists_once_as_chroma_where_and_python_test(tmp_path, monkeypatch):
    from ragscript.store import Store
    cfg = config.load()
    cfg["paths"].update(data_dir=str(tmp_path / "data"), chroma_dir=str(tmp_path / "db"), state_file=str(tmp_path / "db/s.json"))
    fm = "---\nsubgenre: s\ntone: t\npov: first\n---\n"
    for genre, name, word in (("horror", "a", "tide"), ("fantasy", "a", "tide"), ("horror", "b", "lantern"),
                              ("horror", "c", "ladder"), ("horror", "d", "cellar"), ("horror", "e", "window")):
        (tmp_path / "data" / genre).mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / genre / f"{name}.txt").write_text(fm + f"The {word} stood near the old {genre} wall. " * 60, encoding="utf-8")
    st = Store(cfg)
    monkeypatch.setattr(st, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    Ingest(st, cfg).run(lambda text: "{}")
    hits = st.lexical("scripts", "tide", 5, "horror")
    assert hits and all(h["meta"]["genre"] == "horror" for h in hits)  # the fantasy file is filtered out
    assert Store.matches({"genre": "any"}, "horror") and not Store.matches({"genre": "fantasy"}, "horror")
    assert not Store.matches({"genre": "horror", "beat": "hook"}, "horror", "setup")
    assert Store.where("horror", "hook") == {"$and": [{"genre": {"$in": ["horror", "any"]}}, {"beat": "hook"}]}
    assert st.lexical("scripts", "tide", 5, "cthulhu") == []


def test_draft_and_final_issues_judge_the_two_stages_with_named_entry_points():
    from ragscript.refine import draft_issues, final_issues
    cfg = config.load()
    twice = "I walked in. [PAUSE] I see the door. [PAUSE] I knew it. [PAUSE] I find the key. I hear a step. I looked up."
    assert not draft_issues(twice, "", cfg, [], budget=22)  # a draft is not judged on cues
    kinds = [i.kind for i in final_issues(twice, "", cfg, [])]
    assert "pause" in kinds and "tense" in kinds
    assert draft_issues("word " * 400, "", cfg, [], budget=100)[0].kind == "length"
    assert [i.kind for i in final_issues("A fine line.", "", cfg, ["fine line"])] == ["banned"]


def test_interrupted_ingest_keeps_the_progress_of_the_files_already_done(tmp_path, monkeypatch):
    import json as _json
    import pytest
    from ragscript.store import Store
    cfg = config.load()
    cfg["paths"].update(data_dir=str(tmp_path / "data"), chroma_dir=str(tmp_path / "db"), state_file=str(tmp_path / "db/s.json"))
    (tmp_path / "data/horror").mkdir(parents=True)
    fm = """---
subgenre: s
tone: t
pov: first
---
"""
    for name in ("a", "b"):
        (tmp_path / "data/horror" / f"{name}.txt").write_text(fm + f"The {name} tide came in over the rocks. " * 80, encoding="utf-8")
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


def test_dialogue_in_another_tense_is_not_counted_as_a_tense_switch():
    narration = "I walked in. I saw the door. I knew it was locked. "
    speech = 'He said, "I see it, I hear it, I feel it." '
    assert not checks.mixed_tense(narration + speech, CK)
    assert checks.mixed_tense(narration + "I see it. I hear it. I feel it.", CK)


def test_polish_and_fit_outputs_lose_invented_bracket_notes():
    from ragscript import genre
    from ragscript.refine import Refiner
    cfg = config.load()
    g = genre.load(cfg, "horror")
    sections = [(chunking.Beat("hook", 20), "I opened the door and listened for a long quiet moment in the hall.")]
    out = Refiner(lambda stage, vars, **kw: "I opened the door [recurring] and listened for a long quiet moment in the hall.",
                  cfg, []).polish(sections, Story("b", g, "Eli", 750, 5))
    assert "[recurring]" not in out[0]


def test_a_value_that_looks_like_a_placeholder_is_inserted_as_it_is():
    from ragscript import prompts
    out = prompts.stage(config.load(), "revise", ["bad phrase"], text="the lore says <<banned>> here", narrator="n",
                        genre_rules="g", problems="p", target_words=5)
    assert out.count("<<banned>>") == 1 and "bad phrase" in out  # the template's own placeholder was filled


def test_revision_pass_fixes_up_to_the_configured_number_of_sections():
    from ragscript import genre
    from ragscript.refine import Refiner
    cfg = config.load()
    g = genre.load(cfg, "horror")
    bad = "I walked in. [PAUSE] I stood still. [PAUSE] I found the key."
    fixed = "I walked in and stood still. I found the key."
    parts, notes = ["Fine start here.", bad, bad.replace("key", "lock"), bad.replace("key", "door")], []
    calls = []

    def ask(stage, vars, **kw):
        calls.append(stage)
        return fixed

    out = Refiner(ask, cfg, []).revise_weakest(parts, Story("b", g, "Eli", 750, 5, notes=notes))
    assert calls == ["revise"] * cfg["checks"]["revise_sections"]  # two rewrites, not three
    assert sum(1 for p in out if p == fixed) == cfg["checks"]["revise_sections"] and out[0] == parts[0]


def test_report_measures_the_script_and_only_fact_checks_grounded_genres():
    from ragscript import report
    cfg = config.load()
    script = "Captain Briggs sailed in 1872. [PAUSE] {{INVENTED: A cold wind rose.}}"
    kw = dict(target_words=10, minutes=5, refs=[], cited_text={}, banned=[], cfg=cfg, notes=[])
    plain = report.build(script, grounded=False, **kw)
    assert plain.uncited_facts == [] and plain.pause_cues == 1 and plain.invented == ["A cold wind rose."]
    assert report.build(script, grounded=True, **kw).uncited_facts == ["Captain Briggs sailed in 1872."]
    assert plain.sentence_range == "12-16"


def test_attempt_keeps_an_accepted_rewrite_cleans_it_and_reverts_a_rejected_one():
    from ragscript.refine import attempt
    new, kept = attempt("old text.", lambda: "A fine rewrite [recurring] of it.", lambda n: "fine" in n)
    assert kept and new == "A fine rewrite of it."  # the invented note is stripped before the text is judged
    old, kept = attempt("old text.", lambda: "A poor rewrite.", lambda n: "fine" in n)
    assert not kept and old == "old text."


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


def test_a_paragraph_retold_in_new_words_is_dropped_but_a_continuation_is_kept():
    first = ("I checked every monitor in the booth to see if the lag was a local glitch. Each screen showed a different "
             "wing of the warehouse, but the delay remained consistent across every feed. The steady hum of the HVAC "
             "filled the room while I waited for a flicker.")
    reworded = ("I scanned each monitor row to see if the lag affected just one specific camera feed. Every display "
                "showed a different wing of the warehouse, yet the timing stayed consistent across all feeds. The "
                "steady hum of the HVAC filled the booth while I waited for a glitch.")
    new_scene = ("I stepped into the corridor while the sharp chemical sting of floor wax filled my nose. My hands "
                 "gripped the heavy keys at my belt as I walked toward the first door.")
    assert checks.drop_similar(first + "\n\n" + reworded, "", 0.5, 10) == first  # the retelling goes
    assert checks.drop_similar(first + "\n\n" + new_scene, "", 0.5, 10) == first + "\n\n" + new_scene
    assert checks.drop_similar(reworded, first, 0.5, 10) == ""  # also against text from earlier sections
    assert checks.drop_similar("Too short.", "Too short.", 0.5, 10) == "Too short."  # tiny paragraphs are not compared
    assert checks.dedupe(first + "\n\n" + reworded, CK) == first  # and dedupe applies it


def test_words_the_brief_itself_uses_are_not_crutch_words():
    brief = "A night-shift security guard keeps seeing himself on the monitors"
    skip = checks.topic_words(brief)
    assert "monitor" in skip and "guard" in skip
    text = " ".join(["The monitor flickered and the monitor hissed over the heavy door and the heavy latch."] * 3)
    assert [w for w, _ in checks.overused_words(text, CK)] == ["monitor", "heavy"]
    assert [w for w, _ in checks.overused_words(text, CK, skip=skip)] == ["heavy"]  # the topic noun is ignored


def test_a_section_retelling_earlier_text_in_new_words_is_an_issue_and_dedupe_parts_empties_it():
    from ragscript import genre
    from ragscript.refine import Refiner, draft_issues
    cfg = config.load()
    first = ("I checked every monitor in the booth to see if the lag was a local glitch. Each screen showed a different "
             "wing of the warehouse, but the delay remained consistent across every feed. The steady hum of the HVAC "
             "filled the room while I waited for a flicker.")
    retold = ("I scanned each monitor row to see if the lag affected just one specific camera feed. Every display "
              "showed a different wing of the warehouse, yet the timing stayed consistent across all feeds. The "
              "steady hum of the HVAC filled the booth while I waited for a glitch.")
    assert "similar" in [i.kind for i in draft_issues(retold, first, cfg, [], budget=len(retold.split()))]
    story = Story("b", genre.load(cfg, "horror"), "Eli", 750, 5)
    out = Refiner(lambda *a, **k: "", cfg, []).dedupe_parts([first, retold], story)
    assert out == [first, ""] and "Repeat guard removed" in story.notes[0]


def test_the_repeat_guard_only_notes_removals_not_re_flowed_lines():
    from ragscript import genre
    from ragscript.refine import Refiner
    cfg = config.load()
    story = Story("b", genre.load(cfg, "horror"), "Eli", 750, 5)
    lined = "I walked in and looked around the long hall.\nThen I heard a sound from the cellar below the stairs."
    out = Refiner(lambda *a, **k: "", cfg, []).dedupe_parts([lined], story)
    assert checks.spoken_count(out[0]) == checks.spoken_count(lined) and story.notes == []


def test_repeated_wording_lists_unchanged_returns_only():
    from ragscript import checks
    same = ("The steady hum of the vents filled the room. I walked down the hall alone. "
            "The steady hum of the vents filled the room. A door closed behind me somewhere. "
            "The steady hum of the vents filled the room.")
    changed = ("The steady hum of the vents filled the room. I walked down the hall alone. "
               "The hum stalled for one heartbeat. A door closed behind me somewhere. Then the vents breathed again.")
    assert checks.repeated_wording(same)[0][1] == 3
    assert checks.repeated_wording(changed) == []


def test_outline_issues_find_logic_holes():
    from ragscript import checks
    beats = [chunking.Beat(beat="hook", place="booth", change="the monitor shows him early", detail="hum: steady"),
             chunking.Beat(beat="setup", place="", change="the monitor shows him early again", detail="hum: louder"),
             chunking.Beat(beat="reveal", place="hall", change="a door opens by itself", detail="hum: stalls"),
             chunking.Beat(beat="climax", place="closet", change="the handle turns slowly now", detail="hum: dies")]
    found = " | ".join(checks.outline_issues(beats, 3))
    assert 'no "place"' in found and "almost the same thing" in found and "again right after" in found
    assert 'the detail "hum" is used in 4 beats' in found
    assert checks.outline_issues(beats[2:3], 3) == []


def test_critic_issues_reach_the_revision_prompt(tmp_path):
    from tests.support import FakeLLM
    seen = []

    class L(FakeLLM):
        def chat(self, prompt, **kw):
            if kw.get("stage") == "critic":
                return "2: the narrator is in the booth and the hall at once\nrubbish line"
            if kw.get("stage") == "revise":
                seen.append(prompt)
            return super().chat(prompt, **kw)

    write(pipeline(tmp_path, L()))
    assert any("booth and the hall at once" in p for p in seen)


def test_detail_returns_counts_paragraphs():
    from ragscript import checks
    text = "The HVAC hum filled the room.\n\nI walked on.\n\nThe hum stalled.\n\nNothing else."
    assert checks.detail_returns(text, ["hum"]) == [("hum", 2)]


def test_outline_issue_for_a_screen_beat_without_sees():
    from ragscript import checks
    beat = chunking.Beat(beat="reveal", place="booth", change="the double turns its head", summary="He watches the monitor.")
    assert any('"sees"' in x for x in checks.outline_issues([beat], 3))
    assert not any('"sees"' in x for x in checks.outline_issues([chunking.Beat(**{**beat.__dict__, "sees": "the hallway feed on the booth monitor"})], 3))


def test_detail_returns_ignores_articles_in_the_name():
    from ragscript import checks
    text = "The heavy door shut.\n\nThe heavy rain fell.\n\nI saw the turnstile."
    assert checks.detail_returns(text, ["heavy rattle of the turnstile"]) == [("heavy rattle of the turnstile", 1)]  # counted by "turnstile", not "heavy"


def _outline_with_holes(llm_cls_calls, repaired_holes):
    """A FakeLLM whose first outline has one beat without a place; the repair has `repaired_holes` such beats."""
    class L(FakeLLM):
        def chat(self, prompt, **kw):
            out = super().chat(prompt, **kw)
            if kw.get("stage") != "outline":
                return out
            llm_cls_calls.append(1)
            data = json.loads(out)
            holes = 1 if len(llm_cls_calls) == 1 else repaired_holes
            for b in data["beats"][:holes]:
                b["place"] = ""
            return json.dumps(data)

    return L()


def test_outline_repair_is_kept_when_it_has_fewer_holes(tmp_path):
    calls = []
    md = write(pipeline(tmp_path, _outline_with_holes(calls, 0)))["markdown"]
    assert len(calls) == 2 and "Outline logic problems left" not in md


def test_outline_repair_is_dropped_when_it_is_no_better(tmp_path):
    calls = []
    md = write(pipeline(tmp_path, _outline_with_holes(calls, 2)))["markdown"]
    assert len(calls) == 2 and "Outline logic problems left" in md


def test_two_runs_of_the_same_brief_keep_both_files(tmp_path, monkeypatch):
    import time
    stamps = iter(["0101-000001", "0101-000002"])
    monkeypatch.setattr(time, "strftime", lambda fmt: next(stamps))
    a = write(pipeline(tmp_path))["path"]
    b = write(pipeline(tmp_path))["path"]
    assert a != b and a.exists() and b.exists()


def test_detail_returns_ignores_a_name_word_that_is_everywhere():
    from ragscript import checks
    paras = [f"The floor creaked in room {i}." for i in range(6)] + ["The wax smell came back.", "A wax sting stayed."]
    assert checks.detail_returns("\n\n".join(paras), ["floor wax"]) == [("floor wax", 2)]


def test_fill_sees_copies_the_nearest_screen():
    from ragscript import checks
    beats = [chunking.Beat(beat="hook", summary="He watches the monitor.", sees="Monitor 4"),
             chunking.Beat(beat="setup", summary="He paces the hall.", sees=""),
             chunking.Beat(beat="reveal", summary="The screen flickers.", sees="")]
    assert checks.fill_sees(beats) == 1 and beats[2].sees == "Monitor 4" and beats[1].sees == ""


def test_tidy_cleans_layout_but_keeps_the_words():
    from ragscript import checks
    raw = " First line.  \n\n  It moved. [PAUSE]\n\nThen nothing.\n\n[PAUSE]\n\n[PAUSE] The end."
    out = checks.tidy(raw)
    assert out == "First line.\n\nIt moved.\n\n[PAUSE]\n\nThen nothing.\n\n[PAUSE] The end."
    assert checks.spoken_count(out) == checks.spoken_count(raw)


def test_unused_objects_lists_an_object_found_and_never_used():
    from ragscript import checks
    text = "I noticed a set of brass keys on the floor. I walked on. The door opened. I held a flashlight and the flashlight shook."
    assert checks.unused_objects(text) == ["keys"]


def test_a_logic_rewrite_is_kept_only_when_a_second_critic_call_finds_no_hole(tmp_path):
    from tests.support import FakeLLM

    def run(second_answer):
        class L(FakeLLM):
            calls = 0

            def chat(self, prompt, **kw):
                if kw.get("stage") == "critic":
                    self.calls += 1
                    return "2: the narrator is in two places at once" if self.calls == 1 else second_answer
                return super().chat(prompt, **kw)

        return write(pipeline(tmp_path, L()))["markdown"]

    assert "second critic call found no logic hole left" in run("NONE")
    assert "second critic call still found a logic hole" in run("2: still two places")


def _outline_with(holes):
    beats = [chunking.Beat(b, 100, place="" if i < holes else f"room {i}",
                           change=f"event {i} {'alpha beta gamma delta epsilon zeta eta'.split()[i]} {i * 9}")
             for i, b in enumerate(chunking.BEATS)]
    return {"narrator": "Eli", "beats": beats}


def _build(first, second):
    from ragscript.pipeline import build_outline
    answers, notes = iter([first, second]), []
    narrator, beats = build_outline(lambda temps, **kw: next(answers), config.load(), 700, notes)
    return beats, notes


def test_build_outline_keeps_a_repair_with_fewer_holes():
    beats, notes = _build(_outline_with(2), _outline_with(0))
    assert all(b.place for b in beats) and notes == []
    assert sum(b.words for b in beats) == 700 and all(b.orig == b.words for b in beats)


def test_build_outline_drops_a_repair_that_is_no_better():
    beats, notes = _build(_outline_with(1), _outline_with(3))
    assert sum(not b.place for b in beats) == 1 and "Outline logic problems left" in notes[0]


def test_beat_from_json_normalises_the_id_and_rejects_bad_input():
    import pytest
    b = chunking.Beat.from_json({"beat": "Rising Tension", "words": "120", "summary": "s", "place": None})
    assert (b.beat, b.words, b.place) == ("rising_tension", 120, "")
    for bad in ({"beat": "middle", "words": 10}, {"beat": "hook", "words": 0}, {"beat": "hook", "words": "many"}):
        with pytest.raises(ValueError):
            chunking.Beat.from_json(bad)


def test_save_never_overwrites_an_earlier_script(tmp_path, monkeypatch):
    import time
    from ragscript import report
    monkeypatch.setattr(time, "strftime", lambda fmt: "0101-000001")
    a = report.save(tmp_path, "horror", 5, "A Brief!", "one", {"k": 1})
    b = report.save(tmp_path, "horror", 5, "A Brief!", "two", {"k": 2})
    assert a != b and a.read_text(encoding="utf-8") == "one" and b.read_text(encoding="utf-8") == "two"
    assert a.name.startswith("horror-5min-a-brief-") and a.with_suffix(".trace.json").exists()


def test_repeated_rate_outside_the_setting_ignores_the_burger_bun():
    from ragscript import checks
    text = ("The hum of the vents filled the room. I stared at the door. "
            "The hum of the vents filled the hall. I stared at the wall.")
    setting = checks.setting_words("hum of the vents")
    outside, everything = checks.repeated_rate_outside(text, 4, setting), checks.repeated_rate(text, 4)
    assert 0 < outside < everything
    assert checks.repeated_rate_outside(text, 4, set()) == everything


def test_tidy_drops_a_pause_that_is_followed_by_another_on_the_next_line():
    from ragscript import checks
    assert checks.tidy("X.\n[PAUSE]\n[PAUSE]\nY.") == "X.\n[PAUSE]\nY."


def test_the_critic_reply_is_read_whatever_its_list_format():
    from types import SimpleNamespace
    from ragscript.refine import Refiner
    reply = """2: the narrator is in two places
- 3. an object is never used again
Section 4: a fact contradicts an earlier one
5) the same scene is told twice
NONE of this matters
99: not a real section"""
    ref = Refiner(lambda stage, vars_, **kw: reply, config.load(), [])
    found = ref.critique(["a", "b", "c", "d", "e"], SimpleNamespace(narrator="n", outline="o"))
    assert sorted(found) == [1, 2, 3, 4] and found[2][0].message == "an object is never used again"
