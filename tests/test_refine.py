"""Offline tests of the passes after drafting: issues, rewrites, the critic, the ending choice."""
from support import FakeLLM, compare_reply
from support import make_pipeline as pipeline
from support import write_script as write
from test_length_guard import LengthLLM

from ragscript import checks, chunking, config
from ragscript.pipeline import Story


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
    refiner = Refiner(lambda stage, vars, **kw: bad, cfg, [])
    assert refiner.revise_weakest(parts, Story("b", g, "Eli", 750, 5, notes=notes)) == parts and "rejected" in notes[0]


def test_draft_and_final_issues_judge_the_two_stages_with_named_entry_points():
    from ragscript.refine import draft_issues, final_issues
    cfg = config.load()
    twice = "I walked in. [PAUSE] I see the door. [PAUSE] I knew it. [PAUSE] I find the key. I hear a step. I looked up."
    assert not draft_issues(twice, "", cfg, [], budget=22)  # a draft is not judged on cues
    kinds = [i.kind for i in final_issues(twice, "", cfg, [])]
    assert "pause" in kinds and "tense" in kinds
    assert draft_issues("word " * 400, "", cfg, [], budget=100)[0].kind == "length"
    assert [i.kind for i in final_issues("A fine line.", "", cfg, ["fine line"])] == ["banned"]


def test_polish_and_fit_outputs_lose_invented_bracket_notes():
    from ragscript import genre
    from ragscript.refine import Refiner
    cfg = config.load()
    g = genre.load(cfg, "horror")
    sections = [(chunking.Beat("hook", 20), "I opened the door and listened for a long quiet moment in the hall.")]
    noted = "I opened the door [recurring] and listened for a long quiet moment in the hall."
    out = Refiner(lambda stage, vars, **kw: noted, cfg, []).polish(sections, Story("b", g, "Eli", 750, 5))
    assert "[recurring]" not in out[0]


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


def test_attempt_keeps_an_accepted_rewrite_cleans_it_and_reverts_a_rejected_one():
    from ragscript.refine import attempt
    new, kept = attempt("old text.", lambda: "A fine rewrite [recurring] of it.", lambda n: "fine" in n)
    assert kept and new == "A fine rewrite of it."  # the invented note is stripped before the text is judged
    old, kept = attempt("old text.", lambda: "A poor rewrite.", lambda n: "fine" in n)
    assert not kept and old == "old text."


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


def test_a_logic_rewrite_is_dropped_only_when_a_second_critic_call_finds_more_holes(tmp_path):
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

    assert "logic holes: 1 before, 0 after" in run("NONE")
    assert "logic holes: 1 before, 1 after" in run("2: still two places")  # no new hole: the rewrite stays
    assert "found more logic holes" in run("2: one hole\n2: another hole")


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
    found = ref.critique(["a", "b", "c", "d", "e"], SimpleNamespace(narrator="n", outline="o", details=[]))
    assert sorted(found) == [1, 2, 3, 4] and found[2][0].message == "an object is never used again"


def test_compare_asks_twice_with_the_scripts_swapped_and_counts_position_bias(tmp_path):
    seen = []

    class L(FakeLLM):
        def chat(self, prompt, **kw):
            if kw.get("stage") != "compare":
                return super().chat(prompt, **kw)
            seen.append((prompt.index("SCRIPT-ONE"), prompt.index("SCRIPT-TWO")))
            return compare_reply("A")

    result, disagreed = pipeline(tmp_path, L()).refiner.compare("SCRIPT-ONE", "SCRIPT-TWO", "brief", "horror")
    assert seen[0][0] < seen[0][1] and seen[1][0] > seen[1][1]  # the second question has the scripts exchanged
    assert set(result.values()) == {"tie"} and disagreed == 7  # always answering "A" is position bias, so all ties


def test_two_rewrite_rounds_keep_the_critic_issues_for_the_second_round(tmp_path):
    """A logic rewrite in round one once replaced the issue dict with a bool, so round two crashed."""
    from tests.support import FakeLLM

    class L(FakeLLM):
        def chat(self, prompt, **kw):
            if kw.get("stage") == "critic":
                return "2: the narrator is in two places at once\n3: an object appears from nowhere"
            return super().chat(prompt, **kw)

    pl = pipeline(tmp_path, L())
    pl.cfg["checks"]["revise_sections"] = 3
    assert "Revi" in write(pl)["markdown"]  # more than one round ran without an AttributeError


def test_a_detail_past_its_limit_becomes_a_logic_issue_for_the_critic_to_fix(tmp_path):
    from types import SimpleNamespace
    from ragscript.refine import Refiner
    cfg = pipeline(tmp_path).cfg
    cfg["checks"]["motif_max"] = 3
    ref = Refiner(lambda *a, **k: "NONE", cfg, [])
    parts = ["The hum filled the booth.", "The hum stalled.", "The hum rose.", "The hum came back once more."]
    found = ref.critique(parts, SimpleNamespace(narrator="n", outline="o", details=["hum"]))
    assert list(found) == [3] and 'recurring detail "hum"' in found[3][0].message


def test_best_ending_keeps_a_challenger_only_when_it_wins_in_both_orders(tmp_path):
    from types import SimpleNamespace
    from ragscript.refine import Refiner

    def run(compare_reply, parts):
        asked = []

        def ask(stage, tpl_vars, **kw):
            asked.append(stage)
            return "I closed the door and the brass key swung once." if stage == "ending" else compare_reply(tpl_vars)

        pl = pipeline(tmp_path)
        pl.cfg["checks"]["ending_candidates"] = 2
        story = SimpleNamespace(narrator="n", outline="o", payoff="the brass key", brief="b", notes=[],
                                genre=SimpleNamespace(slug="horror"))
        return Refiner(ask, pl.cfg, []).best_ending(parts, story), asked, story.notes

    parts = ["I stood in the booth."] * 3 + ["I heard the lock and the dark closed over the booth and nothing moved."]
    # the challenger wins both orders: it is "B" when it comes second and "A" when it comes first
    prefer_second = lambda v: compare_reply("B" if "closed the door" not in v["script_a"] else "A")
    out, asked, notes = run(prefer_second, parts)
    assert out[-1].startswith("I closed the door") and asked.count("compare") == 2 and any("Ending:" in n for n in notes)
    # always "A" is position bias, so the orders disagree: a tie keeps the current ending
    out, asked, notes = run(lambda v: compare_reply("A"), parts)
    assert out == parts and not notes


def test_every_over_used_detail_gets_a_rewrite_even_past_the_configured_number():
    from ragscript import genre
    from ragscript.refine import Issue, Refiner
    cfg = config.load()
    cfg["checks"]["revise_sections"] = 1
    g = genre.load(cfg, "horror")
    parts = ["Start here.", "I took the key.", "I held the key.", "I lost the key."]
    calls = []
    hole = {i: [Issue("logic", 'the recurring detail "key" already came back 3 times')] for i in (1, 2, 3)}
    Refiner(lambda s, v, **kw: calls.append(s) or "Fresh words now.", cfg, []).revise_weakest(
        parts, Story("b", g, "Eli", 750, 5, notes=[]), hole)
    assert calls.count("revise") == 3


def test_fix_overuse_rewrites_a_section_that_still_holds_the_detail_but_not_the_last_one():
    from ragscript import genre
    from ragscript.refine import Refiner
    cfg = config.load()
    g = genre.load(cfg, "horror")
    old = "I held the brass key tight while the rain hammered the roof above me and the lamps flickered low."
    cfg["checks"]["motif_max"] = 3
    parts = [old, old, old, old]  # the key is in 4 paragraphs, the limit is 3
    new = "I gripped the cold handle tight while the rain hammered the roof above me and the lamps flickered low."
    asked = []

    def ask(stage, vars_, **kw):
        asked.append(vars_["text"])
        return new

    story = Story("b", g, "Eli", 750, 5, notes=[], details=["brass key"])
    out = Refiner(ask, cfg, []).fix_overuse(parts, story)
    assert out == [old, old, new, old] and len(asked) == 1  # the 3rd section, not the ending; then no excess is left
    assert "over-used detail" in story.notes[0]
