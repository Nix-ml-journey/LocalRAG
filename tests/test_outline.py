"""Offline tests of the outline: bad model JSON, repairs, beat fields and the planned ending."""
import json

from support import BEATS, FakeLLM
from support import make_pipeline as pipeline
from support import write_script as write

from ragscript import chunking, config


def test_outline_with_loose_beat_names_and_bad_words_is_fixed_or_retried(tmp_path):
    class L(FakeLLM):
        outlines, section_prompts = 0, []

        def chat(self, prompt, **kw):
            if kw.get("stage") == "outline":
                self.outlines += 1
                names = [b.title().replace("_", " ") for b in BEATS]  # "Hook", "Rising Tension"...
                words = "about 100" if self.outlines == 1 else 107
                beats = [{"beat": n, "title": n, "words": words, "summary": "s", "place": f"room {i}", "detail": "",
                          "change": (f"new event {i * 11} {'ab cd ef gh ij kl mn'.split()[i]} xyz "
                                     f"{'lantern' * (i in (0, 6))}"),
                          "plant": "brass key" if i == 1 else "", "payoff": "brass key" if i == 6 else ""}
                     for i, n in enumerate(names)]
                return json.dumps({"narrator": "Eli", "beats": beats})
            if kw.get("stage") == "section":
                self.section_prompts.append(prompt)
            return super().chat(prompt, **kw)

    llm = L()
    write(pipeline(tmp_path, llm))
    assert llm.outlines == 2, "words that are not a number must make the outline attempt fail"
    assert "Open with this chosen hook" in llm.section_prompts[0], "'Hook' must be read as the hook beat"


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


def _outline_with(holes):
    beats = [chunking.Beat(b, 100, place="" if i < holes else f"room {i}",
                           change=(f"event {i} {'alpha beta gamma delta epsilon zeta eta'.split()[i]} {i * 9} "
                                   f"{'lantern' * (i in (0, 6))}"),
                           plant="lantern" if i == 1 else "", payoff="the lantern" if i == 6 else "")
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


def test_outline_field_values_become_plain_words_so_no_snake_case_reaches_the_narration():
    b = chunking.Beat.from_json({"beat": "setup", "words": 100, "detail": "haptic_hum: steady", "sees": "scanner_screen",
                                 "plant": "gray_synth_wrap"})
    assert (b.detail, b.sees, b.plant) == ("haptic hum: steady", "scanner screen", "gray synth wrap")
    assert b.beat == "setup"


def test_build_outline_cleans_the_narrator_and_ignores_a_plant_copied_into_the_outro():
    first = _outline_with(0)
    first["narrator"] = "Eli, a courier; details: glove_grip, hum"
    first["beats"][-1].plant = "the lantern"  # the model copied the payoff into the outro's plant
    beats, notes = _build(first, first)[0], _build(first, first)[1]
    assert beats[-1].plant == "" and notes == []
    narrator, _ = __import__("ragscript.pipeline", fromlist=["build_outline"]).build_outline(
        lambda temps, **kw: _outline_with(0) | {"narrator": "Eli; glove_grip"}, config.load(), 700, [])
    assert narrator == "Eli; glove grip"


def test_a_detail_labelled_detail_n_takes_the_name_of_the_narrators_nth_detail():
    from ragscript.pipeline import name_details
    beats = [chunking.Beat("hook", 100, detail="Detail 2: The hiss grows sharper"),
             chunking.Beat("setup", 100, detail="Detail 9: x"),
             chunking.Beat("climax", 100, detail="salt crust: stings")]
    bible = "Elias; Detail 1: The salt-crusted grip of his staff; Detail 2: The rhythmic hiss of the tide; Detail 3: itch"
    name_details(bible, beats)
    assert [b.detail for b in beats] == ["rhythmic hiss of the tide: The hiss grows sharper", "Detail 9: x",
                                         "salt crust: stings"]
