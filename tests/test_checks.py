"""Offline tests of the checks on the narration and on the story's logic (checks.py, logic.py)."""
from ragscript import checks, chunking, config

CFG = config.load()
CK = CFG["checks"]


def test_cue_tense_and_crutch_word_checks():
    assert checks.mixed_tense("I walked in. I saw it. I knew. I see the door. I hear it. I feel cold.", CK)
    assert not checks.mixed_tense("I walked in. I saw it. I knew. I found the key. She runs.", CK)
    assert not checks.mixed_tense("She walked in and saw him.", CK)  # third person never fires
    text = " ".join(["The heavy door shut."] * 6 + ["Then silence fell over everything we owned here."] * 2)
    assert checks.overused_words(text, CK)[0][0] == "heavy"


def test_dialogue_in_another_tense_is_not_counted_as_a_tense_switch():
    narration = "I walked in. I saw the door. I knew it was locked. "
    speech = 'He said, "I see it, I hear it, I feel it." '
    assert not checks.mixed_tense(narration + speech, CK)
    assert checks.mixed_tense(narration + "I see it. I hear it. I feel it.", CK)


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
    from ragscript import logic
    beats = [chunking.Beat(beat="hook", place="booth", change="the monitor shows him early", detail="hum: steady"),
             chunking.Beat(beat="setup", place="", change="the monitor shows him early again", detail="hum: louder"),
             chunking.Beat(beat="reveal", place="hall", change="a door opens by itself", detail="hum: stalls"),
             chunking.Beat(beat="climax", place="closet", change="the handle turns slowly now", detail="hum: dies")]
    found = " | ".join(logic.outline_issues(beats, 3))
    assert 'no "place"' in found and "almost the same thing" in found and "again right after" in found
    assert 'the detail "hum" is used in 4 beats' in found
    assert logic.outline_issues(beats[2:3], 3) == []


def test_outline_last_beat_must_use_something_shown_earlier():
    from ragscript import logic
    def beat(name, summary):
        return chunking.Beat(beat=name, place="booth", change=summary, summary=summary)
    story = [beat("hook", "a navy jacket on camera four"), beat("setup", "the guard checks logs"),
             beat("rising_tension", "footsteps near the door"), beat("reveal", "the face turns"),
             beat("climax", "the handle rattles"), beat("aftermath", "he hides inside the closet")]
    bad = logic.outline_issues(story + [beat("outro", "silence settles everywhere")], 3)
    # a word that most beats share (the narrator's name, the setting) is no setup for the ending
    common = [beat(n, f"Elias {s}") for n, s in (("hook", "sees a jacket"), ("setup", "checks logs"),
                                                  ("rising_tension", "hears steps"), ("reveal", "sees the face"))]
    assert logic.outline_issues(common + [beat("climax", "Elias runs"), beat("aftermath", "Elias hides"),
                                           beat("outro", "Elias waits")], 3)
    good = logic.outline_issues(story + [beat("outro", "the navy jacket hangs empty")], 3)
    boots = [beat("hook", "a worn boot by the door")] + story[1:] + [beat("outro", "the boots stand empty")]
    plural = logic.outline_issues(boots, 3)
    assert any("last beat" in x for x in bad) and not any("last beat" in x for x in good + plural)


def test_dropped_then_used_finds_an_object_used_after_it_was_dropped():
    from ragscript import logic
    parts = ["My keys hit the tile while I abandoned my radio to flee.", "The scratching of my keys echoed on the lock.",
             "I picked up the radio and held it close."]
    assert logic.dropped_then_used(parts) == [("keys", 1)]
    assert logic.dropped_then_used(["I dropped my keys.", "I grabbed the keys and gripped them."]) == []
    # an idiom, an abstract loss and an object another person takes are no holes
    for text in ("I threw my hands up in fear.", "I lost the thread of the story."):
        assert logic.dropped_then_used([text, "I looked up and held the thread tight."]) == []
    assert logic.dropped_then_used(["I dropped the keys.", "He took the keys and held them."]) == []
    for text in ("I dropped my keys again.", "I dropped my keys then ran."):  # a small word after the object
        assert logic.dropped_then_used([text, "I gripped the keys."]) == [("keys", 1)]


def test_detail_returns_counts_the_paragraphs_that_bring_a_detail_back():
    from ragscript import logic
    text = "The HVAC hum filled the room.\n\nI walked on.\n\nThe hum stalled.\n\nNothing else."
    assert logic.detail_returns(text, ["hum"]) == [("hum", 2)]
    text = "The heavy door shut.\n\nThe heavy rain fell.\n\nI saw the turnstile."
    # counted by "turnstile", not "heavy"
    assert logic.detail_returns(text, ["heavy rattle of the turnstile"]) == [("heavy rattle of the turnstile", 1)]
    paras = [f"The floor creaked in room {i}." for i in range(6)] + ["The wax smell came back.", "A wax sting stayed."]
    assert logic.detail_returns("\n\n".join(paras), ["floor wax"]) == [("floor wax", 2)]


def test_outline_issue_for_a_screen_beat_without_sees():
    from ragscript import logic
    beat = chunking.Beat(beat="reveal", place="booth", change="the double turns its head", summary="He watches the monitor.")
    assert any('"sees"' in x for x in logic.outline_issues([beat], 3))
    seen = chunking.Beat(**{**beat.__dict__, "sees": "the hallway feed on the booth monitor"})
    assert not any('"sees"' in x for x in logic.outline_issues([seen], 3))


def test_fill_sees_copies_the_nearest_screen():
    from ragscript import logic
    beats = [chunking.Beat(beat="hook", summary="He watches the monitor.", sees="Monitor 4"),
             chunking.Beat(beat="setup", summary="He paces the hall.", sees=""),
             chunking.Beat(beat="reveal", summary="The screen flickers.", sees="")]
    assert logic.fill_sees(beats) == 1 and beats[2].sees == "Monitor 4" and beats[1].sees == ""


def test_tidy_cleans_layout_but_keeps_the_words():
    from ragscript import checks
    raw = " First line.  \n\n  It moved. [PAUSE]\n\nThen nothing.\n\n[PAUSE]\n\n[PAUSE] The end."
    out = checks.tidy(raw)
    assert out == "First line.\n\nIt moved.\n\n[PAUSE]\n\nThen nothing.\n\n[PAUSE] The end."
    assert checks.spoken_count(out) == checks.spoken_count(raw)


def test_unused_objects_lists_an_object_found_and_never_used():
    from ragscript import logic
    text = ("I noticed a set of brass keys on the floor. I walked on. The door opened. "
            "I held a flashlight and the flashlight shook.")
    assert logic.unused_objects(text) == ["keys"]


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


def test_new_in_ending_names_only_things_the_rest_of_the_script_never_mentions():
    from ragscript import logic
    text = ("I hid in the booth with the salt and the logbook.\n\nThe lamp went dark.\n\n"
            "I huddled and took a pinch of salt from the stamp, slowly.")
    assert logic.new_in_ending(text) == ["pinch", "stamp"]  # "salt" was told before; "huddled", "slowly" are not things
    assert logic.new_in_ending("One paragraph only, with a pinch.") == []
    cued = text.replace("I huddled", "[PAUSE] I huddled")  # a last line that opens with a cue still counts as the ending
    assert logic.new_in_ending(cued) == ["pinch", "stamp"]
    assert logic.new_in_ending(text + "\n\n[SFX: wind]") == ["pinch", "stamp"]  # a cue-only paragraph is not a paragraph


def test_outline_must_plant_the_final_image_early_and_the_outro_must_show_it():
    from ragscript import logic

    def story(plant="a brass key on the hook", payoff="the brass key", plant_at=1):
        beats = [chunking.Beat(b, 100, place="booth",
                               change=f"event {i} {'alpha beta gamma delta epsilon zeta eta'.split()[i]}")
                 for i, b in enumerate(chunking.BEATS)]
        beats[plant_at].plant, beats[-1].payoff = plant, payoff
        return logic.outline_issues(beats, 3)

    new = ('no "payoff"', "planted in no beat", "different things", "first four beats")
    assert not any(m in x for x in story() for m in new)  # a planted, matching final image raises none of the new issues
    assert any('no "payoff"' in x for x in story(payoff=""))
    assert any("planted in no beat" in x for x in story(plant=""))
    assert any("different things" in x for x in story(payoff="the red door"))
    assert any("must be in one of the first four beats" in x for x in story(plant_at=5))


def test_detail_overuse_names_the_parts_past_the_limit():
    from ragscript import logic
    parts = ["The hum filled the booth.", "The hum stalled.", "The hum rose again.", "Silence, then a hum."]
    assert logic.detail_overuse(parts, ["hum", ""], 3) == [("hum", 3)]  # the 4th paragraph is past the limit of 3
    two_in_one = ["The hum filled the booth.", "The hum stalled.\n\nThe hum dropped.", "A hum rose.", "Then a hum."]
    assert logic.detail_overuse(two_in_one, ["hum"], 3) == [("hum", 2), ("hum", 3)]  # paragraphs, reported as parts
    assert logic.detail_overuse(parts, ["hum"], 4) == []
    assert logic.detail_overuse(parts, ["gears"], 3) == []


def test_the_last_beat_may_show_the_planted_thing_through_its_payoff_alone():
    from ragscript import logic
    beats = [chunking.Beat(n, 100, place="booth", change=f"event {i} {'alpha beta gamma delta epsilon zeta eta'.split()[i]}",
                           summary=f"scene {i} of the guard")
             for i, n in enumerate(chunking.BEATS)]
    beats[1].plant, beats[-1].payoff = "a brass key", "the brass key"  # nothing else in the outro mentions the key
    assert not any("last beat" in x for x in logic.outline_issues(beats, 3))


def test_outline_flags_a_detail_named_detail_n_and_a_change_that_is_only_a_mood():
    from ragscript import logic
    beats = [chunking.Beat(b, 100, place="booth", detail="Detail 2: sharper" if i == 1 else "",
                           change=("The clerk's growing desperation creates urgency." if i == 2
                                   else f"event {i} {'alpha beta gamma delta epsilon zeta eta'.split()[i]}"))
             for i, b in enumerate(chunking.BEATS)]
    found = " | ".join(logic.outline_issues(beats, 3))
    assert 'calls its detail "detail 2"' in found and "not an event someone could film" in found
    assert "beat 3" in found and found.count("not an event") == 1


def test_outline_flags_a_change_that_only_says_something_stays_the_same():
    from ragscript import logic
    beats = [chunking.Beat(b, 100, place="deck", change="The stranger remains seated and still dripping." if i == 3
                           else f"event {i} {'alpha beta gamma delta epsilon zeta eta'.split()[i]}")
             for i, b in enumerate(chunking.BEATS)]
    found = " | ".join(logic.outline_issues(beats, 3))
    assert "beat 4" in found and found.count("not an event") == 1


def test_stem_keeps_words_that_end_in_us_or_ss():
    from ragscript.checks import stem
    assert stem("nervous") == "nervous" and stem("glass") == "glass" and stem("tickets") == "ticket"


def test_snake_case_leaks_become_words_and_a_third_person_ending_is_flagged():
    from ragscript import logic
    assert checks.tidy("dry as the salt_pans on the ridge.") == "dry as the salt pans on the ridge."
    mine = "I walked down the long hall and I listened to the quiet room around me for a very long time."
    text = "\n\n".join([mine] * 4 + ["The next morning a courier found the desk empty and the old letters lying in gray dust."])
    assert logic.left_first_person(text) and not logic.left_first_person(text.replace("The next morning a", "I saw a"))


def test_trim_tics_keeps_two_uses_cuts_later_ones_before_a_noun_and_repeated_cues():
    ck = {"trim_words": ["heavy"], "trim_keep": 2}
    text = ("Heavy boots hit the floor. A heavy door shut. The heavy key fell. Heavy rain came. The air felt heavy. "
            "[SFX: rustle] a [SFX: rustle] b [SFX: rustle] c")
    out = checks.trim_tics(text, ck)
    assert out.startswith("Heavy boots hit the floor. A heavy door shut. The key fell. Rain came.")
    assert "felt heavy." in out, "after a verb the word is not before a noun and stays"
    assert out.count("[SFX: rustle]") == 2


def test_trim_tics_never_cuts_a_word_that_ends_its_clause_before_a_preposition():
    ck = {"trim_words": ["heavy"], "trim_keep": 0}
    assert checks.trim_tics("The cloth felt heavy in their grip and heavy on my arm.", ck) == \
        "The cloth felt heavy in their grip and heavy on my arm."
