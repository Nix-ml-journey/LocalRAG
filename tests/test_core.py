"""Model-free self-checks: python -m pytest tests"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscript import checks, chunking, config

CFG = config.load()


def test_beat_headers_split():
    text = "## Hook\nA.\n## Setup\nB.\n## Climax\nC.\n## Outro\nD."
    assert [b for b, _ in chunking.chunk(text, CFG)] == ["hook", "setup", "climax", "outro"]


def test_sliding_chunks_have_overlap_and_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    parts = chunking.chunk(text, CFG)
    assert len(parts) > 3 and all(chunking.tokens(t) <= 560 for _, t in parts)


def test_genre_pack_any_name_and_grounding():
    from ragscript import genre
    g = genre.load(CFG, "History", "dry humour")  # case-insensitive; known pack with grounded flag
    assert g.slug == "history" and g.grounded and "GROUNDED" in g.rules and "dry humour" in g.rules
    s = genre.load(CFG, "Sci-Fi ../x")  # unknown genre: default rules, name sanitised
    assert s.slug == "sci-fi-x" and not s.grounded and "GENERIC" in s.rules


def test_unsupported_cites():
    lore = {1: "The Dei Gratia found the Mary Celeste on 4 December 1872."}
    ok = "The ship was found on 4 December 1872. [S1]"
    bad = "The pumps failed on 9 December. [S1]"
    assert not checks.unsupported_cites(ok, lore)
    assert checks.unsupported_cites(bad, lore)[0][1] == ["9"]


def test_grounding_and_repeat_helpers():
    assert checks.reused_quotes('He wrote "We left in haste" again.', 'a note: "we left in haste"')
    assert not checks.reused_quotes('A "fresh quote here" appears.', "nothing quoted")
    flagged = checks.uncited_facts("I walked in. The Mary Celeste sailed in 1872. [S1] It sailed on 4 May 1872. "
                                   "I met Captain Briggs {{INVENTED: a rumour}}.")
    assert flagged == ["It sailed on 4 May 1872."]
    p = "I slam the door and my watch stops at two thirty four"
    assert checks.drop_repeats("Something fresh happens in the cellar below us.\n\n" + p, earlier=p) == \
        "Something fresh happens in the cellar below us."


def test_cited_sentence_with_a_name_missing_from_the_lore_is_flagged():
    lore = {1: "The Dei Gratia found the Mary Celeste near the Azores on 4 December 1872."}
    ok = "Then the Dei Gratia found the Mary Celeste. [S1]"
    invented_name = "Then Captain Morehouse boarded the Mary Celeste. [S1]"
    possessive = "The Mary Celeste’s trail ended near the Azores. [S1]"
    wrapped = "The crew {{INVENTED: of Captain Morehouse}} reached the Azores. [S1]"
    assert not checks.unsupported_cites(ok, lore)
    assert checks.unsupported_cites(invented_name, lore)[0][1] == ["Captain", "Morehouse"]
    assert not checks.unsupported_cites(possessive, lore)  # "Celeste’s" is matched as "Celeste"
    assert not checks.unsupported_cites(wrapped, lore)  # names inside an INVENTED mark are already labelled
    assert not checks.unsupported_cites("Some say the Azores held it, but I’ll never know. [S1]", lore)  # I’ll is no name


def test_mark_invented_wraps_only_rejected_sentences():
    script = "Briggs sailed in 1872. [S1] I found a bottle labelled Mary Celeste. [PAUSE]"
    flagged = checks.uncited_facts(script) + ["Briggs sailed in 1872. [S1]"]
    out, n = checks.mark_invented(script, flagged, lambda s: "Briggs" in s)
    assert n == 1 and "{{INVENTED: I found a bottle labelled Mary Celeste.}}" in out
    assert "Briggs sailed in 1872. [S1]" in out


def test_drop_repeat_sentences():
    old = "The note said we left the wine and nothing else was found aboard."
    new = "A fresh thing happens now in the dark hold. The note said we left the wine and nothing else.\n\n[PAUSE]"
    out = checks.drop_repeat_sentences(new, earlier=old)
    assert "fresh thing" in out and "left the wine" not in out and out.endswith("[PAUSE]")


def test_avg_sentence_words_ignores_cues_and_counts_fragments():
    assert round(checks.avg_sentence_words("I ran. It was dark. [PAUSE] Then nothing."), 2) == 2.33  # 2 + 3 + 2 words, 3 sentences
    long = "The old keeper counted every single iron step as he climbed toward the lamp room at midnight."  # 17 words
    assert checks.avg_sentence_words(long + " {{INVENTED: He hummed a tune.}}") == (17 + 4) / 2
    assert checks.avg_sentence_words("[PAUSE]") == 0.0


def test_prompts_ask_for_natural_sentence_rhythm():
    """The choppy-fragment problem came from nine prompts saying 'short sentences'; keep the rhythm target in place."""
    root = Path(__file__).resolve().parent.parent / "prompts"
    for f in ("stages/section.md", "stages/polish.md", "stages/fit.md"):
        assert "10-15 words" in (root / f).read_text(encoding="utf-8"), f
    for f in root.glob("*.md"):
        assert "short sentences" not in f.read_text(encoding="utf-8").lower(), f"{f.name} still pushes fragments"


def test_clean_narration_strips_tags_and_markers_but_keeps_cues():
    raw = ("The ship sailed in 1872 [S1]. {{INVENTED: I smelled old paper}}. [PAUSE] Then silence.\n"
           "[SFX: creak] The {{INVENTED: cold}} deck [S2] was empty.")
    out = checks.clean_narration(raw)
    assert out == ("The ship sailed in 1872. I smelled old paper. [PAUSE] Then silence.\n"
                   "[SFX: creak] The cold deck was empty.")
    assert checks.clean_narration(out) == out  # idempotent


def test_render_script_is_clean_and_annotated_copy_is_last():
    from ragscript import render
    stats = dict(words=10, target_words=10, word_accuracy=1.0, repeated_phrase_rate=0.0, avg_sentence_words=5.0,
                 overlap_flags=[], banned_hits=[], invented=["x"], unsupported_cites=[], uncited_facts=[], notes=[])
    md = render.render("b", "history", 5, "n", {}, "Fact one [S1]. {{INVENTED: a smell}}.", [(0, "c")], {}, stats)
    main = md[md.index("## Script"): md.index("## Chapters")]
    annotated = md[md.index("## Fact-check copy"):]
    assert "[S1]" not in main and "INVENTED" not in main and "Fact one. a smell." in main
    assert "Fact one [S1]. {{INVENTED: a smell}}." in annotated  # every marker kept for fact-checking
    plain = render.render("b", "horror", 5, "n", {}, "Nothing to strip.", [(0, "c")], {}, stats)
    assert "Fact-check copy" not in plain  # nothing to annotate


def test_overlap_flag_and_repeat_rate():
    src = "the old lighthouse keeper climbed the stairs every night without fail"
    assert checks.overlap_flags(src, [src], 5, 0.8)
    assert not checks.overlap_flags("a completely different sentence about distant mountains today", [src], 5, 0.8)
    p = "I slam the door and my watch stops at two thirty four"
    assert checks.drop_repeats(f"{p}.\n\nSomething new happens in the cellar below.\n\n{p} again.").count("slam") == 1
    assert checks.repeated_rate("a b c d e a b c d e", 4) > 0.5
    assert checks.spoken_count("Hello [PAUSE] world {{INVENTED: red door}}") == 4


def test_curly_apostrophe_is_one_word():
    # models emit both ' and ’; a split would inflate word counts and hide n-gram overlap with straight-quote sources
    assert checks.spoken_count("I don’t know") == checks.spoken_count("I don't know") == 3
    assert checks.words("don’t") == ["don't"]


def test_sentence_split_ignores_initials_and_titles():
    t = 'In 1884 Doyle published "J. Habakuk Jephson". Dr. Smith agreed. It ended.'
    assert re.split(checks.SENT, t) == ['In 1884 Doyle published "J. Habakuk Jephson".', "Dr. Smith agreed.", "It ended."]


def test_mark_invented_never_nests():
    script = "{{INVENTED: I lean in. Records say so.}} Plain claim."
    out, n = checks.mark_invented(script, ["{{INVENTED: I lean in.", "Records say so.}}", "Plain claim."], lambda s: False)
    assert n == 1 and out.count("{{") == 2 and "{{INVENTED: Plain claim.}}" in out


def test_avg_sentence_words_uses_the_shared_splitter():
    assert checks.avg_sentence_words('He read J. Habakuk Jephson aloud. It ended.') == (6 + 2) / 2


def test_verify_marks_only_what_the_model_rejects():
    from ragscript import verify
    lore = {1: "The Mary Celeste sailed in 1872 with 1701 barrels."}
    script = "She sailed in 1872 [S1]. The log ended mid-sentence [S1]."
    out, n = verify.verify(script, lore, lambda l, s: "NO" if "mid-sentence" in s else "YES")
    assert n == 1 and "{{INVENTED: The log ended mid-sentence.}}" in out and "{{INVENTED: She" not in out


def test_source_ledger_numbers_and_reports_sources():
    from ragscript.sources import SourceLedger
    led = SourceLedger()
    c = {"id": "lore:lore_a.md:0", "text": "fact", "meta": {"source": "lore_a.md"}}
    assert led.cite([c, c]).count("[S1]") == 2 and led.cited_text == {1: "fact"}
    led.add_style([{"id": "scripts:x.txt:3", "text": "style", "meta": {"source": "x.txt"}}])
    assert led.used() == {"lore_a.md": [1], "x.txt": []} and led.refs == ["fact", "fact", "style"]


def test_splitter_keeps_mid_sentence_tags_and_abbreviations_whole():
    t = "The ship left in 1861 [S1] before she was sold. He met Dr. Smith, i.e. the clerk. Done."
    assert checks.sentences(t) == ["The ship left in 1861 [S1] before she was sold.", "He met Dr. Smith, i.e. the clerk.", "Done."]


def test_multi_sentence_invented_mark_is_not_reported_as_uncited():
    assert checks.uncited_facts("{{INVENTED: I lean in. Captain Briggs smiled.}} Captain Rose spoke.") == ["Captain Rose spoke."]


def test_words_keep_accented_names_whole():
    assert checks.words("Félix Bruzzone") == ["félix", "bruzzone"]


def test_unlabelled_double_braces_become_invented_marks_and_never_reach_clean_narration():
    from ragscript.pipeline import Pipeline
    fixed = Pipeline._clean("Plain. {{I sat in my office.}} {{invented: kept}} Done.")
    assert fixed == "Plain. {{INVENTED: I sat in my office.}} {{invented: kept}} Done."
    assert checks.clean_narration("A {{stray}} word {{INVENTED: x}}.") == "A stray word x."
