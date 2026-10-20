"""Model-free self-checks: python -m pytest tests"""
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


def test_render_has_annotated_script_and_clean_copy():
    from ragscript import render
    stats = dict(words=10, target_words=10, word_accuracy=1.0, repeated_phrase_rate=0.0, avg_sentence_words=5.0,
                 overlap_flags=[], banned_hits=[], invented=["x"], unsupported_cites=[], uncited_facts=[], notes=[])
    md = render.render("b", "history", 5, "n", {}, "Fact one [S1]. {{INVENTED: a smell}}.", [(0, "c")], {}, stats)
    annotated = md[md.index("## Script"): md.index("## Chapters")]
    clean = md[md.index("## Clean narration"): md.index("## Description")]
    assert "[S1]" in annotated and "{{INVENTED" in annotated  # fact-checking copy keeps every marker
    assert "[S1]" not in clean and "INVENTED" not in clean and "Fact one. a smell." in clean
    plain = render.render("b", "horror", 5, "n", {}, "Nothing to strip.", [(0, "c")], {}, stats)
    assert "can be recorded as is" in plain[plain.index("## Clean narration"): plain.index("## Description")]


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
