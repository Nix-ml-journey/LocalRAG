"""Model-free self-checks: python -m pytest tests"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscript import checks, chunking, config
from ragscript import grounding, text as txt

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


def test_unsupported_cites_flags_numbers_and_names_the_lore_lacks():
    lore = {1: "The Dei Gratia found the Mary Celeste on 4 December 1872."}
    ok = "The ship was found on 4 December 1872. [S1]"
    bad = "The pumps failed on 9 December. [S1]"
    assert not grounding.unsupported_cites(ok, lore)
    assert grounding.unsupported_cites(bad, lore)[0][1] == ["9"]
    lore = {1: "The Dei Gratia found the Mary Celeste near the Azores on 4 December 1872."}
    ok = "Then the Dei Gratia found the Mary Celeste. [S1]"
    invented_name = "Then Captain Morehouse boarded the Mary Celeste. [S1]"
    possessive = "The Mary Celeste’s trail ended near the Azores. [S1]"
    wrapped = "The crew {{INVENTED: of Captain Morehouse}} reached the Azores. [S1]"
    assert not grounding.unsupported_cites(ok, lore)
    assert grounding.unsupported_cites(invented_name, lore)[0][1] == ["Captain", "Morehouse"]
    assert not grounding.unsupported_cites(possessive, lore)  # "Celeste’s" is matched as "Celeste"
    assert not grounding.unsupported_cites(wrapped, lore)  # names inside an INVENTED mark are already labelled
    # I’ll is no name
    assert not grounding.unsupported_cites("Some say the Azores held it, but I’ll never know. [S1]", lore)


def test_grounding_and_repeat_helpers():
    assert checks.reused_quotes('He wrote "We left in haste" again.', 'a note: "we left in haste"')
    assert not checks.reused_quotes('A "fresh quote here" appears.', "nothing quoted")
    flagged = grounding.uncited_facts("I walked in. The Mary Celeste sailed in 1872. [S1] It sailed on 4 May 1872. "
                                   "I met Captain Briggs {{INVENTED: a rumour}}.")
    assert flagged == ["It sailed on 4 May 1872."]
    p = "I slam the door and my watch stops at two thirty four"
    assert checks.drop_repeats("Something fresh happens in the cellar below us.\n\n" + p, p, 4, 0.6) == \
        "Something fresh happens in the cellar below us."


def test_mark_invented_wraps_only_rejected_sentences():
    script = "Briggs sailed in 1872. [S1] I found a bottle labelled Mary Celeste. [PAUSE]"
    flagged = grounding.uncited_facts(script) + ["Briggs sailed in 1872. [S1]"]
    out, n = grounding.mark_invented(script, flagged, lambda s: "Briggs" in s)
    assert n == 1 and "{{INVENTED: I found a bottle labelled Mary Celeste.}}" in out
    assert "Briggs sailed in 1872. [S1]" in out


def test_drop_repeat_sentences():
    old = "The note said we left the wine and nothing else was found aboard."
    new = "A fresh thing happens now in the dark hold. The note said we left the wine and nothing else.\n\n[PAUSE]"
    out = checks.drop_repeat_sentences(new, old, 4, 0.7, 6)
    assert "fresh thing" in out and "left the wine" not in out and out.endswith("[PAUSE]")


def test_avg_sentence_words_ignores_cues_counts_fragments_and_uses_the_shared_splitter():
    # 2 + 3 + 2 words, 3 sentences
    assert round(checks.avg_sentence_words("I ran. It was dark. [PAUSE] Then nothing."), 2) == 2.33
    long = "The old keeper counted every single iron step as he climbed toward the lamp room at midnight."  # 17 words
    assert checks.avg_sentence_words(long + " {{INVENTED: He hummed a tune.}}") == (17 + 4) / 2
    assert checks.avg_sentence_words("[PAUSE]") == 0.0
    assert checks.avg_sentence_words('He read J. Habakuk Jephson aloud. It ended.') == (6 + 2) / 2


def test_prompts_ask_for_natural_sentence_rhythm():
    """The choppy-fragment problem came from nine prompts saying 'short sentences'; keep the rhythm target in place."""
    root = Path(__file__).resolve().parent.parent / "prompts"
    assert config.sentence_range(CFG) == "12-16"
    for f in ("stages/section.md", "stages/polish.md", "stages/fit.md"):
        # filled from config style.sentence_words
        assert "<<sentence_range>> words" in (root / f).read_text(encoding="utf-8"), f
    for f in root.glob("*.md"):
        assert "short sentences" not in f.read_text(encoding="utf-8").lower(), f"{f.name} still pushes fragments"


def test_clean_narration_strips_tags_and_markers_but_keeps_cues():
    raw = ("The ship sailed in 1872 [S1]. {{INVENTED: I smelled old paper}}. [PAUSE] Then silence.\n"
           "[SFX: creak] The {{INVENTED: cold}} deck [S2] was empty.")
    out = grounding.clean_narration(raw)
    assert out == ("The ship sailed in 1872. I smelled old paper. [PAUSE] Then silence.\n"
                   "[SFX: creak] The cold deck was empty.")
    assert grounding.clean_narration(out) == out  # idempotent


def test_render_script_is_clean_and_annotated_copy_is_last():
    from ragscript import report
    rep = report.Report(words=10, target_words=10, word_accuracy=1.0, repeated_phrase_rate=0.0, avg_sentence_words=5.0,
                        sentence_range="10-15", overlap_flags=[], banned_hits=[], invented=["x"], unsupported_cites=[],
                        uncited_facts=[], pause_cues=0, minutes=5, mixed_tense=False, overused_words=[],
                        repeated_wording=[], motif_limit=3)
    doc = report.Doc("b", "history", 5, "n", {}, [(0, "c")], {})
    md = report.render(doc, "Fact one [S1]. {{INVENTED: a smell}}.", rep)
    main = md[md.index("## Script"): md.index("## Chapters")]
    annotated = md[md.index("## Fact-check copy"):]
    assert "[S1]" not in main and "INVENTED" not in main and "Fact one. a smell." in main
    assert "Fact one [S1]. {{INVENTED: a smell}}." in annotated  # every marker kept for fact-checking
    plain = report.render(report.Doc("b", "horror", 5, "n", {}, [(0, "c")], {}), "Nothing to strip.", rep)
    assert "Fact-check copy" not in plain  # nothing to annotate
    assert "about 10-15" in plain  # the sentence target shown is the configured one


def test_chapters_scale_the_drafted_proportions_to_the_finished_length():
    from ragscript import report
    from ragscript.chunking import Beat
    beats = [(Beat("hook", title="A"), 100), (Beat("setup"), 100)]
    assert report.chapters(beats, 300, 150) == [(0, "A"), (60, "setup")]  # 300 words at 150 wpm: the second starts at 1:00


def test_overlap_flag_and_repeat_rate():
    src = "the old lighthouse keeper climbed the stairs every night without fail"
    assert checks.overlap_flags(src, [src], 5, 0.8)
    assert not checks.overlap_flags("a completely different sentence about distant mountains today", [src], 5, 0.8)
    p = "I slam the door and my watch stops at two thirty four"
    text = f"{p}.\n\nSomething new happens in the cellar below.\n\n{p} again."
    assert checks.drop_repeats(text, "", 4, 0.6).count("slam") == 1
    assert checks.repeated_rate("a b c d e a b c d e", 4) > 0.5
    assert checks.spoken_count("Hello [PAUSE] world {{INVENTED: red door}}") == 4


def test_words_keep_curly_apostrophes_and_accented_names_whole():
    # models emit both ' and ’; a split would inflate word counts and hide n-gram overlap with straight-quote sources
    assert checks.spoken_count("I don’t know") == checks.spoken_count("I don't know") == 3
    assert txt.words("don’t") == ["don't"]
    assert txt.words("Félix Bruzzone") == ["félix", "bruzzone"]


def test_splitter_ignores_initials_titles_and_abbreviations_and_keeps_mid_sentence_tags():
    t = 'In 1884 Doyle published "J. Habakuk Jephson". Dr. Smith agreed. It ended.'
    assert re.split(txt.SENT, t) == ['In 1884 Doyle published "J. Habakuk Jephson".', "Dr. Smith agreed.", "It ended."]
    t = "The ship left in 1861 [S1] before she was sold. He met Dr. Smith, i.e. the clerk. Done."
    assert txt.sentences(t) == ["The ship left in 1861 [S1] before she was sold.", "He met Dr. Smith, i.e. the clerk.",
                                "Done."]
    assert len(txt.sentences('He read "J. Habakuk Jephson" aloud. It ended.')) == 2  # a closing quote before a word


def test_mark_invented_never_nests():
    script = "{{INVENTED: I lean in. Records say so.}} Plain claim."
    claims = ["{{INVENTED: I lean in.", "Records say so.}}", "Plain claim."]
    out, n = grounding.mark_invented(script, claims, lambda s: False)
    assert n == 1 and out.count("{{") == 2 and "{{INVENTED: Plain claim.}}" in out


def test_verify_marks_only_what_the_model_rejects():
    lore = {1: "The Mary Celeste sailed in 1872 with 1701 barrels."}
    script = "She sailed in 1872 [S1]. The log ended mid-sentence [S1]."
    out, n = grounding.verify(script, lore, lambda l, s: "NO" if "mid-sentence" in s else "YES")
    assert n == 1 and "{{INVENTED: The log ended mid-sentence.}}" in out and "{{INVENTED: She" not in out


def test_source_ledger_numbers_and_reports_sources():
    from ragscript.sources import SourceLedger
    led = SourceLedger()
    c = {"id": "lore:lore_a.md:0", "text": "fact", "meta": {"source": "lore_a.md"}}
    assert led.cite([c, c]).count("[S1]") == 2 and led.cited_text == {1: "fact"}
    led.add_style([{"id": "scripts:x.txt:3", "text": "style", "meta": {"source": "x.txt"}}])
    assert led.used() == {"lore_a.md": [1], "x.txt": []} and led.refs == ["fact", "fact", "style"]


def test_multi_sentence_invented_mark_is_not_reported_as_uncited():
    text = "{{INVENTED: I lean in. Captain Briggs smiled.}} Captain Rose spoke."
    assert grounding.uncited_facts(text) == ["Captain Rose spoke."]


def test_unlabelled_double_braces_become_invented_marks_and_never_reach_clean_narration():
    from ragscript import refine
    fixed = refine.clean_draft("Plain. {{I sat in my office.}} {{invented: kept}} Done.")
    assert fixed == "Plain. {{INVENTED: I sat in my office.}} {{invented: kept}} Done."
    assert grounding.clean_narration("A {{stray}} word {{INVENTED: x}}.") == "A stray word x."
