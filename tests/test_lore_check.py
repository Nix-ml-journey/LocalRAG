"""Offline tests for eval/lore_check.py (pre-ingest checks of lore notes)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.lore_check import check

FM = "---\ngenre: history\nkind: lore\ntopic: Test\nsource: https://example.org/x\nlicense: CC BY-SA 4.0\n---\n"
BODY = ("The brigantine Mary Celeste sailed from New York on 7 November 1872 with a cargo of alcohol. "
        "Captain Briggs commanded her, and the Dei Gratia found her adrift on 4 December 1872. "
        "The ship was seaworthy and the cargo intact, yet the lifeboat was gone. ") * 9  # ~160 words, names repeat


def test_good_note_has_no_errors():
    r = check(FM + BODY, "lore_mary_celeste.md")
    assert r["errors"] == [] and r["warnings"] == []


def test_each_problem_is_flagged():
    assert any("start with 'lore'" in e for e in check(FM + BODY, "mary_celeste.md")["errors"])
    assert any("no front-matter" in e for e in check(BODY, "lore_x.md")["errors"])
    assert any("not valid YAML" in e for e in check("---\ntopic: [unclosed\n---\n" + BODY, "lore_x.md")["errors"])
    assert any("garbled" in e for e in check(FM + BODY + " Captain Briggs� sailed", "lore_x.md")["errors"])
    assert any("garbled" in e for e in check(FM + BODY + " the shipâ€™s log", "lore_x.md")["errors"])
    assert any("too short" in e for e in check(FM + "Only a few words here.", "lore_x.md")["errors"])
    assert any("long" in w for w in check(FM + BODY * 12, "lore_x.md")["warnings"])
    assert any("lacks: license" in w for w in check(FM.replace("license: CC BY-SA 4.0\n", "") + BODY, "lore_x.md")["warnings"])


def test_short_notes_skip_the_single_mention_warning():
    short = " ".join(BODY.split()[:200]) + " A sailor called Morehouse wrote in 1873."  # 200+ words, under 400
    r = check(FM + short, "lore_x.md")
    assert r["errors"] == [] and not any("only once" in w for w in r["warnings"])


def test_single_mention_names_and_years_are_warned_but_repeated_ones_are_not():
    r = check(FM + BODY + " A sailor called Morehouse wrote in 1873 about it.", "lore_x.md")
    warn = " ".join(r["warnings"])
    assert "Morehouse" in warn and "1873" in warn
    assert "Briggs" not in warn and "1872" not in warn and r["errors"] == []
