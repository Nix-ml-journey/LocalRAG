"""Verify step for fact-grounded genres: sentences the code cannot trust are checked against the lore by the model."""
import re

from . import checks


def verify(script, cited_text, ask, limit=40):
    """Wrap every unsupported sentence as {{INVENTED: ...}}. Returns (script, number marked).

    cited_text: {S number: lore chunk text}. ask(lore, sentence) -> the model's YES/NO answer, lore as sole evidence.
    Checked: sentences naming something with no citation, cited sentences whose numbers/quotes/names are missing from
    their chunk, and every other cited sentence (a [S1] tag proves nothing: the model also cites invented claims)."""
    flagged = (checks.uncited_facts(script) + [s for s, _ in checks.unsupported_cites(script, cited_text)]
               + [s for s in checks.sentences(script) if re.search(r"\[S\d+\]", s)])

    def supported(s):
        sw = set(checks.words(s))  # the 12 chunks sharing most words with the sentence, so its source is in
        near = sorted(cited_text.values(), key=lambda t: -len(sw & set(checks.words(t))))[:12]
        return not ask("\n\n".join(near), s).strip().upper().startswith("NO")  # an unclear answer leaves it alone

    return checks.mark_invented(script, flagged, supported, limit)
