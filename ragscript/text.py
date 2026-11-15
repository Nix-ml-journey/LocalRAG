"""Text primitives every check shares: the sentence splitter and the word tokenizer."""
import re

CUE = re.compile(r"\[[^\]]*\]")
# sentence boundary; a [S#] tag after the full stop stays with its sentence, and the tag itself ends one
# an initial ("J. Habakuk"), a title ("Dr. Smith") or a lower-case continuation ("1861 [S1] before", "i.e. the") is not a sentence end;
# a closing }} of an {{INVENTED}} mark is
SENT = r"(?<=[.!?…”\"\]}])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bMrs\.)(?<!\bCapt\.)\s+(?!\[S\d)(?![a-z])"


def sentences(text: str):
    """The one sentence splitter every check uses (initials and titles do not end a sentence)."""
    return re.split(SENT, text)


def words(t: str):
    return re.findall(r"[\w']+", CUE.sub(" ", t).lower().replace("’", "'"))  # \w: "Félix" is one word
