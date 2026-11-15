"""Every measurement taken on a finished script, in one object: the Checks section of the output, and the numbers the
eval scripts compare. The Report also decides what deserves a warning (`flags`), from config values, so render.py only
prints and the eval scripts only read. Adding a check means adding a field here and a line in render.py."""
from dataclasses import dataclass, field

from . import checks, config, grounding


@dataclass
class Report:
    words: int
    target_words: int
    word_accuracy: float
    repeated_phrase_rate: float
    avg_sentence_words: float
    sentence_range: str             # the configured target, e.g. "10-15", shown next to the measurement
    overlap_flags: list
    banned_hits: list
    invented: list
    unsupported_cites: list
    uncited_facts: list
    pause_cues: int
    minutes: int
    mixed_tense: bool
    overused_words: list
    repeated_images: list = field(default_factory=list)  # [(phrase, count)] returning through the script
    flags: list = field(default_factory=list)            # warnings the Report itself decided on
    notes: list = field(default_factory=list)

    @property
    def pauses_per_minute(self) -> float:
        return self.pause_cues / self.minutes


def build(script, *, target_words, minutes, refs, cited_text, grounded, banned, cfg, notes, brief="") -> Report:
    """Measure `script`. refs: every source chunk text shown to the model; cited_text: {S number: chunk text};
    the two fact checks only run for grounded genres."""
    ck = cfg["checks"]
    words = checks.spoken_count(script)
    r = Report(
        words=words, target_words=target_words,
        word_accuracy=round(1 - abs(words - target_words) / target_words, 3),
        repeated_phrase_rate=round(checks.repeated_rate(script, ck["repeated_ngram"]), 4),
        avg_sentence_words=round(checks.avg_sentence_words(script), 1),
        sentence_range=config.sentence_range(cfg),
        overlap_flags=checks.overlap_flags(script, refs, ck["ngram"], ck["overlap_threshold"]),
        banned_hits=checks.banned_hits(script, banned),
        invented=grounding.invented(script),
        unsupported_cites=grounding.unsupported_cites(script, cited_text) if grounded else [],
        uncited_facts=grounding.uncited_facts(script) if grounded else [],
        pause_cues=script.count("[PAUSE]"), minutes=minutes,
        mixed_tense=checks.mixed_tense(script, ck), overused_words=checks.overused_words(script, ck, skip=checks.topic_words(brief)),
        repeated_images=checks.crowded_phrases(script, ck["phrase_repeat"]), notes=notes)
    if r.mixed_tense:
        r.flags.append("**Tense switches between past and present in the narration (check before recording).**")
    limit = cfg["style"]["max_pauses_per_minute"]
    if r.pauses_per_minute > limit:
        r.flags.append(f"{r.pause_cues} [PAUSE] cues in {minutes} minutes: more than {limit} a minute reads as halting")
    return r
