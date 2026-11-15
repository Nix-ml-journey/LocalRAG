"""Every measurement taken on a finished script, in one object: the Checks section of the output, and the numbers the
eval scripts compare. Adding a check means adding a field here and a line in render.py, nowhere else."""
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
    notes: list = field(default_factory=list)


def build(script, *, target_words, minutes, refs, cited_text, grounded, banned, cfg, notes) -> Report:
    """Measure `script`. refs: every source chunk text shown to the model; cited_text: {S number: chunk text};
    the two fact checks only run for grounded genres."""
    words = checks.spoken_count(script)
    return Report(
        words=words, target_words=target_words,
        word_accuracy=round(1 - abs(words - target_words) / target_words, 3),
        repeated_phrase_rate=round(checks.repeated_rate(script, cfg["checks"]["repeated_ngram"]), 4),
        avg_sentence_words=round(checks.avg_sentence_words(script), 1),
        sentence_range=config.sentence_range(cfg),
        overlap_flags=checks.overlap_flags(script, refs, cfg["checks"]["ngram"], cfg["checks"]["overlap_threshold"]),
        banned_hits=checks.banned_hits(script, banned),
        invented=grounding.invented(script),
        unsupported_cites=grounding.unsupported_cites(script, cited_text) if grounded else [],
        uncited_facts=grounding.uncited_facts(script) if grounded else [],
        pause_cues=script.count("[PAUSE]"), minutes=minutes,
        mixed_tense=checks.mixed_tense(script), overused_words=checks.overused_words(script), notes=notes)
