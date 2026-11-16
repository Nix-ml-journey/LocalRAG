"""The story being written: what every stage after the outline needs to know about it."""
from dataclasses import dataclass, field


@dataclass
class Story:
    brief: str
    genre: object          # genre.Genre: slug, rules, grounded
    narrator: str          # the narrator bible from the outline
    total: int             # target words for the whole script
    minutes: int
    hook: str = ""         # the hook the user picked
    notes: list = field(default_factory=list)  # remarks for the Checks section, added by any stage
    outline: str = ""      # the outline as numbered text, for stages that judge the whole script
