"""Genre packs: any genre name works. data/<slug>/ holds its examples, prompts/<slug>.md its rules."""
import logging
import re
from dataclasses import dataclass

from . import config, loaders

log = logging.getLogger("ragscript")
GROUNDED_RULE = (
    "GROUNDED: real facts (names, dates, places, numbers, quotations) may come ONLY from the numbered lore chunks "
    "[S1], [S2]... End every sentence that states one of those facts with its tag, for example [S2]. "
    "Any detail you invent must be wrapped as {{INVENTED: ...}} and never presented as verified."
)


@dataclass(frozen=True)
class Genre:
    slug: str       # normalised name, equals the data/<slug>/ folder and the retrieval filter
    rules: str      # text injected as <<genre_rules>> (incl. grounded rule and user style notes)
    grounded: bool  # facts must come from lore, invention is marked


def known(cfg) -> list:
    """Genres that already have a data folder (hooks/ is the shared library, not a genre)."""
    d = config.path(cfg, "data_dir")
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name != "hooks") if d.exists() else []


def load(cfg, name: str, style: str = "") -> Genre:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")  # also blocks path tricks like "../x"
    if not slug:
        raise ValueError("genre must contain letters or digits")
    pdir = config.path(cfg, "prompts_dir")
    f = pdir / f"{slug}.md"
    if not f.exists():
        log.warning("no prompts/%s.md; using prompts/default.md (add one to sharpen this genre)", slug)
        f = pdir / "default.md"
    fm, text = loaders.split_front_matter(f.read_text(encoding="utf-8"))
    grounded = bool(fm.get("grounded", False))
    rules = text.strip()
    if grounded:
        rules += "\n\n" + GROUNDED_RULE
    if style.strip():
        rules += f"\n\nExtra style notes from the user (follow them): {style.strip()}"
    return Genre(slug, rules, grounded)
