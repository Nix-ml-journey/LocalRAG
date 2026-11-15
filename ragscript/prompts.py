"""Prompt templates: the only place that loads them and fills their <<placeholders>>.

Shared values (the sentence-length target, the [PAUSE] limit, the banned phrases) come from config and are available to
every template and to the genre rules; anything else is passed in as a variable."""
import re

from . import config


def fill(template, cfg, banned=(), /, **values):
    """`template` with every <<name>> replaced, in one pass: a value that itself contains <<name>> (lore, a brief) is
    inserted as it is. A placeholder with no value stays as written."""
    shared = {"banned": "\n".join(banned), "sentence_range": config.sentence_range(cfg),
              "max_pause": cfg["style"]["max_pause_per_section"]}
    values = {**shared, **values}
    return re.sub(r"<<(\w+)>>", lambda m: str(values.get(m.group(1), m.group(0))), template)


def stage(cfg, name, banned=(), /, **values):
    """The stage template prompts/stages/<name>.md, filled."""
    template = (config.path(cfg, "prompts_dir") / "stages" / f"{name}.md").read_text(encoding="utf-8")
    return fill(template, cfg, banned, **values)
