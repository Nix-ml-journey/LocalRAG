"""Section-by-section drafting: for each outline beat, retrieve style examples and lore, write the section, check it,
retry once, and keep a rolling summary and word budget. `Drafter.draft` is the one interface the pipeline uses."""
import logging

from . import checks
from .refine import clean_reply, draft_issues

log = logging.getLogger("ragscript")


class Drafter:
    """ret.search(...) answers retrieval; ask(stage, vars, tail=, **chat_options) answers model calls; k() is how many
    chunks to retrieve now (it shrinks if the context was halved after an out-of-memory error)."""

    def __init__(self, ret, ask, k, cfg, banned):
        self.ret, self.ask, self.k, self.cfg, self.banned = ret, ask, k, cfg, banned

    def draft(self, story, beats, ledger, outline_txt, stage):
        """The drafted sections [(beat, text)] for every beat of the outline."""
        genre, summary, sections = story.genre, "(nothing written yet)", []
        for i, b in enumerate(beats):
            stage(f"Writing {i+1}/{len(beats)}: {b['beat']}")
            if i:
                self._rebudget(beats, i, sections, story.total)
            k = self.k()  # re-read each beat: an OOM retry may have halved num_ctx
            q = f"{b.get('title','')} {b.get('summary','')}"
            examples = (self.ret.search("scripts", q, genre.slug, beat=b["beat"], k=k)
                        or self.ret.search("scripts", q, genre.slug, k=k))
            if b["beat"] == "outro":  # use the endings library for the last beat
                examples += self.ret.search("hooks", q, genre.slug, beat="ending", k=k)
            lore = self.ret.search("lore", f"{story.brief} {q}", genre.slug, k=k)
            beat_txt = f"{b['beat']}: {b.get('title','')} - {b.get('summary','')}"
            if b["beat"] == "hook":
                beat_txt += f"\nOpen with this chosen hook, polished and extended:\n{story.hook}"
            ledger.add_style(examples)  # so the >80% overlap check also covers style examples
            tail = ""
            if sections:
                tail = ("\n\nEnd of the text written so far. Do NOT repeat its sentences, refrains or images; "
                        "advance the story with new events:\n" + " ".join(sections[-1][1].split()[-120:]))
            vars_ = dict(brief=story.brief, genre_rules=genre.rules, narrator=story.narrator, outline=outline_txt,
                         beat=beat_txt, beat_words=b["words"], summary=summary + tail,
                         style_examples="\n\n".join(f"- {c['text']}" for c in examples) or "(none)",
                         lore=ledger.cite(lore) or "(none)")
            earlier = "\n\n".join(t for _, t in sections)
            text = self._write_section(vars_, b, earlier, genre.grounded)
            # drop re-told paragraphs/sentences now so the rolling budget compensates for the lost words
            text = checks.dedupe(text, self.cfg["checks"], earlier) or text
            sections.append((b, text))
            if i < len(beats) - 1:  # the last summary would never be read
                summary = self.ask("summary", dict(previous_summary=summary, new_text=text),
                                   temperature=0.2, max_tokens=300)
        return sections

    @staticmethod
    def _rebudget(beats, i, sections, total):
        """Rolling budget: spread any over/undershoot so far across the beats still to write. Overshoot counts only
        up to the planned size, because the polish pass trims drafts back to it."""
        done = sum(min(checks.spoken_count(t), x["words"]) for x, t in sections)
        left = (total - done) / sum(x["orig"] for x in beats[i:])
        for x in beats[i:]:
            x["words"] = max(30, round(x["orig"] * min(1.8, max(0.5, left))))

    def _write_section(self, vars_, b, earlier, grounded):
        # models write 3-5x a small word budget, so cap generation and cut the unfinished last sentence
        cap = int(b["words"] * self.cfg["length"]["token_cap"]) + 40
        text = clean_reply(self.ask("section", vars_, max_tokens=cap))
        issues = draft_issues(text, earlier, self.cfg, self.banned, b["words"], grounded)
        if issues:  # one corrective retry
            log.info("retry '%s': %s", b["beat"], "; ".join(i.message[:60] for i in issues))
            tail = "\n\nPrevious attempt:\n" + text + "\n\nRewrite it; " + "; ".join(i.message for i in issues) + "."
            text = clean_reply(self.ask("section", vars_, tail=tail, max_tokens=cap))
        return text
