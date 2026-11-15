"""The story pipeline: hooks -> outline -> draft (draft.py) -> refine (refine.py) -> verify -> package -> markdown.
This module orchestrates; each stage's work lives in its own module."""
import json
import logging
import re

from . import checks, chunking, config, grounding, prompts, render, report, vram
from . import genre as genre_pack
from .draft import Drafter
from .ingest import Ingest
from .llm import LLM
from .refine import Refiner
from .retrieve import Retriever
from .sources import SourceLedger
from .store import Store
from .story import Story

log = logging.getLogger("ragscript")


def parse_json(s: str):
    m = re.search(r"[\[{].*[\]}]", s, re.S)
    return json.loads(m.group(0))


def load_banned(cfg):
    p = config.path(cfg, "banned_phrases")
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


class Pipeline:
    """Dependencies are passed in: `ret` answers retrieval searches, `llm.chat(prompt, stage=...)` answers model
    calls, `store` is only needed for ingest. `Pipeline.from_config()` builds the real ones."""

    def __init__(self, cfg, ret, llm, store=None, banned=()):
        self.cfg, self.ret, self.llm, self.store, self.banned = cfg, ret, llm, store, list(banned)
        self.drafter = Drafter(ret, self._ask, self._k, cfg, self.banned)
        self.refiner = Refiner(self._ask, cfg, self.banned)

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or config.load()
        store = Store(cfg)
        return cls(cfg, Retriever(store, cfg), LLM(cfg), store, load_banned(cfg))

    # ---- helpers -------------------------------------------------------
    def _ask(self, stage, tpl_vars, tail="", **chat):
        """One model call: the `stage` template filled with `tpl_vars` (plus `tail`, text appended after it). The
        stage name travels with the call."""
        return self.llm.chat(prompts.stage(self.cfg, stage, self.banned, **tpl_vars) + tail, stage=stage, **chat)

    def _k(self):
        return max(1, round(self.cfg["retrieval"]["final_k"] * self.llm.scale()))  # shrink if ctx was reduced

    def ingest(self):
        return Ingest(self.store, self.cfg).run(
            lambda text: self._ask("tagger", {"text": text}, temperature=0.1, max_tokens=200))

    # ---- stage 1 -------------------------------------------------------
    def hooks(self, brief, genre, style=""):
        g = genre_pack.load(self.cfg, genre, style)
        ex = self.ret.search("hooks", brief, g.slug, beat="hook", k=self._k())
        out = self._ask("hooks", dict(brief=brief, genre=g.slug, genre_rules=g.rules,
                                      hook_examples="\n\n".join(f"- {c['text']}" for c in ex)))
        hooks = [h.strip() for h in re.split(r"(?m)^\s*\d+[.)]\s+", out)[1:] if h.strip()]
        return hooks[:3] or [out]

    def _json(self, stage, tpl_vars, temps, check=lambda d: None, **kw):
        """Model output parsed as JSON: one attempt per temperature in `temps`; `check` raises if the content is
        invalid. Returns None when no attempt works."""
        for t in temps:
            try:
                data = parse_json(self._ask(stage, tpl_vars, temperature=t, **kw))
                check(data)
                return data
            except Exception as e:
                log.warning("JSON answer rejected (%s)", e)
        return None

    # ---- stage 2 -------------------------------------------------------
    def outline(self, brief, g, minutes, hook, lore_txt):
        total = minutes * self.cfg["script"]["wpm"]
        tpl_vars = dict(brief=brief, genre=g.slug, genre_rules=g.rules, hook=hook, minutes=minutes,
                        total_words=total, lore=lore_txt)

        def valid(d):
            assert d["beats"]
            for b in d["beats"]:  # "Rising Tension" -> rising_tension; the code compares beat ids exactly
                b["beat"] = str(b["beat"]).strip().lower().replace(" ", "_")
                assert b["beat"] in chunking.BEATS and int(b["words"]) > 0

        # thinking can use the whole token budget and return no JSON at all, so the retries run with thinking off
        data = (self._json("outline", tpl_vars, (0.6,), valid, think=self.cfg["llm"]["think_outline"])
                or self._json("outline", tpl_vars, (0.4, 0.2), valid, think=False))
        if data is None:
            raise RuntimeError("could not get a valid outline from the model; try again or switch model")
        beats = data["beats"]
        scale = total / sum(int(b["words"]) for b in beats)  # force budgets to sum to the target
        floor = max(40, round(total * 0.07))  # tiny beats get padded and repeat earlier text
        for b in beats:
            b["words"] = b["orig"] = max(floor, round(int(b["words"]) * scale))
        return data.get("narrator", ""), beats, total

    # ---- stages 3 to 6 -------------------------------------------------
    def write(self, brief, genre, minutes, hook, style="", progress=print):
        g = genre_pack.load(self.cfg, genre, style)
        slug = g.slug  # normalised name: matches data/<slug>/ and the retrieval filter

        def stage(msg):  # every stage reports progress and logs VRAM/RAM under its own name
            progress(msg)
            vram.log_stage(msg)

        ledger, notes = SourceLedger(), []
        lore0 = self.ret.search("lore", brief, slug, k=self._k())
        if g.grounded and not lore0:
            notes.append("No lore chunks found for this fact-grounded brief: every factual claim is unverified.")
        stage("Outline…")
        narrator, beats, total = self.outline(brief, g, minutes, hook, ledger.cite(lore0))
        story = Story(brief, g, narrator, total, minutes, hook, notes)
        outline_txt = "\n".join(f"{i+1}. {b['beat']} ({b['words']}w): {b.get('title','')} - {b.get('summary','')}"
                                for i, b in enumerate(beats))

        sections = self.drafter.draft(story, beats, ledger, outline_txt, stage)
        parts = self.refiner.run(sections, story, stage)
        ck = self.cfg["checks"]
        script = checks.dedupe("\n\n".join(parts), ck)  # final safety net: nothing repeated may reach the output
        if not g.grounded:  # [S#] tags only help fact-checking; ungrounded stories get them leaked by the model
            script = re.sub(r"[ \t]*\[S\d+\]", "", script)
        if g.grounded and ledger.cited_text:
            stage("Verifying…")
            script, marked = grounding.verify(script, ledger.cited_text, lambda lore, sent: self._ask(
                "verify", dict(lore=lore, sentence=sent), temperature=0.0, max_tokens=5))
            if marked:
                notes.append(f"Verify step marked {marked} unsupported sentence(s) as INVENTED.")

        stage("Packaging…")
        pkg = self._json("package", dict(genre=slug, brief=brief, outline=outline_txt,
                                         excerpt=" ".join(script.split()[:500])), (0.7, 0.7),
                         lambda d: d["titles"], max_tokens=800) or {}  # a list or text answer raises here and is retried
        rep = report.build(script, target_words=total, minutes=minutes, refs=ledger.refs, cited_text=ledger.cited_text,
                           grounded=g.grounded, banned=self.banned, cfg=self.cfg, notes=notes, brief=brief)
        doc = render.Doc(brief, slug, minutes, narrator, pkg,
                         render.chapters([(b, checks.spoken_count(t)) for b, t in sections], rep.words,
                                         self.cfg["script"]["wpm"]),
                         ledger.used())
        md = render.render(doc, script, rep)
        out = config.path(self.cfg, "output_dir")
        out.mkdir(exist_ok=True)
        path = out / f"{slug}-{minutes}min-{re.sub(r'[^a-z0-9]+', '-', brief.lower())[:50].strip('-') or 'script'}.md"
        path.write_text(md, encoding="utf-8")
        vram.log_stage("done")
        return {"path": path, "markdown": md, "stats": rep}
