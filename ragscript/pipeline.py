"""The story pipeline: hooks -> outline -> draft (draft.py) -> refine (refine.py) -> verify -> package -> markdown.
This module orchestrates; each stage's work lives in its own module."""
import json
import logging
import re
from dataclasses import asdict, dataclass, field

from . import checks, chunking, config, grounding, logic, prompts, report, vram
from . import genre as genre_pack
from .draft import Drafter
from .ingest import Ingest
from .llm import LLM
from .refine import Refiner
from .sources import SourceLedger
from .store import Retriever, Store

log = logging.getLogger("ragscript")


def parse_json(s: str):
    m = re.search(r"[\[{].*[\]}]", s, re.S)
    return json.loads(m.group(0))


def load_banned(cfg):
    p = config.path(cfg, "banned_phrases")
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


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
    details: list = field(default_factory=list)  # names of the outline's recurring details
    payoff: str = ""       # the outro's final image, which an earlier beat planted


def _valid_outline(d):
    """The outline JSON is usable when it has beats; its beats then become `Beat` objects (a bad one raises)."""
    assert d["beats"]
    d["beats"] = [chunking.Beat.from_json(b) for b in d["beats"]]


def name_details(narrator, beats):
    """A beat whose detail is labelled "Detail 2: ..." takes the name of the narrator's second fixed detail, so the
    detail checks (returns, over-use) look for a real word instead of "detail". No model call."""
    named = {int(n): re.sub(r"(?i)^(?:the|a|an)\s+", "", t).strip()
             for n, t in re.findall(r"(?i)detail\s*(\d)\s*:\s*([^;\n]+)", narrator)}
    for b in beats:
        if (m := re.match(r"(?i)\s*detail\s*(\d)\s*:\s*(.*)", b.detail)) and int(m.group(1)) in named:
            b.detail = f"{named[int(m.group(1))]}: {m.group(2)}"


def build_outline(ask_json, cfg, total, notes=None):
    """The outline as (narrator, [Beat]), the whole job in one place. `ask_json(temps, tail="", **model_options)` asks
    the model and returns the parsed dict (beats already `Beat`s) or None. A beat that shows a screen takes its `sees`
    from a neighbour; if the outline still has logic holes, up to two repairs are asked for, each kept only if it has fewer;
    budgets are then scaled to sum to `total`."""
    # thinking can use the whole token budget and return no JSON at all, so the retries run with thinking off
    data = ask_json((0.6,), think=cfg["llm"]["think_outline"]) or ask_json((0.4, 0.2), think=False)
    if data is None:
        raise RuntimeError("could not get a valid outline from the model; try again or switch model")
    limit, beats = cfg["checks"]["motif_max"], data["beats"]
    beats[-1].plant = ""  # the outro holds the payoff; a plant copied there is no early setup, so it must not count as one
    name_details(str(data.get("narrator", "")), beats)
    logic.fill_sees(beats)
    for _ in range(2):  # up to two repairs, each kept only if it has fewer holes
        if not (issues := logic.outline_issues(beats, limit)):
            break
        again = ask_json((0.4,), think=False,
                         tail="\n\nYour previous outline had these logic problems; fix them and output the "
                              "complete JSON again:\n- " + "\n- ".join(issues))
        if again:
            again["beats"][-1].plant = ""
            name_details(str(again.get("narrator", data.get("narrator", ""))), again["beats"])
            logic.fill_sees(again["beats"])
        if not again or len(logic.outline_issues(again["beats"], limit)) >= len(issues):
            break
        data, beats = again, again["beats"]
    if (issues := logic.outline_issues(beats, limit)) and notes is not None:
        notes.append("Outline logic problems left: " + "; ".join(issues) + ".")
    scale = total / sum(b.words for b in beats)  # force budgets to sum to the target
    floor = max(40, round(total * 0.07))  # tiny beats get padded and repeat earlier text
    for b in beats:
        b.words = b.orig = max(floor, round(b.words * scale))
    return str(data.get("narrator", "")).replace("_", " "), beats  # snake_case in the bible ("glove_grip") reached the story


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
    def outline(self, brief, g, minutes, hook, lore_txt, notes=None):
        total = minutes * self.cfg["script"]["wpm"]
        tpl_vars = dict(brief=brief, genre=g.slug, genre_rules=g.rules, hook=hook, minutes=minutes,
                        total_words=total, lore=lore_txt)

        def ask_json(temps, tail="", **kw):
            return self._json("outline", tpl_vars, temps, _valid_outline, tail=tail, **kw)

        narrator, beats = build_outline(ask_json, self.cfg, total, notes)
        return narrator, beats, total

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
        narrator, beats, total = self.outline(brief, g, minutes, hook, ledger.cite(lore0), notes)
        outline_txt = "\n".join(f"{i+1}. {b.beat} ({b.words}w): {b.title} - {b.summary}"
                                + (f" [narrator at: {b.place}]" if b.place else "")
                                + (f" [planted here: {b.plant}]" if b.plant else "")
                                + (f" [final image: {b.payoff}]" if b.payoff else "") for i, b in enumerate(beats))
        details = [b.detail.split(":")[0].strip() for b in beats]
        story = Story(brief, g, narrator, total, minutes, hook, notes, outline_txt, details, beats[-1].payoff)

        sections = self.drafter.draft(story, beats, ledger, outline_txt, stage)
        parts = self.refiner.run(sections, story, stage)
        ck = self.cfg["checks"]
        script = checks.tidy("\n\n".join(p for p in parts if p))
        safe = checks.dedupe(script, ck)  # final safety net: nothing repeated may reach the output
        if lost := checks.spoken_count(script) - checks.spoken_count(safe):
            notes.append(f"Final repeat guard removed {lost} words after the length fit.")
        script = checks.tidy(safe)  # dedupe can leave a leading space where it cut a sentence
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
                           grounded=g.grounded, banned=self.banned, cfg=self.cfg, notes=notes, brief=brief,
                           details=details,
                           setting=checks.setting_words(brief, *[f"{b.place} {b.sees} {b.detail}" for b in beats]))
        doc = report.Doc(brief, slug, minutes, narrator, pkg,
                         report.chapters([(b, checks.spoken_count(t)) for b, t in sections], rep.words,
                                         self.cfg["script"]["wpm"]),
                         ledger.used())
        md = report.render(doc, script, rep)
        trace = dict(brief=brief, outline=[asdict(b) for b in beats], drafted=[t for _, t in sections], refined=parts,
                     notes=notes)
        path = report.save(config.path(self.cfg, "output_dir"), slug, minutes, brief, md, trace)
        vram.log_stage("done")
        return {"path": path, "markdown": md, "stats": rep}
