"""Multi-stage generation: hooks -> outline -> section-by-section writing -> refine -> checks -> markdown."""
import json
import logging
import re

from . import checks, chunking, config, grounding, render, report, vram
from . import genre as genre_pack
from .llm import LLM
from .refine import Refiner, clean_draft, section_issues, whole_sentences
from .retrieve import Retriever
from .sources import SourceLedger
from .store import Store

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
    """The story pipeline. Dependencies are passed in: `ret` answers retrieval searches, `llm.chat(prompt, stage=...)`
    answers model calls, `store` is only needed for ingest. `Pipeline.from_config()` builds the real ones."""

    def __init__(self, cfg, ret, llm, store=None, banned=()):
        self.cfg, self.ret, self.llm, self.store, self.banned = cfg, ret, llm, store, list(banned)
        self.refiner = Refiner(self._ask, cfg, self.banned)

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or config.load()
        store = Store(cfg)
        return cls(cfg, Retriever(store, cfg), LLM(cfg), store, load_banned(cfg))

    # ---- helpers -------------------------------------------------------
    def _tpl(self, name, **kw):
        t = (config.path(self.cfg, "prompts_dir") / "stages" / f"{name}.md").read_text(encoding="utf-8")
        shared = {"banned": "\n".join(self.banned), "sentence_range": config.sentence_range(self.cfg),
                  "max_pause": self.cfg["style"]["max_pause_per_section"]}
        for k, v in {**shared, **kw}.items():
            t = t.replace(f"<<{k}>>", str(v))
        return t

    def _ask(self, stage, tpl_vars, **chat):
        """One model call: the `stage` template filled with `tpl_vars`. The stage name travels with the call."""
        return self.llm.chat(self._tpl(stage, **tpl_vars), stage=stage, **chat)

    def _k(self):
        return max(1, round(self.cfg["retrieval"]["final_k"] * self.llm.scale()))  # shrink if ctx was reduced

    def ingest(self):
        return self.store.ingest(lambda text: self._ask("tagger", {"text": text}, temperature=0.1, max_tokens=200))

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

    def _write_section(self, prompt, b, earlier, grounded):
        # models write 3-5x a small word budget, so cap generation and cut the unfinished last sentence
        cap = int(b["words"] * self.cfg["length"]["token_cap"]) + 40
        text = clean_draft(whole_sentences(self.llm.chat(prompt, stage="section", max_tokens=cap)))
        problems = section_issues(text, earlier, self.cfg, self.banned, budget=b["words"], grounded=grounded)
        if problems:  # one corrective retry
            log.info("retry '%s': %s", b["beat"], "; ".join(p[:60] for p in problems))
            text = clean_draft(whole_sentences(self.llm.chat(
                prompt + "\n\nPrevious attempt:\n" + text + "\n\nRewrite it; " + "; ".join(problems) + ".",
                stage="section", max_tokens=cap)))
        return text

    # ---- stage 3 + 4 ---------------------------------------------------
    def write(self, brief, genre, minutes, hook, style="", progress=print):
        g = genre_pack.load(self.cfg, genre, style)
        genre = g.slug  # normalised name: matches data/<slug>/ and the retrieval filter
        wpm = self.cfg["script"]["wpm"]

        def stage(msg):  # every stage reports progress and logs VRAM/RAM under its own name
            progress(msg)
            vram.log_stage(msg)

        ledger = SourceLedger()
        cite = ledger.cite

        notes = []
        lore0 = self.ret.search("lore", brief, genre, k=self._k())
        if g.grounded and not lore0:
            notes.append("No lore chunks found for this fact-grounded brief: every factual claim is unverified.")
        stage("Outline…")
        narrator, beats, total = self.outline(brief, g, minutes, hook, cite(lore0))
        outline_txt = "\n".join(f"{i+1}. {b['beat']} ({b['words']}w): {b.get('title','')} - {b.get('summary','')}"
                                for i, b in enumerate(beats))
        summary, sections = "(nothing written yet)", []
        for i, b in enumerate(beats):
            stage(f"Writing {i+1}/{len(beats)}: {b['beat']}")
            if i:  # rolling budget: spread any over/undershoot so far across the beats still to write
                # overshoot counts only up to the planned size: the polish pass trims drafts back to it
                done = sum(min(checks.spoken_count(t), x["words"]) for x, t in sections)
                left = (total - done) / sum(x["orig"] for x in beats[i:])
                for x in beats[i:]:
                    x["words"] = max(30, round(x["orig"] * min(1.8, max(0.5, left))))
            k = self._k()  # re-read each beat: an OOM retry may have halved num_ctx
            q = f"{b.get('title','')} {b.get('summary','')}"
            style = self.ret.search("scripts", q, genre, beat=b["beat"], k=k) or self.ret.search("scripts", q, genre, k=k)
            if b["beat"] == "outro":  # use the endings library for the last beat
                style += self.ret.search("hooks", q, genre, beat="ending", k=k)
            lore = self.ret.search("lore", f"{brief} {q}", genre, k=k)
            beat_txt = f"{b['beat']}: {b.get('title','')} - {b.get('summary','')}"
            if b["beat"] == "hook":
                beat_txt += f"\nOpen with this chosen hook, polished and extended:\n{hook}"
            ledger.add_style(style)  # so the >80% overlap check also covers style examples
            tail = ""
            if sections:
                tail = ("\n\nEnd of the text written so far. Do NOT repeat its sentences, refrains or images; "
                        "advance the story with new events:\n" + " ".join(sections[-1][1].split()[-120:]))
            prompt = self._tpl(
                "section", genre_rules=g.rules, narrator=narrator, outline=outline_txt, beat=beat_txt,
                beat_words=b["words"], summary=summary + tail,
                style_examples="\n\n".join(f"- {c['text']}" for c in style) or "(none)",
                lore=cite(lore) or "(none)")
            earlier = "\n\n".join(t for _, t in sections)
            text = self._write_section(prompt, b, earlier, g.grounded)
            # drop re-told paragraphs/sentences now so the rolling budget compensates for the lost words
            text = checks.dedupe(text, earlier) or text
            sections.append((b, text))
            if i < len(beats) - 1:  # the last summary would never be read
                summary = self._ask("summary", dict(previous_summary=summary, new_text=text),
                                    temperature=0.2, max_tokens=300)

        parts = self.refiner.run(sections, total, narrator, g, notes, stage)
        script = checks.dedupe("\n\n".join(parts))  # final safety net: nothing repeated may reach the output
        if not g.grounded:  # [S#] tags only help fact-checking; ungrounded stories get them leaked by the model
            script = re.sub(r"[ \t]*\[S\d+\]", "", script)
        if g.grounded and ledger.cited_text:
            stage("Verifying…")
            script, marked = grounding.verify(script, ledger.cited_text, lambda lore, sent: self._ask(
                "verify", dict(lore=lore, sentence=sent), temperature=0.0, max_tokens=5))
            if marked:
                notes.append(f"Verify step marked {marked} unsupported sentence(s) as INVENTED.")

        stage("Packaging…")
        pkg = self._json("package", dict(genre=genre, brief=brief, outline=outline_txt,
                                         excerpt=" ".join(script.split()[:500])), (0.7, 0.7),
                         lambda d: d["titles"], max_tokens=800) or {}  # a list or text answer raises here and is retried
        rep = report.build(script, target_words=total, minutes=minutes, refs=ledger.refs, cited_text=ledger.cited_text,
                           grounded=g.grounded, banned=self.banned, cfg=self.cfg, notes=notes)
        doc = render.Doc(brief, genre, minutes, narrator, pkg,
                         render.chapters([(b, checks.spoken_count(t)) for b, t in sections], rep.words, wpm),
                         ledger.used())
        md = render.render(doc, script, rep)
        out = config.path(self.cfg, "output_dir")
        out.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", brief.lower())[:50].strip("-") or "script"
        path = out / f"{genre}-{minutes}min-{slug}.md"
        path.write_text(md, encoding="utf-8")
        vram.log_stage("done")
        return {"path": path, "markdown": md, "stats": rep}
