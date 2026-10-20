"""Multi-stage generation: hooks -> outline -> section-by-section writing -> polish -> checks -> markdown."""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from . import checks, chunking, config, render, vram
from . import genre as genre_pack
from .llm import LLM
from .retrieve import Retriever
from .store import Store

log = logging.getLogger("ragscript")


def parse_json(s: str):
    m = re.search(r"[\[{].*[\]}]", s, re.S)
    return json.loads(m.group(0))


class Pipeline:
    def __init__(self):
        self.cfg = config.load()
        self.store = Store(self.cfg)
        self.ret = Retriever(self.store, self.cfg)
        self.llm = LLM(self.cfg)
        self.banned = self._banned()

    # ---- helpers -------------------------------------------------------
    def _banned(self):
        p = config.path(self.cfg, "banned_phrases")
        if not p.exists():
            return []
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]

    def _tpl(self, name, **kw):
        t = (config.path(self.cfg, "prompts_dir") / "stages" / f"{name}.md").read_text(encoding="utf-8")
        for k, v in kw.items():
            t = t.replace(f"<<{k}>>", str(v))
        return t

    def _k(self):
        return max(1, round(self.cfg["retrieval"]["final_k"] * self.llm.scale()))  # shrink if ctx was reduced

    def ingest(self):
        return self.store.ingest(lambda text: self.llm.chat(self._tpl("tagger", text=text), temperature=0.1, max_tokens=200))

    # ---- stage 1 -------------------------------------------------------
    def hooks(self, brief, genre, style=""):
        g = genre_pack.load(self.cfg, genre, style)
        ex = self.ret.search("hooks", brief, g.slug, beat="hook", k=self._k())
        out = self.llm.chat(self._tpl(
            "hooks", brief=brief, genre=g.slug, genre_rules=g.rules,
            hook_examples="\n\n".join(f"- {c['text']}" for c in ex), banned="\n".join(self.banned)))
        hooks = [h.strip() for h in re.split(r"(?m)^\s*\d+[.)]\s+", out)[1:] if h.strip()]
        return hooks[:3] or [out]

    def _json(self, prompt, temps, check=lambda d: None, **kw):
        """Model output parsed as JSON: one attempt per temperature in `temps`; `check` raises if the content is
        invalid. Returns None when no attempt works."""
        for t in temps:
            try:
                data = parse_json(self.llm.chat(prompt, temperature=t, **kw))
                check(data)
                return data
            except Exception as e:
                log.warning("JSON answer rejected (%s)", e)
        return None

    # ---- stage 2 -------------------------------------------------------
    def outline(self, brief, g, minutes, hook, lore_txt):
        total = minutes * self.cfg["script"]["wpm"]
        prompt = self._tpl("outline", brief=brief, genre=g.slug, genre_rules=g.rules, hook=hook,
                           minutes=minutes, total_words=total, lore=lore_txt)
        def valid(d):
            assert d["beats"]
            for b in d["beats"]:  # "Rising Tension" -> rising_tension; the code compares beat ids exactly
                b["beat"] = str(b["beat"]).strip().lower().replace(" ", "_")
                assert b["beat"] in chunking.BEATS and int(b["words"]) > 0

        data = self._json(prompt, (0.6, 0.2), valid, think=self.cfg["llm"]["think_outline"])
        if data is None:
            raise RuntimeError("could not get a valid outline from the model; try again or switch model")
        beats = data["beats"]
        scale = total / sum(int(b["words"]) for b in beats)  # force budgets to sum to the target
        floor = max(40, round(total * 0.07))  # tiny beats get padded and repeat earlier text
        for b in beats:
            b["words"] = b["orig"] = max(floor, round(int(b["words"]) * scale))
        return data.get("narrator", ""), beats, total

    def _problems(self, text, b, earlier, grounded):
        """Reasons a section draft must be regenerated (empty list = accept)."""
        p, n = [], checks.spoken_count(text)
        # gemma/qwen overshoot small budgets by 1.5-5x every time; the polish pass trims that, so a rewrite is
        # only worth a call when the draft is far too long or too short
        if not 0.75 * b["words"] <= n <= 2.5 * b["words"]:
            p.append(f"it was {n} words but must be about {b['words']}")
        if checks.repeated_rate(text, self.cfg["checks"]["repeated_ngram"]) > 0.08:
            p.append("it repeated its own paragraphs or refrains; mention each recurring detail at most once")
        if earlier and checks.overlap_flags(text, [earlier], 4, 0.6):
            p.append("it repeated earlier paragraphs; write only NEW events that move the story forward")
        if checks.reused_quotes(text, earlier):
            p.append("it repeated a quotation already used earlier; quote each source only once")
        if grounded and (u := checks.uncited_facts(text)):
            p.append("these sentences state names, dates or numbers with neither a [S#] citation nor an "
                     "{{INVENTED: ...}} mark (cite them, mark them, or cut them): " + " | ".join(u[:3]))
        if hits := checks.banned_hits(text, self.banned):
            p.append("it used banned phrases: " + "; ".join(hits))
        return p

    def _write_section(self, prompt, b, earlier, grounded):
        text = self.llm.chat(prompt)
        problems = self._problems(text, b, earlier, grounded)
        if problems:  # one corrective retry
            log.info("retry '%s': %s", b["beat"], "; ".join(p[:60] for p in problems))
            text = self.llm.chat(prompt + "\n\nPrevious attempt:\n" + text + "\n\nRewrite it; " + "; ".join(problems) + ".")
        return text

    def _fit_length(self, parts, budgets, total, narrator, g, notes, max_calls=4):
        """Final length control: while the script is more than 10% off target, rewrite the untried section furthest
        over (or under) its budget. A rewrite is kept only if it moves the total toward the target AND leaves the
        script free of repeats, so this step can never undo the anti-repetition pass."""
        words = [checks.spoken_count(p) for p in parts]
        tried = set()
        for _ in range(max_calls):
            n = sum(words)
            if 0.9 * total <= n <= 1.1 * total:
                break
            sign = 1 if n > total else -1
            i = max((j for j in range(len(parts)) if j not in tried), default=None,
                    key=lambda j: sign * (words[j] - budgets[j]))
            if i is None:
                break
            tried.add(i)
            target = round(min(1.8, max(0.4, 1 + (total - n) / max(words[i], 1))) * words[i])
            new = self.llm.chat(self._tpl(
                "fit", text=parts[i], narrator=narrator, genre_rules=g.rules, banned="\n".join(self.banned),
                current_words=words[i], target_words=target),
                temperature=0.5, max_tokens=int(max(words[i], target) * 2.2))
            trial = parts[:i] + [new] + parts[i + 1:]
            joined, w = "\n\n".join(trial), checks.spoken_count(new)
            # compare words, not layout: dedupe joins a line break after a full stop into a space
            if abs(n - words[i] + w - total) < abs(n - total) and checks.dedupe(joined).split() == joined.split():
                parts, words[i] = trial, w
                notes.append(f"Length fit rewrote section {i + 1}: script is now {sum(words)} words (target {total}).")
        return parts

    # ---- stage 3 + 4 ---------------------------------------------------
    def write(self, brief, genre, minutes, hook, style="", progress=print):
        g = genre_pack.load(self.cfg, genre, style)
        genre = g.slug  # normalised name: matches data/<slug>/ and the retrieval filter
        wpm = self.cfg["script"]["wpm"]

        def stage(msg):  # every stage reports progress and logs VRAM/RAM under its own name
            progress(msg)
            vram.log_stage(msg)

        sources, refs, cited_text = {}, [], {}  # chunk id -> S number ; texts for overlap check ; S number -> text
        style_used = set()

        def cite(chunks):
            lines = []
            for c in chunks:
                n = sources.setdefault(c["id"], len(sources) + 1)
                refs.append(c["text"])
                cited_text[n] = c["text"]
                lines.append(f"[S{n}] ({c['meta'].get('source')}) {c['text']}")
            return "\n\n".join(lines)

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
            q =f"{b.get('title','')} {b.get('summary','')}"
            style = self.ret.search("scripts", q, genre, beat=b["beat"], k=k) or self.ret.search("scripts", q, genre, k=k)
            if b["beat"] == "outro":  # use the endings library for the last beat
                style += self.ret.search("hooks", q, genre, beat="ending", k=k)
            lore = self.ret.search("lore", f"{brief} {q}", genre, k=k)
            beat_txt = f"{b['beat']}: {b.get('title','')} - {b.get('summary','')}"
            if b["beat"] == "hook":
                beat_txt += f"\nOpen with this chosen hook, polished and extended:\n{hook}"
            for c in style:  # style examples are sources too (listed in output, not cited in prose)
                style_used.add(c["meta"]["source"])
                refs.append(c["text"])  # so the >80% overlap check also covers style examples
            tail = ""
            if sections:
                tail = ("\n\nEnd of the text written so far. Do NOT repeat its sentences, refrains or images; "
                        "advance the story with new events:\n" + " ".join(sections[-1][1].split()[-120:]))
            prompt = self._tpl(
                "section", genre_rules=g.rules, narrator=narrator, outline=outline_txt, beat=beat_txt,
                beat_words=b["words"], summary=summary + tail,
                style_examples="\n\n".join(f"- {c['text']}" for c in style) or "(none)",
                lore=cite(lore) or "(none)", banned="\n".join(self.banned))
            earlier = "\n\n".join(t for _, t in sections)
            text = self._write_section(prompt, b, earlier, g.grounded)
            # drop re-told paragraphs/sentences now so the rolling budget compensates for the lost words
            text = checks.dedupe(text, earlier) or text
            sections.append((b, text))
            if i < len(beats) - 1:  # the last summary would never be read
                summary = self.llm.chat(self._tpl("summary", previous_summary=summary, new_text=text),
                                        temperature=0.2, max_tokens=300)

        stage("Polishing…")
        # per section: a whole-script polish call made the 14B model loop and repeat paragraphs
        rn = self.cfg["checks"]["repeated_ngram"]
        def polish(bt):
            b, t = bt
            ref = min(checks.spoken_count(t), b["words"])  # trim an oversize draft to its budget, never pad a short one
            p = self.llm.chat(self._tpl(
                "polish", text=t, narrator=narrator, genre_rules=g.rules,
                banned="\n".join(self.banned), target_words=ref),
                temperature=0.5, max_tokens=int(checks.spoken_count(t) * 2.2))
            ok = (0.7 * ref <= checks.spoken_count(p) <= 1.35 * ref
                  and checks.repeated_rate(p, rn) <= checks.repeated_rate(t, rn) + 0.05)
            return (p if ok else t), ok

        # sections are independent here, so they can share the GPU; this only speeds up when the Ollama
        # server runs with OLLAMA_NUM_PARALLEL >= llm.parallel (otherwise the requests just queue)
        with ThreadPoolExecutor(self.cfg["llm"]["parallel"]) as pool:
            results = list(pool.map(polish, sections))
        for (b, _), (_, ok) in zip(sections, results):
            if not ok:
                notes.append(f"Polish of '{b['beat']}' rejected (length/repetition); draft kept.")
        parts, earlier = [], ""
        for p, _ in results:  # dedupe section by section so the structure survives for the length fit below
            d = checks.dedupe(p, earlier) or p
            if d != p:
                notes.append("Removed paragraphs that repeated earlier ones.")
            parts.append(d)
            earlier += "\n\n" + d
        stage("Fitting length…")
        parts = self._fit_length(parts, [b["words"] for b, _ in sections], total, narrator, g, notes)
        script = checks.dedupe("\n\n".join(parts))  # final safety net: nothing repeated may reach the output
        if not g.grounded:  # [S#] tags only help fact-checking; ungrounded stories get them leaked by the model
            script = re.sub(r"[ \t]*\[S\d+\]", "", script)
        if g.grounded and cited_text:
            # Verify step: ask the model about only the sentences the code flagged, with the lore as sole evidence
            flagged = checks.uncited_facts(script) + [s for s, _ in checks.unsupported_cites(script, cited_text)]

            def supported(s):
                sw = set(checks.words(s))  # the 12 chunks sharing most words with the sentence, so its source is in
                near = sorted(cited_text.values(), key=lambda t: -len(sw & set(checks.words(t))))[:12]
                ans = self.llm.chat(self._tpl("verify", lore="\n\n".join(near), sentence=s), temperature=0.0, max_tokens=5)
                return not ans.strip().upper().startswith("NO")  # an unclear answer leaves the sentence alone

            stage("Verifying…")
            script, marked = checks.mark_invented(script, flagged, supported)
            if marked:
                notes.append(f"Verify step marked {marked} unsupported sentence(s) as INVENTED.")

        stage("Packaging…")
        pkg = self._json(self._tpl("package", genre=genre, brief=brief, outline=outline_txt,
                                   excerpt=" ".join(script.split()[:500])), (0.7, 0.7), lambda d: d["titles"],
                        max_tokens=800) or {}  # a list or text answer raises here and is retried, not crashed on in render
        words_total = checks.spoken_count(script)
        stats = {
            "target_words": total, "words": words_total,
            "word_accuracy": round(1 - abs(words_total - total) / total, 3),
            "repeated_phrase_rate": round(checks.repeated_rate(script, rn), 4),
            "avg_sentence_words": round(checks.avg_sentence_words(script), 1),
            "overlap_flags": checks.overlap_flags(script, refs, self.cfg["checks"]["ngram"],
                                                  self.cfg["checks"]["overlap_threshold"]),
            "banned_hits": checks.banned_hits(script, self.banned),
            "invented": checks.invented(script), "notes": notes,
            "unsupported_cites": checks.unsupported_cites(script, cited_text) if g.grounded else [],
            "uncited_facts": checks.uncited_facts(script) if g.grounded else [],
        }
        # chapter times: each beat's share of the polished length, at wpm
        weights = [checks.spoken_count(t) or 1 for _, t in sections]
        scale = words_total / sum(weights)
        chapters, acc = [], 0
        for (b, _), w in zip(sections, weights):
            chapters.append((int(acc * scale / wpm * 60), b.get("title") or b["beat"]))
            acc += w
        sources_used = {}
        for cid, n in sources.items():
            sources_used.setdefault(cid.split(":", 2)[1], []).append(n)  # id = "<collection>:<file>:<chunk no.>"
        for s in sorted(style_used):
            sources_used.setdefault(s, [])
        md = render.render(brief, genre, minutes, narrator, pkg, script, chapters, sources_used, stats)
        out = config.path(self.cfg, "output_dir")
        out.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", brief.lower())[:50].strip("-") or "script"
        path = out / f"{genre}-{minutes}min-{slug}.md"
        path.write_text(md, encoding="utf-8")
        vram.log_stage("done")
        return {"path": path, "markdown": md, "stats": stats}
