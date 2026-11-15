"""Everything that happens to a drafted section after it is written: the checks that decide whether it is acceptable
(`section_issues`), cleaning of model output, and the three passes that improve the whole script: polish each
section, rewrite the weakest one, fit the total length. `Refiner.run` is the one interface the pipeline uses."""
import re
from concurrent.futures import ThreadPoolExecutor

from . import checks, grounding


def clean_draft(text):
    """Remove bracketed notes the model invented (e.g. "[recurring]"); only real cues and [S#] tags stay.
    A {{...}} the model wrote without the INVENTED: label gets the label."""
    text = grounding.relabel(text)
    return re.sub(r"[ \t]*\[(?!(?:PAUSE|SFX:|B-ROLL:|S\d+\]))[^\]]*\]", "", text, flags=re.I)


def whole_sentences(text):
    """Drop a trailing half-sentence left by the token cap (kept as is when it would lose over half the text)."""
    text = text.rstrip()
    if re.search(r"[.!?…\"”’')\]}]$", text):
        return text
    m = re.search(r"^(.*[.!?…][\"”’')\]}]*)", text, re.S)  # greedy: up to the last sentence end
    return m.group(1) if m and len(m.group(1)) > len(text) / 2 else text


def section_issues(text, earlier, cfg, banned, *, budget=None, grounded=False, style=False):
    """Reasons a section must be rewritten (empty list = accept). The one place section quality is judged.
    budget: word budget for a draft (None skips the length check); grounded: also flag uncited facts;
    style: also flag cue and tense problems (judged on finished sections, not on drafts)."""
    ck, ln, p = cfg["checks"], cfg["length"], []
    n = checks.spoken_count(text)
    # models overshoot small budgets by 1.5-5x every time; the polish pass trims that, so a rewrite is
    # only worth a call when the draft is far too long or too short
    if budget and not ln["draft_min"] * budget <= n <= ln["draft_max"] * budget:
        p.append(f"it was {n} words but must be about {budget}")
    if checks.repeated_rate(text, ck["repeated_ngram"]) > ck["section_repeat"]:
        p.append("it repeated its own paragraphs or refrains; mention each recurring detail at most once")
    if earlier and checks.overlap_flags(text, [earlier], 4, ck["earlier_overlap"]):
        p.append("it repeated earlier paragraphs; write only NEW events that move the story forward")
    if checks.reused_quotes(text, earlier):
        p.append("it repeated a quotation already used earlier; quote each source only once")
    if grounded and (u := grounding.uncited_facts(text)):
        p.append("these sentences state names, dates or numbers with neither a [S#] citation nor an "
                 "{{INVENTED: ...}} mark (cite them, mark them, or cut them): " + " | ".join(u[:3]))
    if hits := checks.banned_hits(text, banned):
        p.append("it used banned phrases: " + "; ".join(hits))
    if style:
        limit = cfg["style"]["max_pause_per_section"]
        if (pauses := text.count("[PAUSE]")) > limit:
            p.append(f"it has {pauses} [PAUSE] cues; keep {limit} at most, right before a reveal")
        if checks.mixed_tense(text):
            p.append("it switches between past and present tense; rewrite everything in the past tense")
        if lines := checks.repeated_lines(text, earlier):
            p.append("it reuses lines already written (" + " | ".join(lines[:3]) + "); write fresh lines instead")
        if images := checks.repeated_phrases(text, earlier, ck["phrase_repeat"]):
            p.append("it brings back images already used earlier in the script (" + ", ".join(
                f"{ph} x{n}" for ph, n in images[:3]) + "); replace them with fresh details")
        if crowded := checks.crowded_words(text, ck["word_repeat"]):
            p.append("it leans on the words " + ", ".join(f"{w} (x{n})" for w, n in crowded[:3]) + "; vary the wording")
    return p


class Refiner:
    """ask(stage, template_vars, **chat_options) -> the model's reply; the pipeline supplies it."""

    def __init__(self, ask, cfg, banned):
        self.ask, self.cfg, self.banned = ask, cfg, banned

    def run(self, sections, total, narrator, g, notes, stage=lambda msg: None):
        """sections: [(beat, drafted text)]. Returns the finished section texts: polished, the weakest rewritten,
        and the total length fitted to `total` words."""
        stage("Polishing…")
        parts = self.polish(sections, narrator, g, notes)
        stage("Revising weakest section…")
        parts = self.revise_weakest(parts, narrator, g, notes)
        stage("Fitting length…")
        return self.fit_length(parts, [b["words"] for b, _ in sections], total, narrator, g, notes)

    def polish(self, sections, narrator, g, notes):
        # per section: a whole-script polish call made the model loop and repeat paragraphs
        rn, ln = self.cfg["checks"]["repeated_ngram"], self.cfg["length"]

        def polish_one(bt):
            b, t = bt
            ref = min(checks.spoken_count(t), b["words"])  # trim an oversize draft to its budget, never pad a short one
            p = clean_draft(whole_sentences(self.ask(
                "polish", dict(text=t, narrator=narrator, genre_rules=g.rules, target_words=ref),
                temperature=0.5, max_tokens=int(checks.spoken_count(t) * 2.2))))
            ok = (ln["polish_min"] * ref <= checks.spoken_count(p) <= ln["polish_max"] * ref
                  and checks.repeated_rate(p, rn) <= checks.repeated_rate(t, rn) + 0.05)
            return (p if ok else t), ok

        # sections are independent here, so they can share the GPU; this only speeds up when the Ollama
        # server runs with OLLAMA_NUM_PARALLEL >= llm.parallel (otherwise the requests just queue)
        with ThreadPoolExecutor(self.cfg["llm"]["parallel"]) as pool:
            results = list(pool.map(polish_one, sections))
        for (b, _), (_, ok) in zip(sections, results):
            if not ok:
                notes.append(f"Polish of '{b['beat']}' rejected (length/repetition); draft kept.")
        parts, earlier = [], ""
        for p, _ in results:  # dedupe section by section so the structure survives for the length fit
            d = checks.dedupe(p, earlier) or p
            if d != p:
                notes.append("Removed paragraphs that repeated earlier ones.")
            parts.append(d)
            earlier += "\n\n" + d
        return parts

    def revise_weakest(self, parts, narrator, g, notes):
        """Rewrite the sections with the most style issues, worst first, at most `checks.revise_sections` of them.
        A rewrite is kept only if it has fewer issues and stays near its length, so this step can only improve the
        script; each section is judged against the text before it as it is now."""
        ln, tried = self.cfg["length"], set()
        for _ in range(self.cfg["checks"]["revise_sections"]):
            earlier, issues = "", []
            for p in parts:
                issues.append(section_issues(p, earlier, self.cfg, self.banned, style=True))
                earlier += "\n\n" + p
            i = max((j for j in range(len(parts)) if j not in tried), key=lambda j: len(issues[j]), default=None)
            if i is None or not issues[i]:
                break
            tried.add(i)
            before = checks.spoken_count(parts[i])
            new = self.ask("revise", dict(text=parts[i], narrator=narrator, genre_rules=g.rules,
                                          problems="\n".join("- " + x for x in issues[i]), target_words=before),
                           temperature=0.5, max_tokens=int(before * 2.2))
            new = clean_draft(whole_sentences(new))
            earlier = "\n\n".join(parts[:i])
            if (ln["polish_min"] * before <= checks.spoken_count(new) <= ln["polish_max"] * before
                    and len(section_issues(new, earlier, self.cfg, self.banned, style=True)) < len(issues[i])):
                notes.append(f"Revised section {i + 1} to fix: " + "; ".join(x.split(";")[0] for x in issues[i]) + ".")
                parts = parts[:i] + [new] + parts[i + 1:]
            else:
                notes.append(f"Revision of section {i + 1} rejected; polished text kept.")
        return parts

    def fit_length(self, parts, budgets, total, narrator, g, notes, max_calls=7):
        """Final length control: while the script is more than fit_tolerance off target, rewrite the untried section
        furthest over (or under) its budget. A rewrite is kept only if it moves the total toward the target AND
        leaves the script free of repeats, so this step can never undo the anti-repetition pass."""
        words = [checks.spoken_count(p) for p in parts]
        tol, tried = self.cfg["length"]["fit_tolerance"], set()
        for _ in range(max_calls):
            n = sum(words)
            if (1 - tol) * total <= n <= (1 + tol) * total:
                break
            sign = 1 if n > total else -1
            i = max((j for j in range(len(parts)) if j not in tried), default=None,
                    key=lambda j: sign * (words[j] - budgets[j]))
            if i is None:
                break
            tried.add(i)
            target = round(min(1.8, max(0.4, 1 + (total - n) / max(words[i], 1))) * words[i])
            new = clean_draft(whole_sentences(self.ask(
                "fit", dict(text=parts[i], narrator=narrator, genre_rules=g.rules, current_words=words[i],
                            target_words=target), temperature=0.5, max_tokens=int(max(words[i], target) * 2.2))))
            trial = parts[:i] + [new] + parts[i + 1:]
            joined, w = "\n\n".join(trial), checks.spoken_count(new)
            # compare words, not layout: dedupe joins a line break after a full stop into a space
            if abs(n - words[i] + w - total) < abs(n - total) and checks.dedupe(joined).split() == joined.split():
                parts, words[i] = trial, w
                notes.append(f"Length fit rewrote section {i + 1}: script is now {sum(words)} words (target {total}).")
        return parts
