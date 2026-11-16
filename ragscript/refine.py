"""Everything that happens to a drafted section after the model writes it: judging it (`draft_issues`, `final_issues`),
cleaning model output (`clean_reply`), the one rule for accepting a rewrite (`attempt`), and the three passes that improve
the whole script: polish each section, rewrite the worst ones, fit the total length. `Refiner.run` is the one interface the
pipeline uses."""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import checks, grounding


@dataclass(frozen=True)
class Issue:
    kind: str          # what was found: "length", "repeats", "pause", "tense", "images", ...
    message: str       # prose for the model, shown in a corrective retry or a rewrite prompt
    detail: str = ""   # short specifics for the Checks notes, e.g. "frantic x3"


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


def clean_reply(text):
    """A model reply as text that can enter the script: no invented notes, no cut-off last sentence."""
    return clean_draft(whole_sentences(text))


def attempt(old, produce, accept):
    """Ask for a replacement of `old` and keep it only if `accept(new)`. Returns (text, kept). The one place that
    cleans a rewrite and decides to keep or revert it, so no pass can make the script worse."""
    new = clean_reply(produce())
    return (new, True) if accept(new) else (old, False)


def _repeat_issues(text, earlier, ck):
    out = []
    if checks.repeated_rate(text, ck["repeated_ngram"]) > ck["section_repeat"]:
        out.append(Issue("repeats", "it repeated its own paragraphs or refrains; use each recurring detail at most 3 times in the story, changed each time"))
    if earlier and checks.overlap_flags(text, [earlier], 4, ck["earlier_overlap"]):
        out.append(Issue("retold", "it repeated earlier paragraphs; write only NEW events that move the story forward"))
    if earlier and checks.drop_similar(text, earlier, ck["dedupe_similar"], ck["dedupe_similar_min"]).split() != text.split():  # words, not layout
        out.append(Issue("similar", "it retells earlier paragraphs in new words; write only NEW events that move the story forward"))
    if checks.reused_quotes(text, earlier):
        out.append(Issue("quote", "it repeated a quotation already used earlier; quote each source only once"))
    return out


def _banned_issues(text, banned):
    hits = checks.banned_hits(text, banned)
    return [Issue("banned", "it used banned phrases: " + "; ".join(hits), ", ".join(hits))] if hits else []


def draft_issues(text, earlier, cfg, banned, budget, grounded=False):
    """Reasons a fresh draft must be regenerated (empty list = accept). Models overshoot small budgets by 1.5-5x every
    time and the polish pass trims that, so length only counts when the draft is far too long or too short."""
    ln, out = cfg["length"], []
    n = checks.spoken_count(text)
    if not ln["draft_min"] * budget <= n <= ln["draft_max"] * budget:
        out.append(Issue("length", f"it was {n} words but must be about {budget}", f"{n} words, budget {budget}"))
    out += _repeat_issues(text, earlier, cfg["checks"])
    if grounded and (u := grounding.uncited_facts(text)):
        out.append(Issue("uncited", "these sentences state names, dates or numbers with neither a [S#] citation nor an "
                                    "{{INVENTED: ...}} mark (cite them, mark them, or cut them): " + " | ".join(u[:3])))
    return out + _banned_issues(text, banned)


def final_issues(text, earlier, cfg, banned):
    """Reasons a finished section is worth rewriting (empty list = fine): the draft checks on repeats and banned
    phrases, plus cue and tense problems."""
    ck, out = cfg["checks"], _repeat_issues(text, earlier, cfg["checks"]) + _banned_issues(text, banned)
    limit = cfg["style"]["max_pause_per_section"]
    if (pauses := text.count("[PAUSE]")) > limit:
        out.append(Issue("pause", f"it has {pauses} [PAUSE] cues; keep {limit} at most, right before a reveal", f"{pauses} cues"))
    if checks.mixed_tense(text, ck):
        out.append(Issue("tense", "it switches between past and present tense; rewrite everything in the past tense"))
    return out


def _note(issues):
    return "; ".join(i.kind + (f" ({i.detail})" if i.detail else "") for i in issues)


class Refiner:
    """ask(stage, template_vars, **chat_options) -> the model's reply; the pipeline supplies it."""

    def __init__(self, ask, cfg, banned):
        self.ask, self.cfg, self.banned = ask, cfg, banned

    def run(self, sections, story, stage=lambda msg: None):
        """sections: [(beat, drafted text)]. Returns the finished section texts: polished, the worst rewritten, and
        the total length fitted to story.total words."""
        stage("Polishing…")
        parts = self.polish(sections, story)
        stage("Revising weakest sections…")
        parts = self.revise_weakest(parts, story, self.critique(parts, story))
        parts = self.dedupe_parts(parts, story)  # before the fit, so the fit sees the real length and makes it up
        stage("Fitting length…")
        return self.fit_length(parts, [b["words"] for b, _ in sections], story)

    def polish(self, sections, story):
        # per section: a whole-script polish call made the model loop and repeat paragraphs
        ck, ln = self.cfg["checks"], self.cfg["length"]

        def polish_one(bt):
            b, t = bt
            ref = min(checks.spoken_count(t), b["words"])  # trim an oversize draft to its budget, never pad a short one
            return attempt(
                t,
                lambda: self.ask("polish", dict(text=t, narrator=story.narrator, genre_rules=story.genre.rules,
                                                target_words=ref),
                                 temperature=0.5, max_tokens=int(checks.spoken_count(t) * 2.2)),
                lambda p: (ln["polish_min"] * ref <= checks.spoken_count(p) <= ln["polish_max"] * ref
                           and checks.repeated_rate(p, ck["repeated_ngram"]) <=
                           checks.repeated_rate(t, ck["repeated_ngram"]) + ck["polish_repeat_slack"]))

        # sections are independent here, so they can share the GPU; this only speeds up when the Ollama
        # server runs with OLLAMA_NUM_PARALLEL >= llm.parallel (otherwise the requests just queue)
        with ThreadPoolExecutor(self.cfg["llm"]["parallel"]) as pool:
            results = list(pool.map(polish_one, sections))
        for (b, _), (_, kept) in zip(sections, results):
            if not kept:
                story.notes.append(f"Polish of '{b['beat']}' rejected (length/repetition); draft kept.")
        return [p for p, _ in results]

    def dedupe_parts(self, parts, story):
        """Run the repeat guard over the finished sections in order, each against the text before it. A section that
        only retells earlier text becomes empty (the length fit then makes the loss up); every removal is noted."""
        ck, out, earlier = self.cfg["checks"], [], ""
        for p in parts:
            d = checks.dedupe(p, ck, earlier)
            if (removed := checks.spoken_count(p) - checks.spoken_count(d)) > 0:  # re-flowed lines alone are no removal
                story.notes.append(f"Repeat guard removed {removed} words that retold earlier text.")
            out.append(d)
            earlier += "\n\n" + d
        return out

    def critique(self, parts, story):
        """One model call that reads the whole script and names logic holes: {section index: [Issue("logic", ...)]}.
        Empty when the model finds none or answers off format."""
        numbered = "\n\n".join(f"[{i + 1}]\n{p}" for i, p in enumerate(parts))
        reply = self.ask("critic", dict(narrator=story.narrator, outline=story.outline, text=numbered),
                         temperature=0.2, max_tokens=300)
        out = {}
        for m in re.finditer(r"(?m)^\s*\[?(\d+)\]?\s*[:.)-]\s*(\S.*)$", reply):
            i = int(m.group(1)) - 1
            if 0 <= i < len(parts) and len(out.get(i, [])) < 2:
                out.setdefault(i, []).append(Issue("logic", m.group(2).strip()))
        return out

    def revise_weakest(self, parts, story, logic=None):
        """Rewrite the sections with the most issues, worst first, at most `checks.revise_sections` of them. Each is
        judged against the text before it as it is now; a rewrite is kept only if it has fewer issues and stays near
        its length."""
        ln, tried = self.cfg["length"], set()
        for _ in range(self.cfg["checks"]["revise_sections"]):
            earlier, issues = "", []
            for p in parts:
                issues.append(final_issues(p, earlier, self.cfg, self.banned) + (logic or {}).get(len(issues), []))
                earlier += "\n\n" + p
            i = max((j for j in range(len(parts)) if j not in tried), key=lambda j: len(issues[j]), default=None)
            if i is None or not issues[i]:
                break
            tried.add(i)
            before, earlier = checks.spoken_count(parts[i]), "\n\n".join(parts[:i])
            new, kept = attempt(
                parts[i],
                lambda: self.ask("revise", dict(text=parts[i], narrator=story.narrator, genre_rules=story.genre.rules,
                                                problems="\n".join("- " + x.message for x in issues[i]), target_words=before),
                                 temperature=0.5, max_tokens=int(before * 2.2)),
                lambda n: (ln["polish_min"] * before <= checks.spoken_count(n) <= ln["polish_max"] * before
                           and len(final_issues(n, earlier, self.cfg, self.banned)) < len(issues[i])))
            if kept:
                story.notes.append(f"Revised section {i + 1} to fix: {_note(issues[i])}.")
                parts = parts[:i] + [new] + parts[i + 1:]
            else:
                story.notes.append(f"Revision of section {i + 1} rejected; polished text kept.")
        return parts

    def fit_length(self, parts, budgets, story, max_calls=7):
        """Final length control: while the script is more than fit_tolerance off target, rewrite the untried section
        furthest over (or under) its budget. A rewrite is kept only if it moves the total toward the target AND
        leaves the script free of repeats, so this step can never undo the anti-repetition pass."""
        total, ck = story.total, self.cfg["checks"]
        words = [checks.spoken_count(p) for p in parts]
        tol, tried = self.cfg["length"]["fit_tolerance"], set()
        for _ in range(max_calls):
            n = sum(words)
            if (1 - tol) * total <= n <= (1 + tol) * total:
                break
            sign = 1 if n > total else -1
            i = max((j for j in range(len(parts)) if j not in tried and words[j]), default=None,
                    key=lambda j: sign * (words[j] - budgets[j]))
            if i is None:
                break
            tried.add(i)
            target = round(min(1.8, max(0.4, 1 + (total - n) / max(words[i], 1))) * words[i])

            def accept(new):
                joined = "\n\n".join(parts[:i] + [new] + parts[i + 1:])
                # compare words, not layout: dedupe joins a line break after a full stop into a space
                return (abs(n - words[i] + checks.spoken_count(new) - total) < abs(n - total)
                        and checks.dedupe(joined, ck).split() == joined.split())

            new, kept = attempt(
                parts[i],
                lambda: self.ask("fit", dict(text=parts[i], narrator=story.narrator, genre_rules=story.genre.rules,
                                             current_words=words[i], target_words=target),
                                 temperature=0.5, max_tokens=int(max(words[i], target) * 2.2)),
                accept)
            if kept:
                parts, words[i] = parts[:i] + [new] + parts[i + 1:], checks.spoken_count(new)
                story.notes.append(f"Length fit rewrote section {i + 1}: script is now {sum(words)} words (target {total}).")
        return parts
