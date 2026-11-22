"""Everything that happens to a drafted section after the model writes it: judging it (`draft_issues`, `final_issues`),
cleaning model output (`clean_reply`), the one rule for accepting a rewrite (`attempt`), and the three passes that improve
the whole script: polish each section, rewrite the worst ones, fit the total length. `Refiner.run` is the one interface the
pipeline uses."""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import checks, grounding, logic


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


# a note right after "the", "with a", "while" ...: it stood in for a noun, and stripping it leaves a hole
_NOTE_IN_SENTENCE = re.compile(r"(?i)\b(?:the|an?|my|his|her|its|their|our|with|while|that|of|and|to|as)[ \t]+"
                               r"\[(?!(?:PAUSE|SFX:|B-ROLL:|S\d+\]))[^\]\n]*\][ \t]*[A-Za-z.,;]")


def attempt(old, produce, accept):
    """Ask for a replacement of `old` and keep it only if `accept(new)`. Returns (text, kept). The one place that
    cleans a rewrite and decides to keep or revert it, so no pass can make the script worse."""
    raw = produce()
    if _NOTE_IN_SENTENCE.search(raw):  # stripping a note from mid-sentence leaves a hole ("while the filled my lungs")
        return old, False
    new = clean_reply(raw)
    return (new, True) if accept(new) else (old, False)


def _repeat_issues(text, earlier, ck):
    out = []
    if checks.repeated_rate(text, ck["repeated_ngram"]) > ck["section_repeat"]:
        out.append(Issue("repeats", "it repeated its own paragraphs or refrains; use each recurring detail at most 4 times "
                                     "in the story, changed each time"))
    if earlier and checks.overlap_flags(text, [earlier], 4, ck["earlier_overlap"]):
        out.append(Issue("retold", "it repeated earlier paragraphs; write only NEW events that move the story forward"))
    # compared by words, not layout
    kept = checks.drop_similar(text, earlier, ck["dedupe_similar"], ck["dedupe_similar_min"]) if earlier else text
    if kept.split() != text.split():
        out.append(Issue("similar", "it retells earlier paragraphs in new words; "
                                    "write only NEW events that move the story forward"))
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
        out.append(Issue("pause", f"it has {pauses} [PAUSE] cues; keep {limit} at most, right before a reveal",
                         f"{pauses} cues"))
    if checks.mixed_tense(text, ck):
        out.append(Issue("tense", "it switches between past and present tense; rewrite everything in the past tense"))
    return out


def _note(issues):
    return "; ".join(i.kind + (f" ({i.detail})" if i.detail else "") for i in issues)


AREAS = ("Premise", "Hook", "Voice", "Prose", "Logic", "Ending")  # the criteria of prompts/stages/compare.md
CRITERIA = AREAS + ("Overall",)
FLIP = {"A": "B", "B": "A", "tie": "tie"}


def parse_compare(reply):
    """{criterion: "A" | "B" | "tie"} from the judge's reply; a missing or malformed line is left out."""
    found = {c: re.search(rf"(?mi)^\s*{c}\s*:\s*(A|B|tie)\s*(?:\||$)", reply) for c in CRITERIA}
    return {c: (m.group(1).upper() if len(m.group(1)) == 1 else "tie") for c, m in found.items() if m}


def combine(first, swapped):
    """Winner per criterion over the two orderings. `swapped` was asked with the scripts exchanged, so its A is the
    original B. When the orderings disagree the judge followed position, not quality: that criterion is a tie."""
    return {c: first[c] if first[c] == FLIP[swapped[c]] else "tie" for c in first if c in swapped}


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
        stage("Choosing the ending…")
        parts = self.best_ending(parts, story)
        parts = self.dedupe_parts(parts, story)  # before the fit, so the fit sees the real length and makes it up
        parts = self.fix_overuse(parts, story)  # the ending and the dedupe can bring a detail back after the revision
        stage("Fitting length…")
        return self.fit_length(parts, [b.words for b, _ in sections], story)

    def polish(self, sections, story):
        # per section: a whole-script polish call made the model loop and repeat paragraphs
        ck, ln = self.cfg["checks"], self.cfg["length"]

        def polish_one(bt):
            b, t = bt
            ref = min(checks.spoken_count(t), b.words)  # trim an oversize draft to its budget, never pad a short one
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
                story.notes.append(f"Polish of '{b.beat}' rejected (length/repetition); draft kept.")
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
        text = "\n\n".join(parts)
        hints = ", ".join(logic.unused_objects(text)) or "(none)"
        ending = ", ".join(logic.new_in_ending(text)) or "(none)"
        state = self.ask("tracker", dict(text=numbered), temperature=0.0, max_tokens=500)  # facts first, then judge
        reply = self.ask("critic", dict(narrator=story.narrator, outline=story.outline, text=numbered, hints=hints,
                                        ending_hints=ending, state=state.strip() or "(none)"),
                         temperature=0.2, max_tokens=300)
        out = {}
        # "2: ...", "2) ...", "- 2. ..." and "Section 2: ..." all name section 2
        for m in re.finditer(r"(?mi)^\s*(?:[-*]\s*)?(?:section\s*)?\[?(\d+)\]?\s*[:.)-]\s*(\S.*)$", reply):
            i = int(m.group(1)) - 1
            if 0 <= i < len(parts) and len(out.get(i, [])) < 2:
                out.setdefault(i, []).append(Issue("logic", m.group(2).strip()))
        for noun, i in logic.dropped_then_used(parts):  # a code check, so the critic cannot miss it
            if not any(noun in x.message for x in out.get(i, [])):
                out.setdefault(i, []).append(Issue("logic", f"the narrator dropped the {noun} earlier but uses it here"))
        last = max((j for j, p in enumerate(parts) if p.strip()), default=0)
        if logic.left_first_person("\n\n".join(parts)):
            out.setdefault(last, []).append(Issue("logic", "the ending leaves the narrator's first person: "
                                                           "write it as I/my, from what the narrator sees"))
        limit = self.cfg["checks"]["motif_max"]
        for name, i in logic.detail_overuse(parts, story.details, limit):  # a detail past its limit: change it here
            out.setdefault(i, []).append(Issue("logic", f'the recurring detail "{name}" already came back {limit} times '
                                                        "in earlier paragraphs; remove it here or change it completely"))
        return out

    def compare(self, a, b, brief, genre):
        """(winner per criterion, how many criteria the two orders disagreed on) for two scripts of one brief. The
        judge is asked twice with the scripts exchanged, and a disagreement is a tie (it followed position)."""
        ask = lambda x, y: parse_compare(self.ask(
            "compare", dict(genre=genre, brief=brief, script_a=x, script_b=y), temperature=0.1, max_tokens=900))
        first, swapped = ask(a, b), ask(b, a)
        result = combine(first, swapped)
        return result, sum(first[c] != FLIP[swapped[c]] for c in result)

    def judge(self, a, b, story):
        """Which of two whole scripts is better: "A", "B" or "tie". The Ending line decides, else Overall."""
        result, _ = self.compare(a, b, story.brief, story.genre.slug)
        return result.get("Ending", "tie") if result.get("Ending", "tie") != "tie" else result.get("Overall", "tie")

    def best_ending(self, parts, story):
        """The ending is the most criticised part and the cheapest to redo: write `checks.ending_candidates - 1` other
        versions of the last section and keep the one the judge prefers (a challenger must win; a tie keeps the
        current text). A version is only judged when it has the right length and no more problems than the current one."""
        n, ln = self.cfg["checks"]["ending_candidates"], self.cfg["length"]
        last = max((j for j, p in enumerate(parts) if p.strip()), default=None)
        if n < 2 or last is None:
            return parts
        before, best = "\n\n".join(parts[:last]), parts[last]
        words, won = checks.spoken_count(best), 0
        for _ in range(n - 1):
            new, kept = attempt(
                best,
                lambda: self.ask("ending", dict(narrator=story.narrator, outline=story.outline,
                                                payoff=story.payoff or "(none)", ending=best, target_words=words,
                                                before=" ".join(before.split()[-250:])),
                                 temperature=0.9, max_tokens=int(words * 2.2)),
                lambda t: (ln["polish_min"] * words <= checks.spoken_count(t) <= ln["polish_max"] * words
                           and len(final_issues(t, before, self.cfg, self.banned)) <=
                           len(final_issues(best, before, self.cfg, self.banned))
                           and len(logic.new_in_ending(f"{before}\n\n{t}")) <=
                           len(logic.new_in_ending(f"{before}\n\n{best}"))))  # no new object in the last image
            if kept and new != best and self.judge(f"{before}\n\n{best}", f"{before}\n\n{new}", story) == "B":
                best, won = new, won + 1
        if won:
            story.notes.append(f"Ending: a rewritten version won the judge's comparison ({won} of {n - 1} challengers).")
        return parts[:last] + [best] + parts[last + 1:]

    def revise_weakest(self, parts, story, logic=None):
        """Rewrite the sections with the most issues, worst first, at most `checks.revise_sections` of them. Each is
        judged against the text before it as it is now; a rewrite is kept only if it has fewer issues and stays near
        its length."""
        ln, tried = self.cfg["length"], set()
        overused = sum(any("recurring detail" in x.message for x in v) for v in (logic or {}).values())
        for _ in range(max(self.cfg["checks"]["revise_sections"], overused)):  # every over-used detail gets its fix
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
                                                target_words=before,
                                                problems="\n".join("- " + x.message for x in issues[i])),
                                 temperature=0.5, max_tokens=int(before * 2.2)),
                lambda n: (ln["polish_min"] * before <= checks.spoken_count(n) <= ln["polish_max"] * before
                           and len(final_issues(n, earlier, self.cfg, self.banned)) < len(issues[i])))
            holes = sum(x.kind == "logic" for x in issues[i])  # not `logic`: the critic's issues, used next round
            left = len(self.critique(parts[:i] + [new] + parts[i + 1:], story).get(i, [])) if kept and holes else 0
            if kept and left > holes:  # the critic is a noisy small model: only a rewrite that adds holes is thrown out
                story.notes.append(f"Revision of section {i + 1} rejected; "
                                   "a second critic call found more logic holes there than before.")
            elif kept:
                story.notes.append(f"Revised section {i + 1} to fix: {_note(issues[i])}."
                                   + (f" (logic holes: {holes} before, {left} after)" if holes else ""))
                parts = parts[:i] + [new] + parts[i + 1:]
            else:
                story.notes.append(f"Revision of section {i + 1} rejected; polished text kept.")
        return parts

    def worn_words(self, parts, story):
        """[(problem for the rewrite, section index)] for the wording a reader hears as a tic: a crutch word ("heavy")
        and a phrase that returns in the same words. The brief's own words and the outline's details are the setting
        and are never listed. The section picked is the one using it most (the last repeat of a phrase)."""
        ck, text = self.cfg["checks"], "\n\n".join(parts)
        skip = checks.topic_words(story.brief) | checks.setting_words(*story.details)
        out = []
        for w, _ in checks.overused_words(text, ck, skip=skip):
            use = [sum(x == w for x in checks.words(grounding.unmark(p))) for p in parts]
            out.append((f'the word "{w}" is used all over the script; use it here at most once, '
                        "with a more specific word for the rest", use.index(max(use))))
        for phrase, n in checks.repeated_wording(text):
            if n > ck["wording_max"]:
                held = [i for i, p in enumerate(parts) if phrase in " ".join(checks.words(grounding.unmark(p)))]
                if held:
                    out.append((f'the phrase "{phrase}" returns in the same words; say it a new way here', held[-1]))
        return out

    def fix_overuse(self, parts, story, max_calls=4):
        """After the ending is chosen, rewrite a section that still holds a detail past `checks.motif_max`, a crutch
        word or a phrase that keeps returning in the same words. The last section is left alone while another can be
        changed (the ending reuses the planted detail on purpose). A rewrite is kept only if it leaves fewer such
        problems and stays near its length."""
        limit, ln = self.cfg["checks"]["motif_max"], self.cfg["length"]
        last = max((j for j, p in enumerate(parts) if p.strip()), default=0)
        count = lambda ps: len(logic.detail_overuse(ps, story.details, limit)) + len(self.worn_words(ps, story))
        for _ in range(max_calls):
            over = logic.detail_overuse(parts, story.details, limit)
            pick = [(p, i) for p, i in self.worn_words(parts, story) if i != last]
            pick += [(f'the recurring detail "{n}" already came back {limit} times; remove it here or change it '
                      "completely (a new image, not the same words)", i) for n, i in over if i != last]
            if not pick and over:  # only the ending is past the limit: thin out the detail in the section before it
                pick = [(f'the recurring detail "{n}" came back too often; change it completely here', i)
                        for n, i in logic.detail_overuse(parts, story.details, 0) if i != last and n == over[0][0]]
            if not pick:
                break
            problem, i = pick[-1]
            before, earlier, worn = checks.spoken_count(parts[i]), "\n\n".join(parts[:i]), count(parts)
            new, kept = attempt(
                parts[i],
                lambda: self.ask("revise", dict(text=parts[i], narrator=story.narrator, genre_rules=story.genre.rules,
                                                target_words=before, problems="- " + problem),
                                 temperature=0.5, max_tokens=int(before * 2.2)),
                lambda n: (ln["polish_min"] * before <= checks.spoken_count(n) <= ln["polish_max"] * before
                           and len(final_issues(n, earlier, self.cfg, self.banned)) <=
                           len(final_issues(parts[i], earlier, self.cfg, self.banned))
                           and count(parts[:i] + [n] + parts[i + 1:]) < worn))
            if not kept:
                break
            parts = parts[:i] + [new] + parts[i + 1:]
            story.notes.append(f"Rewrote section {i + 1} to cut back repeated wording: {problem.split(';')[0]}.")
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
