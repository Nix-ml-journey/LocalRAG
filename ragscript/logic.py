"""Story-logic checks: the holes a reader notices that a word count cannot show. Outline holes (`outline_issues`), the
recurring details and how often they come back, objects that are dropped or never used again, and an ending that
brings in something new. Every check is a word match with no tagger, so each one names a likely hole for a rewrite or
a hint for the critic, never a verdict. The statistics on the narration itself (repeats, tense, crutch words) are in
checks.py."""
import re
from collections import Counter

from .checks import FUNC, content_set, stem
from .grounding import unmark
from .text import CUE, sentences, words


_ABSTRACT = re.compile(r"\b(establish\w*|sense of|growing|presence of|realization|transition|introduc\w+|clarified|"
                       r"tension|urgency)\b", re.I)  # words of a `change` that describes a mood, not an event


def outline_issues(beats, limit):
    """Logic holes in an outline (empty list = fine). Each beat should say where the narrator is (`place`), what new
    thing happens (`change`) and which recurring detail it uses (`detail`, written "name: how it changes"). A beat with
    no place or change, a change that nearly repeats the beat before, a detail used in two beats in a row, or one used
    in more than `limit` beats is a hole."""
    out, uses, prev, prev_detail = [], Counter(), None, ""
    for i, b in enumerate(beats, 1):
        tag = f"beat {i} ({b.beat})"
        for k in ("place", "change"):
            if not str(getattr(b, k)).strip():
                out.append(f'{tag} has no "{k}"')
        if _SCREEN.search(b.summary) and not b.sees.strip():
            out.append(f'{tag} shows a screen or camera but has no "sees" saying what it shows')
        ch = content_set(b.change)
        if prev and min(len(ch), len(prev)) >= 3 and len(ch & prev) / min(len(ch), len(prev)) >= 0.6:
            out.append(f"{tag} changes almost the same thing as the beat before it")
        prev = ch
        if _ABSTRACT.search(b.change):
            out.append(f'{tag} "change" names a feeling or a theme, not an event someone could film')
        name = b.detail.split(":")[0].strip().lower()
        if re.fullmatch(r"detail\s*\d*", name):
            out.append(f'{tag} calls its detail "{name}"; name it by what it is (a sound, a smell, an object)')
        if name:
            uses[name] += 1
            if name == prev_detail:
                out.append(f'{tag} uses the detail "{name}" again right after the beat before')
        prev_detail = name
    out += [f'the detail "{n}" is used in {c} beats; at most {limit}' for n, c in uses.items() if c > limit]
    return out + _ending_plan_issues(beats)


def _ending_plan_issues(beats):
    """The ending is planned: one thing is `plant`ed in an early beat, the outro's `payoff` (its final image) is that
    thing, and the outro reuses something an earlier beat already showed."""
    if len(beats) < 4:
        return []
    out = []
    payoff, plants = beats[-1].payoff.strip(), [(i, b.plant) for i, b in enumerate(beats) if b.plant.strip()]
    if not payoff:
        out.append('the outro has no "payoff": the final image, which is the planted thing shown once more')
    elif not plants:
        out.append(f'the final image "{payoff}" is planted in no beat; set "plant" in one of the first four beats')
    else:
        if late := [beats[i].beat for i, _ in plants if i > 3]:
            out.append(f'"plant" is in the {late[0]} beat; it must be in one of the first four beats')
        stems = lambda t: {stem(w) for w in content_set(t)}
        if not stems(payoff) & stems(" ".join(p for _, p in plants)):
            out.append(f'the planted thing and the final image "{payoff}" are different things')
    # the last beat must also reuse a word an earlier beat showed (words in half the beats, like a name, do not count)
    sets = [{stem(w) for w in content_set(f"{b.title} {b.summary} {b.sees} {b.change} {b.detail} {b.plant}")}
            for b in beats]
    common = {w for w, n in Counter(w for s in sets for w in s).items() if n >= len(beats) / 2}
    seen = set().union(*sets[:-2]) - common
    if not ({stem(w) for w in content_set(f"{beats[-1].summary} {beats[-1].change} {payoff}")} & seen):
        out.append("the last beat uses no object, sound or fact from the earlier beats; "
                   "its final image must come from something the story already showed")
    return out


_ARTICLES = {"the", "and", "his", "her", "its", "for", "our", "their"}


def detail_overuse(parts, names, limit):
    """[(detail, index of the part)] for each recurring detail that comes back in more than `limit` paragraphs of the
    script: every part holding one of the extra paragraphs. The detail is followed by its most specific word, the one
    found in the fewest paragraphs, as in detail_returns."""
    paras = [(i, set(words(unmark(p)))) for i, part in enumerate(parts) for p in re.split(r"\n\s*\n", part)]
    out = []
    for name in dict.fromkeys(n for n in names if n):
        keys = [w for w in words(name) if len(w) >= 3 and w not in FUNC and w not in _ARTICLES]
        counts = {w: n for w in keys if (n := sum(w in s for _, s in paras))}
        if counts:
            key = min(counts, key=counts.get)
            hits = [i for i, s in paras if key in s]  # the part of every paragraph that holds the detail, in order
            out += [(name, i) for i in dict.fromkeys(hits[limit:])]  # the paragraphs past the limit, as parts
    return out


def detail_returns(text, names):
    """How many paragraphs of the script bring back each named recurring detail: [(name, paragraphs)], most first. The
    detail is followed by its most specific word, the one found in the fewest paragraphs, so a common word in the name
    ("floor" in "floor wax") cannot inflate the count."""
    paras = [set(words(unmark(p))) for p in re.split(r"\n\s*\n", text)]
    out = {}
    for name in names:
        key = [w for w in words(name) if len(w) >= 3 and w not in FUNC and w not in _ARTICLES]
        found = [n for w in key if (n := sum(w in p for p in paras))]
        if found and name not in out:
            out[name] = min(found)
    return sorted(out.items(), key=lambda x: -x[1])


_SCREEN = re.compile(r"\b(monitors?|screens?|cameras?|footage)\b", re.I)


def fill_sees(beats):
    """A beat whose summary shows a screen or camera but has no `sees` takes it from the nearest beat that has one (the
    same screen is usually meant). Returns how many beats were filled; no model call is needed."""
    filled = 0
    for i, b in enumerate(beats):
        if _SCREEN.search(b.summary) and not b.sees.strip():
            near = sorted((j for j, o in enumerate(beats) if o.sees.strip()), key=lambda j: abs(j - i))
            if near:
                b.sees = beats[near[0]].sees
                filled += 1
    return filled


_FOUND = re.compile(r"\b(?:noticed|found|spotted|saw|grabbed|held|gripped|pulled out|picked up|spied)\s+(?:an?|the)\s+"
                    r"(?:(?:set|pair|bunch|stack) of\s+)?((?:\w+\s+){0,2}\w+)", re.I)
_PREP = {"on", "in", "at", "near", "from", "under", "beside", "behind", "with", "that", "which", "as", "and", "by", "of",
         "to", "into", "over", "while", "when"}


def unused_objects(text, top=3):
    """Objects the narrator finds or notices that are never mentioned again: a simple word count, so it is only a hint
    for the critic. Returns up to `top` head nouns."""
    w = [stem(x) for x in words(unmark(text))]
    out = []
    for m in _FOUND.finditer(unmark(text)):
        phrase = []
        for x in m.group(1).split():
            if x.lower() in _PREP:
                break
            phrase.append(x.lower())
        head = phrase[-1] if phrase else ""
        if len(head) >= 4 and head not in FUNC and w.count(stem(head)) == 1 and head not in out:
            out.append(head)
    return out[:top]


_DET = re.compile(r"\b(?:a|an|the|my|his|her|its|their|our)\s+(?:[a-z]+\s+)?([a-z]{4,})\b", re.I)


def new_in_ending(text, top=4):
    """Things named in the last paragraph that the rest of the script never mentions: the word after a determiner
    ("a pinch", "the stamp") whose stem occurs nowhere earlier; verbs and adverbs (-ed, -ing, -ly) are skipped. A word
    match with no tagger, so it is only a hint for the critic to look at the ending."""
    paras = [p for p in (CUE.sub("", q).strip() for q in re.split(r"\n\s*\n", text)) if p]  # "[PAUSE] I ran." is a line
    if len(paras) < 2:
        return []
    seen = {stem(w) for w in words(unmark(" ".join(paras[:-1])))}
    out = []
    for m in _DET.finditer(unmark(paras[-1])):
        w = m.group(1).lower()
        if w not in FUNC and not w.endswith(("ed", "ing", "ly")) and stem(w) not in seen and w not in out:
            out.append(w)
    return out[:top]


_DROP = re.compile(r"\b(?:dropped|abandoned|flung|hurled)\s+(?:my|the|his)\s+(\w+(?:\s+\w+)?)"
                   r"|\bmy\s+(\w+)\s+(?:hit|fell to|clattered (?:to|onto|across)|skidded across|slid across)\s+the\b", re.I)
_USE = re.compile(r"\b(?:held|holding|gripp\w*|clutch\w*|scratch\w*|jingl\w*|rattl\w*|unlock\w*|press\w*|using|dial\w*|"
                  r"rais\w*|lift\w*|wav\w*|swung|twist\w*|fumbl\w*|thumb\w*)\b", re.I)
_TAKE = re.compile(r"\b(?:picked|grabbed|snatched|retrieved|scooped|recovered|took|seized|found)\b", re.I)
_BODY = {"hand", "hands", "arm", "arms", "head", "eyes", "hair", "breath"}


def dropped_then_used(parts):
    """Objects the narrator drops or abandons and then uses again with no sentence about taking them back:
    [(object, index of the part that uses it)]. A simple word match: the refiner treats it as a logic issue for that
    section, and a rewrite is kept only if a second look no longer finds it."""
    sents = [(i, s) for i, p in enumerate(parts) for s in sentences(unmark(p))]
    out = []
    for k, (_, s) in enumerate(sents):
        for m in _DROP.finditer(s):
            noun = (m.group(1) or m.group(2)).split()
            # "keys on", "keys again" and "hands up" end at a small word; "heavy flashlight" takes the last
            small = len(noun) > 1 and (noun[1].lower() in _PREP | FUNC or len(noun[1]) < 4)
            noun = noun[0] if small else noun[-1]
            if noun.lower() in _BODY or noun.lower() in FUNC:
                continue
            root = stem(noun.lower())
            for j, later in sents[k + 1:]:
                if root in {stem(x) for x in words(later)}:
                    if _TAKE.search(later):
                        break
                    if _USE.search(later):
                        out.append((noun.lower(), j))
                        break
    return out
