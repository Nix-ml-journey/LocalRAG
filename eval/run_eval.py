"""Evaluate the briefs in briefs.yaml: retrieval hit rate and ranks, and (unless retrieval_only) the scripts themselves:
word-count accuracy, repeated phrases, [PAUSE] density, tense switches and sentence length."""
import dataclasses
import json
import logging
import re
import shutil
import statistics
from collections import Counter

import yaml

from ragscript import config
from ragscript.pipeline import Pipeline
from ragscript.refine import AREAS, FLIP, combine, parse_compare  # shared with the pipeline's ending choice
from ragscript.report import script_of

COLLECTIONS = ("lore", "scripts", "hooks")


def load_briefs():
    return yaml.safe_load((config.ROOT / "eval" / "briefs.yaml").read_text(encoding="utf-8"))


def ranks(pl, brief, k=10):
    """{expected source: its best rank across lore/scripts/hooks, or k+1 when it is not in the top k}. The one place
    that decides whether retrieval found what a brief expects; hit rate and top-k are both derived from it."""
    best = {w: k + 1 for w in brief.get("expect_sources", [])}
    for coll in COLLECTIONS:
        for i, c in enumerate(pl.ret.search(coll, brief["brief"], brief["genre"], k=k), 1):
            for w in best:
                if w in c["meta"]["source"]:
                    best[w] = min(best[w], i)
    return best


def top_k(all_ranks):
    """Share of expected sources ranked within the top 1 / 3 / 10, over every brief's ranks()."""
    flat = [r for rk in all_ranks for r in rk.values()]
    return {f"top{x}": sum(r <= x for r in flat) / len(flat) for x in (1, 3, 10)}


def parse_reader(reply):
    """{area: score} from the reader's reply. An area the reader left out or scored off the 1-5 scale is missing."""
    scores = {a: re.search(rf"(?mi)^\s*{a}\s*:\s*([1-5])\b", reply) for a in AREAS}
    return {a: int(m.group(1)) for a, m in scores.items() if m}


def judge_pair(pl, script_a, script_b, brief, genre):
    """(winner per criterion, how many criteria the two orderings disagreed on) for two scripts of one brief."""
    first = parse_compare(pl.compare(script_a, script_b, brief, genre))
    swapped = parse_compare(pl.compare(script_b, script_a, brief, genre))
    result = combine(first, swapped)
    return result, sum(first[c] != FLIP[swapped[c]] for c in result)


def pairs(dir_a, dir_b):
    """[(script in dir_a, script in dir_b)] for the same brief in two eval label folders under output/; the match is
    the file name without its time stamp."""
    key = lambda p: re.sub(r"-\d{4}-\d{6}$", "", p.stem)
    found = [{key(p): p for p in sorted((config.ROOT / "output" / d).glob("*.md"))} for d in (dir_a, dir_b)]
    return [(found[0][k], found[1][k]) for k in found[0] if k in found[1]]


def _brief_of(path):
    """The brief a saved script was written for: from its trace file in output/, else its title."""
    trace = config.ROOT / "output" / f"{path.stem}.trace.json"
    if trace.exists():
        return json.loads(trace.read_text(encoding="utf-8"))["brief"]
    return path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")


def compare(dir_a, dir_b):
    """Pairwise judge: for each brief written in both eval label folders, ask the model which script is better per
    criterion, twice with the order swapped. The judge is the model set in config.yaml, so a fair comparison
    keeps `llm.model` the same for every run it judges."""
    logging.basicConfig(level=logging.WARNING)
    pl, total = Pipeline.from_config(), Counter()
    for pa, pb in pairs(dir_a, dir_b):
        genre = re.match(r"(.+?)-\d+min-", pa.name).group(1)
        result, disagreed = judge_pair(pl, script_of(pa.read_text(encoding="utf-8")),
                                       script_of(pb.read_text(encoding="utf-8")), _brief_of(pa), genre)
        wins = Counter(v for c, v in result.items() if c != "Overall")
        total.update(wins)
        print(f"{genre:<8} overall={result.get('Overall', '?')}  {dir_a} {wins['A']}  {dir_b} {wins['B']}  tie {wins['tie']}  "
              f"order-disagreements={disagreed}  " + " ".join(f"{c[:4]}={result.get(c, '-')}" for c in AREAS), flush=True)
    print(f"\nCriteria won: {dir_a} {total['A']}, {dir_b} {total['B']}, ties {total['tie']} (A = {dir_a}, B = {dir_b})")


def _over_limit(s):
    """Details over the return limit in one script: the same words coming back, or a detail in too many paragraphs."""
    return sum(n > s.motif_limit for _, n in s.repeated_wording + s.detail_returns)


def run(retrieval_only=False, ids=None, minutes=None, label=None, reader=True):
    """ids: only these brief ids; minutes: write every brief at this length; label: also copy the scripts and a
    stats.json to output/<label>/ (a quick regression run: `cli.py eval --ids b01,b03,b08 --minutes 5 --label x`);
    reader: also have the model score each script as an editor (prompts/stages/reader.md, one extra call per brief)."""
    logging.basicConfig(level=logging.WARNING)
    briefs = [b for b in load_briefs() if not ids or b["id"] in ids]
    pl = Pipeline.from_config()
    final_k = pl.cfg["retrieval"]["final_k"]
    dest = config.ROOT / "output" / label if label else None
    if dest:
        dest.mkdir(parents=True, exist_ok=True)
    hits, all_ranks, reports, read_scores = [], [], [], []
    for b in briefs:
        rk = ranks(pl, b)
        all_ranks.append(rk)
        hit = all(r <= final_k for r in rk.values())  # found among the chunks a script would actually be given
        hits.append(hit)
        line = f"{b['id']:<14} retrieval={'HIT ' if hit else 'MISS'}"
        if not retrieval_only:
            h = pl.hooks(b["brief"], b["genre"])[0]
            r = pl.write(b["brief"], b["genre"], minutes or b["minutes"], h, progress=lambda m: None)
            s = r["stats"]
            reports.append(s)
            if dest:
                shutil.copy(r["path"], dest / r["path"].name)
            if reader:
                try:  # the score is optional: one failed call must not lose a run that took an hour
                    verdict = pl.read(script_of(r["markdown"]), b["brief"], b["genre"])
                except Exception as e:
                    logging.warning("reader failed for %s: %s", b["id"], e)
                    verdict = ""
                scores = parse_reader(verdict)
                read_scores.append(scores)
                line += f" reader={sum(scores.values())}/30" if len(scores) == len(AREAS) else " reader=n/a"
                if dest:
                    (dest / f"{r['path'].stem}.reader.txt").write_text(verdict, encoding="utf-8")
            line += (f" words={s.words}/{s.target_words} rep={s.repeated_phrase_rate:.1%}"
                     f" pauses/min={s.pauses_per_minute:.1f} tense={'MIXED' if s.mixed_tense else 'ok'}"
                     f" sent={s.avg_sentence_words} rep-outside={s.repeated_outside_setting:.1%} same-words={len(s.repeated_wording)}"
                     f" over-limit={_over_limit(s)}")
        print(line, flush=True)
    print(f"\nRetrieval hit rate: {sum(hits)}/{len(hits)}")
    print("Expected sources by rank: " + "  ".join(f"{k}={v:.2f}" for k, v in top_k(all_ranks).items()))
    if reports:
        print(f"Word-count accuracy (mean): {statistics.mean(s.word_accuracy for s in reports):.1%}")
        print(f"Repeated-phrase rate (mean): {statistics.mean(s.repeated_phrase_rate for s in reports):.1%}   "
              f"outside the setting words: {statistics.mean(s.repeated_outside_setting for s in reports):.1%}")
        print(f"[PAUSE] cues per minute (mean): {statistics.mean(s.pauses_per_minute for s in reports):.1f}   "
              f"scripts with mixed tense: {sum(s.mixed_tense for s in reports)}/{len(reports)}   "
              f"sentence length (mean): {statistics.mean(s.avg_sentence_words for s in reports):.1f} words")
        same_words = statistics.mean(len(s.repeated_wording) for s in reports)
        print(f"Details returning in the same words (mean per script): {same_words:.1f}   "
              f"scripts with one over the limit: {sum(_over_limit(s) > 0 for s in reports)}/{len(reports)}")
        if any(read_scores):
            print("Reader score (mean of 1-5): " + "  ".join(
                f"{a}={statistics.mean(s[a] for s in read_scores if a in s):.1f}" for a in AREAS
                if any(a in s for s in read_scores)))
        if dest:
            (dest / "stats.json").write_text(json.dumps([dataclasses.asdict(s) for s in reports], indent=1), encoding="utf-8")
            print(f"saved to {dest}")


if __name__ == "__main__":
    run()
