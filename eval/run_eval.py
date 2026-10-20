"""Evaluate the briefs in briefs.yaml: retrieval hit rate and ranks, word-count accuracy, repeated-phrase rate."""
import logging
import statistics
import time

import yaml

from ragscript import config
from ragscript.pipeline import Pipeline


def load_briefs():
    return yaml.safe_load((config.ROOT / "eval" / "briefs.yaml").read_text(encoding="utf-8"))


def rank_metrics(pl, briefs, k=10):
    """Share of expected sources whose best rank (across lore/scripts/hooks) is within the top 1 / 3 / 10."""
    ranks = []
    for b in briefs:
        best = {}
        for coll in ("lore", "scripts", "hooks"):
            for i, c in enumerate(pl.ret.search(coll, b["brief"], b["genre"], k=k), 1):
                for w in b.get("expect_sources", []):
                    if w in c["meta"]["source"]:
                        best[w] = min(best.get(w, k + 1), i)
        ranks += [best.get(w, k + 1) for w in b.get("expect_sources", [])]
    return {f"top{x}": sum(r <= x for r in ranks) / len(ranks) for x in (1, 3, 10)}


def compare_rerank(model=None):
    """Retrieval only: the same briefs with the reranker off and on, plus the time each pass took."""
    pl, briefs = Pipeline(), load_briefs()
    pl.cfg["reranker"]["model"] = model or pl.cfg["reranker"]["model"]
    rank_metrics(pl, briefs[:1])  # load bge-m3 and build the BM25 indexes outside the timing
    for on in (False, True):
        pl.cfg["reranker"]["enabled"] = on
        if on:
            pl.ret.search("lore", "warm up the reranker", "history")  # load the reranker outside the timing
        t0 = time.time()
        m = rank_metrics(pl, briefs)
        print(f"reranker {'ON ' if on else 'OFF'} ({pl.cfg['reranker']['model'] if on else '-'}): "
              + "  ".join(f"{k}={v:.2f}" for k, v in m.items()) + f"  [{time.time() - t0:.1f}s for {len(briefs) * 3} searches]")


def run(retrieval_only=False):
    logging.basicConfig(level=logging.WARNING)
    briefs = load_briefs()
    pl = Pipeline()
    hits, acc, rep = [], [], []
    for b in briefs:
        got = {c["meta"]["source"] for coll in ("lore", "scripts", "hooks")
               for c in pl.ret.search(coll, b["brief"], b["genre"])}
        want = b.get("expect_sources", [])
        hit = all(any(w in g for g in got) for w in want) if want else True
        hits.append(hit)
        line = f"{b['id']:<14} retrieval={'HIT ' if hit else 'MISS'}"
        if not retrieval_only:
            h = pl.hooks(b["brief"], b["genre"])[0]
            s = pl.write(b["brief"], b["genre"], b["minutes"], h, progress=lambda m: None)["stats"]
            acc.append(s["word_accuracy"])
            rep.append(s["repeated_phrase_rate"])
            line += f" words={s['words']}/{s['target_words']} rep={s['repeated_phrase_rate']:.1%}"
        print(line)
    print(f"\nRetrieval hit rate: {sum(hits)}/{len(hits)}")
    print("Expected sources by rank: " + "  ".join(f"{k}={v:.2f}" for k, v in rank_metrics(pl, briefs).items()))
    if acc:
        print(f"Word-count accuracy (mean): {statistics.mean(acc):.1%}")
        print(f"Repeated-phrase rate (mean): {statistics.mean(rep):.1%}")


if __name__ == "__main__":
    run()
