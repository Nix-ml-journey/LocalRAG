"""CLI: python cli.py ingest | hooks | write | eval"""
import argparse
import logging

from ragscript.config import DURATIONS
from ragscript.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Ollama call buries the VRAM log


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="index data/ (incremental)")
    for name in ("hooks", "write"):
        p = sub.add_parser(name)
        p.add_argument("brief")
        p.add_argument("--genre", required=True, help="any genre, e.g. horror, sci-fi, true crime")
        p.add_argument("--style", default="", help='optional style notes, e.g. "dry humour, short sentences"')
        if name == "write":
            p.add_argument("--minutes", type=int, choices=DURATIONS, default=10)
            p.add_argument("--hook", type=int, choices=(1, 2, 3), default=1, help="which of the 3 hooks to use (1-3)")
    e = sub.add_parser("eval")
    e.add_argument("--retrieval-only", action="store_true")
    e.add_argument("--ids", help="comma-separated brief ids from eval/briefs.yaml, e.g. b01,b03 (default: all)")
    e.add_argument("--minutes", type=int, choices=DURATIONS, help="write every brief at this length instead of its own")
    e.add_argument("--label", help="copy the written scripts and a stats.json to output/<label>/")
    e.add_argument("--no-reader", action="store_true", help="skip the model's editor score of each script")
    a = ap.parse_args()

    if a.cmd == "eval":
        from eval.run_eval import run
        return run(retrieval_only=a.retrieval_only, ids=a.ids.split(",") if a.ids else None, minutes=a.minutes, label=a.label,
                   reader=not a.no_reader)
    pl = Pipeline.from_config()
    if a.cmd == "ingest":
        print(pl.ingest())
    elif a.cmd == "hooks":
        for i, h in enumerate(pl.hooks(a.brief, a.genre, a.style), 1):
            print(f"\n{i}. {h}")
    else:
        hooks = pl.hooks(a.brief, a.genre, a.style)
        r = pl.write(a.brief, a.genre, a.minutes, hooks[min(a.hook, len(hooks)) - 1], a.style)
        print(f"\nSaved {r['path']}  ({r['stats'].words}/{r['stats'].target_words} words)")


if __name__ == "__main__":
    main()
