# Architecture

## The pipeline
1. **Hooks.** Three hook options for the brief, using the hooks collection as examples.
2. **Outline.** One JSON call gives the narrator and 7 beats (hook, setup, rising tension, reveal, climax, aftermath, outro). Each beat has a word budget, `place`, `sees`, `change`, `detail`, and the ending is planned with a `plant` and a `payoff`. Code checks the outline for holes and asks for one repair (`pipeline.build_outline`).
3. **Write.** `Drafter.draft` writes each beat with retrieval per beat, a rolling word budget and a rolling summary labelled "already told", and one corrective retry.
4. **Refine** (`Refiner.run`):
   - polish each section on its own;
   - one critic call lists logic holes in the whole script, and code adds holes the critic can miss (a dropped object used again, a recurring detail past its limit); the section with the most issues is rewritten once, and a logic rewrite is kept only if a second critic call no longer finds the hole;
   - the last section is written several ways and the pairwise judge keeps the best;
   - the repeat guard removes repeated paragraphs, retellings in new words and repeated sentences;
   - the length fit steers the total to within 5% of the target.
5. **Verify** (grounded genres only). The model tags facts `[S#]` and wraps anything it invents as `{{INVENTED: ...}}`; a Verify step re-checks every cited sentence and every sentence naming something without a citation against the lore (at most 40 checks).
6. **Package and report.** Titles, thumbnails, chapters, description, tags, and a Checks section with every measurement.

Retrieval has three collections: `scripts` (style examples), `lore` (facts) and `hooks`. Each is searched with dense vectors (bge-m3) and BM25, merged by reciprocal rank fusion, and filtered by genre and by story beat.

## Code layout
- `ragscript/pipeline.py`: orchestrates hooks, outline (with repair) and the stage modules. Dependencies are passed in (`Pipeline(cfg, retriever, llm)`); `Pipeline.from_config()` builds the real ones. Every model call carries its stage name (`llm.chat(prompt, stage="outline")`), so test doubles never match on prompt text. It saves `<script>.trace.json` (outline, drafted and refined sections) next to each script. `Story` is the shared context; `prompts.py` fills the `<<placeholders>>` of stage templates.
- `ragscript/draft.py`: `Drafter.draft`, the sections beat by beat.
- `ragscript/refine.py`: everything after a section is drafted. `draft_issues()` and `final_issues()` return `Issue(kind, message)` records; `attempt()` keeps a rewrite only if it is better; `Refiner.run` does polish, the critic, the rewrite of the worst section, `best_ending`, the repeat guard and the length fit. `Refiner.compare` is the pairwise judge that both the ending choice and `cli.py compare` use.
- `ragscript/checks.py`: checks on the narration (repeats, overlap, tense, crutch words). `ragscript/logic.py`: story-logic checks (outline holes and the planned ending, detail returns and over-use, dropped objects, a new thing in the ending). `text.py` holds the one sentence splitter; `grounding.py` the `[S#]` citations, `{{INVENTED}}` marks and Verify. Every threshold comes from `config.yaml`.
- `ragscript/report.py`: `Report` holds every measurement of a finished script and decides which warnings to show; `render` turns a `Doc`, the script and the `Report` into markdown. A new check is a field in `Report` and a line in `render`. `report.save` never overwrites an earlier script; `report.script_of` reads the narration back from a saved file.
- `ragscript/store.py` + `ingest.py`: `Store(cfg, embed=None)` is the search index (Chroma plus BM25, injectable embedder) and `Retriever` fuses the two rankings. `chunking.Beat` is the one record of an outline beat. `Ingest` fills the store from `data/`, holds the per-file metadata rules and saves progress after every file. `chunking.py` and `loaders.py` feed it; `sources.py` numbers the cited chunks.
- `ragscript/llm.py` (Ollama, out-of-memory retry), `genre.py`, `vram.py`. Entry points: `cli.py` and `app.py`. `eval/run_eval.py` runs the briefs and the pairwise judge; `eval/lore_check.py` checks lore notes before ingest.
- `config.yaml` holds every tuning number (`style:`, `length:`, `checks:`); `prompts/` holds the stage and genre prompts.

## Tests
`tests/support.py` has the fake model and retriever, `make_pipeline`, `ingest_cfg` and `compare_reply`. Tests are split by area: `test_checks.py` (checks and story logic), `test_outline.py`, `test_refine.py`, `test_eval.py`, `test_pipeline.py` (stages, ingest, prompts, reports), `test_core.py` (text, grounding, chunking), plus `test_length_guard.py` and `test_pipeline_guard.py` (end-to-end length and repeat guards), `test_loaders.py`, `test_lore_check.py` and `test_retrieval.py`. Run them with `python -m pytest tests`; they need no GPU and no Ollama.
