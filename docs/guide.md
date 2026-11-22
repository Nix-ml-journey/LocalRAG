# Guide: using and tuning

## Commands
```bash
python cli.py ingest                                   # incremental: only new or changed files are re-indexed; an interrupted run resumes
python cli.py hooks "A lighthouse keeper finds extra footprints" --genre horror
python cli.py write "The lost colony of Roanoke" --genre history --minutes 10 --hook 2
python cli.py write "A tech finds the ship AI is lying" --genre sci-fi --style "wry, practical details" --minutes 5
python app.py                                          # Gradio UI at http://127.0.0.1:7860
python cli.py eval [--retrieval-only]                  # the 17 briefs in eval/briefs.yaml; a full run adds the script metrics
python cli.py eval --ids b01,b03,b08 --minutes 5 --label NAME   # quick regression: 3 real scripts -> output/NAME/ + stats.json
python cli.py compare NAME_A NAME_B                    # which of two eval runs wrote the better scripts (pairwise, order swapped)
python -m eval.lore_check data/history/lore_*.md       # before ingesting lore: garbled text, front matter, length, names and years seen once
pip install -r requirements-dev.txt && python -m pytest tests   # model-free self-checks
```
Output goes to `output/<genre>-<min>min-<slug>-<MMDD-HHMMSS>.md` (a rerun never overwrites an earlier script): titles (5), thumbnail text (3), narration with `[PAUSE]` / `[SFX: …]` / `[B-ROLL: …]` (the `## Script` section is clean: no `[S#]` tags, no `{{INVENTED}}` braces, ready to record), estimated chapters, description, tags, sources, and a Checks section. Grounded genres end with a **Fact-check copy** that keeps every `[S#]` tag and `{{INVENTED}}` mark.

## Any genre
`--genre` (or the UI box) takes any name: "sci-fi", "true crime", "romance"… The name is lower-cased and hyphenated (`True Crime` → `true-crime`).
- **Examples:** put files in `data/<name>/`, then `ingest`. No folder means no style examples, but it still works.
- **Rules:** an optional `prompts/<name>.md`. Without it, `prompts/default.md` is used (a warning is logged).
- **Fact-grounded genres** (true crime, science, biography): start the prompt file with `---\ngrounded: true\n---`. Facts then come only from `lore_*.md` notes, and invented details are marked `{{INVENTED: …}}`. `history.md` does this already.
- `--style "…"` (or the UI field) adds one-off style notes for a single script.
- The narrator is always first person and the story always uses the 7 beats.

## Adding your material
Drop files (`.txt .md .docx .pdf .epub .srt .vtt`) in `data/<genre>/`, then run `ingest`. A file straight in `data/` is skipped with a warning (the folder is the genre). Keep the original books outside `data/` (this project uses a gitignored `sources/` folder); one file per story or chapter gives better style examples than one file per book.
- **Books:** EPUB chapters are read in reading order; Project Gutenberg headers and footers, PDF page numbers and line-break hyphens are removed automatically (poppler's `pdftotext` is used when it is on PATH). Scanned PDFs need OCR first.
- **Past scripts** go to the style collection. Optional YAML front matter: `subgenre, tone, pov, views, retention`. Missing tags are filled once by the model (cached by file hash).
- **Lore and facts:** name the file `lore_*.md` (or front matter `kind: lore`).
- **Hooks and endings:** `data/hooks/*.md`, blocks separated by a `---` line, first line `TYPE: hook` or `TYPE: ending`.
- **Beats:** chunking uses `## Hook / Setup / Rising tension / Reveal / Climax / Aftermath / Outro` headings (or `[BEAT: climax]`). Without them, 300-500 token chunks with 15% overlap are used and beats are estimated from position.
- Delete a file and the next `ingest` removes it from the index.

## Customising
- `config.yaml`: model, `num_ctx`, retrieval sizes, words per minute, `style:` targets (sentence length, `[PAUSE]` limit), and the `length:` and `checks:` thresholds the results were tuned with.
- `prompts/<genre>.md` are genre rules; `prompts/stages/*.md` are stage templates (placeholders look like `<<name>>`).
- `banned_phrases.txt`: one phrase per line; used in the prompts and checked after generation (one rewrite attempt, then listed in Checks).
- Swap model: `ollama pull mistral-nemo`, then set `llm.model: mistral-nemo`. Models without thinking support are detected automatically.

## Fact-checking and quality checks
Grounded genres take facts only from retrieved lore (`[S#]`); anything invented is wrapped as `{{INVENTED: …}}` and listed under Checks. After polishing, the Verify step asks the model YES or NO ("does the lore state this?") for every cited sentence and every sentence that names something without a citation (at most 40 checks); each NO becomes `{{INVENTED}}`. This reduces but cannot eliminate invention, and paraphrases can be mis-marked: fact-check before publishing. The lore notes are paraphrased from Wikipedia (each file's front matter names the source and check date), so verify them too.

Every output also reports:
- word-count accuracy and average sentence length;
- the repeated-phrase rate, and the same rate outside the setting words (a repeated phrase holding a word from the brief or the outline's place, sees or detail does not count there, because a night-shift story will keep saying "monitor" and "fan"); flagged above `checks.repeat_flag` = 3%;
- details that return in the same words, and the paragraphs that bring back each outline detail (flagged above `checks.motif_max` = 4);
- crutch words and phrases that return in the same words are rewritten (`checks.crowd_min`, `checks.wording_max` = 2); the last tic modifiers in `checks.trim_words` stay `checks.trim_keep` = 2 times and a repeated SFX/B-ROLL cue stays twice;
- banned-phrase hits, passages with over 80% 5-gram overlap with a source, crutch words, past/present tense switches and `[PAUSE]` density.

The weakest-section rewrite fixes tense, `[PAUSE]`, repeat and logic problems automatically; the rest only report.

## Optional: parallel polish
To polish 2 sections at once, set the Ollama server variable `OLLAMA_NUM_PARALLEL=2` (PowerShell: `[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL","2","User")`, then restart Ollama) **and** `llm.parallel: 2` in `config.yaml` (peak GPU 11.8 → 13.3 GB). Undo with `SetEnvironmentVariable("OLLAMA_NUM_PARALLEL",$null,"User")`.

## VRAM troubleshooting
Every stage logs VRAM and RAM and warns above 14.5 GB. On out-of-memory the LLM wrapper halves `num_ctx` (down to `llm.min_ctx`) and retries. If it still fails: lower `llm.num_ctx` (12288 or 8192), set `llm.keep_alive: 0`, close other GPU apps, or use a smaller model. Embeddings are CPU-only.
