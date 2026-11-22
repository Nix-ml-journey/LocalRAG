# Story Script RAG

Local, offline RAG system that writes first-person story scripts (horror, history mysteries, fantasy, Cthulhu, sci-fi, literary, or any genre you name) at a chosen length of 5, 10, 15 or 20 minutes. Everything is free and runs on one consumer GPU (built and measured on an RTX 5060 Ti 16 GB, peak VRAM under 14.5 GB). The LLM runs on the GPU through Ollama; embeddings, ChromaDB and BM25 run on the CPU.

## How it works
```mermaid
flowchart LR
    B[Brief + genre] --> H[3 hook options]
    H --> O[Outline: 7 beats, word budgets,<br/>place / sees / change / detail,<br/>planted ending]
    O --> W[Write each beat]
    R[(Hybrid retrieval<br/>Chroma + BM25 + RRF<br/>style / lore / hooks)] --> W
    W --> P[Polish each beat]
    P --> C[Critic reads the whole script]
    C --> RV[Rewrite the weakest section]
    RV --> E[Last section written 3 ways,<br/>judge keeps the best]
    E --> D[Repeat guard]
    D --> F[Length fit to +/-5%]
    F --> V[Verify facts<br/>grounded genres]
    V --> M[Markdown: titles, thumbnails,<br/>script, chapters, tags, checks]
```
- **Retrieval:** three collections (`scripts` for style, `lore` for facts, `hooks`), each searched with dense vectors (bge-m3) and BM25, merged by reciprocal rank fusion, and filtered by genre and story beat.
- **A planned story:** the outline gives every beat a word budget, a place, what the narrator looks at, one new event and a recurring detail, and it plants the ending early. Code checks the outline for holes and asks for up to two repairs.
- **Checks that catch what a small model misses:** a tracker call writes where the narrator is and what they hold in each section, a critic call compares those lines, and code checks add dropped objects used again and details that come back too often; the weakest section (and every section with an over-used detail) is rewritten, and a fix is dropped if a second look finds more holes than before. After the ending is chosen, crutch words and phrases that return in the same words are rewritten, and the last tics are trimmed by rule.
- **Guards:** repeated and retold paragraphs are removed, and the length is steered to within 5% of the target.
- **Grounded history:** facts are tagged `[S#]` to their lore chunk, invented details are wrapped as `{{INVENTED: ...}}`, and a Verify step re-checks them.

Details: [architecture](docs/architecture.md), [design decisions](docs/design-decisions.md) (what I tested and why), [guide](docs/guide.md) (commands, adding material, tuning).

## Results
Measured on this machine (`num_ctx` 16384). Single runs; quality metrics swing a few points.

| Check | Result |
|---|---|
| Retrieval, 17 briefs (`cli.py eval --retrieval-only`) | right source at top 1: 0.91, top 3: 0.95, top 10: 1.00; 16 of 17 briefs hit (a hit means the source is within the chunks a script is given). Re-measured after the Machen and Wells ingest; the one miss is b10, whose only expected source, the generic `hooks_openers`, ranks 9th |
| Three briefs (horror, history, Cthulhu), 5 min, target 750 | accuracy 97.1% mean (736-780 words), repeats 3.1%, no tense switches, 1.0 `[PAUSE]` per minute, sentences 15.6 words |
| Ten briefs (`cli.py eval --ids b01,...,b10 --minutes 5`), latest code | accuracy 98.2%, repeats 2.2% (0.8% outside the setting words), sentences 16.0 words, no detail over its limit, no tense switches; retrieval hit 9 of 10 |
| Same ten scripts read by me with the six-area reader prompt (1-5 each, 30 total) | mean 18.6 of 30 (range 16-20, four scripts at 20); over four runs 17.3, 17.5, 18.6, 18.6. One reader, so it is a trend, not a measurement; Prose is 2 on all ten |
| Second set of ten (`b18`-`b27`: four Machen-style horror, three War of the Worlds sci-fi, three history), 5 min | accuracy 97.7%, repeats 2.3% (1.5% outside the setting words), sentences 15.4 words, no tense switches; retrieval hit 9 of 10; my reader score 15.9 of 30 (range 12-20). The sci-fi scripts score lowest: they drift into modern settings (trucks, phones, a London flat) instead of Wells's period |
| Six briefs from the two biggest genres (horror `b01 b02 b10 b16`, fantasy `b06 b07`), 5 min | accuracy 98.0%, repeats 1.9% (0.6% outside the setting words), sentences 15.9 words; retrieval hit 5 of 6; my reader score 18.0 of 30 (range 17-21), in line with the first ten |
| History 5-min script with Verify | 722 words (96.3%), repeats 4.3%, 14 sentences marked `{{INVENTED}}` |
| Speed | about 2.5-7 min per 5-min script (about 1 more with the ending choice); decode-bound at ~40-45 tokens/s |
| Offline tests (no GPU, no Ollama) | 116 pass, including an end-to-end repeat guard that fails if the dedupe is removed |

Unedited excerpt (history, "The Mary Celeste", gemma4:12b; `{{INVENTED}}` marks claims the lore does not support, `[S1]` marks facts that do):
> On December 4, 1872, the *Dei Gratia* spotted her sailing erratically in the Atlantic [S1]. No one answered their signals, and a boarding party found the ship completely deserted [S1].
> {{INVENTED: The captain's log ended abruptly, leaving a haunting void in the official record.}}

## Quick start
Needs Python 3.11 to 3.13 and [Ollama](https://ollama.com). **Data is not included:** put your own `.txt .md .docx .pdf .epub` files in `data/<genre>/` (for example `data/horror/`), name fact notes `lore_*.md`, and use only text you have the right to use.

```powershell
.\setup.ps1                  # venv, CPU torch, requirements, ollama pull gemma4:12b
.\.venv\Scripts\Activate.ps1
```
Linux: `bash setup.sh && source .venv/bin/activate`. bge-m3 (~2 GB) downloads on first ingest; after that everything runs offline.

```bash
python cli.py ingest                                   # index data/ (incremental)
python cli.py hooks "A lighthouse keeper finds extra footprints" --genre horror
python cli.py write "The lost colony of Roanoke" --genre history --minutes 10 --hook 2
python app.py                                          # Gradio UI at http://127.0.0.1:7860
python cli.py eval --ids b01,b03,b08 --minutes 5 --label NAME   # quick regression on 3 briefs
python cli.py compare NAME_A NAME_B                    # which of two eval runs is better (pairwise judge)
python -m pytest tests                                 # offline self-checks (pip install -r requirements-dev.txt)
```
Output goes to `output/<genre>-<min>min-<slug>-<MMDD-HHMMSS>.md`: titles, thumbnail text, a clean narration ready to record, estimated chapters, description, tags, sources and a Checks section. More commands and options are in the [guide](docs/guide.md).

## Limitations
- Single user, single machine; no concurrency design beyond the polish step.
- Script quality is judged on a few briefs plus reading, not a user study. Repetition is reduced, not solved (repeated-phrase rates ranged from 1.9% to 6.7% across horror runs).
- Sentences average 15-16 words (target 12-16); read aloud before recording.
- The Verify step is strict and can mark a true paraphrase as `INVENTED`. History facts come from a small set of lore notes (10 topics): fact-check every `{{INVENTED}}` mark and the Checks section before publishing.
- The logic checks are word matches with no tagger; they miss paraphrases and can flag an object another character took.
- The books and scripts used as style data are **not** in this repository.

## Built with AI assistance
This project was developed with the help of AI tools: Claude (Claude Code, with plugins for prompt engineering, refactoring and code review) and Cursor. I chose what to build, picked the data, ran the tests and generation runs on my own GPU, and decided what to keep.
