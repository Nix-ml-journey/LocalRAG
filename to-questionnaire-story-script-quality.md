# Improving story quality and length control in a RAG story-script generator

**Purpose:** I need to decide which general fixes to make next so the generated scripts are both the right length and read as real stories. Fixing one metric (repeats, length) has often made the story worse, and I can't tell which changes are worth the complexity.

**From:** the project owner, **To:** a Senior AI Engineer (RAG & Search), **How your answers will be used:** to rank the next fixes, to write down quality rules the pipeline must follow, and to build a small set of "better story" verdicts.

## Context

The project writes first-person YouTube story scripts (5 to 20 minutes, about 150 words per minute) with a local model (gemma4:12b through Ollama). Retrieval is hybrid (bge-m3 dense plus BM25, merged with reciprocal rank fusion) over style examples, lore and hooks. The pipeline is: hooks, then a JSON outline of 7 beats, then one model call per beat with a rolling word budget and a rolling summary, then per-section polish, one critic call that lists logic holes, a rewrite of the weakest section, a repeat guard, and a length-fit rewrite. Code checks run on repeated phrases, near-duplicate paragraphs, tense and word count.

Already tried: the outline gives each beat a `place` (where the narrator is), `sees` (which screen he watches), `change` (what is new) and `detail` (which recurring detail); code finds outline holes and asks for one repair. The length tolerance is now 5% and polish may grow a section by at most 15%. On three 5-minute briefs this gave word accuracy 97% and repeat rate 3.1%, but the stories still read as uneven.

Two recent horror outputs for the same brief ("a night-shift security guard keeps seeing himself on the monitors a few seconds before he walks into each room"):

- Run A, 724 words (target 750): clear escalation and one good turn (a flashlight click comes from behind his head, not the monitor), but the glitch is explained away three times and one object appears and is never explained.
- Run B, 751 words: good hook and ending image, but the narrator is in two places at once (patrolling a hall while also watching a monitor), one paragraph is generic filler, and the same detail returns 4 to 5 times in near-identical words.

## How to answer

About 30 minutes. Short answers are fine; a ranked list is enough where I ask for one. "I don't know" and "it depends, here is on what" are useful answers. Please flag anything you are unsure about.

## Which fix to try first

### Rank these fixes by expected gain in story quality per unit of effort.

The fixes still open: (1) a critic pass whose rewrite is checked by a second critic call, not trusted; (2) writing the whole script in one pass, with the outline as a plan, instead of one call per beat; (3) generating several outlines or drafts and picking the best with a model judge; (4) a larger or different writing model; (5) few-shot examples of complete good stories in the section prompt.

_Why this matters: each fix costs time on a 12B model that runs at about 46 tokens per second, so I can try only one or two._

>

### Which one of those would you not try, and why?

>

### Which stage of the pipeline would you instrument first to find where story quality or length is lost?

_Why this matters: I can measure after the outline, after each section, after polish and after the length fit, and I need to know which gives the clearest signal._

>

## Story quality rules

### What makes a short first-person horror script read as a story and not as atmosphere with a plot attached?

_Why this matters: I want to write these as rules into the prompts and into checks._

>

### Which of those rules can be checked in code or by a cheap model call, and which need a person?

>

### How should recurring details (a sound, a smell, an object) be controlled: a cap on returns in code, a per-beat assignment in the outline, or only in the prompt?

_Why this matters: each detail may return up to 3 times if it changes each time, but the model copies the wording and goes over the cap._

>

## Length control

### What is the most reliable way to land within 5% of a word target with a small local model?

_Why this matters: asking for "about N words" gives 1.5 to 5 times overshoot on short sections, and a rewrite to fit the length can damage the story._

>

### Would you trim by code (cutting the weakest sentences) or by a model rewrite, and what should the fallback be when a rewrite fails?

>

### How exact does a length target need to be for a YouTube script, and what error would you accept?

>

## Sample verdicts

### Of Run A and Run B above, which is the better story, and what single change would improve it most?

_Why this matters: I am the only reader so far, and I want a second view before I tune against my own taste._

>

### What would you measure to compare two story outputs without reading every one?

>

### How many samples per setting do you need before you trust a difference?

_Why this matters: repeat rates over ten runs ranged from 1.9% to 6.7% on the same brief, so one run proves little._

>

## Anything else?

Anything we didn't ask that we should know?

>
