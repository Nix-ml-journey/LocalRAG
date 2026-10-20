"""Regression tests for the audit findings: bad model JSON, verify evidence, length fit with line breaks, long text
without full stops. Same offline stubs as test_pipeline_guard.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscript import checks, chunking, config
from ragscript.pipeline import Pipeline
from test_length_guard import LengthLLM
from test_pipeline_guard import BEATS, FakeLLM, FakeRetriever

CFG = config.load()


def pipeline(tmp_path, llm, retriever=None):
    p = Pipeline.__new__(Pipeline)
    p.cfg = config.load()
    p.cfg["llm"]["parallel"] = 1
    p.cfg["paths"]["output_dir"] = str(tmp_path)
    p.ret, p.llm, p.banned = retriever or FakeRetriever(), llm, []
    return p


def write(p, genre="horror"):
    return p.write("a guard keeps seeing himself", genre, 5, "hook text", progress=lambda m: None)


def test_package_answer_that_is_not_an_object_is_retried(tmp_path):
    class L(FakeLLM):
        packs = 0

        def chat(self, prompt, **kw):
            if prompt.startswith("You are packaging"):
                self.packs += 1
                return '["a", "b"]' if self.packs == 1 else super().chat(prompt, **kw)
            return super().chat(prompt, **kw)

    llm = L()
    md = write(pipeline(tmp_path, llm))["markdown"]
    assert llm.packs == 2 and "## Title options\n1. a" in md


def test_outline_with_loose_beat_names_and_bad_words_is_fixed_or_retried(tmp_path):
    class L(FakeLLM):
        outlines, section_prompts = 0, []

        def chat(self, prompt, **kw):
            if prompt.startswith("You are a story architect"):
                self.outlines += 1
                names = [b.title().replace("_", " ") for b in BEATS]  # "Hook", "Rising Tension"...
                words = "about 100" if self.outlines == 1 else 107
                beats = [{"beat": n, "title": n, "words": words, "summary": "s"} for n in names]
                return json.dumps({"narrator": "Eli", "beats": beats})
            if prompt.startswith("You are writing one section"):
                self.section_prompts.append(prompt)
            return super().chat(prompt, **kw)

    llm = L()
    write(pipeline(tmp_path, llm))
    assert llm.outlines == 2, "words that are not a number must make the outline attempt fail"
    assert "Open with this chosen hook" in llm.section_prompts[0], "'Hook' must be read as the hook beat"


def test_verify_uses_the_lore_chunk_that_matches_the_sentence(tmp_path):
    fact = "Captain Morehouse sailed from Gibraltar in November."

    class R(FakeRetriever):
        n = 0

        def search(self, coll, *a, **k):
            if coll != "lore":
                return []
            out = []
            for _ in range(4):
                R.n += 1
                text = fact if R.n == 20 else f"Filler note {R.n} about unrelated matters."
                out.append({"id": f"lore:f.md:{R.n}", "text": text, "meta": {"source": "f.md"}})
            return out

    class L(FakeLLM):
        def chat(self, prompt, **kw):
            if prompt.startswith("You are a strict fact checker"):
                lore, sentence = prompt.split("Sentence from the script:")
                return "YES" if "Morehouse" in sentence and "Gibraltar in November" in lore else "NO"
            text = super().chat(prompt, **kw)
            if prompt.startswith("You are writing one section") and not self.written[1:]:
                text = fact + " " + text
                self.written[-1] = text
            return text

    md = write(pipeline(tmp_path, L(), R()), genre="history")["markdown"]
    script = md[md.index("## Script"): md.index("## Chapters")]
    assert "Captain Morehouse sailed" in script
    assert "{{INVENTED: Captain Morehouse" not in script, "the sentence's own source was left out of the check"


def test_length_fit_accepts_a_rewrite_with_a_line_break(tmp_path):
    class L(LengthLLM):
        def chat(self, prompt, **kw):
            out = super().chat(prompt, **kw)
            if prompt.startswith("You are a length editor"):
                w = out.split()
                out = " ".join(w[:5]) + ".\n" + " ".join(w[5:])  # a break after a full stop is what dedupe flattens
            return out

    md = write(pipeline(tmp_path, L(section_words=150)))["markdown"]
    n = checks.spoken_count(md[md.index("## Script") + 9: md.index("## Chapters")])
    assert 675 <= n <= 825, n


def test_text_without_full_stops_is_still_chunked():
    text = " ".join(f"w{i}" for i in range(3000))
    chunks = chunking.sliding(text, CFG["chunking"]["min_tokens"], CFG["chunking"]["max_tokens"], CFG["chunking"]["overlap"])
    assert len(chunks) > 5
    assert all(chunking.tokens(c) <= CFG["chunking"]["max_tokens"] for c in chunks)


def test_deleted_file_leaves_no_chunks_even_if_its_tagging_failed(tmp_path, monkeypatch):
    from ragscript.store import Store
    cfg = config.load()
    cfg["paths"].update(data_dir=str(tmp_path / "data"), chroma_dir=str(tmp_path / "db"), state_file=str(tmp_path / "db/state.json"))
    (tmp_path / "data/horror").mkdir(parents=True)
    f = tmp_path / "data/horror/a.txt"
    f.write_text("The tide came in. " * 80, encoding="utf-8")
    st = Store(cfg)
    monkeypatch.setattr(st, "embed", lambda texts: [[1.0, 0.0] for _ in texts])

    def tagger_down(text):
        raise ConnectionError("ollama is down")

    assert st.ingest(tagger_down)["scripts"] > 0  # indexed, but the file is not marked done
    f.unlink()
    assert st.ingest(tagger_down) == {"scripts": 0, "lore": 0, "hooks": 0}
