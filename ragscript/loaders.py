"""Load and clean source files. Returns (front-matter dict, cleaned text)."""
import html
import posixpath
import re
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET

import yaml

SUPPORTED = {".txt", ".md", ".docx", ".pdf", ".epub", ".srt", ".vtt"}
_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_TS = re.compile(r"^\s*(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{3}\s*-->.*$", re.M)
_FILLER = re.compile(r"\b(um+|uh+|erm|you know|i mean)\b,?\s*", re.I)
# Project Gutenberg wraps every book in a licence header and footer
_PG = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n(.*?)(?:\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG|\Z)",
                 re.S | re.I)


def _epub(p: Path) -> str:
    """Chapters in reading order (container.xml -> OPF spine), tags stripped. Standard library only."""
    z = zipfile.ZipFile(p)
    name = lambda e: e.tag.rsplit("}", 1)[-1]
    opf = next(e.get("full-path") for e in ET.fromstring(z.read("META-INF/container.xml")).iter() if name(e) == "rootfile")
    root = ET.fromstring(z.read(opf))
    href = {e.get("id"): e.get("href") for e in root.iter() if name(e) == "item"}
    out = []
    for e in root.iter():
        if name(e) == "itemref" and e.get("idref") in href:
            raw = z.read(posixpath.normpath(posixpath.join(posixpath.dirname(opf), unquote(href[e.get("idref")]))))
            t = re.sub(r"(?is)<(script|style).*?</\1>", "", raw.decode("utf-8", "replace"))
            t = re.sub(r"(?i)</(p|div|h[1-6]|li|blockquote)>|<br\s*/?>", "\n\n", t)
            out.append(html.unescape(re.sub(r"<[^>]+>", "", t)))
    return re.sub(r"(?m)^[ \t]+|[ \t]+$", "", "\n\n".join(out))  # xhtml indentation becomes stray spaces


def read_raw(p: Path) -> str:
    s = p.suffix.lower()
    if s == ".docx":
        import docx
        return "\n\n".join(para.text for para in docx.Document(str(p)).paragraphs)
    if s == ".pdf":
        from pypdf import PdfReader
        text = "\n\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
        text = unicodedata.normalize("NFKC", text)  # ligature glyphs (ﬀ ﬃ) back to plain letters
        text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)  # bare page numbers
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # words hyphenated across a line break
    if s == ".epub":
        return _epub(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _subtitles(text: str) -> str:
    text = re.sub(r"^WEBVTT.*$", "", text, flags=re.M)
    text = _TS.sub("", text)
    lines, prev = [], None
    for ln in text.splitlines():
        ln = re.sub(r"<[^>]+>", "", ln).strip()
        if not ln or ln.isdigit() or ln == prev:  # drop cue numbers and rolling-caption repeats
            continue
        lines.append(ln)
        prev = ln
    return _FILLER.sub("", " ".join(lines))


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_front_matter(text: str):
    """(front-matter dict, text without it)."""
    m = _FM.match(text)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, text[m.end():]


def load(p: Path):
    raw = read_raw(p)
    if p.suffix.lower() in {".srt", ".vtt"}:
        return {}, clean(_subtitles(raw))
    fm, raw = split_front_matter(raw)
    if m := _PG.search(raw):
        raw = m.group(1)
    return fm, clean(raw)
