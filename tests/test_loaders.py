"""Book extraction: EPUB reading order and tags, Project Gutenberg wrapper."""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragscript import loaders


def make_epub(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf", '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
                   '<item id="b" href="ch%202.xhtml"/><item id="a" href="ch1.xhtml"/></manifest>'
                   '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>')  # spine order differs from manifest order
        z.writestr("OEBPS/ch1.xhtml", "<html><style>p{}</style><body><h1>One</h1><p>It was dark &amp; cold.</p><p>Then silence.</p></body></html>")
        z.writestr("OEBPS/ch 2.xhtml", "<html><body><p>The second chapter begins.</p></body></html>")


def test_epub_is_read_in_spine_order_without_tags(tmp_path):
    f = tmp_path / "book.epub"
    make_epub(f)
    _, text = loaders.load(f)
    assert text.index("It was dark & cold.") < text.index("second chapter")
    assert "<" not in text and "p{}" not in text
    assert "It was dark & cold.\n\nThen silence." in text  # paragraphs survive as blank lines


def test_gutenberg_header_and_footer_are_removed(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("Title page junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nThe real story.\n"
                 "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicence text", encoding="utf-8")
    assert loaders.load(f)[1] == "The real story."


def test_plain_text_is_untouched(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text("Just a story.", encoding="utf-8")
    assert loaders.load(f)[1] == "Just a story."
