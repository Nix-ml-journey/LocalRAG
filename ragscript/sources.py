"""Bookkeeping for the chunks one script was written from."""


class SourceLedger:
    def __init__(self):
        self.ids = {}          # chunk id -> S number
        self.refs = []         # every chunk text shown to the model, for the source-overlap check
        self.cited_text = {}   # S number -> chunk text
        self.style_used = set()

    def cite(self, chunks) -> str:
        """Number the chunks and return them as prompt text: [S3] (file) text."""
        lines = []
        for c in chunks:
            n = self.ids.setdefault(c["id"], len(self.ids) + 1)
            self.refs.append(c["text"])
            self.cited_text[n] = c["text"]
            lines.append(f"[S{n}] ({c['meta'].get('source')}) {c['text']}")
        return "\n\n".join(lines)

    def add_style(self, chunks):
        """Style examples are sources too: listed in the output, never cited in the prose."""
        for c in chunks:
            self.style_used.add(c["meta"]["source"])
            self.refs.append(c["text"])

    def used(self) -> dict:
        """{source file: [S numbers]}; style-only sources map to []."""
        out = {}
        for cid, n in self.ids.items():
            out.setdefault(cid.split(":", 2)[1], []).append(n)  # id = "<collection>:<file>:<chunk no.>"
        for s in sorted(self.style_used):
            out.setdefault(s, [])
        return out
