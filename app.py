"""Gradio UI: python app.py  (http://127.0.0.1:7860)"""
import logging

import gradio as gr

from ragscript import genre as genre_pack
from ragscript.config import DURATIONS
from ragscript.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
pl = Pipeline()


def do_ingest():
    return str(pl.ingest())


def do_hooks(brief, genre, style):
    return gr.Radio(choices=pl.hooks(brief, genre, style), label="Pick a hook", value=None)


def do_write(brief, genre, minutes, style, hook, progress=gr.Progress()):
    if not hook:
        raise gr.Error("Generate hooks and pick one first.")
    r = pl.write(brief, genre, int(minutes), hook, style, progress=lambda m: progress(0, desc=m))
    return r["markdown"], str(r["path"])


with gr.Blocks(title="Story Script RAG") as demo:
    gr.Markdown("## Story Script RAG (local)")
    with gr.Row():
        genre = gr.Dropdown(genre_pack.known(pl.cfg), value="horror", allow_custom_value=True,
                            label="Genre (pick one or type any genre)")
        minutes = gr.Dropdown(DURATIONS, value=10, label="Length (min)")
    brief = gr.Textbox(lines=3, label="Brief")
    style = gr.Textbox(label="Style notes (optional)", placeholder="e.g. dry humour, very short sentences")
    with gr.Row():
        b_ing = gr.Button("Ingest data/")
        b_hook = gr.Button("1. Get 3 hooks")
        b_write = gr.Button("2. Write script", variant="primary")
    ing_out = gr.Textbox(label="Collection sizes", interactive=False)
    hook = gr.Radio(choices=[], label="Pick a hook")
    out = gr.Markdown()
    path = gr.Textbox(label="Saved to", interactive=False)
    b_ing.click(do_ingest, outputs=ing_out)
    b_hook.click(do_hooks, [brief, genre, style], hook)
    b_write.click(do_write, [brief, genre, minutes, style, hook], [out, path])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1")
