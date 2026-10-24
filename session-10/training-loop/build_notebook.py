"""Build train_loop.ipynb from train_loop.py by splitting on its section separators.

One source of truth (train_loop.py). Each `# ===`/`# ---` separator block becomes a
markdown header; the code under it becomes a code cell. Run:  python build_notebook.py
"""
import json, re

src = open("train_loop.py").read()

# 1. module docstring -> intro markdown
m = re.search(r'^"""(.*?)"""', src, re.S)
intro = m.group(1).strip()
body = src[m.end():].lstrip("\n")

SEP = re.compile(r"^#\s*[-=]{10,}\s*$")           # a pure decorative separator line
REQ = re.compile(r"^#\s*---\s*\(\d+\)")            # a numbered requirement header

lines = body.split("\n")
cells = [("md", "# Session 10 — the training loop, made to tell the truth\n\n"
               + intro
               + "\n\n> Run top to bottom. Each section is numbered to match the "
                 "assignment. The theme: **a plausible number is not evidence** — "
                 "so we print the shapes, hand-check a gradient, break the "
                 "accumulation on purpose, and watch the grad norm move before the loss.")]

buf = []


def flush_code():
    global buf
    code = "\n".join(buf).strip("\n")
    if code:
        cells.append(("code", code))
    buf = []


def grab_comment_block(start):
    """Collect consecutive comment lines (skipping pure separators) from `start`."""
    j, comment = start, []
    while j < len(lines) and lines[j].lstrip().startswith("#"):
        if not SEP.match(lines[j]):
            comment.append(re.sub(r"^#\s?", "", lines[j]).rstrip("- "))
        j += 1
    return j, "\n".join(comment).strip()


i = 0
while i < len(lines):
    line = lines[i]
    if SEP.match(line):                 # decorative separator: header lives just below it
        flush_code()
        i, md = grab_comment_block(i + 1)
        if md:
            first, *rest = md.split("\n")
            cells.append(("md", "## " + first + ("\n\n" + "\n".join(rest) if rest else "")))
        continue
    if REQ.match(line):                 # numbered requirement header (has trailing text)
        flush_code()
        i, md = grab_comment_block(i)
        first, *rest = md.split("\n")
        first = re.sub(r"^---\s*", "", first)
        cells.append(("md", "## " + first + ("\n\n" + "\n".join(rest) if rest else "")))
        continue
    buf.append(line)
    i += 1
flush_code()

nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": c.splitlines(keepends=True)}
         if t == "md" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
          "source": c.splitlines(keepends=True)})
        for (t, c) in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

with open("train_loop.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"wrote train_loop.ipynb with {len(cells)} cells "
      f"({sum(t=='code' for t,_ in cells)} code, {sum(t=='md' for t,_ in cells)} md)")
