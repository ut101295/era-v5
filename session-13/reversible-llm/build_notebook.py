"""Build reversible_lab.ipynb from reversible_lab.py by splitting on its section separators.

One source of truth (reversible_lab.py). Each `# ===` separator block becomes a markdown
header; the code under it becomes a code cell. Run:  python build_notebook.py
"""
import json
import re

src = open("reversible_lab.py").read()

m = re.search(r'^"""(.*?)"""', src, re.S)
intro = m.group(1).strip()
body = src[m.end():].lstrip("\n")

SEP = re.compile(r"^#\s*[-=]{10,}\s*$")

lines = body.split("\n")
cells = [("md", "# Session 13 — a reversible LLM, trained three ways\n\n"
               + intro
               + "\n\n> Run top to bottom. We verify the reversible backward against ordinary "
                 "autograd, measure activation memory against depth, and train a 20M-parameter "
                 "model as a standard stack, as a reversible (midpoint) stack, and as a "
                 "reversible stack at a larger batch — reporting final loss, throughput and "
                 "peak memory for each.\n\n> `FULL = False` runs a short local verification; "
                 "set `FULL = True` for the assignment's 20M-params / 50M-tokens run on a GPU.")]

buf = []


def flush_code():
    global buf
    code = "\n".join(buf).strip("\n")
    if code:
        cells.append(("code", code))
    buf = []


def grab_comment_block(start):
    j, comment = start, []
    while j < len(lines) and lines[j].lstrip().startswith("#"):
        if not SEP.match(lines[j]):
            comment.append(re.sub(r"^#\s?", "", lines[j]).rstrip())
        j += 1
    return j, "\n".join(comment).strip()


i = 0
while i < len(lines):
    line = lines[i]
    if SEP.match(line):
        flush_code()
        i, md = grab_comment_block(i + 1)
        if md:
            first, *rest = md.split("\n")
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

with open("reversible_lab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"wrote reversible_lab.ipynb with {len(cells)} cells "
      f"({sum(t=='code' for t,_ in cells)} code, {sum(t=='md' for t,_ in cells)} md)")
