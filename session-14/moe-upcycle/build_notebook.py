"""Build moe_lab.ipynb from moe_lab.py by splitting on its section separators.

One source of truth (moe_lab.py). Each `# ===` separator block becomes a markdown
header; the code under it becomes a code cell. Run:  python build_notebook.py
"""
import json
import re

src = open("moe_lab.py").read()

m = re.search(r'^"""(.*?)"""', src, re.S)
intro = m.group(1).strip()
body = src[m.end():].lstrip("\n")

SEP = re.compile(r"^#\s*[-=]{10,}\s*$")

lines = body.split("\n")
cells = [("md", "# Session 14 — a dense model, upcycled into a Mixture-of-Experts\n\n"
               + intro
               + "\n\n> Run top to bottom. We train a dense model, convert it into an 8-expert "
                 "top-2 MoE by **sparse upcycling**, verify the MoE reproduces the dense output "
                 "at conversion (so loss continues, no jump), keep training it, and — for "
                 "contrast — train the same MoE from scratch under an equal budget.")]

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

with open("moe_lab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"wrote moe_lab.ipynb with {len(cells)} cells "
      f"({sum(t=='code' for t,_ in cells)} code, {sum(t=='md' for t,_ in cells)} md)")
