"""Build zero_sim.ipynb from zero_sim.py by splitting on its section separators.

One source of truth (zero_sim.py). Each `# ===`/`# ---` separator block becomes a
markdown header; the code under it becomes a code cell. Run:  python build_notebook.py
"""
import json, re

src = open("zero_sim.py").read()

m = re.search(r'^"""(.*?)"""', src, re.S)
intro = m.group(1).strip()
body = src[m.end():].lstrip("\n")

SEP = re.compile(r"^#\s*[-=]{10,}\s*$")
REQ = re.compile(r"^#\s*---\s*\(\d+\)")

lines = body.split("\n")
cells = [("md", "# Session 12 — ZeRO on 32 virtual GPUs, simulated and checked\n\n"
               + intro
               + "\n\n> Run top to bottom. We stand up 32 virtual GPUs, measure the memory "
                 "each one holds under DP / ZeRO-1 / ZeRO-2 / ZeRO-3, **prove** every stage "
                 "computes the identical weight, and train a real model under full sharding "
                 "(ZeRO-3) to show the loss fall.")]

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
            comment.append(re.sub(r"^#\s?", "", lines[j]).rstrip("- "))
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
    if REQ.match(line):
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

with open("zero_sim.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"wrote zero_sim.ipynb with {len(cells)} cells "
      f"({sum(t=='code' for t,_ in cells)} code, {sum(t=='md' for t,_ in cells)} md)")
