"""Build loss_harness.ipynb from harness.py by splitting on its section separators.

Keeps one source of truth (harness.py) and emits a clean multi-cell notebook:
each `# ---`/`# ===` separator block becomes a markdown header, the code under it
becomes a code cell. Run:  python build_notebook.py
"""
import json, re

src = open("harness.py").read()

# 1. module docstring -> intro markdown
m = re.search(r'^"""(.*?)"""', src, re.S)
intro = m.group(1).strip()
body = src[m.end():].lstrip("\n")

SEP = re.compile(r"^#\s*[-=]{10,}\s*$")           # a pure decorative separator line
REQ = re.compile(r"^#\s*---\s*\(\d+\)")            # a numbered requirement header, e.g. "# --- (3) ..."

lines = body.split("\n")
cells = [("md", "# Session 9 — the loss harness, made correct and observable\n\n"
               + intro
               + "\n\n> Run top to bottom. Every section is numbered to match the assignment. "
                 "The whole point: **print the strings, not the ids** — the few lines between "
                 "model output and the scalar loss are where silent bugs live.")]

i = 0
pending_md = None
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


while i < len(lines):
    line = lines[i]
    if SEP.match(line):                 # decorative separator: header lives just below it
        flush_code()
        i, md = grab_comment_block(i + 1)
        if md:
            cells.append(("md", "## " + md if "\n" not in md else "## " + md.split(chr(10))[0]
                          + "\n\n" + "\n".join(md.split(chr(10))[1:])))
        continue
    if REQ.match(line):                 # numbered requirement header (has trailing text)
        flush_code()
        i, md = grab_comment_block(i)
        first, *rest = md.split("\n")
        first = re.sub(r"^---\s*", "", first)       # drop the "--- " drawing prefix
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

with open("loss_harness.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"wrote loss_harness.ipynb with {len(cells)} cells "
      f"({sum(t=='code' for t,_ in cells)} code, {sum(t=='md' for t,_ in cells)} md)")
