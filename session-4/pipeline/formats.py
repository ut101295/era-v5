"""
Stage 3 — Format discipline / the ghost-tag trap.

Conversation data arrives from many sources, each marking who is speaking in its
own way. Written into the text as ordinary characters, those markers become "ghost
tags": during pretraining the model learns them as if they were real language, and
during SFT they collide with the tokenizer's real special tokens.

This module does two things:

  1. `render_as_source(pair, i)` — takes one real instruction/response pair and
     writes it in ONE of four incompatible real-world formats, simulating the
     "four sources, four formats" situation from our own audit.

  2. `unify(text)` — detects any of those literal markers and rewrites the turn
     into ONE canonical special-token format, reporting which ghost markers were
     found. `scan_ghost_markers(text)` runs the same detector over pretraining
     text to catch conversation markers that leaked into the pretraining shards.
"""
import re

# --- the four incompatible source formats (literal markers baked into text) ---
FORMATS = ["llama", "chatml_literal", "alpaca", "gemma"]


def render_as_source(instruction, response, i):
    fmt = FORMATS[i % len(FORMATS)]
    if fmt == "llama":
        return fmt, f"[INST] {instruction} [/INST] {response}"
    if fmt == "chatml_literal":
        return fmt, f"<|user|>\n{instruction}\n<|assistant|>\n{response}"
    if fmt == "alpaca":
        return fmt, f"### Human:\n{instruction}\n\n### Assistant:\n{response}"
    # gemma
    return fmt, (f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
                 f"<start_of_turn>model\n{response}<end_of_turn>")


# --- one detector per ghost-marker family ------------------------------------
GHOST_PATTERNS = {
    "llama_inst": re.compile(r"\[/?INST\]"),
    "chatml_literal": re.compile(r"<\|(?:user|assistant|system|im_start|im_end)\|>"),
    "alpaca_header": re.compile(r"###\s*(?:Human|Assistant|Instruction|Response|Input)\s*:?",
                                re.IGNORECASE),
    "gemma_turn": re.compile(r"<(?:start|end)_of_turn>"),
    "openai_role": re.compile(r"<\|(?:endoftext|startoftext)\|>"),
}

# The single canonical format we rewrite everything into (real special tokens).
CANON_USER = "<|im_start|>user"
CANON_ASST = "<|im_start|>assistant"
CANON_END = "<|im_end|>"


def scan_ghost_markers(text):
    """Return {family: count} of ghost markers present in `text` (0s omitted)."""
    found = {}
    for name, pat in GHOST_PATTERNS.items():
        n = len(pat.findall(text))
        if n:
            found[name] = n
    return found


def unify(instruction, response):
    """Emit the canonical special-token conversation for a pair."""
    return (f"{CANON_USER}\n{instruction}{CANON_END}\n"
            f"{CANON_ASST}\n{response}{CANON_END}")


def new_stats():
    return {
        "conversations": 0,
        "sources": {f: 0 for f in FORMATS},
        "ghost_markers_found": 0,
        "ghost_marker_families": {},
        "unified": 0,
        # ghost markers that leaked into the *pretraining* corpus:
        "pretrain_docs_with_ghosts": 0,
        "pretrain_ghost_markers": 0,
    }
