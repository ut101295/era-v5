"""
Stage 2 — Text normalization (the sovereign thread at the character level).

Put every document into one canonical form BEFORE anything else touches it:
  - unescape HTML entities (&amp; -> &)
  - Unicode NFC (one code-point sequence per visible character)
  - strip invisible NOISE (zero-width space, BOM, bidi overrides, replacement char…)
  - strip C0/C1 control & format characters
  - collapse runs of whitespace

The one subtlety that makes this an *Indic* cleaner and not a generic English one:
the zero-width non-joiner (U+200C) and zero-width joiner (U+200D) are LEGITIMATE in
Brahmic scripts and carry real linguistic meaning, so they are kept while every other
invisible character is removed. A cleaner that strips all invisibles mangles Indic
text while believing it is helping.

All special characters below are written as \\u escapes on purpose — invisible
literals in source are unreadable and easy to corrupt.
"""
import html
import re
import unicodedata

ZWNJ = "‌"  # zero-width non-joiner  — KEEP (Brahmic)
ZWJ = "‍"   # zero-width joiner      — KEEP (Brahmic)
KEEP = {ZWNJ, ZWJ, "\n", "\t"}

# Invisible / meaningless characters that are pure noise and must go.
NOISE = {
    "​",  # zero-width space
    "﻿",  # byte-order mark / zero-width no-break space
    "⁠",  # word joiner
    "᠎",  # mongolian vowel separator
    "‎", "‏",  # LRM / RLM
    "‪", "‫", "‬", "‭", "‮",  # bidi embeddings/overrides
    "⁦", "⁧", "⁨", "⁩",  # bidi isolates
    "�",  # replacement character (mojibake tombstone)
    "­",  # soft hyphen
}

_entity = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]{1,31}|#\d{1,7}|#[xX][0-9a-fA-F]{1,6});")
# space, tab, NBSP, thin/en/em spaces (U+2000–200A), narrow NBSP, ideographic space
_ws = re.compile(r"[ \t  -   　]+")
_nl = re.compile(r"\n{3,}")


def normalize(text, stats):
    """Return cleaned text; accumulate character-level counts into `stats`."""
    stats["docs"] += 1

    # 1. HTML entities -> real characters
    n_ent = len(_entity.findall(text))
    if n_ent:
        text = html.unescape(text)
        stats["html_entities"] += n_ent

    # 2. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 3. remove noise + control/format chars, but keep the Brahmic joiners
    out = []
    for ch in text:
        if ch in KEEP:
            if ch in (ZWNJ, ZWJ):
                stats["joiners_kept"] += 1
            out.append(ch)
            continue
        if ch in NOISE:
            stats["noise_removed"] += 1
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf", "Co", "Cs"):
            stats["control_removed"] += 1
            continue
        out.append(ch)
    text = "".join(out)

    # 4. collapse whitespace
    before = len(text)
    text = _ws.sub(" ", text)
    text = _nl.sub("\n\n", text)
    text = text.strip()
    stats["ws_chars_collapsed"] += max(0, before - len(text))

    return text


def new_stats():
    return {
        "docs": 0,
        "html_entities": 0,
        "noise_removed": 0,
        "control_removed": 0,
        "joiners_kept": 0,
        "ws_chars_collapsed": 0,
    }
