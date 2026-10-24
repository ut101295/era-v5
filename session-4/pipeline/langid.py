"""
Stage 7 — Language identification & validation.

Don't trust the folder a document came from — web crawls are mislabeled often
enough that a bad label pollutes the per-language pools and skews the fertility
numbers we use to size each language's budget. Detect the language of each document
at runtime and flag any mismatch against the claimed label.

Detection here is SCRIPT-BASED (no fasttext build): each Unicode block maps to a
language/family. The subtle case the session calls out is Assamese vs Bengali —
they SHARE the Bengali script block, so a pure block check cannot tell them apart.
We disambiguate on the letters that actually differ:

    Assamese uses  ৰ (U+09F0)  and  ৱ (U+09F1)
    Bengali  uses  র (U+09B0)  and  ব (U+09AC)  in those roles

That is exactly why the 300 injected Bengali docs — sitting in the Assamese folder
— light up as mismatches instead of silently polluting the Assamese pool.
"""
import unicodedata

# script block -> coarse language/label
_BLOCKS = [
    ("bengali", 0x0980, 0x09FF),      # Assamese + Bengali (disambiguated below)
    ("devanagari", 0x0900, 0x097F),   # Hindi / Marathi / Nepali / Sanskrit …
    ("gujarati", 0x0A80, 0x0AFF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("arabic", 0x0600, 0x06FF),       # Urdu / Kashmiri / Sindhi
    ("latin", 0x0041, 0x024F),        # English / romanized
]

ASSAMESE_MARKS = ("ৰ", "ৱ")   # U+09F0, U+09F1
BENGALI_MARKS = ("র", "ব")    # U+09B0, U+09AC


def _dominant_script(text):
    counts = {}
    total = 0
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        total += 1
        cp = ord(ch)
        for name, lo, hi in _BLOCKS:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if total == 0:
        return "unknown", 0.0, counts, total
    top = max(counts, key=counts.get)
    return top, counts[top] / total, counts, total


def detect(text):
    """Return (lang_code, confidence, note)."""
    script, conf, counts, total = _dominant_script(text)
    note = ""
    if script == "bengali":
        # ব and র exist in BOTH scripts, so counting them proves nothing. The real
        # signal is the Assamese-only letters ৰ/ৱ, which Bengali never uses. Their
        # presence => Assamese; their absence (within the Bengali block) => Bengali.
        asm = sum(text.count(m) for m in ASSAMESE_MARKS)
        if asm > 0:
            lang = "asm"
        else:
            lang = "ben"
            note = "no_assamese_marks"
    else:
        lang = {
            "devanagari": "hin", "gujarati": "guj", "gurmukhi": "pan",
            "oriya": "ory", "tamil": "tam", "telugu": "tel", "kannada": "kan",
            "malayalam": "mal", "arabic": "urd", "latin": "eng", "unknown": "und",
        }.get(script, "und")

    # code-switching: a large secondary script share
    if len(counts) >= 2:
        ordered = sorted(counts.values(), reverse=True)
        if total > 0 and ordered[1] / total > 0.30:
            note = (note + ";" if note else "") + "code_switched"
    return lang, round(conf, 3), note


# The V4 language-code bug, guarded: a 2-char code used where 3-char was expected,
# which "worked" only because a fallback happened to return the right value.
_ISO2_TO_ISO3 = {"as": "asm", "bn": "ben", "hi": "hin", "te": "tel",
                 "ta": "tam", "mr": "mar", "gu": "guj", "kn": "kan"}


def normalize_lang_code(code):
    """Fail loud instead of silently falling back (the fix for the Telugu bug)."""
    if len(code) == 3:
        return code
    if code in _ISO2_TO_ISO3:
        return _ISO2_TO_ISO3[code]
    raise ValueError(f"unmappable language code: {code!r}")


def new_stats():
    return {
        "docs_in": 0,
        "detected": {},          # lang -> count
        "mismatches": 0,         # claimed != detected
        "mismatch_examples": [],
        "code_switched": 0,
        "removed_mismatch": 0,
        "removed_injected": 0,   # of those, the deliberately-mislabeled Bengali
        "removed_genuine_fp": 0, # genuine Assamese misflagged (the langid FP cost)
    }
