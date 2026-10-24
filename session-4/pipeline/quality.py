"""
Stage 4 — Quality filtering (heuristic cascade + edu-value gate), script-aware.

Two layers:
  Layer 1 — a chain of cheap heuristic rules (Gopher/C4 family) that catch
            obviously-broken documents.
  Layer 2 — a light "educational value" score standing in for the FineWeb-Edu
            classifier (a real deployment trains a small model on LLM labels; we
            use a deterministic proxy so the pipeline stays dependency-free).

The sovereign-thread point: rules tuned on English quietly reject good low-resource
text. Every rule here is SCRIPT-AWARE — it measures characters in the document's
own script and treats the Indic danda (।) as sentence-ending punctuation. To make
the bias visible we ALSO run an English-tuned variant and count how many good Indic
docs it *would* have wrongly dropped (`en_tuned_would_drop`).
"""
import re
import unicodedata

# Unicode ranges for the scripts we care about.
_BENGALI = (0x0980, 0x09FF)      # Assamese + Bengali share this block
_DEVANAGARI = (0x0900, 0x097F)
_INDIC_BLOCKS = [(0x0900, 0x0DFF)]  # Devanagari … Sinhala (covers all our langs)

_TERMINAL_PUNCT = tuple("।॥.?!…")   # includes the Indic danda/double-danda
_word = re.compile(r"\S+")


def _script_ratio(text, blocks):
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    hits = sum(1 for c in letters
               if any(lo <= ord(c) <= hi for lo, hi in blocks))
    return hits / len(letters)


def _symbol_ratio(text):
    """Fraction of non-space characters that are punctuation/symbols.

    Critically, Indic COMBINING MARKS (matras, viramas — category M*) count as
    content, not symbols. Treating them as symbols (the naive `not isalnum()`
    test) wrongly flags perfectly good Brahmic text as a symbol wall — exactly the
    filter bias this stage is meant to avoid.
    """
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return 1.0
    sym = sum(1 for c in nonspace if unicodedata.category(c)[0] in ("P", "S"))
    return sym / len(nonspace)


def _dup_line_fraction(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def heuristic_rules(text, indic=True):
    """Return (passed: bool, failed_rule: str|None). Script-aware when indic=True."""
    words = _word.findall(text)
    n = len(words)

    # R1 — minimum length (kills empties and stubs)
    if n < 3:
        return False, "too_short"

    # R2 — mean word length in a sane band
    mean_wl = sum(len(w) for w in words) / n
    if not (1.5 <= mean_wl <= 30):
        return False, "mean_word_len"

    # R3 — symbol-to-char ratio (spam walls / markup dumps)
    if _symbol_ratio(text) > 0.30:
        return False, "symbol_ratio"

    # R4 — duplicate-line fraction (boilerplate / repeated lines)
    if _dup_line_fraction(text) > 0.50:
        return False, "dup_lines"

    # R5 — script purity. Script-aware: an Indic doc must be mostly Indic script;
    #      the English-tuned variant instead demands mostly Latin, which is the bug.
    if indic:
        if _script_ratio(text, _INDIC_BLOCKS) < 0.50:
            return False, "not_target_script"
    else:
        latin = _script_ratio(text, [(0x0041, 0x024F)])
        if latin < 0.50:
            return False, "not_latin"   # <-- wrongly fires on good Indic text

    # R6 — line-ending punctuation (real prose ends lines in terminal punctuation)
    #      Script-aware: the Indic danda counts. English-tuned only accepts ASCII.
    puncts = _TERMINAL_PUNCT if indic else tuple(".?!")
    if not text.rstrip().endswith(puncts):
        # soft rule — only fails if ALSO very short (avoid over-penalising snippets)
        if n < 8:
            return False, "no_terminal_punct"

    return True, None


def edu_score(text):
    """Deterministic 0–1 proxy for the FineWeb-Edu classifier."""
    words = _word.findall(text)
    n = len(words)
    if n == 0:
        return 0.0
    lines = [ln for ln in text.splitlines() if ln.strip()]
    avg_line_words = n / max(1, len(lines))
    uniq = len(set(words)) / n                      # lexical variety
    length_bonus = min(1.0, n / 40)                 # rewards substantive length
    sentence_bonus = min(1.0, avg_line_words / 12)  # rewards prose over lists
    score = 0.45 * uniq + 0.30 * length_bonus + 0.25 * sentence_bonus
    return round(min(1.0, score), 4)


EDU_THRESHOLD = 0.25


def new_stats():
    return {
        "docs_in": 0,
        "dropped_heuristic": 0,
        "dropped_by_rule": {},
        "dropped_edu": 0,
        "kept": 0,
        "edu_score_sum": 0.0,
        # bias probe: good Indic docs an English-tuned filter would have dropped
        "en_tuned_would_drop": 0,
    }
