"""
Stage 8 — PII removal (regex layer + name layer).

Two layers, mirroring quality filtering:
  Layer 1 — regex for structured identifiers that follow predictable patterns:
            emails, phone numbers (incl. Indian +91 forms), IP addresses, and
            long digit runs (Aadhaar-like). These mask cleanly and safely.
  Layer 2 — a name detector for the things that don't follow a fixed pattern.
            With no ML model available we use a small heuristic name pass; its
            purpose is to make the precision/recall tension concrete rather than
            to be production-grade — and specifically to show that an aggressive
            list catches Indic *common words / place names* as false positives.
"""
import re

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Indian & generic phone numbers: +91 / 10-digit / grouped
PHONE = re.compile(r"(?:(?:\+?91[\-\s]?)?[6-9]\d{9})|(?:\+?\d[\d\-\s]{8,13}\d)")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
URL = re.compile(r"https?://[^\s]+")
AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

# A deliberately aggressive Indic "name" list that ALSO contains common words /
# place names, so the false-positive problem shows up in real numbers.
INDIC_NAME_LIKE = [
    "শৰ্মা", "বৰুৱা", "গগৈ", "দাস", "কলিতা",   # surnames
    "ভাৰত", "অসম", "গুৱাহাটী",                 # place names (false-positive risk)
    "কমল", "জ্যোতি",                             # given names that are also common words
]


def scrub(text, stats):
    """Mask structured PII (safe) then run the name pass (lossy). Returns text."""
    stats["docs"] += 1

    def _sub(pat, tag, key):
        nonlocal text
        n = len(pat.findall(text))
        if n:
            text = pat.sub(tag, text)
            stats[key] += n

    _sub(EMAIL, "[EMAIL]", "emails")
    _sub(URL, "[URL]", "urls")
    _sub(AADHAAR, "[ID]", "ids")
    _sub(IPV4, "[IP]", "ips")
    _sub(PHONE, "[PHONE]", "phones")

    # name layer — count masks and flag likely false positives
    for name in INDIC_NAME_LIKE:
        c = text.count(name)
        if c:
            text = text.replace(name, "[NAME]")
            stats["names_masked"] += c
            # place names / common words masked = false positives
            if name in ("ভাৰত", "অসম", "গুৱাহাটী", "কমল", "জ্যোতি"):
                stats["name_false_positives"] += c
    return text


def new_stats():
    return {
        "docs": 0,
        "emails": 0,
        "phones": 0,
        "ips": 0,
        "urls": 0,
        "ids": 0,
        "names_masked": 0,
        "name_false_positives": 0,
    }
