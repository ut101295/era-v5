"""
fetch_data.py — pull the raw material for the Session 4 cleaning demo.

Three sources, all from the AI4Bharat family we surfaced in Session 3:

  1. PRIMARY  IndicCorpV2 / asm_Beng   — Assamese web text (the corpus we clean).
  2. INJECT   IndicCorpV2 / ben_Beng   — a handful of Bengali docs deliberately
              filed under the Assamese folder, so the language-ID stage has a real
              "the folder says Assamese but it's really Bengali" case to catch
              (Assamese and Bengali share the Bengali script but differ by ৰ/ৱ).
  3. CONV     indic-align / Dolly_T    — real Indic instruction pairs, used by the
              ghost-tag stage to show four conversation formats collapsing into one.

Everything is streamed (never fully downloaded) and capped so the raw corpus lands
in the assignment's 10–100M-token band while staying runnable on a laptop. Output
is deterministic JSONL under data/ (gitignored — regenerate with this script).
"""
import json
import os
import re
import sys
import time

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

# --- sizing knobs (keep the raw corpus in the 10–100M token band) -------------
TARGET_TOKENS = 10_000_000   # stop once the raw Assamese corpus reaches this
MAX_ROWS      = 400_000      # hard safety cap on streamed rows (bounds wall-clock)
N_INJECT      = 300          # Bengali docs mislabeled as Assamese
N_CONV        = 2_000        # Indic instruction pairs for the ghost-tag demo

_tok = re.compile(r"\w+|[^\w\s]", re.UNICODE)
def count_tokens(text: str) -> int:
    return len(_tok.findall(text))


def fetch_primary():
    """Stream Assamese web text until we hit the token target or the row cap."""
    print(f"[primary] streaming IndicCorpV2/asm_Beng "
          f"(target {TARGET_TOKENS:,} tokens, max {MAX_ROWS:,} rows)…")
    ds = load_dataset("ai4bharat/IndicCorpV2", streaming=True, split="asm_Beng")
    out = os.path.join(DATA, "raw_asm.jsonl")
    rows = tokens = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            rec = {"text": text, "claimed_lang": "asm",
                   "source": "IndicCorpV2/asm_Beng", "doc_id": f"asm-{rows}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1
            tokens += count_tokens(text)
            if rows % 25_000 == 0:
                print(f"  … {rows:,} rows, ~{tokens:,} tokens, "
                      f"{rows/(time.time()-t0):.0f} rows/s")
            if tokens >= TARGET_TOKENS or rows >= MAX_ROWS:
                break
    print(f"[primary] wrote {rows:,} rows, ~{tokens:,} tokens -> {out} "
          f"({time.time()-t0:.0f}s)")
    return rows, tokens


def fetch_inject():
    """A few Bengali docs, deliberately stamped claimed_lang='asm'."""
    print(f"[inject] streaming IndicCorpV2/ben_Beng ({N_INJECT} mislabeled docs)…")
    ds = load_dataset("ai4bharat/IndicCorpV2", streaming=True, split="ben_Beng")
    out = os.path.join(DATA, "raw_inject_ben_as_asm.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if not text.strip():
                continue
            rec = {"text": text, "claimed_lang": "asm",  # <-- the lie we will catch
                   "source": "IndicCorpV2/ben_Beng-injected", "doc_id": f"inj-{n}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n >= N_INJECT:
                break
    print(f"[inject] wrote {n} Bengali docs mislabeled as Assamese -> {out}")
    return n


def fetch_conversations():
    """Real Indic instruction pairs (Hindi) for the ghost-tag / format stage."""
    print(f"[conv] streaming indic-align/Dolly_T ({N_CONV} Hindi pairs)…")
    ds = load_dataset("ai4bharat/indic-align", "Dolly_T", streaming=True, split="train")
    out = os.path.join(DATA, "raw_conversations.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            pairs = row.get("hin_Deva") or []
            if not pairs or not pairs[0] or len(pairs[0]) < 2:
                continue
            instruction, response = pairs[0][0], pairs[0][1]
            if not instruction or not response:
                continue
            rec = {"instruction": instruction, "response": response,
                   "lang": "hin", "doc_id": f"conv-{n}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n >= N_CONV:
                break
    print(f"[conv] wrote {n} instruction pairs -> {out}")
    return n


def main():
    summary = {}
    rows, tokens = fetch_primary()
    summary["primary_rows"] = rows
    summary["primary_tokens"] = tokens
    summary["inject_rows"] = fetch_inject()
    summary["conv_rows"] = fetch_conversations()
    with open(os.path.join(DATA, "fetch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[done] fetch summary:", summary)


if __name__ == "__main__":
    main()
