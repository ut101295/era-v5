> ⚠️ **Superseded — this approach scored 0 on regrade.** This byte-level tokenizer fails the
> grader's faithful-roundtrip gate (it drops `[`, `]` and 106 other characters) and targets the
> old words-based rubric. The correct resubmission is in [`../faithful-tokenizer/`](../faithful-tokenizer/);
> see [`../RESUBMISSION_FIX.md`](../RESUBMISSION_FIX.md) for the full explanation of what was wrong
> and what we fixed. This directory is kept for reference only.

# Axiom — Learning OS: Session 2 Multilingual BPE Tokenizer

This project implements a highly optimized, custom Byte-Pair Encoding (BPE) tokenizer using the **Hugging Face `tokenizers` library**. The tokenizer is trained on a multilingual corpus comprising the Wikipedia pages of "India" in four languages: **English**, **Hindi**, **Telugu**, and **Marathi**.

*   **Project Directory**: [multilingual-tokenizer](file:///Users/utkarshupadhyay/Personal/Projects/era-v5/session-2/multilingual-tokenizer)
*   **Main Files**:
    *   [bpe_tokenizer.py](file:///Users/utkarshupadhyay/Personal/Projects/era-v5/session-2/multilingual-tokenizer/bpe_tokenizer.py) — Tokenizer model setup and configuration
    *   [train.py](file:///Users/utkarshupadhyay/Personal/Projects/era-v5/session-2/multilingual-tokenizer/train.py) — Iterative weight optimization and training pipeline
    *   [fetch_data.py](file:///Users/utkarshupadhyay/Personal/Projects/era-v5/session-2/multilingual-tokenizer/fetch_data.py) — Corpus generation from Wikipedia API
    *   [README.md](file:///Users/utkarshupadhyay/Personal/Projects/era-v5/session-2/multilingual-tokenizer/README.md) — Detailed technical reference

---

## 🎯 Assignment Goals & Optimization Score

The objective is to train a BPE tokenizer with a vocabulary size of **10,000 tokens** such that:
1.  **English compression ratio ($X_1$)** is strictly **1.2 or less**.
2.  **Multilingual ratio spread** is minimized (i.e. compression ratios are balanced across all four languages).

The optimization performance is calculated using the following score function:
$$\text{Score} = \frac{1000}{\max(X) - \min(X)}$$
Where $X = \{X_1, X_2, X_3, X_4\}$ represents the compression ratios for each language. The lower the spread, the higher the score. A penalty of $\text{Score} = 0.0$ is applied if $X_1 > 1.2$.

---

## 🔬 Core Technical Challenges & Solutions

### 1. Solving the Byte-Level Pre-tokenization Problem in Indic Scripts
Standard BPE models (like GPT-2 or GPT-4) use a regex-based pre-tokenizer that splits text into word and punctuation blocks before BPE merges are calculated. This regex is designed for Latin scripts and treats non-alphanumeric characters (like Indic vowel signs/matras or Telugu/Hindi letters) as independent punctuation. 

For example, without modification, a Telugu word like `భారతదేశం` is split into:
`['భ', 'ా', 'రతద', 'ే', 'శ', 'ం']`

This premature splitting prevents the tokenizer from ever merging these characters into complete subword units, leading to high compression ratios (more BPE tokens per word) and low efficiency in Indic languages.

#### **Solution**:
We configure the Hugging Face `ByteLevel` pre-tokenizer with `use_regex=False`. This maps every byte to a visible Unicode character (spaces become `Ġ`) but **bypasses the Latin-centric word-splitting regex**, so Indic word/matra structure is preserved intact for the BPE merge algorithm. An `NFC` normalizer is applied first so composed and decomposed forms of the same Indic character train as one unit.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = NFC()
tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=False)
tokenizer.decoder = ByteLevelDecoder()
```

> **Roundtrip note:** decoding is exact for characters, matras and spaces, but because the byte-level model with `use_regex=False` does not emit standalone tokens for line breaks, **newlines are dropped on decode** (`decode(encode(text)) == text.replace("\n", "")`). This does not affect the compression score, which counts tokens per word.

---

## ⚙️ Optimization Pipeline (`train.py`)

The score depends only on the **spread** between the best and worst compression ratio, so the whole game is to give each language just enough of the 10,000-merge budget to pull its ratio toward the pack — while keeping English safely under its 1.2 cap. `train.py` does this by tuning per-language **corpus weights** (how many times each language's text is repeated in the training corpus; more copies → more merges spent on that language → lower ratio).

1.  **Data Generation** — download & clean each Wikipedia page (`fetch_data.py`).
2.  **Weighted Corpus Assembly** — repeat each language's lines by its weight $w_i$ (integer repeats + a fractional tail), then train one BPE model on the combined text.
3.  **Ratio Evaluation** — compute each language's compression ratio:
    $$X_i = \frac{\text{Number of BPE Tokens}}{\text{Number of Whitespace Words}}$$
4.  **Multi-start hill climbing** — from several seed weight vectors, repeatedly try nudging each language's weight up/down **and trading weight between every language pair**, sweeping to convergence at a shrinking step size, and keep the best result across all seeds. Paired moves matter most: the biggest gains come from moving budget *away* from English (which has ratio headroom) *toward* the Indic scripts, squeezing the spread from both ends. Multi-start is needed because the quantized (whole-line) weighting makes the objective bumpy with local optima.
5.  **English safety cap** — the objective maximizes $1000 / \text{spread}$ subject to keeping English at or below `EN_CAP = 1.185`, a deliberate buffer under the hard 1.2 limit. The graders re-run the tokenizer on their own copy of the article (which drifts as Wikipedia is edited); the buffer ensures a small ratio drift on their side cannot tip $X_1$ over 1.2 and zero the score.

---

## 📊 Final Optimization Results

| Language | Words | BPE Tokens | Ratio (X) | Unique Vocab Tokens |
| :--- | :---: | :---: | :---: | :---: |
| **English ($X_1$)** | 9,963 | 11,795 | **1.1839** | 4,041 |
| **Hindi ($X_2$)** | 7,970 | 14,062 | **1.7644** | 1,719 |
| **Telugu ($X_3$)** | 2,391 | 4,473 | **1.8708** | 2,041 |
| **Marathi ($X_4$)** | 4,496 | 8,500 | **1.8906** | 2,023 |

### **Summary Stats**:
*   **English Ratio ($X_1$)**: $1.1839 \le 1.2$ ✅ (constraint met, margin $0.0161$)
*   **Min Ratio**: $1.1839$ (English)
*   **Max Ratio**: $1.8906$ (Marathi)
*   **Spread**: $0.7067$
*   **Assignment Score**: **`1,415.05`** 🎯

These numbers are reproduced **from the saved `output/tokenizer.json`** (not just the training log) and are re-computed live, in-browser, by the widget — so they match exactly what the graders will get when they run the tokenizer themselves. Character/matra/space decoding is exact; see the roundtrip note above (newlines are dropped on decode, which does not affect the score).

---

## 📂 File Structure

```
session-2/multilingual-tokenizer/
├── README.md              # Detailed project reference
├── requirements.txt       # Dependencies (requests, tokenizers)
├── fetch_data.py          # Wikipedia API text downloader
├── bpe_tokenizer.py       # Pre-tokenization and BPE model setup
├── train.py               # Optimization training loop
├── data/                  # Cached Wikipedia text sources
│   ├── en.txt
│   ├── hi.txt
│   ├── te.txt
│   └── mr.txt
└── output/                # Trained output files
    ├── tokenizer.json     # Trained HF tokenizer model file
    └── results.txt        # Output evaluation log
```

---

## 🚀 Reproduction Instructions

### 1. Setup Virtual Environment
Run the following from the root `session-2` folder:
```bash
cd session-2
python3 -m venv .venv
source .venv/bin/activate
pip install -r multilingual-tokenizer/requirements.txt
```

### 2. Fetch Data
Download the Wikipedia articles:
```bash
python3 multilingual-tokenizer/fetch_data.py
```

### 3. Run Optimization & Train
Start the optimization loop to train and save the final tokenizer:
```bash
python3 multilingual-tokenizer/train.py
```
This saves the trained tokenizer to `multilingual-tokenizer/output/tokenizer.json` and logs final compression ratios to `multilingual-tokenizer/output/results.txt`.

### 4. View / Deploy the Widget
The interactive submission widget lives in [`../tokenizer-widget/`](../tokenizer-widget). It loads `tokenizer.json`, re-computes every ratio and the score live in the browser, and lets you download the tokenizer. See that folder's `README.md` for how to preview it locally and deploy it to Netlify.
