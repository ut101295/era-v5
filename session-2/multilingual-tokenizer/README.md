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
We configure the Hugging Face `ByteLevel` pre-tokenizer with `use_regex=False`. This maps bytes to visual Unicode characters (allowing perfect roundtrip decoding) but **bypasses the Latin-centric word-splitting regex**. We then sequence it with a clean `Whitespace()` splitter.

```python
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer, Sequence, Whitespace

tokenizer.pre_tokenizer = Sequence([
    Whitespace(),
    ByteLevelPreTokenizer(add_prefix_space=False, use_regex=False)
])
```
This ensures spaces are mapped to `Ġ` for roundtrip verification, but Indic word structures are preserved intact for the BPE training algorithm.

---

## ⚙️ Optimization Pipeline (`train.py`)

The optimization script performs an iterative coordinate descent on language corpus weights to balance the compression ratios:

1.  **Data Generation**:
    Downloads Wikipedia text for each page, stripping HTML and formatting.
2.  **Weighted Corpus Assembly**:
    Multiplies each language's text corpus by its current weight ($w_i$) to simulate frequency during BPE training. Higher weight gives that language a larger share of the 10,000 vocabulary merge slots.
3.  **BPE Training**:
    Trains the BPE model on the combined weighted text.
4.  **Ratio Evaluation**:
    Computes compression ratio for each language:
    $$X_i = \frac{\text{Number of BPE Tokens}}{\text{Number of Whitespace Words}}$$
5.  **Dynamic Weight Adjustment**:
    If the English ratio $X_1$ exceeds $1.2$, the English corpus weight $w_1$ is multiplied by $1.5$ in the next iteration to allocate more merges to English.
    The weights of the other languages are scaled to minimize the spread:
    $$\Delta w_i = \eta \cdot (X_i - \bar{X})$$

---

## 📊 Final Optimization Results

| Language | Words | BPE Tokens | Ratio (X) | Unique Vocab Tokens |
| :--- | :---: | :---: | :---: | :---: |
| **English ($X_1$)** | 9,963 | 11,528 | **1.1571** | 4,271 |
| **Hindi ($X_2$)** | 7,970 | 14,657 | **1.8390** | 1,651 |
| **Telugu ($X_3$)** | 2,391 | 4,722 | **1.9749** | 2,104 |
| **Marathi ($X_4$)** | 4,496 | 8,557 | **1.9032** | 2,108 |

### **Summary Stats**:
*   **English Ratio ($X_1$)**: $1.1571 \le 1.2$ ✅ (Strict Constraint Met)
*   **Min Ratio**: $1.1571$ (English)
*   **Max Ratio**: $1.9749$ (Telugu)
*   **Spread**: $0.8178$
*   **Assignment Score**: **`1,222.76`** 🎯

All roundtrip decodes are 100% verified (spaces, matras, and special characters are preserved exactly).

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
