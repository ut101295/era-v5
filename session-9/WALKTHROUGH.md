# Session 9 — Walkthrough

A guided tour of the submission, in the order it's easiest to understand and defend.

## The 60-second version
The deliverable is a **loss harness**: a tiny causal transformer plus the handful of lines
between its output and the scalar loss, with *every one of those lines made observable*. It
proves seven things about cross-entropy next-token training (shapes, the shift, padding masks,
document-boundary masks, perplexity, weight tying, chunked-CE memory), then adds a second head
that predicts token t+2 and shows why the far horizon stays harder. Everything is generated
from one file (`harness.py`) into one notebook that runs top to bottom with outputs embedded.

## 1. Why this exists — print the strings
The assignment's own warning: a target shift in the *wrong* direction still gives a beautiful,
falling loss curve. So the harness never asks you to trust the loss. Requirement 2 prints the
**input word beside its target word** (`the→cat`, `cat→ran`, …) and asserts `target[i] ==
input[i+1]`. If the shift were flipped, the column would read as nonsense even while the loss
looked perfect. This is the single most important habit in the whole session.

## 2. The model — honest, not big
`harness.py` builds a ~105k-param causal transformer (`nn.TransformerEncoder`, causal mask,
`norm_first`, GELU), vocab 14 (12 words + `<pad>` + `<eos>`). Small enough to train in seconds,
real enough that every check below produces a meaningful number instead of a toy artifact.

## 3. The seven checks — the actual deliverable
Each requirement is a numbered section in both the script and the notebook:
1. **Shapes** — `(2,6)` tokens → `(2,6,64)` hidden → `(2,6,14)` logits → shift `(2,5,·)` →
   flat `(10,14)`/`(10,)`, with each dim named (B, T, D, V).
2. **The shift** — strings, as above.
3. **Padding mask** — contributing tokens drop **10 → 7** once `<pad>` targets are ignored.
4. **Document packing** — pack `docA <eos> docB`; the target predicting docB's first word from
   docA's `<eos>` is unlearnable. On a *trained* model that boundary term alone is **9.61** vs
   ~5 elsewhere; masking it moves the mean loss 5.61 → 5.24.
5. **Perplexity** — an untrained model should guess ~uniformly: CE ≈ ln(V), PPL ≈ V. We get
   **14.26 vs vocab 14**. (Only true with small-std init — see §5.)
6. **Weight tying** — tied head 105,088 params vs untied 105,984; untied costs one 64×14 matrix (896).
7. **Chunked CE** — the `[N,V]` logits tensor is the memory villain; computing it in chunks
   shrinks the peak logits tensor **16×** for an *identical* loss.

## 4. Part 2 — the second head
A second (untied) head predicts token **t+2**; both losses train jointly (summed). Over 300
steps both fall, but **loss2 (t+2) stays clearly above loss1 (t+1)** and the gap widens. The
corpus is a first-order Markov chain, so `x_t` strongly pins `x_{t+1}` (easy) while `x_{t+2}`
must be reached by marginalising over the still-unknown `x_{t+1}` (irreducibly fuzzier). Same
reason multi-token-prediction / speculative-decoding heads trust the near token more.

## 5. The bug that is also the lesson
The untrained-PPL check only passes because of small-std init. Default `nn.Embedding` init is
N(0,1); tied to the head it makes the untrained model over-confident and PPL explodes. The fix
(`_init_small`, std 0.02) *is* what requirement 5 is testing — a wired-wrong harness fails it.

## 6. One source of truth
`harness.py` is authoritative. `build_notebook.py` splits it on section separators into
`loss_harness.ipynb`, which is then executed with outputs embedded — the script and notebook
cannot drift. `results.txt` is a captured console run.

## 7. Reproduce
```bash
cd loss-harness
pip install -r requirements.txt
python harness.py            # prints every number
# or: open loss_harness.ipynb and Run All
```
Submit: push the repo, paste the GitHub README link into the README and `assignment.md`.
