# Session 14 — Plan

## Goal
Train a dense ("linear") transformer, convert it into a Mixture-of-Experts, and **show both
continue to train and reduce loss**, with training logs in the repo. Binding rule: the conversion
must be *exact* — the MoE at init must reproduce the dense output (verified), so "continues
training" is a property of the construction, not luck.

## Constraints from the spec
- Train a dense/linear model, then convert it into an MoE.
- Model size and data are our call, but must show continued training + falling loss.
- The repo MUST contain training logs.
- Submit a detailed README (GitHub link); 1000 pts.

## Approach — sparse upcycling
1. **Dense baseline** — a small GPT-style byte LM whose FFN is a SwiGLU block; train it so loss
   clearly falls.
2. **MoE layer** — router (Linear d→N) + N SwiGLU experts, top-k routing, weights renormalised to
   sum to 1, plus the Switch-Transformer load-balance auxiliary loss. Log per-expert load.
3. **Upcycle** — copy the trained dense FFN into every expert (+ tiny noise to break symmetry),
   zero the router. Because experts are identical and top-k weights sum to 1, the MoE output at
   init equals the dense output.
4. **Correctness gate** — check the MoE's cross-entropy at conversion matches the dense model's
   (CE from logits, excluding the aux term), and that the MoE's first step loss lands on the dense
   final loss. Everything downstream is only trustworthy because this passes.
5. **Continue training the MoE** — loss keeps falling from where dense stopped.
6. **Contrast run** — train the same MoE *from scratch* under an equal step budget, to show the
   warm start is worth something.
7. **Parameter accounting** — report total vs active-per-token params (capacity vs cost).
8. **Notebook + docs** — build_notebook.py splits the script; execute with outputs + plot;
   README, requirements, session-notes, plan, research, WALKTHROUGH.

## Steps / deliverables
- `moe_lab.py` — source of truth (data, SwiGLU, MoE, LM, upcycle, train, run).
- `fig_moe.png` — left: dense loss → MoE continues (+ scratch overlay); right: per-expert load.
- `results.txt` — param accounting, conversion |Δ|, start/final loss for all three runs.
- `training_log.txt` — **the required training logs** (per-step loss + load-cv lines).
- `moe_lab.ipynb` — executed notebook with embedded output + plot.

## Risks / watch-items
- **Conversion not exact** — if experts aren't true copies or weights don't renormalise, the MoE
  jumps at init. Verify the CE difference is tiny (and the first MoE step lands on the dense final
  loss) before trusting the curve.
- **Router collapse** — without the aux loss a few experts hog all tokens; log load-cv to prove
  balance.
- **"Linear" wording** — read as the dense FFN transformer (what MoE papers upcycle); a bare
  `nn.Linear` can't language-model. Documented in research.md.
- **Local scale** — small model + tiny corpus so the repo is reproducible in minutes; the method
  is identical at larger scale (raise config on a GPU).

## Definition of done
Notebook runs top to bottom with output + plot; conversion CE matches dense + first MoE step lands
on the dense final loss (PASS); dense loss falls;
upcycled MoE continues from the dense loss and falls further; from-scratch MoE under equal budget
visibly lags the upcycled one; experts stay balanced (load-cv small); param accounting shows MoE
has more total params at ~k/N the per-token FFN cost; training_log.txt committed.
