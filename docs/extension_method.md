# Extension Methods

## M3 — Normalized SPDT (candidate, exploratory)

Score_norm(s) = Gain(s,D) - lambda * Var_b[Gain(s,D^(b))]
                / (Gain(s,D)^2 + epsilon)

- Same bootstrap protocol as SPDT (B resamples of the node, candidate
  fixed); normalization makes the penalty scale-free w.r.t. gain magnitude
  (coefficient-of-variation-style), addressing the scale-dependence
  limitation of M2.
- epsilon > 0 stabilizes near-zero gains; behavior at Gain~0: penalty
  ~ Var/eps (bounded since Var is finite for a fixed candidate set).
- Guardrails: invalid candidates rejected as in M2; scores clipped to
  finite range by construction (numpy errstate + finite checks in tests).
- Implemented as StabilityTreeClassifier(normalized_penalty=True).
- Status: exploratory hypothesis; NOT claimed superior or novel
  (see docs/prior_art_extension.csv: no formula-level collision found,
  but negative search is not proof of novelty).

## M5 — ANLS faithful re-implementation (Lee, Sim, Hong 2024, IEEE Access)

Per node: bootstrap replicates -> plain CART-optimal split per replicate ->
accumulate predictor frequencies -> stop when binomial Z of top-2
predictors > Z_alpha (alpha=90%, Z=1.645), minimum 10 replicates before
first test. Stabilized split: modal predictor; continuous cutoff = MEDIAN
of winning cutoffs. Pre-pruning: node smaller than 1% of training set.

Documented deviations: (1) max_iterations=1500 cap (paper used none;
reports up to 47,587 iterations/node); (2) categorical branch omitted
(all datasets numeric); (3) ties broken by first-achieved max frequency.
Verified faithful from open-access PDF pseudocode (Sec. III-A).

## Label noise protocol (Experiment B)

Corruption applied to OUTER TRAINING labels only; outer test stays clean.
Hyperparameters frozen from clean-training inner CV (documented: tuning
uses clean labels; noise applied post-tuning to the training partition —
models are not retuned on corrupted labels). Same corruption realization
(identical seed -> identical mask) across all compared methods.
Corrupted label always differs from the original. Noise levels
{0,5,10,20,30}%, 3 realizations, seeds derived from
SeedSequence([42, repeat, fold, level*1000, realization, dataset_idx]).

## Split stability (Experiment C)

Per outer fold, S=10 models on 90% subsamples (fixed derived seeds):
root-feature agreement (pairwise), root-threshold variation (SD within
modal feature only), prediction disagreement on the fixed outer test.
