---
title: "Stability-Penalized Split Selection for Decision Tree Classification: A Reproducible Multi-Dataset Evaluation"
author: "Rofiq Fauzi (p36202600047@mhs.dinus.ac.id), Magister Artificial Intelligence, Universitas Dian Nuswantoro"
---

**Keywords**--decision trees; split criterion; bootstrap stability; gain variance; nested cross-validation; nonparametric statistical comparison.

## Abstract

The CART decision tree is fast and interpretable, but its structure is
unstable: small changes in the training data can alter the selected split
and, ultimately, the predictions. This study evaluates a very simple
candidate formulation, named *Stability-Penalized Decision Tree* (SPDT),
in which each candidate split is scored as Gini gain minus a weighted
penalty on the variance of that gain over bootstrap resamples of the node
data: `Score(s) = Gain(s,D) - lambda*Var_b[Gain(s,D^(b))]`. A literature
search up to 22 October 2026 found no formula-level equivalent of this
criterion, but the tree-stability research space is crowded; the
formulation is therefore positioned as a candidate, not as a novelty
claim. The evaluation uses 8 UCI tabular classification
datasets with repeated nested cross-validation (5 folds x 3 repeats on
three seeds), hyperparameter tuning in the inner loop, and Macro-F1 as
the primary metric. Results: SPDT versus custom CART (lambda=0) records
6 wins, 0 ties, and 2 losses with a mean Macro-F1
difference of +0.0058. The Friedman test gives chi^2(2)=4.0000,
p=0.1353, Kendall's W=0.2500 -- insufficient evidence to
reject the null hypothesis, so confirmatory post-hoc comparisons were not
run. Ablations show that the penalty strength lambda and the number of
resamples B barely affect Macro-F1, tree-depth constraints matter far
more, and the prediction-stability measure of SPDT is slightly worse than
that of custom CART at roughly 40x the fitting time. These findings are
an honest null result: across the eight small-to-medium datasets tested,
the bootstrap gain-variance penalty provides no benefit sufficient to
outweigh the simplicity of CART, without ruling out benefits in other
data regimes.

## I. Introduction

Decision trees are nonparametric classifiers that recursively partition
the feature space. The CART split criterion selects the candidate
(feature, threshold) that maximizes the reduction of Gini impurity
[@breiman1984cart;@quinlan1986induction]. However, tree structure is
known to be unstable: small perturbations of the training data can change
the selected split [@li2002instability], and consequently the predictions
-- the very phenomenon behind *bagging* and *random forests*
[@breiman1996bagging;@breiman2001randomforests;@dietterich2000ensemble].

Various approaches have reduced this instability at different levels:
significance filtering, structural restriction, ensemble aggregation, or
stable model selection [@briand2009stability;@lee2024anls;@mirzamomen2017stable;@wang2017tsallis].
What is rarely tested directly is the narrower question:

> **Can a bootstrap variance penalty on the split gain, subtracted
> directly from the split score, improve out-of-sample Macro-F1 and/or
> prediction stability compared with standard CART under repeated
> resampling, at acceptable runtime cost?**

Null hypothesis: no systematic improvement over the baseline across
independent datasets. No hypothesis is made that SPDT *must* win.

The claimed contributions are limited to: (i) an explicit candidate
formulation; (ii) a reproducible implementation and audit; (iii) a
leakage-free nested-CV evaluation; (iv) honest ablation and statistical
analysis of both positive and negative outcomes.

## II. Related Work

**Decision tree stability.** Structural tree instability was formalized
by Li and Clarke [@li2002instability]; quantitative stability and
tree-similarity measures were developed, among others, by Turney
[@turney1995stability] and Briand et al. [@briand2009stability], who also
evaluated bootstrap-based node-level stabilization (the NLS/Dannegger
lineage). Lee et al. [@lee2024anls] proposed ANLS, which runs bootstrap
iterations at each node and selects the most frequently chosen predictor
-- the closest resampling-per-node mechanism to this study, but its
aggregation is frequency voting rather than a gain-variance penalty.
Mirzamomen and Kangavari [@mirzamomen2017stable] stabilize trees through
multi-attribute split tests.

**Alternative split criteria.** Two-term Tsallis entropy criteria
[@wang2017tsallis], label-noise-robust criteria [@yang2019pairgain],
bias-corrected Gini split selection [@strobl2007bias], and Hoeffding-bound
data-stream trees [@domingos2000vfdt] show that the "modify the split
score" research space is active. Delise [@delise2026era] scores split
consistency across environments in boosting. Penalizing a resampled score
by its variance is also known outside trees (the mean-variance pattern in
model selection). None of the works we were able to verify contains the
formula `Gain - lambda*Var_bootstrap[Gain]`; nevertheless, a negative
search result is not proof of novelty, and the formulation remains a
candidate.

## III. Method

### A. Proposed split criterion

For a node with dataset D (training portion only), with
`Gini(D) = 1 - Sum_c p(c|D)^2` and split gain

`Gain(s,D) = Gini(D) - (|D_L|/|D|)*Gini(D_L) - (|D_R|/|D|)*Gini(D_R)`,

B bootstrap resamples `D^(b)` are drawn from the node rows. For every
valid candidate s:

`V(s) = Var_b[Gain(s,D^(b))]` (sample variance, ddof=1)

**Main score: `Score(s) = Gain(s,D) - lambda*V(s)`**.

Deterministic policies: candidate thresholds are midpoints between sorted
distinct adjacent values; `min_samples_leaf`/`min_samples_split` are
applied to the original node counts; an empty child partition inside a
resample contributes 0; leaves predict the majority class with
smallest-label tie-breaking; bootstrap seeds derive from
`(random_state, node id)`. For lambda=0 the entire bootstrap computation
is skipped and the estimator degenerates to the custom CART reference
(verified identical to an independent reference implementation in unit
tests). A separate exploratory variant (normalization by `V/(Gain^2+eps)`)
is documented but is not the main method. An important limitation: V(s)
is scale-dependent on the gain.

### B. Evaluation design

Datasets: 8 UCI tabular classification datasets (Table I).
Protocol: repeated stratified nested CV -- outer 5 folds x 3 repeats for
each of the seeds [42, 123, 2026] (45 evaluations per dataset-method,
identical splits for all methods); inner stratified 3-fold tuning with
deterministically derived seeds. Preprocessing is median imputation
fitted on the training portion only (pipeline). Primary metric Macro-F1;
secondary: balanced accuracy, accuracy, fit/predict time, node count,
tree depth. Prediction stability is measured separately as mean pairwise
disagreement among models trained on 10 subsamples (90%) of each outer
training fold, evaluated on the same outer test fold -- reported
separately from accuracy.

| Dataset | N | Features | Classes | Source |
|---|---|---|---|---|
| banknote | 1372 | 4 | 2 | ucimlrepo |
| glass | 214 | 9 | 6 | ucimlrepo |
| ionosphere | 351 | 34 | 2 | ucimlrepo |
| iris | 150 | 4 | 3 | ucimlrepo |
| seeds | 210 | 7 | 3 | uci_static_zip |
| sonar | 208 | 60 | 2 | ucimlrepo |
| wdbc | 569 | 30 | 2 | ucimlrepo |
| wine | 178 | 13 | 3 | ucimlrepo |

Table: Dataset inventory.

### C. Statistical analysis

The unit of inference is the dataset (averaged over 45 outer folds), not
the fold. The three primary methods (sklearn CART, custom CART lambda=0,
SPDT) are compared with the Friedman test using datasets as blocks;
only when the omnibus test provides grounds for it are paired Wilcoxon
signed-rank tests with Holm correction and rank-biserial effect sizes
run. p >= 0.05 is never interpreted as evidence of equivalence. Random
forest is included as a non-equivalent contextual benchmark and is not
part of the primary omnibus test.

## IV. Results and Discussion

| Dataset | cart_sklearn | cart_custom | spdt | random_forest | Delta (spdt-cart_custom) |
|---|---|---|---|---|---|
| banknote | 0.982 +/- 0.008 | 0.981 +/- 0.007 | 0.981 +/- 0.007 | 0.994 +/- 0.004 | +0.0004 |
| glass | 0.620 +/- 0.097 | 0.609 +/- 0.093 | 0.626 +/- 0.093 | 0.742 +/- 0.075 | +0.0166 |
| ionosphere | 0.873 +/- 0.034 | 0.881 +/- 0.038 | 0.879 +/- 0.038 | 0.929 +/- 0.037 | -0.0027 |
| iris | 0.949 +/- 0.033 | 0.945 +/- 0.038 | 0.944 +/- 0.037 | 0.955 +/- 0.034 | -0.0008 |
| seeds | 0.897 +/- 0.032 | 0.906 +/- 0.036 | 0.913 +/- 0.034 | 0.923 +/- 0.038 | +0.0066 |
| sonar | 0.719 +/- 0.065 | 0.706 +/- 0.073 | 0.725 +/- 0.061 | 0.833 +/- 0.065 | +0.0189 |
| wdbc | 0.925 +/- 0.031 | 0.925 +/- 0.030 | 0.928 +/- 0.030 | 0.956 +/- 0.023 | +0.0037 |
| wine | 0.905 +/- 0.047 | 0.905 +/- 0.044 | 0.909 +/- 0.046 | 0.981 +/- 0.024 | +0.0040 |

Table: Macro-F1 mean +/- sd (descriptive, fold-level) per dataset and method.

| Method | Mean rank |
|---|---|
| spdt | 1.5000 |
| cart_sklearn | 2.0000 |
| cart_custom | 2.5000 |

Table: Mean Macro-F1 rank.

| Test | Statistic | df | p | #datasets | Kendall W | Decision |
|---|---|---|---|---|---|---|
| friedman | 4.0000 | 2 | 0.1353 | 8 | 0.2500 | no omnibus evidence; post-hoc not run |

Table: Dataset-level statistical tests.

_Post-hoc not run: the Friedman omnibus test gave no grounds (p >= 0.05), per protocol._

Table: Pairwise (post-hoc) comparisons.

Paired summary SPDT vs custom CART: **6 wins / 0 ties /
2 losses**, mean Macro-F1 difference +0.0058.

| Method | Fit mean (s) | Fit median (s) | Predict mean (s) | Mean nodes | Mean depth |
|---|---|---|---|---|---|
| cart_custom | 0.5340 | 0.4163 | 0.0009 | 26.6722 | 5.1000 |
| cart_sklearn | 0.2693 | 0.2605 | 0.0010 | 27.4222 | 5.2917 |
| random_forest | 1.1169 | 1.0863 | 0.0573 | nan | nan |
| spdt | 21.4302 | 14.6491 | 0.0010 | 28.5944 | 5.4611 |

Table: Computational cost.

| Method | Mean disagreement | SD |
|---|---|---|
| cart_custom | 0.0928 | 0.0853 |
| cart_sklearn | 0.0983 | 0.0885 |
| random_forest | 0.0271 | 0.0337 |
| spdt | 0.1014 | 0.0934 |

Table: Prediction stability (disagreement; not an accuracy metric).

| Variant | Mean macro F1 | SD | Fit mean (s) | Mean nodes |
|---|---|---|---|---|
| A0_lam0 | 0.8599 | 0.1270 | 0.0201 | 37.6611 |
| A1_lam0.1_B10 | 0.8598 | 0.1256 | 0.0860 | 37.6111 |
| A1_lam1_B10 | 0.8601 | 0.1255 | 0.0862 | 37.5278 |
| A2_B20 | 0.8601 | 0.1251 | 0.1136 | 37.4556 |
| A2_B5 | 0.8613 | 0.1243 | 0.0711 | 37.4667 |
| A3_d3_leaf1 | 0.8357 | 0.1592 | 0.0371 | 12.8722 |
| A3_d3_leaf5 | 0.8353 | 0.1546 | 0.0373 | 11.8611 |
| A3_d8_leaf1 | 0.8592 | 0.1272 | 0.0821 | 35.2500 |
| A3_d8_leaf5 | 0.8521 | 0.1381 | 0.0820 | 24.7389 |

Table: Ablation on identical splits.

| parameter | value | frequency | share |
|---|---|---|---|
| max_depth | 5.0000 | 140 | 0.3889 |
| max_depth | 8.0000 | 102 | 0.2833 |
| max_depth | 3.0000 | 92 | 0.2556 |
| min_samples_leaf | 1.0000 | 201 | 0.5583 |
| min_samples_leaf | 2.0000 | 97 | 0.2694 |
| min_samples_leaf | 5.0000 | 62 | 0.1722 |
| min_samples_split | 2.0000 | 252 | 0.7000 |
| min_samples_split | 5.0000 | 69 | 0.1917 |
| min_samples_split | 10.0000 | 39 | 0.1083 |
| n_bootstrap | 5.0000 | 191 | 0.5306 |
| n_bootstrap | 20.0000 | 87 | 0.2417 |
| n_bootstrap | 10.0000 | 82 | 0.2278 |
| penalty_lambda | 10.0000 | 233 | 0.6472 |
| penalty_lambda | 0.0000 | 61 | 0.1694 |
| penalty_lambda | 0.0100 | 41 | 0.1139 |
| penalty_lambda | 1.0000 | 22 | 0.0611 |
| penalty_lambda | 0.1000 | 3 | 0.0083 |

Table: SPDT hyperparameter selection frequency (45 outer folds per dataset).

**Discussion.** The results descriptively favor SPDT but do not reach
confirmatory evidence. SPDT achieves the best mean rank (1.500 vs 2.000
for sklearn CART and 2.500 for custom CART) and wins on 6 of 8 datasets
paired against custom CART, yet the Friedman test finds insufficient
evidence of a difference (p = 0.1353), and the exploratory
SPDT-custom CART comparison gives a raw p of 0.078 -- it does not survive
the 5% level and must not be read as significant. With only 8 datasets,
statistical power is limited; per protocol, the results are reported as
they stand and p >= 0.05 is not interpreted as equivalence.

A second important finding: the stability penalty does not reduce
prediction disagreement -- SPDT is in fact slightly higher (0.101) than
custom CART (0.093) and sklearn CART (0.098). Random forest remains the
most stable (0.027) through ensemble aggregation. The motivating promise
of the method (better prediction stability) therefore does not
materialize on this benchmark; a gain-variance penalty is no substitute
for ensembling.

A third finding from the identical-split ablations: neither the penalty
strength lambda nor the number of resamples B materially changes Macro-F1
(all lambda and B variants fall in the 0.859-0.861 range), whereas
depth/leaf constraints matter far more (depth 3 drops to about 0.836).
Inner tuning tends to select large lambda (10 in 64.7% of folds) and a
small B (5 in 53.1%), indicating the penalty is often selected but has a
small effect on accuracy. The cost is real: SPDT fitting time is roughly
40x that of custom CART. Overall, the effectiveness of the bootstrap
gain-variance penalty is dataset-dependent and small in magnitude;
simple impurity-based splitting remains preferable for these
small-to-medium datasets.

## V. Conclusion

This study evaluated the SPDT candidate formulation reproducibly, with
leakage-free repeated nested CV, controlled ablation on identical splits,
and dataset-level statistical analysis on 8 UCI datasets.
Descriptively, SPDT achieved the best mean rank and won 6 of 8 paired
comparisons against custom CART, but the Friedman test provided
insufficient evidence of a difference (chi^2(2)=4.0000, p=0.1353),
so there is no confirmatory basis for claiming improvement. Moreover, the
prediction-stability measure did not improve and was slightly worse,
while compute cost rose roughly 40x. Under the tested conditions, the
simplicity of CART remains preferable. Limitations: eight small datasets,
Gini impurity only, a single disagreement-based stability proxy, a
formula-level novelty claim that is not established, and the absence of
the closest literature criteria (ANLS, minimax split) as comparators. The
most valuable future work: direct comparison with node-level bootstrap
stabilization from the literature, a theoretically grounded penalty
normalization by gain scale, and evaluation on larger benchmarks.

## References

<div id="refs"></div>
