**Keywords**—decision trees; split criterion; bootstrap stability; gain variance; nested cross-validation; nonparametric statistical comparison.

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
claim. The evaluation uses {{N_DATASETS}} UCI tabular classification
datasets with repeated nested cross-validation (5 folds x 3 repeats on
three seeds), hyperparameter tuning in the inner loop, and Macro-F1 as
the primary metric. Results: SPDT versus custom CART (lambda=0) records
{{WINS}} wins, {{TIES}} ties, and {{LOSSES}} losses with a mean Macro-F1
difference of {{MEAN_DELTA}}. The Friedman test gives chi^2(2)={{FRIEDMAN_STAT}},
p={{FRIEDMAN_P}}, Kendall's W={{KENDALL_W}} — insufficient evidence to
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
— the very phenomenon behind *bagging* and *random forests*
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
— the closest resampling-per-node mechanism to this study, but its
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

Datasets: {{N_DATASETS}} UCI tabular classification datasets (Table I).
Protocol: repeated stratified nested CV — outer 5 folds x 3 repeats for
each of the seeds [42, 123, 2026] (45 evaluations per dataset-method,
identical splits for all methods); inner stratified 3-fold tuning with
deterministically derived seeds. Preprocessing is median imputation
fitted on the training portion only (pipeline). Primary metric Macro-F1;
secondary: balanced accuracy, accuracy, fit/predict time, node count,
tree depth. Prediction stability is measured separately as mean pairwise
disagreement among models trained on 10 subsamples (90%) of each outer
training fold, evaluated on the same outer test fold — reported
separately from accuracy.

{{TABLE_INVENTORY}}

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

{{TABLE_PERFORMANCE}}

Table: Macro-F1 mean +/- sd (descriptive, fold-level) per dataset and method.

{{TABLE_MEAN_RANK}}

Table: Mean Macro-F1 rank.

{{TABLE_STATISTICS}}

Table: Dataset-level statistical tests.

{{TABLE_POSTHOC}}

Table: Pairwise (post-hoc) comparisons.

Paired summary SPDT vs custom CART: **{{WINS}} wins / {{TIES}} ties /
{{LOSSES}} losses**, mean Macro-F1 difference {{MEAN_DELTA}}.

{{TABLE_RUNTIME}}

Table: Computational cost.

{{TABLE_STABILITY}}

Table: Prediction stability (disagreement; not an accuracy metric).

{{TABLE_ABLATION}}

Table: Ablation on identical splits.

{{TABLE_HYPERPARAMS}}

Table: SPDT hyperparameter selection frequency (45 outer folds per dataset).

**Discussion.** The results descriptively favor SPDT but do not reach
confirmatory evidence. SPDT achieves the best mean rank (1.500 vs 2.000
for sklearn CART and 2.500 for custom CART) and wins on 6 of 8 datasets
paired against custom CART, yet the Friedman test finds insufficient
evidence of a difference (p = {{FRIEDMAN_P}}), and the exploratory
SPDT-custom CART comparison gives a raw p of 0.078 — it does not survive
the 5% level and must not be read as significant. With only 8 datasets,
statistical power is limited; per protocol, the results are reported as
they stand and p >= 0.05 is not interpreted as equivalence.

A second important finding: the stability penalty does not reduce
prediction disagreement — SPDT is in fact slightly higher (0.101) than
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
and dataset-level statistical analysis on {{N_DATASETS}} UCI datasets.
Descriptively, SPDT achieved the best mean rank and won 6 of 8 paired
comparisons against custom CART, but the Friedman test provided
insufficient evidence of a difference (chi^2(2)={{FRIEDMAN_STAT}}, p={{FRIEDMAN_P}}),
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
