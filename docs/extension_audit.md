# Extension Audit (Stage 1) — 2026-10-22

## Verdict: PASS — existing results reconcile; extension may proceed.

| Check | Method | Result |
|---|---|---|
| Original results reproducible | `validate_reproducibility.py` (3 probes) | PASS bit-exact |
| Dataset integrity | metadata + SHA256, 8/8 | OK |
| Paper reference values == artifacts | Friedman 4.0000/p 0.1353/W 0.2500; fit 0.5340/21.4302; disagreement 0.0928/0.1014 | OK |
| Backup | `backups/pre_extension_results_paper.tar.gz` + git tag `pre-extension` (8a4788e) | OK |
| No overwrite policy | extension writes only to `results/extension/`, `figures/*extension*`/`noise_*`/`root_*`/`runtime_vs_*`, `logs/extension_*` | OK |

## Frozen reference values (must not change)

6W/0T/2L, mean Δ +0.0058, χ²(2)=4.0000, p=0.1353, Kendall W=0.2500,
CART-cust disagreement 0.0928, SPDT 0.1014, fit 0.5340s / 21.4302s.

## Environment snapshot

Python 3.12.3, pinned requirements.txt (numpy 2.3.4, scipy 1.16.3,
pandas 2.3.3, scikit-learn 1.9.1, statsmodels 0.15.0),
seeds [42,123,2026], global 20261021 in legacy pipeline.
Recorded to logs/extension_run_manifest.json by the extension runner.
