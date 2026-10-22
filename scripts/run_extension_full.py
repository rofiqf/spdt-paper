"""Stage 7 full extension run: all 8 datasets, A/B/C in one pass, then D
(runtime tradeoff incl. memory probe) and the stats table."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import extension_evaluation as ee
from src import extension_statistics as estat


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def main():
    t0 = time.perf_counter()
    ycfg = yaml.safe_load((ROOT / "configs/extension.yaml").read_text())
    cfg = ee.default_cfg()
    cfg.update({k: ycfg[k] for k in ("tune_B", "epsilon", "noise_levels",
                                     "noise_realizations",
                                     "stability_subsamples", "rf_trees")})
    cfg["pilot_datasets"] = ycfg["pilot_datasets"]
    cfg["full_datasets"] = ycfg["full_datasets"]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full", "git_commit": git_commit(),
        "python": platform.python_version(), "platform": platform.platform(),
        "cfg": {k: cfg[k] for k in ("tune_B", "epsilon", "noise_levels",
                                    "noise_realizations",
                                    "stability_subsamples", "rf_trees")},
        "datasets": cfg["full_datasets"],
    }
    rows_a, rows_b, rows_c = ee.run_datasets(cfg, cfg["full_datasets"],
                                             manifest)

    a = pd.DataFrame(rows_a)
    b = pd.DataFrame(rows_b)
    c = pd.DataFrame(rows_c)

    # ---- D: runtime tradeoff ------------------------------------------
    rt = (a.groupby("method", as_index=False)
          .agg(fit_mean=("fit_seconds", "mean"),
               fit_median=("fit_seconds", "median"),
               predict_mean=("predict_seconds", "mean"),
               nodes_mean=("node_count", "mean"),
               candidates_mean=("candidates_evaluated", "mean"),
               boot_gain_evals_mean=("bootstrap_gain_evaluations", "mean"),
               f1_mean=("f1_macro", "mean")))
    peak_mb = ee.memory_probe(cfg)
    rt["peak_memory_mb_probe"] = [peak_mb if m in ("spdt", "spdt_norm")
                                  else None for m in rt["method"]]
    ee._atomic_write(ee.EXT / "runtime_tradeoff.csv", rt)

    # ---- stats ---------------------------------------------------------
    stats_df = estat.summarize_extension(a, b, c)
    ee._atomic_write(ee.EXT / "statistical_tests.csv", stats_df)

    manifest["elapsed_seconds"] = time.perf_counter() - t0
    manifest["peak_memory_mb_probe"] = peak_mb
    (ROOT / "logs" / "extension_run_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps({"elapsed_s": round(manifest["elapsed_seconds"], 1),
                      "rows": manifest["rows"],
                      "peak_mb": round(peak_mb, 1)}, indent=2))


if __name__ == "__main__":
    main()
