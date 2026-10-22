"""Stage 6: full nested repeated stratified CV sesuai config yang sudah
di-freeze (Stage 5). Menghasilkan results/raw/fold_metrics.csv,
results/raw/stability_metrics.csv, dan logs/run_manifest.json.
"""

from __future__ import annotations

import argparse
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

from src import data as data_mod
from src.evaluate import build_model_specs, run_dataset


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset dataset untuk resume parsial")
    args = ap.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    specs = build_model_specs(config)

    out_raw = ROOT / "results" / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    ckpt = out_raw / "checkpoints"

    names = args.datasets or list(config["datasets"].keys())
    stab_n = int(config.get("stability", {}).get("n_resamples", 0))

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "datasets": {},
    }

    all_rows, all_stab, failures = [], [], {}
    for name in names:
        t0 = time.perf_counter()
        try:
            drop = config["datasets"][name].get("drop_columns")
            X, y = data_mod.load_dataset(name, drop_columns=drop)
            rows, stab = run_dataset(name, X, y, config, specs, ckpt,
                                     stability_resamples=stab_n,
                                     verbose=True)
            all_rows += rows
            all_stab += stab
            manifest["datasets"][name] = {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - t0,
            }
        except Exception as e:  # log & lanjut; jangan fabrikasi
            failures[name] = repr(e)
            manifest["datasets"][name] = {"status": "failed",
                                          "error": repr(e)}
        # tulis ulang hasil kumulatif setiap dataset (atomic-ish)
        if all_rows:
            tmp = out_raw / ".fold_metrics.tmp"
            pd.DataFrame(all_rows).to_csv(tmp, index=False)
            tmp.rename(out_raw / "fold_metrics.csv")
        if all_stab:
            tmp = out_raw / ".stab.tmp"
            pd.DataFrame(all_stab).to_csv(tmp, index=False)
            tmp.rename(out_raw / "stability_metrics.csv")

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failures"] = failures
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"done. failures={failures}")
    print(f"fold rows: {len(all_rows)}, stability rows: {len(all_stab)}")


if __name__ == "__main__":
    main()
