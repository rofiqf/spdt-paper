"""Jalankan ulang SATU metode di semua dataset (outer splits identik) dan
ganti barisnya di results/raw/fold_metrics.csv + stability_metrics.csv
secara atomik. Dipakai untuk memperbaiki hasil yang ternyata bergantung
seed (mis. cart_sklearn sebelum random_state di-fix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod
from src.evaluate import build_model_specs, run_dataset

RAW = ROOT / "results" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    specs_all = build_model_specs(config)
    if args.method not in specs_all:
        raise SystemExit(f"metode tidak dikenal: {args.method}")
    spec = {args.method: specs_all[args.method]}
    stab_n = int(config.get("stability", {}).get("n_resamples", 0))

    new_rows, new_stab = [], []
    for name, dc in config["datasets"].items():
        X, y = data_mod.load_dataset(name, drop_columns=dc.get("drop_columns"))
        rows, stab = run_dataset(name, X, y, config, spec,
                                 RAW / "checkpoints",
                                 stability_resamples=stab_n, verbose=True)
        new_rows += rows
        new_stab += stab

    fm_path = RAW / "fold_metrics.csv"
    fm = pd.read_csv(fm_path)
    fm = fm[fm.method != args.method]
    fm = pd.concat([fm, pd.DataFrame(new_rows)], ignore_index=True)
    tmp = RAW / ".fm.tmp"
    fm.to_csv(tmp, index=False)
    tmp.rename(fm_path)

    sm_path = RAW / "stability_metrics.csv"
    sm = pd.read_csv(sm_path)
    sm = sm[sm.method != args.method]
    sm = pd.concat([sm, pd.DataFrame(new_stab)], ignore_index=True)
    tmp = RAW / ".sm.tmp"
    sm.to_csv(tmp, index=False)
    tmp.rename(sm_path)
    print(f"replaced {args.method}: {len(new_rows)} fold rows, "
          f"{len(new_stab)} stability rows")


if __name__ == "__main__":
    main()
