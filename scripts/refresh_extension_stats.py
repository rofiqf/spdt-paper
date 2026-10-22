"""Re-run extension statistics from existing CSVs (no refits)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import extension_statistics as estat
from src.extension_evaluation import EXT


def main():
    a = pd.read_csv(EXT / "normalized_penalty.csv")
    b = pd.read_csv(EXT / "noise_robustness.csv")
    c = pd.read_csv(EXT / "split_stability.csv")
    df = estat.summarize_extension(a, b, c)
    tmp = (EXT / "statistical_tests.csv").with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(EXT / "statistical_tests.csv")
    h = df[df.analysis.str.contains("holm")]
    print(h[["subset", "p", "holm_p", "decision"]].round(4).to_string(
        index=False))
    print("rows:", len(df))


if __name__ == "__main__":
    main()
