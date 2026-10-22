"""Stage 5 pilot: iris/wine/wdbc through the extension pipeline; runtime
estimate; NO confirmatory claims. Output to results/extension/pilot_*.csv
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import extension_evaluation as ee


def main():
    t0 = time.perf_counter()
    ycfg = yaml.safe_load((ROOT / "configs/extension.yaml").read_text())
    cfg = ee.default_cfg()
    cfg.update({k: ycfg[k] for k in ("tune_B", "epsilon", "noise_levels",
                                     "noise_realizations",
                                     "stability_subsamples", "rf_trees")})
    cfg["pilot_datasets"] = ycfg["pilot_datasets"]
    cfg["full_datasets"] = ycfg["full_datasets"]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "pilot", "cfg": {k: cfg[k] for k in
                                         ("tune_B", "epsilon",
                                          "noise_levels",
                                          "noise_realizations",
                                          "stability_subsamples",
                                          "rf_trees")}}
    rows_a, rows_b, rows_c = ee.run_datasets(cfg, cfg["pilot_datasets"],
                                             manifest)
    elapsed = time.perf_counter() - t0
    n_full = len(cfg["full_datasets"])
    pilot_avg = elapsed / len(cfg["pilot_datasets"])
    proj = pilot_avg * n_full
    manifest["elapsed_seconds"] = elapsed
    manifest["projected_full_seconds"] = proj
    ee.EXT.mkdir(parents=True, exist_ok=True)
    for tag, rows in [("pilot_normalized_penalty", rows_a),
                      ("pilot_noise_robustness", rows_b),
                      ("pilot_split_stability", rows_c)]:
        ee._atomic_write(ee.EXT / f"{tag}.csv", pd.DataFrame(rows))
    (ROOT / "logs" / "extension_run_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps({"elapsed_s": round(elapsed, 1),
                      "projected_full_s": round(proj, 1),
                      "rows": manifest["rows"]}, indent=2))


if __name__ == "__main__":
    main()
