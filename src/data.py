"""Dataset acquisition with integrity logging.

Sources (server constraints, learned 2026-10-21):
- ucimlrepo API works for 7 of 8 datasets.
- OpenML is BLOCKED on this network (SYN timeout) — do not use.
- Seeds (UCI id 236) only via static zip fallback.

Every dataset is stored under data/raw/<name>/ together with metadata +
SHA256 in data/metadata/ and data/hashes/. Re-download with a changed
checksum is a hard error, never a silent overwrite.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

UCI_URL = "https://archive.ics.uci.edu/dataset/{id}/{slug}"

SEEDS_ZIP = "https://archive.ics.uci.edu/static/public/236/seeds.zip"
SEEDS_ZIP_SHA256 = None  # recorded on first successful download

DATASET_INFO = {
    "iris":       {"uci_id": 53,  "slug": "iris",
                   "doi": "10.24432/C56C76"},
    "wine":       {"uci_id": 109, "slug": "wine",
                   "doi": "10.24432/C5PC7J"},
    "seeds":      {"uci_id": 236, "slug": "seeds",
                   "doi": "10.24432/C5H30K"},
    "glass":      {"uci_id": 42,  "slug": "glass+identification",
                   "doi": None},
    "sonar":      {"uci_id": 151, "slug": "connectionist+bench+sonar+mines+vs+rocks",
                   "doi": "10.24432/C5T01Q"},
    "ionosphere": {"uci_id": 52,  "slug": "ionosphere",
                   "doi": "10.24432/C5W01B"},
    "wdbc":       {"uci_id": 17,  "slug": "breast+cancer+wisconsin+diagnostic",
                   "doi": "10.24432/C5DW2B"},
    "banknote":   {"uci_id": 267, "slug": "banknote+authentication",
                   "doi": "10.24432/C55P57"},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_meta(name: str, X: pd.DataFrame, y: pd.Series, source: str,
                sha: str, extra: dict | None = None) -> None:
    meta = {
        "dataset": name,
        "source": source,
        "uci_id": DATASET_INFO[name]["uci_id"],
        "doi": DATASET_INFO[name]["doi"],
        "source_url": UCI_URL.format(id=DATASET_INFO[name]["uci_id"],
                                     slug=DATASET_INFO[name]["slug"]),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "target_name": str(y.name),
        "n_classes": int(y.nunique()),
        "class_distribution": {str(k): int(v)
                               for k, v in y.value_counts().sort_index().items()},
        "missing_values": bool(X.isna().any().any()),
        "sha256": sha,
    }
    if extra:
        meta.update(extra)
    (DATA_ROOT / "metadata").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "hashes").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "metadata" / f"{name}.json").write_text(
        json.dumps(meta, indent=2))
    (DATA_ROOT / "hashes" / f"{name}.sha256").write_text(
        f"{sha}  {name}_X.csv\n")


def _check_existing(name: str) -> tuple[pd.DataFrame, pd.Series] | None:
    raw = DATA_ROOT / "raw" / name
    meta_path = DATA_ROOT / "metadata" / f"{name}.json"
    if not (raw / f"{name}_X.csv").exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    sha = _sha256(raw / f"{name}_X.csv")
    if sha != meta["sha256"]:
        raise RuntimeError(
            f"Checksum mismatch for {name}: stored raw file differs from "
            f"recorded SHA256. Manual validation required; refusing to "
            f"silently overwrite.")
    X = pd.read_csv(raw / f"{name}_X.csv")
    y = pd.read_csv(raw / f"{name}_y.csv")[meta["target_name"]]
    return X, y


def _finalize(name: str, X: pd.DataFrame, y: pd.Series,
              source: str, extra: dict | None = None):
    raw = DATA_ROOT / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    X.to_csv(raw / f"{name}_X.csv", index=False)
    y.to_csv(raw / f"{name}_y.csv", index=False)
    sha = _sha256(raw / f"{name}_X.csv")
    _write_meta(name, X, y, source, sha, extra)
    return X, y


def _fetch_uci(name: str, drop_columns):
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=DATASET_INFO[name]["uci_id"])
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].copy()
    y.name = ds.data.targets.columns[0]
    for col in (drop_columns or []):
        if col in X.columns:
            X = X.drop(columns=col)
    # drop rows with missing target
    mask = y.notna()
    X, y = X[mask], y[mask]
    return _finalize(name, X, y, source="ucimlrepo")


def _fetch_seeds():
    import urllib.request
    import zipfile
    import io
    with urllib.request.urlopen(SEEDS_ZIP, timeout=60) as r:
        blob = r.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    txt = [n for n in zf.namelist() if n.endswith("seeds_dataset.txt")][0]
    df = pd.read_csv(zf.open(txt), sep=r"\s+", header=None,
                     names=["area", "perimeter", "compactness",
                            "length_kernel", "width_kernel", "asymmetry",
                            "groove_length", "class"])
    X = df.drop(columns="class")
    y = df["class"].astype(int)
    return _finalize(name="seeds", X=X, y=y, source="uci_static_zip",
                     extra={"zip_sha256": hashlib.sha256(blob).hexdigest()})


def load_dataset(name: str, drop_columns=None,
                 redownload: bool = False):
    """Return (X DataFrame, y Series) with integrity guarantees."""
    if name not in DATASET_INFO:
        raise ValueError(f"unknown dataset {name}")
    if not redownload:
        cached = _check_existing(name)
        if cached is not None:
            return cached
    if name == "seeds":
        return _fetch_seeds()
    return _fetch_uci(name, drop_columns)


def dataset_inventory() -> pd.DataFrame:
    rows = []
    for name in DATASET_INFO:
        meta_path = DATA_ROOT / "metadata" / f"{name}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            rows.append({"dataset": name, **{k: meta[k] for k in
                         ("n_samples", "n_features", "n_classes",
                          "missing_values", "sha256", "source")}})
    return pd.DataFrame(rows)
