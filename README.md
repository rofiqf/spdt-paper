# Stability-Penalized Split Selection for Decision Tree Classification

Proyek eksperimen reproducible untuk mengevaluasi kriteria split decision tree
yang dipenalti variansi bootstrap:

    Score(s) = Gain(s, D) − λ · Var_b[Gain(s, D^(b))]

**Status metode: proposed / candidate formulation — klaim novelty TIDAK
diasumsikan.** Prior-art gate wajib dilakukan sebelum eksperimen besar
(lihat `docs/prior_art_matrix.csv`).

## Lingkungan

- Python 3.12.x (Ubuntu), paket ter-pin di `requirements.txt`
- venv lokal: `python3.12 -m venv --without-pip .venv` + get-pip bootstrap
  (server ini tidak punya `ensurepip`/`python3.12-venv`)
- Selalu gunakan `.venv/bin/python`

## Alur kerja

```bash
.venv/bin/python -m pytest -q                 # unit tests
.venv/bin/python scripts/run_pilot.py         # pilot kecil (Stage 4)
.venv/bin/python scripts/run_full.py          # nested CV penuh (Stage 6)
.venv/bin/python scripts/run_ablation.py      # ablation split identik (Stage 7)
.venv/bin/python scripts/analyze_results.py   # ringkasan statistik
.venv/bin/python scripts/make_figures.py      # semua figure
.venv/bin/python scripts/validate_reproducibility.py  # repro-bit-exact
bash scripts/build_pdf.sh                     # paper.pdf IEEEtran dari draft.md
```

Lihat `docs/method.md` untuk definisi metode, `docs/limitations.md`
untuk keterbatasan wajib, dan `docs/protocol_freeze.md` untuk protokol
beku + addendum (fix determinisme seed CART sklearn).

## Struktur

```
configs/      konfigurasi eksperimen (freeze di Stage 5)
src/          implementasi (tree, data, evaluate, statistics, plots, baselines)
scripts/      runner pilot & full
tests/        unit tests (kebenaran CART, leakage, reproducibility)
docs/         prior-art matrix, metode, keterbatasan
results/      fold metrics (raw) & ringkasan (summary)
figures/      plot hasil
paper/        draf paper (Bahasa Indonesia) + references.bib
logs/         log eksekusi & manifest
data/         dataset mentah + metadata + sha256
```

## Catatan jaringan server

OpenML terblokir di server ini; dataset diperoleh via `ucimlrepo` (UCI API)
atau static zip UCI (fallback Seeds). Lihat `src/data.py`.
