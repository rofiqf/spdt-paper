# HANDOVER — Proyek SPDT (per 2026-10-22)

Baca file ini dulu sebelum mengerjakan apa pun di repo ini.

## Status: DUA PAPER LENGKAP + SATU EKSTENSI EKSPERIMEN LENGKAP

| Deliverable | File | Keterangan |
|---|---|---|
| Paper IEEE extended (utama, submit ini) | `paper/paper_ieee_extended.pdf` | 6 halaman, 16 ref, hasil baseline + ekstensi |
| Paper IEEE baseline (versi editorial) | `paper/paper_ieee_final.pdf` | 4 halaman, hasil baseline saja — dipertahankan sebagai versi ringkas |
| Paper pandoc lama | `paper/paper.pdf`, `paper/paper_en.pdf` | versi sebelum format IEEE (arsip) |

## Hasil kunci (jangan diubah tanpa bukti)

- Baseline: SPDT vs CART-custom 6W/0T/2L, Δ+0.0058; Friedman χ²(2)=4.0000, p=0.1353;
  stabilitas prediksi SPDT lebih buruk; overhead ~40x.
- Ekstensi: normalisasi TIDAK membantu di data bersih (3W/5L vs CART);
  sinyal robustness eksploratori di noise 10–20% (CI exclude nol);
  stabilitas split terburuk justru varian ternormalisasi;
  ANLS terbaik secara deskriptif (5W/3L, Δ+0.0066) dan memisahkan
  stabilitas split ≠ stabilitas prediksi.

## Reproduce

```bash
cd /home/ropix/spdt-paper
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/bin:$PATH"
.venv/bin/python -m pytest -q                    # 37 tests
# baseline (2.9 jam):
.venv/bin/python scripts/run_full.py && scripts/run_ablation.py
# ekstensi (2.6 jam):
.venv/bin/python scripts/run_extension_full.py && scripts/run_extension_ablation.py
# paper:
.venv/bin/python scripts/make_ieee_tables.py && scripts/make_extension_tables.py
cd paper && latexmk -pdf main.tex && cp main.pdf paper_ieee_extended.pdf
```

## State teknis server

- venv `.venv` (Python 3.12.3, paket ter-pin, disalin dari venv ALD — PyPI lambat).
- Tools: pandoc+TinyTeX+latexmk di PATH via ~/.bashrc; OpenML TERBLOKIR
  (Seeds via static zip UCI, sudah di `data/raw` + hash).
- Dataset 8 UCI ter-cache dengan SHA256; jangan di-restart ulang pipeline
  kecuali diminta — hasil ada di `results/` dan `results/extension/`.
- Git: HEAD `83b1f0a`; tag `pre-extension` menandai snapshot sebelum
  ekstensi; backup `backups/pre_extension_results_paper.tar.gz`.

## ITEM TERBUKA (untuk user / sesi baru)

1. **6 co-author fakultas** — identitas belum dikonfirmasi; TODO ada di
   `paper/main.tex`. Paper BELUM submission-ready sampai ini terisi.
2. **Turnitin** — belum dijalankan; submit `paper_ieee_extended.pdf` ke
   institusi (target ≤18%). Server ini tidak bisa menjalankannya.
3. **Push GitHub** — PAT lama sudah di-revoke; butuh token baru
   (`gh auth login --with-token`) lalu push repo ini sebagai link eksperimen.
4. **Video penjelasan** — deliverable course; skrip/outline belum dibuat.
5. Verifikasi Scopus tidak diperlukan (aturan dosen = portal sumber).

Dokumen pendukung: `docs/experiment_protocol.md` (freeze + 2 amendemen),
`docs/experiment_summary.md`, `docs/submission_readiness.md`,
`docs/prior_art_extension.csv`, `docs/extension_audit.md`.
