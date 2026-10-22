# HANDOVER — Proyek SPDT (sesi ditutup 2026-10-22)

Baca file ini dulu sebelum mengerjakan apa pun di repo ini.

## Status akhir sesi: SUBMISSION-READY (teknis), menunggu Turnitin + video

**Repo publik: https://github.com/rofiqf/spdt-paper** (branch `master`,
tag `pre-extension`). README sudah bergaya dokumentasi mahasiswa untuk
Project Mid Prof. Guruh. Build artifact LaTeX sudah di-.gitignore.

## Deliverable course

| Item | Status |
|---|---|
| Final paper IEEE 6 halaman | ✅ `paper/paper_ieee_extended.pdf` (8 penulis: mahasiswa first + 7 dosen; 16 referensi terverifikasi Crossref; bagian Acknowledgment penggunaan AI/Kimi) |
| Link eksperimen | ✅ repo GitHub di atas |
| Video penjelasan | ⏳ **belum direkam** — skrip siap: `docs/video_script.md` (8 slide, ±8 menit) |
| Turnitin ≤18% | ⏳ **sedang diperiksa user** — submit `submission/paper/paper_ieee_extended.pdf`; bila ada bagian flagged, parafrase lalu recompile (`latexmk -pdf main.tex` di `paper/`) |
| Token GitHub | ⚠️ **WAJIB di-revoke user** (terekspos di chat sesi ini) |

## Hasil kunci (jangan diubah tanpa bukti; selalu verifikasi ulang)

- Baseline: SPDT vs CART-custom 6W/0T/2L, Δ+0.0058; Friedman χ²(2)=4.0000,
  p=0.1353; stabilitas prediksi SPDT lebih buruk; fit 21.43s vs 0.53s.
- Ekstensi: normalisasi gagal di data bersih (3W/5L vs CART, Δ−0.0008→
  dibaca −0.0001 di paper untuk norm-cust; norm vs SPDT 2W/6L, Δ−0.0028);
  sinyal noise 20% **bertahan setelah Holm** (p_Holm=0.017) — eksploratori;
  stabilitas split terburuk justru varian ternormalisasi (0.782);
  ANLS terbaik deskriptif (5W/3L, Δ+0.0066); kontrol depth-matched:
  SPDT asli −0.0021 vs kontrol (ada efek nyata kecil), norm +0.0007
  (= efek kedalaman saja).
- Audit independen: verdict MINOR REVISION, 12/13 klaim angka PASS saat
  verifikasi, F1–F4 sudah dieksekusi (Holm, depth-control, repro
  bit-exact ekstensi PASS). Paket lengkap di `audit/`.

## Reproduce (singkat; detail di README.md)

```bash
cd /home/ropix/spdt-paper
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/bin:$PATH"
.venv/bin/python -m pytest -q                     # 37 tests
.venv/bin/python scripts/validate_reproducibility.py            # baseline bit-exact
.venv/bin/python scripts/validate_extension_reproducibility.py  # ekstensi bit-exact
# full rerun bila perlu: run_full.py → run_ablation.py →
#   run_extension_full.py (±2.6 jam) → run_extension_ablation.py
cd paper && latexmk -pdf main.tex && cp main.pdf paper_ieee_extended.pdf
```

## State teknis server

- venv `.venv` (Python 3.12.3; disalin dari venv ALD karena PyPI lambat).
- OpenML TERBLOKIR; Seeds via static zip UCI; 8 dataset ter-cache + hash.
- Paket `balance` TIDAK tersedia di mirror TinyTeX — layout halaman
  terakhir standar (kolom akhir ±85% penuh, itu normal IEEE, jangan
  "diperbaiki" paksa).
- TinyTeX menambah paket baru via `tlmgr install <pkg>`; beberapa paket
  kecil (mis. `xurl`, `caption`) berhasil; yang tidak ada di mirror
  yihui gagal — jangan andalkan paket LaTeX langka.

## Struktur penting

- `submission/` — bundle siap kumpul (PDF + ringkasan CSV + protokol +
  MANIFEST.sha256; regenerasi manifest setiap PDF berubah).
- `audit/` — AUDIT_REPORT, ISSUES_PRIORITIZED (P1: video+Turnitin),
  PRIOR_ART_MATRIX, EXPERIMENT_PLAN (E-1 konfirmatori noise butuh
  persetujuan user), AGENT_FIX_TASKS (F1–F4 selesai; F5 tertunda).
- `docs/` — experiment_protocol.md (freeze + 2 amendemen), video_script,
  prior-art CSV, submission_readiness.
- `results/extension_e2/` — hasil kontrol depth-matched (F3).
- Backup pra-ekstensi: `backups/pre_extension_results_paper.tar.gz`
  (juga di-tag git `pre-extension`).

## Kontak antar-sesi

Sesi baru cukup dibrief: "Lanjutkan proyek SPDT di
/home/ropix/spdt-paper, baca HANDOVER.md." Prioritas berikutnya bila
dilanjutkan: (1) hasil Turnitin → parafrase bila perlu; (2) rekam video;
(3) opsional E-1/E-3 dari audit/EXPERIMENT_PLAN.md (eksperimen
konfirmatori; butuh setuju user karena praregistrasi).
