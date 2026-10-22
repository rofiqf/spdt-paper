# AGENT FIX TASKS — urutan eksekusi siap-jalan (2026-10-22)

Setiap tugas: implementasi → pengujian → artefak keluaran. Tidak ada
yang menimpa hasil lama; semua tulis ke lokasi baru atau regenerasi
dokumen.

## F1 — Koreksi numerik P2-1 + catatan P1-1/P1-2/P2-2 (±20 menit)

1. `paper/main.tex`: ubah `-0.0029` → `-0.0028` (Δ norm vs SPDT).
2. §Robustness Analysis: tambah kalimat: "Five noise levels were examined
   as an exploratory family without across-level correction; per-level
   p-values are descriptive. A single pre-registered confirmatory level
   is left to future work (see Experiment E-1 in the project audit)."
3. §Computational Cost: tambah kalimat basis waktu: "Extension timings
   report refits of tuned configurations; the staged tuning search
   (S1–S3) is the dominant cost and is reported separately from the
   baseline study's full-grid timings; cross-version timing comparisons
   use different bases and must not be read as controlled."
4. §Limitations: tambah "no official ANLS code release exists for
   cross-validation of our re-implementation."
- **Pengujian**: recompile `latexmk -pdf main.tex` → 0 overfull, 0
  undefined; grep angka baru ada.
- **Artefak**: `paper/paper_ieee_extended.pdf` (refresh), submission
  bundle + MANIFEST diregenerasi.

## F2 — Holm across noise levels di statistik (±15 menit, tanpa rerun)

1. `src/extension_statistics.py`: pada ringkasan B, tambah baris
  `B_friedman_per_level_holm`: kumpulkan p per 5 level, terapkan
  `multipletests(method='holm')`, tulis p_holm.
2. Jalankan ulang `summarize_extension` memakai CSV yang ada (fungsi
  murni, tidak perlu fit ulang) — skrip kecil
  `scripts/refresh_extension_stats.py`.
- **Pengujian**: `statistical_tests.csv` memuat kolom p_holm untuk
  family noise; angka p mentah tidak berubah.
- **Artefak**: `results/extension/statistical_tests.csv` (versi baru),
  tercatat di experiment_summary.md.

## F3 — Eksperimen E-2 kontrol kedalaman (±15 menit)

1. `scripts/run_depth_matched_control.py`: baca
  `logs/extension_tuned_params.json`; untuk tiap (dataset, seed, repeat,
  fold): fit CART-custom λ=0 dengan max_depth = depth hasil tuning SPDT
  (dari struct); evaluasi pada test fold; tulis
  `results/extension_e2/depth_matched.csv` (baris per metode
  cart_custom_depthmatched + salinan spdt/spdt_norm dari A untuk
  perbandingan paired).
2. Hitung paired diff deskriptif; tambah 1 paragraf di §Results/Discussion
  (dengan angka aktual, tanpa klaim baru).
- **Pengujian**: unit test kecil (depth benar terpasang; split identik
  dengan outer_splits).
- **Artefak**: CSV E-2 + (opsional) 1 kalimat di paper.

## F4 — Uji reproduksibilitas bit-exact ekstensi (±10 menit)

1. `scripts/validate_extension_reproducibility.py`: mirip versi baseline —
  rehitung 3 probe fold (mis. iris/cart_custom, wine/spdt_norm,
  banknote/anls) dan bandingkan dengan normalized_penalty.csv.
  Catatan: ANLS diharapkan bit-identik (semua seeded); bila RF ikut
  dipilih sebagai probe, gunakan toleransi (RF fit sendiri deterministik
  dengan random_state, harus bit-identik juga).
- **Pengujian**: PASS/FAIL jelas; FAIL → investigasi sebelum lanjut.
- **Artefak**: hasil dicatat di audit/AUDIT_REPORT.md (seksi bukti).

## F5 — Eksekusi E-1 & E-3 (butuh keputusan user, ±1 jam komputasi)

Hanya setelah F1–F4 bersih dan user menyetujui praregistrasi E-1
(level 20%, family Holm 2 perbandingan, dataset tambahan Spambase +
Wine Quality red). Implementasi: `src/noise.py: corrupt_features`,
config `configs/extension_e1.yaml`, runner adaptasi dari
extension_evaluation. Lihat `audit/EXPERIMENT_PLAN.md` untuk protokol
lengkap. **Gate: tidak mulai sebelum persetujuan tertulis user.**

## Urutan eksekusi yang disarankan

F1 → F2 → F4 → F3 → (F5 setelah persetujuan). Estimasi total tanpa F5:
±1 jam kerja agent.
