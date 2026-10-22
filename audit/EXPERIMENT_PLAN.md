# EXPERIMENT PLAN — Maksimal 3 Eksperimen Tambahan Paling Bernilai

Dipilih berdasarkan audit: (1) mengangkat sinyal eksploratori terkuat ke
status konfirmatori, (2) menutup keberatan "efeknya sebenarnya dari
kedalaman?", (3) melengkapi RQ2 tanpa biaya besar. Semua reusable
infrastruktur yang ada.

## E-1. Konfirmatori Label-Noise (PRIORITAS TERTINGGI)

- **Hipotesis (praregistrasi):** pada noise label 20% (level tunggal yang
  dipilih sebelum melihat data), SPDT-norm > CART-custom pada Macro-F1
  per-dataset (paired, dataset sebagai unit).
- **Kontrol:** protokol persis Experimen B yang ada (mask identik, test
  bersih, tuning frozen bersih) kecuali: level TUNGGAL 20%, 5 realisasi
  (naik dari 2), Holm correction atas family kecil yang didefinisikan
  sebelumnya (2 perbandingan primer: norm vs CART, norm vs SPDT).
- **Tambahan dataset (provenance terverifikasi via ucimlrepo/UCI API —
  jalan di server ini):** Spambase (UCI 94, 4601×57×2) dan Wine Quality
  red (UCI 186, 1599×11×6). Sama-sama memenuhi syarat course (UCI,
  klasifikasi, lisensi terdokumentasi). Ini juga menjawab kelemahan
  "hanya 8 dataset kecil" secara terbatas.
- **Metrik:** Macro-F1 primer; balanced accuracy; degradasi vs bersih.
- **Seed:** corruption seeds lanjutan (realization 3..7 via
  corruption_seed yang ada), split identik.
- **Statistik:** Wilcoxon signed-rank per-dataset + Holm (family 2);
  laporkan effect size r_rb dan CI bootstrap.
- **Biaya:** 10 dataset × 45 fold × (3 metode primer × 5 realisasi)
  refit + tuning ulang untuk 2 dataset baru. Estimasi: tuning baru 2
  dataset ≈ 45×(S1-S3) ≈ 15-25 menit; noise ≈ 10×45×15×0.1s ≈ 11 menit.
  **Total ≈ 30-45 menit.**

## E-2. Kontrol Kedalaman (Complexity-Matched CART)

- **Hipotesis:** jika sebagian efek SPDT/SPDT-norm berasal dari pembatasan
  kompleksitas implisit, CART dengan kedalaman disamakan (depth-matched)
  akan menyamai penalti variansi pada data bersih.
- **Kontrol:** untuk setiap outer fold, depth dari konfigurasi terpilih
  SPDT dipakai sebagai max_depth CART-custom; evaluasi pada split identik
  (bisa dikerjakan murni dari logs/extension_tuned_params.json — tidak
  perlu tuning ulang).
- **Metrik:** Macro-F1; paired diff vs SPDT dan vs SPDT-norm.
- **Seed:** reuse penuh (deterministik).
- **Statistik:** deskriptif + Wilcoxon eksploratori (diberi label).
- **Biaya:** 8×45 refit CART cepat. **Total ≈ 2-3 menit.**

## E-3. Robustness Feature-Noise (melengkapi RQ2)

- **Hipotesis:** sinyal robustness SPDT-norm juga muncul (atau lebih
  kuat) ketika fitur (bukan label) dikorupsi Gaussian dengan σ relatif
  {0.05, 0.1, 0.2} terhadap std fitur train — protokol tanpa label
  noise, lebih dekat ke sensor noise nyata.
- **Kontrol:** korupsi hanya X train (fit std train, seed deterministik
  per fold/realisasi), mask identik antar metode, test bersih, tuning
  frozen; 2 realisasi; level {0, 0.05, 0.1, 0.2}.
- **Metrik:** Macro-F1 + degradasi; perbandingan CART vs SPDT vs SPDT-norm.
- **Statistik:** eksploratori (CI bootstrap), eksplisit tidak konfirmatori
  kecuali E-1 memberi landasan.
- **Biaya:** 8×45×4 level×2 realisasi×3 metode refit. **Total ≈ 10-15
  menit** (implementasi corrupt_features ≈ 20 baris di src/noise.py).

## Urutan & gate

E-2 dulu (murah, langsung) → E-3 (murah) → E-1 (butuh perhatian
praregistrasi). Bila E-1 konfirmatori tidak signifikan, klaim robustness
ditarik sepenuhnya dari diskusi (paper sudah hati-hati; ini ujian
keberanian ilmiahnya). Semua output ke results/extension_e{1,2,3}/ baru —
tidak menimpa apa pun.
