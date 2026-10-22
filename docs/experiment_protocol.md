# Experiment Protocol — Extension (FROZEN 2026-10-22, Stage 6)

Di-freeze setelah pilot Stage 5; perubahan setelah ini wajib didokumentasikan
sebagai deviasi.

## Desain umum (mewarisi baseline)

- 8 dataset UCI yang sama + hash yang sama (iris, wine, seeds, glass,
  sonar, ionosphere, wdbc, banknote).
- Outer: RepeatedStratifiedKFold 5 fold x 3 repeats, seed [42,123,2026]
  -> 45 evaluasi per dataset-metode; split IDENTIK untuk semua metode
  (fungsi outer_splits yang sama dengan baseline).
- Inner: StratifiedKFold 3-fold, seed = inner_seed(seed, repeat, fold).
- Preprocessing: median imputer dalam Pipeline (fit training only).
- Unit inferensi: dataset (rata-rata 45 fold), bukan fold.

## Metode

- M0 cart_sklearn: structural GridSearchCV (36 konfigurasi).
- M1 cart_custom: lambda=0 (fast path, tanpa bootstrap).
- M2 spdt: penalti variansi asli.
- M3 spdt_norm: penalti ternormalisasi (eps=1e-4 saat tuning).
- M4 random_forest: 300 pohon, kontekstual (bukan ekuivalen).
- M5 anls: parameter default paper (Z=1.645, min 10 iter, min-leaf 1%,
  cap 1500); TIDAK ditune.

## Staged tuning (deviasi terdokumentasi dari Cartesian penuh)

S1 struktur {depth x leaf x split} via CART custom lambda=0 -> theta*.
S2 lambda in {0.1, 1, 10} untuk M2 pada theta*, B=10.
S3 lambda in {0.1, 1, 10} untuk M3 pada theta*, B=10, eps=1e-4.
Motivasi: kendali runtime; semua keputusan hanya dari training fold.

## Eksperimen

- A normalized_penalty: evaluasi bersih 6 metode + root split + counter
  kandidat/bootstrap-gain-eval. Output: normalized_penalty.csv.
- B noise_robustness: level {0,.05,.10,.20,.30} x 2 realisasi, mask
  identik antar metode, test bersih, parameter frozen. ANLS DIKECUALIKAN
  (biaya 75 fit bootstrap-heavy per fold tidak proporsional; fokus M0-M3).
  Output: noise_robustness.csv.
- C split_stability: S=8 sub-sampel 90%, agreement fitur root, variasi
  threshold (fitur modal saja), disagreement prediksi. Output:
  split_stability.csv.

## Amendemen pasca-pilot (2026-10-22, sebelum full run)

Pilot iris memakan ~13 menit; bottleneck = RandomForest 300 pohon dievaluasi
86x per fold (A:1, B:75, C:10). Amendemen (dokumentasi wajib):
1. rf_trees 300 -> 100 (hanya ekstensi; hasil RF baseline lama tidak diubah);
2. noise_realizations 3 -> 2;
3. stability_subsamples 10 -> 8.
Proyeksi penuh setelah amendemen: ~2.5-4 jam. Protocol dianggap FROZEN
setelah amendemen ini; deviasi berikutnya (bila ada) wajib didokumentasikan
ulang.
- D runtime_tradeoff: agregasi waktu + counter + peak memory probe
  (tracemalloc, satu proses, wdbc) — estimasi terdokumentasi, bukan
  benchmark terkontrol. Output: runtime_tradeoff.csv.
- E ablation: struktur frozen per fold (dari tuned params JSON),
  varian lambda/B/eps/depth/leaf pada split identik. Output: ablation.csv.

## Statistik

Konfirmatori: Friedman omnibus atas 4 metode primer (M0,M1,M2,M3) per
eksperimen; post-hoc Wilcoxon+Holm HANYA bila omnibus signifikan.
Deskriptif/eksploratori (diberi label): semua perbandingan berpasangan
dengan bootstrap-percentile CI 95% atas delta level-dataset; Friedman
per-level noise; degradasi vs kondisi bersih. p>=0.05 tidak pernah
diartikan kesetaraan.

## Larangan tetap

Tidak tuning pada outer test; tidak menyembunyikan hasil negatif; tidak
mengklaim novelty tanpa verifikasi formula-level; ANLS tidak dilabeli
karya kami.
