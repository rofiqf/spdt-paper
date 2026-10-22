# Experiment Summary — Extension (2026-10-22)

## Yang dijalankan (semua nyata, dari artefak results/extension/)

- **Experimen A** (normalized_penalty.csv, 2160 baris): 8 dataset × 45
  outer fold × 6 metode (CART-skl, CART-cust, SPDT, SPDT-norm, RF, ANLS),
  0 NaN.
- **Experimen B** (noise_robustness.csv, 18000 baris): 5 level × 2
  realisasi × 5 metode (ANLS dikecualikan, terdokumentasi), mask identik,
  test bersih.
- **Experimen C** (split_stability.csv, 1440 baris): S=8 sub-sampel;
  agreement fitur root, variasi threshold, disagreement prediksi.
- **Experimen D** (runtime_tradeoff.csv): refit time, counter kandidat &
  bootstrap-gain-eval, memory probe 2.27 MB.
- **Experimen E** (ablation.csv, 5040 baris): 14 varian, struktur frozen,
  split identik.
- Runtime total Stage 7: ±2,65 jam (4 vCPU); ablation ±6 menit.

## Hasil utama (deskriptif kecuali dinyatakan lain)

| RQ | Jawaban singkat |
|---|---|
| RQ1 | Normalisasi **tidak** meningkatkan Macro-F1 bersih: 3W/5L vs CART (Δ −0.0001), 2W/6L vs SPDT; Friedman χ²(3)=3.75, p=0.290 (ns) |
| RQ2 | Sinyal eksploratori: degradasi paling lambat pada noise 10–20% (median +0.0136 CI [0.0015, 0.0186]; +0.0217 CI [0.0132, 0.0333]); Friedman per-level p=0.024/0.003 — **eksploratori, belum konfirmatori** |
| RQ3 | **Tidak**: varian norm memiliki root-feature agreement terendah (0.782 vs 0.83 CART); disagreement prediksi 0.105 (terburuk kedua) |
| RQ4 | Refit <0.1 s, tetapi tuning staged puluhan fit inner per fold; efek bersih tidak sepadan; sinyal noise mulai mendekati sepadan |
| RQ5 | ANLS (re-implementation faithful) terbaik secara deskriptif di antara pohon tunggal (5W/3L vs CART, Δ +0.0066), stabilitas split tertinggi (0.849) tetapi disagreement prediksi tinggi (0.109) — stabilitas split ≠ stabilitas prediksi |

## Baseline lama (dipegang tetap)

6W/0T/2L, Δ +0.0058; χ²(2)=4.0000, p=0.1353; disagreement 0.0928/0.1014;
fit 0.5340/21.4302 s — tidak diubah, tersimpan di results/summary/.

## Statistik

Konfirmatori: Friedman omnibus (4 primer). Eksploratori (berlabel):
semua perbandingan berpasangan dengan bootstrap-percentile CI 95%;
Friedman per-level noise. p ≥ 0.05 tidak pernah diartikan kesetaraan.
Detail: results/extension/statistical_tests.csv.
