# Protocol Freeze — Stage 5 (2026-10-22)

Protokol eksperimen penuh dibekukan setelah pilot (Stage 4) dan tidak boleh
diubah berdasarkan hasil outer-test. File resmi: `configs/experiment.yaml`.

## Ringkasan pilot yang menginformasi freeze

- Fit median terukur (benchmark mentah, 5 ulangan):
  - iris (150×4):  λ>0, B=10 → 0.009 s; λ=0 → 0.002 s
  - wine (178×13): λ>0, B=10 → 0.033 s; λ=0 → 0.008 s
  - wdbc (569×30): λ>0, B=10 → 0.19 s; λ=0 → 0.039 s
- Proyeksi runtime full grid (540 konfigurasi spdt × 45 outer fold × 8
  dataset, paralel 4 vCPU): ±3–5 jam — dapat diterima, grid praregistrasi
  TIDAK disempitkan berdasarkan performa.
- Pipeline nested CV terbukti jalan end-to-end; hasil pilot bersifat
  diagnostik dan tidak digunakan sebagai klaim temuan.

## Keputusan freeze

1. Outer: 5 folds × 3 repeats, tiga seed terpisah [42, 123, 2026]
   (RepeatedStratifiedKFold per seed) → 45 evaluasi outer per dataset-metode.
2. Inner: StratifiedKFold 3-fold, seed diturunkan deterministik
   `SeedSequence([seed, repeat, fold])`.
3. Grid spdt penuh sesuai spec: λ ∈ {0, 0.01, 0.1, 1, 10},
   B ∈ {5, 10, 20}, max_depth ∈ {3, 5, 8, None},
   min_samples_leaf ∈ {1, 2, 5}, min_samples_split ∈ {2, 5, 10}.
   λ = 0 di grid ganda sebagai ablation A0 yang tersimpan otomatis.
4. Metode primer perbandingan: cart_sklearn, cart_custom (λ=0),
   spdt; random_forest = benchmark kontekstual opsional (diaktifkan).
5. Scoring tuning: f1_macro (sama dengan metrik primer).
6. Preprocessing: SimpleImputer median dalam Pipeline sklearn (fit hanya
   pada training fold); tanpa standardisasi (pohon).
7. Stabilitas: S=10 resample sub-sampel 90% per outer fold, prediksi pada
   outer test yang sama; disagreement pairwise dilaporkan terpisah.
8. Metrik tersimpan per fold: f1_macro, balanced_accuracy, accuracy,
   fit/predict time, node_count, tree_depth, best_params, inner_seed.

## Larangan tetap berlaku

- Tidak menyempitkan grid berdasarkan hasil outer test.
- Tidak menambah/mengurangi dataset setelah freeze.
- p ≥ 0.05 tidak boleh diartikan sebagai bukti kesetaraan.
- Fold = observasi deskriptif; inferensi pada level dataset.

## Addendum pasca-eksperimen (2026-10-22)

Validasi reproducibility menemukan bahwa `sklearn.tree.
DecisionTreeClassifier(random_state=None)` **nondeterministik antar-fit**
(tie-breaking mengonsumsi RNG global) pada sklearn 1.9.1, sehingga baris
`cart_sklearn` hasil run utama (2.9 jam) tidak dapat direproduksi bit-exact.
Tindakan: `make_sklearn_cart` diberi `random_state=0` (default tetap),
metode `cart_sklearn` dijalankan ulang di seluruh dataset dengan outer
splits identik (`scripts/rerun_method.py`), dan baris lama diganti secara
atomik. Metode lain (`cart_custom`, `spdt`, `random_forest`) ternyata sudah
deterministik (seed eksplisit) dan hasilnya tidak berubah. Keputusan ini
bukan tuning berdasarkan outer test — hanya perbaikan determinisme.
