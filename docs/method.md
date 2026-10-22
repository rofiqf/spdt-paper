# Metode: Stability-Penalized Split Selection

**Status: proposed / candidate formulation.** Hasil gate prior-art (2026-10-22):
tidak ditemukan ekuivalen formula-level dari `Score = Gain − λ·Var_b[Gain]`
dalam literatur yang berhasil diperiksa, tetapi ruang "decision tree
stability" cukup ramai (lihat `docs/prior_art_matrix.csv`). Klaim novelty:
**NOVELTY_NOT_ESTABLISHED** — formulasi diposisikan sebagai kandidat yang
perlu verifikasi lanjutan, dengan tetap mengutip karya terdekat secara
terbuka.

## Definisi

Untuk node dengan dataset D (hanya bagian training yang sah):

- `Gini(D) = 1 − Σ_c p(c|D)²`
- `Gain(s, D) = Gini(D) − (|D_L|/|D|)·Gini(D_L) − (|D_R|/|D|)·Gini(D_R)`
  untuk kandidat split s = (fitur, threshold).

Dari D dibangun B bootstrap resample `D^(b)` (sampling dengan penggantian,
hanya dari baris node). Untuk setiap kandidat s (fitur & threshold tetap):

- `V(s) = Var_b[ Gain(s, D^(b)) ]`  (varians sampel, ddof=1)
- **Kriteria utama:** `Score(s) = Gain(s, D) − λ·V(s)`
- Varian eksploratori terpisah (bukan metode utama):
  `Score_norm(s) = Gain(s, D) − λ·V(s) / (Gain(s, D)² + ε)`

Split terpilih = kandidat valid ber-Score tertinggi **dengan Gain asli > 0**.
Jika tidak ada kandidat yang memenuhi syarat, node menjadi daun.

## Kebijakan deterministik yang di-dokumentasikan

1. Threshold kandidat = titik tengah pasangan nilai unik terurut yang
   berurutan (midpoints adjacent distinct values).
2. `min_samples_leaf` dan `min_samples_split` diterapkan pada **jumlah
   observasi asli node** (bukan jumlah resample).
3. Partisi child kosong di dalam resample memberi kontribusi 0 pada gain
   resample (suku Gini terbobot dihilangkan; Gini(empty) = 0).
4. Daun: kelas mayoritas; seri diputus oleh label terkecil (urutan sorted).
5. Seed bootstrap diturunkan deterministik dari `(random_state, id node)`
   via `numpy.random.SeedSequence`.
6. λ = 0 menonaktifkan seluruh komputasi bootstrap → identik dengan CART
   custom referensi (pencarian kandidat, stopping, tie-breaking sama).

## Mengapa penalti variansi (bukan reward)

V(s) bersifat skala-tergantung (limitasi yang wajib dilaporkan): nilai
absolut gain berbeda antar node/dataset, sehingga λ tidak langsung
terinterpretasi lintas dataset. Justru karena itu λ dipilih via nested CV
per dataset, dan ablation sensitivitas λ dilaporkan apa adanya.

## Alasan desain

Instabilitas pohon kecil terjadi karena split yang terpilih sangat peka
terhadap variasi data (Li 2002; Dietterich 2000). Kriteria yang umum ada:
filter signifikansi (ctree), voting bootstrap per node (NLS/ANLS),
agregasi ensemble (bagging), pembatasan struktur (IFN), atau minimax child
risk. Formulasi ini menempatkan stabilitas sebagai **besaran kontinu
per-split**: mean−variance (risk-averse) scoring langsung pada gain.

## Kompleksitas

Per node per fitur: `O(B·n·C)` untuk membangun tensor rank-resample dan
`O(B·n·C)` kumulatif, dengan n = ukuran node, C = jumlah kelas. Implementasi
vectorized NumPy; brute force sudah memadai untuk dataset kecil–menengah
spec ini.
