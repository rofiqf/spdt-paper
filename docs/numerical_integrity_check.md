# Numerical Integrity Check — IEEE Revision (2026-10-22)

Metodologi: tabel LaTeX di-**generate** dari artefak `results/` oleh
`scripts/make_ieee_tables.py` (satu sumber kebenaran), lalu setiap
kelompok angka diverifikasi ulang terhadap CSV oleh skrip ini.
Kesimpulan: **SEMUA PASS — angka identik dengan paper sebelum revisi**.

| Item | Status | Detail |
|---|---|---|
| Dataset inventory (8 dataset, N/fitur/kelas) | PASS | semua baris metadata ada di Tabel I |
| Macro-F1 per dataset x metode + delta SPDT-CART | PASS | 32 nilai mean + 8 delta cocok dengan fold_metrics.csv |
| Friedman chi2=4.0000, p=0.1353, W=0.2500, df=2 | PASS | cocok statistical_tests.csv |
| Prediction disagreement (4 metode, mean+SD) | PASS | SPDT 0.1014 > CART-cust 0.0928 (negatif, dipertahankan) |
| Fit/predict time (4 metode) | PASS | SPDT 21.43s vs CART-cust 0.53s (~40x) dipertahankan |
| Ablation (9 varian: F1 + fit time) | PASS | cocok ablation.csv; semua varian hadir |
| Hyperparameter frequencies (24 baris penuh) | PASS | seluruh tabel dipertahankan di paper (Tabel VIII) |
| W/T/L = 6/0/2, mean delta +0.0058 | PASS | cocok; klaim deskriptif tidak berubah |

Catatan: tidak ada satu pun angka eksperimen yang diubah dalam revisi
editorial ini; semua perbedaan bersifat format (penyajian, jumlah
desimal konsisten 4, label metode pendek). Klaim statistik tidak
berubah: Friedman p=0.1353 (n.s.), stabilitas SPDT lebih buruk,
overhead ~40x, hasil nol dilaporkan apa adanya.