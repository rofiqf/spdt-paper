# Keterbatasan yang Wajib Dilaporkan

1. **Novelty tidak established.** Prior-art gate (2026-10-22) tidak menemukan
   ekuivalen formula-level, tetapi beberapa karya sangat dekat secara konsep
   (ANLS Lee et al. 2024; NLS/Dannegger via Briand et al. 2009; era splitting
   Delise 2026; MinimaxSplit Luo 2025; ctree Hothorn 2006; VFDT Domingos &
   Hulten 2000). Paper tidak boleh mengklaim metode baru secara mutlak.
2. **V(s) scale-dependent.** Variansi gain bergantung pada skala gain itu
   sendiri; λ tidak portable antar dataset/node. Penyetelan λ per dataset
   melalui nested CV adalah keharusan, bukan pilihan.
3. **Biaya komputasi.** Setiap node mengevaluasi B resample untuk semua
   kandidat split; overhead besar vs CART murni dan tumbuh dengan ukuran
   node × jumlah fitur × B.
4. **Unit inferensi terbatas.** 8 dataset (target ≥6) kecil–menengah;
   kekuatan uji Friedman/Wilcoxon rendah; hasil p ≥ 0.05 TIDAK boleh
   diartikan sebagai bukti kesetaraan.
5. **Fold CV berkorelasi.** 45 outer fold per dataset-method bersifat
   deskriptif; rata-rata per dataset adalah satu-satunya unit inferensi.
6. **Metrik stabilitas terpisah.** Disagreement antar model ter-resample
   dilaporkan terpisah dari akurasi; penalti bootstrap ≠ bukti stabilitas
   terbukti meningkat secara kausal.
7. **Hanya impuritas Gini, axis-parallel, numeric.** Tidak menangani
   kategorik native, missing value internal (imputasi median upstream),
   pruning pasca-pohon, atau metrik lain.
8. **Bootstrap resample dari node kecil.** Untuk node kecil, gain resample
   berisiko degenerate/bernilai sama; kebijakan empty-child sudah
   ditetapkan tetapi tetap kaku.
9. **Varian eksploratori normalized penalty** belum divalidasi sebagai
   metode utama dan hanya boleh dibahas sebagai eksplorasi.
10. **Belum tentu berlaku** untuk dataset berskala besar/high-dimensional
    modern; semua dataset benchmark berukuran kecil (150–1.372 sampel).
