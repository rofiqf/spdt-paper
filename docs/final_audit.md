# Final Audit — Proyek SPDT (2026-10-22)

- [x] 8 dataset terunduh, skema + SHA256 tervalidasi (`data/metadata/`, `data/hashes/`)
- [x] Reproducibility unit tests lulus (16 passed)
- [x] No-leakage tests lulus (outer splits deterministik, imputer train-only,
      tuning invarian terhadap konten outer test)
- [x] Custom CART (λ=0) identik dengan implementasi referensi independen
- [x] Protokol freeze tertulis (`docs/protocol_freeze.md`)
- [x] Main nested CV: 1440 fold-rows (8 × 4 × 45), 0 NaN, 0 fold gagal
- [x] Stabilitas: 1440 baris disagreement, dilaporkan terpisah dari akurasi
- [x] Ablation pada split identik: 3240 baris (9 varian × 45 × 8)
- [x] Statistik level-dataset: Friedman + mean ranks; post-hoc HANYA bila
      omnibus signifikan (tidak terpenuhi → tidak dijalankan, sesuai protokol)
- [x] Fold-level tidak pernah dipakai sebagai sampel independen
- [x] Fix determinisme: cart_sklearn di-re-run dengan random_state tetap;
      baris lama diganti atomik (`scripts/rerun_method.py`)
- [x] Validasi reproducibility bit-exact terhadap artefak tersimpan
- [x] Semua angka paper terlacak ke artefak results/ (build_paper gagal
      bila artefak hilang — tidak ada angka fabrikasi)
- [x] Paper IEEEtran 2 kolom, 0 error kompilasi, caption tabel benar
- [x] Prior-art gate + matrix lengkap (`docs/prior_art_matrix.csv`),
      status NOVELTY_NOT_ESTABLISHED, karya terdekat dikutip terbuka
- [x] Referensi hanya dari portal izin (IEEE/ACM/ScienceDirect/Springer)
      + 1 buku fondasi yang ditandai; karya arXiv/T&F tidak dikutip bernomor
- [x] Hasil negatif dilaporkan (stabilitas tidak membaik; Friedman ns)
- [ ] **Item terbuka untuk user**: isi identitas `\author{Anonymous}` di
      `paper/paper.tex` (atau regenerate via build_pdf.sh setelah mengubah
      -V author), cek Turnitin institusi terhadap `paper/paper.pdf`
