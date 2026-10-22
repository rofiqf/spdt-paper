# Submission Readiness — SPDT IEEE Paper (2026-10-22)

**Status keseluruhan: BELUM "SUBMISSION READY"** — gate author metadata
belum terpenuhi (lihat bawah). Secara teknis/format: lengkap.

## Quality gates

- [x] Official IEEE conference class/template: IEEEtran.cls (conference) + IEEEtran.bst
- [x] Author block proper (\IEEEauthorblockN/A) — nama mahasiswa sebagai first author
- [x] Abstract + IEEEkeywords benar
- [x] Semua persamaan typeset (eq. 1-4), notasi statistik benar
- [x] Sitasi numerik IEEE [1]-[16] urutan kemunculan pertama
- [x] Referensi IEEE bernomor; 15/16 dari portal izin; 1 buku fondasi ditandai
- [x] Tabel muat margin, caption konsisten, dekat pembahasan; 0 overfull
- [x] Tanpa placeholder yatim/caption kosong
- [x] Hasil negatif & limitasi dipertahankan; klaim tidak berlebihan
- [x] Integritas numerik 8/8 PASS (docs/numerical_integrity_check.md)
- [x] Semua halaman diinspeksi visual (render PNG per halaman)
- [x] Perintah kompilasi terdokumentasi (paper/README.md)
- [ ] **Author metadata lengkap: 6 co-author fakultas BELUM dikonfirmasi user**
- [ ] **Turnitin: UNVERIFIED — tidak ada laporan aktual; jangan klaim lolos**

## TODO wajib sebelum submit

1. **TODO-AUTHORSHIP**: konfirmasikan identitas 6 dosen co-author ke user,
   lalu tambahkan di `\author` block `paper/main.tex` (template penambahan
   sudah disiapkan sebagai komentar TODO di file). First author tetap
   mahasiswa.
2. **TODO-TURNITIN**: submit `paper_ieee_final.pdf` ke Turnitin institusi;
   jika >18%, parafrase bagian yang di-flag. Tidak bisa dijalankan dari
   server ini.
3. Opsional: hyperlink eksperimen (repo GitHub) & video penjelasan sesuai
   permintaan course deliverables.

## Risiko saintifik yang tersisa (disampaikan apa adanya)

- Klaim novelty **belum established** — pencarian formula-level negatif
  bukan bukti kebaruan (paper menyatakan ini eksplisit).
- Tidak ada bukti peningkatan signifikan (Friedman p=0.1353); keunggulan
  bersifat deskriptif.
- Stabilitas prediksi SPDT sedikit **lebih buruk** dari CART custom.
- Overhead komputasi ~40x.
- Comparator terdekat literatur (ANLS, minimax split) belum dijalankan.
- CART sklearn sempat nondeterministik (sudah di-fix + rerun; lihat
  docs/protocol_freeze.md addendum).
