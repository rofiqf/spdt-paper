# ISSUES PRIORITIZED — Audit SPDT Extended (2026-10-22)

## P0 — menggugurkan validitas

**Tidak ada.** Tidak ditemukan leakage (noise hanya train, mask identik,
tuning training-only, unit inferensi dataset), fabrikasi angka (12/13
klaim PASS, 1 nit pembulatan), atau klaim berlebihan (semua klaim
didukung label konfirmatori/eksploratori).

## P1 — wajib sebelum submit

| # | Masalah | Tindakan | Kriteria selesai |
|---|---|---|---|
| P1-1 | ~~Multiple testing per-level noise~~ **SELESAI (F2)**: Holm across levels; hanya 20% signifikan (p=0.017); paper diperbarui | done | done |
| P1-2 | Basis waktu tidak seragam antar-versi: baseline lama 21.43 s (GridSearchCV total) vs extended 0.068 s (refit saja) — pembaca bisa membandingkan apel-jeruk | Tambah 1 kalimat di §Computational Cost: angka extended = refit konfigurasi terpilih; tuning staged (S1–S3) terpisah dan mendominasi; bandingkan hanya dalam versi yang sama | kalimat ada; tidak ada tabel yang membandingkan antar basis |
| P1-3 | Video penjelasan belum direkam (deliverable course) | Rekam per skrip `docs/video_script.md` (±8 menit) | file video terserahkan |
| P1-4 | Turnitin belum ada laporan | Submit `submission/paper/paper_ieee_extended.pdf` ke institusi | laporan similarity ≤18% atau parafrase bagian flagged |

## P2 — penyempurnaan (tidak menghalangi submit)

| # | Masalah | Tindakan |
|---|---|---|
| P2-1 | ~~Δ −0.0029~~ **SELESAI (F1)**: kini −0.0028 | done |
| P2-2 | ANLS tanpa validasi silang official code | tambah 1 kalimat limitasi (sudah ada sebagian): "no official code release to cross-validate" |
| P2-3 | S=8 (extended) vs S=10 (legacy) membandingkan disagreement antar-versi | sudah dibedakan sebagai eksperimen terpisah; tambah footnote 1 angka di Tabel stabilitas extended |
| P2-4 | Repo belum di-push (experiment link) | push ke GitHub setelah token tersedia |
| P2-5 | CI bootstrap 8 dataset rapuh terhadap outlier dataset | pertimbangkan laporkan juga median Δ di §Results (sudah ada di tabel statistik; cukup rujuk) |
| P2-6 | Gambar noise memakai mean sederhana | opsional: tambah pita min-max antar dataset di caption (tidak wajib) |

## Rekomendasi eksperimen lanjutan (paling bernilai)

Lihat `EXPERIMENT_PLAN.md` (maks 3): konfirmatori noise ter-praregistrasi
+ dataset tambahan; kontrol kedalaman; robustness feature-noise.
