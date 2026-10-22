# Keterbatasan (diperbarui untuk ekstensi, 2026-10-22)

Keterbatasan baseline tetap berlaku (lihat git history versi sebelumnya):
8 dataset kecil, Gini saja, proxy stabilitas tunggal, novelty formula-level
belum established, klaim "belum pernah" dilarang.

## Keterbatasan tambahan ekstensi

1. **Power statistik** tetap 8 dataset; sinyal robustness noise bersifat
   eksploratori (bukan jalur konfirmatori), multi-level, dan perlu
   replikasi independen pada benchmark lebih besar.
2. **Protokol noise**: label dikorupsi independen dari fitur; noise nyata
   sering tidak uniform. Tuning dibekukan dari label bersih — salah satu
   dari beberapa protokol yang defensible.
3. **ANLS**: re-implementation faithful dengan deviasi terdokumentasi
   (cap iterasi 400; cabang kategoris dihilangkan karena data numerik).
   Tidak ada official code release untuk validasi silang.
4. **Waktu refit vs tuning**: tabel runtime memuat refit konfigurasi
   terpilih; overhead tuning staged (S1–S3) terpisah dan mendominasi —
   jangan membandingkan angka refit dengan waktu grid search penuh.
5. **Amendemen protokol pasca-pilot** (RF 100 pohon, 2 realisasi, S=8,
   cap ANLS 400) terdokumentasi, namun tetap deviasi dari rencana awal.
6. **Comparator alternatif** (minimax split, conditional inference trees)
   dianalisis tetapi tidak dibenchmark.
7. Stabilitas split diukur hanya di root; stabilitas node dalam tidak
   dilaporkan.
