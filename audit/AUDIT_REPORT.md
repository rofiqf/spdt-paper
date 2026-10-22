# AUDIT REPORT — paper_ieee_extended.pdf (2026-10-22)

**Verdict: MINOR REVISION** — validitas ilmiah intak, angka terverifikasi,
tidak ada P0; beberapa P1/P2 administratif-statistik sebelum submit.

## Ringkasan bukti

### 1. Konsistensi angka (paper ↔ CSV, verifikasi independen ulang 2026-10-22)

13 klaim numerik dicek ulang dari `results/` mentah: **12 PASS, 1 nit**.

| Klaim paper | Sumber | Hasil |
|---|---|---|
| norm vs CART 3W/5L, Δ−0.0001 | normalized_penalty.csv | PASS |
| norm vs SPDT 2W/6L, Δ−0.0029 | idem | **NIT: mean aktual −0.00284 → −0.0028** (selisih pembulatan 1e-4; tidak mengubah interpretasi) |
| ANLS vs CART 5W/3L, Δ+0.0066 | idem | PASS |
| Friedman χ²(3)=3.75, p=0.290 | statistical_tests.csv | PASS |
| noise 20%: +0.0217, CI [0.0132, 0.0333] | idem | PASS |
| stabilitas: agreement norm 0.782 terendah; ANLS 0.849; disagreement 0.105 vs 0.093 | split_stability.csv | PASS |
| ablation A3n_lam0.1 = 0.8617 tertinggi | ablation.csv | PASS |
| runtime: ~177k bootstrap gain evals; ANLS 2.206 s | runtime_tradeoff.csv | PASS |
| baseline lama χ²(2)=4.0000, p=0.1353, W=0.2500 | results/summary/ | PASS (tidak berubah) |
| konsistensi B(level 0) == A | a vs b | PASS (bit-close) |

### 2. Audit metode (paper ↔ kode)

- Gini/gain/varians/score sesuai `src/tree.py` (normalized: `gain −
  λ·V/(gain²+eps)`, guard errstate + uji near-zero gain). PASS.
- Staged tuning S1–S3 hanya training fold; ANLS parameter default paper
  dengan cap 400 terdokumentasi. PASS.
- Noise: `src/noise.py` deterministik, test fold bersih (test unit
  `test_test_labels_never_modified`, `test_identical_mask`), label hasil
  korupsi selalu berbeda. PASS.
- Potensi catatan: tuning dibekukan dari label bersih = satu dari
  beberapa protokol defensible; sudah diakui di Limitasi §V.

### 3. Audit statistik

- Unit inferensi = dataset (45 fold dirata-rata). PASS.
- Jalur konfirmatori tunggal (Friedman bersih) tidak signifikan; post-hoc
  tidak dijalankan — sesuai protokol. PASS.
- **Temuan P1:** per-level Friedman noise (5 level) tanpa koreksi
  family; paper sudah berlabel "exploratory", tetapi audit merekomendasikan
  eksplisit menambah catatan Holm-across-levels atau menurunkan klaim.
- CI bootstrap percentile atas 8 dataset bersifat deskriptif — sudah
  berlabel; jangan dibaca sebagai bukti konfirmatori.

### 4. Kepatuhan

- Format IEEEtran resmi, 6 halaman, 0 overfull/undefined; author 7
  (mahasiswa first) — PASS.
- Referensi 16: 14 portal izin + Wiley (Scopus-indexed) + SpringerLink —
  PASS (lihat PRIOR_ART_MATRIX.csv).
- **Belum:** Turnitin (tanpa laporan — tidak diklaim), video (skrip ada,
  rekaman belum), tautan eksperimen (repo lokal; push GitHub menunggu
  token).

### 5. Perbandingan versi (ringkas)

| Aspek | ALD-WKNN | SPDT baseline | SPDT extended |
|---|---|---|---|
| Novelty (posisi gate) | tabrakan terdokumentasi (Han 2025) → replikasi | kandidat, ruang ramai | kandidat; normalisasi juga tanpa tabrakan; klaim tetap ditahan |
| Ketelitian eksperimen | nested CV 15 outer | 45 outer + staged tuning | + noise + stabilitas split + ANLS + counter + 2 amendemen terdokumentasi |
| Dampak hasil | deskriptif +, p=0.07 | null jujur | null konfirmatori + sinyal eksploratori noise + pemisahan dua jenis stabilitas |
| Biaya komputasi | ~15 menit | ~3 jam | ~5.5 jam total |
| Risiko penolakan reviewer | klaim novelty lemah | hasil nol sendirian | terkendali; keberatan terbesar (komparator terdekat) sudah dijawab |

## Keterbatasan audit ini

- Audit angka memakai CSV hasil, bukan re-run penuh (reproducibility
  baseline sudah pernah PASS bit-exact; ekstensi belum diuji ulang
  bit-exact penuh — lihat AGENT_FIX_TASKS.md F4).
- ANLS tidak punya official code untuk validasi silang (diklaim
  faithful dari pseudocode; risiko kecil).
- Kesimpulan novelty tetap berbasis pencarian negatif — bukan bukti.
