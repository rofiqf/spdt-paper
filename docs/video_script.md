# Skrip Video Penjelasan — SPDT (±8 menit)

Bahasa: Indonesia. Gaya: presentasi tugas kuliah. Semua visual sudah ada
di repo (figures/, tabel di paper) — tidak perlu membuat gambar baru.

---

## SLIDE 1 — Judul (0:00–0:30)

**Visual:** judul paper: *Evaluating Normalized Bootstrap Gain-Variance
Penalties for Robust Decision Tree Classification* + nama + NIM + dosen.

**Narasi:**
> "Assalamu'alaikum. Saya Rofiq Fauzi. Presentasi ini membahas evaluasi
> reproducible terhadap sebuah modifikasi kriteria split pada pohon
> keputusan, yaitu penalti variansi bootstrap atas gain Gini — termasuk
> varian ternormalisasinya — dan perbandingannya dengan metode
> stabilisasi published terdekat, ANLS."

## SLIDE 2 — Latar Belakang (0:30–1:30)

**Visual:** satu kalimat besar: *"Pohon keputusan tidak stabil: perubahan
kecil data → split berubah → prediksi berubah."*

**Narasi:**
> "Pohon keputusan CART cepat dan interpretabel, tetapi strukturnya
> tidak stabil: perturbasi kecil pada data latih dapat mengubah split
> yang terpilih. Ini fenomena lama yang melatarbelakangi bagging dan
> random forest. Pertanyaan penelitian kami lebih sempit: bisakah
> ketidakstabilan itu ditekan langsung di kriteria split, tanpa ensemble?"

## SLIDE 3 — Metode: SPDT & Varian Normalisasi (1:30–3:00)

**Visual:** persamaan (1)–(5) dari paper, satu per satu muncul.
`Score = Gain − λ·Var_bootstrap[Gain]`, lalu
`Score_norm = Gain − λ·Var/(Gain²+ε)`.

**Narasi:**
> "Formulasi kandidat kami, SPDT: skor split = gain Gini dikurangi
> lambda kali variansi gain yang dihitung dari B resample bootstrap
> data node. Neighbor yang gain-nya rapuh terhadap resampling
> dipenalti. Masalahnya: penalti ini bergantung skala gain, sehingga
> lambda sulit diartikan lintas dataset. Maka kami uji hipotesis
> kedua — varian ternormalisasi yang membagi penalti dengan
> Gain kuadrat plus epsilon. Ini hipotesis eksploratori, bukan klaim
> metode baru. Ketiga, kami bandingkan dengan ANLS, metode published
> 2024 di IEEE Access, yang kami re-implementasikan secara faithful
> dari pseudocode paper-nya."

## SLIDE 4 — Desain Eksperimen (3:00–4:00)

**Visual:** Tabel I (8 dataset) + skema nested CV sederhana
(teks: outer 5×3×3 seed = 45 evaluasi; inner 3-fold tuning).

**Narasi:**
> "Evaluasi memakai 8 dataset UCI, nested cross-validation berulang —
> 45 evaluasi per dataset-metode, split identik untuk semua metode,
> tuning hanya di inner loop, tanpa leakage. Unit statistiknya dataset,
> bukan fold. Untuk robustness, label latih dikorupsi 5 sampai 30
> persen dengan mask yang sama untuk semua metode, test tetap bersih."

## SLIDE 5 — Hasil 1: Data Bersih (4:00–5:00)

**Visual:** Tabel extended (Macro-F1 per dataset) + baris hasil
Friedman: χ²(3)=3.75, p=0.29.

**Narasi:**
> "Di data bersih, hasilnya jujur: varian ternormalisasi TIDAK
> meningkatkan Macro-F1 — tiga menang, lima kalah dari CART, rata-rata
> selisih hampir nol. Uji Friedman tidak menemukan perbedaan signifikan.
> ANLS secara deskriptif terbaik di antara pohon tunggal, tapi tetap
> tidak signifikan secara statistik."

## SLIDE 6 — Hasil 2: Robustness Noise (5:00–6:00)

**Visual:** `figures/noise_robustness_macro_f1.png` (layar penuh).

**Narasi:**
> "Di sinilah sinyal menarik muncul. Grafik Macro-F1 melawan level
> noise: semua metode menurun, tetapi SPDT ternormalisasi — garis hijau
> — menurun paling lambat di antara pohon tunggal. Pada noise 10 sampai
> 20 persen, selisih berpasangan terhadap CART punya confidence
> interval yang tidak memuat nol. Penting: ini analisis eksploratori —
> kami tidak mengklaim robust; sinyal ini butuh replikasi konfirmatori."

## SLIDE 7 — Hasil 3: Stabilitas (6:00–7:00)

**Visual:** dua figure berdampingan: `root_split_agreement.png` dan
`prediction_disagreement.png`.

**Narasi:**
> "Untuk stabilitas, hasilnya mematahkan asumsi: penalti variansi justru
> TIDak menstabilkan — varian ternormalisasi punya kesepakatan fitur
> root terendah. Dan ANLS, yang paling stabil di split, justru punya
> disagreement prediksi tinggi. Kesimpulan empiris: stabilitas split
> dan stabilitas prediksi adalah dua hal berbeda — temuan yang kami
> laporkan terbuka."

## SLIDE 8 — Kesimpulan (7:00–8:00)

**Visual:** 3 bullet: (1) normalisasi gagal di data bersih, (2) sinyal
robustness eksploratori di noise, (3) voting per-node (ANLS) lebih
menjanjikan daripada penalti variansi.

**Narasi:**
> "Kesimpulan: di kondisi teruji, CART sederhana tetap preferable untuk
> data bersih; satu-satunya sinyal manfaat adalah degradasi lebih lambat
> di bawah label noise 10 sampai 20 persen — eksploratori; dan arah
> stabilisasi yang lebih menjanjikan adalah voting per-node ala ANLS,
> bukan penalti variansi gain. Semua kode, data, dan hasil tersedia
> publik dan reproducible. Terima kasih."

---

## Catatan produksi

- Durasi baca narasi: ±8 menit normal; boleh dipangkas dengan
  melewati Slide 2.
- Semua visual ada di `figures/` dan tabel di `paper/paper_ieee_extended.pdf`
  (screenshot bagian tabel untuk slide).
- Rekam suara + slide (PowerPoint/Canva/Google Slides) cukup; tidak
  perlu animasi.
- Jika dosen meminta demo kode singkat: jalankan
  `.venv/bin/python -m pytest -q` (37 passed) sebagai bukti reproducibility.
