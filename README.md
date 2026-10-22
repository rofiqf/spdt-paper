# SPDT — Stability-Penalized Decision Tree

Repositori dokumentasi **Project Mid Semester** mata kuliah Machine
Learning, dosen pengampu **Prof. Dr. Guruh Fajar Shidik, S.Kom., M.Cs.**
(Magister Artificial Intelligence, Universitas Dian Nuswantoro).

> Tugas: pengembangan metode ML yang sederhana namun orisinal, paper
> format prosiding IEEE, dikerjakan individu, mahasiswa sebagai penulis
> pertama + 7 dosen co-author, referensi dari Scopus/IEEE/ACM/
> ScienceDirect/SpringerLink, Turnitin ≤18%.

## Tentang proyek ini

Saya menguji satu ide sederhana pada **pohon keputusan CART**: kalau
gain Gini sebuah calon split "goyang" (bervariasi besar) ketika data
node di-bootstrap ulang, mungkin split itu kurang bisa dipercaya.
Dari situ lahir kriteria kandidat:

```
Score(s) = Gain(s,D) − λ · Var_bootstrap[Gain(s,D^(b))]
```

dan varian ternormalisasinya `Gain − λ·Var/(Gain²+ε)`.

**Hasilnya jujur, tidak semua bagus:**

- Di 8 dataset UCI dengan nested CV (45 evaluasi per dataset-metode),
  penalti variansi **tidak terbukti** meningkatkan Macro-F1
  (Friedman p=0.135 untuk SPDT; p=0.290 untuk varian ternormalisasi).
- Justru muncul satu sinyal menarik: di bawah **label noise 20%**,
  varian ternormalisasi menurun paling lambat (setelah koreksi Holm,
  p=0.017 — tapi tetap eksploratori, perlu replikasi).
- Penalti variansi **tidak** memperbaiki stabilitas split maupun
  prediksi.
- Comparator terdekat dari literatur (ANLS, IEEE Access 2024) yang saya
  re-implementasikan faithful ternyata lebih menjanjikan secara
  deskriptif — dan membuktikan bahwa *stabilitas split ≠ stabilitas
  prediksi*.

Kesimpulan lengkap ada di
[`paper/paper_ieee_extended.pdf`](paper/paper_ieee_extended.pdf)
(IEEE conference, 6 halaman).

## Isi repositori

```
paper/               naskah IEEE + PDF final + references.bib
src/                 implementasi (tree.py, anls.py, noise, stabilitas, engine)
scripts/             runner eksperimen + validasi reproduksibilitas
configs/             konfigurasi beku (baseline + ekstensi)
results/             hasil baseline · results/extension/ hasil ekstensi
figures/             seluruh grafik (termasuk 4 figure ekstensi)
tests/               37 unit tests
docs/                protokol, ringkasan hasil, prior-art, skrip video, audit
audit/               paket audit (verdict MINOR REVISION, bukti, rencana)
submission/          bundle siap kumpul (PDF + ringkasan + manifest checksum)
HANDOVER.md          catatan antar-sesi untuk melanjutkan proyek
```

## Cara menjalankan ulang (reproduce)

```bash
# lingkungan
python3.12 -m venv --without-pip .venv
curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install -r requirements.txt

# tes (37 passed)
.venv/bin/python -m pytest -q

# baseline (±3 jam) lalu ekstensi (±2.6 jam)
.venv/bin/python scripts/run_full.py
.venv/bin/python scripts/run_ablation.py
.venv/bin/python scripts/run_extension_full.py
.venv/bin/python scripts/run_extension_ablation.py

# validasi: hasil harus bit-exact sama dengan CSV tersimpan
.venv/bin/python scripts/validate_reproducibility.py
.venv/bin/python scripts/validate_extension_reproducibility.py

# rebuild paper (butuh TinyTeX/pdflatex)
cd paper && latexmk -pdf main.tex && cp main.pdf paper_ieee_extended.pdf
```

## Penulis

**Rofiq Fauzi** (penulis pertama, mahasiswa) — bersama 7 dosen
co-author dari Universitas Dian Nuswantoro: Guruh Fajar Shidik,
Rindra Yusianto, Tita Talitha, Aripin, Farrikh Alzami,
Ratih Setyaningrum, Firman Wahyudi.

## Catatan integritas

- Seluruh angka dihasilkan dari menjalankan pipeline ter-seed di atas;
  klaim numerik paper diverifikasi ulang terhadap CSV tersimpan
  (lihat `audit/`).
- Penggunaan AI (Kimi, Moonshot AI) diakui di bagian Acknowledgment
  paper: untuk implementasi kode, dukungan penelusuran literatur, dan
  penyuntingan bahasa; semua hasil tetap dieksekusi dan diverifikasi
  secara nyata.
- Klaim kebaruan formulasi **sengaja ditahan** — penelusuran tidak
  menemukan ekuivalen formula-level, tetapi hasil pencarian negatif
  bukan bukti kebaruan.
