# IEEE Format Audit — Revisi Editorial (2026-10-22)

Sumber sebelum revisi: `paper/template_en.md` + `scripts/build_paper.py`
(pandoc pipeline) → `paper/paper_en.pdf` (setara dengan "paper_en(1).pdf"
yang disebut dalam tugas). Backup tak-terubah: `paper/paper_en_v1_backup.pdf`,
snapshot git `48b7b5a`.

Naskah baru ditulis ulang sebagai LaTeX asli: `paper/main.tex` +
`paper/tables_generated.tex` (generate dari `results/`) →
`paper/paper_ieee_final.pdf` (IEEEtran conference resmi, 4 halaman).

## Defect → status perbaikan

| # | Defect (dari brief) | Lokasi lama | Status |
|---|---|---|---|
| 1 | Title/author block bukan format IEEE conference | judul = section biasa; author 1 baris | FIXED: `\IEEEauthorblockN/\IEEEauthorblockA`, affiliation Semarang, email mahasiswa |
| 2 | Abstract/keywords order & style | "## Abstrak" heading; keywords paragraf | FIXED: `abstract` env (bold "Abstract—") + `IEEEkeywords` ("Index Terms—") |
| 3 | Matematika plain text (lambda*V(s), chi^2, D^(b)) | inline code | FIXED: eq. (1)–(4) numbered; `\operatorname{Gini}`, `\operatorname{Var}_b`, `\chi^2`, `D^{(b)}`, boxed score; λ, B, D, s dijelaskan |
| 4 | Sitasi author-year | `[@key]` pandoc | FIXED: `\cite{...}` numerik IEEE via IEEEtran.bst |
| 5 | Referensi tidak bernomor/bukan IEEE style | daftar tanpa nomor | FIXED: `\bibliographystyle{IEEEtran}`, [1]–[16] urutan kemunculan pertama |
| 6 | Tabel terdistribusi buruk di halaman akhir | 5 tabel menumpuk p4/p5 | FIXED: 8 tabel tersebar p2(1), p3(5), rapi; tanpa halaman float kosong |
| 7 | Tabel jauh dari mention pertama | float menumpuk | FIXED: `tables_generated.tex` di-\input tepat setelah paragraf pembuka Hasil; semua tabel dirujuk `Table~\ref{...}` |
| 8 | Placeholder "Table: Pairwise (post-hoc) comparisons." | caption yatim | FIXED: dihapus; diganti kalimat eksplisit di body (post-hoc tidak dijalankan) |
| 9 | Caption/unit/significant digits tidak konsisten | campuran 3–6 desimal | FIXED: booktabs, caption di atas, 4 desimal seragam (mean±sd 3 desimal), singkatan: CART-skl/CART-cust/SPDT/RF didefinisikan di teks |
| 10 | Overfull/clipped/orphan | 1 overfull 97pt | FIXED: 0 overfull, 0 undefined citation/ref (verified di main.log) |

## Catatan proses

- Template resmi: `IEEEtran.cls` (conference mode) + `IEEEtran.bst`,
  keduanya paket IEEE resmi dari TeX Live/TinyTeX — bukan tiruan.
- Bahasa: Inggris (sesuai brief).
- Angka eksperimen: **tidak ada yang diubah** (lihat
  `docs/numerical_integrity_check.md`, 8/8 PASS).
- Limitasi & hasil negatif dipertahankan penuh (stabilitas lebih buruk,
  ~40x overhead, p=0.1353 n.s., novelty belum established).
