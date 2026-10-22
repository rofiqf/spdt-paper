# Submission Readiness — Extended Paper (2026-10-22)

**Status: BELUM "SUBMISSION READY"** — gate author metadata belum
terpenuhi. Secara teknis/format/ilmiah: lengkap.

## Gates

- [x] IEEEtran conference resmi; 6 halaman; 0 overfull; 0 undefined
- [x] First author mahasiswa; blok author IEEE benar
- [x] Abstract + IEEEkeywords; semua persamaan typeset (eq. 1-5)
- [x] Sitasi numerik IEEE; 16 referensi (14 portal izin + Wiley Scopus
      + SpringerLink); 2 referensi baru ter-verifikasi DOI
- [x] 13 tabel + 3 figure, semua di-\ref dari teks; caption konsisten
- [x] Integritas numerik spot-check PASS (nilai tabel = CSV)
- [x] Hasil negatif & positif & inkonklusif dilaporkan apa adanya;
      tanpa klaim superior/signifikan/robust yang tak didukung
- [x] 37 unit tests passed; protokol freeze + 2 amendemen terdokumentasi
- [x] **6 co-author fakultas TERKONFIRMASI** (2026-10-22, user; first author mahasiswa)
- [ ] **Turnitin UNVERIFIED** — belum ada laporan aktual

## Rebuild

```bash
cd paper && export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/bin:$PATH"
latexmk -pdf -interaction=nonstopmode main.tex && cp main.pdf paper_ieee_extended.pdf
```

## Deliverable paths

paper/paper_ieee_extended.pdf · paper/main.tex · paper/references.bib ·
results/extension/*.csv (6 file) · figures/noise_robustness_macro_f1.png ·
figures/root_split_agreement.png · figures/prediction_disagreement.png ·
figures/runtime_vs_macro_f1.png · logs/extension_run_manifest.json ·
docs/{experiment_summary,experiment_protocol,extension_method,
prior_art_extension,limitations,submission_readiness}.md
