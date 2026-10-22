# SPDT IEEE Paper — Build Instructions

File naskah IEEE conference (IEEEtran, mode conference):

- `main.tex` — naskah utama (abstrak, section, sitasi numerik, referensi)
- `tables_generated.tex` — 8 tabel, **generate otomatis dari `results/`**
  (jangan edit angka manual; regenerate lewat skrip)
- `references.bib` — 16 referensi terverifikasi (15 portal izin + 1 buku
  fondasi yang ditandai)
- `paper_ieee_final.pdf` — hasil kompilasi final

## Build

```bash
cd paper
# regenerate tabel dari artefak hasil (wajib setelah ada perubahan results/)
../.venv/bin/python ../scripts/make_ieee_tables.py
# kompilasi penuh (pdflatex + bibtex + rerun sampai stabil)
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$HOME/bin:$PATH"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cp main.pdf paper_ieee_final.pdf
```

Log build tersimpan di `../logs/latex_build.log`.
Verifikasi integritas angka: `../.venv/bin/python ../scripts/check_numerical_integrity.py`

## TODO sebelum submission

Lihat `../docs/submission_readiness.md` — co-author fakultas belum
dikonfirmasi (jangan fabrikasi nama), Turnitin belum dijalankan.
