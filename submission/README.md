# Submission Bundle — SPDT (Stability-Penalized Decision Tree)

Disusun 2026-10-22. Satu paket untuk pengumpulan tugas; data mentah
lengkap tersedia di repo (results/ dan results/extension/).

## Isi

```
paper/
  paper_ieee_extended.pdf   <- NASKAH SUBMIT (IEEE conference, 6 halaman)
results_summary/
  baseline_dataset_metrics.csv    Macro-F1 per dataset-metode (baseline)
  baseline_statistical_tests.csv  Friedman + eksploratif baseline
  baseline_stability.csv          disagreement baseline
  baseline_runtime.csv            waktu baseline
  extension_statistical_tests.csv semua uji statistik ekstensi (dengan CI)
  extension_runtime.csv           biaya komputasi + counter ekstensi
docs/
  experiment_protocol.md     protokol beku + 2 amendemen terdokumentasi
  experiment_summary.md      ringkasan hasil RQ1-RQ5
  limitations.md             keterbatasan (baseline + ekstensi)
  prior_art_extension.csv    audit prior-art (11 entri terverifikasi)
  extension_audit.md         verifikasi hasil lama sebelum ekstensi
MANIFEST.sha256              checksum seluruh berkas di paket ini
```

## Yang SENGAJA tidak disertakan (tersedia di repo)

- CSV fold-level mentah (ribuan baris): results/raw/, results/extension/
- Figure: figures/
- Video penjelasan: lihat docs/video_script.md (skrip siap rekam)

## Integritas

Verifikasi: `sha256sum -c MANIFEST.sha256`
Reproduksi penuh: lihat HANDOVER.md di root repo.
