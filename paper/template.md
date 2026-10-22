# Stability-Penalized Split Selection for Decision Tree Classification: Sebuah Evaluasi Multi-Dataset yang Reproducible

**Kata kunci**—pohon keputusan; kriteria split; stabilitas bootstrap; variansi gain; *nested cross-validation*; perbandingan statistik nonparametrik.

## Abstrak

Pohon keputusan CART dikenal cepat dan interpretabel, namun struktur pohonnya
tidak stabil: perubahan kecil pada data latih dapat mengubah split yang
terpilih dan akhirnya prediksi. Penelitian ini mengevaluasi sebuah formulasi
kandidat yang sangat sederhana, dinamai *Stability-Penalized Decision Tree*
(SPDT), di mana skor setiap kandidat split dihitung sebagai gain Gini dikurangi
penalti berbobot terhadap variansi gain tersebut atas resample bootstrap dari
data node: `Score(s) = Gain(s,D) − λ·Var_b[Gain(s,D^(b))]`. Penelusuran
literatur hingga 22 Oktober 2026 tidak menemukan ekuivalen formula-level dari
kriteria tersebut, tetapi ruang riset stabilitas pohon sudah ramai; karena itu
formulasi diposisikan sebagai kandidat, bukan klaim metode baru. Evaluasi
dilakukan pada {{N_DATASETS}} dataset klasifikasi tabular UCI dengan *nested
cross-validation* berulang (5 fold × 3 ulangan pada tiga seed), penyetelan
hyperparameter di inner loop, dan Macro-F1 sebagai metrik primer. Hasil:
SPDT vs CART custom (λ=0) mencatat {{WINS}} menang, {{TIES}} seri, {{LOSSES}}
kalah dengan rata-rata Δ Macro-F1 {{MEAN_DELTA}}. Uji Friedman menunjukkan
χ²(3)={{FRIEDMAN_STAT}}, p={{FRIEDMAN_P}}, Kendall W={{KENDALL_W}}.
**[Kesimpulan akhir diisi setelah angka final terverifikasi dari artefak.]**
Temuan dibahas bersama ablation, biaya komputasi, dan ukuran stabilitas
prediksi yang dilaporkan terpisah dari akurasi.

## I. Pendahuluan

Pohon keputusan adalah metode klasifikasi nonparametrik yang mempartisi ruang
fitur secara rekursif. Kriteria split CART memilih kandidat (fitur, ambang)
yang memaksimalkan penurunan impuritas Gini [@breiman1984cart;@quinlan1986induction].
Namun demikian, struktur pohon diketahui tidak stabil: perturbasi kecil pada
data latih dapat mengubah split yang terpilih [@li2002instability], dan
konsekuensinya prediksi berubah — fenomena inilah yang melatarbelakangi
*bagging* dan *random forests* [@breiman1996bagging;@breiman2001randomforests;@dietterich2000ensemble].

Berbagai pendekatan telah mengurangi instabilitas tersebut pada tingkat yang
berbeda: filter signifikansi statistik, pembatasan struktural, agregasi
ensemble, atau pemilihan model yang stabil [@briand2009stability;@lee2024anls;@mirzamomen2017stable;@wang2017tsallis].
Yang jarang diuji secara langsung adalah pertanyaan yang lebih sempit ini:

> **Apakah penalti variansi bootstrap atas gain split, dikurangkan langsung
> dari skor split, dapat meningkatkan Macro-F1 out-of-sample dan/atau
> stabilitas prediksi dibandingkan CART standar pada resampling berulang,
> dengan biaya runtime yang dapat diterima?**

Hipotesis nol: tidak ada peningkatan sistematis dibandingkan baseline
lintas dataset independen. Tidak ada hipotesis bahwa SPDT *harus* unggul.

Kontribusi yang diklaim terbatas pada: (i) formulasi kandidat yang eksplisit;
(ii) implementasi dan audit reproducible; (iii) evaluasi nested-CV yang
bebas *leakage*; (iv) ablation dan analisis statistik yang jujur terhadap
hasil positif maupun negatif.

## II. Karya Terkait

**Stabilitas pohon keputusan.** Instabilitas struktural pohon dibahas
secara formal oleh Li dan Clarke [@li2002instability]; pengukuran kuantitatif
stabilitas dan kemiripan antar pohon dikembangkan antara lain oleh Turney
[@turney1995stability] serta Briand dkk. [@briand2009stability], yang juga
mengevaluasi stabilisasi node berbasis bootstrap (lini NLS/Dannegger).
Lee dkk. [@lee2024anls] mengusulkan ANLS yang menjalankan iterasi bootstrap
pada setiap node dan memilih prediktor terpilih terbanyak — mekanisme
resampling-node yang paling dekat dengan penelitian ini, tetapi agregasinya
memakai voting frekuensi, bukan penalti variansi gain. Mirzamomen dan
Kangavari [@mirzamomen2017stable] menstabilkan melalui uji split
multi-atribut.

**Kriteria split alternatif.** Kriteria berbasis entropi Tsallis dua-suku
[@wang2017tsallis], kriteria *robust* terhadap label noise [@yang2019pairgain],
pemilihan split tak-bias berbasis statistik Gini [@strobl2007bias], dan
pohon arus data dengan batas keyakinan Hoeffding [@domingos2000vfdt]
menunjukkan bahwa ruang "mengubah skor split" sudah aktif diteliti. Delise
[@delise2026era] menilai konsistensi gain lintas *environment* pada boosting.
Penalti skor oleh variansi estimasi resampled juga dikenal di luar pohon
(pola mean−variance pada seleksi model). Tidak satu pun dari karya yang
berhasil diverifikasi memuat rumus
`Gain − λ·Var_bootstrap[Gain]`; namun demikian temuan negatif penelusuran
bukan bukti kebaruan, sehingga formulasi tetap dicap sebagai kandidat.

## III. Metode

### A. Kriteria split yang diusulkan

Untuk node dengan data D (hanya bagian latih yang sah), dengan
`Gini(D) = 1 − Σ_c p(c|D)²` dan gain split

`Gain(s,D) = Gini(D) − (|D_L|/|D|)·Gini(D_L) − (|D_R|/|D|)·Gini(D_R)`,

dibangun B resample bootstrap `D^(b)` dari baris node. Untuk setiap kandidat
s yang valid:

`V(s) = Var_b[Gain(s,D^(b))]` (varians sampel, ddof=1)

**Skor utama: `Score(s) = Gain(s,D) − λ·V(s)`**.

Kebijakan deterministik: ambang kandidat = titik tengah nilai unik terurut;
`min_samples_leaf`/`min_samples_split` pada jumlah asli node; partisi child
kosong dalam resample berkontribusi 0; daun memilih kelas mayoritas dengan
tie-break label terkecil; seed bootstrap diturunkan dari `(random_state,
id node)`. Untuk λ=0 seluruh komputasi bootstrap dilewati dan estimator
terdegenerasi menjadi CART custom referensi (diverifikasi identik dengan
implementasi referensi independen pada pengujian unit). Sebuah varian
eksploratori terpisah (normalisasi `V/(Gain²+ε)`) dicatat tetapi bukan
metode utama. Keterbatasan penting: V(s) *scale-dependent* terhadap gain.

### B. Desain evaluasi

Dataset: {{N_DATASETS}} dataset klasifikasi tabular UCI (Tabel I).
Protokol: *repeated stratified nested CV* — outer 5 fold × 3 ulangan untuk
masing-masing seed [42, 123, 2026] (45 evaluasi per dataset-metode, split
identik untuk semua metode); inner 3-fold stratified untuk penyetelan dengan
seed deterministik. Preprocessing hanya imputasi median yang di-fit pada
bagian latih (pipeline). Metrik primer Macro-F1; sekunder: balanced accuracy,
accuracy, waktu fit/prediksi, jumlah node, kedalaman. Stabilitas prediksi
diukur terpisah sebagai *disagreement* rata-rata pairwise antar model yang
dilatih pada 10 sub-sampel 90% dari data latih outer, dievaluasi pada outer
test yang sama — dilaporkan terpisah dari akurasi.

TABEL I. Inventaris dataset.

{{TABLE_INVENTORY}}

### C. Analisis statistik

Unit inferensi adalah dataset (rata-rata 45 outer fold), bukan fold.
Perbandingan tiga metode primer (CART sklearn, CART custom λ=0, SPDT)
menggunakan uji Friedman dengan dataset sebagai blok; bila omnibus
memberikan alasan, dilanjutkan Wilcoxon signed-rank berpasangan dengan
koreksi Holm dan effect size rank-biserial. p ≥ 0.05 tidak pernah
diartikan sebagai bukti kesetaraan. Random forest disertakan sebagai
benchmark kontekstual non-ekuivalen dan tidak masuk uji omnibus primer.

## IV. Hasil dan Pembahasan

TABEL II. Macro-F1 mean ± sd (deskriptif, fold-level) per dataset dan metode.

{{TABLE_PERFORMANCE}}

TABEL III. Mean rank Macro-F1.

{{TABLE_MEAN_RANK}}

TABEL IV. Uji statistik dataset-level.

{{TABLE_STATISTICS}}

TABEL V. Perbandingan berpasangan (post-hoc).

{{TABLE_POSTHOC}}

Ringkasan berpasangan SPDT vs CART custom: **{{WINS}} menang / {{TIES}} seri
/ {{LOSSES}} kalah**, rata-rata Δ Macro-F1 {{MEAN_DELTA}}.

TABEL VI. Biaya komputasi.

{{TABLE_RUNTIME}}

TABEL VII. Stabilitas prediksi (disagreement; bukan metrik akurasi).

{{TABLE_STABILITY}}

TABEL VIII. Ablation pada split identik.

{{TABLE_ABLATION}}

TABEL IX. Frekuensi seleksi hyperparameter SPDT (45 outer fold × dataset).

{{TABLE_HYPERPARAMS}}

**Pembahasan.** [Diisi setelah angka final: pola menang/kalah per dataset,
interpretasi sesuai matriks interpretasi yang telah didefinisikan, jawaban
atas apakah penalti variansi membantu sebagai *modifier*, sensitivitas λ dan
B, biaya komputasi aktual, dan konsistensi hasil statistik dengan klaim.]

## V. Kesimpulan

[Kesimpulan diisi setelah hasil final — draf sementara:] Penelitian ini
mengevaluasi formulasi kandidat SPDT secara reproducible dengan nested-CV
bebas leakage, ablation, dan analisis statistik level-dataset pada
{{N_DATASETS}} dataset UCI. Hasil eksperimen menunjukkan **[diisi dari
artefak]**. Keterbatasan: delapan dataset kecil, impuritas Gini saja,
representasi stabilitas berbasis satu proxy, dan klaim kebaruan yang belum
established secara formula-level.

## Referensi

<div id="refs"></div>
