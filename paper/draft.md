---
title: "Stability-Penalized Split Selection for Decision Tree Classification: Sebuah Evaluasi Multi-Dataset yang Reproducible"
author: "Rofiq Fauzi (p36202600047@mhs.dinus.ac.id), Magister Artificial Intelligence, Universitas Dian Nuswantoro"
---

**Kata kunci**--pohon keputusan; kriteria split; stabilitas bootstrap; variansi gain; *nested cross-validation*; perbandingan statistik nonparametrik.

## Abstrak

Pohon keputusan CART dikenal cepat dan interpretabel, namun struktur pohonnya
tidak stabil: perubahan kecil pada data latih dapat mengubah split yang
terpilih dan akhirnya prediksi. Penelitian ini mengevaluasi sebuah formulasi
kandidat yang sangat sederhana, dinamai *Stability-Penalized Decision Tree*
(SPDT), di mana skor setiap kandidat split dihitung sebagai gain Gini dikurangi
penalti berbobot terhadap variansi gain tersebut atas resample bootstrap dari
data node: `Score(s) = Gain(s,D) - lambda*Var_b[Gain(s,D^(b))]`. Penelusuran
literatur hingga 22 Oktober 2026 tidak menemukan ekuivalen formula-level dari
kriteria tersebut, tetapi ruang riset stabilitas pohon sudah ramai; karena itu
formulasi diposisikan sebagai kandidat, bukan klaim metode baru. Evaluasi
dilakukan pada 8 dataset klasifikasi tabular UCI dengan *nested
cross-validation* berulang (5 fold x 3 ulangan pada tiga seed), penyetelan
hyperparameter di inner loop, dan Macro-F1 sebagai metrik primer. Hasil:
SPDT vs CART custom (lambda=0) mencatat 6 menang, 0 seri, 2
kalah dengan rata-rata Delta Macro-F1 +0.0058. Uji Friedman menunjukkan
chi^2(2)=4.0000, p=0.1353, Kendall W=0.2500 -- tidak
cukup bukti untuk menolak hipotesis nol, sehingga perbandingan post-hoc
konfirmatori tidak dijalankan. Ablation menunjukkan besar lambda dan jumlah
resample B nyaris tidak memengaruhi Macro-F1, pembatasan kedalaman lebih
menentukan, dan ukuran stabilitas prediksi SPDT sedikit lebih buruk
daripada CART custom dengan overhead waktu fit +/-40x. Temuan ini merupakan
hasil nol yang jujur: pada delapan dataset kecil-menengah yang diuji,
penalti variansi bootstrap atas gain split tidak memberikan manfaat yang
cukup untuk menandingi kesederhanaan CART, tanpa menutup kemungkinan
manfaat pada regime data lain.

## I. Pendahuluan

Pohon keputusan adalah metode klasifikasi nonparametrik yang mempartisi ruang
fitur secara rekursif. Kriteria split CART memilih kandidat (fitur, ambang)
yang memaksimalkan penurunan impuritas Gini [@breiman1984cart;@quinlan1986induction].
Namun demikian, struktur pohon diketahui tidak stabil: perturbasi kecil pada
data latih dapat mengubah split yang terpilih [@li2002instability], dan
konsekuensinya prediksi berubah -- fenomena inilah yang melatarbelakangi
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
pada setiap node dan memilih prediktor terpilih terbanyak -- mekanisme
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
(pola mean-variance pada seleksi model). Tidak satu pun dari karya yang
berhasil diverifikasi memuat rumus
`Gain - lambda*Var_bootstrap[Gain]`; namun demikian temuan negatif penelusuran
bukan bukti kebaruan, sehingga formulasi tetap dicap sebagai kandidat.

## III. Metode

### A. Kriteria split yang diusulkan

Untuk node dengan data D (hanya bagian latih yang sah), dengan
`Gini(D) = 1 - Sum_c p(c|D)^2` dan gain split

`Gain(s,D) = Gini(D) - (|D_L|/|D|)*Gini(D_L) - (|D_R|/|D|)*Gini(D_R)`,

dibangun B resample bootstrap `D^(b)` dari baris node. Untuk setiap kandidat
s yang valid:

`V(s) = Var_b[Gain(s,D^(b))]` (varians sampel, ddof=1)

**Skor utama: `Score(s) = Gain(s,D) - lambda*V(s)`**.

Kebijakan deterministik: ambang kandidat = titik tengah nilai unik terurut;
`min_samples_leaf`/`min_samples_split` pada jumlah asli node; partisi child
kosong dalam resample berkontribusi 0; daun memilih kelas mayoritas dengan
tie-break label terkecil; seed bootstrap diturunkan dari `(random_state,
id node)`. Untuk lambda=0 seluruh komputasi bootstrap dilewati dan estimator
terdegenerasi menjadi CART custom referensi (diverifikasi identik dengan
implementasi referensi independen pada pengujian unit). Sebuah varian
eksploratori terpisah (normalisasi `V/(Gain^2+eps)`) dicatat tetapi bukan
metode utama. Keterbatasan penting: V(s) *scale-dependent* terhadap gain.

### B. Desain evaluasi

Dataset: 8 dataset klasifikasi tabular UCI (Tabel I).
Protokol: *repeated stratified nested CV* -- outer 5 fold x 3 ulangan untuk
masing-masing seed [42, 123, 2026] (45 evaluasi per dataset-metode, split
identik untuk semua metode); inner 3-fold stratified untuk penyetelan dengan
seed deterministik. Preprocessing hanya imputasi median yang di-fit pada
bagian latih (pipeline). Metrik primer Macro-F1; sekunder: balanced accuracy,
accuracy, waktu fit/prediksi, jumlah node, kedalaman. Stabilitas prediksi
diukur terpisah sebagai *disagreement* rata-rata pairwise antar model yang
dilatih pada 10 sub-sampel 90% dari data latih outer, dievaluasi pada outer
test yang sama -- dilaporkan terpisah dari akurasi.

| Dataset | N | Fitur | Kelas | Sumber |
|---|---|---|---|---|
| banknote | 1372 | 4 | 2 | ucimlrepo |
| glass | 214 | 9 | 6 | ucimlrepo |
| ionosphere | 351 | 34 | 2 | ucimlrepo |
| iris | 150 | 4 | 3 | ucimlrepo |
| seeds | 210 | 7 | 3 | uci_static_zip |
| sonar | 208 | 60 | 2 | ucimlrepo |
| wdbc | 569 | 30 | 2 | ucimlrepo |
| wine | 178 | 13 | 3 | ucimlrepo |

Table: Inventaris dataset.

### C. Analisis statistik

Unit inferensi adalah dataset (rata-rata 45 outer fold), bukan fold.
Perbandingan tiga metode primer (CART sklearn, CART custom lambda=0, SPDT)
menggunakan uji Friedman dengan dataset sebagai blok; bila omnibus
memberikan alasan, dilanjutkan Wilcoxon signed-rank berpasangan dengan
koreksi Holm dan effect size rank-biserial. p >= 0.05 tidak pernah
diartikan sebagai bukti kesetaraan. Random forest disertakan sebagai
benchmark kontekstual non-ekuivalen dan tidak masuk uji omnibus primer.

## IV. Hasil dan Pembahasan

| Dataset | CART-skl | CART-cust | SPDT | RF | SPDT-CART |
|---|---|---|---|---|---|
| banknote | 0.982 +/- 0.008 | 0.981 +/- 0.007 | 0.981 +/- 0.007 | 0.994 +/- 0.004 | +0.0004 |
| glass | 0.620 +/- 0.097 | 0.609 +/- 0.093 | 0.626 +/- 0.093 | 0.742 +/- 0.075 | +0.0166 |
| ionosphere | 0.873 +/- 0.034 | 0.881 +/- 0.038 | 0.879 +/- 0.038 | 0.929 +/- 0.037 | -0.0027 |
| iris | 0.949 +/- 0.033 | 0.945 +/- 0.038 | 0.944 +/- 0.037 | 0.955 +/- 0.034 | -0.0008 |
| seeds | 0.897 +/- 0.032 | 0.906 +/- 0.036 | 0.913 +/- 0.034 | 0.923 +/- 0.038 | +0.0066 |
| sonar | 0.719 +/- 0.065 | 0.706 +/- 0.073 | 0.725 +/- 0.061 | 0.833 +/- 0.065 | +0.0189 |
| wdbc | 0.925 +/- 0.031 | 0.925 +/- 0.030 | 0.928 +/- 0.030 | 0.956 +/- 0.023 | +0.0037 |
| wine | 0.905 +/- 0.047 | 0.905 +/- 0.044 | 0.909 +/- 0.046 | 0.981 +/- 0.024 | +0.0040 |

Table: Macro-F1 mean +/- sd (deskriptif, fold-level) per dataset dan metode.

| Metode | Mean rank |
|---|---|
| SPDT | 1.500 |
| CART-skl | 2.000 |
| CART-cust | 2.500 |

Table: Mean rank Macro-F1.

| Uji | Statistik | db | p | #dataset | Kendall W | Keputusan |
|---|---|---|---|---|---|---|
| friedman | 4.0000 | 2 | 0.1353 | 8 | 0.2500 | tidak ada bukti omnibus; post-hoc tidak dijalankan |

Table: Uji statistik dataset-level.

_Post-hoc tidak dijalankan: uji omnibus Friedman tidak memberikan alasan (p >= 0.05), sesuai protokol._

Table: Perbandingan berpasangan (post-hoc).

Ringkasan berpasangan SPDT vs CART custom: **6 menang / 0 seri
/ 2 kalah**, rata-rata Delta Macro-F1 +0.0058.

| Metode | Fit (s) | Fit median (s) | Prediksi (s) | Node | Kedalaman |
|---|---|---|---|---|---|
| CART-cust | 0.5340 | 0.4163 | 0.0009 | 26.6722 | 5.1000 |
| CART-skl | 0.2693 | 0.2605 | 0.0010 | 27.4222 | 5.2917 |
| RF | 1.1169 | 1.0863 | 0.0573 | - | - |
| SPDT | 21.4302 | 14.6491 | 0.0010 | 28.5944 | 5.4611 |

Table: Biaya komputasi.

| Metode | Disagreement rata-rata | SD |
|---|---|---|
| CART-cust | 0.0928 | 0.0853 |
| CART-skl | 0.0983 | 0.0885 |
| RF | 0.0271 | 0.0337 |
| SPDT | 0.1014 | 0.0934 |

Table: Stabilitas prediksi (disagreement; bukan metrik akurasi).

| Varian | F1 makro | SD | Fit (s) | Node |
|---|---|---|---|---|
| A0 (lam=0) | 0.8599 | 0.1270 | 0.0201 | 37.6611 |
| A1 (lam=.1) | 0.8598 | 0.1256 | 0.0860 | 37.6111 |
| A1 (lam=1) | 0.8601 | 0.1255 | 0.0862 | 37.5278 |
| A2 (B=20) | 0.8601 | 0.1251 | 0.1136 | 37.4556 |
| A2 (B=5) | 0.8613 | 0.1243 | 0.0711 | 37.4667 |
| A3 d3 l1 | 0.8357 | 0.1592 | 0.0371 | 12.8722 |
| A3 d3 l5 | 0.8353 | 0.1546 | 0.0373 | 11.8611 |
| A3 d8 l1 | 0.8592 | 0.1272 | 0.0821 | 35.2500 |
| A3 d8 l5 | 0.8521 | 0.1381 | 0.0820 | 24.7389 |

Table: Ablation pada split identik.

| parameter | value | frequency | share |
|---|---|---|---|
| max_depth | 5.0000 | 140 | 0.3889 |
| max_depth | 8.0000 | 102 | 0.2833 |
| min_samples_leaf | 1.0000 | 201 | 0.5583 |
| min_samples_leaf | 2.0000 | 97 | 0.2694 |
| min_samples_split | 2.0000 | 252 | 0.7000 |
| min_samples_split | 5.0000 | 69 | 0.1917 |
| n_bootstrap | 5.0000 | 191 | 0.5306 |
| n_bootstrap | 20.0000 | 87 | 0.2417 |
| penalty_lambda | 10.0000 | 233 | 0.6472 |
| penalty_lambda | 0.0000 | 61 | 0.1694 |

Table: Frekuensi seleksi hyperparameter SPDT (45 outer fold per dataset).

**Pembahasan.** Hasil bersifat deskriptif condong mendukung SPDT tetapi
tidak mencapai bukti konfirmatori. SPDT mencatat mean rank terbaik (1.500
vs 2.000 CART sklearn dan 2.500 CART custom) dan menang pada 6 dari 8
dataset berpasangan terhadap CART custom, namun uji Friedman tidak
menemukan bukti perbedaan yang cukup (p = 0.1353) dan perbandingan
eksploratif SPDT-CART custom menghasilkan p mentah 0.078 -- tidak bertahan
pada taraf 5% dan tidak boleh ditafsirkan sebagai signifikan. Dengan hanya
8 dataset, power statistik terbatas; sesuai protokol, hasil ini dilaporkan
apa adanya dan p >= 0.05 tidak diartikan sebagai kesetaraan.

Temuan penting kedua: penalti stabilitas tidak menurunkan *prediction
disagreement* -- SPDT justru sedikit lebih tinggi (0.101) dibanding CART
custom (0.093) dan CART sklearn (0.101). Random forest tetap paling stabil
(0.027) melalui agregasi ensemble. Dengan demikian, janji motivasi metode
(stabilitas prediksi lebih baik) **tidak terwujud** pada benchmark ini;
penalti variansi gain bukan pengganti ensemble.

Temuan ketiga dari ablation pada split identik: besar penalti lambda maupun
jumlah resample B nyaris tidak mengubah Macro-F1 (semua varian lambda dan B di
kisaran 0.859-0.861), sedangkan pembatasan kedalaman/ukuran daun jauh
lebih menentukan (depth 3 turun ke +/-0.836). Penyetelan inner cenderung
memilih lambda besar (10 pada 64.7% fold) dan B kecil (5 pada 53.1%), yang
mengindikasikan penalti sering dipilih tetapi efeknya pada akurasi kecil.
Biayanya nyata: waktu fit SPDT +/-40x CART custom. Secara keseluruhan,
efektivitas penalti variansi bootstrap bergantung dataset dan kecil
magnitude-nya; simple distance/impurity splitting tetap preferable untuk
dataset kecil-menengah ini.

## V. Kesimpulan

Penelitian ini mengevaluasi formulasi kandidat SPDT secara reproducible
dengan nested repeated CV bebas leakage, ablation terkontrol pada split
identik, dan analisis statistik level-dataset pada 8 dataset
UCI. Secara deskriptif SPDT memperoleh mean rank terbaik dan menang
berpasangan pada 6 dari 8 dataset terhadap CART custom, tetapi uji
Friedman tidak memberikan bukti perbedaan yang cukup
(chi^2(2)=4.0000, p=0.1353), sehingga tidak ada dasar
konfirmatori untuk mengklaim peningkatan. Lebih lanjut, ukuran stabilitas
prediksi tidak membaik dan justru sedikit memburuk, sementara biaya
komputasi naik +/-40x. Pada kondisi yang diuji, kesederhanaan CART tetap
preferable. Keterbatasan: delapan dataset kecil, impuritas Gini saja,
proxy stabilitas berbasis disagreement, klaim kebaruan formula-level yang
belum established, dan belum dibandingkannya kriteria terdekat dari
literatur (ANLS, minimax split). Pekerjaan lanjutan yang paling bernilai:
membandingkan langsung dengan stabilisasi bootstrap-node dari literatur,
mengkaji normalisasi penalti oleh skala gain, dan menguji pada benchmark
yang lebih besar.

## Referensi

<div id="refs"></div>
