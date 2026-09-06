# Lentera MVA — Aplikasi Analisis Multivariat

Aplikasi web untuk analisis statistik multivariat: unggah berkas CSV/Excel, pilih
metode, lalu baca hasil beserta uji asumsi, visualisasi, dan penjelasan
interpretasinya dalam bahasa Indonesia.

## Metode yang tersedia

| Kelompok | Metode |
| --- | --- |
| Entri data | Membuat tabel baru langsung di aplikasi (definisi kolom, skala Likert, butir kuesioner bernomor), menyunting data aktif, dan memeriksa kelengkapan isian |
| Instrumen | Alpha Cronbach dengan statistik per butir dan alpha-if-deleted, omega McDonald, composite reliability, AVE, serta validitas diskriminan Fornell-Larcker |
| Eksplorasi | Statistik deskriptif, tabel frekuensi, uji normalitas (Shapiro-Wilk, D'Agostino, KS), normalitas multivariat Mardia, pencilan IQR & jarak Mahalanobis |
| Korelasi & asumsi | Korelasi Pearson/Spearman/Kendall dengan uji signifikansi, korelasi parsial, KMO, Bartlett's test of sphericity, VIF, Box's M, Levene |
| Reduksi dimensi | PCA (scree plot, biplot, komunalitas, analisis paralel Horn), analisis faktor eksploratori (principal / principal axis factoring / maximum likelihood) dengan rotasi varimax & promax |
| Pengelompokan | K-Means (elbow, silhouette, Calinski-Harabasz, Davies-Bouldin), hierarki + dendrogram, DBSCAN, profil klaster dengan uji ANOVA |
| Pemodelan | Regresi linear berganda (koefisien baku, ANOVA, VIF, uji asumsi klasik, stepwise), regresi logistik biner (odds ratio, ROC/AUC, matriks konfusi), dan regresi moderasi/MRA (suku interaksi, ΔR², simple slopes, Johnson-Neyman) |
| Uji beda & hubungan | Analisis diskriminan linear/kuadratik (fungsi kanonik, Wilks' lambda), MANOVA satu jalur + Hotelling's T², MANCOVA dengan rata-rata terkoreksi dan uji homogenitas kemiringan, korelasi kanonik dengan indeks redundansi |
| Model struktural | CFA (muatan terstandardisasi, CR/AVE, indeks kecocokan), analisis jalur dengan dekomposisi efek langsung/tidak langsung, SEM penuh, serta uji mediasi bootstrap dengan VAF |
| Ringkasan kesimpulan | Tiga halaman ringkasan terpisah — Eksekutif, Akademik, Profesional — yang menuliskan satu hasil analisis dengan lampu status, peringkat pendorong, matriks prioritas, tabel bergaya APA, paragraf siap salin, rekomendasi, dan keterbatasan |

## Menjalankan aplikasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka di <http://localhost:8503> (port diatur pada `.streamlit/config.toml`; bila port itu sedang dipakai, jalankan dengan `--server.port 8504`). Mulai dari halaman **Beranda & Data**
untuk mengunggah berkas, atau tekan *Muat contoh data nasabah* untuk mencoba
seluruh metode dengan data contoh.

## Tiga halaman ringkasan

Kelompok menu **Ringkasan Kesimpulan** berisi tiga halaman terpisah yang dapat dipilih
pengguna. Ketiganya menjalankan rangkaian metode yang sama pada data aktif, namun
menuliskan hasilnya untuk pembaca yang berbeda:

| Halaman | Untuk siapa | Isinya |
| --- | --- | --- |
| **Ringkasan Eksekutif** | Pimpinan dan pembaca non-statistik | Kesimpulan utama, lampu status pemeriksaan, peringkat pendorong beserta matriks prioritas (kepentingan terhadap kinerja), rekomendasi tindakan berprioritas, dan batas kesimpulan — tanpa notasi statistik |
| **Ringkasan Akademik** | Mahasiswa dan dosen | Pelaporan bergaya jurnal (statistik uji, derajat bebas, p-value, ukuran efek), tabel bergaya APA, paragraf siap salin untuk bab metode/hasil/pembahasan, keterbatasan, dan rujukan ambang |
| **Ringkasan Profesional** | Analis dan praktisi | Metrik kunci model, kontribusi fitur, temuan teknis, ringkasan pemeriksaan asumsi, tindak lanjut berprioritas, dan risiko pemakaian |

Pengaturan cakupan analisis (variabel, target, prediktor, kelompok) dibuat sekali dan
berlaku untuk ketiga halaman; hasil perhitungan dipakai ulang sehingga berpindah
halaman tidak menghitung ulang. Tiap halaman menyediakan unduhan laporan HTML mandiri
(siap dicetak menjadi PDF) dan Markdown untuk registernya, serta satu berkas HTML
berisi ketiga ringkasan sekaligus untuk pembaca campuran. Seluruh angka dihitung ulang
dari data yang sedang aktif — tidak ada nilai contoh yang ditanam.

## Paket langganan

Fitur dibatasi menurut paket yang berlaku (`lentera_mva/langganan.py`):

| Paket | Batas data | Cakupan |
| --- | --- | --- |
| Gratis | 300 baris · 10 kolom | Eksplorasi, korelasi, PCA/EFA, klaster, regresi, entri data, ringkasan eksekutif |
| Pro | 50.000 baris · 100 kolom | Seluruh metode termasuk instrumen dan SEM, ketiga ringkasan, unduhan laporan |
| Institusi | 500.000 baris · 500 kolom | Sama seperti Pro dengan batas data jauh lebih besar |

**Penagihan belum terpasang.** Halaman *Akun & Langganan* saat ini berjalan dalam
mode uji coba: paket dapat diganti bebas untuk mencoba batasannya. Status paket
disimpan pada sesi; ketika basis data pengguna tersedia, hanya fungsi
`ui.paket_aktif()` yang perlu diubah.

## Format data yang didukung

- CSV/TSV/TXT — pemisah kolom dideteksi otomatis (koma, titik koma, atau tab)
- Excel `.xlsx`, `.xlsm`, `.xls` — sheet dapat dipilih saat mengunggah

Setiap baris adalah satu observasi dan setiap kolom satu variabel. Kolom kategorik
otomatis diubah menjadi variabel dummy pada analisis regresi.

## Struktur proyek

```
app.py                 Entri aplikasi & navigasi halaman
views/                 Halaman antarmuka Streamlit (satu berkas per metode)
lentera_mva/           Pustaka perhitungan (murni pandas/numpy, tanpa Streamlit)
  io_utils.py          Pemuatan berkas dan profil data
  preprocessing.py     Missing value, penskalaan, encoding, matriks desain
  descriptive.py       Deskriptif, normalitas, Mardia, pencilan
  assumptions.py       KMO, Bartlett, VIF, Box's M, Levene
  correlation.py       Matriks korelasi, signifikansi, korelasi parsial
  pca_analysis.py      PCA dan analisis paralel Horn
  factor_analysis.py   EFA, rotasi varimax & promax
  clustering.py        K-Means, hierarki, DBSCAN, profil klaster
  regression.py        Regresi linear & logistik, stepwise, prediksi
  discriminant.py      LDA/QDA, fungsi kanonik, Wilks' lambda
  manova.py            MANOVA satu jalur, Hotelling's T²
  cca.py               Korelasi kanonik
  data_entry.py        Pembuatan dan pembersihan data hasil entri manual
  reliability.py       Alpha, omega, CR, AVE, dan validitas diskriminan
  moderation.py        Regresi moderasi: interaksi, simple slopes, Johnson-Neyman
  mancova.py           MANCOVA, ANCOVA univariat, rata-rata terkoreksi
  sem_analysis.py      CFA, analisis jalur, SEM, dan mediasi bootstrap (semopy)
  langganan.py         Paket langganan dan pembatasan fitur
  narrative.py         Penyusun kesimpulan naratif tiga register pembaca
  report_html.py       Laporan HTML mandiri yang dapat diunduh
  kesimpulan_ui.py     Komponen bersama ketiga halaman ringkasan
  plots.py             Visualisasi Plotly
  ui.py                Komponen antarmuka bersama
data/                  Contoh data
scripts/               Pembuat contoh data sintetis
tests/                 Uji perhitungan dan uji asap halaman
```

Pustaka `lentera_mva` dapat dipakai langsung tanpa antarmuka:

```python
import pandas as pd
from lentera_mva import pca_analysis, regression

df = pd.read_csv("data/contoh_data_nasabah.csv")
hasil = pca_analysis.run_pca(df[["skor_kredit", "rasio_utang_pendapatan", "saldo_tabungan"]])
print(hasil.variance_table())
```

## Pengujian

```bash
pytest tests/ -q
```

Terdapat tiga lapis pengujian: kebenaran perhitungan statistik (koefisien regresi
dipulihkan dari data simulasi, klaster yang ditanam berhasil ditemukan, Hotelling's
T² konsisten dengan MANOVA), penyusunan narasi kesimpulan (ketiga register benar-benar
berbeda, register awam bebas notasi statistik, metode yang gagal dicatat alih-alih
menggagalkan laporan), dan uji asap yang merender setiap halaman Streamlit.

## Contoh data

`data/contoh_data_nasabah.csv` berisi 400 nasabah sintetis dengan tiga faktor laten
(kapasitas ekonomi, kedisiplinan bayar, intensitas transaksi), tiga segmen usaha, dan
status gagal bayar. Data dapat dibangkitkan ulang dengan:

```bash
python scripts/generate_sample_data.py
```
