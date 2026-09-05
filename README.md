# Lentera MVA — Aplikasi Analisis Multivariat

Aplikasi web untuk analisis statistik multivariat: unggah berkas CSV/Excel, pilih
metode, lalu baca hasil beserta uji asumsi, visualisasi, dan penjelasan
interpretasinya dalam bahasa Indonesia.

## Metode yang tersedia

| Kelompok | Metode |
| --- | --- |
| Eksplorasi | Statistik deskriptif, tabel frekuensi, uji normalitas (Shapiro-Wilk, D'Agostino, KS), normalitas multivariat Mardia, pencilan IQR & jarak Mahalanobis |
| Korelasi & asumsi | Korelasi Pearson/Spearman/Kendall dengan uji signifikansi, korelasi parsial, KMO, Bartlett's test of sphericity, VIF, Box's M, Levene |
| Reduksi dimensi | PCA (scree plot, biplot, komunalitas, analisis paralel Horn), analisis faktor eksploratori (principal / principal axis factoring / maximum likelihood) dengan rotasi varimax & promax |
| Pengelompokan | K-Means (elbow, silhouette, Calinski-Harabasz, Davies-Bouldin), hierarki + dendrogram, DBSCAN, profil klaster dengan uji ANOVA |
| Pemodelan | Regresi linear berganda (koefisien baku, ANOVA, VIF, uji asumsi klasik, stepwise) dan regresi logistik biner (odds ratio, ROC/AUC, matriks konfusi) |
| Uji beda & hubungan | Analisis diskriminan linear/kuadratik (fungsi kanonik, Wilks' lambda), MANOVA satu jalur + Hotelling's T², korelasi kanonik dengan indeks redundansi |
| Pelaporan | Kesimpulan otomatis dalam tiga register pembaca (eksekutif/awam, akademik, profesional) lengkap dengan lampu status, peringkat pendorong, matriks prioritas, tabel bergaya APA, paragraf siap salin, rekomendasi, dan keterbatasan |

## Menjalankan aplikasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka di <http://localhost:8502> (port diatur pada `.streamlit/config.toml`). Mulai dari halaman **Beranda & Data**
untuk mengunggah berkas, atau tekan *Muat contoh data nasabah* untuk mencoba
seluruh metode dengan data contoh.

## Halaman Kesimpulan Analisis

Halaman **Kesimpulan Analisis** menjalankan seluruh rangkaian metode sekaligus pada
data aktif, lalu menuliskan hasilnya sebagai narasi untuk tiga pembaca berbeda:

- **Eksekutif & pengguna awam** — pernyataan kesimpulan utama, lampu status
  pemeriksaan, peringkat pendorong beserta matriks prioritas (kepentingan terhadap
  kinerja), rekomendasi tindakan, dan batas kesimpulan; ditulis tanpa notasi statistik.
- **Akademik (mahasiswa & dosen)** — pelaporan bergaya jurnal lengkap dengan statistik
  uji, derajat bebas, p-value, dan ukuran efek; tabel bergaya APA (deskriptif +
  korelasi berbintang, regresi, MANOVA, ringkasan asumsi); paragraf siap salin untuk
  bab metode, hasil, dan pembahasan; keterbatasan serta rujukan ambang yang dipakai.
- **Profesional & analis** — metrik kunci model, kontribusi fitur, temuan teknis,
  ringkasan pemeriksaan asumsi, tindak lanjut berprioritas, dan risiko pemakaian.

Setiap register dapat diunduh sebagai laporan HTML mandiri (siap dicetak menjadi PDF)
atau berkas Markdown. Seluruh angka dihitung ulang dari data yang sedang aktif —
tidak ada nilai contoh yang ditanam.

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
  narrative.py         Penyusun kesimpulan naratif tiga register pembaca
  report_html.py       Laporan HTML mandiri yang dapat diunduh
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
