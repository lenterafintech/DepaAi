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
| Uji non-parametrik | Mann-Whitney U, Kolmogorov-Smirnov dua sampel, median Mood, Kruskal-Wallis dengan uji lanjutan Dunn (koreksi Holm/Bonferroni), Wilcoxon peringkat bertanda, uji tanda, Friedman + Kendall's W, chi-square kebebasan + Cramér's V, uji eksak Fisher, korelasi Spearman & Kendall — seluruhnya dilaporkan bersama ukuran efek dan padanan parametriknya |
| Uji beda & hubungan | Analisis diskriminan linear/kuadratik (fungsi kanonik, Wilks' lambda), MANOVA satu jalur + Hotelling's T², MANCOVA dengan rata-rata terkoreksi dan uji homogenitas kemiringan, korelasi kanonik dengan indeks redundansi |
| Model struktural | CFA (muatan terstandardisasi, CR/AVE, indeks kecocokan termasuk SRMR), analisis jalur dengan dekomposisi efek langsung/tidak langsung, SEM penuh, uji mediasi bootstrap dengan VAF, serta pilihan estimator ML / FIML / DWLS / ULS / GLS beserta saran otomatis sesuai ciri data |
| Ekspor & reproduksi | Laporan Lengkap maupun Ringkasan diekspor ke Word, PDF, Excel, PowerPoint, HTML, Markdown, JSON, atau satu paket ZIP; disertai sintaks Python dan R yang menjalankan ulang analisis yang sama |
| Ringkasan kesimpulan | Tiga halaman ringkasan terpisah — Eksekutif, Akademik, Profesional — yang menuliskan satu hasil analisis dengan lampu status, peringkat pendorong, matriks prioritas, tabel bergaya APA, paragraf siap salin, rekomendasi, dan keterbatasan |

## Menjalankan aplikasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Sesudah pemasangan sekali itu, aplikasi cukup dijalankan lewat skrip di akar proyek —
`jalankan.cmd` (Windows, dapat diklik dua kali) atau `./jalankan.sh` (Linux/macOS) —
yang selalu membuka <http://localhost:8503> tanpa perlu diatur ulang.

Port sengaja **tidak** dikunci di `.streamlit/config.toml`. Layanan hosting seperti
Streamlit Community Cloud memeriksa kesehatan aplikasi pada port bawaannya sendiri
(8501); mengunci 8503 di berkas konfigurasi membuat aplikasi berjalan di port yang
tidak diperiksa, sehingga penerapan gagal dengan pesan
`dial tcp 127.0.0.1:8501: connect: connection refused`. Karena itu 8503 ditetapkan
oleh skrip peluncur, yang hanya berlaku di komputer sendiri.

Mulai dari halaman **Beranda & Data** untuk mengunggah berkas, atau tekan *Muat contoh
data nasabah* untuk mencoba seluruh metode dengan data contoh.

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
halaman tidak menghitung ulang. Seluruh angka dihitung ulang dari data yang sedang
aktif — tidak ada nilai contoh yang ditanam.

### Ekspor laporan

Tiap halaman ringkasan memuat panel **Ekspor hasil analisis** dengan dua pilihan ragam
dan delapan pilihan format:

| Ragam | Isinya |
| --- | --- |
| **Ringkasan** | Register halaman yang sedang dibuka saja — ringkas untuk dibagikan |
| **Laporan Lengkap** | Ketiga register sekaligus, seluruh tabel hasil, kalimat siap salin, rujukan ambang, dan catatan analisis yang tidak dapat dijalankan |

| Format | Kegunaan |
| --- | --- |
| Word (`.docx`) | Dokumen siap disunting dan dicetak |
| PDF | Tata letak tetap untuk lampiran resmi |
| Excel (`.xlsx`) | Tiap bagian menjadi lembar tersendiri; angkanya siap diolah ulang |
| PowerPoint (`.pptx`) | Slide siap dipresentasikan |
| HTML | Dibuka di peramban mana pun, dapat dicetak menjadi PDF |
| Markdown | Teks polos untuk disunting lebih lanjut |
| JSON | Data terstruktur untuk diolah sistem lain |
| Sintaks Python / R | Skrip yang menjalankan ulang analisis yang sama |
| Paket lengkap (ZIP) | Seluruh format di atas, setiap tabel dalam CSV, dan kedua sintaks |

Isi seluruh format disusun dari satu sumber yang sama (`lentera_mva/ekspor.py`),
sehingga angka dan kesimpulannya tidak pernah berbeda antar berkas.

### Sintaks yang dapat dijalankan ulang

`lentera_mva/sintaks.py` menuliskan langkah analisis yang benar-benar dipilih pengguna
menjadi skrip. Versi Python memakai pustaka yang persis sama dengan aplikasi
(pandas, scipy, scikit-learn, statsmodels), sehingga angkanya identik; versi R memakai
padanan terdekat (`psych`, `car`, `MASS`, `CCA`, `lmtest`) sebagai pemeriksaan silang.
Skrip hanya memuat jalur berkas, bukan datanya.

## Akun dan paket langganan

Pengguna mendaftar dengan surel dan kata sandi; akun tersimpan pada SQLite
(`data/pengguna.db`, tidak ikut ke dalam repositori). Kata sandi disimpan sebagai
turunan PBKDF2-HMAC-SHA256 dengan garam acak per pengguna, bukan dalam bentuk asli.

**Setiap akun baru mendapat 3 hari uji coba dengan seluruh fitur terbuka.** Setelah
masa itu berakhir, akun berlanjut pada paket yang dipilih saat mendaftar.

| Paket | Harga | Batas data | Cakupan |
| --- | --- | --- | --- |
| Gratis | Gratis | 300 baris · 10 kolom | Eksplorasi, korelasi, PCA/EFA, klaster, regresi, entri data, ringkasan eksekutif |
| Mahasiswa & Pengajar | Rp 49.000/bulan | 5.000 baris · 60 kolom | Seluruh metode termasuk instrumen, MANOVA/MANCOVA, CFA dan SEM; ringkasan akademik; unduhan laporan |
| Profesional | Rp 149.000/bulan | 100.000 baris · 150 kolom | Seluruh fitur dan ketiga ringkasan |
| Institusi (Khusus) | Sesuai kesepakatan | 1.000.000 baris · 500 kolom | Seluruh fitur untuk kampus dan perusahaan; ketentuan disepakati tersendiri |

**Penagihan belum terpasang.** Selama masa perkenalan seluruh paket dapat
diaktifkan tanpa pembayaran, dan halaman *Akun & Langganan* menyatakan hal itu
secara terbuka. Rencana pembayaran memakai Doku; hosting memakai VPS Hostinger.

Yang belum ada pada lapisan akun, dan diperlukan sebelum aplikasi dibuka untuk
umum: verifikasi alamat surel, pemulihan kata sandi, dan pembatasan percobaan masuk.

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
  sem_analysis.py      CFA, analisis jalur, SEM, estimator ML/FIML/DWLS, SRMR, mediasi bootstrap
  nonparametrik.py     Uji non-parametrik, uji lanjutan Dunn, dan koreksi ganda
  ekspor.py            Ekspor laporan ke Word, PDF, Excel, PPT, HTML, MD, JSON, ZIP
  sintaks.py           Pembangkit sintaks Python dan R yang dapat dijalankan ulang
  langganan.py         Paket langganan dan pembatasan fitur
  pengguna.py          Basis data akun, autentikasi, dan masa uji coba
  narrative.py         Penyusun kesimpulan naratif tiga register pembaca
  report_html.py       Laporan HTML mandiri yang dapat diunduh
  kesimpulan_ui.py     Komponen bersama ketiga halaman ringkasan
  plots.py             Visualisasi Plotly
  ui.py                Komponen antarmuka bersama
data/                  Contoh data
scripts/               Pembuat contoh data sintetis
jalankan.cmd           Peluncur Windows pada port 8503
jalankan.sh            Peluncur Linux/macOS pada port 8503
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

Uji non-parametrik dibandingkan langsung dengan `scipy.stats` sebagai acuan, dan
ekspor diperiksa dengan membuka kembali berkas hasilnya (docx, xlsx, pptx, dan pdf
dibaca ulang oleh pustakanya masing-masing) — bukan sekadar memeriksa ukuran berkas.
Sintaks Python yang dibangkitkan ikut diuji dengan `compile()` agar tidak pernah
menghasilkan skrip yang cacat.

## Contoh data

`data/contoh_data_nasabah.csv` berisi 400 nasabah sintetis dengan tiga faktor laten
(kapasitas ekonomi, kedisiplinan bayar, intensitas transaksi), tiga segmen usaha, dan
status gagal bayar. Data dapat dibangkitkan ulang dengan:

```bash
python scripts/generate_sample_data.py
```
