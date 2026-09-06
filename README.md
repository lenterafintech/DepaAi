# NalarData — Aplikasi Analisis Multivariat

Aplikasi web untuk analisis statistik multivariat: unggah berkas CSV/Excel, pilih
metode, lalu baca hasil beserta uji asumsi, visualisasi, dan penjelasan
interpretasinya dalam bahasa Indonesia.

## Metode yang tersedia

| Kelompok | Metode |
| --- | --- |
| Proyek | Menyimpan data aktif, hasil pada Laporan Hasil, dan pengaturan cakupan analisis ke satu berkas `.nalardata`, lalu membukanya kembali pada sesi berikutnya |
| Entri data | Membuat tabel baru langsung di aplikasi (definisi kolom, skala Likert, butir kuesioner bernomor), menyunting data aktif, memeriksa kelengkapan isian, serta meringkas beberapa butir menjadi satu variabel konstruk (rata-rata, jumlah, atau skor baku) dengan aturan butir minimal terisi |
| Instrumen | Alpha Cronbach dengan statistik per butir dan alpha-if-deleted, reliabilitas belah-dua Spearman-Brown, omega McDonald, composite reliability, AVE, serta validitas diskriminan Fornell-Larcker dan HTMT |
| Audit kualitas data | Satu keputusan tentang kesiapan data: nilai hilang bertingkat, baris kembar, kolom kosong dan tidak beragam, angka yang tersimpan sebagai teks, nilai tak hingga, pencilan, kemencengan, dan kategori yang tidak seragam — masing-masing disertai dampak dan saran, tanpa mengubah data |
| Impor data | CSV, TSV, Excel, dan **SPSS `.sav`** — label nilai SPSS diterapkan apa adanya, sehingga kolom kategori tidak muncul sebagai angka tanpa makna |
| Eksplorasi | Statistik deskriptif, tabel frekuensi, uji normalitas (Shapiro-Wilk, D'Agostino, KS), normalitas multivariat Mardia, pencilan IQR & jarak Mahalanobis |
| Korelasi & asumsi | Korelasi Pearson/Spearman/Kendall dengan uji signifikansi, korelasi parsial, KMO, Bartlett's test of sphericity, VIF, Box's M, Levene |
| Reduksi dimensi | PCA (scree plot, biplot, komunalitas, analisis paralel Horn), analisis faktor eksploratori (principal / principal axis factoring / maximum likelihood) dengan rotasi varimax & promax |
| Pengelompokan | K-Means (elbow, silhouette, Calinski-Harabasz, Davies-Bouldin), hierarki + dendrogram, DBSCAN, profil klaster dengan uji ANOVA |
| Pemodelan | Regresi linear berganda (koefisien baku, ANOVA, VIF, uji asumsi klasik termasuk Breusch-Pagan dan White, galat baku robust HC0–HC3 dengan anjuran otomatis, stepwise), regresi logistik biner (odds ratio, ROC/AUC, matriks konfusi), dan regresi moderasi/MRA (suku interaksi, ΔR², simple slopes, Johnson-Neyman) |
| Uji non-parametrik | Mann-Whitney U, Kolmogorov-Smirnov dua sampel, median Mood, Kruskal-Wallis dengan uji lanjutan Dunn (koreksi Holm/Bonferroni), Wilcoxon peringkat bertanda, uji tanda, Friedman + Kendall's W, chi-square kebebasan + Cramér's V, uji eksak Fisher, korelasi Spearman & Kendall — seluruhnya dilaporkan bersama ukuran efek dan padanan parametriknya |
| Uji beda & hubungan | Analisis diskriminan linear/kuadratik (fungsi kanonik, Wilks' lambda), MANOVA satu jalur + Hotelling's T², MANCOVA dengan rata-rata terkoreksi dan uji homogenitas kemiringan, korelasi kanonik dengan indeks redundansi |
| Model struktural | CFA (muatan terstandardisasi, CR/AVE, indeks kecocokan termasuk SRMR), analisis jalur dengan dekomposisi efek langsung/tidak langsung, SEM penuh, uji mediasi bootstrap dengan VAF, pilihan estimator ML / FIML / DWLS / ULS / GLS beserta saran otomatis sesuai ciri data, serta spesifikasi model bebas bergaya lavaan lengkap dengan pemeriksaan sintaks |
| Ekspor & reproduksi | Laporan Lengkap maupun Ringkasan diekspor ke Word, PDF, Excel, PowerPoint, HTML, Markdown, JSON, atau satu paket ZIP; disertai sintaks Python, R, SPSS, AMOS, dan Mplus yang menjalankan ulang analisis yang sama |
| Laporan analisis | Tiga halaman menurut pembaca — Umum, Mahasiswa & Pengajar, Profesional — masing-masing tersedia dalam dua bentuk: **Ringkasan** dan **Laporan Lengkap**. Ditambah **Laporan Hasil**, yang mengumpulkan analisis yang benar-benar dijalankan pengguna |

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

`.streamlit/config.toml` mencantumkan port **8501** karena berkas itu dipakai saat
penerapan: Streamlit Community Cloud memeriksa kesehatan aplikasi pada port tersebut,
dan mengunci 8503 di sana membuat aplikasi berjalan di port yang tidak diperiksa
sehingga penerapan gagal dengan pesan
`dial tcp 127.0.0.1:8501: connect: connection refused`. Port 8503 ditetapkan oleh
skrip peluncur lewat `--server.port`, yang mengalahkan berkas konfigurasi di komputer
sendiri tanpa mengganggu penerapan.

Mulai dari halaman **Beranda & Data** untuk mengunggah berkas, atau tekan *Muat contoh
data nasabah* untuk mencoba seluruh metode dengan data contoh.

## Laporan analisis: pembaca × kedalaman

Kelompok menu **Laporan Analisis** memisahkan dua hal yang sering tertukar:

- **Pembaca** menentukan bahasanya — Umum, Akademik, atau Profesional.
- **Kedalaman** menentukan bentuk dokumennya — Ringkasan atau Laporan Lengkap.

Keduanya sumbu yang saling bebas, sehingga tersedia enam dokumen. Laporan Lengkap
**bukan** gabungan ketiga ringkasan: ia dokumen tersendiri yang tetap ditulis dalam
satu register, hanya lebih dalam. Menggabungkan ketiganya justru memaksa pembaca
membaca hal yang sama tiga kali.

| | Ringkasan | Laporan Lengkap |
| --- | --- | --- |
| Isi | Kesimpulan utama, status pemeriksaan, peringkat pendorong, rekomendasi, batas kesimpulan | Seluruh isi ringkasan, ditambah uraian tiap temuan, seluruh tabel hasil, rujukan ambang, dan catatan analisis yang tidak dapat dijalankan |
| Akademik | — | Ditambah kalimat siap salin bergaya jurnal |

Ketiga halaman menjalankan rangkaian metode yang sama pada data aktif, namun
menuliskan hasilnya untuk pembaca yang berbeda:

| Halaman | Untuk siapa | Isinya |
| --- | --- | --- |
| **Umum** | Pimpinan dan pembaca non-statistik | Kesimpulan utama, lampu status pemeriksaan, peringkat pendorong beserta matriks prioritas (kepentingan terhadap kinerja), rekomendasi tindakan berprioritas, dan batas kesimpulan — tanpa notasi statistik |
| **Mahasiswa & Pengajar** | Mahasiswa, dosen, dan pengajar | Pelaporan bergaya jurnal (statistik uji, derajat bebas, p-value, ukuran efek), tabel bergaya APA, paragraf siap salin untuk bab metode/hasil/pembahasan, keterbatasan, dan rujukan ambang |
| **Profesional** | Analis dan praktisi | Metrik kunci model, kontribusi fitur, temuan teknis, ringkasan pemeriksaan asumsi, tindak lanjut berprioritas, dan risiko pemakaian |

Pengaturan cakupan analisis (variabel, target, prediktor, kelompok) dibuat sekali dan
berlaku untuk ketiga halaman; hasil perhitungan dipakai ulang sehingga berpindah
halaman tidak menghitung ulang. Seluruh angka dihitung ulang dari data yang sedang
aktif — tidak ada nilai contoh yang ditanam.

### Menyimpan pekerjaan: berkas proyek

Aplikasi ini **tidak menyimpan apa pun di server**. Data yang diunggah, hasil pada
Laporan Hasil, dan pengaturan cakupan analisis semuanya hidup di dalam sesi, sehingga
hilang ketika peramban ditutup atau layanan hosting menidurkan aplikasi.

Berkas proyek `.nalardata` menutup celah itu — satu arsip berisi:

```
proyek.json        manifest: format, versi, waktu pembuatan
data.csv           data aktif, selalu ada dan dapat dibaca manusia
data.parquet       salinan yang mempertahankan tipe data
konfigurasi.json   pilihan variabel pada halaman laporan
keranjang.json     daftar hasil yang disimpan
keranjang/NN.csv   tabel tiap hasil
```

CSV selalu ditulis meski Parquet berhasil, sehingga proyek tetap terbuka di komputer
tanpa engine Parquet. Simpan lewat panel di bawah halaman **Beranda & Data**, buka
lewat tab **Buka proyek** di halaman yang sama.

Berkas proyek berasal dari luar aplikasi, jadi diperiksa sebelum isinya dibaca: ukuran
arsip maksimal 250 MB, ukuran setelah dibuka maksimal 500 MB, dan jumlah berkas
internal maksimal 500. Batas kedua yang menahan *zip bomb* — arsip kecil yang
mengembang menjadi raksasa dan menghabiskan memori server.

### Laporan Hasil — analisis yang Anda jalankan sendiri

Ketiga halaman di atas disusun otomatis oleh mesin narasi dari rangkaian metodenya
sendiri. Halaman **Laporan Hasil** melengkapinya dari arah sebaliknya: setiap tabel
hasil di halaman metode punya tombol *Simpan ke laporan*, dan yang tersimpan
dikumpulkan di sini menjadi satu dokumen — persis analisis yang Anda pilih dan
jalankan, lengkap dengan judul laporan dan nama penyusun.

Penyimpanan sengaja atas permintaan, bukan otomatis: menangkap setiap tabel yang
pernah terlihat akan memenuhi laporan dengan percobaan yang tidak jadi dipakai. Hasil
disimpan sebagai salinan, sehingga tetap utuh ketika data aktif berubah — dan hasil
yang berasal dari data lain ditandai agar tidak tercampur tanpa disadari.

### Ekspor laporan

Tiap halaman ringkasan memuat panel **Ekspor hasil analisis** dengan dua pilihan ragam
dan delapan pilihan format:

Yang diekspor mengikuti pilihan *Bentuk laporan* di bagian atas halaman, sehingga
berkasnya persis sama dengan yang tampil di layar.

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
| Sintaks SPSS | Perintah `.sps` yang setara, untuk diperiksa ulang di SPSS |
| Spesifikasi AMOS | Daftar jalur dan langkah untuk dipindahkan ke AMOS |
| Input Mplus | Berkas `.inp` siap dijalankan |
| Paket lengkap (ZIP) | Seluruh format di atas, setiap tabel dalam CSV, dan kedua sintaks |

Seluruh penulis format kini melewati satu lapisan dokumen netral (`Dokumen` di
`nalardata/ekspor.py`), sehingga angka dan kesimpulannya tidak pernah berbeda antar
berkas — dan sumber yang sama dapat berupa laporan naratif maupun Laporan Hasil.

### Galat baku robust pada regresi

Uji asumsi klasik menjalankan **dua** uji homoskedastisitas, bukan satu: Breusch-Pagan
hanya menangkap ragam yang berubah sebanding prediktor, sementara ragam yang membesar
simetris (misalnya menurut `|x|`) lolos darinya dan hanya tertangkap uji **White**.
Bila salah satu menolak, aplikasi menganjurkan galat baku **HC3** beserta alasannya.

Memilih galat baku robust tidak mengubah koefisien B sedikit pun — yang berubah hanya
galat baku, nilai t, p, dan selang kepercayaannya. Tabel ANOVA tetap memakai uji F
berbasis jumlah kuadrat agar sebanding lintas pilihan; uji Wald robust dilaporkan
terpisah. Ketika rancangan tidak memadai untuk uji White (misalnya karena prediktor
dummy membuatnya rank-deficient), hasilnya dinyatakan *tidak dapat diuji* alih-alih
menampilkan angka semu.

### Spesifikasi model SEM bebas

Selain menyusun model lewat pilihan, tab **Sintaks sendiri** menerima model yang
diketik langsung dengan tata tulis gaya lavaan (`=~`, `~`, `~~`) — berguna untuk model
yang tidak tertampung pilihan, seperti kovarians residual, atau model yang sudah
ditulis untuk lavaan maupun AMOS. Sintaks diperiksa lebih dulu dan kesalahan yang lazim
(salah ketik nama variabel, operator keliru, konstruk berbutir tunggal) diterjemahkan
menjadi pesan yang dapat ditindaklanjuti, bukan kegagalan mesin estimasi yang samar.

### Sintaks yang dapat dijalankan ulang

`nalardata/sintaks.py` menuliskan langkah analisis yang benar-benar dipilih pengguna
menjadi skrip untuk lima perangkat:

| Berkas | Perangkat | Sifat |
| --- | --- | --- |
| `analisis.py` | Python | Pustaka yang persis sama dengan aplikasi, angkanya identik |
| `analisis.R` | R | Padanan terdekat (`psych`, `car`, `MASS`, `CCA`, `lmtest`) |
| `analisis.sps` | SPSS | Perintah setara: `REGRESSION`, `FACTOR`, `GLM`, `DISCRIMINANT` |
| `model_amos.txt` | AMOS | Spesifikasi jalur beserta langkah menggambarnya |
| `analisis.inp` | Mplus | Berkas input dengan `ESTIMATOR = MLR` |

SPSS mendominasi kampus di Indonesia dan AMOS mendominasi SEM di sana, sehingga
pembimbing kerap meminta hasil diperiksa ulang di perangkat itu.

Mplus membatasi nama variabel 8 karakter. Pemotongan lugas berbahaya —
`pendapatan_bulanan` dan `pendapatan_tahunan` sama-sama menjadi `pendapat` dan
menghasilkan berkas rusak tanpa peringatan — sehingga tabrakan diselesaikan dengan
penomoran dan seluruh pemetaannya dilaporkan di dalam berkas.

Skrip hanya memuat jalur berkas, bukan datanya. Paket ZIP menyertakan
`sintaks/BACA_DULU.txt` yang menjelaskan cara memperoleh `data.csv` yang dirujuk
seluruh skrip itu.

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

### Grafik pada berkas laporan

Grafik di layar dibuat dengan Plotly, tetapi Plotly memerlukan `kaleido` untuk
menghasilkan gambar. Berkas Word, PDF, dan PowerPoint hanya menerima gambar, sehingga
grafik untuk laporan digambar ulang memakai **matplotlib** — menghindari satu
dependensi tanpa mengorbankan apa pun, karena media cetak memang tidak punya
interaksi yang hilang.

Dua grafik ikut ke laporan: **status pemeriksaan** dan **peringkat pendorong**. Warnanya
memakai palet yang sama dengan grafik di layar, yang sudah lolos pemeriksaan
keterbacaan bagi pembaca dengan buta warna. Arah pengaruh dibawa warna **sekaligus**
tanda angkanya, sehingga tidak ada makna yang hilang bila warnanya tidak terbaca.

Word, PDF, PowerPoint, dan HTML memuat gambarnya (HTML menyematkannya sebagai data URI
agar berkasnya tetap mandiri). Markdown dan JSON hanya menandai keberadaannya supaya
berkasnya tetap ringan dan dapat dibaca.

## Antarmuka

Warna hidup sebagai **token CSS** yang disusun dari satu palet Python, bukan ditulis
berulang di banyak tempat. Palet gelap adalah langkah yang dipilih sendiri — bukan
pembalikan otomatis dari palet terang — dan yang menentukan adalah **tema Streamlit**,
bukan `prefers-color-scheme` peramban: pengguna dapat memilih tema di dalam aplikasi,
dan pilihan itu tidak selalu sama dengan pengaturan sistem operasinya.

Tiga komponen dipakai bersama seluruh halaman:

| Komponen | Kegunaan |
| --- | --- |
| Bilah status | Data aktif, ukurannya, dan paket yang berlaku — di **atas** halaman, bukan tersembunyi di sidebar. Kekeliruan paling mahal adalah menganalisis data yang salah tanpa menyadarinya |
| Judul bagian | Batang aksen, kicker, judul, dan keterangan; seragam di semua halaman |
| Keadaan kosong | Mengarahkan langkah berikutnya, bukan sekadar memberi tahu bahwa isinya kosong |

Gerak dimatikan bagi pengguna yang memintanya lewat `prefers-reduced-motion`.

## Format data yang didukung

- CSV/TSV/TXT — pemisah kolom dideteksi otomatis (koma, titik koma, atau tab)
- Excel `.xlsx`, `.xlsm`, `.xls` — sheet dapat dipilih saat mengunggah

Setiap baris adalah satu observasi dan setiap kolom satu variabel. Kolom kategorik
otomatis diubah menjadi variabel dummy pada analisis regresi.

## Struktur proyek

```
app.py                 Entri aplikasi & navigasi halaman
views/                 Halaman antarmuka Streamlit (satu berkas per metode)
nalardata/           Pustaka perhitungan (murni pandas/numpy, tanpa Streamlit)
  io_utils.py          Pemuatan berkas dan profil data
  preprocessing.py     Missing value, penskalaan, encoding, matriks desain
  descriptive.py       Deskriptif, normalitas, Mardia, pencilan
  assumptions.py       KMO, Bartlett, VIF, Box's M, Levene
  correlation.py       Matriks korelasi, signifikansi, korelasi parsial
  pca_analysis.py      PCA dan analisis paralel Horn
  factor_analysis.py   EFA, rotasi varimax & promax
  clustering.py        K-Means, hierarki, DBSCAN, profil klaster
  regression.py        Regresi linear & logistik, galat baku robust, stepwise, prediksi
  discriminant.py      LDA/QDA, fungsi kanonik, Wilks' lambda
  manova.py            MANOVA satu jalur, Hotelling's T²
  cca.py               Korelasi kanonik
  data_entry.py        Pembuatan data, pembersihan, dan variabel gabungan dari butir
  reliability.py       Alpha, Spearman-Brown, omega, CR, AVE, Fornell-Larcker, HTMT
  audit.py             Audit kualitas data: temuan bertingkat beserta dampak dan sarannya
  grafik.py            Grafik statis matplotlib untuk disisipkan ke berkas laporan
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
  kesimpulan_ui.py     Komponen bersama ketiga halaman laporan
  keranjang.py         Keranjang hasil: kumpulan analisis yang disimpan pengguna
  proyek.py            Simpan/buka berkas proyek .nalardata beserta batas keamanannya
  plots.py             Visualisasi Plotly
  ui.py                Komponen antarmuka bersama
data/                  Contoh data
scripts/               Pembuat contoh data sintetis
jalankan.cmd           Peluncur Windows pada port 8503
jalankan.sh            Peluncur Linux/macOS pada port 8503
tests/                 Uji perhitungan dan uji asap halaman
```

Pustaka `nalardata` dapat dipakai langsung tanpa antarmuka:

```python
import pandas as pd
from nalardata import pca_analysis, regression

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
