"""Pemilih Metode Terpandu: memilih uji dengan melihat data, bukan diagram alur.

Pertanyaan yang paling sering diajukan penulis skripsi adalah "uji apa yang harus
saya pakai". Jawaban yang beredar berbentuk diagram alur di buku teks — berguna,
tetapi buta: ia tidak tahu apakah data Anda menceng, apakah ragam antar kelompok
sama, atau apakah salah satu kelompok hanya berisi tujuh orang.

Aplikasi ini memegang datanya, jadi ia dapat memeriksa. Yang ditanyakan hanyalah
yang **tidak mungkin** disimpulkan dari data:

* apa tujuan penelitiannya;
* variabel mana yang menjadi outcome;
* apakah pengamatannya berpasangan atau saling bebas.

Sisanya dihitung: skala tiap variabel dibaca dari kamus, jumlah kelompok, normalitas
per kelompok, keseragaman ragam, dan ukuran sel terkecil.

Setiap saran wajib menyertakan **alasan mengapa alternatifnya tidak dipilih**. Bagian
itu yang mengajari, dan yang menyelamatkan pengguna ketika penguji bertanya "mengapa
tidak memakai ANOVA?". Saran tanpa alasan hanya memindahkan ketergantungan, dari
buku teks ke aplikasi.

Batasnya dinyatakan terbuka: pemandu membaca **bentuk** data, bukan **maksud**
penelitian. Ia tidak tahu apakah pengamatan Anda benar-benar saling bebas, apakah
variabelnya benar-benar mengukur yang Anda maksud, atau apakah pertanyaan
penelitiannya sudah tepat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from nalardata import kamus as km
from nalardata import pagar
from nalardata import proyek_penelitian as pp

# --------------------------------------------------------------------------- #
# Kosakata
# --------------------------------------------------------------------------- #

TUJUAN = {
    "membandingkan": "Membandingkan kelompok",
    "membandingkan_banyak_outcome": "Membandingkan kelompok pada beberapa hasil sekaligus",
    "menghubungkan": "Menguji hubungan antar variabel",
    "memperkirakan_nilai": "Menjelaskan atau memperkirakan nilai angka",
    "memperkirakan_kategori": "Memperkirakan kategori atau keputusan",
    "menguji_mediasi": "Menguji apakah satu variabel menjadi perantara (mediasi)",
    "menguji_moderasi": "Menguji apakah satu variabel memperkuat/memperlemah pengaruh (moderasi)",
    "meringkas": "Meringkas banyak variabel menjadi sedikit dimensi",
    "mengelompokkan": "Mengelompokkan responden yang mirip",
    "menguji_model": "Menguji model teoretis antar konstruk",
    "mutu_instrumen": "Memeriksa mutu kuesioner",
    "menganalisis_panel": "Menganalisis data panel (entitas x waktu)",
    "meramalkan_waktu": "Meramalkan deret waktu",
}

PERTANYAAN_TUJUAN = {
    "membandingkan": "Apakah kelompok A berbeda dari kelompok B?",
    "membandingkan_banyak_outcome": "Apakah kelompok berbeda pada beberapa ukuran hasil sekaligus?",
    "menghubungkan": "Apakah dua variabel bergerak bersamaan?",
    "memperkirakan_nilai": "Faktor apa yang menjelaskan naik-turunnya sebuah angka?",
    "memperkirakan_kategori": "Faktor apa yang membedakan yang 'ya' dari yang 'tidak'?",
    "menguji_mediasi": "Apakah pengaruh X terhadap Y mengalir lewat variabel perantara M?",
    "menguji_moderasi": "Apakah kekuatan pengaruh X terhadap Y berubah pada tingkat variabel lain?",
    "meringkas": "Belasan butir kuesioner ini sebenarnya mengukur berapa hal?",
    "mengelompokkan": "Ada berapa tipe responden dalam data ini?",
    "menguji_model": "Apakah model hubungan antar konstruk saya didukung data?",
    "mutu_instrumen": "Apakah kuesioner saya valid dan reliabel?",
    "menganalisis_panel": "Bagaimana pengaruh X terhadap Y bila unit yang sama diamati berulang kali?",
    "meramalkan_waktu": "Bagaimana kemungkinan nilainya di masa depan, dari pola masa lalunya sendiri?",
}

TERPENUHI = "terpenuhi"
DILANGGAR = "dilanggar"
TIDAK_DIUJI = "tidak diuji"

# Status informasi sebuah Rekomendasi — bukan status metode, melainkan seberapa
# bisa diandalkan rekomendasi itu sendiri (lihat Rekomendasi.status_informasi).
CUKUP = "cukup"
PERLU_KONFIRMASI = "perlu_konfirmasi"
TIDAK_CUKUP = "tidak_cukup"

ALFA = 0.05
MIN_SEL = 5  # frekuensi harapan minimum pada tabel silang
MIN_KELOMPOK = 3  # anggota minimum agar sebuah kelompok masih dapat diuji
MIN_SHAPIRO = 3

# Ambang untuk aturan parametrik-vs-nonparametrik yang tidak lagi bersandar pada
# Shapiro-Wilk sebagai satu-satunya sakelar (lihat periksa_bentuk_sebaran/
# periksa_pencilan_ekstrem serta _dua_kelompok/_banyak_kelompok di bawah).
MIN_BESAR = 30  # ambang "sampel besar" — sudah jadi catatan tak resmi di
# periksa_ukuran_kelompok dan dipakai nonparametrik.perlu_nonparametrik();
# di sini dijadikan eksplisit dan disatukan, bukan angka baru.
SKEW_BERAT = 2.0  # skewness di atas ini tergolong "penyimpangan berat"
OUTLIER_EKSTREM_K = 3.0  # pagar IQR pencilan "ekstrem" — beda level dari k=1.5
# yang dipakai Rapor Data untuk pencilan biasa


@dataclass
class Syarat:
    """Satu asumsi beserta keadaannya pada data ini."""

    nama: str
    status: str
    rincian: str

    @property
    def terpenuhi(self) -> bool:
        return self.status == TERPENUHI

    @property
    def dilanggar(self) -> bool:
        return self.status == DILANGGAR


# Metode yang benar-benar dapat dijalankan aplikasi ini, beserta halaman dan
# kunci sesi yang dipakai mengoper penetapan variabelnya.
#
# Daftar ini bukan dokumentasi, melainkan pagar: satu uji menuntut setiap metode
# yang dapat disarankan pemandu ada di sini. Tanpa itu, pemandu pernah menyarankan
# uji-t dan ANOVA yang sama sekali belum ada di aplikasi - pengguna yang menuruti
# sarannya tiba di halaman yang tidak dapat menjalankannya.
METODE_TERSEDIA: dict[str, str] = {
    "Uji-t sampel bebas": "Uji Beda",
    "Uji-t Welch": "Uji Beda",
    "Uji-t berpasangan": "Uji Beda",
    "One-Way ANOVA": "Uji Beda",
    "Welch ANOVA": "Uji Beda",
    "Mann-Whitney U": "Uji Beda",
    "Wilcoxon signed-rank": "Uji Beda",
    "Kruskal-Wallis": "Uji Beda",
    "Friedman": "Uji Beda",
    "ANOVA ukur ulang": "MANOVA",
    "MANOVA": "MANOVA",
    "Chi-square": "Uji Beda",
    "Uji eksak Fisher": "Uji Beda",
    "Korelasi Pearson": "Korelasi & Asumsi",
    "Korelasi Spearman": "Korelasi & Asumsi",
    "Korelasi Kendall tau": "Korelasi & Asumsi",
    "Regresi linear berganda": "Regresi",
    "Regresi logistik biner": "Regresi",
    "Regresi Moderasi (MRA)": "Regresi Moderasi (MRA)",
    "Analisis diskriminan": "Analisis Diskriminan",
    "Analisis Faktor Eksploratori (EFA)": "Analisis Faktor",
    "Analisis Komponen Utama (PCA)": "PCA",
    "Analisis klaster": "Analisis Klaster",
    "CFA / Analisis Jalur / SEM": "CFA, Jalur & SEM",
    "Uji validitas dan reliabilitas": "Reliabilitas & Validitas",
    "Regresi Panel": "Regresi Panel",
    "ARIMA": "Deret Waktu (ARIMA)",
}


@dataclass
class Saran:
    """Satu metode yang disarankan atau ditolak, selalu beserta alasannya."""

    metode: str
    halaman: str
    alasan: str
    syarat: list[Syarat] = field(default_factory=list)
    lanjutan: str = ""
    pembanding: str = ""
    peringatan: str = ""
    ditolak_karena: str = ""
    konfig: dict = field(default_factory=dict)
    # Ruas struktur tambahan — semua berdefault aman (kosong) supaya pemanggilan
    # Saran(...) yang sudah ada di seluruh berkas ini tidak wajib diubah, hanya
    # diisi di titik yang relevan. Lihat sarankan() untuk status_bukti dan
    # keterbatasan_desain (diisi dari Rencana penelitian bila tersedia).
    method_key: str = ""
    variant: str = ""
    status_bukti: str = ""
    keterbatasan_desain: list[str] = field(default_factory=list)
    ukuran_efek: str = ""
    tidak_dapat_diperiksa: list[str] = field(
        default_factory=lambda: [
            "Independensi antar-pengamatan",
            "Validitas konstruk yang diukur",
        ]
    )

    def __post_init__(self) -> None:
        # Halaman diambil dari daftar metode, bukan dituliskan ulang di tiap
        # cabang: nama halaman berubah ketika navigasi ditata ulang, dan salinan
        # yang tersebar akan menunjuk tempat yang sudah tidak ada.
        if self.metode in METODE_TERSEDIA:
            self.halaman = METODE_TERSEDIA[self.metode]
        if not self.method_key:
            self.method_key = _slugify(self.metode)

    @property
    def dipilih(self) -> bool:
        return not self.ditolak_karena

    @property
    def tersedia(self) -> bool:
        """Apakah metode ini benar-benar dapat dijalankan di aplikasi ini.

        Saran yang menunjuk metode yang belum ada tetap berguna — pengguna perlu
        tahu apa yang sebenarnya paling tepat — tetapi harus ditandai, bukan
        dibiarkan tampak seperti tombol yang tinggal ditekan.
        """
        return self.metode in METODE_TERSEDIA


@dataclass
class Rekomendasi:
    """Hasil pemandu: satu metode utama, alternatifnya, dan apa yang masih kurang."""

    utama: Saran | None = None
    alternatif: list[Saran] = field(default_factory=list)
    catatan: list[str] = field(default_factory=list)
    belum_terjawab: list[str] = field(default_factory=list)
    konfig: dict = field(default_factory=dict)
    status_informasi: str = CUKUP
    perlu_konfirmasi: list[str] = field(default_factory=list)

    @property
    def berhasil(self) -> bool:
        return self.utama is not None

    def ringkas(self) -> pd.DataFrame:
        """Tabel metode utama dan alternatifnya, agar bedanya terlihat berdampingan."""
        baris = []
        for saran in ([self.utama] if self.utama else []) + self.alternatif:
            baris.append(
                {
                    "Metode": saran.metode,
                    "Status": "Disarankan" if saran.dipilih else "Tidak dipilih",
                    "Alasan": saran.alasan if saran.dipilih else saran.ditolak_karena,
                    "Halaman": saran.halaman,
                }
            )
        return pd.DataFrame(baris)


def _slugify(teks: str) -> str:
    """Slug stabil dari nama metode, mis. 'Uji-t Welch' -> 'uji_t_welch'.

    Dipakai sebagai kunci mesin (method_key) yang tidak berubah walau label
    tampilannya disunting redaksinya — beda dari ``metode`` yang boleh berubah
    kata-katanya kapan saja untuk kejelasan pembaca.
    """
    huruf = [c.lower() if c.isalnum() else "_" for c in teks]
    slug = re.sub(r"_+", "_", "".join(huruf)).strip("_")
    return slug


# --------------------------------------------------------------------------- #
# Pemeriksaan data
# --------------------------------------------------------------------------- #


def _bersih(df: pd.DataFrame, kolom: list[str]) -> pd.DataFrame:
    ada = [k for k in kolom if k in df.columns]
    return df[ada].dropna() if ada else pd.DataFrame()


def periksa_normalitas(df: pd.DataFrame, outcome: str, kelompok: str | None) -> Syarat:
    """Shapiro-Wilk pada tiap kelompok, bukan pada data gabungan.

    Normalitas yang dituntut uji-t dan ANOVA adalah normalitas **dalam** kelompok.
    Data gabungan dari dua kelompok yang masing-masing normal namun berbeda rata-rata
    akan tampak dwipuncak, dan menguji gabungan itu akan menolak normalitas yang
    sebenarnya terpenuhi.
    """
    if outcome not in df.columns:
        return Syarat("Normalitas", TIDAK_DIUJI, "Variabel terikat belum dipilih.")

    nilai = pd.to_numeric(df[outcome], errors="coerce")
    if kelompok and kelompok in df.columns:
        bagian = [
            pd.to_numeric(g, errors="coerce").dropna()
            for _, g in nilai.groupby(df[kelompok])
        ]
    else:
        bagian = [nilai.dropna()]

    diuji, ditolak = 0, 0
    for contoh in bagian:
        if len(contoh) < MIN_SHAPIRO or contoh.nunique() < 2:
            continue
        diuji += 1
        if float(stats.shapiro(contoh)[1]) < ALFA:
            ditolak += 1

    if diuji == 0:
        return Syarat(
            "Normalitas", TIDAK_DIUJI, "Kelompoknya terlalu kecil untuk diuji normalitas."
        )
    if ditolak == 0:
        satuan = "kelompok" if diuji > 1 else "seluruh data"
        return Syarat(
            "Normalitas",
            TERPENUHI,
            f"Shapiro-Wilk tidak menolak normalitas pada {diuji} {satuan}.",
        )
    return Syarat(
        "Normalitas",
        DILANGGAR,
        f"Shapiro-Wilk menolak normalitas pada {ditolak} dari {diuji} kelompok "
        f"(p < {ALFA:.2f}).",
    )


def periksa_homogenitas(df: pd.DataFrame, outcome: str, kelompok: str) -> Syarat:
    """Uji Levene: apakah ragam antar kelompok cukup seragam."""
    bersih = _bersih(df, [outcome, kelompok])
    if bersih.empty or bersih[kelompok].nunique() < 2:
        return Syarat("Keseragaman ragam", TIDAK_DIUJI, "Kelompoknya kurang dari dua.")

    bagian = [
        pd.to_numeric(g, errors="coerce").dropna()
        for _, g in bersih[outcome].groupby(bersih[kelompok])
    ]
    bagian = [b for b in bagian if len(b) >= 2 and b.nunique() > 1]
    if len(bagian) < 2:
        return Syarat(
            "Keseragaman ragam", TIDAK_DIUJI, "Terlalu sedikit data pada tiap kelompok."
        )

    p = float(stats.levene(*bagian, center="median")[1])
    if p >= ALFA:
        return Syarat(
            "Keseragaman ragam", TERPENUHI, f"Levene p = {p:.3f}, ragam antar kelompok seragam."
        )
    return Syarat(
        "Keseragaman ragam",
        DILANGGAR,
        f"Levene p = {p:.3f}, ragam antar kelompok berbeda nyata.",
    )


def periksa_ukuran_kelompok(df: pd.DataFrame, kelompok: str) -> Syarat:
    """Kelompok yang terlalu kecil membuat uji apa pun menjadi tidak stabil."""
    if kelompok not in df.columns:
        return Syarat("Ukuran kelompok", TIDAK_DIUJI, "Penanda kelompok belum dipilih.")

    jumlah = df[kelompok].value_counts()
    if jumlah.empty:
        return Syarat("Ukuran kelompok", TIDAK_DIUJI, "Penanda kelompok kosong.")

    terkecil = int(jumlah.min())
    if terkecil < MIN_KELOMPOK:
        return Syarat(
            "Ukuran kelompok",
            DILANGGAR,
            f"Kelompok terkecil hanya berisi {terkecil} pengamatan "
            f"('{jumlah.idxmin()}').",
        )
    if terkecil < MIN_BESAR:
        return Syarat(
            "Ukuran kelompok",
            TERPENUHI,
            f"Kelompok terkecil berisi {terkecil} pengamatan — cukup untuk diuji, "
            f"namun uji non-parametrik lebih aman di bawah {MIN_BESAR}.",
        )
    return Syarat(
        "Ukuran kelompok", TERPENUHI, f"Kelompok terkecil berisi {terkecil} pengamatan."
    )


def ukuran_tergolong_besar(df: pd.DataFrame, kelompok: str | None) -> bool:
    """Kelompok terkecil (atau seluruh data bila tanpa kelompok) sudah >= MIN_BESAR.

    Dipakai sebagai salah satu syarat teorema limit pusat: pada sampel sebesar ini,
    sebaran rata-rata sampel sudah mendekati normal walau data mentahnya tidak,
    sehingga Shapiro-Wilk yang signifikan tidak lagi otomatis berarti uji parametrik
    tidak layak dipakai.
    """
    if kelompok and kelompok in df.columns:
        jumlah = df[kelompok].value_counts()
        return bool(not jumlah.empty and jumlah.min() >= MIN_BESAR)
    return len(df) >= MIN_BESAR


def periksa_bentuk_sebaran(df: pd.DataFrame, outcome: str, kelompok: str | None) -> Syarat:
    """Skewness per kelompok — pelengkap Shapiro-Wilk yang menilai SEBERAPA berat
    penyimpangan dari normal, bukan sekadar signifikan/tidak signifikan.

    Shapiro-Wilk menolak normalitas pada penyimpangan sekecil apa pun bila sampel
    cukup besar, sehingga tidak dapat dipakai sendirian untuk menilai keparahan.
    """
    if outcome not in df.columns:
        return Syarat("Bentuk sebaran", TIDAK_DIUJI, "Variabel terikat belum dipilih.")

    nilai = pd.to_numeric(df[outcome], errors="coerce")
    if kelompok and kelompok in df.columns:
        bagian = [g.dropna() for _, g in nilai.groupby(df[kelompok])]
    else:
        bagian = [nilai.dropna()]

    skew_maks = 0.0
    diuji = 0
    for contoh in bagian:
        if len(contoh) < 3 or contoh.nunique() < 2:
            continue
        diuji += 1
        skew_maks = max(skew_maks, abs(float(stats.skew(contoh))))

    if diuji == 0:
        return Syarat("Bentuk sebaran", TIDAK_DIUJI, "Kelompoknya terlalu kecil untuk dinilai.")
    if skew_maks > SKEW_BERAT:
        return Syarat(
            "Bentuk sebaran",
            DILANGGAR,
            f"Skewness tertinggi antar kelompok {skew_maks:.2f} — tergolong "
            f"penyimpangan berat (di atas {SKEW_BERAT:.0f}).",
        )
    return Syarat(
        "Bentuk sebaran",
        TERPENUHI,
        f"Skewness tertinggi antar kelompok {skew_maks:.2f} — tidak tergolong berat.",
    )


def periksa_pencilan_ekstrem(df: pd.DataFrame, outcome: str, kelompok: str | None) -> Syarat:
    """Pencilan IQR ekstrem (k=3,0) per kelompok — beda level dari pencilan biasa
    (k=1,5) yang dipakai Rapor Data. Memakai ulang descriptive.univariate_outliers,
    bukan menulis ulang aturan IQR.
    """
    from nalardata import descriptive

    if outcome not in df.columns:
        return Syarat("Pencilan ekstrem", TIDAK_DIUJI, "Variabel terikat belum dipilih.")

    nilai = pd.to_numeric(df[outcome], errors="coerce")
    if kelompok and kelompok in df.columns:
        bagian = {str(nama): g.dropna() for nama, g in nilai.groupby(df[kelompok])}
    else:
        bagian = {outcome: nilai.dropna()}

    bermasalah = []
    for nama, contoh in bagian.items():
        if len(contoh) < 4:
            continue
        ringkas = descriptive.univariate_outliers(
            pd.DataFrame({outcome: contoh}), k=OUTLIER_EKSTREM_K
        )
        jumlah = int(ringkas.loc[0, "Jumlah Pencilan"]) if not ringkas.empty else 0
        if jumlah:
            bermasalah.append(f"{jumlah} pada '{nama}'")

    if not bermasalah:
        return Syarat(
            "Pencilan ekstrem", TERPENUHI, "Tidak ditemukan pencilan ekstrem pada kelompok mana pun."
        )
    return Syarat(
        "Pencilan ekstrem",
        DILANGGAR,
        "Ditemukan pencilan ekstrem: " + ", ".join(bermasalah) + ".",
    )


def periksa_frekuensi_harapan(df: pd.DataFrame, satu: str, dua: str) -> Syarat:
    """Chi-square menuntut frekuensi harapan minimal lima pada sebagian besar sel."""
    bersih = _bersih(df, [satu, dua])
    if bersih.empty:
        return Syarat("Frekuensi harapan", TIDAK_DIUJI, "Data kosong setelah dibersihkan.")

    silang = pd.crosstab(bersih[satu], bersih[dua])
    if silang.shape[0] < 2 or silang.shape[1] < 2:
        return Syarat(
            "Frekuensi harapan", TIDAK_DIUJI, "Tabel silang perlu sekurang-kurangnya 2x2."
        )

    harapan = stats.chi2_contingency(silang)[3]
    kecil = int((harapan < MIN_SEL).sum())
    total = int(harapan.size)
    bagian = kecil / total
    if kecil == 0:
        return Syarat(
            "Frekuensi harapan", TERPENUHI, f"Seluruh {total} sel berharapan minimal {MIN_SEL}."
        )
    if bagian <= 0.20:
        return Syarat(
            "Frekuensi harapan",
            TERPENUHI,
            f"{kecil} dari {total} sel ({bagian:.0%}) berharapan di bawah {MIN_SEL} — "
            "masih di dalam batas yang lazim diterima.",
        )
    return Syarat(
        "Frekuensi harapan",
        DILANGGAR,
        f"{kecil} dari {total} sel ({bagian:.0%}) berharapan di bawah {MIN_SEL}.",
    )


def periksa_multikolinearitas(df: pd.DataFrame, prediktor: list[str]) -> Syarat:
    """VIF antar prediktor; di atas 10 lazim dianggap bermasalah."""
    from nalardata import assumptions

    bersih = _bersih(df, prediktor)
    if bersih.shape[1] < 2 or len(bersih) < 10:
        return Syarat(
            "Multikolinearitas", TIDAK_DIUJI, "Perlu sekurang-kurangnya dua prediktor."
        )
    try:
        tabel = assumptions.vif(bersih)
    except Exception:  # noqa: BLE001 - VIF gagal pada matriks singular
        return Syarat(
            "Multikolinearitas",
            DILANGGAR,
            "VIF tidak dapat dihitung, lazimnya karena ada prediktor yang merupakan "
            "kombinasi persis dari prediktor lain.",
        )

    # Kolom dirujuk dengan namanya: tabel VIF memuat R2 lebih dulu, dan membaca
    # kolom kedua secara posisi akan mengambil R2 sehingga multikolinearitas
    # setinggi apa pun tidak pernah terdeteksi.
    if "VIF" not in tabel.columns:
        return Syarat("Multikolinearitas", TIDAK_DIUJI, "Tabel VIF tidak berbentuk seperti dugaan.")
    nilai = pd.to_numeric(tabel["VIF"], errors="coerce")
    if not np.isfinite(nilai).any():
        return Syarat(
            "Multikolinearitas",
            DILANGGAR,
            "VIF tidak terhingga, yang berarti ada prediktor yang merupakan kombinasi "
            "persis dari prediktor lain.",
        )
    puncak = float(nilai.max())
    if puncak >= 10:
        nama = tabel.loc[nilai.idxmax(), "Variabel"]
        return Syarat(
            "Multikolinearitas", DILANGGAR, f"VIF tertinggi {puncak:,.1f} pada '{nama}'."
        )
    return Syarat("Multikolinearitas", TERPENUHI, f"VIF tertinggi {puncak:.1f}, di bawah 10.")


def periksa_kelayakan_faktor(df: pd.DataFrame, variabel: list[str]) -> list[Syarat]:
    """KMO dan Bartlett: apakah data ini layak difaktorkan sama sekali."""
    from nalardata import assumptions

    bersih = _bersih(df, variabel)
    if bersih.shape[1] < 3 or len(bersih) < 20:
        return [
            Syarat(
                "Kelayakan faktor",
                TIDAK_DIUJI,
                "Perlu sekurang-kurangnya tiga variabel dan dua puluh pengamatan.",
            )
        ]

    hasil = []
    try:
        nilai = float(assumptions.kmo(bersih).overall)
        hasil.append(
            Syarat(
                "KMO",
                TERPENUHI if nilai >= 0.5 else DILANGGAR,
                f"KMO = {nilai:.3f}" + ("" if nilai >= 0.5 else ", di bawah batas 0,50."),
            )
        )
    except Exception:  # noqa: BLE001
        hasil.append(Syarat("KMO", TIDAK_DIUJI, "KMO tidak dapat dihitung pada data ini."))

    try:
        p = float(assumptions.bartlett_sphericity(bersih).p_value)
        hasil.append(
            Syarat(
                "Bartlett",
                TERPENUHI if p < ALFA else DILANGGAR,
                f"Bartlett p = {p:.4f}"
                + (
                    ""
                    if p < ALFA
                    else " — korelasi antar variabel tidak berbeda dari matriks identitas."
                ),
            )
        )
    except Exception:  # noqa: BLE001
        hasil.append(Syarat("Bartlett", TIDAK_DIUJI, "Uji Bartlett tidak dapat dijalankan."))
    return hasil


# --------------------------------------------------------------------------- #
# Pemandu
# --------------------------------------------------------------------------- #


def sarankan(
    df: pd.DataFrame,
    kamus: km.Kamus,
    tujuan: str,
    outcome: str | None = None,
    prediktor: list[str] | None = None,
    kelompok: str | None = None,
    berpasangan: bool = False,
    penelitian: pp.ProyekPenelitian | None = None,
) -> Rekomendasi:
    """Sarankan metode dengan memeriksa data yang sungguh ada.

    ``berpasangan`` adalah satu-satunya hal yang ditanyakan dan tidak dihitung:
    tidak ada cara membaca dari angka apakah dua kolom berasal dari orang yang sama
    diukur dua kali atau dari dua orang berbeda.

    ``penelitian`` (Rencana/Ruang Proyek) bersifat opsional dan tidak pernah
    mengubah METODE yang disarankan — bentuk data yang memutuskan itu, bukan
    desain penelitian. Yang dipengaruhi hanyalah BAHASA alasannya (kata kerja
    "berpengaruh terhadap" vs "berhubungan dengan", lewat ``pagar.kata_hubungan``)
    pada tujuan yang benar-benar menyusun kalimat hubungan sebab-akibat/asosiatif.
    """
    if tujuan not in TUJUAN:
        raise ValueError(f"Tujuan '{tujuan}' tidak dikenal. Pilih dari {list(TUJUAN)}.")

    prediktor = [p for p in (prediktor or []) if p in df.columns]
    if outcome is not None and outcome not in df.columns:
        outcome = None
    if kelompok is not None and kelompok not in df.columns:
        kelompok = None

    penanganan = {
        "membandingkan": _membandingkan,
        "membandingkan_banyak_outcome": _membandingkan_banyak_outcome,
        "menghubungkan": _menghubungkan,
        "memperkirakan_nilai": _memperkirakan_nilai,
        "memperkirakan_kategori": _memperkirakan_kategori,
        "menguji_mediasi": _mediasi,
        "menguji_moderasi": _moderasi,
        "meringkas": _meringkas,
        "mengelompokkan": _mengelompokkan,
        "menguji_model": _menguji_model,
        "mutu_instrumen": _mutu_instrumen,
        "menganalisis_panel": _panel,
        "meramalkan_waktu": _arima,
    }
    hasil = penanganan[tujuan](df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian)

    # Penetapan variabel diikutkan pada hasil agar halaman metode dapat terbuka
    # sudah terisi. Tanpa ini, pengguna yang baru saja memberi tahu pemandu
    # variabel mana yang dipakai harus memilihnya sekali lagi di halaman berikut.
    hasil.konfig = {
        "tujuan": tujuan,
        "outcome": outcome or "",
        "prediktor": list(prediktor),
        "kelompok": kelompok or "",
        "berpasangan": bool(berpasangan),
    }
    if hasil.utama is not None:
        hasil.utama.konfig = dict(hasil.konfig, metode=hasil.utama.metode)
        # Dipindah dari lapisan tampilan (views/pemandu.py) supaya jadi bagian
        # struktur hasil, bukan dihitung ulang di titik render — pagar.py dan
        # penelitian.batas_kesimpulan() sudah ada dan sudah teruji, di sini
        # hanya disambungkan.
        hasil.utama.status_bukti = pagar.label_eksploratori(hasil.utama.metode, penelitian)
        if penelitian is not None:
            hasil.utama.keterbatasan_desain = penelitian.batas_kesimpulan()

    # Status informasi: seberapa bisa diandalkan rekomendasi ini, terpisah dari
    # metode yang dipilih. Hanya variabel yang BENAR-BENAR dipakai yang diperiksa —
    # bukan seluruh dataset — supaya peringatannya spesifik dan dapat ditindaklanjuti.
    if hasil.belum_terjawab:
        hasil.status_informasi = TIDAK_CUKUP
    else:
        dipakai = [n for n in ([outcome, kelompok] + list(prediktor)) if n]
        dipakai = list(dict.fromkeys(dipakai))
        hasil.perlu_konfirmasi = [
            n for n in dipakai if n in kamus and kamus[n].perlu_diperiksa
        ]
        hasil.status_informasi = PERLU_KONFIRMASI if hasil.perlu_konfirmasi else CUKUP
    return hasil


def _numerik(kamus: km.Kamus, nama: str | None) -> bool:
    """Numerik menurut kamus, bukan menurut dtype pandas.

    Skor Likert bertipe angka tetapi ordinal, dan memperlakukannya sebagai numerik
    adalah kekeliruan yang paling sering terjadi tanpa disadari.
    """
    return bool(nama and nama in kamus and kamus[nama].numerik)


def _ordinal(kamus: km.Kamus, nama: str | None) -> bool:
    return bool(nama and nama in kamus and kamus[nama].skala == km.ORDINAL)


# --------------------------------------------------------------------------- #
# Membandingkan kelompok
# --------------------------------------------------------------------------- #


def _membandingkan(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    hasil = Rekomendasi()

    if berpasangan:
        return _membandingkan_berpasangan(df, kamus, outcome, prediktor)

    if not outcome:
        hasil.belum_terjawab.append("Variabel mana yang dibandingkan antar kelompok?")
        return hasil
    if not kelompok:
        hasil.belum_terjawab.append("Kolom mana yang menandai kelompoknya?")
        return hasil

    bersih = _bersih(df, [outcome, kelompok])
    k = int(bersih[kelompok].nunique()) if not bersih.empty else 0
    if k < 2:
        hasil.catatan.append(
            f"Kolom '{kelompok}' hanya memuat {k} kelompok, sehingga tidak ada yang "
            "dapat dibandingkan."
        )
        return hasil

    ukuran = periksa_ukuran_kelompok(bersih, kelompok)

    # Outcome kategorik: bandingkan sebaran, bukan rata-rata.
    if not _numerik(kamus, outcome) and not _ordinal(kamus, outcome):
        sel = periksa_frekuensi_harapan(bersih, outcome, kelompok)
        if sel.dilanggar:
            hasil.utama = Saran(
                metode="Uji eksak Fisher",
                halaman="Uji Non-parametrik",
                alasan=(
                    f"Variabel terikat '{outcome}' berskala kategori dan sebagian sel "
                    "berfrekuensi harapan kecil, sehingga hampiran chi-square kurang "
                    "dapat dipercaya."
                ),
                syarat=[sel, ukuran],
                pembanding="SPSS: Analyze ▸ Descriptive Statistics ▸ Crosstabs ▸ Exact",
            )
            hasil.alternatif.append(
                Saran(
                    metode="Chi-square",
                    halaman="Uji Non-parametrik",
                    alasan="",
                    ditolak_karena=sel.rincian
                    + " Chi-square menuntut frekuensi harapan minimal lima pada "
                    "sebagian besar sel.",
                )
            )
            return hasil

        hasil.utama = Saran(
            metode="Chi-square",
            halaman="Uji Non-parametrik",
            alasan=(
                f"Variabel terikat '{outcome}' dan penanda kelompok '{kelompok}' "
                "sama-sama berskala kategori, sehingga yang dibandingkan adalah "
                "sebarannya."
            ),
            syarat=[sel, ukuran],
            lanjutan="Laporkan Cramér's V sebagai ukuran kekuatan hubungannya.",
            pembanding="SPSS: Analyze ▸ Descriptive Statistics ▸ Crosstabs ▸ Chi-square",
        )
        hasil.alternatif.append(
            Saran(
                metode="Uji-t / ANOVA",
                halaman="Uji Non-parametrik",
                alasan="",
                ditolak_karena=(
                    f"'{outcome}' berskala kategori, sehingga rata-ratanya tidak "
                    "bermakna. Bila menurut Anda variabel ini sebenarnya angka, "
                    "ubah skalanya pada Kamus Variabel."
                ),
            )
        )
        return hasil

    normal = periksa_normalitas(bersih, outcome, kelompok)
    seragam = periksa_homogenitas(bersih, outcome, kelompok)
    sebaran = periksa_bentuk_sebaran(bersih, outcome, kelompok)
    pencilan = periksa_pencilan_ekstrem(bersih, outcome, kelompok)
    syarat = [normal, seragam, sebaran, pencilan, ukuran]

    ordinal = _ordinal(kamus, outcome)
    besar = ukuran_tergolong_besar(bersih, kelompok)

    if k == 2:
        return _dua_kelompok(
            hasil, outcome, kelompok, syarat, ordinal, normal, seragam, sebaran, pencilan, besar
        )
    return _banyak_kelompok(
        hasil, outcome, kelompok, k, syarat, ordinal, normal, seragam, sebaran, pencilan, besar
    )


def _dua_kelompok(hasil, outcome, kelompok, syarat, ordinal, normal, seragam, sebaran, pencilan, besar):
    if ordinal:
        alasan = (
            f"'{outcome}' berskala ordinal, sehingga jarak antar tingkatnya belum tentu "
            "sama dan rata-rata kurang bermakna."
        )
        return _dua_kelompok_nonparametrik(hasil, syarat, alasan)

    if pencilan.dilanggar:
        alasan = (
            f"{pencilan.rincian} Pencilan seekstrem ini menarik rata-rata secara tidak "
            "wajar dan tidak dapat dipertanggungjawabkan sebagai bagian dari sebaran normal."
        )
        return _dua_kelompok_nonparametrik(hasil, syarat, alasan)

    if not besar and sebaran.dilanggar:
        alasan = (
            f"Ukuran kelompok belum tergolong besar (di bawah {MIN_BESAR}) dan "
            f"{sebaran.rincian.lower()} Pada sampel sekecil ini, penyimpangan seberat "
            "itu cukup mengganggu keakuratan uji parametrik."
        )
        return _dua_kelompok_nonparametrik(hasil, syarat, alasan)

    # Lolos seluruh gerbang non-parametrik: outcome benar-benar numerik, kelompok
    # bebas, ukuran kelompok memadai atau penyimpangannya tidak berat, dan tidak ada
    # pencilan ekstrem. Parametrik layak dipakai walau Shapiro-Wilk masih signifikan.
    catatan_clt = ""
    if normal.dilanggar:
        sebab_aman = "ukuran kelompok sudah tergolong besar" if besar else "penyimpangannya tidak tergolong berat"
        catatan_clt = (
            f" Shapiro-Wilk memang menolak normalitas ({normal.rincian.lower()}), "
            f"tetapi {sebab_aman} dan tidak ditemukan pencilan ekstrem, sehingga uji "
            "parametrik tetap tahan dipakai."
        )

    if seragam.dilanggar:
        hasil.utama = Saran(
            metode="Uji-t Welch",
            halaman="Uji Non-parametrik",
            alasan=(
                "Dua kelompok bebas dengan ragam tidak seragam. "
                f"{seragam.rincian}{catatan_clt} Welch tidak menuntut ragam yang sama, "
                "sehingga jadi pilihan parametrik yang lebih aman di sini."
            ),
            syarat=syarat,
            lanjutan="Laporkan Cohen's d sebagai ukuran efeknya.",
            pembanding="SPSS: Independent-Samples T Test, baris 'Equal variances not assumed'",
        )
        hasil.alternatif.append(
            Saran(
                metode="Uji-t sampel bebas (ragam sama)",
                halaman="Uji Non-parametrik",
                alasan="",
                ditolak_karena=seragam.rincian,
            )
        )
        return hasil

    hasil.utama = Saran(
        metode="Uji-t sampel bebas",
        halaman="Uji Non-parametrik",
        alasan=(
            f"Dua kelompok bebas, ragam antar kelompok seragam ({seragam.rincian.lower()})."
            f"{catatan_clt}"
        ),
        syarat=syarat,
        lanjutan="Laporkan Cohen's d sebagai ukuran efeknya.",
        pembanding="SPSS: Analyze ▸ Compare Means ▸ Independent-Samples T Test",
    )
    hasil.alternatif.append(
        Saran(
            metode="Mann-Whitney U",
            halaman="Uji Non-parametrik",
            alasan="",
            ditolak_karena=(
                "Seluruh asumsi uji-t terpenuhi, sehingga uji parametrik lebih peka "
                "menemukan perbedaan yang memang ada. Mann-Whitney tetap sah dipakai "
                "bila Anda ingin lebih berhati-hati."
            ),
        )
    )
    return hasil


def _dua_kelompok_nonparametrik(hasil, syarat, alasan):
    hasil.utama = Saran(
        metode="Mann-Whitney U",
        halaman="Uji Non-parametrik",
        alasan=f"Dua kelompok bebas dibandingkan tanpa asumsi normalitas. {alasan}",
        syarat=syarat,
        lanjutan="Laporkan rank-biserial correlation sebagai ukuran efeknya.",
        pembanding="SPSS: Analyze ▸ Nonparametric Tests ▸ Independent Samples",
    )
    hasil.alternatif.append(
        Saran(
            metode="Uji-t sampel bebas",
            halaman="Uji Non-parametrik",
            alasan="",
            ditolak_karena=alasan,
            peringatan=(
                "Boleh tetap dipakai bila sampel tiap kelompok besar, namun "
                "pelanggaran asumsinya wajib disebutkan pada laporan."
            ),
        )
    )
    return hasil


def _banyak_kelompok(hasil, outcome, kelompok, k, syarat, ordinal, normal, seragam, sebaran, pencilan, besar):
    if ordinal:
        alasan = f"'{outcome}' berskala ordinal."
        return _banyak_kelompok_nonparametrik(hasil, k, syarat, alasan)

    if pencilan.dilanggar:
        alasan = (
            f"{pencilan.rincian} Pencilan seekstrem ini tidak dapat dipertanggungjawabkan "
            "sebagai bagian dari sebaran normal."
        )
        return _banyak_kelompok_nonparametrik(hasil, k, syarat, alasan)

    if not besar and sebaran.dilanggar:
        alasan = (
            f"Ukuran kelompok belum tergolong besar (di bawah {MIN_BESAR}) dan "
            f"{sebaran.rincian.lower()} Pada sampel sekecil ini, penyimpangan seberat "
            "itu cukup mengganggu keakuratan ANOVA."
        )
        return _banyak_kelompok_nonparametrik(hasil, k, syarat, alasan)

    catatan_clt = ""
    if normal.dilanggar:
        sebab_aman = "ukuran kelompok sudah tergolong besar" if besar else "penyimpangannya tidak tergolong berat"
        catatan_clt = (
            f" Shapiro-Wilk memang menolak normalitas pada sebagian kelompok "
            f"({normal.rincian.lower()}), tetapi {sebab_aman} dan tidak ditemukan "
            "pencilan ekstrem, sehingga ANOVA tetap tahan dipakai."
        )

    if seragam.dilanggar:
        hasil.utama = Saran(
            metode="Welch ANOVA",
            halaman="Uji Non-parametrik",
            alasan=(
                f"{k} kelompok bebas dengan ragam tidak seragam. {seragam.rincian}"
                f"{catatan_clt} Welch ANOVA tidak menuntut ragam yang sama, sehingga "
                "jadi pilihan parametrik yang lebih aman di sini."
            ),
            syarat=syarat,
            lanjutan="Uji lanjut Games-Howell, yang juga tidak menuntut ragam seragam.",
            pembanding="SPSS: One-Way ANOVA ▸ Options ▸ Welch",
        )
        hasil.alternatif.append(
            Saran(
                metode="One-Way ANOVA",
                halaman="Uji Non-parametrik",
                alasan="",
                ditolak_karena=seragam.rincian,
            )
        )
        hasil.alternatif.append(
            Saran(
                metode="Kruskal-Wallis",
                halaman="Uji Non-parametrik",
                alasan="",
                ditolak_karena=(
                    "Sebarannya normal, sehingga uji parametrik lebih peka. "
                    "Kruskal-Wallis tetap dapat dipakai sebagai pemeriksaan silang."
                ),
            )
        )
        return hasil

    hasil.utama = Saran(
        metode="One-Way ANOVA",
        halaman="Uji Non-parametrik",
        alasan=(
            f"{k} kelompok bebas, ragam antar kelompok seragam ({seragam.rincian.lower()})."
            f"{catatan_clt}"
        ),
        syarat=syarat,
        lanjutan="Uji lanjut Tukey HSD untuk mengetahui pasangan mana yang berbeda.",
        pembanding="SPSS: Analyze ▸ Compare Means ▸ One-Way ANOVA",
    )
    hasil.alternatif.append(
        Saran(
            metode="Kruskal-Wallis",
            halaman="Uji Non-parametrik",
            alasan="",
            ditolak_karena=(
                "Seluruh asumsi ANOVA terpenuhi, sehingga uji parametrik lebih peka "
                "menemukan perbedaan yang memang ada."
            ),
        )
    )
    return hasil


def _banyak_kelompok_nonparametrik(hasil, k, syarat, alasan):
    hasil.utama = Saran(
        metode="Kruskal-Wallis",
        halaman="Uji Non-parametrik",
        alasan=f"{k} kelompok bebas dibandingkan tanpa asumsi normalitas. {alasan}",
        syarat=syarat,
        lanjutan="Uji lanjut Dunn dengan koreksi Holm untuk mengetahui pasangan mana yang berbeda.",
        pembanding="SPSS: Analyze ▸ Nonparametric Tests ▸ Independent Samples",
    )
    hasil.alternatif.append(
        Saran(
            metode="One-Way ANOVA",
            halaman="Uji Non-parametrik",
            alasan="",
            ditolak_karena=alasan,
            peringatan=(
                "Boleh tetap dipakai bila tiap kelompok berisi cukup banyak "
                "pengamatan, dengan menyebutkan pelanggaran asumsinya."
            ),
        )
    )
    return hasil


def _membandingkan_berpasangan(df, kamus, outcome, prediktor) -> Rekomendasi:
    """Pengukuran berulang pada unit yang sama, tersimpan sebagai beberapa kolom.

    Asumsi diperiksa pada SELISIH antarpengukuran (dua kolom) atau residual
    antar-kondisi (tiga kolom atau lebih) — bukan pada salah satu kolom mentah
    secara terpisah, karena itulah yang sebenarnya disyaratkan uji-t berpasangan
    dan ANOVA pengukuran berulang.
    """
    hasil = Rekomendasi()
    kolom = [k for k in ([outcome] if outcome else []) + list(prediktor) if k]
    kolom = list(dict.fromkeys(kolom))

    if len(kolom) < 2:
        hasil.belum_terjawab.append(
            "Kolom mana saja yang memuat pengukuran berulang pada unit yang sama?"
        )
        return hasil

    bersih = _bersih(df, kolom)
    ordinal = any(_ordinal(kamus, k) for k in kolom)
    besar = len(bersih) >= MIN_BESAR

    if len(kolom) == 2:
        selisih = pd.DataFrame({"selisih": bersih[kolom[0]] - bersih[kolom[1]]})
        normal = periksa_normalitas(selisih, "selisih", None)
        sebaran = periksa_bentuk_sebaran(selisih, "selisih", None)
        pencilan = periksa_pencilan_ekstrem(selisih, "selisih", None)
        syarat = [normal, sebaran, pencilan]

        if ordinal:
            alasan = "Salah satu atau kedua pengukuran berskala ordinal, sehingga jarak antar tingkat belum tentu sama."
            return _berpasangan_nonparametrik(hasil, syarat, alasan)
        if pencilan.dilanggar:
            alasan = (
                f"{pencilan.rincian} Pencilan seekstrem pada selisih kedua pengukuran "
                "ini tidak dapat dipertanggungjawabkan."
            )
            return _berpasangan_nonparametrik(hasil, syarat, alasan)
        if not besar and sebaran.dilanggar:
            alasan = (
                f"Jumlah pasangan belum tergolong besar (di bawah {MIN_BESAR}) dan "
                f"{sebaran.rincian.lower()} Pada sampel sekecil ini, penyimpangan "
                "seberat itu cukup mengganggu keakuratan uji-t berpasangan."
            )
            return _berpasangan_nonparametrik(hasil, syarat, alasan)

        if normal.dilanggar:
            sebab_aman = "jumlah pasangan sudah tergolong besar" if besar else "penyimpangannya tidak tergolong berat"
            alasan_utama = (
                "Dua pengukuran pada unit yang sama. Shapiro-Wilk pada selisihnya "
                f"signifikan ({normal.rincian.lower()}), tetapi {sebab_aman} dan "
                "tidak ditemukan pencilan ekstrem, sehingga uji-t berpasangan tetap "
                "tahan dipakai."
            )
        else:
            alasan_utama = "Dua pengukuran pada unit yang sama, selisihnya bersebaran normal."

        hasil.utama = Saran(
            metode="Uji-t berpasangan",
            halaman="Uji Non-parametrik",
            alasan=alasan_utama,
            syarat=syarat,
            lanjutan="Laporkan Cohen's d untuk sampel berpasangan.",
            pembanding="SPSS: Analyze ▸ Compare Means ▸ Paired-Samples T Test",
        )
        hasil.alternatif.append(
            Saran(
                metode="Wilcoxon signed-rank",
                halaman="Uji Non-parametrik",
                alasan="",
                ditolak_karena="Asumsi pada selisihnya cukup terpenuhi, sehingga uji-t lebih peka.",
            )
        )
        return hasil

    # Tiga kondisi atau lebih: periksa residual antar-kondisi (nilai dikurangi
    # rata-rata subjek itu sendiri), bukan salah satu kolom mentah.
    residual = bersih[kolom].sub(bersih[kolom].mean(axis=1), axis=0)
    residual_gab = pd.DataFrame({"residual": residual.to_numpy().ravel()})
    normal = periksa_normalitas(residual_gab, "residual", None)
    sebaran = periksa_bentuk_sebaran(residual_gab, "residual", None)
    pencilan = periksa_pencilan_ekstrem(residual_gab, "residual", None)
    syarat = [normal, sebaran, pencilan]

    if ordinal:
        alasan = "Salah satu kolom pengukuran berskala ordinal."
        return _berpasangan_banyak_nonparametrik(hasil, len(kolom), syarat, alasan)
    if pencilan.dilanggar:
        alasan = f"{pencilan.rincian} Pencilan seekstrem ini tidak dapat dipertanggungjawabkan."
        return _berpasangan_banyak_nonparametrik(hasil, len(kolom), syarat, alasan)
    if not besar and sebaran.dilanggar:
        alasan = (
            f"Jumlah subjek belum tergolong besar (di bawah {MIN_BESAR}) dan "
            f"{sebaran.rincian.lower()} Pada sampel sekecil ini, penyimpangan seberat "
            "itu cukup mengganggu keakuratan ANOVA pengukuran berulang."
        )
        return _berpasangan_banyak_nonparametrik(hasil, len(kolom), syarat, alasan)

    if normal.dilanggar:
        sebab_aman = "jumlah subjek sudah tergolong besar" if besar else "penyimpangannya tidak tergolong berat"
        alasan_utama = (
            f"{len(kolom)} pengukuran berulang pada unit yang sama. Shapiro-Wilk pada "
            f"residual antar-kondisi signifikan ({normal.rincian.lower()}), tetapi "
            f"{sebab_aman} dan tidak ditemukan pencilan ekstrem, sehingga ANOVA "
            "pengukuran berulang tetap tahan dipakai."
        )
    else:
        alasan_utama = (
            f"{len(kolom)} pengukuran berulang pada unit yang sama; residual "
            "antar-kondisinya bersebaran normal."
        )

    hasil.utama = Saran(
        metode="ANOVA ukur ulang",
        halaman="MANOVA",
        alasan=alasan_utama,
        syarat=syarat,
        lanjutan=(
            "Periksa sphericity Mauchly pada halaman metode; pakai koreksi "
            "Greenhouse-Geisser bila dilanggar."
        ),
        pembanding="SPSS: Analyze ▸ General Linear Model ▸ Repeated Measures",
    )
    hasil.alternatif.append(
        Saran(
            metode="Friedman",
            halaman="Uji Non-parametrik",
            alasan="",
            ditolak_karena=(
                "Asumsi pada residual antar-kondisi cukup terpenuhi, sehingga ANOVA "
                "pengukuran berulang lebih peka menemukan perbedaan yang memang ada. "
                "Friedman tetap dapat dipakai sebagai pemeriksaan silang."
            ),
        )
    )
    return hasil


def _berpasangan_nonparametrik(hasil, syarat, alasan):
    hasil.utama = Saran(
        metode="Wilcoxon signed-rank",
        halaman="Uji Non-parametrik",
        alasan=f"Dua pengukuran pada unit yang sama dibandingkan tanpa asumsi normalitas. {alasan}",
        syarat=syarat,
        pembanding="SPSS: Analyze ▸ Nonparametric Tests ▸ Related Samples",
    )
    hasil.alternatif.append(
        Saran(
            metode="Uji-t berpasangan",
            halaman="Uji Non-parametrik",
            alasan="",
            ditolak_karena=alasan,
        )
    )
    return hasil


def _berpasangan_banyak_nonparametrik(hasil, jumlah_kolom, syarat, alasan):
    hasil.utama = Saran(
        metode="Friedman",
        halaman="Uji Non-parametrik",
        alasan=(
            f"{jumlah_kolom} pengukuran berulang pada unit yang sama, dibandingkan "
            f"tanpa asumsi normalitas. {alasan}"
        ),
        syarat=syarat,
        lanjutan="Laporkan Kendall's W sebagai ukuran kesepakatan antar pengukuran.",
        pembanding="SPSS: Analyze ▸ Nonparametric Tests ▸ Related Samples ▸ Friedman",
    )
    hasil.alternatif.append(
        Saran(
            metode="ANOVA ukur ulang",
            halaman="MANOVA",
            alasan="",
            ditolak_karena=alasan,
        )
    )
    return hasil


def _membandingkan_banyak_outcome(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    """MANOVA: beberapa variabel hasil dibandingkan sekaligus antar kelompok.

    ``prediktor`` di sini bukan penjelas seperti pada regresi, melainkan daftar
    variabel hasil (dependen) yang diuji bersama — mengikuti bentuk widget
    multiselect yang sama seperti tujuan lain, bukan parameter baru.
    """
    hasil = Rekomendasi()
    variabel = [v for v in prediktor if _numerik(kamus, v)]
    if not kelompok:
        hasil.belum_terjawab.append("Kolom mana yang menandai kelompoknya?")
    if len(variabel) < 2:
        hasil.belum_terjawab.append(
            "Variabel hasil (dependen) mana saja yang dibandingkan sekaligus? "
            "Perlu sekurang-kurangnya dua."
        )
    if hasil.belum_terjawab:
        return hasil

    bersih = _bersih(df, variabel + [kelompok])
    k = int(bersih[kelompok].nunique()) if not bersih.empty else 0
    if k < 2:
        hasil.catatan.append(
            f"Kolom '{kelompok}' hanya memuat {k} kelompok, sehingga tidak ada yang "
            "dapat dibandingkan."
        )
        return hasil

    ukuran = periksa_ukuran_kelompok(bersih, kelompok)
    normalitas = [periksa_normalitas(bersih, v, kelompok) for v in variabel]
    langgar_normal = [s for s in normalitas if s.dilanggar]
    sebaran = Syarat(
        "Normalitas per variabel hasil",
        DILANGGAR if langgar_normal else TERPENUHI,
        (
            f"{len(langgar_normal)} dari {len(variabel)} variabel hasil menolak "
            "normalitas univariat pada Shapiro-Wilk — indikasi awal, bukan uji "
            "normalitas multivariat sesungguhnya (diperiksa penuh pada halaman MANOVA)."
            if langgar_normal
            else f"Tidak satu pun dari {len(variabel)} variabel hasil menolak normalitas "
            "univariat pada Shapiro-Wilk — indikasi awal, bukan uji normalitas "
            "multivariat sesungguhnya (diperiksa penuh pada halaman MANOVA)."
        ),
    )
    syarat = [ukuran, sebaran]

    hasil.utama = Saran(
        metode="MANOVA",
        halaman="MANOVA",
        alasan=(
            f"{len(variabel)} variabel hasil dibandingkan sekaligus antar {k} kelompok "
            f"pada '{kelompok}'. MANOVA memperhitungkan korelasi antar variabel hasil "
            "dan menjaga tingkat kesalahan tipe I dibanding menjalankan ANOVA terpisah "
            "untuk tiap variabel."
        ),
        syarat=syarat,
        lanjutan=(
            "Box's M dan normalitas multivariat diperiksa penuh pada halaman MANOVA. "
            "Bila ada variabel kovariat yang ingin dikendalikan, pakai tab MANCOVA "
            "pada halaman yang sama alih-alih menjalankan MANOVA biasa."
        ),
        ukuran_efek="Eta-squared parsial per variabel hasil (tab ANOVA Lanjutan).",
        pembanding="SPSS: Analyze ▸ General Linear Model ▸ Multivariate",
    )
    hasil.alternatif.append(
        Saran(
            metode="ANOVA terpisah untuk tiap variabel hasil",
            halaman="Uji Beda",
            alasan="",
            ditolak_karena=(
                "Menjalankan ANOVA satu-satu untuk tiap variabel hasil menaikkan "
                "peluang kesalahan tipe I secara keseluruhan dan mengabaikan korelasi "
                "antar variabel hasil yang justru diperhitungkan MANOVA."
            ),
        )
    )
    return hasil


# --------------------------------------------------------------------------- #
# Menguji hubungan
# --------------------------------------------------------------------------- #


def _menghubungkan(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    hasil = Rekomendasi()
    kolom = [k for k in ([outcome] if outcome else []) + list(prediktor) if k]
    kolom = list(dict.fromkeys(kolom))

    if len(kolom) < 2:
        hasil.belum_terjawab.append("Dua variabel mana yang ingin Anda hubungkan?")
        return hasil

    kata = pagar.kata_hubungan(penelitian)
    kalimat_hubungan = f" Laporkan sebagai '{kolom[0]} {kata} {kolom[1]}'."

    semua_numerik = all(_numerik(kamus, k) for k in kolom)
    ada_ordinal = any(_ordinal(kamus, k) for k in kolom)
    semua_kategorik = all(k in kamus and kamus[k].kategorik for k in kolom)

    if semua_kategorik and not ada_ordinal:
        sel = periksa_frekuensi_harapan(df, kolom[0], kolom[1])
        hasil.utama = Saran(
            metode="Chi-square" if not sel.dilanggar else "Uji eksak Fisher",
            halaman="Uji Non-parametrik",
            alasan="Kedua variabel berskala kategori tanpa urutan." + kalimat_hubungan,
            syarat=[sel],
            lanjutan="Laporkan Cramér's V sebagai ukuran kekuatan hubungannya.",
            pembanding="SPSS: Analyze ▸ Descriptive Statistics ▸ Crosstabs",
        )
        hasil.alternatif.append(
            Saran(
                metode="Korelasi Pearson",
                halaman="Korelasi & Asumsi",
                alasan="",
                ditolak_karena=(
                    "Pearson menuntut variabel berskala angka; kategori tanpa urutan "
                    "tidak dapat dikorelasikan dengan cara itu."
                ),
            )
        )
        return hasil

    normal = periksa_normalitas(df, kolom[0], None)
    if ada_ordinal or normal.dilanggar or not semua_numerik:
        alasan = (
            "Salah satu variabel berskala ordinal, sehingga yang dibandingkan adalah "
            "urutannya, bukan jaraknya."
            if ada_ordinal
            else normal.rincian
        )
        hasil.utama = Saran(
            metode="Korelasi Spearman",
            halaman="Korelasi & Asumsi",
            alasan=f"Hubungan diukur atas peringkat, bukan nilai mentah. {alasan}{kalimat_hubungan}",
            syarat=[normal],
            pembanding="SPSS: Analyze ▸ Correlate ▸ Bivariate ▸ Spearman",
        )
        hasil.alternatif.append(
            Saran(
                metode="Korelasi Pearson",
                halaman="Korelasi & Asumsi",
                alasan="",
                ditolak_karena=alasan,
            )
        )
        hasil.alternatif.append(
            Saran(
                metode="Korelasi Kendall tau",
                halaman="Korelasi & Asumsi",
                alasan="",
                ditolak_karena=(
                    "Setara Spearman dan lebih tahan pada sampel kecil dengan banyak "
                    "nilai kembar, namun kurang dikenal pembaca."
                ),
            )
        )
        return hasil

    hasil.utama = Saran(
        metode="Korelasi Pearson",
        halaman="Korelasi & Asumsi",
        alasan=f"Kedua variabel berskala angka dan sebarannya normal.{kalimat_hubungan}",
        syarat=[normal],
        lanjutan="Periksa diagram pencar: Pearson hanya menangkap hubungan yang lurus.",
        pembanding="SPSS: Analyze ▸ Correlate ▸ Bivariate ▸ Pearson",
    )
    hasil.alternatif.append(
        Saran(
            metode="Korelasi Spearman",
            halaman="Korelasi & Asumsi",
            alasan="",
            ditolak_karena=(
                "Asumsi Pearson terpenuhi, sehingga Pearson lebih peka. Spearman tetap "
                "berguna sebagai pemeriksaan silang bila ada pencilan."
            ),
        )
    )
    return hasil


# --------------------------------------------------------------------------- #
# Memperkirakan nilai
# --------------------------------------------------------------------------- #


def _periksa_normalitas_residual(regression, df, outcome, prediktor) -> Syarat:
    """Normalitas RESIDUAL model, bukan outcome mentah.

    Regresi mengasumsikan galat model bersebaran normal, bukan variabel terikatnya
    sendiri — outcome yang menceng sekalipun dapat punya residual yang mendekati
    normal begitu prediktornya ikut menjelaskan sebagian kemencengan itu. Memakai
    ulang ``regression.linear_regression`` (yang sudah menghitung Jarque-Bera pada
    residualnya), bukan menulis ulang fit model atau uji normalitas.
    """
    try:
        model = regression.linear_regression(df, outcome, prediktor)
    except Exception:  # noqa: BLE001 - model gagal fit pada data ekstrem
        return Syarat("Normalitas residual", TIDAK_DIUJI, "Model belum dapat diperiksa.")

    baris = model.diagnostics[model.diagnostics["Asumsi"].str.contains("Normalitas residual")]
    if baris.empty:
        return Syarat(
            "Normalitas residual", TIDAK_DIUJI, "Diagnostik residual tidak tersedia untuk model ini."
        )

    p = float(baris.iloc[0]["p-value"])
    if not np.isfinite(p):
        return Syarat("Normalitas residual", TIDAK_DIUJI, "Nilai p Jarque-Bera tidak terhingga.")
    if p < ALFA:
        return Syarat(
            "Normalitas residual", DILANGGAR, f"Jarque-Bera menolak normalitas residual (p = {p:.4f})."
        )
    return Syarat(
        "Normalitas residual", TERPENUHI, f"Jarque-Bera tidak menolak normalitas residual (p = {p:.4f})."
    )


def _memperkirakan_nilai(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    from nalardata import regression

    hasil = Rekomendasi()
    if not outcome:
        hasil.belum_terjawab.append("Angka mana yang ingin Anda jelaskan?")
    if not prediktor:
        hasil.belum_terjawab.append("Variabel mana yang Anda duga menjelaskannya?")
    if hasil.belum_terjawab:
        return hasil

    if not _numerik(kamus, outcome):
        hasil.catatan.append(
            f"'{outcome}' tidak berskala angka menurut Kamus Variabel, sehingga "
            "regresi linear tidak sesuai."
        )
        hasil.alternatif.append(
            Saran(
                metode="Regresi logistik",
                halaman="Regresi",
                alasan="",
                ditolak_karena=(
                    "Bila yang Anda perkirakan sebenarnya kategori, pilih tujuan "
                    "'Memperkirakan kategori'."
                ),
            )
        )
        return hasil

    kolinear = periksa_multikolinearitas(df, prediktor)
    normal = _periksa_normalitas_residual(regression, df, outcome, prediktor)
    syarat = [kolinear, normal]

    try:
        jenis, alasan_galat = regression.saran_galat_baku(df, outcome, prediktor)
    except Exception:  # noqa: BLE001 - saran galat baku gagal pada data ekstrem
        jenis, alasan_galat = "nonrobust", ""

    peringatan = ""
    if jenis != "nonrobust":
        peringatan = f"Pakai galat baku {jenis.upper()}. {alasan_galat}"

    kata_benda = pagar.kata_benda_hubungan(penelitian)
    hasil.utama = Saran(
        metode="Regresi linear berganda",
        halaman="Regresi",
        alasan=(
            f"'{outcome}' berskala angka dan dijelaskan oleh {len(prediktor)} prediktor "
            f"sekaligus. Laporkan koefisien tiap prediktor sebagai besar {kata_benda}nya "
            f"terhadap '{outcome}'."
        ),
        syarat=syarat,
        lanjutan=(
            "Periksa uji asumsi klasik pada panel Regresi: normalitas residual, "
            "heteroskedastisitas, autokorelasi, dan linearitas."
        ),
        peringatan=peringatan,
        pembanding="SPSS: Analyze ▸ Regression ▸ Linear",
    )

    if kolinear.dilanggar:
        hasil.alternatif.append(
            Saran(
                metode="Kurangi prediktor atau gabungkan",
                halaman="Kamus Variabel",
                alasan="",
                ditolak_karena=(
                    f"{kolinear.rincian} Koefisien tiap prediktor menjadi tidak stabil "
                    "dan tandanya dapat berbalik, meskipun kecocokan model keseluruhan "
                    "tetap baik."
                ),
            )
        )

    hasil.alternatif.append(
        Saran(
            metode="Korelasi",
            halaman="Korelasi & Asumsi",
            alasan="",
            ditolak_karena=(
                "Korelasi hanya menghubungkan dua variabel sekaligus dan tidak dapat "
                "mengendalikan variabel lain."
            ),
        )
    )
    return hasil


# --------------------------------------------------------------------------- #
# Memperkirakan kategori
# --------------------------------------------------------------------------- #


def _memperkirakan_kategori(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    hasil = Rekomendasi()
    if not outcome:
        hasil.belum_terjawab.append("Kategori mana yang ingin Anda perkirakan?")
    if not prediktor:
        hasil.belum_terjawab.append("Variabel mana yang Anda duga membedakannya?")
    if hasil.belum_terjawab:
        return hasil

    bersih = _bersih(df, [outcome] + prediktor)
    kelas = int(bersih[outcome].nunique()) if not bersih.empty else 0
    if kelas < 2:
        hasil.catatan.append(f"'{outcome}' hanya memuat {kelas} kategori.")
        return hasil

    jumlah = bersih[outcome].value_counts()
    timpang = float(jumlah.min() / jumlah.sum())
    syarat = [
        Syarat(
            "Keseimbangan kelas",
            DILANGGAR if timpang < 0.10 else TERPENUHI,
            f"Kelas terkecil berisi {timpang:.1%} dari data"
            + (
                ". Model cenderung mengabaikan kelas minoritas dan tetap tampak akurat."
                if timpang < 0.10
                else "."
            ),
        ),
        periksa_multikolinearitas(bersih, prediktor),
    ]

    if kelas == 2:
        hasil.utama = Saran(
            metode="Regresi logistik biner",
            halaman="Regresi",
            alasan=(
                f"'{outcome}' memuat dua kategori, dan regresi logistik memberi odds "
                "ratio yang langsung dapat ditafsirkan."
            ),
            syarat=syarat,
            lanjutan="Laporkan odds ratio beserta selang kepercayaannya, bukan hanya nilai p.",
            pembanding="SPSS: Analyze ▸ Regression ▸ Binary Logistic",
        )
        hasil.alternatif.append(
            Saran(
                metode="Analisis diskriminan",
                halaman="Analisis Diskriminan",
                alasan="",
                ditolak_karena=(
                    "Diskriminan menuntut prediktor bersebaran normal multivariat dan "
                    "matriks ragam yang sama antar kelompok — syarat yang lebih berat "
                    "daripada regresi logistik."
                ),
            )
        )
        return hasil

    hasil.utama = Saran(
        metode="Analisis diskriminan",
        halaman="Analisis Diskriminan",
        alasan=(
            f"'{outcome}' memuat {kelas} kategori. Analisis diskriminan menangani lebih "
            "dari dua kelompok sekaligus dan menunjukkan variabel mana yang paling "
            "membedakannya."
        ),
        syarat=syarat + [Syarat("Normalitas multivariat", TIDAK_DIUJI, "Periksa pada halaman metode.")],
        lanjutan="Periksa Box's M untuk kesamaan matriks ragam antar kelompok.",
        pembanding="SPSS: Analyze ▸ Classify ▸ Discriminant",
    )
    hasil.alternatif.append(
        Saran(
            metode="Regresi logistik multinomial",
            halaman="(belum tersedia)",
            alasan="",
            ditolak_karena=(
                "Logistik multinomial belum tersedia di aplikasi ini. Ia menuntut asumsi "
                "lebih ringan daripada diskriminan, jadi pertimbangkan memeriksanya "
                "silang di SPSS atau R."
            ),
        )
    )
    return hasil


# --------------------------------------------------------------------------- #
# Meringkas, mengelompokkan, model, instrumen
# --------------------------------------------------------------------------- #


def _meringkas(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    hasil = Rekomendasi()
    variabel = [v for v in prediktor if v in df.columns]
    if len(variabel) < 3:
        hasil.belum_terjawab.append(
            "Variabel mana saja yang ingin diringkas? Perlu sekurang-kurangnya tiga."
        )
        return hasil

    syarat = periksa_kelayakan_faktor(df, variabel)
    gagal = [s for s in syarat if s.dilanggar]
    if gagal:
        hasil.catatan.append(
            "Data ini belum layak difaktorkan: " + " ".join(s.rincian for s in gagal)
        )

    hasil.utama = Saran(
        metode="Analisis Faktor Eksploratori (EFA)",
        halaman="Analisis Faktor",
        alasan=(
            f"{len(variabel)} variabel diringkas menjadi beberapa dimensi bersama. EFA "
            "menganggap dimensi itu sebagai konstruk laten yang menjadi sebab kesamaan "
            "jawaban — anggapan yang sesuai untuk butir kuesioner."
        ),
        syarat=syarat,
        lanjutan="Periksa muatan faktor dan lakukan rotasi bila dimensinya lebih dari satu.",
        pembanding="SPSS: Analyze ▸ Dimension Reduction ▸ Factor",
    )
    hasil.alternatif.append(
        Saran(
            metode="Analisis Komponen Utama (PCA)",
            halaman="PCA",
            alasan="",
            ditolak_karena=(
                "PCA meringkas tanpa menganggap ada konstruk laten. Pilih PCA bila "
                "tujuan Anda semata memampatkan data, misalnya untuk dipakai sebagai "
                "masukan analisis lain."
            ),
        )
    )
    return hasil


def _mengelompokkan(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    hasil = Rekomendasi()
    variabel = [v for v in prediktor if _numerik(kamus, v)]
    if len(variabel) < 2:
        hasil.belum_terjawab.append(
            "Variabel angka mana saja yang menjadi dasar pengelompokan? Perlu "
            "sekurang-kurangnya dua."
        )
        return hasil

    n = len(_bersih(df, variabel))
    hasil.utama = Saran(
        metode="Analisis klaster",
        halaman="Analisis Klaster",
        alasan=(
            f"{n} pengamatan dikelompokkan menurut kemiripan pada {len(variabel)} "
            "variabel. Kelompoknya ditemukan dari data, bukan ditetapkan sebelumnya."
        ),
        syarat=[
            Syarat(
                "Skala variabel",
                TERPENUHI,
                "Seluruh variabel berskala angka. Bakukan lebih dulu bila satuannya "
                "berbeda jauh, agar variabel bersatuan besar tidak mendominasi jarak.",
            )
        ],
        lanjutan=(
            "Klaster tidak punya jawaban benar tunggal. Bandingkan beberapa jumlah "
            "klaster, lalu pilih yang paling masuk akal secara teori."
        ),
        pembanding="SPSS: Analyze ▸ Classify ▸ K-Means Cluster",
    )
    hasil.alternatif.append(
        Saran(
            metode="Analisis diskriminan",
            halaman="Analisis Diskriminan",
            alasan="",
            ditolak_karena=(
                "Diskriminan dipakai bila kelompoknya sudah diketahui sejak awal; "
                "klaster dipakai ketika kelompoknya justru yang dicari."
            ),
        )
    )
    return hasil


def _menguji_model(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    from nalardata import sem_analysis

    hasil = Rekomendasi()
    variabel = [v for v in prediktor if v in df.columns]
    if len(variabel) < 3:
        hasil.belum_terjawab.append(
            "Indikator mana saja yang menyusun konstruk Anda? Perlu sekurang-kurangnya tiga."
        )
        return hasil

    n = len(_bersih(df, variabel))
    syarat = [
        Syarat(
            "Ukuran sampel",
            DILANGGAR if n < 200 else TERPENUHI,
            f"{n} pengamatan"
            + (
                ". Model laten lazimnya menuntut sekurang-kurangnya 200."
                if n < 200
                else "."
            ),
        )
    ]
    try:
        estimator, alasan_estimator = sem_analysis.saran_estimator(df, variabel)
    except Exception:  # noqa: BLE001
        estimator, alasan_estimator = "ML", ""

    hasil.utama = Saran(
        metode="CFA / Analisis Jalur / SEM",
        halaman="CFA, Jalur & SEM",
        alasan=(
            "Model hubungan antar konstruk laten diuji sekaligus, bukan sepotong demi "
            "sepotong lewat beberapa regresi terpisah."
        ),
        syarat=syarat,
        lanjutan=(
            "Periksa model pengukuran lebih dulu (muatan, AVE, CR, HTMT), baru model "
            "strukturalnya. Model struktural di atas pengukuran yang buruk tidak berarti."
        ),
        peringatan=f"Estimator yang disarankan: {estimator}. {alasan_estimator}".strip(),
        pembanding="AMOS, Mplus, atau lavaan di R",
    )
    hasil.alternatif.append(
        Saran(
            # Nama ini sengaja menghindari kata "regresi berganda" saja — itu nama
            # metode yang sudah tersedia di halaman Regresi, dan pengguna yang
            # membaca sekilas dapat salah paham aplikasi ini tidak punya regresi
            # berganda sama sekali, padahal yang tidak ada hanya cara bertahapnya.
            metode="Menguji tiap jalur satu-satu lewat regresi terpisah",
            halaman="Regresi",
            alasan="",
            ditolak_karena=(
                "Ini berbeda dari regresi berganda biasa (tersedia di halaman "
                "Regresi): di sini setiap jalur model diuji satu-satu lewat regresi "
                "terpisah, bertahap. Cara itu mengabaikan galat pengukuran dan "
                "tidak menghasilkan indeks kecocokan model secara keseluruhan."
            ),
        )
    )
    return hasil


# --------------------------------------------------------------------------- #
# Mediasi dan moderasi
# --------------------------------------------------------------------------- #


def _mediasi(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    """Mediasi: X mempengaruhi Y lewat perantara M. ``prediktor[0]`` = X,
    ``prediktor[1]`` = M — urutan yang diminta di widget, bukan aturan baru."""
    hasil = Rekomendasi()
    if not outcome:
        hasil.belum_terjawab.append("Variabel mana yang menjadi hasil akhir (Y)?")
    if len(prediktor) < 2:
        hasil.belum_terjawab.append(
            "Variabel bebas (X) dan variabel perantara (mediator, M) yang mana? "
            "Pilih X terlebih dulu, lalu M."
        )
    if hasil.belum_terjawab:
        return hasil

    x, m = prediktor[0], prediktor[1]
    n = len(_bersih(df, [outcome, x, m]))
    syarat = [
        Syarat(
            "Ukuran sampel",
            DILANGGAR if n < 100 else TERPENUHI,
            f"{n} pengamatan"
            + (
                ". Bootstrap efek tidak langsung lebih stabil pada sekurang-kurangnya 100."
                if n < 100
                else "."
            ),
        )
    ]

    hasil.utama = Saran(
        metode="CFA / Analisis Jalur / SEM",
        halaman="CFA, Jalur & SEM",
        alasan=(
            f"Diuji apakah pengaruh '{x}' terhadap '{outcome}' mengalir lewat '{m}' "
            "sebagai perantara. Efek tidak langsung (X → M → Y) diuji dengan bootstrap, "
            "bukan uji Sobel, karena sebaran hasil kali dua koefisien jarang normal."
        ),
        syarat=syarat,
        lanjutan=(
            "Bangun jalur X → M dan M → Y (plus X → Y untuk efek langsung) pada panel "
            "CFA, Jalur & SEM, lalu baca efek langsung, tidak langsung, dan total "
            "beserta interval kepercayaan bootstrap-nya."
        ),
        ukuran_efek="Proporsi efek yang dimediasi (efek tidak langsung ÷ efek total).",
        pembanding="PROCESS macro (Hayes) Model 4 pada SPSS, atau paket lavaan di R",
    )
    hasil.alternatif.append(
        Saran(
            metode="Uji Sobel",
            halaman="CFA, Jalur & SEM",
            alasan="",
            ditolak_karena=(
                "Uji Sobel mengasumsikan hasil kali dua koefisien (a×b) berdistribusi "
                "normal — asumsi yang lazim dilanggar pada sampel sedang atau kecil. "
                "Bootstrap tidak menuntut asumsi itu."
            ),
        )
    )
    return hasil


def _moderasi(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    """Moderasi: kekuatan pengaruh X terhadap Y berubah menurut M. ``prediktor[0]``
    = X, ``prediktor[1]`` = moderator M."""
    hasil = Rekomendasi()
    if not outcome:
        hasil.belum_terjawab.append("Variabel mana yang menjadi hasil (Y)?")
    if len(prediktor) < 2:
        hasil.belum_terjawab.append(
            "Variabel bebas (X) dan variabel moderator yang mana? Pilih X terlebih "
            "dulu, lalu moderatornya."
        )
    if hasil.belum_terjawab:
        return hasil

    x, m = prediktor[0], prediktor[1]
    if not all(_numerik(kamus, v) for v in (outcome, x, m)):
        hasil.catatan.append(
            "Regresi moderasi pada aplikasi ini menuntut Y, X, dan moderator "
            "sama-sama berskala angka."
        )

    kolinear = periksa_multikolinearitas(df, [x, m])
    syarat = [kolinear]

    hasil.utama = Saran(
        metode="Regresi Moderasi (MRA)",
        halaman="Regresi Moderasi (MRA)",
        alasan=(
            f"Diuji apakah besar pengaruh '{x}' terhadap '{outcome}' berubah pada "
            f"tingkat '{m}' yang berbeda, lewat suku interaksi X × moderator — bukan "
            "dengan membandingkan koefisien pada beberapa regresi terpisah."
        ),
        syarat=syarat,
        lanjutan=(
            "Pusatkan (mean-center) X dan moderator sebelum membentuk suku interaksi, "
            "lalu baca kemiringan sederhana pada nilai moderator rendah, rata-rata, "
            "dan tinggi, serta titik Johnson-Neyman bila tersedia."
        ),
        ukuran_efek="Perubahan R² akibat suku interaksi (ΔR²).",
        pembanding="PROCESS macro (Hayes) Model 1 pada SPSS",
    )
    hasil.alternatif.append(
        Saran(
            metode="Regresi linear berganda tanpa suku interaksi",
            halaman="Regresi",
            alasan="",
            ditolak_karena=(
                "Tanpa suku interaksi, model ini mengasumsikan pengaruh X terhadap Y "
                "sama besarnya pada semua tingkat moderator — anggapan yang justru "
                "ingin diuji, bukan ditetapkan sejak awal."
            ),
        )
    )
    return hasil


def _mutu_instrumen(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    hasil = Rekomendasi()
    butir = [v for v in prediktor if v in df.columns]
    if len(butir) < 3:
        hasil.belum_terjawab.append(
            "Butir mana saja yang menyusun satu konstruk? Perlu sekurang-kurangnya tiga."
        )
        return hasil

    n = len(_bersih(df, butir))
    hasil.utama = Saran(
        metode="Uji validitas dan reliabilitas",
        halaman="Reliabilitas & Validitas",
        alasan=(
            f"{len(butir)} butir diperiksa apakah benar-benar mengukur satu hal yang "
            "sama, sebelum skornya dipakai pada analisis lain."
        ),
        syarat=[
            Syarat("Jumlah butir", TERPENUHI, f"{len(butir)} butir."),
            Syarat("Ukuran sampel", TERPENUHI if n >= 30 else DILANGGAR, f"{n} responden."),
        ],
        lanjutan=(
            "Laporkan Cronbach alpha bersama McDonald omega. Alpha mengandaikan seluruh "
            "butir berbobot sama; omega tidak. Namun keunggulan omega tidak berlaku "
            "di segala keadaan: pada skala berbutir sedikit dengan muatan rendah, "
            "omega justru dapat kurang tepat daripada alpha, dan pada skala lima butir "
            "atau lebih keduanya lazimnya berselisih tipis."
        ),
        pembanding="SPSS: Analyze ▸ Scale ▸ Reliability Analysis",
    )
    hasil.alternatif.append(
        Saran(
            metode="Analisis Faktor Konfirmatori (CFA)",
            halaman="CFA, Jalur & SEM",
            alasan="",
            ditolak_karena=(
                "CFA menguji struktur yang sudah Anda tetapkan dan memberi AVE serta "
                "CR. Pakai sesudah alpha, bila strukturnya sudah jelas dari teori."
            ),
        )
    )
    return hasil


# --------------------------------------------------------------------------- #
# Data panel dan deret waktu
# --------------------------------------------------------------------------- #


def _panel(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    """Data panel: ``kelompok`` menjadi entitas (unit yang diamati berulang),
    bukan penanda kelompok pembanding seperti pada tujuan 'membandingkan'."""
    hasil = Rekomendasi()
    if not kelompok:
        hasil.belum_terjawab.append(
            "Kolom mana yang menandai entitas (unit) yang diamati berulang, "
            "misalnya kode perusahaan atau nama individu?"
        )
    if not outcome:
        hasil.belum_terjawab.append("Angka mana yang menjadi hasil (Y)?")
    if not prediktor:
        hasil.belum_terjawab.append("Variabel mana yang menjadi prediktor (X)?")
    if hasil.belum_terjawab:
        return hasil

    bersih = _bersih(df, [outcome, kelompok, *prediktor])
    n_entitas = int(bersih[kelompok].nunique()) if not bersih.empty else 0
    if n_entitas < 2:
        hasil.catatan.append(
            f"Kolom '{kelompok}' hanya memuat {n_entitas} entitas, sehingga bukan data panel."
        )
        return hasil

    ulangan = bersih.groupby(kelompok).size()
    terkecil = int(ulangan.min()) if not ulangan.empty else 0
    ulang_syarat = Syarat(
        "Pengamatan berulang per entitas",
        TERPENUHI if terkecil >= 2 else DILANGGAR,
        f"Entitas dengan pengamatan tersedikit punya {terkecil} baris."
        + ("" if terkecil >= 2 else " Data panel menuntut tiap entitas diamati lebih dari sekali."),
    )
    syarat = [ulang_syarat]

    hasil.utama = Saran(
        metode="Regresi Panel",
        halaman="Regresi Panel",
        alasan=(
            f"'{kelompok}' diamati berulang sepanjang waktu, sehingga pengamatan pada "
            "entitas yang sama cenderung lebih mirip satu sama lain — melanggar "
            "independensi yang dituntut OLS biasa. Regresi panel (pengaruh tetap "
            "atau pengaruh acak, dipilih lewat uji Hausman) mengendalikan hal itu."
        ),
        syarat=syarat,
        lanjutan=(
            "Uji Hausman pada halaman Regresi Panel memutuskan antara pengaruh tetap "
            "dan pengaruh acak berdasarkan data Anda sendiri, bukan ditetapkan di sini."
        ),
        pembanding="Stata: xtreg, fe / xtreg, re, xtoverid — atau paket plm/lme4 di R",
    )
    hasil.alternatif.append(
        Saran(
            metode="Regresi linear berganda",
            halaman="Regresi",
            alasan="",
            ditolak_karena=(
                "Regresi biasa mengabaikan bahwa pengamatan dari entitas yang sama "
                "saling berkorelasi, sehingga galat baku dan p-value-nya dapat "
                "menyesatkan — tampak lebih meyakinkan daripada seharusnya."
            ),
        )
    )
    return hasil


def _arima(df, kamus, outcome, prediktor, kelompok, berpasangan, penelitian) -> Rekomendasi:
    hasil = Rekomendasi()
    if not outcome:
        hasil.belum_terjawab.append("Angka mana yang ingin diramalkan?")
        return hasil

    nilai = pd.to_numeric(df[outcome], errors="coerce").dropna() if outcome in df.columns else pd.Series(dtype=float)
    n = len(nilai)
    panjang = Syarat(
        "Panjang deret",
        TERPENUHI if n >= 15 else DILANGGAR,
        f"{n} titik waktu yang valid." + ("" if n >= 15 else " ARIMA menuntut sekurang-kurangnya 15."),
    )

    hasil.utama = Saran(
        metode="ARIMA",
        halaman="Deret Waktu (ARIMA)",
        alasan=(
            f"'{outcome}' diramalkan dari polanya sendiri di masa lalu, bukan dari "
            "variabel lain. Deret diuji stasioner (Augmented Dickey-Fuller) dan "
            "di-differencing otomatis bila belum, sebelum order ARIMA dicari lewat AIC."
        ),
        syarat=[panjang],
        lanjutan=(
            "Periksa Ljung-Box pada residual setelah model terpasang: p ≥ 0,05 "
            "berarti pola pada deret sudah cukup tertangkap."
        ),
        pembanding="EViews, Minitab, atau fungsi auto.arima() pada paket forecast di R",
    )
    hasil.alternatif.append(
        Saran(
            metode="Regresi linear berganda",
            halaman="Regresi",
            alasan="",
            ditolak_karena=(
                "Regresi biasa menuntut variabel penjelas terpisah dan mengasumsikan "
                "pengamatan saling bebas — asumsi yang dilanggar oleh deret waktu, "
                "yang nilainya justru berkorelasi dengan dirinya sendiri di masa lalu."
            ),
        )
    )
    return hasil
