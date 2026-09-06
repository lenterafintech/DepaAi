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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from nalardata import kamus as km

# --------------------------------------------------------------------------- #
# Kosakata
# --------------------------------------------------------------------------- #

TUJUAN = {
    "membandingkan": "Membandingkan kelompok",
    "menghubungkan": "Menguji hubungan antar variabel",
    "memperkirakan_nilai": "Menjelaskan atau memperkirakan nilai angka",
    "memperkirakan_kategori": "Memperkirakan kategori atau keputusan",
    "meringkas": "Meringkas banyak variabel menjadi sedikit dimensi",
    "mengelompokkan": "Mengelompokkan responden yang mirip",
    "menguji_model": "Menguji model teoretis antar konstruk",
    "mutu_instrumen": "Memeriksa mutu kuesioner",
}

PERTANYAAN_TUJUAN = {
    "membandingkan": "Apakah kelompok A berbeda dari kelompok B?",
    "menghubungkan": "Apakah dua variabel bergerak bersamaan?",
    "memperkirakan_nilai": "Faktor apa yang menjelaskan naik-turunnya sebuah angka?",
    "memperkirakan_kategori": "Faktor apa yang membedakan yang 'ya' dari yang 'tidak'?",
    "meringkas": "Belasan butir kuesioner ini sebenarnya mengukur berapa hal?",
    "mengelompokkan": "Ada berapa tipe responden dalam data ini?",
    "menguji_model": "Apakah model hubungan antar konstruk saya didukung data?",
    "mutu_instrumen": "Apakah kuesioner saya valid dan reliabel?",
}

TERPENUHI = "terpenuhi"
DILANGGAR = "dilanggar"
TIDAK_DIUJI = "tidak diuji"

ALFA = 0.05
MIN_SEL = 5  # frekuensi harapan minimum pada tabel silang
MIN_KELOMPOK = 3  # anggota minimum agar sebuah kelompok masih dapat diuji
MIN_SHAPIRO = 3


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
    "Chi-square": "Uji Beda",
    "Uji eksak Fisher": "Uji Beda",
    "Korelasi Pearson": "Korelasi & Asumsi",
    "Korelasi Spearman": "Korelasi & Asumsi",
    "Korelasi Kendall tau": "Korelasi & Asumsi",
    "Regresi linear berganda": "Regresi",
    "Regresi logistik biner": "Regresi",
    "Analisis diskriminan": "Analisis Diskriminan",
    "Analisis Faktor Eksploratori (EFA)": "Analisis Faktor",
    "Analisis Komponen Utama (PCA)": "PCA",
    "Analisis klaster": "Analisis Klaster",
    "CFA / Analisis Jalur / SEM": "CFA, Jalur & SEM",
    "Uji validitas dan reliabilitas": "Reliabilitas & Validitas",
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

    def __post_init__(self) -> None:
        # Halaman diambil dari daftar metode, bukan dituliskan ulang di tiap
        # cabang: nama halaman berubah ketika navigasi ditata ulang, dan salinan
        # yang tersebar akan menunjuk tempat yang sudah tidak ada.
        if self.metode in METODE_TERSEDIA:
            self.halaman = METODE_TERSEDIA[self.metode]

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
    if terkecil < 30:
        return Syarat(
            "Ukuran kelompok",
            TERPENUHI,
            f"Kelompok terkecil berisi {terkecil} pengamatan — cukup untuk diuji, "
            "namun uji non-parametrik lebih aman di bawah 30.",
        )
    return Syarat(
        "Ukuran kelompok", TERPENUHI, f"Kelompok terkecil berisi {terkecil} pengamatan."
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
) -> Rekomendasi:
    """Sarankan metode dengan memeriksa data yang sungguh ada.

    ``berpasangan`` adalah satu-satunya hal yang ditanyakan dan tidak dihitung:
    tidak ada cara membaca dari angka apakah dua kolom berasal dari orang yang sama
    diukur dua kali atau dari dua orang berbeda.
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
        "menghubungkan": _menghubungkan,
        "memperkirakan_nilai": _memperkirakan_nilai,
        "memperkirakan_kategori": _memperkirakan_kategori,
        "meringkas": _meringkas,
        "mengelompokkan": _mengelompokkan,
        "menguji_model": _menguji_model,
        "mutu_instrumen": _mutu_instrumen,
    }
    hasil = penanganan[tujuan](df, kamus, outcome, prediktor, kelompok, berpasangan)

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


def _membandingkan(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
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
    syarat = [normal, seragam, ukuran]

    ordinal = _ordinal(kamus, outcome)
    perlu_nonpar = ordinal or normal.dilanggar or ukuran.dilanggar

    if k == 2:
        return _dua_kelompok(hasil, outcome, kelompok, syarat, ordinal, normal, seragam, perlu_nonpar)
    return _banyak_kelompok(
        hasil, outcome, kelompok, k, syarat, ordinal, normal, seragam, perlu_nonpar
    )


def _dua_kelompok(hasil, outcome, kelompok, syarat, ordinal, normal, seragam, perlu_nonpar):
    if perlu_nonpar:
        alasan = (
            f"'{outcome}' berskala ordinal, sehingga jarak antar tingkatnya belum tentu "
            "sama dan rata-rata kurang bermakna."
            if ordinal
            else normal.rincian
        )
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

    if seragam.dilanggar:
        hasil.utama = Saran(
            metode="Uji-t Welch",
            halaman="Uji Non-parametrik",
            alasan=(
                "Dua kelompok bebas dengan sebaran normal, namun ragamnya tidak "
                f"seragam. {seragam.rincian} Welch tidak menuntut ragam yang sama."
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
            f"Dua kelompok bebas, sebaran '{outcome}' normal pada tiap kelompok, dan "
            "ragamnya seragam."
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


def _banyak_kelompok(hasil, outcome, kelompok, k, syarat, ordinal, normal, seragam, perlu_nonpar):
    if perlu_nonpar:
        alasan = (
            f"'{outcome}' berskala ordinal." if ordinal else normal.rincian
        )
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

    if seragam.dilanggar:
        hasil.utama = Saran(
            metode="Welch ANOVA",
            halaman="Uji Non-parametrik",
            alasan=(
                f"{k} kelompok bebas dengan sebaran normal, namun ragamnya tidak "
                f"seragam. {seragam.rincian} Welch ANOVA tidak menuntut ragam yang sama."
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
            f"{k} kelompok bebas, sebaran '{outcome}' normal pada tiap kelompok, dan "
            "ragamnya seragam."
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


def _membandingkan_berpasangan(df, kamus, outcome, prediktor) -> Rekomendasi:
    """Pengukuran berulang pada unit yang sama, tersimpan sebagai beberapa kolom."""
    hasil = Rekomendasi()
    kolom = [k for k in ([outcome] if outcome else []) + list(prediktor) if k]
    kolom = list(dict.fromkeys(kolom))

    if len(kolom) < 2:
        hasil.belum_terjawab.append(
            "Kolom mana saja yang memuat pengukuran berulang pada unit yang sama?"
        )
        return hasil

    bersih = _bersih(df, kolom)
    normal = periksa_normalitas(bersih, kolom[0], None)
    ordinal = any(_ordinal(kamus, k) for k in kolom)
    perlu_nonpar = ordinal or normal.dilanggar

    if len(kolom) == 2:
        if perlu_nonpar:
            hasil.utama = Saran(
                metode="Wilcoxon signed-rank",
                halaman="Uji Non-parametrik",
                alasan=(
                    "Dua pengukuran pada unit yang sama dibandingkan tanpa asumsi "
                    f"normalitas. {'Skalanya ordinal.' if ordinal else normal.rincian}"
                ),
                syarat=[normal],
                pembanding="SPSS: Analyze ▸ Nonparametric Tests ▸ Related Samples",
            )
            hasil.alternatif.append(
                Saran(
                    metode="Uji-t berpasangan",
                    halaman="Uji Non-parametrik",
                    alasan="",
                    ditolak_karena="Skalanya ordinal." if ordinal else normal.rincian,
                )
            )
            return hasil

        hasil.utama = Saran(
            metode="Uji-t berpasangan",
            halaman="Uji Non-parametrik",
            alasan="Dua pengukuran pada unit yang sama, selisihnya bersebaran normal.",
            syarat=[normal],
            lanjutan="Laporkan Cohen's d untuk sampel berpasangan.",
            pembanding="SPSS: Analyze ▸ Compare Means ▸ Paired-Samples T Test",
        )
        hasil.alternatif.append(
            Saran(
                metode="Wilcoxon signed-rank",
                halaman="Uji Non-parametrik",
                alasan="",
                ditolak_karena="Asumsi normalitas terpenuhi, sehingga uji-t lebih peka.",
            )
        )
        return hasil

    hasil.utama = Saran(
        metode="Friedman",
        halaman="Uji Non-parametrik",
        alasan=(
            f"{len(kolom)} pengukuran berulang pada unit yang sama. Friedman tidak "
            "menuntut normalitas dan menangani lebih dari dua pengukuran sekaligus."
        ),
        syarat=[normal],
        lanjutan="Laporkan Kendall's W sebagai ukuran kesepakatan antar pengukuran.",
        pembanding="SPSS: Analyze ▸ Nonparametric Tests ▸ Related Samples ▸ Friedman",
    )
    hasil.alternatif.append(
        Saran(
            metode="ANOVA ukur ulang",
            halaman="(belum tersedia)",
            alasan="",
            ditolak_karena=(
                "ANOVA ukur ulang belum tersedia di aplikasi ini. Friedman menjawab "
                "pertanyaan yang sama tanpa asumsi normalitas dan kesamaan ragam."
            ),
        )
    )
    return hasil


# --------------------------------------------------------------------------- #
# Menguji hubungan
# --------------------------------------------------------------------------- #


def _menghubungkan(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
    hasil = Rekomendasi()
    kolom = [k for k in ([outcome] if outcome else []) + list(prediktor) if k]
    kolom = list(dict.fromkeys(kolom))

    if len(kolom) < 2:
        hasil.belum_terjawab.append("Dua variabel mana yang ingin Anda hubungkan?")
        return hasil

    semua_numerik = all(_numerik(kamus, k) for k in kolom)
    ada_ordinal = any(_ordinal(kamus, k) for k in kolom)
    semua_kategorik = all(k in kamus and kamus[k].kategorik for k in kolom)

    if semua_kategorik and not ada_ordinal:
        sel = periksa_frekuensi_harapan(df, kolom[0], kolom[1])
        hasil.utama = Saran(
            metode="Chi-square" if not sel.dilanggar else "Uji eksak Fisher",
            halaman="Uji Non-parametrik",
            alasan="Kedua variabel berskala kategori tanpa urutan.",
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
            alasan=f"Hubungan diukur atas peringkat, bukan nilai mentah. {alasan}",
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
        alasan="Kedua variabel berskala angka dan sebarannya normal.",
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


def _memperkirakan_nilai(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
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
    normal = periksa_normalitas(df, outcome, None)
    syarat = [kolinear, normal]

    try:
        jenis, alasan_galat = regression.saran_galat_baku(df, outcome, prediktor)
    except Exception:  # noqa: BLE001 - saran galat baku gagal pada data ekstrem
        jenis, alasan_galat = "nonrobust", ""

    peringatan = ""
    if jenis != "nonrobust":
        peringatan = f"Pakai galat baku {jenis.upper()}. {alasan_galat}"

    hasil.utama = Saran(
        metode="Regresi linear berganda",
        halaman="Regresi",
        alasan=(
            f"'{outcome}' berskala angka dan dijelaskan oleh {len(prediktor)} prediktor "
            "sekaligus."
        ),
        syarat=syarat,
        lanjutan=(
            "Periksa uji asumsi klasik pada halaman Regresi: normalitas residual, "
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


def _memperkirakan_kategori(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
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


def _meringkas(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
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


def _mengelompokkan(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
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


def _menguji_model(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
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
            metode="Regresi berganda berurutan",
            halaman="Regresi",
            alasan="",
            ditolak_karena=(
                "Menguji model bertahap lewat beberapa regresi terpisah mengabaikan "
                "galat pengukuran dan tidak menghasilkan indeks kecocokan model."
            ),
        )
    )
    return hasil


def _mutu_instrumen(df, kamus, outcome, prediktor, kelompok, berpasangan) -> Rekomendasi:
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
