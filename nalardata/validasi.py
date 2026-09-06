"""Validasi lintas software: membuktikan angkanya, bukan mengklaimnya.

Pengguna Indonesia tidak akan memercayai perangkat baru sebelum angkanya cocok
dengan SPSS atau R. Klaim "hasil kami akurat" tidak menyelesaikan apa pun — yang
menyelesaikannya adalah menunjukkan perbandingannya, termasuk ketika berbeda.

Acuan di sini **beku dan bersumber**: nilainya berasal dari keluaran R yang
terdokumentasi luas pada dataset klasik yang ikut disertakan (mtcars, PlantGrowth,
sleep). Menjalankan R secara langsung akan lebih baik, tetapi menuntut R terpasang
di setiap tempat aplikasi ini dijalankan — dan acuan beku yang jujur lebih berguna
daripada acuan hidup yang tidak pernah tersedia.

Yang dibandingkan adalah **fungsi aplikasi ini sendiri**, bukan pustaka di baliknya.
Membandingkan scipy dengan R hanya menguji scipy; yang perlu diuji adalah lapisan
yang ditulis di sini, karena di lapisan itulah kekeliruan penulisan kode terjadi.

Metode yang belum punya acuan disebut apa adanya sebagai belum divalidasi. Halaman
kesesuaian menampilkannya terbuka: daftar yang menyembunyikan lubangnya sendiri
tidak dapat dipercaya untuk hal lain mana pun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ACUAN = Path(__file__).resolve().parents[1] / "validasi" / "acuan"

SESUAI = "Sesuai"
TOLERANSI = "Sesuai dalam toleransi"
BERBEDA = "Berbeda"
GAGAL = "Tidak dapat dijalankan"
BELUM = "Belum divalidasi"

# Beda relatif yang masih dianggap sama. Angka yang dilaporkan R lazimnya dibulatkan
# ke empat sampai lima digit berarti, sehingga menuntut kesamaan mutlak akan
# melaporkan selisih pembulatan sebagai perbedaan.
TOLERANSI_KETAT = 1e-6
TOLERANSI_LONGGAR = 1e-3


@dataclass
class Butir:
    """Satu besaran yang dibandingkan beserta nilai acuannya."""

    nama: str
    harapan: float
    toleransi: float = TOLERANSI_KETAT


@dataclass
class Acuan:
    """Satu pembandingan: metode aplikasi terhadap keluaran perangkat lain."""

    metode: str
    fitur: str
    pembanding: str
    sumber: str
    data: str
    butir: list[Butir] = field(default_factory=list)


def _muat(nama: str) -> pd.DataFrame:
    berkas = ACUAN / nama
    if not berkas.exists():
        raise FileNotFoundError(f"Berkas acuan '{nama}' tidak ditemukan.")
    return pd.read_csv(berkas)


# --------------------------------------------------------------------------- #
# Perhitungan yang diuji — memakai fungsi aplikasi, bukan pustaka di baliknya
# --------------------------------------------------------------------------- #


def _regresi_mtcars() -> dict[str, float]:
    from nalardata import regression

    hasil = regression.linear_regression(_muat("mtcars.csv"), "mpg", ["wt", "hp"])
    koef = hasil.coefficients.set_index("Variabel")["B"]
    return {
        "Intersep": float(koef["const"]),
        "B wt": float(koef["wt"]),
        "B hp": float(koef["hp"]),
        "R²": float(hasil.model.rsquared),
        "F": float(hasil.model.fvalue),
    }


def _korelasi_mtcars() -> dict[str, float]:
    from nalardata import correlation

    hasil = correlation.correlation_matrix(_muat("mtcars.csv")[["mpg", "wt"]], "pearson")
    return {"r (mpg, wt)": float(hasil.matrix.loc["mpg", "wt"])}


def _anova_plantgrowth() -> dict[str, float]:
    from nalardata import parametrik as par

    data = _muat("plantgrowth.csv")
    hasil = par.anova_satu_arah(data["weight"], data["group"])
    return {"F": float(hasil.statistik), "p": float(hasil.p_value)}


def _welch_anova_plantgrowth() -> dict[str, float]:
    from nalardata import parametrik as par

    data = _muat("plantgrowth.csv")
    hasil = par.welch_anova(data["weight"], data["group"])
    return {"F": float(hasil.statistik), "p": float(hasil.p_value)}


def _kruskal_plantgrowth() -> dict[str, float]:
    from nalardata import nonparametrik as npar

    data = _muat("plantgrowth.csv")
    hasil = npar.kruskal_wallis(data["weight"], data["group"])
    return {"H": float(hasil.statistik), "p": float(hasil.p_value)}


def _welch_sleep() -> dict[str, float]:
    from nalardata import parametrik as par

    data = _muat("sleep.csv")
    hasil = par.uji_t_bebas(data["extra"], data["group"].astype(str), ragam_sama=False)
    return {"t": float(hasil.statistik), "p": float(hasil.p_value)}


def _uji_t_sleep() -> dict[str, float]:
    from nalardata import parametrik as par

    data = _muat("sleep.csv")
    hasil = par.uji_t_bebas(data["extra"], data["group"].astype(str), ragam_sama=True)
    return {"t": float(hasil.statistik), "p": float(hasil.p_value)}


def _tukey_plantgrowth() -> dict[str, float]:
    from nalardata import parametrik as par

    data = _muat("plantgrowth.csv")
    tabel = par.tukey(data["weight"], data["group"])
    baris = tabel[(tabel["Kelompok 1"] == "trt1") & (tabel["Kelompok 2"] == "trt2")]
    return {
        "Selisih trt1-trt2": float(baris["Selisih rata-rata"].iloc[0]),
        "p disesuaikan": float(baris["p disesuaikan"].iloc[0]),
    }


def _normalitas_mtcars() -> dict[str, float]:
    from nalardata import descriptive

    tabel = descriptive.normality_tests(_muat("mtcars.csv")[["mpg"]])
    baris = tabel.iloc[0]
    return {"Shapiro-W": float(baris["Shapiro-W"]), "p": float(baris["p (Shapiro)"])}


PERHITUNGAN = {
    "regresi_mtcars": _regresi_mtcars,
    "welch_anova_plantgrowth": _welch_anova_plantgrowth,
    "uji_t_sleep": _uji_t_sleep,
    "tukey_plantgrowth": _tukey_plantgrowth,
    "korelasi_mtcars": _korelasi_mtcars,
    "anova_plantgrowth": _anova_plantgrowth,
    "kruskal_plantgrowth": _kruskal_plantgrowth,
    "welch_sleep": _welch_sleep,
    "normalitas_mtcars": _normalitas_mtcars,
}


# --------------------------------------------------------------------------- #
# Acuan
# --------------------------------------------------------------------------- #

DAFTAR: dict[str, Acuan] = {
    "regresi_mtcars": Acuan(
        metode="Regresi linear berganda",
        fitur="regresi",
        pembanding="R — lm(mpg ~ wt + hp, data = mtcars)",
        sumber="Keluaran R 4.x pada dataset bawaan mtcars",
        data="mtcars.csv",
        butir=[
            Butir("Intersep", 37.22727, TOLERANSI_LONGGAR),
            Butir("B wt", -3.87783, TOLERANSI_LONGGAR),
            Butir("B hp", -0.03177, TOLERANSI_LONGGAR),
            Butir("R²", 0.8268, TOLERANSI_LONGGAR),
            Butir("F", 69.21, TOLERANSI_LONGGAR),
        ],
    ),
    "korelasi_mtcars": Acuan(
        metode="Korelasi Pearson",
        fitur="dasar",
        pembanding="R — cor(mtcars$mpg, mtcars$wt)",
        sumber="Keluaran R 4.x pada dataset bawaan mtcars",
        data="mtcars.csv",
        butir=[Butir("r (mpg, wt)", -0.8676594, TOLERANSI_LONGGAR)],
    ),
    "anova_plantgrowth": Acuan(
        metode="ANOVA satu arah",
        fitur="nonparametrik",
        pembanding="R — summary(aov(weight ~ group, data = PlantGrowth))",
        sumber="Keluaran R 4.x pada dataset bawaan PlantGrowth",
        data="plantgrowth.csv",
        butir=[Butir("F", 4.846, TOLERANSI_LONGGAR), Butir("p", 0.01591, TOLERANSI_LONGGAR)],
    ),
    "kruskal_plantgrowth": Acuan(
        metode="Kruskal-Wallis",
        fitur="nonparametrik",
        pembanding="R — kruskal.test(weight ~ group, data = PlantGrowth)",
        sumber="Keluaran R 4.x pada dataset bawaan PlantGrowth",
        data="plantgrowth.csv",
        butir=[Butir("H", 7.9882, TOLERANSI_LONGGAR), Butir("p", 0.01842, TOLERANSI_LONGGAR)],
    ),
    "uji_t_sleep": Acuan(
        metode="Uji-t sampel bebas",
        fitur="nonparametrik",
        pembanding="R — t.test(extra ~ group, data = sleep, var.equal = TRUE)",
        sumber="Keluaran R 4.x pada dataset bawaan sleep",
        data="sleep.csv",
        butir=[Butir("t", -1.8608, TOLERANSI_LONGGAR), Butir("p", 0.07919, TOLERANSI_LONGGAR)],
    ),
    "welch_anova_plantgrowth": Acuan(
        metode="Welch ANOVA",
        fitur="nonparametrik",
        pembanding="statsmodels — anova_oneway(use_var='unequal')",
        sumber=(
            "Implementasi Python kedua, bukan R. Acuan yang lebih lemah daripada "
            "acuan lain di sini, dan disebutkan apa adanya."
        ),
        data="plantgrowth.csv",
        butir=[Butir("F", 5.1810, TOLERANSI_LONGGAR), Butir("p", 0.01739, TOLERANSI_LONGGAR)],
    ),
    "tukey_plantgrowth": Acuan(
        metode="Uji lanjutan Tukey HSD",
        fitur="nonparametrik",
        pembanding="R — TukeyHSD(aov(weight ~ group, data = PlantGrowth))",
        sumber="Keluaran R 4.x pada dataset bawaan PlantGrowth",
        data="plantgrowth.csv",
        butir=[
            Butir("Selisih trt1-trt2", 0.865, TOLERANSI_LONGGAR),
            Butir("p disesuaikan", 0.012, 0.05),
        ],
    ),
    "welch_sleep": Acuan(
        metode="Uji-t Welch",
        fitur="nonparametrik",
        pembanding="R — t.test(extra ~ group, data = sleep)",
        sumber="Keluaran R 4.x pada dataset bawaan sleep",
        data="sleep.csv",
        butir=[Butir("t", -1.8608, TOLERANSI_LONGGAR), Butir("p", 0.07939, TOLERANSI_LONGGAR)],
    ),
    "normalitas_mtcars": Acuan(
        metode="Uji normalitas Shapiro-Wilk",
        fitur="dasar",
        pembanding="R — shapiro.test(mtcars$mpg)",
        sumber="Keluaran R 4.x pada dataset bawaan mtcars",
        data="mtcars.csv",
        butir=[Butir("Shapiro-W", 0.94756, TOLERANSI_LONGGAR), Butir("p", 0.1229, TOLERANSI_LONGGAR)],
    ),
}

# Metode aplikasi yang belum punya acuan. Disebutkan agar cakupan validasi
# terbaca apa adanya, bukan tersirat dari daftar yang tampak lengkap.
BELUM_DIVALIDASI = {
    "Regresi logistik": (
        "Perlu acuan glm binomial di R beserta dataset yang boleh ikut disertakan."
    ),
    "Games-Howell": (
        "Perlu acuan paket rstatix di R; belum ada nilai terbit pada dataset bawaan."
    ),
    "MANOVA": "Perlu acuan SPSS; keluaran R memakai parameterisasi berbeda.",
    "Analisis diskriminan": (
        "Perlu acuan SPSS; keluaran MASS::lda di R tidak menyertakan uji Wilks."
    ),
    "Analisis faktor (EFA)": "Hasil bergantung metode rotasi; acuan harus menyebut rotasinya.",
    "PCA": (
        "Perlu acuan prcomp di R; tanda komponen dapat berbeda dan harus dibakukan."
    ),
    "Analisis klaster": "K-Means bergantung benih acak; acuan harus mengunci benihnya.",
    "CFA / SEM": (
        "Perlu acuan lavaan atau Mplus beserta datanya, dan estimatornya harus disebut."
    ),
    "Korelasi kanonik": (
        "Perlu acuan SPSS; tanda muatan kanonik dapat berbeda antar implementasi."
    ),
    "Reliabilitas (alpha, omega)": (
        "Perlu acuan psych::alpha dan psych::omega di R beserta datanya."
    ),
    "Regresi moderasi (MRA)": (
        "Perlu acuan PROCESS macro Hayes, yang keluarannya tidak diterbitkan bebas."
    ),
}


# --------------------------------------------------------------------------- #
# Perbandingan
# --------------------------------------------------------------------------- #


def _status(diperoleh: float, harapan: float, toleransi: float) -> str:
    if not np.isfinite(diperoleh) or not np.isfinite(harapan):
        return BERBEDA
    beda = abs(diperoleh - harapan)
    skala = max(abs(harapan), 1e-12)
    if beda <= abs(harapan) * TOLERANSI_KETAT or beda < 1e-12:
        return SESUAI
    return TOLERANSI if beda / skala <= toleransi else BERBEDA


def jalankan(kode: str | None = None) -> pd.DataFrame:
    """Bandingkan hasil aplikasi dengan acuan; satu baris per besaran."""
    if kode is not None and kode not in DAFTAR:
        raise ValueError(f"Acuan '{kode}' tidak dikenal. Pilih dari {list(DAFTAR)}.")
    pilihan = {kode: DAFTAR[kode]} if kode else DAFTAR

    baris = []
    for nama, acuan in pilihan.items():
        try:
            diperoleh = PERHITUNGAN[nama]()
        except Exception as galat:  # noqa: BLE001 - kegagalan dilaporkan, bukan disembunyikan
            baris.append(
                {
                    "Metode": acuan.metode,
                    "Pembanding": acuan.pembanding,
                    "Besaran": "—",
                    "NalarData": np.nan,
                    "Acuan": np.nan,
                    "Selisih": np.nan,
                    "Status": GAGAL,
                    "Keterangan": str(galat)[:160],
                }
            )
            continue

        for butir in acuan.butir:
            nilai = float(diperoleh.get(butir.nama, np.nan))
            baris.append(
                {
                    "Metode": acuan.metode,
                    "Pembanding": acuan.pembanding,
                    "Besaran": butir.nama,
                    "NalarData": nilai,
                    "Acuan": butir.harapan,
                    "Selisih": abs(nilai - butir.harapan),
                    "Status": _status(nilai, butir.harapan, butir.toleransi),
                    "Keterangan": acuan.sumber,
                }
            )
    return pd.DataFrame(baris)


def ringkas(hasil: pd.DataFrame | None = None) -> pd.DataFrame:
    """Status per metode, satu baris per metode."""
    hasil = jalankan() if hasil is None else hasil
    baris = []
    for metode, bagian in hasil.groupby("Metode", sort=False):
        status = set(bagian["Status"])
        if GAGAL in status:
            keseluruhan = GAGAL
        elif BERBEDA in status:
            keseluruhan = BERBEDA
        elif TOLERANSI in status:
            keseluruhan = TOLERANSI
        else:
            keseluruhan = SESUAI
        baris.append(
            {
                "Metode": metode,
                "Pembanding": bagian["Pembanding"].iloc[0],
                "Besaran diperiksa": len(bagian),
                "Status": keseluruhan,
            }
        )
    for metode, alasan in BELUM_DIVALIDASI.items():
        baris.append(
            {
                "Metode": metode,
                "Pembanding": "—",
                "Besaran diperiksa": 0,
                "Status": BELUM,
            }
        )
    return pd.DataFrame(baris)


def cakupan() -> dict[str, int]:
    """Berapa metode sudah divalidasi dan berapa yang belum, tanpa dibulatkan ke atas."""
    tabel = ringkas()
    return {
        "Metode divalidasi": int((tabel["Status"] != BELUM).sum()),
        "Belum divalidasi": int((tabel["Status"] == BELUM).sum()),
        "Besaran diperiksa": int(tabel["Besaran diperiksa"].sum()),
    }
