"""Ukuran sampel: berapa responden yang diperlukan, dan mengapa angka itu.

Pertanyaan ini datang jauh sebelum data ada, dan hampir selalu dijawab dengan satu
rumus: Slovin. Rumus itu memang mudah dipakai, tetapi ia hanya memperhitungkan besar
populasi dan galat yang ditoleransi. Ia tidak tahu seberapa besar pengaruh yang hendak
dideteksi, tidak memperhitungkan kuesioner yang tidak kembali, dan sama sekali tidak
menjamin bahwa sampelnya mewakili populasi — keterwakilan ditentukan **cara mengambil
sampel**, bukan banyaknya.

Modul ini menyediakan Slovin karena memang itu yang diminta pembimbing di banyak
kampus, tetapi selalu berdampingan dengan hitungan daya uji, agar pengguna melihat
bahwa keduanya menjawab pertanyaan yang berbeda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Ambang lazim. Dikumpulkan agar dapat ditinjau dan dikutip, bukan tersebar di kode.
ALFA = 0.05
DAYA = 0.80

# Patokan besar pengaruh menurut Cohen (1988). Dipakai sebagai titik awal, bukan
# kebenaran: besar pengaruh yang layak dikejar bergantung pada bidangnya.
EFEK_D = {"kecil": 0.20, "sedang": 0.50, "besar": 0.80}
EFEK_F = {"kecil": 0.10, "sedang": 0.25, "besar": 0.40}
EFEK_F2 = {"kecil": 0.02, "sedang": 0.15, "besar": 0.35}
EFEK_R = {"kecil": 0.10, "sedang": 0.30, "besar": 0.50}


@dataclass
class Hitungan:
    """Satu hasil hitungan ukuran sampel beserta dasarnya."""

    metode: str
    n: int
    dasar: str
    catatan: str = ""


# --------------------------------------------------------------------------- #
# Rumus berbasis populasi
# --------------------------------------------------------------------------- #


def slovin(populasi: int, galat: float = 0.05) -> Hitungan:
    """Rumus Slovin: n = N / (1 + N e²).

    Galat 5 persen untuk penelitian yang menuntut ketelitian, 10 persen untuk
    penelitian penjajakan berpopulasi besar.
    """
    if populasi <= 0:
        raise ValueError("Ukuran populasi harus lebih besar dari nol.")
    if not 0 < galat < 1:
        raise ValueError("Galat yang ditoleransi harus di antara 0 dan 1.")

    n = populasi / (1 + populasi * galat**2)
    return Hitungan(
        metode="Slovin",
        n=int(np.ceil(n)),
        dasar=f"Populasi {populasi:,} dengan galat {galat:.0%}".replace(",", "."),
        catatan=(
            "Slovin hanya memperhitungkan besar populasi dan galat. Ia tidak "
            "memperhitungkan besar pengaruh yang hendak dideteksi maupun kuesioner "
            "yang tidak kembali, dan tidak menjamin keterwakilan."
        ),
    )


def cochran(
    proporsi: float = 0.5, galat: float = 0.05, alfa: float = ALFA, populasi: int | None = None
) -> Hitungan:
    """Rumus Cochran untuk proporsi, dengan koreksi populasi terhingga bila diketahui.

    Proporsi 0,5 dipakai bila belum ada dugaan, karena menghasilkan sampel terbesar
    — pilihan paling aman ketika keadaan populasi belum diketahui.
    """
    if not 0 < proporsi < 1:
        raise ValueError("Proporsi harus di antara 0 dan 1.")
    if not 0 < galat < 1:
        raise ValueError("Galat yang ditoleransi harus di antara 0 dan 1.")

    z = float(stats.norm.ppf(1 - alfa / 2))
    n0 = (z**2) * proporsi * (1 - proporsi) / (galat**2)

    catatan = (
        "Proporsi 0,5 menghasilkan sampel terbesar, jadi aman dipakai bila keadaan "
        "populasi belum diketahui."
        if abs(proporsi - 0.5) < 1e-9
        else ""
    )
    if populasi and populasi > 0:
        n = n0 / (1 + (n0 - 1) / populasi)
        dasar = (
            f"Proporsi {proporsi:.2f}, galat {galat:.0%}, populasi "
            f"{populasi:,}".replace(",", ".")
        )
    else:
        n = n0
        dasar = f"Proporsi {proporsi:.2f}, galat {galat:.0%}, populasi tak terhingga"

    return Hitungan(metode="Cochran", n=int(np.ceil(n)), dasar=dasar, catatan=catatan)


# --------------------------------------------------------------------------- #
# Daya uji
# --------------------------------------------------------------------------- #


def daya_uji_t(d: float = 0.5, daya: float = DAYA, alfa: float = ALFA) -> Hitungan:
    """Ukuran sampel per kelompok untuk uji-t dua sampel bebas."""
    from statsmodels.stats.power import TTestIndPower

    _periksa_daya(daya, alfa)
    if d <= 0:
        raise ValueError("Besar pengaruh harus lebih besar dari nol.")

    per_kelompok = float(TTestIndPower().solve_power(effect_size=d, power=daya, alpha=alfa))
    n = int(np.ceil(per_kelompok))
    return Hitungan(
        metode="Daya uji — uji-t dua kelompok",
        n=n * 2,
        dasar=f"d = {d:.2f}, daya {daya:.0%}, alfa {alfa:.2f}",
        catatan=f"{n} responden per kelompok, {n * 2} seluruhnya.",
    )


def daya_uji_anova(
    kelompok: int, f: float = 0.25, daya: float = DAYA, alfa: float = ALFA
) -> Hitungan:
    """Ukuran sampel total untuk ANOVA satu arah."""
    from statsmodels.stats.power import FTestAnovaPower

    _periksa_daya(daya, alfa)
    if kelompok < 2:
        raise ValueError("ANOVA memerlukan sekurang-kurangnya dua kelompok.")
    if f <= 0:
        raise ValueError("Besar pengaruh harus lebih besar dari nol.")

    total = float(
        FTestAnovaPower().solve_power(effect_size=f, k_groups=kelompok, power=daya, alpha=alfa)
    )
    n = int(np.ceil(total))
    per_kelompok = int(np.ceil(n / kelompok))
    return Hitungan(
        metode=f"Daya uji — ANOVA {kelompok} kelompok",
        n=per_kelompok * kelompok,
        dasar=f"f = {f:.2f}, daya {daya:.0%}, alfa {alfa:.2f}",
        catatan=f"Sekitar {per_kelompok} responden per kelompok.",
    )


def daya_uji_regresi(
    prediktor: int, f2: float = 0.15, daya: float = DAYA, alfa: float = ALFA
) -> Hitungan:
    """Ukuran sampel untuk uji F keseluruhan pada regresi berganda.

    Dihitung langsung dari sebaran F non-sentral dengan parameter kenon-sentralan
    ``lambda = f2 x N`` (Cohen, 1988), lalu dicari N terkecil yang mencapai daya
    yang diminta.
    """
    _periksa_daya(daya, alfa)
    if prediktor < 1:
        raise ValueError("Regresi memerlukan sekurang-kurangnya satu prediktor.")
    if f2 <= 0:
        raise ValueError("Besar pengaruh harus lebih besar dari nol.")

    df1 = prediktor
    for n in range(prediktor + 2, 100_001):
        df2 = n - prediktor - 1
        if df2 < 1:
            continue
        kritis = float(stats.f.ppf(1 - alfa, df1, df2))
        tercapai = float(stats.ncf.sf(kritis, df1, df2, f2 * n))
        if tercapai >= daya:
            return Hitungan(
                metode=f"Daya uji — regresi {prediktor} prediktor",
                n=n,
                dasar=f"f² = {f2:.2f}, daya {daya:.0%}, alfa {alfa:.2f}",
                catatan=(
                    "Angka ini untuk uji F keseluruhan model. Menguji koefisien satu "
                    "prediktor tertentu menuntut sampel lebih besar."
                ),
            )
    raise ValueError("Ukuran sampel yang diperlukan melebihi batas yang wajar dihitung.")


def daya_uji_korelasi(r: float = 0.3, daya: float = DAYA, alfa: float = ALFA) -> Hitungan:
    """Ukuran sampel untuk uji korelasi, lewat transformasi Fisher z."""
    _periksa_daya(daya, alfa)
    if not 0 < abs(r) < 1:
        raise ValueError("Koefisien korelasi harus di antara -1 dan 1, dan bukan nol.")

    z_alfa = float(stats.norm.ppf(1 - alfa / 2))
    z_daya = float(stats.norm.ppf(daya))
    zr = np.arctanh(abs(r))
    n = int(np.ceil(((z_alfa + z_daya) / zr) ** 2 + 3))
    return Hitungan(
        metode="Daya uji — korelasi",
        n=n,
        dasar=f"r = {r:.2f}, daya {daya:.0%}, alfa {alfa:.2f}",
        catatan=(
            "Dihitung dengan hampiran Fisher z, yang lazimnya satu responden lebih "
            "besar daripada metode eksak — selisih yang tidak mengubah keputusan."
        ),
    )


def _periksa_daya(daya: float, alfa: float) -> None:
    if not 0 < daya < 1:
        raise ValueError("Daya uji harus di antara 0 dan 1.")
    if not 0 < alfa < 1:
        raise ValueError("Alfa harus di antara 0 dan 1.")


# --------------------------------------------------------------------------- #
# Cadangan non-respons
# --------------------------------------------------------------------------- #


def dengan_cadangan(n: int, tingkat_kembali: float = 0.8) -> int:
    """Jumlah kuesioner yang perlu disebar agar ``n`` kembali terisi lengkap."""
    if n <= 0:
        raise ValueError("Ukuran sampel harus lebih besar dari nol.")
    if not 0 < tingkat_kembali <= 1:
        raise ValueError("Tingkat pengembalian harus di antara 0 dan 1.")
    return int(np.ceil(n / tingkat_kembali))


# --------------------------------------------------------------------------- #
# Perbandingan
# --------------------------------------------------------------------------- #


def bandingkan(hitungan: list[Hitungan], tingkat_kembali: float | None = None) -> pd.DataFrame:
    """Sandingkan beberapa hitungan agar bedanya terlihat, bukan dipilih diam-diam."""
    if not hitungan:
        raise ValueError("Tidak ada hitungan untuk dibandingkan.")

    baris = []
    for h in hitungan:
        isi = {
            "Metode": h.metode,
            "Ukuran sampel": h.n,
            "Dasar": h.dasar,
            "Catatan": h.catatan,
        }
        if tingkat_kembali:
            isi["Perlu disebar"] = dengan_cadangan(h.n, tingkat_kembali)
        baris.append(isi)
    return pd.DataFrame(baris)
