"""Reliabilitas dan validitas instrumen pengukuran.

Menyediakan koefisien alpha Cronbach beserta statistik per butir, omega McDonald,
composite reliability (CR) dan average variance extracted (AVE) dari muatan faktor
tunggal, serta pemeriksaan validitas diskriminan kriteria Fornell-Larcker.

Ambang yang dipakai mengikuti Hair dkk. (2019): alpha dan CR >= 0,70; AVE >= 0,50;
muatan butir >= 0,50. Validitas diskriminan terpenuhi bila akar AVE tiap konstruk
melampaui seluruh korelasinya dengan konstruk lain (Fornell & Larcker, 1981).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from nalardata import preprocessing

AMBANG_ALPHA = 0.70
AMBANG_CR = 0.70
AMBANG_AVE = 0.50
AMBANG_MUATAN = 0.50


@dataclass
class HasilAlpha:
    """Koefisien alpha Cronbach beserta rincian per butir."""

    alpha: float
    n_item: int
    n_observasi: int
    item: pd.DataFrame  # per butir: rata-rata, SD, korelasi item-total, alpha jika dibuang
    spearman_brown: float = float("nan")  # reliabilitas belah-dua terkoreksi
    korelasi_belahan: float = float("nan")  # korelasi mentah antar kedua belahan

    def interpretasi(self) -> str:
        if self.alpha >= 0.9:
            return "Sangat baik"
        if self.alpha >= 0.8:
            return "Baik"
        if self.alpha >= AMBANG_ALPHA:
            return "Dapat diterima"
        if self.alpha >= 0.6:
            return "Meragukan"
        return "Tidak dapat diterima"

    def butir_bermasalah(self) -> list[str]:
        """Butir yang justru menaikkan alpha bila dibuang, atau korelasinya rendah."""
        naik = self.item.loc[self.item["Alpha jika dibuang"] > self.alpha, "Butir"]
        lemah = self.item.loc[self.item["Korelasi item-total"] < 0.3, "Butir"]
        return sorted(set(naik) | set(lemah))


@dataclass
class HasilKonstruk:
    """Ringkasan reliabilitas dan validitas satu konstruk."""

    nama: str
    butir: list[str]
    alpha: float
    omega: float
    cr: float
    ave: float
    muatan: pd.Series

    @property
    def akar_ave(self) -> float:
        return float(np.sqrt(self.ave))

    def memenuhi(self) -> bool:
        return self.alpha >= AMBANG_ALPHA and self.cr >= AMBANG_CR and self.ave >= AMBANG_AVE

    def catatan(self) -> str:
        kurang = []
        if self.alpha < AMBANG_ALPHA:
            kurang.append("alpha")
        if self.cr < AMBANG_CR:
            kurang.append("CR")
        if self.ave < AMBANG_AVE:
            kurang.append("AVE")
        lemah = self.muatan[self.muatan.abs() < AMBANG_MUATAN].index.tolist()
        if lemah:
            kurang.append(f"muatan {', '.join(lemah)}")
        return "Memenuhi" if not kurang else "Belum memenuhi: " + ", ".join(kurang)


def alpha_cronbach(df: pd.DataFrame) -> HasilAlpha:
    """Alpha Cronbach beserta korelasi item-total terkoreksi dan alpha-if-deleted."""
    data = preprocessing.clean_subset(df, list(df.columns))
    k = data.shape[1]
    if k < 2:
        raise ValueError("Alpha Cronbach memerlukan minimal 2 butir.")
    if len(data) < 3:
        raise ValueError("Alpha Cronbach memerlukan minimal 3 observasi lengkap.")

    def _alpha(sub: pd.DataFrame) -> float:
        jumlah_butir = sub.shape[1]
        if jumlah_butir < 2:
            return float("nan")
        ragam_butir = sub.var(ddof=1).sum()
        ragam_total = sub.sum(axis=1).var(ddof=1)
        if ragam_total == 0:
            return float("nan")
        return float(jumlah_butir / (jumlah_butir - 1) * (1 - ragam_butir / ragam_total))

    alpha = _alpha(data)
    baris = []
    for butir in data.columns:
        sisa = data.drop(columns=[butir])
        # Korelasi item-total terkoreksi: butir dibandingkan dengan total butir lain.
        korelasi = float(data[butir].corr(sisa.sum(axis=1)))
        baris.append(
            {
                "Butir": butir,
                "Rata-rata": float(data[butir].mean()),
                "SD": float(data[butir].std(ddof=1)),
                "Korelasi item-total": korelasi,
                "Alpha jika dibuang": _alpha(sisa),
            }
        )
    sb, r_belah = belah_dua(data)
    return HasilAlpha(
        alpha=alpha,
        n_item=k,
        n_observasi=len(data),
        item=pd.DataFrame(baris),
        spearman_brown=sb,
        korelasi_belahan=r_belah,
    )


def belah_dua(df: pd.DataFrame) -> tuple[float, float]:
    """Reliabilitas belah-dua Spearman-Brown beserta korelasi mentah antar belahan.

    Butir dibelah ganjil-genap, bukan separuh-awal separuh-akhir, karena urutan
    butir pada kuesioner kerap membawa efek kelelahan atau urutan tema yang membuat
    pembelahan berurutan bias.

    Koreksi Spearman-Brown mengembalikan korelasi antar belahan ke panjang tes
    penuh: r_sb = 2r / (1 + r). Tanpa koreksi itu, angkanya meremehkan reliabilitas
    karena tiap belahan hanya separuh panjang instrumen.
    """
    data = preprocessing.clean_subset(df, list(df.columns))
    if data.shape[1] < 2:
        return float("nan"), float("nan")

    ganjil = data.iloc[:, 0::2].sum(axis=1)
    genap = data.iloc[:, 1::2].sum(axis=1)
    if ganjil.std(ddof=1) == 0 or genap.std(ddof=1) == 0:
        return float("nan"), float("nan")

    r = float(ganjil.corr(genap))
    if not np.isfinite(r) or r <= -1:
        return float("nan"), r
    return float(2 * r / (1 + r)), r


def muatan_faktor_tunggal(df: pd.DataFrame) -> pd.Series:
    """Muatan butir pada satu faktor bersama, dasar perhitungan omega, CR, dan AVE."""
    data = preprocessing.clean_subset(df, list(df.columns))
    z = preprocessing.scale(data, "z-score")
    model = FactorAnalysis(n_components=1, random_state=0).fit(z.to_numpy())
    muatan = pd.Series(model.components_[0], index=data.columns, name="Muatan")
    # Tanda faktor bersifat arbitrer; disepakati mayoritas butir bermuatan positif.
    if (muatan < 0).sum() > len(muatan) / 2:
        muatan = -muatan
    return muatan


def cr_ave(muatan: pd.Series) -> tuple[float, float]:
    """Composite reliability dan average variance extracted dari muatan faktor.

    Nilai mutlak muatan dipakai karena butir berarah terbalik bermuatan negatif;
    tanpa itu jumlah muatan saling meniadakan dan CR jatuh secara keliru.
    """
    lam = np.abs(muatan.to_numpy(dtype=float))
    galat = 1 - lam**2
    penyebut = lam.sum() ** 2 + galat.sum()
    cr = float(lam.sum() ** 2 / penyebut) if penyebut else float("nan")
    ave = float((lam**2).mean())
    return cr, ave


def omega_mcdonald(muatan: pd.Series) -> float:
    """Omega McDonald; secara perhitungan identik dengan CR pada model satu faktor."""
    cr, _ = cr_ave(muatan)
    return cr


def analisis_konstruk(df: pd.DataFrame, konstruk: dict[str, list[str]]) -> list[HasilKonstruk]:
    """Hitung reliabilitas dan validitas konvergen untuk tiap konstruk."""
    hasil: list[HasilKonstruk] = []
    for nama, butir in konstruk.items():
        tersedia = [b for b in butir if b in df.columns]
        if len(tersedia) < 2:
            raise ValueError(f"Konstruk '{nama}' memerlukan minimal 2 butir.")
        subset = df[tersedia]
        muatan = muatan_faktor_tunggal(subset)
        cr, ave = cr_ave(muatan)
        hasil.append(
            HasilKonstruk(
                nama=nama,
                butir=tersedia,
                alpha=alpha_cronbach(subset).alpha,
                omega=omega_mcdonald(muatan),
                cr=cr,
                ave=ave,
                muatan=muatan,
            )
        )
    return hasil


def tabel_konstruk(hasil: list[HasilKonstruk]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Konstruk": [h.nama for h in hasil],
            "Jumlah butir": [len(h.butir) for h in hasil],
            "Alpha": [h.alpha for h in hasil],
            "Omega": [h.omega for h in hasil],
            "CR": [h.cr for h in hasil],
            "AVE": [h.ave for h in hasil],
            "√AVE": [h.akar_ave for h in hasil],
            "Keputusan": [h.catatan() for h in hasil],
        }
    )


def skor_konstruk(df: pd.DataFrame, hasil: list[HasilKonstruk]) -> pd.DataFrame:
    """Skor komposit tiap konstruk sebagai rata-rata butirnya."""
    return pd.DataFrame({h.nama: df[h.butir].mean(axis=1) for h in hasil})


def fornell_larcker(df: pd.DataFrame, hasil: list[HasilKonstruk]) -> pd.DataFrame:
    """Matriks Fornell-Larcker: akar AVE pada diagonal, korelasi di bawahnya."""
    skor = skor_konstruk(df, hasil).dropna()
    korelasi = skor.corr()
    nama = [h.nama for h in hasil]
    matriks = pd.DataFrame(index=nama, columns=nama, dtype=float)
    for i, a in enumerate(nama):
        for j, b in enumerate(nama):
            if i == j:
                matriks.loc[a, b] = hasil[i].akar_ave
            elif i > j:
                matriks.loc[a, b] = float(korelasi.loc[a, b])
            else:
                matriks.loc[a, b] = np.nan
    return matriks


def htmt(df: pd.DataFrame, konstruk: dict[str, list[str]]) -> pd.DataFrame:
    """Rasio Heterotrait-Monotrait (Henseler dkk., 2015) untuk validitas diskriminan.

    Membandingkan rata-rata korelasi antar butir dari konstruk berbeda (heterotrait)
    dengan rata-rata korelasi antar butir di dalam konstruk yang sama (monotrait).
    Pada model kongenerik bermuatan seragam, hasilnya menaksir korelasi antar konstruk
    itu sendiri; ketika muatannya beragam, taksirannya sedikit lebih tinggi.

    Henseler dkk. melaporkan bahwa kriteria ini lebih peka daripada Fornell-Larcker
    dalam mengenali konstruk yang sebenarnya tumpang tindih. Keduanya disediakan
    berdampingan, bukan saling menggantikan, karena penelaah berbeda meminta yang
    berbeda.

    Ambang 0,85 dipakai untuk konstruk yang secara konsep berbeda tegas, sedangkan
    0,90 untuk konstruk yang memang berdekatan maknanya.
    """
    nama = [k for k, butir in konstruk.items() if len(butir) >= 2]
    if len(nama) < 2:
        raise ValueError("HTMT memerlukan minimal 2 konstruk dengan masing-masing 2 butir.")

    semua = [b for k in nama for b in konstruk[k]]
    hilang = [b for b in semua if b not in df.columns]
    if hilang:
        raise ValueError(f"Butir tidak ada dalam data: {', '.join(hilang)}.")

    korelasi = preprocessing.clean_subset(df, semua).corr().abs()

    def _rata_monotrait(butir: list[str]) -> float:
        """Rata-rata korelasi antar butir di dalam satu konstruk."""
        nilai = [
            korelasi.loc[a, b]
            for i, a in enumerate(butir)
            for b in butir[i + 1 :]
        ]
        return float(np.mean(nilai)) if nilai else float("nan")

    baris = []
    for i, a in enumerate(nama):
        for b in nama[i + 1 :]:
            hetero = [korelasi.loc[x, y] for x in konstruk[a] for y in konstruk[b]]
            rata_hetero = float(np.mean(hetero)) if hetero else float("nan")
            mono = np.sqrt(_rata_monotrait(konstruk[a]) * _rata_monotrait(konstruk[b]))
            nilai = rata_hetero / mono if mono and np.isfinite(mono) and mono > 0 else float("nan")
            baris.append(
                {
                    "Konstruk A": a,
                    "Konstruk B": b,
                    "HTMT": nilai,
                    "Keputusan (0,85)": (
                        "Terpenuhi" if np.isfinite(nilai) and nilai < 0.85 else "Tidak terpenuhi"
                    ),
                    "Keputusan (0,90)": (
                        "Terpenuhi" if np.isfinite(nilai) and nilai < 0.90 else "Tidak terpenuhi"
                    ),
                }
            )
    return pd.DataFrame(baris).sort_values("HTMT", ascending=False).reset_index(drop=True)


def periksa_diskriminan(df: pd.DataFrame, hasil: list[HasilKonstruk]) -> pd.DataFrame:
    """Pasangan konstruk yang melanggar kriteria Fornell-Larcker."""
    skor = skor_konstruk(df, hasil).dropna()
    korelasi = skor.corr()
    baris = []
    for i, a in enumerate(hasil):
        for b in hasil[i + 1 :]:
            r = float(korelasi.loc[a.nama, b.nama])
            batas = min(a.akar_ave, b.akar_ave)
            baris.append(
                {
                    "Pasangan": f"{a.nama} – {b.nama}",
                    "Korelasi": r,
                    "√AVE terkecil": batas,
                    "Keputusan": "Terpenuhi" if abs(r) < batas else "Tidak terpenuhi",
                }
            )
    return pd.DataFrame(baris)


def tebak_konstruk(kolom: list[str]) -> dict[str, list[str]]:
    """Kelompokkan butir bernomor menjadi konstruk berdasarkan awalan namanya.

    Butir kuesioner lazim dinamai KUAL1, KUAL2, … sehingga awalan sebelum angka
    dapat dipakai sebagai tebakan awal pengelompokan konstruk.
    """
    kelompok: dict[str, list[str]] = {}
    for nama in kolom:
        teks = str(nama)
        awalan = teks.rstrip("0123456789")
        if awalan == teks or not awalan:
            continue
        kelompok.setdefault(awalan.rstrip("_-"), []).append(teks)
    return {k: v for k, v in kelompok.items() if len(v) >= 2}
