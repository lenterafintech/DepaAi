"""Uji beda parametrik: uji-t dan ANOVA satu arah beserta uji lanjutannya.

Modul ini menutup lubang yang ditemukan saat menyusun validasi lintas software:
Pemandu Uji menyarankan uji-t sampel bebas, uji-t Welch, One-Way ANOVA, dan Welch
ANOVA — sedangkan aplikasi hanya memiliki padanan non-parametriknya. Pengguna yang
menuruti saran pemandu tiba di halaman yang tidak dapat menjalankan uji yang
disarankan kepadanya.

Bentuk hasilnya sengaja sama dengan uji non-parametrik (``HasilUji``), sehingga
halaman Uji Beda menampilkan keduanya dengan tata letak yang sama dan pengguna
dapat membandingkannya berdampingan — yang justru merupakan cara paling baik
memahami apa yang hilang ketika asumsi dilanggar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from nalardata.nonparametrik import HasilUji, _deskriptif

# Patokan Cohen (1988), dipakai untuk menafsirkan besaran efek.
AMBANG_D = ((0.2, "kecil"), (0.5, "sedang"), (0.8, "besar"))
AMBANG_ETA = ((0.01, "kecil"), (0.06, "sedang"), (0.14, "besar"))


def _tafsir(nilai: float, ambang) -> str:
    if not np.isfinite(nilai):
        return "tidak dapat dihitung"
    besar = abs(nilai)
    label = "sangat kecil"
    for batas, sebutan in ambang:
        if besar >= batas:
            label = sebutan
    return label


def _dua_kelompok(nilai: pd.Series, kelompok: pd.Series) -> dict[str, np.ndarray]:
    bingkai = pd.DataFrame(
        {"nilai": pd.to_numeric(nilai, errors="coerce"), "grup": kelompok}
    ).dropna()
    isi = {
        str(nama): bagian["nilai"].to_numpy(dtype=float)
        for nama, bagian in bingkai.groupby("grup", observed=True)
        if len(bagian) >= 2
    }
    return isi


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d dengan simpangan baku gabungan."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    gabungan = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    )
    return float((a.mean() - b.mean()) / gabungan) if gabungan > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Uji-t
# --------------------------------------------------------------------------- #


def uji_t_bebas(
    nilai: pd.Series, kelompok: pd.Series, ragam_sama: bool = True
) -> HasilUji:
    """Uji-t dua sampel bebas.

    ``ragam_sama=False`` menghasilkan uji Welch, yang tidak menuntut ragam kedua
    kelompok sama. Welch lebih aman dipakai sebagai bawaan pada data nyata, namun
    pilihannya diserahkan pengguna karena pelaporan skripsi lazimnya menyebut
    keduanya secara terpisah.
    """
    isi = _dua_kelompok(nilai, kelompok)
    if len(isi) != 2:
        raise ValueError(
            f"Uji-t sampel bebas menuntut tepat 2 kelompok berisi minimal 2 data; "
            f"ditemukan {len(isi)}."
        )

    (nama_a, a), (nama_b, b) = isi.items()
    t, p = stats.ttest_ind(a, b, equal_var=ragam_sama)
    if ragam_sama:
        db = len(a) + len(b) - 2
    else:
        # Koreksi Welch-Satterthwaite.
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        db = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))

    d = _cohen_d(a, b)
    return HasilUji(
        nama="Uji-t Welch" if not ragam_sama else "Uji-t sampel bebas",
        kode="t_welch" if not ragam_sama else "t_bebas",
        statistik=float(t),
        p_value=float(p),
        n=len(a) + len(b),
        label_statistik="t",
        efek_nama="Cohen's d",
        efek_nilai=d,
        efek_tafsir=_tafsir(d, AMBANG_D),
        keterangan=f"Derajat bebas = {db:.3f}. Membandingkan '{nama_a}' dan '{nama_b}'.",
        tabel=_deskriptif(isi),
        catatan=(
            []
            if not ragam_sama
            else [
                "Uji ini mengandaikan ragam kedua kelompok sama. Bila uji Levene "
                "menolaknya, pakai Welch."
            ]
        ),
    )


def uji_t_berpasangan(a: pd.Series, b: pd.Series, nama_a: str = "Sebelum", nama_b: str = "Sesudah") -> HasilUji:
    """Uji-t untuk dua pengukuran pada unit yang sama."""
    bingkai = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(bingkai) < 3:
        raise ValueError("Uji-t berpasangan menuntut minimal 3 pasangan lengkap.")

    x, y = bingkai["a"].to_numpy(float), bingkai["b"].to_numpy(float)
    t, p = stats.ttest_rel(x, y)
    selisih = x - y
    d = float(selisih.mean() / selisih.std(ddof=1)) if selisih.std(ddof=1) > 0 else float("nan")
    return HasilUji(
        nama="Uji-t berpasangan",
        kode="t_berpasangan",
        statistik=float(t),
        p_value=float(p),
        n=len(bingkai),
        label_statistik="t",
        efek_nama="Cohen's d (berpasangan)",
        efek_nilai=d,
        efek_tafsir=_tafsir(d, AMBANG_D),
        keterangan=f"Derajat bebas = {len(bingkai) - 1}. Selisih {nama_a} dikurangi {nama_b}.",
        tabel=_deskriptif({nama_a: x, nama_b: y}),
    )


def uji_t_satu_sampel(nilai: pd.Series, pembanding: float = 0.0) -> HasilUji:
    """Uji-t satu sampel terhadap sebuah nilai acuan."""
    x = pd.to_numeric(nilai, errors="coerce").dropna().to_numpy(float)
    if len(x) < 3:
        raise ValueError("Uji-t satu sampel menuntut minimal 3 data.")

    t, p = stats.ttest_1samp(x, pembanding)
    d = float((x.mean() - pembanding) / x.std(ddof=1)) if x.std(ddof=1) > 0 else float("nan")
    return HasilUji(
        nama="Uji-t satu sampel",
        kode="t_satu",
        statistik=float(t),
        p_value=float(p),
        n=len(x),
        label_statistik="t",
        efek_nama="Cohen's d",
        efek_nilai=d,
        efek_tafsir=_tafsir(d, AMBANG_D),
        keterangan=f"Derajat bebas = {len(x) - 1}. Dibandingkan dengan {pembanding:g}.",
        tabel=_deskriptif({"Sampel": x}),
    )


# --------------------------------------------------------------------------- #
# ANOVA
# --------------------------------------------------------------------------- #


def anova_satu_arah(nilai: pd.Series, kelompok: pd.Series) -> HasilUji:
    """ANOVA satu arah: tiga kelompok bebas atau lebih dengan ragam yang seragam."""
    isi = _dua_kelompok(nilai, kelompok)
    if len(isi) < 2:
        raise ValueError("ANOVA menuntut minimal 2 kelompok berisi masing-masing 2 data.")

    f, p = stats.f_oneway(*isi.values())
    semua = np.concatenate(list(isi.values()))
    n, k = len(semua), len(isi)
    rerata = semua.mean()
    ss_antar = sum(len(v) * (v.mean() - rerata) ** 2 for v in isi.values())
    ss_total = float(((semua - rerata) ** 2).sum())
    eta2 = ss_antar / ss_total if ss_total > 0 else float("nan")

    return HasilUji(
        nama=f"One-Way ANOVA ({k} kelompok)",
        kode="anova",
        statistik=float(f),
        p_value=float(p),
        n=n,
        label_statistik="F",
        efek_nama="eta kuadrat",
        efek_nilai=float(eta2),
        efek_tafsir=_tafsir(eta2, AMBANG_ETA),
        keterangan=f"Derajat bebas = ({k - 1}, {n - k}).",
        tabel=_deskriptif(isi),
        catatan=(
            ["Uji ini hanya menyatakan ada perbedaan, belum menunjuk kelompok mana; "
             "lihat uji lanjutan Tukey HSD."]
            if p < 0.05
            else []
        ),
    )


def welch_anova(nilai: pd.Series, kelompok: pd.Series) -> HasilUji:
    """Welch ANOVA: tidak menuntut ragam antar kelompok sama.

    Dihitung langsung dari rumusnya karena scipy belum menyediakannya, lalu
    diperiksa terhadap keluaran ``oneway.test`` di R.
    """
    isi = _dua_kelompok(nilai, kelompok)
    if len(isi) < 2:
        raise ValueError("Welch ANOVA menuntut minimal 2 kelompok berisi masing-masing 2 data.")

    n = np.array([len(v) for v in isi.values()], dtype=float)
    rerata = np.array([v.mean() for v in isi.values()])
    ragam = np.array([v.var(ddof=1) for v in isi.values()])
    if np.any(ragam <= 0):
        raise ValueError("Salah satu kelompok tidak memiliki keragaman sama sekali.")

    bobot = n / ragam
    rerata_bobot = float((bobot * rerata).sum() / bobot.sum())
    k = len(isi)

    pembilang = float((bobot * (rerata - rerata_bobot) ** 2).sum()) / (k - 1)
    lambda_ = float((((1 - bobot / bobot.sum()) ** 2) / (n - 1)).sum())
    penyebut = 1 + (2 * (k - 2) / (k**2 - 1)) * lambda_
    f = pembilang / penyebut
    db2 = (k**2 - 1) / (3 * lambda_)
    p = float(stats.f.sf(f, k - 1, db2))

    semua = np.concatenate(list(isi.values()))
    total = float(((semua - semua.mean()) ** 2).sum())
    ss_antar = float(sum(len(v) * (v.mean() - semua.mean()) ** 2 for v in isi.values()))
    eta2 = ss_antar / total if total > 0 else float("nan")

    return HasilUji(
        nama=f"Welch ANOVA ({k} kelompok)",
        kode="welch_anova",
        statistik=float(f),
        p_value=p,
        n=int(n.sum()),
        label_statistik="F",
        efek_nama="eta kuadrat",
        efek_nilai=float(eta2),
        efek_tafsir=_tafsir(eta2, AMBANG_ETA),
        keterangan=f"Derajat bebas = ({k - 1}; {db2:.3f}).",
        tabel=_deskriptif(isi),
        catatan=[
            "Welch ANOVA tidak menuntut ragam antar kelompok sama. Uji lanjutannya "
            "adalah Games-Howell, bukan Tukey."
        ],
    )


# --------------------------------------------------------------------------- #
# Uji lanjutan
# --------------------------------------------------------------------------- #


def tukey(nilai: pd.Series, kelompok: pd.Series, alfa: float = 0.05) -> pd.DataFrame:
    """Uji lanjutan Tukey HSD: pasangan kelompok mana yang berbeda."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    bingkai = pd.DataFrame(
        {"nilai": pd.to_numeric(nilai, errors="coerce"), "grup": kelompok.astype(str)}
    ).dropna()
    if bingkai["grup"].nunique() < 2:
        raise ValueError("Uji Tukey menuntut minimal 2 kelompok.")

    hasil = pairwise_tukeyhsd(bingkai["nilai"], bingkai["grup"], alpha=alfa)
    tabel = pd.DataFrame(hasil.summary().data[1:], columns=hasil.summary().data[0])
    return pd.DataFrame(
        {
            "Kelompok 1": tabel["group1"],
            "Kelompok 2": tabel["group2"],
            "Selisih rata-rata": pd.to_numeric(tabel["meandiff"]),
            "Batas bawah": pd.to_numeric(tabel["lower"]),
            "Batas atas": pd.to_numeric(tabel["upper"]),
            "p disesuaikan": pd.to_numeric(tabel["p-adj"]),
            "Berbeda": [
                "Ya" if str(v) in {"True", "true"} else "Tidak" for v in tabel["reject"]
            ],
        }
    )


def games_howell(nilai: pd.Series, kelompok: pd.Series) -> pd.DataFrame:
    """Uji lanjutan Games-Howell: tidak menuntut ragam maupun ukuran kelompok sama."""
    isi = _dua_kelompok(nilai, kelompok)
    if len(isi) < 2:
        raise ValueError("Games-Howell menuntut minimal 2 kelompok.")

    nama = list(isi)
    baris = []
    for i in range(len(nama)):
        for j in range(i + 1, len(nama)):
            a, b = isi[nama[i]], isi[nama[j]]
            na, nb = len(a), len(b)
            va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
            selisih = float(a.mean() - b.mean())
            galat = float(np.sqrt(va + vb))
            db = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
            t = selisih / galat if galat > 0 else float("nan")
            # Sebaran rentang terstudentkan memakai q = sqrt(2) x |t|.
            p = float(stats.studentized_range.sf(abs(t) * np.sqrt(2), len(nama), db))
            baris.append(
                {
                    "Kelompok 1": nama[i],
                    "Kelompok 2": nama[j],
                    "Selisih rata-rata": selisih,
                    "Galat baku": galat,
                    "df": float(db),
                    "p": min(p, 1.0),
                    "Berbeda": "Ya" if p < 0.05 else "Tidak",
                }
            )
    return pd.DataFrame(baris)
