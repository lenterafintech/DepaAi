"""Analisis faktor konfirmatori (CFA), analisis jalur, dan SEM.

Ketiganya memakai satu mesin estimasi yang sama (semopy) dan berbeda hanya pada
spesifikasi modelnya:

- **CFA** hanya memuat model pengukuran: konstruk laten diukur oleh butir-butirnya.
- **Analisis jalur** hanya memuat model struktural antar variabel teramati.
- **SEM** memuat keduanya sekaligus.

Modul ini menyediakan penyusun spesifikasi, penjalan model, indeks kecocokan
beserta ambangnya, reliabilitas konstruk dari muatan terstandardisasi, serta uji
mediasi dengan bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import semopy

import re

from lentera_mva import preprocessing

# Nama variabel dan konstruk harus berupa pengenal sederhana; spasi atau tanda baca
# membuat spesifikasi model ditolak oleh mesin estimasi.
POLA_NAMA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Ambang kecocokan model yang lazim dikutip pada naskah akademik.
AMBANG_FIT: dict[str, tuple[str, str]] = {
    "chi2/df": ("< 3,00", "Hair dkk. (2019)"),
    "CFI": ("> 0,90", "Hu & Bentler (1999)"),
    "TLI": ("> 0,90", "Hu & Bentler (1999)"),
    "RMSEA": ("< 0,08", "Hu & Bentler (1999)"),
    "GFI": ("> 0,90", "Hair dkk. (2019)"),
    "AGFI": ("> 0,90", "Hair dkk. (2019)"),
    "NFI": ("> 0,90", "Hair dkk. (2019)"),
}


@dataclass
class HasilSEM:
    """Hasil estimasi satu model pengukuran dan/atau struktural."""

    model: semopy.Model
    spesifikasi: str
    estimasi: pd.DataFrame
    statistik: pd.Series
    laten: list[str]
    teramati: list[str]
    n: int
    catatan: list[str] = field(default_factory=list)

    def muatan(self) -> pd.DataFrame:
        """Muatan butir pada konstruk laten (baris berjenis 'pengukuran')."""
        return self.estimasi[self.estimasi["Jenis"] == "Muatan"].reset_index(drop=True)

    def jalur(self) -> pd.DataFrame:
        """Jalur struktural antar variabel (regresi)."""
        return self.estimasi[self.estimasi["Jenis"] == "Jalur"].reset_index(drop=True)

    def kovarians(self) -> pd.DataFrame:
        return self.estimasi[self.estimasi["Jenis"] == "Kovarians"].reset_index(drop=True)

    def cocok(self) -> bool:
        tabel = tabel_kecocokan(self)
        keputusan = tabel[tabel["Indeks"] != "Chi-square"]["Keputusan"]
        return bool((keputusan == "Memenuhi").mean() >= 0.6)


# --------------------------------------------------------------------------- #
# Penyusun spesifikasi model
# --------------------------------------------------------------------------- #


def periksa_nama(nama: str, peran: str = "Nama") -> None:
    """Pastikan nama aman dipakai dalam spesifikasi model."""
    if not POLA_NAMA.match(str(nama)):
        raise ValueError(
            f"{peran} '{nama}' tidak dapat dipakai: gunakan huruf, angka, dan garis "
            "bawah tanpa spasi (misalnya 'kualitas_layanan')."
        )


def spesifikasi_cfa(konstruk: dict[str, list[str]]) -> str:
    """Model pengukuran: tiap konstruk laten diukur oleh butir-butirnya."""
    if len(konstruk) < 1:
        raise ValueError("CFA memerlukan minimal satu konstruk.")
    for nama, butir in konstruk.items():
        periksa_nama(nama, "Nama konstruk")
        for b in butir:
            periksa_nama(b, "Nama variabel")
        if len(butir) < 2:
            raise ValueError(f"Konstruk '{nama}' memerlukan minimal 2 butir.")
    baris = [f"{nama} =~ " + " + ".join(butir) for nama, butir in konstruk.items()]
    return "\n".join(baris)


def spesifikasi_jalur(jalur: dict[str, list[str]]) -> str:
    """Model struktural: tiap variabel terikat beserta prediktornya."""
    if not jalur:
        raise ValueError("Model jalur memerlukan minimal satu hubungan.")
    baris = []
    for terikat, penjelas in jalur.items():
        periksa_nama(terikat, "Nama variabel terikat")
        for p in penjelas:
            periksa_nama(p, "Nama prediktor")
        bersih = [p for p in penjelas if p != terikat]
        if not bersih:
            raise ValueError(f"Variabel '{terikat}' tidak punya prediktor.")
        baris.append(f"{terikat} ~ " + " + ".join(bersih))
    return "\n".join(baris)


def spesifikasi_sem(konstruk: dict[str, list[str]], jalur: dict[str, list[str]]) -> str:
    """Gabungan model pengukuran dan struktural."""
    return spesifikasi_cfa(konstruk) + "\n" + spesifikasi_jalur(jalur)


# --------------------------------------------------------------------------- #
# Estimasi
# --------------------------------------------------------------------------- #


def _jenis_baris(lval: str, op: str, rval: str, laten: set[str]) -> str:
    """Bedakan muatan pengukuran dari jalur struktural.

    semopy melaporkan muatan sebagai "butir ~ konstruk", bentuk yang sama dengan
    jalur struktural, sehingga pembedanya adalah apakah ruas kirinya variabel
    teramati sementara ruas kanannya konstruk laten.
    """
    if op == "~~":
        return "Kovarians"
    if op == "~" and rval in laten and lval not in laten:
        return "Muatan"
    return "Jalur"


def jalankan(
    df: pd.DataFrame,
    spesifikasi: str,
    standardisasi: bool = True,
) -> HasilSEM:
    """Estimasi model dengan maximum likelihood.

    Data distandardisasi secara bawaan karena variabel bersatuan sangat berbeda
    (rupiah berdampingan dengan skala 1–5) membuat optimasi sulit konvergen dan
    koefisiennya tidak sebanding satu sama lain.
    """
    teks = spesifikasi.strip()
    if not teks:
        raise ValueError("Spesifikasi model masih kosong.")

    variabel = _variabel_dalam(teks)
    hilang = [v for v in variabel if v not in df.columns]
    if hilang:
        raise ValueError(f"Variabel tidak ada dalam data: {', '.join(hilang)}.")

    data = preprocessing.clean_subset(df, variabel)
    if len(data) < len(variabel) * 5:
        catatan_sampel = (
            f"Ukuran sampel {len(data)} tergolong kecil untuk {len(variabel)} variabel; "
            "estimasi SEM lazim menuntut minimal 5–10 observasi per variabel."
        )
    else:
        catatan_sampel = ""

    kerja = preprocessing.scale(data, "z-score") if standardisasi else data
    model = semopy.Model(teks)
    model.fit(kerja)

    mentah = model.inspect(std_est=True)
    laten = set(_laten_dalam(teks))
    estimasi = pd.DataFrame(
        {
            "Dari": mentah["rval"],
            "Ke": mentah["lval"],
            "Operator": mentah["op"],
            "Jenis": [
                _jenis_baris(lv, op, rv, laten)
                for lv, op, rv in zip(mentah["lval"], mentah["op"], mentah["rval"])
            ],
            "Estimasi": pd.to_numeric(mentah["Estimate"], errors="coerce"),
            "Estimasi baku": pd.to_numeric(mentah["Est. Std"], errors="coerce"),
            "Std. Error": pd.to_numeric(mentah["Std. Err"], errors="coerce"),
            "z": pd.to_numeric(mentah["z-value"], errors="coerce"),
            "p-value": pd.to_numeric(mentah["p-value"], errors="coerce"),
        }
    )
    estimasi["Signifikan"] = np.where(
        estimasi["p-value"].isna(), "Acuan", np.where(estimasi["p-value"] < 0.05, "Ya", "Tidak")
    )

    statistik = semopy.calc_stats(model).iloc[0]
    catatan = [catatan_sampel] if catatan_sampel else []

    return HasilSEM(
        model=model,
        spesifikasi=teks,
        estimasi=estimasi,
        statistik=statistik,
        laten=sorted(laten),
        teramati=[v for v in variabel],
        n=len(data),
        catatan=catatan,
    )


def _laten_dalam(spesifikasi: str) -> list[str]:
    """Nama konstruk laten, yaitu ruas kiri dari baris berlambang '=~'."""
    laten = []
    for baris in spesifikasi.splitlines():
        if "=~" in baris:
            laten.append(baris.split("=~")[0].strip())
    return laten


def _variabel_dalam(spesifikasi: str) -> list[str]:
    """Variabel teramati: seluruh nama pada spesifikasi selain konstruk laten."""
    laten = set(_laten_dalam(spesifikasi))
    nama: list[str] = []
    for baris in spesifikasi.splitlines():
        teks = baris.strip()
        if not teks or teks.startswith("#"):
            continue
        for penanda in ("=~", "~~", "~"):
            if penanda in teks:
                kiri, kanan = teks.split(penanda, 1)
                bagian = [kiri] + kanan.split("+")
                break
        else:
            continue
        for potongan in bagian:
            bersih = potongan.strip()
            if bersih and bersih not in laten and bersih not in nama:
                nama.append(bersih)
    return nama


# --------------------------------------------------------------------------- #
# Pelaporan
# --------------------------------------------------------------------------- #


def tabel_kecocokan(hasil: HasilSEM) -> pd.DataFrame:
    """Indeks kecocokan model beserta ambang dan keputusannya."""
    s = hasil.statistik
    chi2 = float(s["chi2"])
    dof = float(s["DoF"])
    rasio = chi2 / dof if dof else float("nan")
    nilai = {
        "chi2/df": rasio,
        "CFI": float(s["CFI"]),
        "TLI": float(s["TLI"]),
        "RMSEA": float(s["RMSEA"]),
        "GFI": float(s["GFI"]),
        "AGFI": float(s["AGFI"]),
        "NFI": float(s["NFI"]),
    }
    baris = [
        {
            "Indeks": "Chi-square",
            "Nilai": chi2,
            "Kriteria": "p > 0,05",
            "Rujukan": "—",
            "Keputusan": "Memenuhi" if float(s["chi2 p-value"]) > 0.05 else "Tidak memenuhi",
            "Keterangan": f"df = {int(dof)}; p = {float(s['chi2 p-value']):.4f}",
        }
    ]
    for indeks, angka in nilai.items():
        kriteria, rujukan = AMBANG_FIT[indeks]
        if indeks in ("chi2/df", "RMSEA"):
            batas = 3.0 if indeks == "chi2/df" else 0.08
            memenuhi = np.isfinite(angka) and angka < batas
        else:
            memenuhi = np.isfinite(angka) and angka > 0.90
        baris.append(
            {
                "Indeks": indeks,
                "Nilai": angka,
                "Kriteria": kriteria,
                "Rujukan": rujukan,
                "Keputusan": "Memenuhi" if memenuhi else "Tidak memenuhi",
                "Keterangan": "",
            }
        )
    return pd.DataFrame(baris)


def catatan_chi_square(hasil: HasilSEM) -> str:
    """Kalimat yang perlu ada di naskah bila uji chi-square signifikan."""
    p = float(hasil.statistik["chi2 p-value"])
    if p > 0.05:
        return ""
    return (
        f"Uji chi-square signifikan (p = {p:.4f}), sehingga secara ketat hipotesis "
        "kecocokan sempurna ditolak. Uji ini diketahui sangat peka terhadap ukuran "
        f"sampel dan hampir selalu signifikan pada N > 200 (di sini N = {hasil.n}), "
        "sehingga penilaian kecocokan bertumpu pada indeks pendamping. Laporkan "
        "keduanya — melaporkan hanya indeks yang lolos adalah praktik yang lazim "
        "dikritik penelaah."
    )


def reliabilitas_konstruk(hasil: HasilSEM) -> pd.DataFrame:
    """CR dan AVE tiap konstruk laten, dihitung dari muatan terstandardisasi."""
    muatan = hasil.muatan()
    baris = []
    for konstruk in hasil.laten:
        lam = muatan.loc[muatan["Dari"] == konstruk, "Estimasi baku"].to_numpy(dtype=float)
        lam = lam[np.isfinite(lam)]
        if lam.size < 2:
            continue
        terbalik = int((lam < 0).sum())
        # Muatan negatif berasal dari butir berarah terbalik; nilai mutlaknya dipakai
        # agar jumlah muatan tidak saling meniadakan dan CR tidak jatuh keliru.
        besar = np.abs(lam)
        galat = 1 - besar**2
        penyebut = besar.sum() ** 2 + galat.sum()
        cr = float(besar.sum() ** 2 / penyebut) if penyebut else float("nan")
        ave = float((besar**2).mean())
        baris.append(
            {
                "Konstruk": konstruk,
                "Jumlah butir": int(lam.size),
                "Muatan minimum": float(np.min(besar)),
                "Butir arah terbalik": terbalik,
                "CR": cr,
                "AVE": ave,
                "√AVE": float(np.sqrt(ave)),
                "Keputusan": "Memenuhi" if cr >= 0.7 and ave >= 0.5 else "Belum memenuhi",
            }
        )
    return pd.DataFrame(baris)


def efek_langsung_tidak_langsung(
    hasil: HasilSEM, x: str, m: str, y: str
) -> dict[str, float]:
    """Dekomposisi efek X → M → Y dari koefisien baku model."""
    jalur = hasil.jalur().set_index(["Dari", "Ke"])

    def _ambil(dari: str, ke: str) -> float:
        try:
            return float(jalur.loc[(dari, ke), "Estimasi baku"])
        except KeyError:
            return float("nan")

    a = _ambil(x, m)
    b = _ambil(m, y)
    langsung = _ambil(x, y)
    tidak_langsung = a * b
    total = (0.0 if not np.isfinite(langsung) else langsung) + tidak_langsung
    vaf = tidak_langsung / total * 100 if total else float("nan")
    return {
        "a (X→M)": a,
        "b (M→Y)": b,
        "Efek langsung (X→Y)": langsung,
        "Efek tidak langsung": tidak_langsung,
        "Efek total": total,
        "VAF (%)": vaf,
    }


def bootstrap_mediasi(
    df: pd.DataFrame,
    spesifikasi: str,
    x: str,
    m: str,
    y: str,
    n_boot: int = 200,
    seed: int = 0,
    standardisasi: bool = True,
) -> pd.DataFrame:
    """Uji mediasi dengan bootstrap: interval kepercayaan efek tidak langsung.

    Interval bootstrap dipakai karena sebaran hasil kali dua koefisien (a×b) tidak
    normal, sehingga uji Sobel yang mengandaikan normalitas cenderung keliru.
    """
    variabel = _variabel_dalam(spesifikasi)
    data = preprocessing.clean_subset(df, variabel)
    rng = np.random.default_rng(seed)
    nilai: list[float] = []

    for _ in range(int(n_boot)):
        contoh = data.iloc[rng.integers(0, len(data), len(data))]
        try:
            ulang = jalankan(contoh, spesifikasi, standardisasi=standardisasi)
            efek = efek_langsung_tidak_langsung(ulang, x, m, y)
        except Exception:  # noqa: BLE001 - resample yang gagal konvergen dilewati
            continue
        if np.isfinite(efek["Efek tidak langsung"]):
            nilai.append(efek["Efek tidak langsung"])

    if len(nilai) < 20:
        raise ValueError(
            "Bootstrap gagal: terlalu sedikit resample yang konvergen. Sederhanakan "
            "model atau perbesar ukuran sampel."
        )

    asli = jalankan(df, spesifikasi, standardisasi=standardisasi)
    efek = efek_langsung_tidak_langsung(asli, x, m, y)
    bawah, atas = np.percentile(nilai, [2.5, 97.5])
    signifikan = not (bawah <= 0 <= atas)
    vaf = efek["VAF (%)"]
    if not signifikan:
        bentuk = "Tidak ada mediasi"
    elif np.isfinite(efek["Efek langsung (X→Y)"]) and abs(efek["Efek langsung (X→Y)"]) > 1e-9:
        bentuk = "Mediasi parsial" if vaf < 80 else "Mediasi hampir penuh"
    else:
        bentuk = "Mediasi penuh (jalur langsung tidak diestimasi)"

    return pd.DataFrame(
        [
            {
                "Jalur": f"{x} → {m} → {y}",
                "Efek langsung": efek["Efek langsung (X→Y)"],
                "Efek tidak langsung": efek["Efek tidak langsung"],
                "Efek total": efek["Efek total"],
                "SE bootstrap": float(np.std(nilai, ddof=1)),
                "IK 95% Bawah": float(bawah),
                "IK 95% Atas": float(atas),
                "VAF (%)": vaf,
                "Resample berhasil": len(nilai),
                "Keputusan": bentuk,
            }
        ]
    )
