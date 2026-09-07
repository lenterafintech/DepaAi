"""PLS-SEM (Partial Least Squares Structural Equation Modeling).

Pelengkap CB-SEM/lavaan-style yang sudah ada di ``sem_analysis.py`` — bukan
penggantinya. CB-SEM (dipakai `sem_analysis.jalankan`) mengestimasi model dengan
memaksimalkan kecocokan matriks kovarians teramati dengan yang tersirat model,
menuntut sampel relatif besar dan indikator berdistribusi mendekati normal.
PLS-SEM memaksimalkan varians yang dijelaskan (mirip regresi berantai),
sehingga tetap dapat diestimasi pada sampel kecil, model kompleks, atau
konstruk formatif — dengan harga tidak ada uji kecocokan model keseluruhan
seperti chi-square/CFI/RMSEA.

Hanya indikator **reflektif** (Mode A) yang didukung: tiap indikator dianggap
akibat dari konstruknya, cocok untuk mayoritas kuesioner sikap/persepsi. Model
formatif (Mode B, indikator membentuk konstruknya) sengaja belum didukung —
salah menandai arah pengukuran formatif sebagai reflektif adalah kekeliruan
metodologis yang sering luput, dan lebih aman menolaknya eksplisit daripada
diam-diam mengestimasi dengan asumsi yang salah.

Algoritma mengikuti skema pembobotan **jalur (path weighting scheme)** —
skema baku SmartPLS dan `plspm` — supaya angkanya sebisa mungkin sebanding
ketika pengguna memeriksa ulang di perangkat itu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nalardata import preprocessing, reliability as rb

POLA_NAMA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _periksa_nama(nama: str, peran: str = "Nama") -> None:
    if not POLA_NAMA.match(str(nama)):
        raise ValueError(
            f"{peran} '{nama}' tidak dapat dipakai: gunakan huruf, angka, dan garis "
            "bawah tanpa spasi (misalnya 'kualitas_layanan')."
        )


def _periksa_spesifikasi(
    df: pd.DataFrame, konstruk: dict[str, list[str]], jalur: dict[str, list[str]]
) -> None:
    if len(konstruk) < 2:
        raise ValueError("PLS-SEM memerlukan minimal 2 konstruk.")
    for nama, indikator in konstruk.items():
        _periksa_nama(nama, "Nama konstruk")
        if len(indikator) < 1:
            raise ValueError(f"Konstruk '{nama}' memerlukan minimal 1 indikator.")
        hilang = [i for i in indikator if i not in df.columns]
        if hilang:
            raise ValueError(f"Indikator tidak ditemukan pada data: {', '.join(hilang)}.")
    if not jalur:
        raise ValueError("Model jalur memerlukan minimal satu hubungan struktural.")
    for terikat, sumber in jalur.items():
        if terikat not in konstruk:
            raise ValueError(f"'{terikat}' pada model jalur bukan nama konstruk yang ada.")
        for s in sumber:
            if s not in konstruk:
                raise ValueError(f"'{s}' pada model jalur bukan nama konstruk yang ada.")
            if s == terikat:
                raise ValueError(f"Konstruk '{terikat}' tidak boleh menjadi prediktor dirinya sendiri.")


def _tetangga(jalur: dict[str, list[str]], konstruk_names: list[str]) -> dict[str, set[str]]:
    """Konstruk yang terhubung ke tiap konstruk lewat jalur manapun (dua arah)."""
    tetangga: dict[str, set[str]] = {k: set() for k in konstruk_names}
    for terikat, sumber in jalur.items():
        for s in sumber:
            tetangga[terikat].add(s)
            tetangga[s].add(terikat)
    return tetangga


def _standarkan(x: np.ndarray) -> np.ndarray:
    pusat = x - x.mean(axis=0, keepdims=True)
    sd = pusat.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return pusat / sd


@dataclass
class HasilPLS:
    """Estimasi PLS-SEM: model pengukuran (bobot, muatan) dan model struktural (jalur)."""

    konstruk: dict[str, list[str]]
    jalur: dict[str, list[str]]
    n: int
    bobot: pd.DataFrame  # Konstruk, Indikator, Bobot
    muatan: pd.DataFrame  # Konstruk, Indikator, Muatan
    skor: pd.DataFrame  # n x k, satu kolom per konstruk (standar)
    jalur_koefisien: pd.DataFrame  # Terikat, Sumber, Beta
    r2: pd.Series  # indeks = konstruk endogen
    iterasi: int
    konvergen: bool

    def muatan_konstruk(self, nama: str) -> pd.Series:
        sub = self.muatan[self.muatan["Konstruk"] == nama]
        return pd.Series(sub["Muatan"].to_numpy(), index=sub["Indikator"].to_numpy())

    def cr_ave(self) -> pd.DataFrame:
        """Composite reliability dan AVE tiap konstruk, dari muatan yang diestimasi."""
        baris = []
        for nama in self.konstruk:
            muatan = self.muatan_konstruk(nama)
            if len(muatan) < 2:
                # Konstruk berindikator tunggal: CR/AVE tidak bermakna (selalu 1).
                baris.append({"Konstruk": nama, "CR": 1.0, "AVE": 1.0})
                continue
            cr, ave = rb.cr_ave(muatan)
            baris.append({"Konstruk": nama, "CR": cr, "AVE": ave})
        return pd.DataFrame(baris)


def _estimasi(
    df: pd.DataFrame,
    konstruk: dict[str, list[str]],
    jalur: dict[str, list[str]],
    maks_iter: int = 300,
    toleransi: float = 1e-7,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, bool]:
    """Inti algoritma PLS: kembalikan bobot akhir, skor konstruk, iterasi, dan status konvergen."""
    nama_konstruk = list(konstruk)
    x = {k: _standarkan(df[indikator].to_numpy(float)) for k, indikator in konstruk.items()}
    tetangga = _tetangga(jalur, nama_konstruk)

    # Bobot awal: rata-rata sederhana tiap indikator (bobot seragam, lalu dinormalkan).
    bobot = {k: np.ones(x[k].shape[1]) / x[k].shape[1] for k in nama_konstruk}

    def skor_dari_bobot(b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: _standarkan((x[k] @ b[k]).reshape(-1, 1)).ravel() for k in nama_konstruk}

    y = skor_dari_bobot(bobot)
    konvergen = False
    iterasi = 0
    for iterasi in range(1, maks_iter + 1):
        # --- Estimasi dalam (inner estimation), skema pembobotan jalur ---
        z: dict[str, np.ndarray] = {}
        for k in nama_konstruk:
            sumber_k = jalur.get(k, [])  # konstruk yang menjadi prediktor k
            pengikut_k = [t for t, s in jalur.items() if k in s]  # konstruk yang diprediksi k
            kontribusi = np.zeros(len(df))
            if sumber_k:
                mat = np.column_stack([y[s] for s in sumber_k])
                beta, *_ = np.linalg.lstsq(mat, y[k], rcond=None)
                kontribusi = kontribusi + mat @ beta
            for t in pengikut_k:
                r = float(np.corrcoef(y[k], y[t])[0, 1])
                kontribusi = kontribusi + r * y[t]
            if not sumber_k and not pengikut_k:
                kontribusi = y[k]  # konstruk terisolasi: pertahankan skor apa adanya
            z[k] = _standarkan(kontribusi.reshape(-1, 1)).ravel() if np.std(kontribusi) > 0 else kontribusi

        # --- Estimasi luar (outer estimation), Mode A reflektif ---
        bobot_baru: dict[str, np.ndarray] = {}
        for k in nama_konstruk:
            xk = x[k]
            if xk.shape[1] == 1:
                bobot_baru[k] = np.array([1.0])
                continue
            w = (xk.T @ z[k]) / len(df)
            skala = np.std(xk @ w, ddof=1)
            bobot_baru[k] = w / skala if skala > 0 else w

        selisih = max(
            float(np.max(np.abs(bobot_baru[k] - bobot[k]))) if bobot[k].shape == bobot_baru[k].shape else 1.0
            for k in nama_konstruk
        )
        bobot = bobot_baru
        y = skor_dari_bobot(bobot)
        if selisih < toleransi:
            konvergen = True
            break

    return bobot, y, iterasi, konvergen


def jalankan(
    df: pd.DataFrame,
    konstruk: dict[str, list[str]],
    jalur: dict[str, list[str]],
    maks_iter: int = 300,
    toleransi: float = 1e-7,
) -> HasilPLS:
    """Estimasi PLS-SEM: bobot dan muatan model pengukuran, koefisien model struktural."""
    _periksa_spesifikasi(df, konstruk, jalur)
    semua_indikator = [i for indikator in konstruk.values() for i in indikator]
    bersih = preprocessing.clean_subset(df, semua_indikator)
    if len(bersih) < len(konstruk) + 5:
        raise ValueError("Observasi lengkap terlalu sedikit untuk jumlah konstruk ini.")

    bobot, skor, iterasi, konvergen = _estimasi(bersih, konstruk, jalur, maks_iter, toleransi)

    baris_bobot, baris_muatan = [], []
    for k, indikator in konstruk.items():
        xk = _standarkan(bersih[indikator].to_numpy(float))
        for i, nama_ind in enumerate(indikator):
            baris_bobot.append({"Konstruk": k, "Indikator": nama_ind, "Bobot": float(bobot[k][i])})
            muatan_i = float(np.corrcoef(xk[:, i], skor[k])[0, 1])
            baris_muatan.append({"Konstruk": k, "Indikator": nama_ind, "Muatan": muatan_i})

    baris_jalur = []
    r2 = {}
    for terikat, sumber in jalur.items():
        mat = np.column_stack([skor[s] for s in sumber])
        beta, *_ = np.linalg.lstsq(mat, skor[terikat], rcond=None)
        prediksi = mat @ beta
        ss_total = float(np.sum((skor[terikat] - skor[terikat].mean()) ** 2))
        ss_sisa = float(np.sum((skor[terikat] - prediksi) ** 2))
        r2[terikat] = 1 - ss_sisa / ss_total if ss_total > 0 else float("nan")
        for s, b in zip(sumber, beta):
            baris_jalur.append({"Terikat": terikat, "Sumber": s, "Beta": float(b)})

    return HasilPLS(
        konstruk=dict(konstruk),
        jalur=dict(jalur),
        n=len(bersih),
        bobot=pd.DataFrame(baris_bobot),
        muatan=pd.DataFrame(baris_muatan),
        skor=pd.DataFrame(skor, index=bersih.index),
        jalur_koefisien=pd.DataFrame(baris_jalur),
        r2=pd.Series(r2),
        iterasi=iterasi,
        konvergen=konvergen,
    )


def bootstrap_jalur(
    df: pd.DataFrame,
    konstruk: dict[str, list[str]],
    jalur: dict[str, list[str]],
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Signifikansi koefisien jalur lewat bootstrap — PLS-SEM tidak mengasumsikan
    sebaran tertentu, sehingga tidak punya uji-t analitik seperti CB-SEM/regresi."""
    asli = jalankan(df, konstruk, jalur)
    semua_indikator = [i for indikator in konstruk.values() for i in indikator]
    bersih = preprocessing.clean_subset(df, semua_indikator)
    rng = np.random.default_rng(seed)

    tumpuk: dict[tuple[str, str], list[float]] = {
        (b["Terikat"], b["Sumber"]): [] for b in asli.jalur_koefisien.to_dict("records")
    }
    gagal = 0
    for _ in range(int(n_boot)):
        contoh = bersih.iloc[rng.integers(0, len(bersih), len(bersih))].reset_index(drop=True)
        try:
            ulang = jalankan(contoh, konstruk, jalur)
        except Exception:  # noqa: BLE001 - resample yang gagal konvergen/singular dilewati
            gagal += 1
            continue
        for baris in ulang.jalur_koefisien.to_dict("records"):
            tumpuk[(baris["Terikat"], baris["Sumber"])].append(baris["Beta"])

    hasil = []
    for baris in asli.jalur_koefisien.to_dict("records"):
        nilai = np.array(tumpuk[(baris["Terikat"], baris["Sumber"])])
        if len(nilai) < 30:
            hasil.append(
                {
                    "Terikat": baris["Terikat"],
                    "Sumber": baris["Sumber"],
                    "Beta": baris["Beta"],
                    "SE Bootstrap": float("nan"),
                    "p-value": float("nan"),
                    "IK 95% Bawah": float("nan"),
                    "IK 95% Atas": float("nan"),
                    "Signifikan": "Tidak dapat ditentukan",
                }
            )
            continue
        se = float(nilai.std(ddof=1))
        t = baris["Beta"] / se if se > 0 else float("nan")
        p = float(2 * (1 - _cdf_normal_baku(abs(t)))) if np.isfinite(t) else float("nan")
        hasil.append(
            {
                "Terikat": baris["Terikat"],
                "Sumber": baris["Sumber"],
                "Beta": baris["Beta"],
                "SE Bootstrap": se,
                "p-value": p,
                "IK 95% Bawah": float(np.percentile(nilai, 2.5)),
                "IK 95% Atas": float(np.percentile(nilai, 97.5)),
                "Signifikan": "Ya" if np.isfinite(p) and p < 0.05 else "Tidak",
            }
        )
    return pd.DataFrame(hasil)


def _cdf_normal_baku(z: float) -> float:
    from scipy import stats

    return float(stats.norm.cdf(z))
