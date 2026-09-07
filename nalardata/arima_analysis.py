"""Deret waktu: uji stasioneritas (ADF), pencarian order otomatis, dan ARIMA.

ARIMA menuntut deret yang stasioner — rata-rata dan ragamnya tidak berubah
sepanjang waktu — sebelum pola autoregresif/rata-rata bergeraknya dapat
diestimasi dengan andal. Tiga langkah di sini mengikuti urutan yang lazim
diajarkan (Box-Jenkins), tetapi dijalankan otomatis alih-alih dibaca manual
dari plot ACF/PACF:

1. **Uji Augmented Dickey-Fuller (ADF)** pada level; bila belum stasioner,
   deret di-differencing dan diuji ulang, sampai stasioner atau mencapai
   batas differencing (``MAKS_D``).
2. **Pencarian order** ``(p, d, q)`` lewat grid sederhana yang meminimumkan
   AIC, dengan ``d`` sudah ditetapkan dari langkah 1 — bukan pencarian tiga
   dimensi penuh yang mahal.
3. **Fit ARIMA** pada order terpilih, dengan Ljung-Box pada residual sebagai
   pemeriksaan bahwa pola autokorelasi memang sudah tertangkap model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

ALFA = 0.05
MAKS_D = 2
MAKS_P = 3
MAKS_Q = 3


@dataclass
class HasilStasioneritas:
    """Hasil uji ADF, beserta berapa kali differencing sampai stasioner."""

    statistik: float
    p_value: float
    d: int
    stasioner: bool
    rincian: str


def uji_stasioneritas(series: pd.Series, maks_d: int = MAKS_D) -> HasilStasioneritas:
    """ADF pada level, differencing otomatis sampai stasioner atau ``maks_d`` tercapai."""
    nilai = pd.to_numeric(series, errors="coerce").dropna()
    if len(nilai) < 10:
        raise ValueError("Perlu sekurang-kurangnya 10 titik waktu untuk uji stasioneritas.")

    kerja = nilai.copy()
    d = 0
    while True:
        stat, p = adfuller(kerja, autolag="AIC", result_object=False)[:2]
        stasioner = bool(p < ALFA)
        if stasioner or d >= maks_d:
            rincian = f"ADF pada d={d}: statistik = {stat:.3f}, p = {p:.3f}."
            if not stasioner:
                rincian += f" Belum stasioner setelah differencing maksimum (d={maks_d})."
            return HasilStasioneritas(
                statistik=float(stat), p_value=float(p), d=d, stasioner=stasioner, rincian=rincian
            )
        kerja = kerja.diff().dropna()
        d += 1


def cari_order(
    series: pd.Series, d: int, maks_p: int = MAKS_P, maks_q: int = MAKS_Q
) -> tuple[int, int, int]:
    """Order ``(p, d, q)`` yang meminimumkan AIC pada grid ``p``/``q`` sederhana.

    ``d`` ditetapkan dari luar (biasanya dari :func:`uji_stasioneritas`) alih-alih
    ikut dicari, karena mencari ketiganya sekaligus jauh lebih mahal dan jarang
    mengubah kesimpulan dibanding menetapkan ``d`` dari ujinya sendiri.
    """
    nilai = pd.to_numeric(series, errors="coerce").dropna()
    terbaik: tuple[float, tuple[int, int, int]] | None = None
    for p in range(maks_p + 1):
        for q in range(maks_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(nilai, order=(p, d, q)).fit()
            except Exception:  # noqa: BLE001 - sebagian order gagal konvergen, dilewati
                continue
            if terbaik is None or model.aic < terbaik[0]:
                terbaik = (float(model.aic), (p, d, q))
    if terbaik is None:
        raise ValueError(
            "Tidak ditemukan order ARIMA yang dapat difit pada data ini — coba "
            "periksa kembali deretnya, atau kurangi batas p/q."
        )
    return terbaik[1]


@dataclass
class HasilARIMA:
    """Model ARIMA terfit, beserta cara membaca residual dan meramalkannya."""

    model: object  # ARIMAResultsWrapper — tidak diekspor tipenya oleh statsmodels
    order: tuple[int, int, int]
    aic: float
    bic: float
    n: int

    def ringkasan_koefisien(self) -> pd.DataFrame:
        ik = self.model.conf_int()
        return pd.DataFrame(
            {
                "Parameter": list(self.model.params.index),
                "Koefisien": self.model.params.to_numpy(),
                "Std. Error": self.model.bse.to_numpy(),
                "p-value": self.model.pvalues.to_numpy(),
                "IK 95% Bawah": ik.iloc[:, 0].to_numpy(),
                "IK 95% Atas": ik.iloc[:, 1].to_numpy(),
            }
        )

    def uji_ljung_box(self, lags: int = 10) -> pd.DataFrame:
        """Ljung-Box pada residual: H0 — tidak ada autokorelasi tersisa."""
        resid = self.model.resid
        batas = max(1, min(lags, len(resid) // 2 - 1))
        return acorr_ljungbox(resid, lags=[batas], return_df=True).reset_index(drop=True)

    def residual_bersih(self, alpha: float = ALFA) -> bool:
        """Apakah residualnya sudah bersih dari autokorelasi (Ljung-Box tidak signifikan)."""
        tabel = self.uji_ljung_box()
        if tabel.empty or "lb_pvalue" not in tabel.columns:
            return False
        return bool((tabel["lb_pvalue"] >= alpha).all())

    def ramalkan(self, langkah: int) -> pd.DataFrame:
        if langkah < 1:
            raise ValueError("Jumlah langkah peramalan harus sekurang-kurangnya satu.")
        prediksi = self.model.get_forecast(steps=langkah)
        ik = prediksi.conf_int(alpha=ALFA)
        return pd.DataFrame(
            {
                "Langkah": range(1, langkah + 1),
                "Perkiraan": np.asarray(prediksi.predicted_mean),
                "IK 95% Bawah": np.asarray(ik.iloc[:, 0]),
                "IK 95% Atas": np.asarray(ik.iloc[:, 1]),
            }
        )


def fit_arima(series: pd.Series, order: tuple[int, int, int] | None = None) -> HasilARIMA:
    """Fit ARIMA; bila ``order`` tidak diberikan, dicari otomatis lewat langkah
    1-2 pada docstring modul (ADF lalu grid AIC)."""
    nilai = pd.to_numeric(series, errors="coerce").dropna()
    if len(nilai) < 15:
        raise ValueError("Perlu sekurang-kurangnya 15 titik waktu untuk ARIMA.")

    if order is None:
        d = uji_stasioneritas(nilai).d
        order = cari_order(nilai, d)

    model = ARIMA(nilai, order=order).fit()
    return HasilARIMA(
        model=model, order=tuple(order), aic=float(model.aic), bic=float(model.bic), n=len(nilai)
    )
