"""Regresi data panel: pengaruh tetap dan pengaruh acak, beserta uji Hausman.

Data panel (mis. beberapa perusahaan diamati beberapa tahun berturut-turut)
melanggar independensi standar OLS: pengamatan pada entitas yang sama lebih
mirip satu sama lain daripada pada entitas berbeda. Dua model mengatasinya
dengan cara berbeda:

* **Pengaruh tetap (fixed effects)** memberi tiap entitas intersepnya sendiri,
  sehingga apa pun yang khas pada entitas itu (dan tidak berubah sepanjang
  waktu) ikut terkendalikan tanpa perlu diukur. Diestimasi di sini lewat
  variabel dummy entitas (LSDV) — secara aljabar identik dengan estimator
  within (demeaning), tetapi galat baku dan derajat bebasnya otomatis benar
  karena dihitung dari model OLS biasa, bukan dari data yang sudah dipusatkan
  secara manual.
* **Pengaruh acak (random effects)** memperlakukan efek entitas sebagai
  bagian dari galat, bukan parameter yang diestimasi terpisah — lebih efisien
  bila layak dipakai. Statsmodels tidak menyediakan estimator RE-GLS klasik
  (Swamy-Arora) di luar paket ``linearmodels``, sehingga di sini dipakai model
  campuran (``MixedLM``) berintersep acak per entitas sebagai penggantinya —
  pendekatan yang lazim dipakai ketika ``linearmodels`` tidak tersedia, dan
  dinyatakan begitu adanya pada catatan hasil, bukan disamarkan sebagai
  RE-GLS murni.

**Uji Hausman** memutuskan di antara keduanya: bila efek entitas berkorelasi
dengan prediktor, pengaruh acak menjadi tidak konsisten dan pengaruh tetap
wajib dipakai; bila tidak, pengaruh acak lebih efisien.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from nalardata import preprocessing


@dataclass
class HasilPanel:
    """Hasil regresi panel: model pengaruh tetap dan pengaruh acak berdampingan."""

    fe: sm.regression.linear_model.RegressionResultsWrapper
    re: object  # MixedLMResultsWrapper — diketik longgar, statsmodels tidak mengekspor tipenya
    y: str
    prediktor: list[str]
    entitas: str
    waktu: str | None
    efek_waktu: bool
    n_entitas: int
    n_observasi: int
    koefisien_fe: pd.DataFrame
    koefisien_re: pd.DataFrame
    r2_within: float

    def hausman(self) -> dict[str, float]:
        """Statistik Hausman: H0 — pengaruh acak konsisten (dan lebih efisien)."""
        b_fe = self.fe.params.loc[self.prediktor].to_numpy()
        b_re = self.re.fe_params.loc[self.prediktor].to_numpy()
        v_fe = self.fe.cov_params().loc[self.prediktor, self.prediktor].to_numpy()
        v_re = self.re.cov_params().loc[self.prediktor, self.prediktor].to_numpy()
        beda = b_fe - b_re
        v_beda = v_fe - v_re
        df = len(self.prediktor)

        try:
            stat = float(beda @ np.linalg.pinv(v_beda) @ beda)
        except np.linalg.LinAlgError:
            return {"statistik": float("nan"), "df": df, "p-value": float("nan")}
        if not np.isfinite(stat) or stat < 0:
            return {"statistik": float("nan"), "df": df, "p-value": float("nan")}
        return {"statistik": stat, "df": df, "p-value": float(stats.chi2.sf(stat, df))}

    def kesimpulan(self, alpha: float = 0.05) -> str:
        h = self.hausman()
        if not np.isfinite(h["p-value"]):
            return (
                "Uji Hausman tidak dapat dihitung pada data ini (selisih matriks "
                "ragam kedua model tidak dapat dibalik). Pengaruh tetap adalah "
                "pilihan yang lebih aman di sini — ia tetap konsisten baik "
                "pengaruh acak layak dipakai atau tidak."
            )
        if h["p-value"] < alpha:
            return (
                f"Uji Hausman signifikan (χ² = {h['statistik']:.2f}, "
                f"p = {h['p-value']:.3f}): efek entitas kemungkinan berkorelasi "
                "dengan prediktor. Pakai model **pengaruh tetap** — pengaruh acak "
                "tidak konsisten pada kondisi ini."
            )
        return (
            f"Uji Hausman tidak signifikan (χ² = {h['statistik']:.2f}, "
            f"p = {h['p-value']:.3f}): tidak ada bukti efek entitas berkorelasi "
            "dengan prediktor. Model **pengaruh acak** dapat dipakai dan lebih "
            "efisien."
        )


def _rapikan_koefisien(model, nama_baris: list[str]) -> pd.DataFrame:
    ik = model.conf_int()
    return pd.DataFrame(
        {
            "Variabel": nama_baris,
            "B": model.params.loc[nama_baris].to_numpy(),
            "Std. Error": model.bse.loc[nama_baris].to_numpy(),
            "t / z": model.tvalues.loc[nama_baris].to_numpy(),
            "p-value": model.pvalues.loc[nama_baris].to_numpy(),
            "IK 95% Bawah": ik.loc[nama_baris].iloc[:, 0].to_numpy(),
            "IK 95% Atas": ik.loc[nama_baris].iloc[:, 1].to_numpy(),
            "Signifikan": [
                "Ya" if p < 0.05 else "Tidak" for p in model.pvalues.loc[nama_baris]
            ],
        }
    )


def _r2_within(data: pd.DataFrame, y: str, prediktor: list[str], entitas: str, params) -> float:
    """R² dari bagian dalam-entitas saja (within) — ukuran kecocokan yang lazim
    dilaporkan untuk pengaruh tetap, berbeda dari R² keseluruhan model dummy
    (LSDV) yang ikut menghitung daya jelas dummy entitasnya sendiri."""
    y_demean = data[y] - data.groupby(entitas)[y].transform("mean")
    duga = sum(
        float(params[p]) * (data[p] - data.groupby(entitas)[p].transform("mean"))
        for p in prediktor
    )
    residu = y_demean - duga
    sst = float((y_demean**2).sum())
    if sst <= 0:
        return float("nan")
    return 1.0 - float((residu**2).sum()) / sst


def regresi_panel(
    df: pd.DataFrame,
    y: str,
    prediktor: list[str],
    entitas: str,
    waktu: str | None = None,
) -> HasilPanel:
    """Estimasi pengaruh tetap (LSDV) dan pengaruh acak (intersep acak per
    entitas) sekaligus, agar keduanya dapat dibandingkan lewat uji Hausman."""
    kolom = [y, entitas, *prediktor] + ([waktu] if waktu else [])
    hilang = [k for k in kolom if k not in df.columns]
    if hilang:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(hilang)}.")
    if not prediktor:
        raise ValueError("Perlu sekurang-kurangnya satu prediktor.")

    data = preprocessing.clean_subset(df, kolom).copy()
    data[entitas] = data[entitas].astype(str)
    n_entitas = int(data[entitas].nunique())
    if n_entitas < 2:
        raise ValueError("Data panel memerlukan sekurang-kurangnya dua entitas.")
    if len(data) < len(prediktor) + n_entitas + 5:
        raise ValueError("Observasi lengkap terlalu sedikit untuk regresi panel ini.")

    dummy_entitas = pd.get_dummies(data[entitas], prefix="entitas", drop_first=True)
    bagian_fe = [data[prediktor].astype(float), dummy_entitas.astype(float)]
    bagian_re = [data[prediktor].astype(float)]
    efek_waktu = bool(waktu)
    if waktu:
        # Dummy waktu ikut disertakan pada KEDUA model, bukan hanya FE: uji Hausman
        # membandingkan koefisien prediktor pada bagian tetap kedua model, dan
        # perbandingan itu hanya sahih bila keduanya mengendalikan hal yang sama.
        # Tanpa ini, selisih koefisien bisa muncul semata karena RE tidak
        # mengendalikan tren waktu yang justru dikendalikan FE, bukan karena
        # efek entitas benar-benar berkorelasi dengan prediktor.
        dummy_waktu = pd.get_dummies(data[waktu].astype(str), prefix="waktu", drop_first=True)
        bagian_fe.append(dummy_waktu.astype(float))
        bagian_re.append(dummy_waktu.astype(float))
    X_fe = sm.add_constant(pd.concat(bagian_fe, axis=1), has_constant="add")
    fe = sm.OLS(data[y].astype(float), X_fe).fit()

    X_re = sm.add_constant(pd.concat(bagian_re, axis=1), has_constant="add")
    re = sm.MixedLM(data[y].astype(float), X_re, groups=data[entitas]).fit(reml=False)

    baris = ["const", *prediktor]
    return HasilPanel(
        fe=fe,
        re=re,
        y=y,
        prediktor=list(prediktor),
        entitas=entitas,
        waktu=waktu,
        efek_waktu=efek_waktu,
        n_entitas=n_entitas,
        n_observasi=len(data),
        koefisien_fe=_rapikan_koefisien(fe, baris),
        koefisien_re=_rapikan_koefisien(re, baris),
        r2_within=_r2_within(data, y, prediktor, entitas, fe.params),
    )
