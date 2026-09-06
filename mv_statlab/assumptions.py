"""Uji asumsi yang mendasari analisis multivariat.

Berisi uji kelayakan data untuk analisis faktor (KMO, Bartlett), diagnosis
multikolinearitas (VIF), dan uji homogenitas matriks kovarians (Box's M).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BartlettResult:
    chi_square: float
    df: int
    p_value: float

    @property
    def adequate(self) -> bool:
        return self.p_value < 0.05

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Uji": "Bartlett's Test of Sphericity",
                    "Chi-square": self.chi_square,
                    "df": self.df,
                    "p-value": self.p_value,
                    "Kesimpulan": (
                        "Layak difaktorkan (p < 0,05)"
                        if self.adequate
                        else "Belum layak difaktorkan (p >= 0,05)"
                    ),
                }
            ]
        )


def bartlett_sphericity(df: pd.DataFrame) -> BartlettResult:
    """Uji Bartlett: apakah matriks korelasi berbeda dari matriks identitas."""
    X = df.select_dtypes(include=np.number).dropna()
    n, p = X.shape
    if p < 2:
        raise ValueError("Uji Bartlett memerlukan minimal 2 variabel.")
    R = np.corrcoef(X.to_numpy(float), rowvar=False)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        logdet = np.log(max(np.linalg.det(R), 1e-12))
    chi2 = -((n - 1) - (2 * p + 5) / 6.0) * logdet
    dof = p * (p - 1) // 2
    return BartlettResult(
        chi_square=float(chi2), df=int(dof), p_value=float(stats.chi2.sf(chi2, dof))
    )


@dataclass
class KMOResult:
    overall: float
    per_variable: pd.Series

    @property
    def interpretation(self) -> str:
        v = self.overall
        if v >= 0.9:
            return "Sangat baik (marvelous)"
        if v >= 0.8:
            return "Baik (meritorious)"
        if v >= 0.7:
            return "Cukup (middling)"
        if v >= 0.6:
            return "Sedang (mediocre)"
        if v >= 0.5:
            return "Buruk (miserable)"
        return "Tidak layak (unacceptable)"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Variabel": self.per_variable.index,
                "MSA": self.per_variable.to_numpy(),
                "Kelayakan": np.where(
                    self.per_variable.to_numpy() >= 0.5, "Layak", "Pertimbangkan dibuang"
                ),
            }
        )


def kmo(df: pd.DataFrame) -> KMOResult:
    """Kaiser-Meyer-Olkin measure of sampling adequacy (keseluruhan dan per variabel)."""
    X = df.select_dtypes(include=np.number).dropna()
    if X.shape[1] < 2:
        raise ValueError("KMO memerlukan minimal 2 variabel.")
    R = np.corrcoef(X.to_numpy(float), rowvar=False)
    R_inv = np.linalg.pinv(R)
    d = np.sqrt(np.diag(R_inv))
    partial = -R_inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    corr = R.copy()
    np.fill_diagonal(corr, 0.0)

    corr_sq = (corr**2).sum()
    partial_sq = (partial**2).sum()
    overall = corr_sq / (corr_sq + partial_sq)

    col_corr = (corr**2).sum(axis=0)
    col_partial = (partial**2).sum(axis=0)
    msa = col_corr / (col_corr + col_partial)
    return KMOResult(
        overall=float(overall), per_variable=pd.Series(msa, index=X.columns, name="MSA")
    )


def vif(df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor tiap variabel (deteksi multikolinearitas)."""
    X = df.select_dtypes(include=np.number).dropna().astype(float)
    if X.shape[1] < 2:
        raise ValueError("VIF memerlukan minimal 2 variabel.")
    rows = []
    for col in X.columns:
        y = X[col].to_numpy()
        others = X.drop(columns=[col]).to_numpy()
        A = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else 0.0
        value = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)
        rows.append(
            {
                "Variabel": col,
                "R2": float(r2),
                "VIF": float(value),
                "Tolerance": float(1 - r2),
                "Status": "Multikolinear" if value >= 10 else "Aman",
            }
        )
    return pd.DataFrame(rows)


@dataclass
class BoxMResult:
    statistic: float
    chi_square: float
    df: int
    p_value: float

    @property
    def homogeneous(self) -> bool:
        return self.p_value > 0.05

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Uji": "Box's M",
                    "M": self.statistic,
                    "Chi-square": self.chi_square,
                    "df": self.df,
                    "p-value": self.p_value,
                    "Kesimpulan": (
                        "Matriks kovarians homogen"
                        if self.homogeneous
                        else "Matriks kovarians tidak homogen"
                    ),
                }
            ]
        )


def box_m(df: pd.DataFrame, group: pd.Series) -> BoxMResult:
    """Uji Box's M untuk homogenitas matriks kovarians antar kelompok."""
    X = df.select_dtypes(include=np.number)
    data = pd.concat([X, group.rename("__grup__")], axis=1).dropna()
    p = X.shape[1]
    groups = list(data["__grup__"].unique())
    g = len(groups)
    if g < 2:
        raise ValueError("Box's M memerlukan minimal 2 kelompok.")

    pooled = np.zeros((p, p))
    total_df = 0
    log_terms = 0.0
    inv_df_sum = 0.0
    for name in groups:
        Xi = data.loc[data["__grup__"] == name, X.columns].to_numpy(float)
        ni = len(Xi)
        if ni <= p:
            raise ValueError(
                f"Kelompok '{name}' hanya punya {ni} observasi untuk {p} variabel; "
                "Box's M butuh n > p di setiap kelompok."
            )
        Si = np.cov(Xi, rowvar=False)
        pooled += (ni - 1) * Si
        total_df += ni - 1
        sign, logdet = np.linalg.slogdet(Si)
        log_terms += (ni - 1) * logdet
        inv_df_sum += 1.0 / (ni - 1)

    pooled /= total_df
    _, log_pooled = np.linalg.slogdet(pooled)
    M = total_df * log_pooled - log_terms
    c = (inv_df_sum - 1.0 / total_df) * (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    chi2 = M * (1 - c)
    dof = (g - 1) * p * (p + 1) // 2
    return BoxMResult(
        statistic=float(M),
        chi_square=float(chi2),
        df=int(dof),
        p_value=float(stats.chi2.sf(chi2, dof)),
    )


def levene_by_variable(df: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Uji Levene per variabel untuk homogenitas varians antar kelompok."""
    X = df.select_dtypes(include=np.number)
    data = pd.concat([X, group.rename("__grup__")], axis=1).dropna()
    rows = []
    for col in X.columns:
        samples = [
            grp[col].to_numpy(float) for _, grp in data.groupby("__grup__", observed=True)
        ]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) < 2:
            continue
        stat, p = stats.levene(*samples, center="median")
        rows.append(
            {
                "Variabel": col,
                "Levene W": float(stat),
                "p-value": float(p),
                "Kesimpulan": "Homogen" if p > 0.05 else "Tidak homogen",
            }
        )
    return pd.DataFrame(rows)
