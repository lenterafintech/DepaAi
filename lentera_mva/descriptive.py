"""Statistik deskriptif, uji normalitas, dan deteksi pencilan multivariat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Statistik deskriptif lengkap untuk kolom numerik."""
    num = df.select_dtypes(include=np.number)
    rows = []
    for col in num.columns:
        s = num[col].dropna()
        n = len(s)
        sd = float(s.std(ddof=1)) if n > 1 else np.nan
        rows.append(
            {
                "Variabel": col,
                "N": n,
                "Missing": int(num[col].isna().sum()),
                "Mean": float(s.mean()) if n else np.nan,
                "Std. Dev": sd,
                "Std. Error": sd / np.sqrt(n) if n > 1 else np.nan,
                "Min": float(s.min()) if n else np.nan,
                "Q1": float(s.quantile(0.25)) if n else np.nan,
                "Median": float(s.median()) if n else np.nan,
                "Q3": float(s.quantile(0.75)) if n else np.nan,
                "Max": float(s.max()) if n else np.nan,
                "Skewness": float(s.skew()) if n > 2 else np.nan,
                "Kurtosis": float(s.kurtosis()) if n > 3 else np.nan,
                "CV (%)": float(sd / s.mean() * 100) if n > 1 and s.mean() != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def frequency_table(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    return pd.DataFrame(
        {
            "Kategori": counts.index.astype(str),
            "Frekuensi": counts.to_numpy(),
            "Persen (%)": (counts / counts.sum() * 100).round(2).to_numpy(),
        }
    )


def normality_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Uji normalitas univariat: Shapiro-Wilk, D'Agostino, dan Kolmogorov-Smirnov."""
    num = df.select_dtypes(include=np.number)
    rows = []
    for col in num.columns:
        s = num[col].dropna().to_numpy()
        n = len(s)
        row: dict[str, object] = {"Variabel": col, "N": n}
        if n >= 3 and np.std(s) > 0:
            sw_stat, sw_p = stats.shapiro(s[:5000])
            row["Shapiro-W"] = float(sw_stat)
            row["p (Shapiro)"] = float(sw_p)
        else:
            row["Shapiro-W"] = np.nan
            row["p (Shapiro)"] = np.nan
        if n >= 20 and np.std(s) > 0:
            k2, k2_p = stats.normaltest(s)
            row["D'Agostino K2"] = float(k2)
            row["p (K2)"] = float(k2_p)
        else:
            row["D'Agostino K2"] = np.nan
            row["p (K2)"] = np.nan
        if n >= 3 and np.std(s, ddof=1) > 0:
            ks_stat, ks_p = stats.kstest(
                (s - s.mean()) / s.std(ddof=1), "norm"
            )
            row["KS"] = float(ks_stat)
            row["p (KS)"] = float(ks_p)
        else:
            row["KS"] = np.nan
            row["p (KS)"] = np.nan
        p_ref = row["p (Shapiro)"] if not pd.isna(row["p (Shapiro)"]) else row["p (KS)"]
        row["Kesimpulan"] = (
            "Tidak normal" if (not pd.isna(p_ref) and p_ref < alpha) else "Normal"
        )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class MardiaResult:
    skewness: float
    skew_chi2: float
    skew_df: int
    skew_p: float
    kurtosis: float
    kurt_z: float
    kurt_p: float
    n: int
    p: int

    @property
    def multivariate_normal(self) -> bool:
        return self.skew_p > 0.05 and self.kurt_p > 0.05

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Uji": "Mardia Skewness",
                    "Statistik": self.skewness,
                    "Chi-square / Z": self.skew_chi2,
                    "df": self.skew_df,
                    "p-value": self.skew_p,
                },
                {
                    "Uji": "Mardia Kurtosis",
                    "Statistik": self.kurtosis,
                    "Chi-square / Z": self.kurt_z,
                    "df": np.nan,
                    "p-value": self.kurt_p,
                },
            ]
        )


def mardia_test(df: pd.DataFrame) -> MardiaResult:
    """Uji normalitas multivariat Mardia (skewness & kurtosis)."""
    X = df.select_dtypes(include=np.number).dropna().to_numpy(float)
    n, p = X.shape
    if n <= p:
        raise ValueError("Jumlah observasi harus lebih besar dari jumlah variabel.")
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    S_inv = np.linalg.pinv(S)
    M = Xc @ S_inv @ Xc.T

    b1p = float((M**3).sum() / n**2)
    skew_chi2 = n * b1p / 6.0
    skew_df = p * (p + 1) * (p + 2) // 6
    skew_p = float(stats.chi2.sf(skew_chi2, skew_df))

    b2p = float((np.diag(M) ** 2).mean())
    kurt_z = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    kurt_p = float(2 * stats.norm.sf(abs(kurt_z)))

    return MardiaResult(
        skewness=b1p,
        skew_chi2=float(skew_chi2),
        skew_df=int(skew_df),
        skew_p=skew_p,
        kurtosis=b2p,
        kurt_z=float(kurt_z),
        kurt_p=kurt_p,
        n=n,
        p=p,
    )


def mahalanobis_outliers(df: pd.DataFrame, alpha: float = 0.001) -> pd.DataFrame:
    """Jarak Mahalanobis tiap observasi beserta penandaan pencilan multivariat."""
    num = df.select_dtypes(include=np.number).dropna()
    X = num.to_numpy(float)
    n, p = X.shape
    if n <= p:
        raise ValueError("Jumlah observasi harus lebih besar dari jumlah variabel.")
    Xc = X - X.mean(axis=0)
    S_inv = np.linalg.pinv(np.cov(X, rowvar=False))
    d2 = np.einsum("ij,jk,ik->i", Xc, S_inv, Xc)
    cutoff = float(stats.chi2.ppf(1 - alpha, p))
    return pd.DataFrame(
        {
            "Indeks": num.index,
            "Mahalanobis D2": d2,
            "p-value": stats.chi2.sf(d2, p),
            "Cutoff": cutoff,
            "Pencilan": np.where(d2 > cutoff, "Ya", "Tidak"),
        }
    )


def univariate_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Deteksi pencilan univariat dengan aturan IQR."""
    num = df.select_dtypes(include=np.number)
    rows = []
    for col in num.columns:
        s = num[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        mask = (s < low) | (s > high)
        rows.append(
            {
                "Variabel": col,
                "Batas Bawah": float(low),
                "Batas Atas": float(high),
                "Jumlah Pencilan": int(mask.sum()),
                "% Pencilan": round(float(mask.mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)
