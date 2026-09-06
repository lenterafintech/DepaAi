"""Analisis korelasi: matriks korelasi, signifikansi, dan korelasi parsial."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ("pearson", "spearman", "kendall")


@dataclass
class CorrelationResult:
    matrix: pd.DataFrame
    p_values: pd.DataFrame
    n: int
    method: str

    def significant_pairs(self, alpha: float = 0.05) -> pd.DataFrame:
        cols = list(self.matrix.columns)
        rows = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                r = float(self.matrix.loc[a, b])
                p = float(self.p_values.loc[a, b])
                rows.append(
                    {
                        "Variabel 1": a,
                        "Variabel 2": b,
                        "r": r,
                        "p-value": p,
                        "Kekuatan": strength_label(r),
                        "Signifikan": "Ya" if p < alpha else "Tidak",
                    }
                )
        out = pd.DataFrame(rows)
        return out.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(
            drop=True
        )


def strength_label(r: float) -> str:
    a = abs(r)
    if a >= 0.8:
        return "Sangat kuat"
    if a >= 0.6:
        return "Kuat"
    if a >= 0.4:
        return "Sedang"
    if a >= 0.2:
        return "Lemah"
    return "Sangat lemah"


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> CorrelationResult:
    """Matriks korelasi beserta matriks p-value untuk setiap pasangan variabel."""
    if method not in METHODS:
        raise ValueError(f"Metode '{method}' tidak dikenal. Pilih dari {METHODS}.")
    X = df.select_dtypes(include=np.number).dropna()
    if X.shape[1] < 2:
        raise ValueError("Analisis korelasi memerlukan minimal 2 variabel numerik.")

    cols = list(X.columns)
    corr = X.corr(method=method)
    pvals = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    test = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }[method]
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            res = test(X[a].to_numpy(float), X[b].to_numpy(float))
            p = float(res[1])
            pvals.loc[a, b] = pvals.loc[b, a] = p
    for col in cols:
        pvals.loc[col, col] = 0.0
    return CorrelationResult(matrix=corr, p_values=pvals, n=len(X), method=method)


def partial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Korelasi parsial: hubungan dua variabel dengan mengontrol variabel lainnya."""
    X = df.select_dtypes(include=np.number).dropna()
    if X.shape[1] < 3:
        raise ValueError("Korelasi parsial memerlukan minimal 3 variabel.")
    R = np.corrcoef(X.to_numpy(float), rowvar=False)
    R_inv = np.linalg.pinv(R)
    d = np.sqrt(np.diag(R_inv))
    pc = -R_inv / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pd.DataFrame(pc, index=X.columns, columns=X.columns)


def covariance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df.select_dtypes(include=np.number).dropna()
    return X.cov()
