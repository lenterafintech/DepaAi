"""Analisis korelasi kanonik (Canonical Correlation Analysis)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg, stats


@dataclass
class CCAResult:
    correlations: np.ndarray
    significance: pd.DataFrame
    x_weights: pd.DataFrame
    y_weights: pd.DataFrame
    x_loadings: pd.DataFrame
    y_loadings: pd.DataFrame
    x_scores: pd.DataFrame
    y_scores: pd.DataFrame
    redundancy: pd.DataFrame
    n: int

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Fungsi Kanonik": [f"CV{i + 1}" for i in range(len(self.correlations))],
                "Korelasi Kanonik": self.correlations,
                "R2 Kanonik": self.correlations**2,
            }
        )


def _inv_sqrt(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    vals, vecs = linalg.eigh(matrix)
    vals = np.clip(vals, tol, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def run_cca(df: pd.DataFrame, x_vars: list[str], y_vars: list[str]) -> CCAResult:
    """Hubungan antara dua gugus variabel melalui variat kanonik.

    Signifikansi tiap fungsi diuji dengan pendekatan Bartlett terhadap
    Wilks' lambda kumulatif.
    """
    overlap = set(x_vars) & set(y_vars)
    if overlap:
        raise ValueError(f"Variabel {sorted(overlap)} tidak boleh ada di kedua gugus.")
    if not x_vars or not y_vars:
        raise ValueError("Kedua gugus variabel harus terisi.")

    data = df[[*x_vars, *y_vars]].dropna()
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Variabel '{col}' harus numerik untuk korelasi kanonik.")

    X = data[x_vars].to_numpy(float)
    Y = data[y_vars].to_numpy(float)
    n, p = X.shape
    q = Y.shape[1]
    if n <= p + q:
        raise ValueError("Jumlah observasi harus lebih besar dari total variabel kedua gugus.")

    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Yz = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    Sxx = np.cov(Xz, rowvar=False).reshape(p, p)
    Syy = np.cov(Yz, rowvar=False).reshape(q, q)
    Sxy = np.cov(Xz, Yz, rowvar=False)[:p, p:]

    Sxx_inv = _inv_sqrt(Sxx)
    Syy_inv = _inv_sqrt(Syy)
    U, s, Vt = np.linalg.svd(Sxx_inv @ Sxy @ Syy_inv)

    k = min(p, q)
    corrs = np.clip(s[:k], 0.0, 1.0)
    names = [f"CV{i + 1}" for i in range(k)]

    a = Sxx_inv @ U[:, :k]
    b = Syy_inv @ Vt.T[:, :k]
    x_scores = pd.DataFrame(Xz @ a, index=data.index, columns=names)
    y_scores = pd.DataFrame(Yz @ b, index=data.index, columns=names)

    x_loadings = pd.DataFrame(
        [[np.corrcoef(Xz[:, i], x_scores[c])[0, 1] for c in names] for i in range(p)],
        index=x_vars,
        columns=names,
    )
    y_loadings = pd.DataFrame(
        [[np.corrcoef(Yz[:, j], y_scores[c])[0, 1] for c in names] for j in range(q)],
        index=y_vars,
        columns=names,
    )

    rows = []
    for i in range(k):
        lam = float(np.prod(1 - corrs[i:] ** 2))
        chi2 = -(n - 1 - (p + q + 1) / 2.0) * np.log(max(lam, 1e-300))
        dof = max((p - i) * (q - i), 1)
        p_value = float(stats.chi2.sf(chi2, dof))
        rows.append(
            {
                "Fungsi": f"{i + 1} sampai {k}",
                "Korelasi Kanonik": float(corrs[i]),
                "Wilks' Lambda": lam,
                "Chi-square": float(chi2),
                "df": dof,
                "p-value": p_value,
                "Signifikan": "Ya" if p_value < 0.05 else "Tidak",
            }
        )
    significance = pd.DataFrame(rows)

    redundancy = pd.DataFrame(
        {
            "Fungsi": names,
            "Varians X dijelaskan variat X (%)": [
                float((x_loadings[c] ** 2).mean() * 100) for c in names
            ],
            "Redundansi X|Y (%)": [
                float((x_loadings[c] ** 2).mean() * corrs[i] ** 2 * 100)
                for i, c in enumerate(names)
            ],
            "Varians Y dijelaskan variat Y (%)": [
                float((y_loadings[c] ** 2).mean() * 100) for c in names
            ],
            "Redundansi Y|X (%)": [
                float((y_loadings[c] ** 2).mean() * corrs[i] ** 2 * 100)
                for i, c in enumerate(names)
            ],
        }
    )

    return CCAResult(
        correlations=corrs,
        significance=significance,
        x_weights=pd.DataFrame(a, index=x_vars, columns=names),
        y_weights=pd.DataFrame(b, index=y_vars, columns=names),
        x_loadings=x_loadings,
        y_loadings=y_loadings,
        x_scores=x_scores,
        y_scores=y_scores,
        redundancy=redundancy,
        n=n,
    )
