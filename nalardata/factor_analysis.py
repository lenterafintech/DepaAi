"""Analisis faktor eksploratori (EFA) dengan rotasi varimax/promax.

Metode ekstraksi yang tersedia:
- ``principal``: principal component extraction
- ``paf``: principal axis factoring (iteratif terhadap komunalitas)
- ``ml``: maximum likelihood (scikit-learn ``FactorAnalysis``)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EXTRACTION_METHODS = ("principal", "paf", "ml")
ROTATIONS = ("tanpa rotasi", "varimax", "promax")


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 200, tol: float = 1e-6):
    """Rotasi ortogonal varimax. Mengembalikan (loadings terotasi, matriks rotasi)."""
    p, k = loadings.shape
    if k < 2:
        return loadings.copy(), np.eye(k)
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        L = loadings @ R
        B = loadings.T @ (
            L**3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))
        )
        u, s, vh = np.linalg.svd(B)
        R = u @ vh
        d = float(s.sum())
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return loadings @ R, R


def promax(loadings: np.ndarray, kappa: float = 4.0):
    """Rotasi oblique promax berbasis solusi varimax."""
    if loadings.shape[1] < 2:
        return loadings.copy(), np.eye(loadings.shape[1])
    L, _ = varimax(loadings)
    P = np.sign(L) * np.abs(L) ** kappa
    U = np.linalg.lstsq(L, P, rcond=None)[0]
    d = np.diag(np.linalg.pinv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    return L @ U, U


@dataclass
class FactorResult:
    loadings: pd.DataFrame
    communalities: pd.Series
    uniquenesses: pd.Series
    eigenvalues: np.ndarray
    variance: pd.DataFrame
    scores: pd.DataFrame
    factor_correlation: pd.DataFrame | None
    method: str
    rotation: str
    n_factors: int

    def dominant_loadings(self, threshold: float = 0.5) -> pd.DataFrame:
        """Pemetaan tiap variabel ke faktor dengan muatan tertinggi."""
        abs_load = self.loadings.abs()
        best = abs_load.idxmax(axis=1)
        rows = []
        for var in self.loadings.index:
            factor = best[var]
            value = float(self.loadings.loc[var, factor])
            rows.append(
                {
                    "Variabel": var,
                    "Faktor Dominan": factor,
                    "Muatan": value,
                    "Status": (
                        "Valid" if abs(value) >= threshold else "Muatan rendah"
                    ),
                }
            )
        return pd.DataFrame(rows)


def _paf_loadings(R: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-6):
    """Principal axis factoring: iterasi komunalitas pada diagonal matriks korelasi."""
    R_work = R.copy()
    R_inv = np.linalg.pinv(R)
    comm = 1.0 - 1.0 / np.diag(R_inv)
    comm = np.clip(comm, 0.01, 0.99)
    loadings = np.zeros((R.shape[0], k))
    for _ in range(max_iter):
        np.fill_diagonal(R_work, comm)
        eigvals, eigvecs = np.linalg.eigh(R_work)
        order = np.argsort(eigvals)[::-1][:k]
        vals = np.clip(eigvals[order], 0.0, None)
        loadings = eigvecs[:, order] * np.sqrt(vals)
        new_comm = (loadings**2).sum(axis=1)
        if np.max(np.abs(new_comm - comm)) < tol:
            comm = new_comm
            break
        comm = np.clip(new_comm, 0.0, 1.0)
    return loadings


def run_factor_analysis(
    df: pd.DataFrame,
    n_factors: int = 2,
    method: str = "paf",
    rotation: str = "varimax",
) -> FactorResult:
    """Jalankan analisis faktor eksploratori pada variabel numerik."""
    if method not in EXTRACTION_METHODS:
        raise ValueError(f"Metode '{method}' tidak dikenal. Pilih dari {EXTRACTION_METHODS}.")
    if rotation not in ROTATIONS:
        raise ValueError(f"Rotasi '{rotation}' tidak dikenal. Pilih dari {ROTATIONS}.")

    X_df = df.select_dtypes(include=np.number).dropna()
    p = X_df.shape[1]
    if p < 2:
        raise ValueError("Analisis faktor memerlukan minimal 2 variabel numerik.")
    if not 1 <= n_factors <= p:
        raise ValueError(f"Jumlah faktor harus antara 1 dan {p}.")

    X = X_df.to_numpy(float)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - X.mean(axis=0)) / sd
    R = np.corrcoef(X, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(R))[::-1]

    if method == "ml":
        from sklearn.decomposition import FactorAnalysis

        fa = FactorAnalysis(n_components=n_factors, random_state=0)
        fa.fit(Z)
        raw = fa.components_.T
    elif method == "paf":
        raw = _paf_loadings(R, n_factors)
    else:
        eigvals, eigvecs = np.linalg.eigh(R)
        order = np.argsort(eigvals)[::-1][:n_factors]
        raw = eigvecs[:, order] * np.sqrt(np.clip(eigvals[order], 0, None))

    factor_corr = None
    if rotation == "varimax":
        rotated, _ = varimax(raw)
    elif rotation == "promax":
        rotated, U = promax(raw)
        Uinv = np.linalg.pinv(U)
        phi = Uinv @ Uinv.T
        d = np.sqrt(np.diag(phi))
        phi = phi / np.outer(d, d)
        factor_corr = pd.DataFrame(
            phi,
            index=[f"F{i + 1}" for i in range(n_factors)],
            columns=[f"F{i + 1}" for i in range(n_factors)],
        )
    else:
        rotated = raw

    for j in range(rotated.shape[1]):
        if rotated[:, j].sum() < 0:
            rotated[:, j] *= -1

    names = [f"F{i + 1}" for i in range(n_factors)]
    loadings = pd.DataFrame(rotated, index=X_df.columns, columns=names)
    communalities = (loadings**2).sum(axis=1).rename("Komunalitas")
    uniquenesses = (1 - communalities).rename("Keunikan")

    ss = (loadings**2).sum(axis=0)
    variance = pd.DataFrame(
        {
            "Faktor": names,
            "SS Loadings": ss.to_numpy(),
            "% Varians": (ss / p * 100).to_numpy(),
            "% Kumulatif": (ss.cumsum() / p * 100).to_numpy(),
        }
    )

    weights = np.linalg.pinv(R) @ rotated
    scores = pd.DataFrame(Z @ weights, index=X_df.index, columns=names)

    return FactorResult(
        loadings=loadings,
        communalities=communalities,
        uniquenesses=uniquenesses,
        eigenvalues=eigenvalues,
        variance=variance,
        scores=scores,
        factor_correlation=factor_corr,
        method=method,
        rotation=rotation,
        n_factors=n_factors,
    )
