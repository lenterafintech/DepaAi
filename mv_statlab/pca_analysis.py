"""Principal Component Analysis (PCA)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    eigenvalues: np.ndarray
    explained_ratio: np.ndarray
    cumulative_ratio: np.ndarray
    loadings: pd.DataFrame
    eigenvectors: pd.DataFrame
    scores: pd.DataFrame
    communalities: pd.Series
    n_components: int
    standardized: bool
    variables: list[str] = field(default_factory=list)

    @property
    def kaiser_components(self) -> int:
        """Jumlah komponen dengan eigenvalue > 1 (kriteria Kaiser)."""
        return int((self.eigenvalues > 1).sum())

    def variance_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Komponen": [f"PC{i + 1}" for i in range(len(self.eigenvalues))],
                "Eigenvalue": self.eigenvalues,
                "% Varians": self.explained_ratio * 100,
                "% Kumulatif": self.cumulative_ratio * 100,
            }
        )

    def components_needed(self, target: float = 0.80) -> int:
        idx = np.searchsorted(self.cumulative_ratio, target) + 1
        return int(min(idx, len(self.cumulative_ratio)))


def run_pca(
    df: pd.DataFrame, n_components: int | None = None, standardize: bool = True
) -> PCAResult:
    """Jalankan PCA pada matriks korelasi (standardize=True) atau kovarians."""
    X_df = df.select_dtypes(include=np.number).dropna()
    if X_df.shape[1] < 2:
        raise ValueError("PCA memerlukan minimal 2 variabel numerik.")
    if X_df.shape[0] <= X_df.shape[1]:
        raise ValueError("Jumlah observasi harus lebih besar dari jumlah variabel.")

    X = X_df.to_numpy(float)
    mean = X.mean(axis=0)
    Xc = X - mean
    if standardize:
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xc = Xc / sd
    S = np.cov(Xc, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    # Arahkan setiap vektor agar muatan dominan bernilai positif supaya
    # tanda komponen stabil antar-pemanggilan.
    for j in range(eigvecs.shape[1]):
        if eigvecs[np.argmax(np.abs(eigvecs[:, j])), j] < 0:
            eigvecs[:, j] *= -1

    total = eigvals.sum()
    ratio = eigvals / total if total > 0 else np.zeros_like(eigvals)
    cumulative = np.cumsum(ratio)

    k = n_components or int(max((eigvals > 1).sum(), 1))
    k = int(min(max(k, 1), X_df.shape[1]))

    names = [f"PC{i + 1}" for i in range(k)]
    loadings = pd.DataFrame(
        eigvecs[:, :k] * np.sqrt(eigvals[:k]), index=X_df.columns, columns=names
    )
    scores = pd.DataFrame(Xc @ eigvecs[:, :k], index=X_df.index, columns=names)
    communalities = (loadings**2).sum(axis=1).rename("Komunalitas")

    return PCAResult(
        eigenvalues=eigvals,
        explained_ratio=ratio,
        cumulative_ratio=cumulative,
        loadings=loadings,
        eigenvectors=pd.DataFrame(
            eigvecs[:, :k], index=X_df.columns, columns=names
        ),
        scores=scores,
        communalities=communalities,
        n_components=k,
        standardized=standardize,
        variables=list(X_df.columns),
    )


def parallel_analysis(
    df: pd.DataFrame, n_iter: int = 200, percentile: float = 95.0, seed: int = 42
) -> pd.DataFrame:
    """Analisis paralel Horn: bandingkan eigenvalue data dengan data acak.

    Komponen dipertahankan selama eigenvalue aktual melebihi eigenvalue acak.
    """
    X = df.select_dtypes(include=np.number).dropna().to_numpy(float)
    n, p = X.shape
    actual = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]

    rng = np.random.default_rng(seed)
    simulated = np.empty((n_iter, p))
    for i in range(n_iter):
        R = np.corrcoef(rng.standard_normal((n, p)), rowvar=False)
        simulated[i] = np.sort(np.linalg.eigvalsh(R))[::-1]
    threshold = np.percentile(simulated, percentile, axis=0)

    return pd.DataFrame(
        {
            "Komponen": [f"PC{i + 1}" for i in range(p)],
            "Eigenvalue Data": actual,
            f"Eigenvalue Acak (p{int(percentile)})": threshold,
            "Dipertahankan": np.where(actual > threshold, "Ya", "Tidak"),
        }
    )
