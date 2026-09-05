"""Analisis diskriminan linear (LDA) dan kuadratik (QDA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


@dataclass
class DiscriminantResult:
    model: object
    kind: str
    labels: pd.Series
    predictions: pd.Series
    scores: pd.DataFrame | None
    coefficients: pd.DataFrame | None
    group_means: pd.DataFrame
    eigenvalues: pd.DataFrame | None
    wilks: pd.DataFrame | None
    confusion: pd.DataFrame
    accuracy: float
    cv_accuracy: float
    classes: list

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Metrik": "Akurasi klasifikasi (data latih)", "Nilai": self.accuracy},
                {"Metrik": "Akurasi validasi silang (5-fold)", "Nilai": self.cv_accuracy},
                {"Metrik": "Jumlah kelompok", "Nilai": len(self.classes)},
                {"Metrik": "Jumlah observasi", "Nilai": len(self.labels)},
            ]
        )


def _canonical_eigenvalues(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eigenvalue dari W^-1 B untuk fungsi diskriminan kanonik."""
    overall = X.mean(axis=0)
    W = np.zeros((X.shape[1], X.shape[1]))
    B = np.zeros_like(W)
    for cls in np.unique(y):
        Xi = X[y == cls]
        ni = len(Xi)
        mi = Xi.mean(axis=0)
        Xi_c = Xi - mi
        W += Xi_c.T @ Xi_c
        diff = (mi - overall).reshape(-1, 1)
        B += ni * (diff @ diff.T)
    eigvals = np.linalg.eigvals(np.linalg.pinv(W) @ B)
    eigvals = np.real(eigvals)
    eigvals = np.sort(np.clip(eigvals, 0.0, None))[::-1]
    n_func = min(len(np.unique(y)) - 1, X.shape[1])
    return eigvals[:n_func]


def _wilks_table(eigvals: np.ndarray, n: int, p: int, g: int) -> pd.DataFrame:
    """Uji Wilks' lambda berurutan untuk setiap fungsi diskriminan."""
    rows = []
    for k in range(len(eigvals)):
        lam = float(np.prod(1.0 / (1.0 + eigvals[k:])))
        chi2 = -(n - 1 - (p + g) / 2.0) * np.log(lam)
        dof = (p - k) * (g - 1 - k)
        dof = max(int(dof), 1)
        rows.append(
            {
                "Fungsi": f"{k + 1} sampai {len(eigvals)}",
                "Wilks' Lambda": lam,
                "Chi-square": float(chi2),
                "df": dof,
                "p-value": float(stats.chi2.sf(chi2, dof)),
                "Signifikan": "Ya" if stats.chi2.sf(chi2, dof) < 0.05 else "Tidak",
            }
        )
    return pd.DataFrame(rows)


def run_discriminant(
    df: pd.DataFrame,
    group: str,
    predictors: list[str],
    kind: str = "linear",
) -> DiscriminantResult:
    """Analisis diskriminan untuk memprediksi keanggotaan kelompok."""
    if kind not in ("linear", "kuadratik"):
        raise ValueError("Jenis analisis harus 'linear' atau 'kuadratik'.")
    if group in predictors:
        raise ValueError("Variabel kelompok tidak boleh menjadi prediktor.")
    if len(predictors) < 2:
        raise ValueError("Analisis diskriminan memerlukan minimal 2 variabel prediktor.")

    data = df[[group, *predictors]].dropna()
    numeric = [c for c in predictors if pd.api.types.is_numeric_dtype(data[c])]
    if len(numeric) < 2:
        raise ValueError("Minimal 2 prediktor numerik dibutuhkan.")

    X_df = data[numeric].astype(float)
    X = X_df.to_numpy()
    y = data[group].astype(str).to_numpy()
    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("Variabel kelompok harus memiliki minimal 2 kategori.")

    if kind == "linear":
        model = LinearDiscriminantAnalysis(solver="svd")
    else:
        model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)
    pred = model.predict(X)

    n_splits = int(min(5, pd.Series(y).value_counts().min()))
    if n_splits >= 2:
        cv = float(cross_val_score(model, X, y, cv=n_splits).mean())
    else:
        cv = float("nan")

    group_means = X_df.groupby(y).mean()
    group_means.index.name = group
    group_means = group_means.reset_index()

    scores = coefficients = eigen_table = wilks = None
    if kind == "linear":
        transformed = model.transform(X)
        names = [f"Fungsi {i + 1}" for i in range(transformed.shape[1])]
        scores = pd.DataFrame(transformed, index=data.index, columns=names)
        coefficients = pd.DataFrame(
            model.scalings_[:, : transformed.shape[1]], index=numeric, columns=names
        )
        eigvals = _canonical_eigenvalues(X, y)
        if len(eigvals):
            total = eigvals.sum()
            eigen_table = pd.DataFrame(
                {
                    "Fungsi": [f"Fungsi {i + 1}" for i in range(len(eigvals))],
                    "Eigenvalue": eigvals,
                    "% Varians": eigvals / total * 100 if total > 0 else eigvals,
                    "% Kumulatif": np.cumsum(eigvals) / total * 100
                    if total > 0
                    else eigvals,
                    "Korelasi Kanonik": np.sqrt(eigvals / (1 + eigvals)),
                }
            )
            wilks = _wilks_table(eigvals, len(X), len(numeric), len(classes))

    cm = confusion_matrix(y, pred, labels=classes)
    confusion = pd.DataFrame(
        cm,
        index=[f"Aktual {c}" for c in classes],
        columns=[f"Prediksi {c}" for c in classes],
    )

    return DiscriminantResult(
        model=model,
        kind=kind,
        labels=pd.Series(y, index=data.index, name=group),
        predictions=pd.Series(pred, index=data.index, name="Prediksi"),
        scores=scores,
        coefficients=coefficients,
        group_means=group_means,
        eigenvalues=eigen_table,
        wilks=wilks,
        confusion=confusion,
        accuracy=float(accuracy_score(y, pred)),
        cv_accuracy=cv,
        classes=classes,
    )
