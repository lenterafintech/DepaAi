"""Analisis klaster: K-Means, hierarki (agglomerative), dan DBSCAN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

LINKAGE_METHODS = ("ward", "complete", "average", "single")
DISTANCE_METRICS = ("euclidean", "cityblock", "correlation")


@dataclass
class ClusterResult:
    labels: pd.Series
    method: str
    n_clusters: int
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None
    centers: pd.DataFrame | None
    inertia: float | None = None
    linkage_matrix: np.ndarray | None = None

    def sizes(self) -> pd.DataFrame:
        counts = self.labels.value_counts().sort_index()
        return pd.DataFrame(
            {
                "Klaster": counts.index.astype(str),
                "Anggota": counts.to_numpy(),
                "Persen (%)": (counts / counts.sum() * 100).round(2).to_numpy(),
            }
        )

    def quality_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Metrik": "Silhouette Score",
                    "Nilai": self.silhouette,
                    "Interpretasi": _silhouette_label(self.silhouette),
                },
                {
                    "Metrik": "Calinski-Harabasz",
                    "Nilai": self.calinski_harabasz,
                    "Interpretasi": "Makin besar makin baik",
                },
                {
                    "Metrik": "Davies-Bouldin",
                    "Nilai": self.davies_bouldin,
                    "Interpretasi": "Makin kecil makin baik",
                },
            ]
        )


def _silhouette_label(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "Tidak tersedia"
    if value >= 0.7:
        return "Struktur klaster kuat"
    if value >= 0.5:
        return "Struktur klaster layak"
    if value >= 0.25:
        return "Struktur klaster lemah"
    return "Tidak ada struktur berarti"


def _metrics(X: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None, float | None]:
    valid = labels[labels >= 0]
    if len(set(valid)) < 2 or len(valid) < 3:
        return None, None, None
    mask = labels >= 0
    Xv, lv = X[mask], labels[mask]
    return (
        float(silhouette_score(Xv, lv)),
        float(calinski_harabasz_score(Xv, lv)),
        float(davies_bouldin_score(Xv, lv)),
    )


def kmeans_diagnostics(
    df: pd.DataFrame, k_min: int = 2, k_max: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Metrik untuk memilih jumlah klaster: inertia (elbow) dan silhouette."""
    X = df.select_dtypes(include=np.number).dropna().to_numpy(float)
    k_max = int(min(k_max, len(X) - 1))
    rows = []
    for k in range(max(2, k_min), k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil, ch, db = _metrics(X, model.labels_)
        rows.append(
            {
                "k": k,
                "Inertia (WSS)": float(model.inertia_),
                "Silhouette": sil,
                "Calinski-Harabasz": ch,
                "Davies-Bouldin": db,
            }
        )
    return pd.DataFrame(rows)


def run_kmeans(df: pd.DataFrame, n_clusters: int = 3, seed: int = 42) -> ClusterResult:
    X_df = df.select_dtypes(include=np.number).dropna()
    X = X_df.to_numpy(float)
    if n_clusters < 2 or n_clusters >= len(X):
        raise ValueError("Jumlah klaster harus >= 2 dan lebih kecil dari jumlah observasi.")
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
    sil, ch, db = _metrics(X, model.labels_)
    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=X_df.columns,
        index=[f"Klaster {i + 1}" for i in range(n_clusters)],
    )
    return ClusterResult(
        labels=pd.Series(model.labels_ + 1, index=X_df.index, name="Klaster"),
        method="K-Means",
        n_clusters=n_clusters,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        centers=centers,
        inertia=float(model.inertia_),
    )


def run_hierarchical(
    df: pd.DataFrame,
    n_clusters: int = 3,
    method: str = "ward",
    metric: str = "euclidean",
) -> ClusterResult:
    if method not in LINKAGE_METHODS:
        raise ValueError(f"Metode linkage '{method}' tidak dikenal.")
    X_df = df.select_dtypes(include=np.number).dropna()
    X = X_df.to_numpy(float)
    if method == "ward":
        metric = "euclidean"
    Z = linkage(X, method=method, metric=metric)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    sil, ch, db = _metrics(X, labels)
    centers = (
        pd.DataFrame(X, columns=X_df.columns)
        .groupby(labels)
        .mean()
        .rename(index=lambda i: f"Klaster {i}")
    )
    return ClusterResult(
        labels=pd.Series(labels, index=X_df.index, name="Klaster"),
        method=f"Hierarki ({method})",
        n_clusters=int(len(np.unique(labels))),
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        centers=centers,
        linkage_matrix=Z,
    )


def run_dbscan(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> ClusterResult:
    X_df = df.select_dtypes(include=np.number).dropna()
    X = X_df.to_numpy(float)
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = model.labels_
    sil, ch, db = _metrics(X, labels)
    n_clusters = int(len(set(labels) - {-1}))
    # Label -1 dipertahankan sebagai penanda noise, klaster lain mulai dari 1.
    display = pd.Series(
        np.where(labels < 0, -1, labels + 1), index=X_df.index, name="Klaster"
    )
    return ClusterResult(
        labels=display,
        method="DBSCAN",
        n_clusters=n_clusters,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        centers=None,
    )


def profile_clusters(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Rata-rata tiap variabel per klaster beserta uji beda antar klaster (ANOVA F)."""
    X = df.select_dtypes(include=np.number).loc[labels.index]
    data = X.copy()
    data["__klaster__"] = labels.to_numpy()
    means = data.groupby("__klaster__").mean().T
    means.columns = [f"Klaster {c}" for c in means.columns]
    means.insert(0, "Rata-rata Total", X.mean())

    f_stats, p_values = [], []
    for col in X.columns:
        groups = [g[col].to_numpy() for _, g in data.groupby("__klaster__") if len(g) > 1]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
        else:
            f, p = np.nan, np.nan
        f_stats.append(float(f))
        p_values.append(float(p))
    means["F (ANOVA)"] = f_stats
    means["p-value"] = p_values
    means["Pembeda"] = np.where(np.array(p_values) < 0.05, "Signifikan", "Tidak")
    return means.reset_index().rename(columns={"index": "Variabel"})


def silhouette_detail(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    X = df.select_dtypes(include=np.number).loc[labels.index].to_numpy(float)
    lab = labels.to_numpy()
    mask = lab >= 0
    values = np.full(len(lab), np.nan)
    if len(set(lab[mask])) >= 2:
        values[mask] = silhouette_samples(X[mask], lab[mask])
    return pd.DataFrame({"Klaster": lab, "Silhouette": values}, index=labels.index)
