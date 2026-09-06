"""MANOVA satu jalur dan uji lanjutan (ANOVA univariat, Hotelling's T2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA


@dataclass
class ManovaResult:
    multivariate: pd.DataFrame
    univariate: pd.DataFrame
    group_means: pd.DataFrame
    group_sizes: pd.DataFrame
    dependents: list[str]
    factor: str

    def conclusion(self, alpha: float = 0.05) -> str:
        wilks = self.multivariate[self.multivariate["Statistik"] == "Wilks' lambda"]
        if wilks.empty or pd.isna(wilks["p-value"].iloc[0]):
            return "Hasil uji multivariat tidak tersedia."
        p = float(wilks["p-value"].iloc[0])
        if p < alpha:
            return (
                f"Terdapat perbedaan rata-rata vektor variabel dependen antar kelompok "
                f"'{self.factor}' (Wilks' lambda, p = {p:.4f} < {alpha})."
            )
        return (
            f"Tidak terdapat perbedaan signifikan antar kelompok '{self.factor}' "
            f"(Wilks' lambda, p = {p:.4f} >= {alpha})."
        )


def run_manova(df: pd.DataFrame, dependents: list[str], factor: str) -> ManovaResult:
    """MANOVA satu jalur: apakah vektor rata-rata berbeda antar kelompok."""
    if len(dependents) < 2:
        raise ValueError("MANOVA memerlukan minimal 2 variabel dependen numerik.")
    if factor in dependents:
        raise ValueError("Variabel faktor tidak boleh menjadi variabel dependen.")

    data = df[[*dependents, factor]].dropna()
    for col in dependents:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Variabel dependen '{col}' harus numerik.")
    if data[factor].nunique() < 2:
        raise ValueError("Variabel faktor harus memiliki minimal 2 kelompok.")

    # Nama kolom disederhanakan agar aman dipakai dalam formula statsmodels.
    safe = {col: f"v{i}" for i, col in enumerate(dependents)}
    work = data.rename(columns=safe)
    work["grp"] = data[factor].astype(str).to_numpy()
    formula = f"{' + '.join(safe.values())} ~ grp"
    fitted = MANOVA.from_formula(formula, data=work)
    table = fitted.mv_test().results["grp"]["stat"]

    multivariate = pd.DataFrame(
        {
            "Statistik": table.index,
            "Nilai": table["Value"].to_numpy(),
            "F": table["F Value"].to_numpy(),
            "df Hipotesis": table["Num DF"].to_numpy(),
            "df Galat": table["Den DF"].to_numpy(),
            "p-value": table["Pr > F"].to_numpy(),
        }
    )
    multivariate["Signifikan"] = np.where(multivariate["p-value"] < 0.05, "Ya", "Tidak")

    rows = []
    for col in dependents:
        groups = [g[col].to_numpy(float) for _, g in data.groupby(factor, observed=True)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p = stats.f_oneway(*groups)
        grand = data[col].to_numpy(float)
        ss_total = ((grand - grand.mean()) ** 2).sum()
        ss_between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
        rows.append(
            {
                "Variabel": col,
                "F": float(f_stat),
                "p-value": float(p),
                "Eta-squared": float(ss_between / ss_total) if ss_total > 0 else np.nan,
                "Signifikan": "Ya" if p < 0.05 else "Tidak",
            }
        )
    univariate = pd.DataFrame(rows)

    group_means = data.groupby(factor, observed=True)[dependents].mean().reset_index()
    sizes = data[factor].value_counts().sort_index()
    group_sizes = pd.DataFrame(
        {"Kelompok": sizes.index.astype(str), "N": sizes.to_numpy()}
    )

    return ManovaResult(
        multivariate=multivariate,
        univariate=univariate,
        group_means=group_means,
        group_sizes=group_sizes,
        dependents=list(dependents),
        factor=factor,
    )


def hotelling_t2(df: pd.DataFrame, dependents: list[str], factor: str) -> pd.DataFrame:
    """Hotelling's T2 untuk perbandingan dua kelompok pada banyak variabel."""
    data = df[[*dependents, factor]].dropna()
    groups = list(data[factor].unique())
    if len(groups) != 2:
        raise ValueError("Hotelling's T2 memerlukan tepat 2 kelompok.")

    X1 = data.loc[data[factor] == groups[0], dependents].to_numpy(float)
    X2 = data.loc[data[factor] == groups[1], dependents].to_numpy(float)
    n1, n2 = len(X1), len(X2)
    p = len(dependents)
    if n1 + n2 - p - 1 <= 0:
        raise ValueError("Jumlah observasi tidak cukup untuk jumlah variabel yang dipilih.")

    diff = X1.mean(axis=0) - X2.mean(axis=0)
    S_pooled = ((n1 - 1) * np.cov(X1, rowvar=False) + (n2 - 1) * np.cov(X2, rowvar=False)) / (
        n1 + n2 - 2
    )
    t2 = float(diff @ np.linalg.pinv(S_pooled) @ diff * (n1 * n2) / (n1 + n2))
    f_stat = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    p_value = float(stats.f.sf(f_stat, p, n1 + n2 - p - 1))
    return pd.DataFrame(
        [
            {
                "Kelompok 1": str(groups[0]),
                "Kelompok 2": str(groups[1]),
                "Hotelling T2": t2,
                "F": float(f_stat),
                "df1": p,
                "df2": n1 + n2 - p - 1,
                "p-value": p_value,
                "Signifikan": "Ya" if p_value < 0.05 else "Tidak",
            }
        ]
    )
