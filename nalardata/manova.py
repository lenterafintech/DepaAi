"""MANOVA satu jalur, ANOVA berulang, dan uji lanjutan (Hotelling's T2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM


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


# --------------------------------------------------------------------------- #
# ANOVA pengukuran berulang (repeated measures), dengan uji sphericity
# --------------------------------------------------------------------------- #


@dataclass
class MauchlyResult:
    """Uji sphericity Mauchly: apakah ragam selisih antar semua pasangan kondisi sama.

    Dengan tepat 2 kondisi, sphericity otomatis terpenuhi — hanya ada satu selisih
    yang mungkin dibentuk, sehingga tidak ada "kesamaan ragam antar selisih" yang
    perlu diuji. Konvensi ini dipakai SPSS, R (rstatix), dan seluruh pustaka statistik
    lain, bukan penyederhanaan yang dibuat di sini.
    """

    w: float
    chi_square: float
    df: int
    p_value: float
    berlaku: bool  # sphericity dapat diuji (k >= 3)

    @property
    def terpenuhi(self) -> bool:
        return (not self.berlaku) or bool(self.p_value > 0.05)


def _kontras_ortonormal(k: int) -> np.ndarray:
    """Basis ortonormal (k x k-1) tegak lurus vektor konstan.

    Hasil uji Mauchly tidak bergantung pada basis kontras spesifik yang dipilih
    selama basisnya ortonormal — QR di sini hanya cara yang stabil secara numerik
    untuk membangunnya, bukan pilihan yang memengaruhi hasil akhir.
    """
    konstan = np.ones((k, 1)) / np.sqrt(k)
    awal = np.eye(k)[:, :-1]
    q, _ = np.linalg.qr(np.hstack([konstan, awal]))
    return q[:, 1:k]


def mauchly_sphericity(data: np.ndarray) -> tuple[MauchlyResult, float, float]:
    """Uji Mauchly beserta epsilon koreksi Greenhouse-Geisser dan Huynh-Feldt.

    ``data`` berbentuk lebar: n subjek x k kondisi. Rumus mengikuti Maxwell &
    Delaney (2004) dan identik dengan yang dipakai SPSS/R: kovarians ditransformasi
    ke basis kontras ortonormal (k-1 dimensi) lebih dulu, karena sphericity adalah
    sifat pada SELISIH antar kondisi, bukan pada kondisi mentahnya.
    """
    n, k = data.shape
    p = k - 1
    if p < 1:
        raise ValueError("Pengukuran berulang memerlukan minimal 2 kondisi.")
    if n <= k:
        raise ValueError(
            f"Jumlah subjek ({n}) harus lebih besar daripada jumlah kondisi ({k})."
        )

    if p == 1:
        # Tepat 2 kondisi: sphericity trivial, epsilon selalu 1 (tidak ada koreksi).
        return MauchlyResult(w=1.0, chi_square=0.0, df=0, p_value=float("nan"), berlaku=False), 1.0, 1.0

    s = np.cov(data, rowvar=False, ddof=1)
    m = _kontras_ortonormal(k)
    sigma = m.T @ s @ m
    eigval = np.linalg.eigvalsh(sigma)
    eigval = np.clip(eigval, 1e-12, None)  # derau numerik dapat membuatnya sedikit negatif

    tr = float(np.sum(eigval))
    det = float(np.prod(eigval))
    w = det / (tr / p) ** p

    f = n - 1
    d = 1 - (2 * p**2 + p + 2) / (6 * p * f)
    chi2 = float(-f * d * np.log(w)) if w > 0 else float("inf")
    df_chi2 = int(p * (p + 1) / 2 - 1)
    p_value = float(stats.chi2.sf(chi2, df_chi2)) if df_chi2 > 0 else float("nan")

    gg = min(1.0, float(tr**2 / (p * np.sum(eigval**2))))
    gg = max(gg, 1.0 / p)  # batas bawah baku epsilon
    hf_mentah = (n * p * gg - 2) / (p * (n - 1 - p * gg))
    hf = min(1.0, float(hf_mentah)) if np.isfinite(hf_mentah) else gg
    hf = max(hf, gg)  # Huynh-Feldt tidak pernah lebih rendah daripada Greenhouse-Geisser

    mauchly = MauchlyResult(w=w, chi_square=chi2, df=df_chi2, p_value=p_value, berlaku=True)
    return mauchly, gg, hf


@dataclass
class RepeatedMeasuresResult:
    """ANOVA pengukuran berulang satu faktor, beserta uji dan koreksi sphericity."""

    kondisi: list[str]
    n: int
    ringkasan: pd.DataFrame  # rata-rata & SD tiap kondisi
    anova: pd.DataFrame  # F, df, p tanpa koreksi
    mauchly: MauchlyResult
    gg_epsilon: float
    hf_epsilon: float
    terkoreksi: pd.DataFrame  # baris Greenhouse-Geisser dan Huynh-Feldt
    catatan: list[str] = field(default_factory=list)

    def kesimpulan(self, alpha: float = 0.05) -> str:
        p_dipakai = (
            float(self.terkoreksi.iloc[0]["p-value"])
            if not self.mauchly.terpenuhi
            else float(self.anova.iloc[0]["Pr > F"])
        )
        bentuk = "terkoreksi Greenhouse-Geisser" if not self.mauchly.terpenuhi else "tanpa koreksi"
        if p_dipakai < alpha:
            return f"Rata-rata berbeda signifikan antar kondisi ({bentuk}, p = {p_dipakai:.4f})."
        return f"Rata-rata tidak berbeda signifikan antar kondisi ({bentuk}, p = {p_dipakai:.4f})."


def run_repeated_measures(df: pd.DataFrame, kondisi: list[str]) -> RepeatedMeasuresResult:
    """ANOVA satu faktor pengukuran berulang: subjek yang sama diukur pada k kondisi.

    Berbeda dari ``run_manova`` (kelompok berbeda-subjek), di sini subjeknya sama pada
    seluruh kondisi — sehingga korelasi antar kondisi harus diperhitungkan, dan
    sphericity (ragam selisih antar kondisi yang seragam) menjadi asumsi kunci yang
    tidak muncul pada ANOVA/MANOVA antar-subjek biasa.
    """
    if len(kondisi) < 2:
        raise ValueError("Pengukuran berulang memerlukan minimal 2 kondisi.")
    lebar = df[kondisi].dropna()
    if len(lebar) < len(kondisi) + 2:
        raise ValueError("Jumlah subjek lengkap terlalu sedikit untuk jumlah kondisi ini.")

    n = len(lebar)
    ringkasan = pd.DataFrame(
        {
            "Kondisi": kondisi,
            "Rata-rata": [float(lebar[k].mean()) for k in kondisi],
            "SD": [float(lebar[k].std(ddof=1)) for k in kondisi],
            "n": n,
        }
    )

    panjang = lebar.reset_index(drop=True).reset_index(names="subjek").melt(
        id_vars="subjek", value_vars=kondisi, var_name="kondisi", value_name="nilai"
    )
    anova = AnovaRM(panjang, depvar="nilai", subject="subjek", within=["kondisi"]).fit().anova_table

    mauchly, gg, hf = mauchly_sphericity(lebar.to_numpy(float))

    df1 = float(anova.iloc[0]["Num DF"])
    df2 = float(anova.iloc[0]["Den DF"])
    f_stat = float(anova.iloc[0]["F Value"])
    baris_koreksi = []
    for nama, eps in (("Greenhouse-Geisser", gg), ("Huynh-Feldt", hf)):
        df1_k, df2_k = df1 * eps, df2 * eps
        p_k = float(stats.f.sf(f_stat, df1_k, df2_k))
        baris_koreksi.append(
            {
                "Koreksi": nama,
                "Epsilon": eps,
                "df1": df1_k,
                "df2": df2_k,
                "F": f_stat,
                "p-value": p_k,
                "Signifikan": "Ya" if p_k < 0.05 else "Tidak",
            }
        )
    terkoreksi = pd.DataFrame(baris_koreksi)

    catatan = []
    if not mauchly.terpenuhi:
        catatan.append(
            f"Sphericity dilanggar (Mauchly's W = {mauchly.w:.3f}, "
            f"χ²({mauchly.df}) = {mauchly.chi_square:.2f}, p = {mauchly.p_value:.4f}). "
            f"Derajat bebas dikoreksi Greenhouse-Geisser (ε = {gg:.3f}) karena lebih "
            "konservatif daripada Huynh-Feldt; laporkan versi terkoreksi, bukan yang mentah."
        )
    elif not mauchly.berlaku:
        catatan.append(
            "Hanya 2 kondisi: sphericity otomatis terpenuhi, tidak ada koreksi yang diperlukan."
        )
    else:
        catatan.append(
            f"Sphericity terpenuhi (Mauchly's W = {mauchly.w:.3f}, p = {mauchly.p_value:.4f}); "
            "ANOVA tanpa koreksi sah dipakai."
        )

    return RepeatedMeasuresResult(
        kondisi=list(kondisi),
        n=n,
        ringkasan=ringkasan,
        anova=anova,
        mauchly=mauchly,
        gg_epsilon=gg,
        hf_epsilon=hf,
        terkoreksi=terkoreksi,
        catatan=catatan,
    )
