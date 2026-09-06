"""MANCOVA: uji beda antar kelompok dengan kovariat dikendalikan.

MANCOVA menjawab pertanyaan yang tidak dapat dijawab MANOVA biasa: apakah kelompok
tetap berbeda setelah pengaruh variabel lain (kovariat) disingkirkan. Kovariat
lazimnya variabel yang sudah berbeda sejak awal antar kelompok — misalnya usia atau
lama usaha — sehingga tanpa dikendalikan, perbedaan yang terlihat bisa jadi berasal
dari kovariat itu, bukan dari kelompoknya.

Selain uji multivariat, modul ini menyediakan ANCOVA univariat per variabel dependen,
rata-rata terkoreksi (estimated marginal means), serta pemeriksaan asumsi kemiringan
regresi yang homogen — asumsi khas MANCOVA yang sering terlewat dilaporkan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA

from mv_statlab import preprocessing


@dataclass
class MancovaResult:
    multivariate: pd.DataFrame
    univariate: pd.DataFrame
    rata_terkoreksi: pd.DataFrame
    rata_mentah: pd.DataFrame
    pengaruh_kovariat: pd.DataFrame
    homogenitas_slope: pd.DataFrame
    group_sizes: pd.DataFrame
    dependents: list[str]
    factor: str
    covariates: list[str]
    n: int

    def signifikan(self, alpha: float = 0.05) -> bool:
        wilks = self.multivariate[self.multivariate["Statistik"] == "Wilks' lambda"]
        return bool(not wilks.empty and float(wilks["p-value"].iloc[0]) < alpha)

    def conclusion(self, alpha: float = 0.05) -> str:
        wilks = self.multivariate[self.multivariate["Statistik"] == "Wilks' lambda"]
        if wilks.empty or pd.isna(wilks["p-value"].iloc[0]):
            return "Hasil uji multivariat tidak tersedia."
        p = float(wilks["p-value"].iloc[0])
        kovariat = ", ".join(self.covariates)
        if p < alpha:
            return (
                f"Setelah {kovariat} dikendalikan, kelompok '{self.factor}' tetap "
                f"berbeda pada gabungan variabel dependen (Wilks' lambda, p = {p:.4f})."
            )
        return (
            f"Setelah {kovariat} dikendalikan, tidak terdapat perbedaan signifikan "
            f"antar kelompok '{self.factor}' (Wilks' lambda, p = {p:.4f})."
        )

    def slope_homogen(self, alpha: float = 0.05) -> bool:
        """True bila tidak ada interaksi faktor × kovariat yang signifikan."""
        if self.homogenitas_slope.empty:
            return True
        return bool((self.homogenitas_slope["p-value"] >= alpha).all())


def _tabel_multivariat(tabel: pd.DataFrame) -> pd.DataFrame:
    hasil = pd.DataFrame(
        {
            "Statistik": tabel.index,
            "Nilai": tabel["Value"].to_numpy(),
            "F": tabel["F Value"].to_numpy(),
            "df Hipotesis": tabel["Num DF"].to_numpy(),
            "df Galat": tabel["Den DF"].to_numpy(),
            "p-value": tabel["Pr > F"].to_numpy(),
        }
    )
    hasil["Signifikan"] = np.where(hasil["p-value"] < 0.05, "Ya", "Tidak")
    return hasil


def run_mancova(
    df: pd.DataFrame,
    dependents: list[str],
    factor: str,
    covariates: list[str],
) -> MancovaResult:
    """MANCOVA satu jalur dengan satu atau lebih kovariat numerik."""
    if len(dependents) < 2:
        raise ValueError("MANCOVA memerlukan minimal 2 variabel dependen numerik.")
    if not covariates:
        raise ValueError("MANCOVA memerlukan minimal 1 kovariat; tanpa itu gunakan MANOVA.")
    tumpang = set(dependents) & set(covariates)
    if tumpang:
        raise ValueError(
            f"Variabel tidak boleh menjadi dependen sekaligus kovariat: {', '.join(sorted(tumpang))}."
        )
    if factor in dependents or factor in covariates:
        raise ValueError("Variabel faktor tidak boleh dipakai sebagai dependen atau kovariat.")

    kolom = [*dependents, *covariates, factor]
    hilang = [k for k in kolom if k not in df.columns]
    if hilang:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(hilang)}.")

    data = df[kolom].dropna()
    for kol in [*dependents, *covariates]:
        if not pd.api.types.is_numeric_dtype(data[kol]):
            raise ValueError(f"Variabel '{kol}' harus numerik.")
    if data[factor].nunique() < 2:
        raise ValueError("Variabel faktor harus memiliki minimal 2 kelompok.")
    if len(data) <= len(kolom) + data[factor].nunique():
        raise ValueError("Observasi lengkap terlalu sedikit untuk MANCOVA.")

    # Nama kolom disederhanakan agar aman dipakai dalam formula statsmodels.
    nama_dv = {kol: f"dv{i}" for i, kol in enumerate(dependents)}
    nama_kov = {kol: f"kov{i}" for i, kol in enumerate(covariates)}
    kerja = data.rename(columns={**nama_dv, **nama_kov}).copy()
    kerja["grp"] = data[factor].astype(str).to_numpy()

    sisi_kanan = "grp + " + " + ".join(nama_kov.values())
    formula = f"{' + '.join(nama_dv.values())} ~ {sisi_kanan}"
    uji = MANOVA.from_formula(formula, data=kerja).mv_test()
    multivariate = _tabel_multivariat(uji.results["grp"]["stat"])

    baris_kovariat = []
    for asli, aman in nama_kov.items():
        tabel = _tabel_multivariat(uji.results[aman]["stat"])
        wilks = tabel[tabel["Statistik"] == "Wilks' lambda"].iloc[0]
        baris_kovariat.append(
            {
                "Kovariat": asli,
                "Wilks' Lambda": float(wilks["Nilai"]),
                "F": float(wilks["F"]),
                "p-value": float(wilks["p-value"]),
                "Signifikan": "Ya" if float(wilks["p-value"]) < 0.05 else "Tidak",
            }
        )
    pengaruh_kovariat = pd.DataFrame(baris_kovariat)

    # ANCOVA univariat dan rata-rata terkoreksi per variabel dependen.
    baris_univariat = []
    terkoreksi: dict[str, dict[str, float]] = {}
    rata_kovariat = {aman: float(kerja[aman].mean()) for aman in nama_kov.values()}
    kelompok = sorted(kerja["grp"].unique())

    for asli, aman in nama_dv.items():
        model = smf.ols(f"{aman} ~ grp + " + " + ".join(nama_kov.values()), data=kerja).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        if "grp" in anova.index:
            f = float(anova.loc["grp", "F"])
            p = float(anova.loc["grp", "PR(>F)"])
            ss_efek = float(anova.loc["grp", "sum_sq"])
            ss_galat = float(anova.loc["Residual", "sum_sq"])
            eta_parsial = ss_efek / (ss_efek + ss_galat) if (ss_efek + ss_galat) else np.nan
        else:  # pragma: no cover - hanya bila formula gagal terbentuk
            f = p = eta_parsial = float("nan")
        baris_univariat.append(
            {
                "Variabel": asli,
                "F": f,
                "p-value": p,
                "Eta-squared parsial": eta_parsial,
                "Signifikan": "Ya" if np.isfinite(p) and p < 0.05 else "Tidak",
            }
        )
        # Rata-rata terkoreksi: prediksi tiap kelompok saat kovariat pada rata-ratanya.
        titik = pd.DataFrame({"grp": kelompok, **{k: [v] * len(kelompok) for k, v in rata_kovariat.items()}})
        terkoreksi[asli] = dict(zip(kelompok, model.predict(titik).to_numpy(float)))

    univariate = pd.DataFrame(baris_univariat)
    rata_terkoreksi = pd.DataFrame(
        {factor: kelompok, **{dv: [terkoreksi[dv][g] for g in kelompok] for dv in dependents}}
    )
    rata_mentah = data.groupby(factor, observed=True)[dependents].mean().reset_index()

    # Asumsi kemiringan regresi homogen: interaksi faktor × kovariat harus tak signifikan.
    baris_slope = []
    for asli, aman in nama_dv.items():
        for kov_asli, kov_aman in nama_kov.items():
            model = smf.ols(f"{aman} ~ grp * {kov_aman}", data=kerja).fit()
            anova = sm.stats.anova_lm(model, typ=2)
            kunci = [i for i in anova.index if ":" in i]
            if not kunci:
                continue
            p = float(anova.loc[kunci[0], "PR(>F)"])
            baris_slope.append(
                {
                    "Variabel dependen": asli,
                    "Kovariat": kov_asli,
                    "F interaksi": float(anova.loc[kunci[0], "F"]),
                    "p-value": p,
                    "Kesimpulan": "Homogen" if p >= 0.05 else "Tidak homogen",
                }
            )
    homogenitas_slope = pd.DataFrame(baris_slope)

    ukuran = kerja["grp"].value_counts().sort_index()
    group_sizes = pd.DataFrame({"Kelompok": ukuran.index.astype(str), "N": ukuran.to_numpy()})

    return MancovaResult(
        multivariate=multivariate,
        univariate=univariate,
        rata_terkoreksi=rata_terkoreksi,
        rata_mentah=rata_mentah,
        pengaruh_kovariat=pengaruh_kovariat,
        homogenitas_slope=homogenitas_slope,
        group_sizes=group_sizes,
        dependents=list(dependents),
        factor=factor,
        covariates=list(covariates),
        n=len(data),
    )


def bandingkan_rata(hasil: MancovaResult) -> pd.DataFrame:
    """Sandingkan rata-rata mentah dan terkoreksi agar efek kovariat terlihat."""
    baris = []
    for _, mentah in hasil.rata_mentah.iterrows():
        kelompok = str(mentah[hasil.factor])
        koreksi = hasil.rata_terkoreksi[
            hasil.rata_terkoreksi[hasil.factor].astype(str) == kelompok
        ]
        if koreksi.empty:
            continue
        for dv in hasil.dependents:
            nilai_mentah = float(mentah[dv])
            nilai_koreksi = float(koreksi[dv].iloc[0])
            baris.append(
                {
                    "Kelompok": kelompok,
                    "Variabel": dv,
                    "Rata-rata mentah": nilai_mentah,
                    "Rata-rata terkoreksi": nilai_koreksi,
                    "Selisih": nilai_koreksi - nilai_mentah,
                }
            )
    return pd.DataFrame(baris)
