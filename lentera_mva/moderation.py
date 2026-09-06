"""Analisis regresi moderasi (moderated regression analysis).

Menguji apakah pengaruh sebuah prediktor terhadap variabel terikat berubah
menurut tingkat variabel moderator, melalui suku interaksi X × M.

Tiga hal yang membuat pembacaan hasil tidak menyesatkan disediakan di sini:
koefisien interaksi beserta perubahan R² yang ditimbulkannya, kemiringan
sederhana (simple slopes) pada beberapa tingkat moderator, serta rentang nilai
moderator tempat pengaruh X benar-benar signifikan (Johnson-Neyman).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from lentera_mva import preprocessing


@dataclass
class HasilModerasi:
    """Hasil regresi moderasi beserta pembanding model tanpa interaksi."""

    model: sm.regression.linear_model.RegressionResultsWrapper
    model_utama: sm.regression.linear_model.RegressionResultsWrapper
    y: str
    x: str
    m: str
    kontrol: list[str]
    dipusatkan: bool
    koefisien: pd.DataFrame
    rata_m: float
    sd_m: float
    rata_x: float
    sd_x: float
    n: int

    @property
    def nama_interaksi(self) -> str:
        return f"{self.x}_x_{self.m}"

    @property
    def delta_r2(self) -> float:
        """Tambahan ragam yang dijelaskan oleh suku interaksi."""
        return float(self.model.rsquared - self.model_utama.rsquared)

    def uji_perubahan(self) -> dict[str, float]:
        """Uji F atas tambahan R² akibat masuknya suku interaksi."""
        df1 = int(self.model.df_model - self.model_utama.df_model)
        df2 = int(self.model.df_resid)
        sisa = 1 - self.model.rsquared
        f = float(self.delta_r2 / df1 / (sisa / df2)) if df1 and sisa > 0 else float("nan")
        p = float(stats.f.sf(f, df1, df2)) if np.isfinite(f) else float("nan")
        return {"F": f, "df1": df1, "df2": df2, "p-value": p, "Delta R2": self.delta_r2}

    def koefisien_interaksi(self) -> pd.Series:
        return self.koefisien.set_index("Variabel").loc[self.nama_interaksi]

    def signifikan(self, alpha: float = 0.05) -> bool:
        return bool(self.koefisien_interaksi()["p-value"] < alpha)

    def kesimpulan(self, alpha: float = 0.05) -> str:
        if self.signifikan(alpha):
            return (
                f"{self.m} memoderasi pengaruh {self.x} terhadap {self.y}: pengaruhnya "
                "berbeda pada tingkat moderator yang berbeda."
            )
        return (
            f"Tidak ada bukti {self.m} memoderasi pengaruh {self.x} terhadap {self.y}; "
            "pengaruh X dapat dianggap seragam pada seluruh tingkat moderator."
        )


def _rapikan_koefisien(model, nama: list[str]) -> pd.DataFrame:
    ik = model.conf_int()
    return pd.DataFrame(
        {
            "Variabel": nama,
            "B": model.params.to_numpy(),
            "Std. Error": model.bse.to_numpy(),
            "t": model.tvalues.to_numpy(),
            "p-value": model.pvalues.to_numpy(),
            "IK 95% Bawah": ik.iloc[:, 0].to_numpy(),
            "IK 95% Atas": ik.iloc[:, 1].to_numpy(),
            "Signifikan": ["Ya" if p < 0.05 else "Tidak" for p in model.pvalues],
        }
    )


def regresi_moderasi(
    df: pd.DataFrame,
    y: str,
    x: str,
    m: str,
    kontrol: list[str] | None = None,
    pusatkan: bool = True,
) -> HasilModerasi:
    """Estimasi model Y = X + M + X×M (+ kontrol).

    Pemusatan (mean centering) dianjurkan karena tanpa itu koefisien X dan M pada
    model interaksi bermakna "pengaruh saat variabel lain bernilai nol" — nilai
    yang sering tidak pernah ada dalam data, sekaligus menaikkan multikolinearitas.
    """
    kontrol = [k for k in (kontrol or []) if k not in {y, x, m}]
    kolom = [y, x, m, *kontrol]
    hilang = [k for k in kolom if k not in df.columns]
    if hilang:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(hilang)}.")
    if x == m:
        raise ValueError("Prediktor dan moderator harus variabel yang berbeda.")

    data = preprocessing.clean_subset(df, kolom)
    if len(data) < len(kolom) + 5:
        raise ValueError("Observasi lengkap terlalu sedikit untuk model moderasi.")

    rata_m = float(data[m].mean())
    sd_m = float(data[m].std(ddof=1))
    rata_x = float(data[x].mean())
    sd_x = float(data[x].std(ddof=1))
    kerja = data.copy()
    if pusatkan:
        for kol in [x, m, *kontrol]:
            kerja[kol] = kerja[kol] - kerja[kol].mean()

    nama_interaksi = f"{x}_x_{m}"
    kerja[nama_interaksi] = kerja[x] * kerja[m]

    prediktor_utama = [x, m, *kontrol]
    X_utama = sm.add_constant(kerja[prediktor_utama], has_constant="add")
    X_penuh = sm.add_constant(kerja[[*prediktor_utama, nama_interaksi]], has_constant="add")
    model_utama = sm.OLS(kerja[y], X_utama).fit()
    model = sm.OLS(kerja[y], X_penuh).fit()

    return HasilModerasi(
        model=model,
        model_utama=model_utama,
        y=y,
        x=x,
        m=m,
        kontrol=kontrol,
        dipusatkan=pusatkan,
        koefisien=_rapikan_koefisien(model, list(X_penuh.columns)),
        rata_m=rata_m,
        sd_m=sd_m,
        rata_x=rata_x,
        sd_x=sd_x,
        n=len(data),
    )


def _slope_pada(hasil: HasilModerasi, nilai_m: float) -> tuple[float, float, float, float]:
    """Kemiringan X pada satu nilai moderator, beserta galat baku dan ujinya."""
    params = hasil.model.params
    cov = hasil.model.cov_params()
    b_x = float(params[hasil.x])
    b_int = float(params[hasil.nama_interaksi])
    var_x = float(cov.loc[hasil.x, hasil.x])
    var_int = float(cov.loc[hasil.nama_interaksi, hasil.nama_interaksi])
    kov = float(cov.loc[hasil.x, hasil.nama_interaksi])

    slope = b_x + b_int * nilai_m
    ragam = var_x + 2 * nilai_m * kov + nilai_m**2 * var_int
    se = float(np.sqrt(ragam)) if ragam > 0 else float("nan")
    t = slope / se if se and np.isfinite(se) else float("nan")
    p = float(2 * stats.t.sf(abs(t), hasil.model.df_resid)) if np.isfinite(t) else float("nan")
    return slope, se, t, p


def simple_slopes(hasil: HasilModerasi, kelipatan_sd: float = 1.0) -> pd.DataFrame:
    """Kemiringan X pada moderator rendah (−1 SD), rata-rata, dan tinggi (+1 SD)."""
    # Model dipusatkan memakai skala terpusat; nilai moderator dinyatakan relatif
    # terhadap rata-ratanya agar tabel terbaca dalam satuan aslinya.
    titik = {
        f"Rendah (−{kelipatan_sd:g} SD)": -kelipatan_sd * hasil.sd_m,
        "Rata-rata": 0.0,
        f"Tinggi (+{kelipatan_sd:g} SD)": kelipatan_sd * hasil.sd_m,
    }
    baris = []
    for label, geser in titik.items():
        nilai_model = geser if hasil.dipusatkan else hasil.rata_m + geser
        slope, se, t, p = _slope_pada(hasil, nilai_model)
        baris.append(
            {
                "Tingkat moderator": label,
                f"Nilai {hasil.m}": hasil.rata_m + geser,
                "Kemiringan X": slope,
                "Std. Error": se,
                "t": t,
                "p-value": p,
                "Signifikan": "Ya" if np.isfinite(p) and p < 0.05 else "Tidak",
            }
        )
    return pd.DataFrame(baris)


def johnson_neyman(hasil: HasilModerasi, alpha: float = 0.05) -> dict[str, object]:
    """Rentang nilai moderator tempat pengaruh X signifikan.

    Batas diperoleh dengan menyelesaikan persamaan kuadrat yang menyamakan nilai t
    kemiringan sederhana dengan nilai kritisnya, sehingga tidak bergantung pada
    pilihan titik ±1 SD yang sifatnya sembarang.
    """
    params = hasil.model.params
    cov = hasil.model.cov_params()
    b1 = float(params[hasil.x])
    b3 = float(params[hasil.nama_interaksi])
    v11 = float(cov.loc[hasil.x, hasil.x])
    v33 = float(cov.loc[hasil.nama_interaksi, hasil.nama_interaksi])
    v13 = float(cov.loc[hasil.x, hasil.nama_interaksi])
    t_kritis = float(stats.t.ppf(1 - alpha / 2, hasil.model.df_resid))

    a = b3**2 - t_kritis**2 * v33
    b = 2 * (b1 * b3 - t_kritis**2 * v13)
    c = b1**2 - t_kritis**2 * v11
    diskriminan = b**2 - 4 * a * c

    if abs(a) < 1e-12 or diskriminan < 0:
        return {"ada_batas": False, "batas": [], "catatan": "Tidak ada titik potong nyata."}

    akar = sorted(
        [
            (-b - np.sqrt(diskriminan)) / (2 * a),
            (-b + np.sqrt(diskriminan)) / (2 * a),
        ]
    )
    # Batas dikembalikan dalam satuan asli moderator.
    geser = hasil.rata_m if hasil.dipusatkan else 0.0
    batas = [float(r + geser) for r in akar]
    return {"ada_batas": True, "batas": batas, "catatan": ""}


def rentang_signifikan(hasil: HasilModerasi, alpha: float = 0.05) -> pd.DataFrame:
    """Ringkasan Johnson-Neyman: batas, serta apakah batas itu ada di dalam data."""
    jn = johnson_neyman(hasil, alpha)
    if not jn["ada_batas"]:
        return pd.DataFrame(
            [{"Batas": "-", "Di dalam rentang data": "-", "Keterangan": jn["catatan"]}]
        )
    minimum = hasil.rata_m - 3 * hasil.sd_m
    maksimum = hasil.rata_m + 3 * hasil.sd_m
    baris = []
    for nilai in jn["batas"]:
        di_dalam = minimum <= nilai <= maksimum
        baris.append(
            {
                "Batas": float(nilai),
                "Di dalam rentang data": "Ya" if di_dalam else "Tidak",
                "Keterangan": (
                    "Pengaruh X berubah status signifikansi pada nilai moderator ini."
                    if di_dalam
                    else "Di luar rentang nilai moderator yang teramati; abaikan."
                ),
            }
        )
    return pd.DataFrame(baris)


def data_plot_slopes(hasil: HasilModerasi, kelipatan_sd: float = 1.0) -> pd.DataFrame:
    """Titik-titik garis prediksi Y terhadap X pada tiga tingkat moderator."""
    params = hasil.model.params
    b0 = float(params["const"])
    b_x = float(params[hasil.x])
    b_m = float(params[hasil.m])
    b_int = float(params[hasil.nama_interaksi])

    # Sumbu X direntangkan ±2 SD di sekitar rata-ratanya, lalu dikembalikan ke
    # satuan asli agar sumbu grafik dapat dibaca pengguna.
    geser_x = np.linspace(-2, 2, 25) * hasil.sd_x
    baris = []
    for label, geser_m in {
        f"{hasil.m} rendah (−{kelipatan_sd:g} SD)": -kelipatan_sd * hasil.sd_m,
        f"{hasil.m} rata-rata": 0.0,
        f"{hasil.m} tinggi (+{kelipatan_sd:g} SD)": kelipatan_sd * hasil.sd_m,
    }.items():
        for g in geser_x:
            nilai_x = g if hasil.dipusatkan else hasil.rata_x + g
            nilai_m = geser_m if hasil.dipusatkan else hasil.rata_m + geser_m
            y_duga = b0 + b_x * nilai_x + b_m * nilai_m + b_int * nilai_x * nilai_m
            baris.append(
                {
                    "Tingkat moderator": label,
                    hasil.x: hasil.rata_x + g,
                    hasil.y: float(y_duga),
                }
            )
    return pd.DataFrame(baris)
