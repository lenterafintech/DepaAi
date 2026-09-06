"""Regresi linear berganda dan regresi logistik biner."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from nalardata.assumptions import vif
from nalardata.preprocessing import design_matrix

# Jenis galat baku yang disediakan. Pilihan ini tidak mengubah koefisien B sama
# sekali — yang berubah hanya galat baku, nilai t, p, dan selang kepercayaannya.
# Keterangannya ditampilkan apa adanya agar pengguna memilih dengan sadar.
GALAT_BAKU: dict[str, dict[str, str]] = {
    "nonrobust": {
        "nama": "Biasa (OLS)",
        "kapan": "Ragam residual seragam (homoskedastisitas terpenuhi).",
        "catatan": (
            "Pilihan baku. Bila ragam residual ternyata tidak seragam, nilai p-nya "
            "menjadi terlalu optimistis sehingga pengaruh tampak lebih meyakinkan "
            "daripada yang sebenarnya."
        ),
    },
    "HC3": {
        "nama": "Robust HC3",
        "kapan": "Ragam residual tidak seragam, terutama pada sampel kecil-menengah.",
        "catatan": (
            "Anjuran umum ketika homoskedastisitas tidak terpenuhi (MacKinnon & White, "
            "1985). Paling berhati-hati di antara keluarga HC, sehingga aman dipakai "
            "sebagai pilihan pertama."
        ),
    },
    "HC1": {
        "nama": "Robust HC1",
        "kapan": "Sampel besar; padanan bawaan pada Stata.",
        "catatan": (
            "Koreksi derajat bebas sederhana. Berguna bila hasil ingin dibandingkan "
            "dengan keluaran Stata."
        ),
    },
    "HC0": {
        "nama": "Robust HC0 (White)",
        "kapan": "Rumusan asli White; sampel besar.",
        "catatan": (
            "Tanpa koreksi sampel kecil, sehingga cenderung terlalu longgar pada n kecil."
        ),
    },
    "HC2": {
        "nama": "Robust HC2",
        "kapan": "Jalan tengah antara HC1 dan HC3.",
        "catatan": "Mengoreksi leverage tanpa sekonservatif HC3.",
    },
}
GALAT_BAKU_BAWAAN = "nonrobust"


@dataclass
class LinearRegressionResult:
    model: object
    coefficients: pd.DataFrame
    fit: pd.DataFrame
    anova: pd.DataFrame
    diagnostics: pd.DataFrame
    vif: pd.DataFrame
    residuals: pd.Series
    fitted: pd.Series
    y_name: str
    predictors: list[str] = field(default_factory=list)
    cov_type: str = GALAT_BAKU_BAWAAN
    catatan: list[str] = field(default_factory=list)

    @property
    def nama_galat_baku(self) -> str:
        return GALAT_BAKU.get(self.cov_type, {}).get("nama", self.cov_type)

    @property
    def robust(self) -> bool:
        return self.cov_type != GALAT_BAKU_BAWAAN

    @property
    def r_squared(self) -> float:
        return float(self.model.rsquared)

    def equation(self, digits: int = 4) -> str:
        params = self.model.params
        terms = [f"{params.iloc[0]:.{digits}g}"]
        for name, value in params.iloc[1:].items():
            sign = "+" if value >= 0 else "-"
            terms.append(f" {sign} {abs(value):.{digits}g}·{name}")
        return f"{self.y_name} = " + "".join(terms)


def linear_regression(
    df: pd.DataFrame,
    y: str,
    predictors: list[str],
    alpha: float = 0.05,
    cov_type: str = GALAT_BAKU_BAWAAN,
) -> LinearRegressionResult:
    """Regresi linear berganda (OLS) lengkap dengan uji asumsi klasik.

    ``cov_type`` memilih cara galat baku dihitung. Koefisien B tidak terpengaruh —
    estimasi titiknya tetap OLS — yang berubah hanya galat baku beserta nilai t, p,
    dan selang kepercayaan yang diturunkan darinya. Lihat ``GALAT_BAKU``.
    """
    if y in predictors:
        raise ValueError("Variabel dependen tidak boleh menjadi prediktor.")
    if not predictors:
        raise ValueError("Pilih minimal satu variabel prediktor.")

    data = df[[y, *predictors]].dropna()
    if not pd.api.types.is_numeric_dtype(data[y]):
        raise ValueError(f"Variabel dependen '{y}' harus numerik.")
    if cov_type not in GALAT_BAKU:
        raise ValueError(
            f"Jenis galat baku '{cov_type}' tidak dikenal. "
            f"Pilih dari: {', '.join(GALAT_BAKU)}."
        )

    X, _ = design_matrix(data, predictors)
    X = sm.add_constant(X, has_constant="add")
    dasar = sm.OLS(data[y].astype(float), X)
    # Model klasik selalu dipasang: uji asumsi, jumlah kuadrat, dan uji F berbasis
    # SS diturunkan darinya sehingga tetap sebanding lintas pilihan galat baku.
    model_ols = dasar.fit()
    catatan: list[str] = []
    if cov_type == GALAT_BAKU_BAWAAN:
        model = model_ols
    else:
        model = dasar.fit(cov_type=cov_type)
        catatan.append(
            f"Galat baku dihitung dengan {GALAT_BAKU[cov_type]['nama']}. "
            "Koefisien B tetap sama dengan OLS biasa; yang berubah hanya galat baku, "
            "nilai t, p, dan selang kepercayaannya."
        )

    conf = model.conf_int(alpha=alpha)
    std_coef = _standardized_coefficients(model, data[y].astype(float), X)
    coefficients = pd.DataFrame(
        {
            "Variabel": model.params.index,
            "B": model.params.to_numpy(),
            "Std. Error": model.bse.to_numpy(),
            "Beta (baku)": std_coef,
            "t": model.tvalues.to_numpy(),
            "p-value": model.pvalues.to_numpy(),
            f"CI {int((1 - alpha) * 100)}% Bawah": conf[0].to_numpy(),
            f"CI {int((1 - alpha) * 100)}% Atas": conf[1].to_numpy(),
            "Signifikan": np.where(model.pvalues.to_numpy() < alpha, "Ya", "Tidak"),
        }
    )

    fit = pd.DataFrame(
        [
            {
                "Metrik": "R-squared",
                "Nilai": float(model.rsquared),
                "Keterangan": f"{model.rsquared * 100:.2f}% variasi {y} dijelaskan model",
            },
            {
                "Metrik": "Adjusted R-squared",
                "Nilai": float(model.rsquared_adj),
                "Keterangan": "Disesuaikan dengan jumlah prediktor",
            },
            {
                "Metrik": "Std. Error of Estimate",
                "Nilai": float(np.sqrt(model.mse_resid)),
                "Keterangan": "Simpangan baku residual",
            },
            {"Metrik": "AIC", "Nilai": float(model.aic), "Keterangan": "Makin kecil makin baik"},
            {"Metrik": "BIC", "Nilai": float(model.bic), "Keterangan": "Makin kecil makin baik"},
            {"Metrik": "N", "Nilai": int(model.nobs), "Keterangan": "Jumlah observasi terpakai"},
        ]
    )
    if cov_type != GALAT_BAKU_BAWAAN:
        # Dengan galat baku robust, uji kelayakan model berbentuk Wald, bukan rasio
        # jumlah kuadrat. Keduanya ditampilkan agar perbedaannya tidak menyesatkan.
        fit = pd.concat(
            [
                fit,
                pd.DataFrame(
                    [
                        {
                            "Metrik": "Uji F (Wald robust)",
                            "Nilai": float(model.fvalue),
                            "Keterangan": (
                                f"p = {float(model.f_pvalue):.4g}; memakai "
                                f"{GALAT_BAKU[cov_type]['nama']}"
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    anova = pd.DataFrame(
        [
            {
                "Sumber": "Regresi",
                "Sum of Squares": float(model_ols.ess),
                "df": float(model_ols.df_model),
                "Mean Square": (
                    float(model_ols.ess / model_ols.df_model) if model_ols.df_model else np.nan
                ),
                "F": float(model_ols.fvalue),
                "p-value": float(model_ols.f_pvalue),
            },
            {
                "Sumber": "Residual",
                "Sum of Squares": float(model_ols.ssr),
                "df": float(model_ols.df_resid),
                "Mean Square": float(model_ols.mse_resid),
                "F": np.nan,
                "p-value": np.nan,
            },
            {
                "Sumber": "Total",
                "Sum of Squares": float(model_ols.centered_tss),
                "df": float(model_ols.df_model + model_ols.df_resid),
                "Mean Square": np.nan,
                "F": np.nan,
                "p-value": np.nan,
            },
        ]
    )

    resid = pd.Series(model_ols.resid, index=data.index, name="Residual")
    diagnostics = _linear_diagnostics(model_ols, X, resid)
    tidak_seragam = bool(
        (
            diagnostics.loc[
                diagnostics["Asumsi"].str.contains("Homoskedastisitas"), "Kesimpulan"
            ]
            == "Tidak terpenuhi"
        ).any()
    )
    if tidak_seragam and cov_type == GALAT_BAKU_BAWAAN:
        catatan.append(
            "Ragam residual tidak seragam, sehingga galat baku biasa membuat nilai p "
            "terlalu optimistis. Pilih galat baku robust HC3 untuk memperbaikinya."
        )
    try:
        vif_table = vif(X.drop(columns=["const"]))
    except ValueError:
        vif_table = pd.DataFrame(columns=["Variabel", "R2", "VIF", "Tolerance", "Status"])

    return LinearRegressionResult(
        model=model,
        coefficients=coefficients,
        fit=fit,
        anova=anova,
        diagnostics=diagnostics,
        vif=vif_table,
        residuals=resid,
        fitted=pd.Series(model_ols.fittedvalues, index=data.index, name="Prediksi"),
        y_name=y,
        predictors=list(predictors),
        cov_type=cov_type,
        catatan=catatan,
    )


def saran_galat_baku(df: pd.DataFrame, y: str, predictors: list[str]) -> tuple[str, str]:
    """Jenis galat baku yang sesuai untuk data ini, beserta alasannya.

    Anjuran, bukan keharusan: pengguna tetap memilih sendiri, dan alasannya
    ditampilkan agar pilihan itu dapat dipertanggungjawabkan saat dilaporkan.
    """
    try:
        awal = linear_regression(df, y, predictors)
    except Exception:  # noqa: BLE001 - model gagal, biarkan pilihan bawaan
        return GALAT_BAKU_BAWAAN, "Model belum dapat diperiksa; pilihan baku dipakai."

    baris = awal.diagnostics[awal.diagnostics["Asumsi"].str.contains("Homoskedastisitas")]
    if baris.empty:
        return GALAT_BAKU_BAWAAN, "Homoskedastisitas tidak dapat diuji."

    # Cukup satu uji yang menolak untuk menganjurkan galat baku robust: keduanya
    # peka pada bentuk ketidakseragaman yang berbeda.
    menolak = [
        (str(b["Asumsi"]).split("(")[-1].rstrip(")"), float(b["p-value"]))
        for _, b in baris.iterrows()
        if np.isfinite(b["p-value"]) and float(b["p-value"]) < 0.05
    ]
    if menolak:
        rincian = "; ".join(f"{nama} p = {p:.4f}" for nama, p in menolak)
        return "HC3", (
            f"Homoskedastisitas ditolak ({rincian}), sehingga galat baku biasa "
            "membuat nilai p terlalu optimistis."
        )
    rincian = "; ".join(
        f"{str(b['Asumsi']).split('(')[-1].rstrip(')')} p = {float(b['p-value']):.4f}"
        for _, b in baris.iterrows()
        if np.isfinite(b["p-value"])
    )
    return GALAT_BAKU_BAWAAN, (
        f"Ragam residual cukup seragam ({rincian}); galat baku biasa memadai."
    )


def _standardized_coefficients(model, y: pd.Series, X: pd.DataFrame) -> np.ndarray:
    sy = y.std(ddof=1)
    betas = []
    for name in model.params.index:
        if name == "const" or sy == 0:
            betas.append(np.nan)
            continue
        sx = X[name].std(ddof=1)
        betas.append(float(model.params[name] * sx / sy) if sx > 0 else np.nan)
    return np.array(betas)


def _linear_diagnostics(model, X: pd.DataFrame, resid: pd.Series) -> pd.DataFrame:
    dw = float(durbin_watson(resid.to_numpy()))
    jb_stat, jb_p, _, _ = jarque_bera(resid.to_numpy())
    bp_stat, bp_p, _, _ = het_breuschpagan(resid.to_numpy(), X.to_numpy(float))
    # Breusch-Pagan hanya menangkap ragam yang berubah sebanding prediktor. Ragam
    # yang membesar simetris (misalnya menurut |x|) lolos dari uji itu, sehingga uji
    # White — yang menyertakan suku kuadrat dan perkalian silang — ikut dijalankan.
    try:
        with warnings.catch_warnings():
            # Rancangan White memuat suku kuadrat dan perkalian silang, yang menjadi
            # rank-deficient bila ada prediktor dummy. Statistiknya lalu tidak dapat
            # dipercaya, jadi peringatan itu diperlakukan sebagai kegagalan dan
            # hasilnya dilaporkan sebagai tidak dapat diuji — bukan angka semu.
            warnings.simplefilter("error")
            white_stat, white_p, _, _ = het_white(resid.to_numpy(), X.to_numpy(float))
    except Exception:  # noqa: BLE001 - rancangan tidak memadai untuk uji White
        white_stat = white_p = np.nan
    return pd.DataFrame(
        [
            {
                "Asumsi": "Normalitas residual (Jarque-Bera)",
                "Statistik": float(jb_stat),
                "p-value": float(jb_p),
                "Kesimpulan": "Terpenuhi" if jb_p > 0.05 else "Tidak terpenuhi",
            },
            {
                "Asumsi": "Homoskedastisitas (Breusch-Pagan)",
                "Statistik": float(bp_stat),
                "p-value": float(bp_p),
                "Kesimpulan": "Terpenuhi" if bp_p > 0.05 else "Tidak terpenuhi",
            },
            {
                "Asumsi": "Homoskedastisitas (White)",
                "Statistik": float(white_stat),
                "p-value": float(white_p),
                "Kesimpulan": (
                    "Tidak dapat diuji"
                    if not np.isfinite(white_p)
                    else ("Terpenuhi" if white_p > 0.05 else "Tidak terpenuhi")
                ),
            },
            {
                "Asumsi": "Non-autokorelasi (Durbin-Watson)",
                "Statistik": dw,
                "p-value": np.nan,
                "Kesimpulan": "Terpenuhi" if 1.5 <= dw <= 2.5 else "Perlu diperiksa",
            },
        ]
    )


@dataclass
class LogisticRegressionResult:
    model: object
    coefficients: pd.DataFrame
    fit: pd.DataFrame
    confusion: pd.DataFrame
    performance: pd.DataFrame
    roc: pd.DataFrame
    auc: float
    probabilities: pd.Series
    classes: tuple
    y_name: str


def logistic_regression(
    df: pd.DataFrame,
    y: str,
    predictors: list[str],
    positive_class: object | None = None,
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> LogisticRegressionResult:
    """Regresi logistik biner dengan odds ratio dan evaluasi klasifikasi."""
    if y in predictors:
        raise ValueError("Variabel dependen tidak boleh menjadi prediktor.")
    if not predictors:
        raise ValueError("Pilih minimal satu variabel prediktor.")

    data = df[[y, *predictors]].dropna()
    levels = pd.Series(data[y].unique()).sort_values().tolist()
    if len(levels) != 2:
        raise ValueError(
            f"Regresi logistik biner memerlukan tepat 2 kategori pada '{y}', "
            f"ditemukan {len(levels)}."
        )
    positive = positive_class if positive_class is not None else levels[-1]
    if positive not in levels:
        raise ValueError(f"Kategori '{positive}' tidak ada pada variabel '{y}'.")
    negative = [lv for lv in levels if lv != positive][0]

    y_bin = (data[y] == positive).astype(int)
    X, _ = design_matrix(data, predictors)
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y_bin, X).fit(disp=False, maxiter=200)

    conf = model.conf_int(alpha=alpha)
    coefficients = pd.DataFrame(
        {
            "Variabel": model.params.index,
            "B": model.params.to_numpy(),
            "Std. Error": model.bse.to_numpy(),
            "Wald z": model.tvalues.to_numpy(),
            "p-value": model.pvalues.to_numpy(),
            "Odds Ratio": np.exp(model.params.to_numpy()),
            "OR Bawah": np.exp(conf[0].to_numpy()),
            "OR Atas": np.exp(conf[1].to_numpy()),
            "Signifikan": np.where(model.pvalues.to_numpy() < alpha, "Ya", "Tidak"),
        }
    )

    ll, ll0 = float(model.llf), float(model.llnull)
    n = int(model.nobs)
    cox_snell = 1 - np.exp(2 * (ll0 - ll) / n)
    nagelkerke = cox_snell / (1 - np.exp(2 * ll0 / n))
    lr_stat = float(model.llr)
    fit = pd.DataFrame(
        [
            {"Metrik": "-2 Log Likelihood", "Nilai": -2 * ll},
            {"Metrik": "Cox & Snell R2", "Nilai": float(cox_snell)},
            {"Metrik": "Nagelkerke R2", "Nilai": float(nagelkerke)},
            {"Metrik": "McFadden R2", "Nilai": float(model.prsquared)},
            {"Metrik": "Likelihood Ratio Chi-square", "Nilai": lr_stat},
            {"Metrik": "p-value (model)", "Nilai": float(model.llr_pvalue)},
            {"Metrik": "N", "Nilai": n},
        ]
    )

    prob = pd.Series(model.predict(X), index=data.index, name="Probabilitas")
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_bin, pred, labels=[0, 1])
    confusion = pd.DataFrame(
        cm,
        index=[f"Aktual {negative}", f"Aktual {positive}"],
        columns=[f"Prediksi {negative}", f"Prediksi {positive}"],
    )

    auc = float(roc_auc_score(y_bin, prob)) if y_bin.nunique() == 2 else np.nan
    fpr, tpr, thr = roc_curve(y_bin, prob)
    performance = pd.DataFrame(
        [
            {"Metrik": "Akurasi", "Nilai": float(accuracy_score(y_bin, pred))},
            {"Metrik": "Presisi", "Nilai": float(precision_score(y_bin, pred, zero_division=0))},
            {"Metrik": "Recall (Sensitivitas)", "Nilai": float(recall_score(y_bin, pred, zero_division=0))},
            {
                "Metrik": "Spesifisitas",
                "Nilai": float(cm[0, 0] / cm[0].sum()) if cm[0].sum() else np.nan,
            },
            {"Metrik": "F1-Score", "Nilai": float(f1_score(y_bin, pred, zero_division=0))},
            {"Metrik": "AUC", "Nilai": auc},
        ]
    )

    return LogisticRegressionResult(
        model=model,
        coefficients=coefficients,
        fit=fit,
        confusion=confusion,
        performance=performance,
        roc=pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr}),
        auc=auc,
        probabilities=prob,
        classes=(negative, positive),
        y_name=y,
    )


def stepwise_selection(
    df: pd.DataFrame,
    y: str,
    predictors: list[str],
    p_enter: float = 0.05,
    p_remove: float = 0.10,
) -> tuple[list[str], pd.DataFrame]:
    """Seleksi variabel forward-backward berdasarkan p-value."""
    data = df[[y, *predictors]].dropna()
    X_all, _ = design_matrix(data, predictors)
    y_vec = data[y].astype(float)
    included: list[str] = []
    log: list[dict] = []

    while True:
        changed = False
        excluded = [c for c in X_all.columns if c not in included]
        new_p = pd.Series(dtype=float)
        for col in excluded:
            Xc = sm.add_constant(X_all[[*included, col]], has_constant="add")
            new_p[col] = sm.OLS(y_vec, Xc).fit().pvalues[col]
        if not new_p.empty and new_p.min() < p_enter:
            best = new_p.idxmin()
            included.append(best)
            log.append({"Langkah": "Masuk", "Variabel": best, "p-value": float(new_p.min())})
            changed = True

        if included:
            Xc = sm.add_constant(X_all[included], has_constant="add")
            pvals = sm.OLS(y_vec, Xc).fit().pvalues.iloc[1:]
            if not pvals.empty and pvals.max() > p_remove:
                worst = pvals.idxmax()
                included.remove(worst)
                log.append({"Langkah": "Keluar", "Variabel": worst, "p-value": float(pvals.max())})
                changed = True
        if not changed:
            break
    return included, pd.DataFrame(log)


def predict_linear(result: LinearRegressionResult, new_data: pd.DataFrame) -> pd.Series:
    """Prediksi nilai baru dari model regresi linear terlatih."""
    X, _ = design_matrix(new_data, result.predictors)
    X = sm.add_constant(X, has_constant="add")
    X = X.reindex(columns=result.model.params.index, fill_value=0.0)
    return pd.Series(result.model.predict(X), index=new_data.index, name="Prediksi")


def correlation_significance(r: float, n: int) -> float:
    """p-value dua sisi untuk koefisien korelasi Pearson."""
    if n <= 2 or abs(r) >= 1:
        return 0.0
    t = r * np.sqrt((n - 2) / (1 - r**2))
    return float(2 * stats.t.sf(abs(t), n - 2))
