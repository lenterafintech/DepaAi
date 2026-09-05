"""Uji kebenaran perhitungan modul analisis multivariat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lentera_mva import (
    assumptions,
    cca,
    clustering,
    correlation,
    descriptive,
    discriminant,
    factor_analysis,
    manova,
    pca_analysis,
    preprocessing,
    regression,
)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Data sintetis dengan dua faktor laten dan tiga kelompok yang terpisah."""
    rng = np.random.default_rng(11)
    n = 300
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "x1": 0.9 * f1 + rng.normal(0, 0.35, n),
            "x2": 0.85 * f1 + rng.normal(0, 0.4, n),
            "x3": 0.8 * f1 + rng.normal(0, 0.45, n),
            "y1": 0.9 * f2 + rng.normal(0, 0.35, n),
            "y2": 0.85 * f2 + rng.normal(0, 0.4, n),
            "y3": 0.75 * f2 + rng.normal(0, 0.5, n),
        }
    )
    df["target"] = 2.5 + 1.5 * df["x1"] - 0.8 * df["y1"] + rng.normal(0, 0.5, n)
    df["kelompok"] = pd.cut(f1, [-np.inf, -0.5, 0.5, np.inf], labels=["A", "B", "C"]).astype(str)
    df["biner"] = (df["target"] > df["target"].median()).astype(int)
    return df


def test_describe_menghitung_statistik_dasar():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 10.0, 10.0, 10.0]})
    out = descriptive.describe(df).set_index("Variabel")
    assert out.loc["a", "Mean"] == pytest.approx(2.5)
    assert out.loc["a", "Median"] == pytest.approx(2.5)
    assert out.loc["a", "Std. Dev"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert out.loc["b", "Std. Dev"] == pytest.approx(0.0)


def test_missing_dan_penskalaan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
    assert len(preprocessing.handle_missing(df, "hapus baris")) == 1
    filled = preprocessing.handle_missing(df, "rata-rata")
    assert filled["a"].tolist() == [1.0, 2.0, 3.0]
    assert filled["b"].isna().sum() == 0

    scaled = preprocessing.scale(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), "z-score")
    assert scaled["a"].mean() == pytest.approx(0.0, abs=1e-12)
    assert scaled["a"].std(ddof=1) == pytest.approx(1.0)


def test_mardia_menolak_data_tak_normal(data):
    normal = data[["x1", "y1"]]
    assert descriptive.mardia_test(normal).skew_p > 0.01

    rng = np.random.default_rng(3)
    miring = pd.DataFrame(
        {"a": rng.exponential(1, 400), "b": rng.exponential(1, 400)}
    )
    assert descriptive.mardia_test(miring).skew_p < 0.01


def test_mahalanobis_menandai_pencilan_yang_disisipkan():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(0, 1, (200, 3)), columns=["a", "b", "c"])
    df.loc[len(df)] = [12.0, -11.0, 10.0]
    out = descriptive.mahalanobis_outliers(df, alpha=0.001)
    assert out.iloc[-1]["Pencilan"] == "Ya"
    assert (out["Pencilan"] == "Ya").sum() < 10


def test_korelasi_dan_signifikansi(data):
    result = correlation.correlation_matrix(data[["x1", "x2", "y1"]])
    assert result.matrix.loc["x1", "x2"] > 0.7
    assert result.p_values.loc["x1", "x2"] < 0.001
    assert result.p_values.loc["x1", "x1"] == 0.0
    pairs = result.significant_pairs()
    assert pairs.iloc[0]["Variabel 1"] in {"x1", "x2"}
    assert set(result.matrix.columns) == {"x1", "x2", "y1"}


def test_korelasi_parsial_meredam_hubungan_semu():
    rng = np.random.default_rng(9)
    z = rng.normal(0, 1, 500)
    df = pd.DataFrame(
        {"z": z, "a": z + rng.normal(0, 0.3, 500), "b": z + rng.normal(0, 0.3, 500)}
    )
    biasa = df.corr().loc["a", "b"]
    parsial = correlation.partial_correlation(df).loc["a", "b"]
    assert biasa > 0.8
    assert abs(parsial) < 0.2


def test_kmo_dan_bartlett(data):
    subset = data[["x1", "x2", "x3", "y1", "y2", "y3"]]
    kmo_result = assumptions.kmo(subset)
    assert 0.0 <= kmo_result.overall <= 1.0
    assert kmo_result.overall > 0.5
    assert assumptions.bartlett_sphericity(subset).p_value < 0.001


def test_vif_mendeteksi_multikolinearitas():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 200)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(0, 0.01, 200), "c": rng.normal(0, 1, 200)})
    table = assumptions.vif(df).set_index("Variabel")
    assert table.loc["a", "VIF"] > 10
    assert table.loc["c", "VIF"] < 2


def test_pca_konsisten_dengan_teori(data):
    subset = data[["x1", "x2", "x3", "y1", "y2", "y3"]]
    result = pca_analysis.run_pca(subset, n_components=2)
    # Pada matriks korelasi, total eigenvalue sama dengan jumlah variabel.
    assert result.eigenvalues.sum() == pytest.approx(subset.shape[1], rel=1e-6)
    assert result.cumulative_ratio[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result.eigenvalues) <= 1e-9)
    assert result.kaiser_components == 2
    assert result.scores.shape == (len(subset), 2)
    # Komponen utama saling ortogonal.
    assert result.scores.corr().iloc[0, 1] == pytest.approx(0.0, abs=1e-8)


def test_analisis_paralel_menemukan_dua_faktor(data):
    subset = data[["x1", "x2", "x3", "y1", "y2", "y3"]]
    out = pca_analysis.parallel_analysis(subset, n_iter=50)
    assert (out["Dipertahankan"] == "Ya").sum() == 2


@pytest.mark.parametrize("method", ["principal", "paf", "ml"])
def test_analisis_faktor_memisahkan_dua_gugus(data, method):
    subset = data[["x1", "x2", "x3", "y1", "y2", "y3"]]
    result = factor_analysis.run_factor_analysis(subset, 2, method, "varimax")
    mapping = result.dominant_loadings().set_index("Variabel")["Faktor Dominan"]
    assert mapping["x1"] == mapping["x2"] == mapping["x3"]
    assert mapping["y1"] == mapping["y2"] == mapping["y3"]
    assert mapping["x1"] != mapping["y1"]
    assert (result.communalities <= 1.001).all()


def test_varimax_mempertahankan_komunalitas():
    rng = np.random.default_rng(1)
    loadings = rng.normal(0, 0.5, (8, 3))
    rotated, rotation = factor_analysis.varimax(loadings)
    np.testing.assert_allclose(
        (loadings**2).sum(axis=1), (rotated**2).sum(axis=1), rtol=1e-8
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)


def test_kmeans_menemukan_klaster_yang_dibuat():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        np.vstack(
            [
                rng.normal(-6, 0.5, (60, 2)),
                rng.normal(0, 0.5, (60, 2)),
                rng.normal(6, 0.5, (60, 2)),
            ]
        ),
        columns=["a", "b"],
    )
    result = clustering.run_kmeans(df, 3)
    assert result.silhouette > 0.7
    assert sorted(result.sizes()["Anggota"].tolist()) == [60, 60, 60]

    diagnostics = clustering.kmeans_diagnostics(df, 2, 6)
    assert int(diagnostics.loc[diagnostics["Silhouette"].idxmax(), "k"]) == 3


def test_hierarchical_dan_dbscan():
    rng = np.random.default_rng(6)
    df = pd.DataFrame(
        np.vstack([rng.normal(-5, 0.4, (50, 2)), rng.normal(5, 0.4, (50, 2))]),
        columns=["a", "b"],
    )
    hier = clustering.run_hierarchical(df, 2)
    assert hier.n_clusters == 2
    assert hier.linkage_matrix is not None
    assert hier.silhouette > 0.7

    db = clustering.run_dbscan(df, eps=1.0, min_samples=5)
    assert db.n_clusters == 2


def test_profil_klaster_menandai_variabel_pembeda():
    rng = np.random.default_rng(8)
    df = pd.DataFrame(
        {
            "pembeda": np.concatenate([rng.normal(0, 1, 80), rng.normal(8, 1, 80)]),
            "netral": rng.normal(0, 1, 160),
        }
    )
    labels = pd.Series([1] * 80 + [2] * 80, index=df.index, name="Klaster")
    profile = clustering.profile_clusters(df, labels).set_index("Variabel")
    assert profile.loc["pembeda", "Pembeda"] == "Signifikan"
    assert profile.loc["netral", "p-value"] > 0.01


def test_regresi_linear_memulihkan_koefisien(data):
    result = regression.linear_regression(data, "target", ["x1", "y1"])
    coef = result.coefficients.set_index("Variabel")
    assert coef.loc["x1", "B"] == pytest.approx(1.5, abs=0.1)
    assert coef.loc["y1", "B"] == pytest.approx(-0.8, abs=0.1)
    assert coef.loc["const", "B"] == pytest.approx(2.5, abs=0.15)
    assert result.r_squared > 0.85
    assert set(result.diagnostics["Asumsi"]).issuperset({"Normalitas residual (Jarque-Bera)"})
    assert "target =" in result.equation()


def test_regresi_linear_menerima_prediktor_kategorik(data):
    result = regression.linear_regression(data, "target", ["x1", "kelompok"])
    names = result.coefficients["Variabel"].tolist()
    assert any(name.startswith("kelompok_") for name in names)


def test_regresi_logistik_menghasilkan_odds_ratio(data):
    result = regression.logistic_regression(data, "biner", ["x1", "y1"], positive_class=1)
    coef = result.coefficients.set_index("Variabel")
    assert coef.loc["x1", "Odds Ratio"] > 1
    assert coef.loc["y1", "Odds Ratio"] < 1
    assert result.auc > 0.9
    assert result.confusion.to_numpy().sum() == len(data)
    assert result.classes == (0, 1)


def test_regresi_logistik_menolak_target_multikelas(data):
    with pytest.raises(ValueError, match="tepat 2 kategori"):
        regression.logistic_regression(data, "kelompok", ["x1"])


def test_stepwise_membuang_prediktor_tak_relevan(data):
    rng = np.random.default_rng(12)
    df = data.copy()
    df["bising"] = rng.normal(0, 1, len(df))
    chosen, log = regression.stepwise_selection(df, "target", ["x1", "y1", "bising"])
    assert set(chosen) == {"x1", "y1"}
    assert not log.empty


def test_prediksi_linear_untuk_data_baru(data):
    result = regression.linear_regression(data, "target", ["x1", "y1"])
    new = pd.DataFrame({"x1": [0.0, 1.0], "y1": [0.0, 0.0]})
    pred = regression.predict_linear(result, new)
    assert pred.iloc[0] == pytest.approx(2.5, abs=0.15)
    assert pred.iloc[1] - pred.iloc[0] == pytest.approx(1.5, abs=0.1)


def test_diskriminan_mengklasifikasi_dengan_baik(data):
    result = discriminant.run_discriminant(data, "kelompok", ["x1", "x2", "x3"])
    assert result.accuracy > 0.7
    assert result.eigenvalues is not None
    assert result.wilks.iloc[0]["p-value"] < 0.001
    assert result.confusion.to_numpy().sum() == len(data)
    assert len(result.classes) == 3


def test_diskriminan_kuadratik_berjalan(data):
    result = discriminant.run_discriminant(data, "kelompok", ["x1", "x2", "x3"], "kuadratik")
    assert result.accuracy > 0.7
    assert result.eigenvalues is None


def test_manova_mendeteksi_perbedaan_kelompok(data):
    result = manova.run_manova(data, ["x1", "x2", "x3"], "kelompok")
    wilks = result.multivariate.set_index("Statistik").loc["Wilks' lambda"]
    assert wilks["p-value"] < 0.001
    assert set(result.multivariate["Statistik"]).issuperset({"Pillai's trace", "Wilks' lambda"})
    assert (result.univariate["Signifikan"] == "Ya").all()
    assert "perbedaan" in result.conclusion()


def test_manova_pada_kelompok_acak_tidak_signifikan():
    rng = np.random.default_rng(15)
    df = pd.DataFrame(rng.normal(0, 1, (240, 3)), columns=["a", "b", "c"])
    df["grup"] = rng.choice(["p", "q"], 240)
    result = manova.run_manova(df, ["a", "b", "c"], "grup")
    wilks = result.multivariate.set_index("Statistik").loc["Wilks' lambda"]
    assert wilks["p-value"] > 0.05


def test_hotelling_t2_sepakat_dengan_manova():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(1.2, 1, (60, 2))]), columns=["a", "b"]
    )
    df["grup"] = ["p"] * 60 + ["q"] * 60
    t2 = manova.hotelling_t2(df, ["a", "b"], "grup")
    mv = manova.run_manova(df, ["a", "b"], "grup")
    wilks_p = mv.multivariate.set_index("Statistik").loc["Wilks' lambda", "p-value"]
    assert t2["p-value"].iloc[0] == pytest.approx(float(wilks_p), rel=1e-6)


def test_box_m_membedakan_kovarians():
    rng = np.random.default_rng(21)
    sama = pd.DataFrame(rng.normal(0, 1, (300, 2)), columns=["a", "b"])
    grup_sama = pd.Series(rng.choice(["p", "q"], 300))
    assert assumptions.box_m(sama, grup_sama).p_value > 0.01

    beda = pd.DataFrame(
        np.vstack([rng.normal(0, 1, (150, 2)), rng.normal(0, 4, (150, 2))]), columns=["a", "b"]
    )
    grup_beda = pd.Series(["p"] * 150 + ["q"] * 150)
    assert assumptions.box_m(beda, grup_beda).p_value < 0.001


def test_cca_memulihkan_hubungan_antar_gugus(data):
    result = cca.run_cca(data, ["x1", "x2", "x3"], ["y1", "y2", "y3"])
    assert len(result.correlations) == 3
    assert np.all(np.diff(result.correlations) <= 1e-9)
    assert np.all((result.correlations >= 0) & (result.correlations <= 1))
    assert result.x_scores.shape[1] == 3

    rng = np.random.default_rng(23)
    df = data.copy()
    df["z1"] = df["x1"] * 0.9 + rng.normal(0, 0.2, len(df))
    df["z2"] = df["x2"] * 0.9 + rng.normal(0, 0.2, len(df))
    kuat = cca.run_cca(df, ["x1", "x2"], ["z1", "z2"])
    assert kuat.correlations[0] > 0.9
    assert kuat.significance.iloc[0]["p-value"] < 0.001


def test_cca_menolak_variabel_bertumpang_tindih(data):
    with pytest.raises(ValueError, match="kedua gugus"):
        cca.run_cca(data, ["x1", "x2"], ["x2", "y1"])
