"""Uji regresi moderasi: interaksi, simple slopes, dan Johnson-Neyman."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nalardata import moderation as mo


@pytest.fixture(scope="module")
def data_moderasi() -> pd.DataFrame:
    """Data dengan interaksi X×M yang ditanam sebesar 0,45."""
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    m = rng.normal(size=n)
    y = 0.3 * x + 0.2 * m + 0.45 * x * m + rng.normal(scale=0.7, size=n)
    return pd.DataFrame({"x": x, "m": m, "y": y, "kontrol": rng.normal(size=n)})


@pytest.fixture(scope="module")
def data_tanpa_moderasi() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=n)
    m = rng.normal(size=n)
    return pd.DataFrame({"x": x, "m": m, "y": 0.4 * x + 0.3 * m + rng.normal(size=n)})


def test_interaksi_yang_ditanam_terdeteksi(data_moderasi):
    hasil = mo.regresi_moderasi(data_moderasi, "y", "x", "m")
    koef = hasil.koefisien_interaksi()
    assert koef["p-value"] < 0.001
    assert koef["B"] == pytest.approx(0.45, abs=0.1)  # koefisien yang ditanam dipulihkan
    assert hasil.signifikan()
    assert "memoderasi" in hasil.kesimpulan()


def test_tanpa_interaksi_tidak_ditemukan_moderasi(data_tanpa_moderasi):
    hasil = mo.regresi_moderasi(data_tanpa_moderasi, "y", "x", "m")
    assert not hasil.signifikan()
    assert "Tidak ada bukti" in hasil.kesimpulan()
    assert hasil.delta_r2 < 0.01


def test_uji_perubahan_r2(data_moderasi):
    hasil = mo.regresi_moderasi(data_moderasi, "y", "x", "m")
    ubah = hasil.uji_perubahan()
    assert ubah["df1"] == 1
    assert ubah["Delta R2"] > 0.1
    assert ubah["p-value"] < 0.001
    # Uji F atas perubahan R² setara dengan uji t koefisien interaksi.
    assert ubah["F"] == pytest.approx(hasil.koefisien_interaksi()["t"] ** 2, rel=1e-6)


def test_simple_slopes_berbeda_arah(data_moderasi):
    hasil = mo.regresi_moderasi(data_moderasi, "y", "x", "m")
    slopes = mo.simple_slopes(hasil)
    assert list(slopes["Tingkat moderator"]) == [
        "Rendah (−1 SD)", "Rata-rata", "Tinggi (+1 SD)",
    ]
    rendah, tinggi = slopes.iloc[0], slopes.iloc[2]
    assert rendah["Kemiringan X"] < tinggi["Kemiringan X"]  # interaksi positif
    assert rendah["Signifikan"] == "Ya" and tinggi["Signifikan"] == "Ya"


def test_kemiringan_rata_rata_sama_dengan_koefisien_x(data_moderasi):
    hasil = mo.regresi_moderasi(data_moderasi, "y", "x", "m")
    slopes = mo.simple_slopes(hasil)
    b_x = float(hasil.koefisien.set_index("Variabel").loc["x", "B"])
    assert slopes.loc[1, "Kemiringan X"] == pytest.approx(b_x)


def test_johnson_neyman_memberi_batas(data_moderasi):
    hasil = mo.regresi_moderasi(data_moderasi, "y", "x", "m")
    jn = mo.johnson_neyman(hasil)
    assert jn["ada_batas"] and len(jn["batas"]) == 2
    assert jn["batas"][0] < jn["batas"][1]
    tabel = mo.rentang_signifikan(hasil)
    assert len(tabel) == 2 and set(tabel["Di dalam rentang data"]) <= {"Ya", "Tidak"}


def test_pemusatan_menurunkan_multikolinearitas(data_moderasi):
    dipusatkan = mo.regresi_moderasi(data_moderasi, "y", "x", "m", pusatkan=True)
    mentah = mo.regresi_moderasi(data_moderasi, "y", "x", "m", pusatkan=False)
    # Model setara: R² identik, hanya penafsiran koefisiennya yang berbeda.
    assert dipusatkan.model.rsquared == pytest.approx(mentah.model.rsquared, rel=1e-8)
    assert dipusatkan.koefisien_interaksi()["B"] == pytest.approx(
        mentah.koefisien_interaksi()["B"], rel=1e-6
    )


def test_variabel_kontrol_masuk_model(data_moderasi):
    hasil = mo.regresi_moderasi(data_moderasi, "y", "x", "m", kontrol=["kontrol"])
    assert "kontrol" in list(hasil.koefisien["Variabel"])
    assert hasil.kontrol == ["kontrol"]


def test_penolakan_masukan_keliru(data_moderasi):
    with pytest.raises(ValueError):
        mo.regresi_moderasi(data_moderasi, "y", "x", "x")
    with pytest.raises(ValueError):
        mo.regresi_moderasi(data_moderasi, "y", "x", "tidak_ada")


def test_data_plot_slopes_tiga_garis(data_moderasi):
    hasil = mo.regresi_moderasi(data_moderasi, "y", "x", "m")
    plot = mo.data_plot_slopes(hasil)
    assert plot["Tingkat moderator"].nunique() == 3
    assert set(plot.columns) == {"Tingkat moderator", "x", "y"}
