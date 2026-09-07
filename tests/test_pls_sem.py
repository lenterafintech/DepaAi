"""Uji PLS-SEM: kasus berindikator tunggal harus persis sama dengan regresi/korelasi
biasa (PLS pada bentuk paling sederhana adalah regresi berantai), dan kasus
berindikator banyak harus memulihkan struktur yang ditanam."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nalardata import pls_sem as pls


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std(ddof=1)


def test_indikator_tunggal_identik_dengan_korelasi():
    """Konstruk berindikator satu tidak punya bobot untuk diestimasi — beta PLS-nya
    harus persis sama dengan korelasi Pearson, bukan sekadar mendekati."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    y = 0.6 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x, "y1": y})

    hasil = pls.jalankan(df, {"X": ["x1"], "Y": ["y1"]}, {"Y": ["X"]})
    assert hasil.konvergen
    beta = float(hasil.jalur_koefisien["Beta"].iloc[0])
    r = float(np.corrcoef(x, y)[0, 1])
    assert beta == pytest.approx(r, abs=1e-6)
    assert float(hasil.r2["Y"]) == pytest.approx(r**2, abs=1e-6)


def test_dua_prediktor_indikator_tunggal_identik_dengan_ols_standar():
    """Dengan seluruh konstruk berindikator satu, model struktural PLS harus sama
    persis dengan regresi OLS pada data yang distandardisasi."""
    rng = np.random.default_rng(1)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 0.4 * x1 - 0.3 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y})

    hasil = pls.jalankan(df, {"X1": ["x1"], "X2": ["x2"], "Y": ["y1"]}, {"Y": ["X1", "X2"]})
    mat = np.column_stack([_z(x1), _z(x2)])
    beta_ols, *_ = np.linalg.lstsq(mat, _z(y), rcond=None)

    peta = dict(zip(hasil.jalur_koefisien["Sumber"], hasil.jalur_koefisien["Beta"]))
    assert peta["X1"] == pytest.approx(beta_ols[0], abs=1e-6)
    assert peta["X2"] == pytest.approx(beta_ols[1], abs=1e-6)


def test_mediasi_indikator_tunggal_identik_dengan_dua_regresi_berantai():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(0, 1, n)
    m = 0.5 * x + rng.normal(0, 1, n)
    y = 0.3 * x + 0.4 * m + rng.normal(0, 1, n)
    df = pd.DataFrame({"x1": x, "m1": m, "y1": y})

    hasil = pls.jalankan(
        df, {"X": ["x1"], "M": ["m1"], "Y": ["y1"]}, {"M": ["X"], "Y": ["X", "M"]}
    )
    beta_xm, *_ = np.linalg.lstsq(_z(x).reshape(-1, 1), _z(m), rcond=None)
    mat_y = np.column_stack([_z(x), _z(m)])
    beta_y, *_ = np.linalg.lstsq(mat_y, _z(y), rcond=None)

    baris_m = hasil.jalur_koefisien[hasil.jalur_koefisien["Terikat"] == "M"].iloc[0]
    assert baris_m["Beta"] == pytest.approx(beta_xm[0], abs=1e-6)
    peta_y = {
        r["Sumber"]: r["Beta"]
        for _, r in hasil.jalur_koefisien[hasil.jalur_koefisien["Terikat"] == "Y"].iterrows()
    }
    assert peta_y["X"] == pytest.approx(beta_y[0], abs=1e-6)
    assert peta_y["M"] == pytest.approx(beta_y[1], abs=1e-6)


@pytest.fixture(scope="module")
def data_survei() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    kual = rng.normal(0, 1, n)
    puas = 0.5 * kual + rng.normal(0, 1, n) * np.sqrt(1 - 0.5**2)
    return pd.DataFrame(
        {
            "KUAL1": kual * 0.8 + rng.normal(0, 0.5, n),
            "KUAL2": kual * 0.75 + rng.normal(0, 0.5, n),
            "KUAL3": kual * 0.7 + rng.normal(0, 0.5, n),
            "PUAS1": puas * 0.8 + rng.normal(0, 0.5, n),
            "PUAS2": puas * 0.75 + rng.normal(0, 0.5, n),
            "PUAS3": puas * 0.7 + rng.normal(0, 0.5, n),
        }
    )


def test_model_reflektif_memulihkan_muatan_dan_jalur_yang_ditanam(data_survei):
    konstruk = {"KUAL": ["KUAL1", "KUAL2", "KUAL3"], "PUAS": ["PUAS1", "PUAS2", "PUAS3"]}
    hasil = pls.jalankan(data_survei, konstruk, {"PUAS": ["KUAL"]})
    assert hasil.konvergen

    # Muatan yang ditanam berkisar 0,7-0,8; PLS harus memulihkannya jauh di atas
    # ambang minimal 0,5 yang lazim dipakai (Hair dkk., 2019).
    assert (hasil.muatan["Muatan"] > 0.7).all()

    beta = float(hasil.jalur_koefisien["Beta"].iloc[0])
    # PLS diketahui sedikit meremehkan jalur struktural dibanding korelasi laten
    # sebenarnya (0,5) karena estimasi berbasis komposit, bukan CB-SEM yang
    # didisatenuasi — tetap harus positif dan cukup besar, bukan meleset arah.
    assert 0.2 < beta < 0.6

    cr_ave = hasil.cr_ave()
    assert (cr_ave["CR"] > 0.7).all()
    assert (cr_ave["AVE"] > 0.5).all()


def test_bootstrap_menangkap_jalur_signifikan(data_survei):
    konstruk = {"KUAL": ["KUAL1", "KUAL2", "KUAL3"], "PUAS": ["PUAS1", "PUAS2", "PUAS3"]}
    boot = pls.bootstrap_jalur(data_survei, konstruk, {"PUAS": ["KUAL"]}, n_boot=200, seed=3)
    assert len(boot) == 1
    baris = boot.iloc[0]
    assert baris["Signifikan"] == "Ya"
    assert baris["IK 95% Bawah"] > 0  # selang tidak melewati nol


def test_jalur_tak_bertumbuk_dengan_konstruk_terisolasi_tidak_gagal():
    """Konstruk yang tidak muncul di model jalur mana pun tidak boleh membuat
    algoritma gagal — skor konstruknya cukup dipertahankan apa adanya."""
    rng = np.random.default_rng(4)
    n = 150
    df = pd.DataFrame(
        {
            "a1": rng.normal(0, 1, n),
            "a2": rng.normal(0, 1, n),
            "b1": rng.normal(0, 1, n),
            "c1": rng.normal(0, 1, n),
        }
    )
    hasil = pls.jalankan(df, {"A": ["a1", "a2"], "B": ["b1"], "C": ["c1"]}, {"B": ["A"]})
    assert hasil.konvergen
    assert "C" in hasil.skor.columns


def test_menolak_konstruk_kurang_dari_dua():
    df = pd.DataFrame({"a1": [1, 2, 3, 4, 5, 6, 7, 8]})
    with pytest.raises(ValueError):
        pls.jalankan(df, {"A": ["a1"]}, {})


def test_menolak_nama_konstruk_pada_jalur_yang_tidak_ada():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"a1": rng.normal(0, 1, 50), "b1": rng.normal(0, 1, 50)})
    with pytest.raises(ValueError):
        pls.jalankan(df, {"A": ["a1"], "B": ["b1"]}, {"C": ["A"]})


def test_menolak_indikator_yang_tidak_ada_pada_data():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"a1": rng.normal(0, 1, 50), "b1": rng.normal(0, 1, 50)})
    with pytest.raises(ValueError):
        pls.jalankan(df, {"A": ["a1", "tidak_ada"], "B": ["b1"]}, {"B": ["A"]})
