"""Uji deret waktu: uji stasioneritas (ADF), pencarian order, dan ARIMA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nalardata import arima_analysis as ar


@pytest.fixture(scope="module")
def acak() -> np.random.Generator:
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def deret_stasioner(acak) -> pd.Series:
    return pd.Series(acak.normal(0, 1, 80))


@pytest.fixture(scope="module")
def deret_random_walk(acak) -> pd.Series:
    """Random walk: bukti buku teks deret tidak stasioner (perlu differencing)."""
    langkah = acak.normal(0, 1, 150)
    return pd.Series(np.cumsum(langkah))


def test_deret_stasioner_terdeteksi_tanpa_differencing(deret_stasioner):
    hasil = ar.uji_stasioneritas(deret_stasioner)
    assert hasil.stasioner
    assert hasil.d == 0


def test_random_walk_perlu_differencing(deret_random_walk):
    hasil = ar.uji_stasioneritas(deret_random_walk)
    assert hasil.stasioner
    assert hasil.d >= 1


def test_deret_terlalu_pendek_ditolak(acak):
    with pytest.raises(ValueError):
        ar.uji_stasioneritas(pd.Series(acak.normal(0, 1, 5)))


def test_cari_order_mengembalikan_d_yang_diminta(deret_stasioner):
    order = ar.cari_order(deret_stasioner, d=0, maks_p=2, maks_q=2)
    assert order[1] == 0
    assert 0 <= order[0] <= 2
    assert 0 <= order[2] <= 2


def test_fit_arima_otomatis_pada_random_walk(deret_random_walk):
    hasil = ar.fit_arima(deret_random_walk)
    assert hasil.order[1] >= 1  # differencing otomatis terpakai
    assert hasil.n == len(deret_random_walk)
    assert np.isfinite(hasil.aic)
    assert np.isfinite(hasil.bic)


def test_fit_arima_order_eksplisit_dihormati(deret_stasioner):
    hasil = ar.fit_arima(deret_stasioner, order=(1, 0, 0))
    assert hasil.order == (1, 0, 0)


def test_fit_arima_deret_terlalu_pendek_ditolak(acak):
    with pytest.raises(ValueError):
        ar.fit_arima(pd.Series(acak.normal(0, 1, 10)))


def test_ringkasan_koefisien_berbentuk_tabel(deret_stasioner):
    hasil = ar.fit_arima(deret_stasioner, order=(1, 0, 0))
    tabel = hasil.ringkasan_koefisien()
    assert "ar.L1" in list(tabel["Parameter"])
    assert {"Koefisien", "Std. Error", "p-value"} <= set(tabel.columns)


def test_ljung_box_pada_deret_acak_tidak_signifikan(deret_stasioner):
    """Derau putih murni: setelah ARIMA(0,0,0) (hanya intersep), residual pada
    dasarnya deret aslinya sendiri — semestinya lolos Ljung-Box."""
    hasil = ar.fit_arima(deret_stasioner, order=(0, 0, 0))
    tabel = hasil.uji_ljung_box()
    assert "lb_pvalue" in tabel.columns
    assert bool((tabel["lb_pvalue"] >= 0.01).all())


def test_ramalkan_mengembalikan_jumlah_langkah_yang_diminta(deret_stasioner):
    hasil = ar.fit_arima(deret_stasioner, order=(1, 0, 0))
    ramalan = hasil.ramalkan(5)
    assert len(ramalan) == 5
    assert list(ramalan["Langkah"]) == [1, 2, 3, 4, 5]
    assert (ramalan["IK 95% Bawah"] <= ramalan["Perkiraan"]).all()
    assert (ramalan["Perkiraan"] <= ramalan["IK 95% Atas"]).all()


def test_ramalkan_langkah_kurang_dari_satu_ditolak(deret_stasioner):
    hasil = ar.fit_arima(deret_stasioner, order=(1, 0, 0))
    with pytest.raises(ValueError):
        hasil.ramalkan(0)
