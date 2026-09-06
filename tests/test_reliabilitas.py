"""Uji perhitungan reliabilitas dan validitas instrumen."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lentera_mva import reliability as rb


@pytest.fixture(scope="module")
def kuesioner() -> pd.DataFrame:
    """Data simulasi: dua konstruk laten, masing-masing empat butir."""
    rng = np.random.default_rng(7)
    n = 300
    kualitas = rng.normal(size=n)
    kepuasan = 0.6 * kualitas + rng.normal(scale=0.8, size=n)
    data = {}
    for i in range(1, 5):
        data[f"KUAL{i}"] = kualitas + rng.normal(scale=0.5, size=n)
        data[f"PUAS{i}"] = kepuasan + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(data)


def test_alpha_tinggi_untuk_butir_satu_konstruk(kuesioner):
    hasil = rb.alpha_cronbach(kuesioner[[f"KUAL{i}" for i in range(1, 5)]])
    assert hasil.alpha > 0.8
    assert hasil.n_item == 4 and hasil.n_observasi == 300
    assert hasil.interpretasi() in {"Baik", "Sangat baik"}
    assert hasil.butir_bermasalah() == []
    assert list(hasil.item.columns) == [
        "Butir", "Rata-rata", "SD", "Korelasi item-total", "Alpha jika dibuang",
    ]


def test_alpha_rendah_saat_butir_tidak_berkaitan():
    rng = np.random.default_rng(3)
    acak = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    hasil = rb.alpha_cronbach(acak)
    assert hasil.alpha < 0.4
    assert hasil.interpretasi() == "Tidak dapat diterima"


def test_butir_menyimpang_terdeteksi(kuesioner):
    rng = np.random.default_rng(11)
    campur = kuesioner[[f"KUAL{i}" for i in range(1, 5)]].copy()
    campur["ASING"] = rng.normal(size=len(campur))  # butir yang tidak mengukur konstruk
    hasil = rb.alpha_cronbach(campur)
    assert "ASING" in hasil.butir_bermasalah()


def test_alpha_menolak_data_terlalu_sedikit():
    with pytest.raises(ValueError):
        rb.alpha_cronbach(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError):
        rb.alpha_cronbach(pd.DataFrame({"a": [1, 2], "b": [2, 3]}))


def test_muatan_faktor_positif_dan_cr_ave_konsisten(kuesioner):
    muatan = rb.muatan_faktor_tunggal(kuesioner[[f"PUAS{i}" for i in range(1, 5)]])
    assert (muatan > 0).all()  # tanda faktor dibakukan agar mayoritas positif
    cr, ave = rb.cr_ave(muatan)
    assert 0 < ave < 1
    assert cr > ave  # CR selalu melampaui AVE pada muatan yang seragam
    assert rb.omega_mcdonald(muatan) == pytest.approx(cr)


def test_analisis_konstruk_dan_tabelnya(kuesioner):
    konstruk = {
        "Kualitas": [f"KUAL{i}" for i in range(1, 5)],
        "Kepuasan": [f"PUAS{i}" for i in range(1, 5)],
    }
    hasil = rb.analisis_konstruk(kuesioner, konstruk)
    assert [h.nama for h in hasil] == ["Kualitas", "Kepuasan"]
    assert all(h.memenuhi() for h in hasil)
    assert all(h.catatan() == "Memenuhi" for h in hasil)
    tabel = rb.tabel_konstruk(hasil)
    assert list(tabel["Konstruk"]) == ["Kualitas", "Kepuasan"]
    assert (tabel["AVE"] > rb.AMBANG_AVE).all()


def test_konstruk_kurang_butir_ditolak(kuesioner):
    with pytest.raises(ValueError):
        rb.analisis_konstruk(kuesioner, {"Kualitas": ["KUAL1"]})


def test_fornell_larcker_dan_pemeriksaan_diskriminan(kuesioner):
    konstruk = {
        "Kualitas": [f"KUAL{i}" for i in range(1, 5)],
        "Kepuasan": [f"PUAS{i}" for i in range(1, 5)],
    }
    hasil = rb.analisis_konstruk(kuesioner, konstruk)
    matriks = rb.fornell_larcker(kuesioner, hasil)
    assert matriks.loc["Kualitas", "Kualitas"] == pytest.approx(hasil[0].akar_ave)
    assert np.isnan(matriks.loc["Kualitas", "Kepuasan"])  # segitiga atas dikosongkan
    assert not np.isnan(matriks.loc["Kepuasan", "Kualitas"])

    periksa = rb.periksa_diskriminan(kuesioner, hasil)
    assert len(periksa) == 1
    assert periksa.loc[0, "Keputusan"] == "Terpenuhi"


def test_diskriminan_gagal_saat_dua_konstruk_hampir_sama():
    rng = np.random.default_rng(5)
    laten = rng.normal(size=300)
    data = {f"A{i}": laten + rng.normal(scale=0.3, size=300) for i in range(1, 4)}
    data |= {f"B{i}": laten + rng.normal(scale=0.3, size=300) for i in range(1, 4)}
    df = pd.DataFrame(data)
    hasil = rb.analisis_konstruk(df, {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]})
    assert rb.periksa_diskriminan(df, hasil).loc[0, "Keputusan"] == "Tidak terpenuhi"


def test_tebak_konstruk_dari_pola_penamaan():
    kolom = ["KUAL1", "KUAL2", "KUAL3", "PUAS1", "PUAS2", "usia", "LOY1"]
    tebakan = rb.tebak_konstruk(kolom)
    assert tebakan == {"KUAL": ["KUAL1", "KUAL2", "KUAL3"], "PUAS": ["PUAS1", "PUAS2"]}


def test_skor_konstruk_rata_rata_butir(kuesioner):
    hasil = rb.analisis_konstruk(kuesioner, {"Kualitas": ["KUAL1", "KUAL2"]})
    skor = rb.skor_konstruk(kuesioner, hasil)
    diharapkan = kuesioner[["KUAL1", "KUAL2"]].mean(axis=1)
    assert np.allclose(skor["Kualitas"], diharapkan)
