"""Uji regresi data panel: pengaruh tetap, pengaruh acak, dan uji Hausman."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nalardata import panel_analysis as pnl


def _panel(rng, n_entitas=10, n_waktu=12, korelasi_efek=0.0):
    """Panel seimbang. ``korelasi_efek`` > 0 membuat efek entitas berkorelasi
    dengan x1 — pelanggaran asumsi pengaruh acak yang harus terdeteksi Hausman."""
    baris = []
    for e in range(n_entitas):
        alpha = rng.normal(0, 3)
        for t in range(n_waktu):
            x1 = rng.normal(5, 2) + korelasi_efek * alpha
            x2 = rng.normal(0, 1)
            y = 2 + 1.2 * x1 - 0.8 * x2 + alpha + rng.normal(0, 1)
            baris.append(
                {"entitas": f"E{e}", "waktu": 2010 + t, "x1": x1, "x2": x2, "y": y}
            )
    return pd.DataFrame(baris)


@pytest.fixture(scope="module")
def acak() -> np.random.Generator:
    return np.random.default_rng(42)


def test_koefisien_fe_dan_re_mendekati_nilai_sebenarnya(acak):
    df = _panel(acak)
    hasil = pnl.regresi_panel(df, "y", ["x1", "x2"], "entitas", "waktu")
    b_fe = hasil.fe.params
    assert b_fe["x1"] == pytest.approx(1.2, abs=0.15)
    assert b_fe["x2"] == pytest.approx(-0.8, abs=0.15)
    b_re = hasil.re.fe_params
    assert b_re["x1"] == pytest.approx(1.2, abs=0.15)


def test_r2_within_antara_nol_dan_satu(acak):
    df = _panel(acak)
    hasil = pnl.regresi_panel(df, "y", ["x1", "x2"], "entitas", "waktu")
    assert 0.0 <= hasil.r2_within <= 1.0


def test_hausman_tidak_signifikan_saat_efek_tak_berkorelasi(acak):
    """Tanpa korelasi antara efek entitas dan prediktor, RE konsisten — Hausman
    semestinya tidak menolak H0."""
    df = _panel(acak, korelasi_efek=0.0)
    hasil = pnl.regresi_panel(df, "y", ["x1", "x2"], "entitas", "waktu")
    h = hasil.hausman()
    assert h["df"] == 2
    assert np.isnan(h["p-value"]) or h["p-value"] >= 0.05
    assert "acak" in hasil.kesimpulan().lower() or "hausman tidak dapat" in hasil.kesimpulan().lower()


def test_hausman_signifikan_saat_efek_berkorelasi(acak):
    """Efek entitas yang sengaja dibuat berkorelasi dengan x1 membuat RE tidak
    konsisten — Hausman semestinya menolak H0 dan merekomendasikan FE."""
    df = _panel(acak, n_entitas=12, n_waktu=15, korelasi_efek=0.9)
    hasil = pnl.regresi_panel(df, "y", ["x1"], "entitas", "waktu")
    h = hasil.hausman()
    assert h["p-value"] < 0.05
    assert "pengaruh tetap" in hasil.kesimpulan().lower()


def test_tanpa_kolom_waktu_tetap_berjalan(acak):
    df = _panel(acak).drop(columns=["waktu"])
    hasil = pnl.regresi_panel(df, "y", ["x1", "x2"], "entitas")
    assert hasil.waktu is None
    assert not hasil.efek_waktu


def test_kolom_tidak_ditemukan_ditolak(acak):
    df = _panel(acak)
    with pytest.raises(ValueError):
        pnl.regresi_panel(df, "y", ["tidak_ada"], "entitas")


def test_entitas_tunggal_ditolak(acak):
    df = _panel(acak, n_entitas=1)
    with pytest.raises(ValueError):
        pnl.regresi_panel(df, "y", ["x1"], "entitas")


def test_tanpa_prediktor_ditolak(acak):
    df = _panel(acak)
    with pytest.raises(ValueError):
        pnl.regresi_panel(df, "y", [], "entitas")


def test_koefisien_tabel_berisi_baris_yang_diminta(acak):
    df = _panel(acak)
    hasil = pnl.regresi_panel(df, "y", ["x1", "x2"], "entitas", "waktu")
    assert list(hasil.koefisien_fe["Variabel"]) == ["const", "x1", "x2"]
    assert list(hasil.koefisien_re["Variabel"]) == ["const", "x1", "x2"]
