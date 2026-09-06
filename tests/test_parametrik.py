"""Uji beda parametrik, diperiksa terhadap keluaran R yang terbit.

Modul ini lahir dari lubang yang ditemukan saat menyusun validasi lintas software:
Pemandu Uji menyarankan empat uji parametrik yang sama sekali belum ada di
aplikasi. Ujinya karena itu memeriksa dua hal sekaligus — angkanya benar, dan
uji yang disarankan pemandu benar-benar dapat dijalankan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalardata import parametrik as par

ROOT = Path(__file__).resolve().parents[1]
ACUAN = ROOT / "validasi" / "acuan"


@pytest.fixture(scope="module")
def plantgrowth() -> pd.DataFrame:
    return pd.read_csv(ACUAN / "plantgrowth.csv")


@pytest.fixture(scope="module")
def sleep() -> pd.DataFrame:
    return pd.read_csv(ACUAN / "sleep.csv")


# --------------------------------------------------------------------------- #
# Cocok dengan R
# --------------------------------------------------------------------------- #


def test_uji_t_bebas_cocok_dengan_r(sleep):
    """R: t.test(extra ~ group, data = sleep, var.equal = TRUE) → t = -1.8608."""
    hasil = par.uji_t_bebas(sleep["extra"], sleep["group"].astype(str), ragam_sama=True)
    assert hasil.statistik == pytest.approx(-1.8608, abs=1e-4)
    assert hasil.p_value == pytest.approx(0.07919, abs=1e-5)


def test_uji_t_welch_cocok_dengan_r(sleep):
    """R: t.test(extra ~ group, data = sleep) → t = -1.8608, df = 17.776."""
    hasil = par.uji_t_bebas(sleep["extra"], sleep["group"].astype(str), ragam_sama=False)
    assert hasil.statistik == pytest.approx(-1.8608, abs=1e-4)
    assert hasil.p_value == pytest.approx(0.07939, abs=1e-5)
    assert "17.776" in hasil.keterangan


def test_anova_satu_arah_cocok_dengan_r(plantgrowth):
    """R: summary(aov(weight ~ group, data = PlantGrowth)) → F = 4.846, p = 0.01591."""
    hasil = par.anova_satu_arah(plantgrowth["weight"], plantgrowth["group"])
    assert hasil.statistik == pytest.approx(4.846, abs=1e-3)
    assert hasil.p_value == pytest.approx(0.01591, abs=1e-5)


def test_welch_anova_cocok_dengan_statsmodels(plantgrowth):
    """Diperiksa terhadap statsmodels, bukan nilai R yang diingat.

    Nilai R yang semula dipakai ternyata keliru — tertukar dengan F ANOVA biasa.
    Derajat bebas penyebutnya yang benar, 17,128, tetap menjadi penanda bahwa
    rumus Welch-Satterthwaite-nya sudah tepat.
    """
    from statsmodels.stats.oneway import anova_oneway

    kelompok = [x["weight"].to_numpy() for _, x in plantgrowth.groupby("group")]
    acuan = anova_oneway(kelompok, use_var="unequal", welch_correction=True)
    hasil = par.welch_anova(plantgrowth["weight"], plantgrowth["group"])
    assert hasil.statistik == pytest.approx(float(acuan.statistic), rel=1e-6)
    assert hasil.p_value == pytest.approx(float(acuan.pvalue), rel=1e-6)
    assert "17.128" in hasil.keterangan


def test_tukey_cocok_dengan_r(plantgrowth):
    """R: TukeyHSD → trt1 vs trt2 selisih 0,865, p disesuaikan 0,012."""
    tabel = par.tukey(plantgrowth["weight"], plantgrowth["group"])
    baris = tabel[(tabel["Kelompok 1"] == "trt1") & (tabel["Kelompok 2"] == "trt2")]
    assert float(baris["Selisih rata-rata"].iloc[0]) == pytest.approx(0.865, abs=1e-3)
    assert float(baris["p disesuaikan"].iloc[0]) == pytest.approx(0.012, abs=1e-3)


def test_welch_anova_dua_kelompok_sama_dengan_kuadrat_t_welch(sleep):
    """Pemeriksaan silang internal: pada dua kelompok, F Welch = t Welch kuadrat."""
    f = par.welch_anova(sleep["extra"], sleep["group"].astype(str))
    t = par.uji_t_bebas(sleep["extra"], sleep["group"].astype(str), ragam_sama=False)
    assert f.statistik == pytest.approx(t.statistik**2, rel=1e-8)
    assert f.p_value == pytest.approx(t.p_value, rel=1e-8)


# --------------------------------------------------------------------------- #
# Besaran efek dan uji lanjutan
# --------------------------------------------------------------------------- #


def test_cohen_d_dilaporkan_bersama_uji_t(sleep):
    hasil = par.uji_t_bebas(sleep["extra"], sleep["group"].astype(str))
    assert hasil.efek_nama == "Cohen's d"
    assert np.isfinite(hasil.efek_nilai)
    assert hasil.efek_tafsir in {"sangat kecil", "kecil", "sedang", "besar"}


def test_eta_kuadrat_dilaporkan_bersama_anova(plantgrowth):
    hasil = par.anova_satu_arah(plantgrowth["weight"], plantgrowth["group"])
    assert hasil.efek_nama == "eta kuadrat"
    assert 0 <= hasil.efek_nilai <= 1


def test_anova_signifikan_mengarahkan_ke_uji_lanjutan(plantgrowth):
    hasil = par.anova_satu_arah(plantgrowth["weight"], plantgrowth["group"])
    assert any("Tukey" in c for c in hasil.catatan)


def test_uji_t_ragam_sama_mengingatkan_syaratnya(sleep):
    hasil = par.uji_t_bebas(sleep["extra"], sleep["group"].astype(str), ragam_sama=True)
    assert any("Levene" in c for c in hasil.catatan)


def test_welch_anova_mengarahkan_ke_games_howell(plantgrowth):
    hasil = par.welch_anova(plantgrowth["weight"], plantgrowth["group"])
    assert any("Games-Howell" in c for c in hasil.catatan)


def test_games_howell_menandai_pasangan_yang_berbeda(plantgrowth):
    tabel = par.games_howell(plantgrowth["weight"], plantgrowth["group"])
    assert len(tabel) == 3
    baris = tabel[(tabel["Kelompok 1"] == "trt1") & (tabel["Kelompok 2"] == "trt2")]
    assert baris["Berbeda"].iloc[0] == "Ya"


# --------------------------------------------------------------------------- #
# Uji berpasangan dan satu sampel
# --------------------------------------------------------------------------- #


def test_uji_t_berpasangan_cocok_dengan_r(sleep):
    """R: t.test(extra ~ group, data = sleep, paired = TRUE) → t = -4.0621."""
    a = sleep[sleep["group"] == 1]["extra"].reset_index(drop=True)
    b = sleep[sleep["group"] == 2]["extra"].reset_index(drop=True)
    hasil = par.uji_t_berpasangan(a, b)
    assert hasil.statistik == pytest.approx(-4.0621, abs=1e-4)
    assert hasil.p_value == pytest.approx(0.002833, abs=1e-6)


def test_uji_t_satu_sampel_membandingkan_dengan_acuan(sleep):
    hasil = par.uji_t_satu_sampel(sleep["extra"], 0.0)
    assert hasil.n == 20
    assert "Dibandingkan dengan 0" in hasil.keterangan


# --------------------------------------------------------------------------- #
# Masukan yang tidak sah
# --------------------------------------------------------------------------- #


def test_uji_t_menolak_kelompok_selain_dua(plantgrowth):
    with pytest.raises(ValueError, match="tepat 2 kelompok"):
        par.uji_t_bebas(plantgrowth["weight"], plantgrowth["group"])


def test_anova_menolak_kelompok_tunggal():
    df = pd.DataFrame({"y": [1.0, 2, 3, 4], "g": ["A"] * 4})
    with pytest.raises(ValueError, match="minimal 2 kelompok"):
        par.anova_satu_arah(df["y"], df["g"])


def test_welch_anova_menolak_kelompok_tanpa_keragaman():
    df = pd.DataFrame({"y": [1.0, 1, 1, 5, 6, 7], "g": ["A"] * 3 + ["B"] * 3})
    with pytest.raises(ValueError, match="tidak memiliki keragaman"):
        par.welch_anova(df["y"], df["g"])


def test_uji_t_berpasangan_menolak_pasangan_terlalu_sedikit():
    with pytest.raises(ValueError, match="minimal 3 pasangan"):
        par.uji_t_berpasangan(pd.Series([1.0, 2]), pd.Series([2.0, 3]))
