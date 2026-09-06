"""Uji modul non-parametrik: ketepatan angka, ukuran efek, dan penolakan masukan cacat."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nalardata import nonparametrik as npar

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


@pytest.fixture(scope="module")
def acak() -> np.random.Generator:
    return np.random.default_rng(11)


# --------------------------------------------------------------------------- #
# Dua kelompok bebas
# --------------------------------------------------------------------------- #


def test_mann_whitney_sama_dengan_scipy(acak):
    a = pd.Series(acak.normal(10, 2, 60))
    b = pd.Series(acak.normal(12, 2, 55))
    hasil = npar.mann_whitney(a, b, "A", "B")
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert hasil.statistik == pytest.approx(u)
    assert hasil.p_value == pytest.approx(p)
    assert hasil.n == 115
    assert -1 <= hasil.efek_nilai <= 1
    assert hasil.padanan == "Uji-t sampel bebas"


def test_mann_whitney_menemukan_perbedaan_yang_ditanam(acak):
    a = pd.Series(acak.normal(10, 1, 80))
    b = pd.Series(acak.normal(13, 1, 80))
    hasil = npar.mann_whitney(a, b)
    assert hasil.signifikan
    assert hasil.efek_tafsir == "besar"


def test_mann_whitney_tidak_menemukan_perbedaan_semu(acak):
    a = pd.Series(acak.normal(10, 1, 200))
    b = pd.Series(acak.normal(10, 1, 200))
    assert not npar.mann_whitney(a, b).signifikan


def test_kelompok_terlalu_kecil_ditolak():
    with pytest.raises(ValueError, match="minimal 2"):
        npar.mann_whitney(pd.Series([1.0]), pd.Series([2.0, 3.0]))


def test_sampel_kecil_diberi_catatan(acak):
    hasil = npar.mann_whitney(
        pd.Series(acak.normal(size=8)), pd.Series(acak.normal(size=8))
    )
    assert hasil.catatan and "kurang dari 20" in hasil.catatan[0]


def test_ks_dan_mood_berjalan(data):
    seg = data["segmen_usaha"]
    tingkat = sorted(seg.dropna().unique())
    a = data.loc[seg == tingkat[0], "skor_kredit"]
    b = data.loc[seg == tingkat[1], "skor_kredit"]
    ks = npar.kolmogorov_smirnov_2(a, b)
    assert 0 <= ks.statistik <= 1
    mood = npar.mood_median(a, b)
    assert mood.tabel is not None and len(mood.tabel) == 2


# --------------------------------------------------------------------------- #
# Banyak kelompok
# --------------------------------------------------------------------------- #


def test_kruskal_sama_dengan_scipy(data):
    hasil = npar.kruskal_wallis(data["skor_kredit"], data["segmen_usaha"])
    kelompok = [
        bagian["skor_kredit"].to_numpy(float)
        for _, bagian in data.groupby("segmen_usaha")
    ]
    h, p = stats.kruskal(*kelompok)
    assert hasil.statistik == pytest.approx(h)
    assert hasil.p_value == pytest.approx(p)
    assert hasil.efek_nilai <= 1
    assert list(hasil.tabel.columns)[:2] == ["Kelompok", "n"]


def test_kruskal_menolak_kelompok_tunggal(data):
    satu = data.assign(tetap="A")
    with pytest.raises(ValueError, match="minimal 2 kelompok"):
        npar.kruskal_wallis(satu["skor_kredit"], satu["tetap"])


def test_dunn_menunjuk_pasangan_yang_berbeda(acak):
    df = pd.DataFrame(
        {
            "nilai": np.concatenate(
                [acak.normal(10, 1, 50), acak.normal(10, 1, 50), acak.normal(16, 1, 50)]
            ),
            "grup": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
        }
    )
    hasil = npar.dunn(df["nilai"], df["grup"])
    assert len(hasil) == 3
    pasangan = {
        frozenset([b["Kelompok A"], b["Kelompok B"]]): b["Keputusan"]
        for _, b in hasil.iterrows()
    }
    assert pasangan[frozenset(["A", "B"])] == "Tidak berbeda"
    assert pasangan[frozenset(["A", "C"])] == "Berbeda bermakna"
    assert pasangan[frozenset(["B", "C"])] == "Berbeda bermakna"


def test_koreksi_ganda_makin_ketat():
    p = np.array([0.01, 0.02, 0.04])
    tanpa = npar._koreksi_ganda(p, "tanpa")
    holm = npar._koreksi_ganda(p, "holm")
    bonf = npar._koreksi_ganda(p, "bonferroni")
    assert np.all(tanpa <= holm + 1e-12)
    assert np.all(holm <= bonf + 1e-12)
    # Holm menjaga urutan nilai p tetap monoton.
    assert np.all(np.diff(holm[np.argsort(p)]) >= -1e-12)
    assert np.all(npar._koreksi_ganda(np.array([0.9, 0.95]), "bonferroni") <= 1.0)


# --------------------------------------------------------------------------- #
# Berpasangan
# --------------------------------------------------------------------------- #


def test_wilcoxon_sama_dengan_scipy(acak):
    x = pd.Series(acak.normal(50, 8, 40))
    y = x + acak.normal(3, 2, 40)
    hasil = npar.wilcoxon(x, y)
    w, p = stats.wilcoxon(x, y)
    assert hasil.statistik == pytest.approx(w)
    assert hasil.p_value == pytest.approx(p)
    assert hasil.signifikan
    assert hasil.n == 40


def test_wilcoxon_menolak_pasangan_terlalu_sedikit():
    with pytest.raises(ValueError, match="minimal 5"):
        npar.wilcoxon(pd.Series([1.0, 2, 3]), pd.Series([2.0, 3, 4]))


def test_wilcoxon_menolak_data_tanpa_perbedaan():
    nilai = pd.Series([1.0, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="bernilai sama"):
        npar.wilcoxon(nilai, nilai)


def test_uji_tanda_hanya_menghitung_arah():
    x = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
    y = pd.Series([2.0, 3, 4, 5, 6, 7, 8, 0])
    hasil = npar.uji_tanda(x, y)
    assert hasil.statistik == 1  # hanya satu pasangan yang naik
    assert hasil.n == 8


def test_friedman_sama_dengan_scipy(acak):
    df = pd.DataFrame(
        {
            "t1": acak.normal(10, 2, 40),
            "t2": acak.normal(12, 2, 40),
            "t3": acak.normal(14, 2, 40),
        }
    )
    hasil = npar.friedman(df, ["t1", "t2", "t3"])
    chi2, p = stats.friedmanchisquare(df["t1"], df["t2"], df["t3"])
    assert hasil.statistik == pytest.approx(chi2)
    assert hasil.p_value == pytest.approx(p)
    assert 0 <= hasil.efek_nilai <= 1
    with pytest.raises(ValueError, match="minimal 3 pengukuran"):
        npar.friedman(df, ["t1", "t2"])


# --------------------------------------------------------------------------- #
# Kategorik
# --------------------------------------------------------------------------- #


def test_chi_square_sama_dengan_scipy(data):
    hasil = npar.chi_square(data["segmen_usaha"], data["gagal_bayar"], "segmen", "gagal")
    silang = pd.crosstab(data["segmen_usaha"], data["gagal_bayar"])
    chi2, p, _, _ = stats.chi2_contingency(silang)
    assert hasil.statistik == pytest.approx(chi2)
    assert hasil.p_value == pytest.approx(p)
    assert 0 <= hasil.efek_nilai <= 1


def test_chi_square_memperingatkan_sel_kecil():
    a = pd.Series(["x"] * 20 + ["y"] * 3)
    b = pd.Series(["p"] * 11 + ["q"] * 11 + ["p"])
    hasil = npar.chi_square(a, b)
    assert hasil.catatan and "frekuensi harapan" in hasil.catatan[0]


def test_fisher_hanya_untuk_tabel_2x2(data):
    with pytest.raises(ValueError, match="2×2"):
        npar.fisher_eksak(data["segmen_usaha"], data["gagal_bayar"])
    dua = data[data["segmen_usaha"] != data["segmen_usaha"].unique()[0]]
    hasil = npar.fisher_eksak(dua["segmen_usaha"], dua["gagal_bayar"])
    assert np.isfinite(hasil.p_value)


# --------------------------------------------------------------------------- #
# Korelasi peringkat
# --------------------------------------------------------------------------- #


def test_korelasi_peringkat_sama_dengan_scipy(data):
    spearman = npar.korelasi_peringkat(data["skor_kredit"], data["rasio_utang_pendapatan"])
    rho, p = stats.spearmanr(data["skor_kredit"], data["rasio_utang_pendapatan"])
    assert spearman.statistik == pytest.approx(rho)
    assert spearman.p_value == pytest.approx(p)
    kendall = npar.korelasi_peringkat(
        data["skor_kredit"], data["rasio_utang_pendapatan"], "kendall"
    )
    tau, p_tau = stats.kendalltau(data["skor_kredit"], data["rasio_utang_pendapatan"])
    assert kendall.statistik == pytest.approx(tau)
    assert kendall.p_value == pytest.approx(p_tau)


def test_korelasi_peringkat_tahan_hubungan_melengkung():
    x = pd.Series(np.arange(1, 51, dtype=float))
    y = x**3
    # Hubungan monoton sempurna: rho = 1 meski bentuknya tidak lurus.
    assert npar.korelasi_peringkat(x, y).statistik == pytest.approx(1.0)


def test_matriks_peringkat_simetris(data):
    kolom = ["skor_kredit", "usia", "saldo_tabungan"]
    koef, p = npar.matriks_peringkat(data, kolom)
    assert list(koef.columns) == kolom
    for a in kolom:
        for b in kolom:
            assert koef.loc[a, b] == pytest.approx(koef.loc[b, a])
            assert p.loc[a, b] == pytest.approx(p.loc[b, a])


# --------------------------------------------------------------------------- #
# Saran dan pelaporan
# --------------------------------------------------------------------------- #


def test_saran_nonparametrik_pada_data_menceng(acak):
    menceng = pd.DataFrame({"a": acak.exponential(2, 300)})
    perlu, alasan = npar.perlu_nonparametrik(menceng, ["a"])
    assert perlu and alasan


def test_saran_pada_sampel_kecil(acak):
    kecil = pd.DataFrame({"a": acak.normal(size=12)})
    perlu, alasan = npar.perlu_nonparametrik(kecil, ["a"])
    assert perlu and "12 baris" in alasan


def test_saran_tidak_memaksa_pada_data_normal(acak):
    normal = pd.DataFrame({"a": acak.normal(size=500), "b": acak.normal(size=500)})
    perlu, _ = npar.perlu_nonparametrik(normal, ["a", "b"])
    assert not perlu


def test_ringkas_memuat_angka_penting(acak):
    hasil = npar.mann_whitney(
        pd.Series(acak.normal(10, 1, 40)), pd.Series(acak.normal(13, 1, 40))
    )
    teks = hasil.ringkas()
    assert "U =" in teks and "p " in teks and "rank-biserial" in teks
    # Angka ditulis dengan koma sebagai pemisah desimal.
    assert "," in teks


def test_nilai_hilang_dibuang_berpasangan():
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    y = pd.Series([2.0, np.nan, 3.0, 5.0, 6.0, 7.0, 9.0])
    hasil = npar.wilcoxon(x, y)
    assert hasil.n == 5
