"""Uji kalkulator ukuran sampel.

Angka yang dihasilkan dibandingkan dengan hasil G*Power dan rumus baku, bukan
sekadar diperiksa "tidak galat". Kalkulator ukuran sampel yang meleset akan
menyesatkan penelitian sejak sebelum datanya ada.
"""

from __future__ import annotations

import pytest

from nalardata import sampel as sp


# --------------------------------------------------------------------------- #
# Rumus berbasis populasi
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "populasi, galat, harapan",
    [
        (1000, 0.05, 286),
        (1000, 0.10, 91),
        (100, 0.05, 80),
        (10_000, 0.05, 385),
    ],
)
def test_slovin_sesuai_rumus_baku(populasi, galat, harapan):
    assert sp.slovin(populasi, galat).n == harapan


def test_slovin_menyebutkan_batasnya():
    """Rumus yang paling sering dipakai adalah yang paling sering disalahpahami."""
    catatan = sp.slovin(500).catatan
    assert "keterwakilan" in catatan
    assert "tidak kembali" in catatan


def test_cochran_proporsi_setengah_menghasilkan_angka_klasik():
    """p = 0,5 dengan galat 5 persen menghasilkan 385, angka yang dikenal luas."""
    assert sp.cochran(0.5, 0.05).n == 385


def test_cochran_dengan_populasi_terhingga_lebih_kecil():
    besar = sp.cochran(0.5, 0.05).n
    kecil = sp.cochran(0.5, 0.05, populasi=500).n
    assert kecil < besar


@pytest.mark.parametrize("populasi", [0, -5])
def test_populasi_tidak_masuk_akal_ditolak(populasi):
    with pytest.raises(ValueError, match="lebih besar dari nol"):
        sp.slovin(populasi)


@pytest.mark.parametrize("galat", [0, 1, 1.5, -0.1])
def test_galat_di_luar_rentang_ditolak(galat):
    with pytest.raises(ValueError, match="di antara 0 dan 1"):
        sp.slovin(1000, galat)


# --------------------------------------------------------------------------- #
# Daya uji — dibandingkan dengan G*Power
# --------------------------------------------------------------------------- #


def test_uji_t_efek_sedang_sama_dengan_gpower():
    """d = 0,5, daya 0,80, alfa 0,05 menghasilkan 64 per kelompok."""
    hasil = sp.daya_uji_t(0.5)
    assert hasil.n == 128
    assert "64 responden per kelompok" in hasil.catatan


def test_anova_tiga_kelompok_sama_dengan_gpower():
    """f = 0,25, tiga kelompok, daya 0,80 menghasilkan 159 responden."""
    assert sp.daya_uji_anova(3, 0.25).n == 159


def test_regresi_tiga_prediktor_sama_dengan_gpower():
    """f² = 0,15, tiga prediktor, daya 0,80 menghasilkan 77 responden."""
    assert sp.daya_uji_regresi(3, 0.15).n == 77


def test_korelasi_efek_sedang_mendekati_gpower():
    """Hampiran Fisher z memberi 85; metode eksak memberi 84."""
    hasil = sp.daya_uji_korelasi(0.3)
    assert hasil.n == 85
    assert "Fisher" in hasil.catatan


def test_pengaruh_makin_kecil_menuntut_sampel_makin_besar():
    kecil = sp.daya_uji_t(sp.EFEK_D["kecil"]).n
    sedang = sp.daya_uji_t(sp.EFEK_D["sedang"]).n
    besar = sp.daya_uji_t(sp.EFEK_D["besar"]).n
    assert kecil > sedang > besar


def test_daya_lebih_tinggi_menuntut_sampel_lebih_besar():
    assert sp.daya_uji_t(0.5, daya=0.95).n > sp.daya_uji_t(0.5, daya=0.80).n


def test_prediktor_lebih_banyak_menuntut_sampel_lebih_besar():
    assert sp.daya_uji_regresi(8, 0.15).n > sp.daya_uji_regresi(2, 0.15).n


def test_anova_kurang_dari_dua_kelompok_ditolak():
    with pytest.raises(ValueError, match="dua kelompok"):
        sp.daya_uji_anova(1)


def test_regresi_tanpa_prediktor_ditolak():
    with pytest.raises(ValueError, match="satu prediktor"):
        sp.daya_uji_regresi(0)


@pytest.mark.parametrize("daya", [0, 1, 1.2])
def test_daya_di_luar_rentang_ditolak(daya):
    with pytest.raises(ValueError, match="di antara 0 dan 1"):
        sp.daya_uji_t(0.5, daya=daya)


def test_besar_pengaruh_nol_ditolak():
    with pytest.raises(ValueError, match="lebih besar dari nol"):
        sp.daya_uji_t(0)


# --------------------------------------------------------------------------- #
# Cadangan dan perbandingan
# --------------------------------------------------------------------------- #


def test_cadangan_non_respons_menaikkan_jumlah_sebaran():
    assert sp.dengan_cadangan(100, 0.8) == 125
    assert sp.dengan_cadangan(100, 1.0) == 100


def test_tingkat_pengembalian_tidak_masuk_akal_ditolak():
    with pytest.raises(ValueError, match="di antara 0 dan 1"):
        sp.dengan_cadangan(100, 1.5)


def test_perbandingan_menyandingkan_seluruh_metode():
    tabel = sp.bandingkan(
        [sp.slovin(1000), sp.daya_uji_t(0.5), sp.daya_uji_regresi(3)],
        tingkat_kembali=0.8,
    )
    assert len(tabel) == 3
    assert "Perlu disebar" in tabel.columns
    assert (tabel["Perlu disebar"] >= tabel["Ukuran sampel"]).all()


def test_perbandingan_kosong_ditolak():
    with pytest.raises(ValueError, match="Tidak ada hitungan"):
        sp.bandingkan([])


def test_slovin_dan_daya_uji_menjawab_pertanyaan_berbeda():
    """Inti pesan halaman ini: keduanya tidak boleh dianggap saling menggantikan.

    Untuk populasi besar Slovin berhenti di sekitar 400 berapa pun besar pengaruh
    yang dicari, sedangkan daya uji terus membesar ketika pengaruhnya kecil.
    """
    slovin = sp.slovin(1_000_000, 0.05).n
    daya_kecil = sp.daya_uji_t(sp.EFEK_D["kecil"]).n
    assert daya_kecil > slovin
