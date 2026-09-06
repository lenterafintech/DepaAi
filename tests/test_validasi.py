"""Uji harness validasi lintas software.

Uji ini adalah pagar terakhir terhadap kekeliruan penulisan kode pada lapisan yang
ditulis di sini: bila salah satu fungsi aplikasi mulai menghasilkan angka yang
berbeda dari R, uji ini gagal sebelum penggunanya menemukannya sendiri di sidang.
"""

from __future__ import annotations

import pytest

from nalardata import validasi as vl


@pytest.fixture(scope="module")
def hasil():
    return vl.jalankan()


def test_seluruh_pembandingan_dapat_dijalankan(hasil):
    gagal = hasil[hasil["Status"] == vl.GAGAL]
    assert gagal.empty, gagal[["Metode", "Keterangan"]].to_dict("records")


def test_tidak_ada_besaran_yang_berbeda_dari_acuan(hasil):
    """Selisih yang tersisa hanya boleh berasal dari pembulatan angka acuan."""
    berbeda = hasil[hasil["Status"] == vl.BERBEDA]
    assert berbeda.empty, berbeda[["Metode", "Besaran", "NalarData", "Acuan"]].to_dict(
        "records"
    )


@pytest.mark.parametrize("kode", list(vl.DAFTAR))
def test_setiap_acuan_menyebut_sumbernya(kode):
    """Acuan tanpa sumber tidak dapat diperiksa siapa pun."""
    acuan = vl.DAFTAR[kode]
    assert acuan.sumber.strip()
    assert acuan.pembanding.strip()
    assert acuan.butir


def test_acuan_welch_anova_mengaku_bukan_dari_r():
    """Acuan yang lebih lemah harus menyebut dirinya lebih lemah.

    Nilai R yang semula dipakai untuk Welch ANOVA ternyata keliru diingat -
    tertukar dengan F ANOVA biasa. Penggantinya berasal dari statsmodels, yang
    merupakan implementasi Python kedua, bukan pembanding yang benar-benar bebas.
    """
    acuan = vl.DAFTAR["welch_anova_plantgrowth"]
    assert "bukan R" in acuan.sumber
    assert "lebih lemah" in acuan.sumber


def test_ringkasan_memuat_metode_yang_belum_divalidasi():
    """Daftar yang menyembunyikan lubangnya sendiri tidak dapat dipercaya."""
    ringkas = vl.ringkas()
    belum = ringkas[ringkas["Status"] == vl.BELUM]
    assert len(belum) == len(vl.BELUM_DIVALIDASI)
    assert "CFA / SEM" in set(belum["Metode"])


def test_cakupan_tidak_dibulatkan_ke_atas():
    cakupan = vl.cakupan()
    assert cakupan["Metode divalidasi"] == len(
        {a.metode for a in vl.DAFTAR.values()}
    )
    assert cakupan["Belum divalidasi"] == len(vl.BELUM_DIVALIDASI)


def test_setiap_metode_belum_divalidasi_menyebut_sebabnya():
    for alasan in vl.BELUM_DIVALIDASI.values():
        assert len(alasan) > 20


def test_acuan_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        vl.jalankan("acuan_yang_tidak_ada")


def test_selisih_kecil_dinilai_sesuai_bukan_berbeda():
    assert vl._status(4.846088, 4.846, vl.TOLERANSI_LONGGAR) == vl.TOLERANSI
    assert vl._status(4.846, 4.846, vl.TOLERANSI_LONGGAR) == vl.SESUAI
    assert vl._status(5.9, 4.846, vl.TOLERANSI_LONGGAR) == vl.BERBEDA


def test_berkas_data_acuan_ikut_disertakan():
    """Acuan tanpa datanya tidak dapat dijalankan ulang siapa pun."""
    for acuan in vl.DAFTAR.values():
        assert (vl.ACUAN / acuan.data).exists(), acuan.data
