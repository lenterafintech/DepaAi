"""Uji aturan paket langganan dan pembatasan fitur."""

from __future__ import annotations

from lentera_mva import langganan as lg


def test_paket_tidak_dikenal_jatuh_ke_gratis():
    assert lg.ambil_paket(None).kode == "gratis"
    assert lg.ambil_paket("tidak-ada").kode == "gratis"
    assert lg.ambil_paket("PRO").kode == "pro"


def test_paket_gratis_membatasi_metode_lanjutan():
    gratis = lg.PAKET["gratis"]
    assert gratis.punya("regresi")
    assert not gratis.punya("sem")
    assert not gratis.punya("unduh_laporan")
    assert lg.periksa_fitur(gratis, "regresi") is None
    pelanggaran = lg.periksa_fitur(gratis, "sem")
    assert pelanggaran is not None
    assert pelanggaran.jenis == "fitur"
    assert pelanggaran.saran_paket == "pro"


def test_paket_pro_membuka_seluruh_fitur():
    pro = lg.PAKET["pro"]
    assert set(pro.fitur) == set(lg.FITUR)
    assert pro.fitur_terkunci() == []
    assert all(lg.periksa_fitur(pro, kode) is None for kode in lg.FITUR)


def test_batas_ukuran_data():
    gratis = lg.PAKET["gratis"]
    assert lg.periksa_ukuran(gratis, 100, 5) is None
    baris = lg.periksa_ukuran(gratis, 5_000, 5)
    assert baris is not None and baris.jenis == "baris"
    assert "300" in baris.pesan and baris.saran_paket == "pro"
    kolom = lg.periksa_ukuran(gratis, 100, 50)
    assert kolom is not None and kolom.jenis == "variabel"


def test_saran_paket_memilih_yang_termurah_dan_mencukupi():
    assert lg.paket_terkecil_dengan("dasar") == "gratis"
    assert lg.paket_terkecil_dengan("sem") == "pro"
    assert lg.paket_terkecil_untuk_ukuran(100, 5) == "gratis"
    assert lg.paket_terkecil_untuk_ukuran(10_000, 20) == "pro"
    assert lg.paket_terkecil_untuk_ukuran(400_000, 20) == "institusi"
    assert lg.paket_terkecil_untuk_ukuran(10_000_000, 20) is None


def test_ringkas_paket_memakai_format_rupiah():
    ringkas = dict(lg.ringkas_paket(lg.PAKET["pro"]))
    assert ringkas["Harga"] == "Rp 99.000/bulan"
    assert "50.000 baris" in ringkas["Batas data"]
    assert dict(lg.ringkas_paket(lg.PAKET["gratis"]))["Harga"] == "Gratis"
