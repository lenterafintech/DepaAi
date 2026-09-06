"""Uji aturan paket langganan dan pembatasan fitur."""

from __future__ import annotations

from nalardata import langganan as lg


def test_paket_tidak_dikenal_jatuh_ke_gratis():
    assert lg.ambil_paket(None).kode == "gratis"
    assert lg.ambil_paket("tidak-ada").kode == "gratis"
    assert lg.ambil_paket("PROFESIONAL").kode == "profesional"


def test_paket_gratis_membatasi_metode_lanjutan():
    gratis = lg.PAKET["gratis"]
    assert gratis.punya("regresi")
    assert not gratis.punya("sem")
    assert not gratis.punya("unduh_laporan")
    assert lg.periksa_fitur(gratis, "regresi") is None
    pelanggaran = lg.periksa_fitur(gratis, "sem")
    assert pelanggaran is not None
    assert pelanggaran.jenis == "fitur"
    # Paket termurah yang memuat SEM adalah Mahasiswa, bukan yang paling lengkap.
    assert pelanggaran.saran_paket == "mahasiswa"


def test_paket_profesional_membuka_seluruh_fitur():
    pro = lg.PAKET["profesional"]
    assert set(pro.fitur) == set(lg.FITUR)
    assert pro.fitur_terkunci() == []
    assert all(lg.periksa_fitur(pro, kode) is None for kode in lg.FITUR)


def test_empat_tingkatan_terurut():
    assert [p.kode for p in lg.urut_tingkatan()] == [
        "gratis", "mahasiswa", "profesional", "institusi",
    ]


def test_paket_mahasiswa_membuka_metode_akademik_bukan_ringkasan_profesional():
    mahasiswa = lg.PAKET["mahasiswa"]
    assert mahasiswa.punya("sem") and mahasiswa.punya("instrumen")
    assert mahasiswa.punya("ringkasan_akademik") and mahasiswa.punya("unduh_laporan")
    assert not mahasiswa.punya("ringkasan_profesional")


def test_harga_institusi_bersifat_kesepakatan():
    assert lg.harga_tampil(lg.PAKET["institusi"]) == "Sesuai kesepakatan"
    assert lg.harga_tampil(lg.PAKET["gratis"]) == "Gratis"
    assert lg.harga_tampil(lg.PAKET["mahasiswa"]) == "Rp 49.000/bulan"


def test_batas_ukuran_data():
    gratis = lg.PAKET["gratis"]
    assert lg.periksa_ukuran(gratis, 100, 5) is None
    baris = lg.periksa_ukuran(gratis, 5_000, 5)
    assert baris is not None and baris.jenis == "baris"
    # 5.000 baris masih muat pada paket Mahasiswa, jadi itulah yang disarankan.
    assert "300" in baris.pesan and baris.saran_paket == "mahasiswa"
    # Kalimatnya tetap utuh: pemisah ribuan tidak merusak tanda baca.
    assert "baris, sedangkan" in baris.pesan
    kolom = lg.periksa_ukuran(gratis, 100, 50)
    assert kolom is not None and kolom.jenis == "variabel"


def test_saran_paket_memilih_yang_termurah_dan_mencukupi():
    assert lg.paket_terkecil_dengan("dasar") == "gratis"
    assert lg.paket_terkecil_dengan("sem") == "mahasiswa"
    assert lg.paket_terkecil_dengan("ringkasan_profesional") == "profesional"
    assert lg.paket_terkecil_untuk_ukuran(100, 5) == "gratis"
    assert lg.paket_terkecil_untuk_ukuran(4_000, 20) == "mahasiswa"
    assert lg.paket_terkecil_untuk_ukuran(50_000, 20) == "profesional"
    assert lg.paket_terkecil_untuk_ukuran(400_000, 20) == "institusi"
    assert lg.paket_terkecil_untuk_ukuran(10_000_000, 20) is None


def test_ringkas_paket_memakai_format_rupiah():
    ringkas = dict(lg.ringkas_paket(lg.PAKET["profesional"]))
    assert ringkas["Harga"] == "Rp 149.000/bulan"
    assert "100.000 baris" in ringkas["Batas data"]
    assert dict(lg.ringkas_paket(lg.PAKET["gratis"]))["Harga"] == "Gratis"
