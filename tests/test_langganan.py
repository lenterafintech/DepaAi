"""Uji aturan paket langganan dan pembatasan fitur."""

from __future__ import annotations

from pathlib import Path

from nalardata import langganan as lg

ROOT = Path(__file__).resolve().parents[1]


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
    # Angka batas dibaca dari paketnya, bukan ditulis ulang di sini: uji yang
    # mengunci angka harfiah akan patah setiap kali batasnya ditinjau ulang.
    gratis = lg.PAKET["gratis"]
    assert lg.periksa_ukuran(gratis, 100, 5) is None

    baris = lg.periksa_ukuran(gratis, gratis.maks_baris + 1, 5)
    assert baris is not None and baris.jenis == "baris"
    assert lg._ribuan(gratis.maks_baris) in baris.pesan
    assert baris.saran_paket == "mahasiswa"
    # Kalimatnya tetap utuh: pemisah ribuan tidak merusak tanda baca.
    assert "baris, sedangkan" in baris.pesan

    kolom = lg.periksa_ukuran(gratis, 100, gratis.maks_variabel + 1)
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
    profesional = lg.PAKET["profesional"]
    assert lg._ribuan(profesional.maks_baris) + " baris" in ringkas["Batas data"]
    assert dict(lg.ringkas_paket(lg.PAKET["gratis"]))["Harga"] == "Gratis"


# --------------------------------------------------------------------------- #
# Contoh data bawaan
# --------------------------------------------------------------------------- #


def _contoh():
    import pandas as pd

    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


def test_contoh_data_bawaan_muat_pada_paket_gratis():
    """Onboarding tidak boleh terbentur dinding berbayar.

    Contoh data sengaja dibuat cukup besar agar seluruh metode dapat dijalankan
    di atasnya - SEM menuntut sekurang-kurangnya 200 responden, dan 18 kolom
    itulah yang membuat PCA, analisis faktor, dan korelasi kanonik bermakna.
    Karena contohnya tidak dapat dikecilkan, batas paket Gratis yang harus
    memuatnya. Uji ini menjaga agar keduanya tidak pernah berselisih lagi.
    """
    df = _contoh()
    assert lg.periksa_ukuran(lg.ambil_paket("gratis"), len(df), df.shape[1]) is None


def test_paket_gratis_memuat_kuesioner_skripsi_yang_lazim():
    """100-400 responden dengan 20-40 butir adalah ukuran skripsi yang lazim.

    Pembeda antar paket adalah metodenya, bukan banyaknya baris. Membatasi
    ukuran justru menutup pintu bagi pengguna yang paling membutuhkan pemandu
    uji, yakni yang belum mampu berlangganan.
    """
    gratis = lg.ambil_paket("gratis")
    assert lg.periksa_ukuran(gratis, 400, 45) is None
    assert lg.periksa_ukuran(gratis, 1_000, 50) is None


def test_batas_paket_naik_menurut_tingkatannya():
    tingkat = lg.urut_tingkatan()
    for lebih_rendah, lebih_tinggi in zip(tingkat, tingkat[1:]):
        assert lebih_tinggi.maks_baris > lebih_rendah.maks_baris
        assert lebih_tinggi.maks_variabel > lebih_rendah.maks_variabel


def test_data_yang_terlalu_besar_tetap_ditolak():
    pelanggaran = lg.periksa_ukuran(lg.ambil_paket("gratis"), 5_000, 10)
    assert pelanggaran is not None


def test_simulasi_sidang_punya_kode_fiturnya_sendiri():
    """Halaman yang menumpang kode fitur lain menampilkan pesan kunci yang keliru.

    Simulasi Sidang sempat memakai kode 'ringkasan_akademik', sehingga pengguna
    paket Gratis yang membukanya diberi tahu bahwa "Ringkasan akademik tidak
    termasuk paket Gratis" - halaman yang sama sekali berbeda.
    """
    assert "sidang" in lg.FITUR
    assert not lg.ambil_paket("gratis").punya("sidang")
    assert lg.ambil_paket("mahasiswa").punya("sidang")
