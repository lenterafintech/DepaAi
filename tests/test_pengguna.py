"""Uji basis data pengguna, autentikasi, dan masa uji coba."""

from __future__ import annotations

from datetime import timedelta

import pytest

from lentera_mva import langganan as lg
from lentera_mva import pengguna as pg


@pytest.fixture
def db(tmp_path):
    """Basis data sementara agar pengujian tidak menyentuh data sungguhan."""
    berkas = tmp_path / "uji.db"
    pg.siapkan(berkas)
    return berkas


def buat_akun(db, surel="andi@contoh.com", paket="gratis"):
    return pg.daftar(surel, "Andi Pratama", "rahasia123", paket=paket, berkas=db)


def test_pendaftaran_menyimpan_akun(db):
    akun = buat_akun(db)
    assert akun.surel == "andi@contoh.com"
    assert akun.nama == "Andi Pratama"
    assert akun.paket == "gratis"
    assert akun.aktif
    assert pg.jumlah_pengguna(db) == 1


def test_surel_dibakukan_dan_divalidasi(db):
    akun = pg.daftar("  Budi@Contoh.COM ", "Budi", "rahasia123", berkas=db)
    assert akun.surel == "budi@contoh.com"
    with pytest.raises(pg.GalatPengguna):
        pg.daftar("bukan-surel", "Cici", "rahasia123", berkas=db)


def test_surel_ganda_ditolak(db):
    buat_akun(db)
    with pytest.raises(pg.GalatPengguna, match="sudah terdaftar"):
        buat_akun(db)


def test_kata_sandi_lemah_ditolak(db):
    with pytest.raises(pg.GalatPengguna, match="minimal"):
        pg.daftar("d@contoh.com", "Dedi", "abc12", berkas=db)
    with pytest.raises(pg.GalatPengguna, match="huruf dan angka"):
        pg.daftar("d@contoh.com", "Dedi", "hanyahuruf", berkas=db)


def test_kata_sandi_tidak_disimpan_apa_adanya(db):
    buat_akun(db)
    with pg.koneksi(db) as conn:
        baris = conn.execute("SELECT sandi_hash, garam FROM pengguna").fetchone()
    assert "rahasia123" not in baris["sandi_hash"]
    assert len(baris["sandi_hash"]) == 64  # SHA256 dalam heksadesimal
    assert baris["garam"]


def test_masuk_berhasil_dan_gagal(db):
    buat_akun(db)
    akun = pg.masuk("andi@contoh.com", "rahasia123", berkas=db)
    assert akun.terakhir_masuk is not None
    with pytest.raises(pg.GalatPengguna, match="Surel atau kata sandi salah"):
        pg.masuk("andi@contoh.com", "salah123", berkas=db)
    # Pesan gagal sama untuk surel tak dikenal, agar tidak jadi cara menebak akun.
    with pytest.raises(pg.GalatPengguna, match="Surel atau kata sandi salah"):
        pg.masuk("tidakada@contoh.com", "rahasia123", berkas=db)


def test_garam_berbeda_antar_pengguna(db):
    pg.daftar("a@contoh.com", "A", "rahasia123", berkas=db)
    pg.daftar("b@contoh.com", "B", "rahasia123", berkas=db)
    with pg.koneksi(db) as conn:
        baris = conn.execute("SELECT sandi_hash, garam FROM pengguna").fetchall()
    # Kata sandi identik tetap menghasilkan hash berbeda karena garamnya acak.
    assert baris[0]["garam"] != baris[1]["garam"]
    assert baris[0]["sandi_hash"] != baris[1]["sandi_hash"]


def test_uji_coba_membuka_seluruh_fitur(db):
    akun = buat_akun(db, paket="gratis")
    assert akun.dalam_uji_coba()
    assert akun.paket_efektif() == pg.PAKET_UJI_COBA
    assert lg.ambil_paket(akun.paket_efektif()).punya("sem")
    assert "uji coba" in akun.alasan_paket().lower()
    assert timedelta(days=pg.HARI_UJI_COBA) - akun.sisa_uji_coba() < timedelta(minutes=1)


def test_setelah_uji_coba_kembali_ke_paket_pilihan(db):
    buat_akun(db, paket="gratis")
    akun = pg.akhiri_uji_coba("andi@contoh.com", berkas=db)
    assert not akun.dalam_uji_coba()
    assert akun.paket_efektif() == "gratis"
    assert akun.sisa_uji_coba() == timedelta(0)
    assert not lg.ambil_paket(akun.paket_efektif()).punya("sem")


def test_uji_coba_tidak_menurunkan_paket_yang_lebih_tinggi(db):
    """Uji coba membuka fitur, bukan mengganti paket yang sudah dipilih pengguna."""
    akun = buat_akun(db, paket="institusi")
    assert akun.dalam_uji_coba()
    assert akun.paket == "institusi"
    setelah = pg.akhiri_uji_coba("andi@contoh.com", berkas=db)
    assert setelah.paket_efektif() == "institusi"


def test_set_paket_dan_masa_berlaku(db):
    buat_akun(db)
    akun = pg.set_paket("andi@contoh.com", "mahasiswa", berkas=db)
    assert akun.paket == "mahasiswa"
    assert akun.langganan_berlaku()  # tanpa tanggal akhir selama masa perkenalan

    kadaluarsa = pg._sekarang() - timedelta(days=1)
    akun = pg.set_paket("andi@contoh.com", "mahasiswa", kadaluarsa, berkas=db)
    pg.akhiri_uji_coba("andi@contoh.com", berkas=db)
    akun = pg.ambil("andi@contoh.com", berkas=db)
    assert not akun.langganan_berlaku()
    assert akun.paket_efektif() == "gratis"
    assert "berakhir" in akun.alasan_paket()


def test_ganti_sandi(db):
    buat_akun(db)
    pg.ganti_sandi("andi@contoh.com", "rahasia123", "sandibaru9", berkas=db)
    assert pg.masuk("andi@contoh.com", "sandibaru9", berkas=db)
    with pytest.raises(pg.GalatPengguna):
        pg.masuk("andi@contoh.com", "rahasia123", berkas=db)
    with pytest.raises(pg.GalatPengguna):  # sandi lama harus benar
        pg.ganti_sandi("andi@contoh.com", "salah123", "sandilain9", berkas=db)


def test_ambil_dengan_id_dan_daftar(db):
    akun = buat_akun(db)
    assert pg.ambil_dengan_id(akun.id, berkas=db).surel == akun.surel
    assert pg.ambil_dengan_id(9999, berkas=db) is None
    pg.daftar("z@contoh.com", "Zaki", "rahasia123", berkas=db)
    semua = pg.daftar_pengguna(berkas=db)
    assert [p.surel for p in semua] == ["z@contoh.com", "andi@contoh.com"]


def test_pengguna_tidak_ditemukan(db):
    with pytest.raises(pg.GalatPengguna, match="tidak ditemukan"):
        pg.ambil("hantu@contoh.com", berkas=db)
