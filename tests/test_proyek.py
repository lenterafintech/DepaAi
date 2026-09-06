"""Uji simpan dan buka berkas proyek, termasuk penolakan arsip yang tidak aman."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nalardata import kamus as km
from nalardata import keranjang as kr
from nalardata import proyek as pr
from nalardata import proyek_penelitian as pp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


@pytest.fixture
def isi_keranjang() -> kr.Keranjang:
    k = kr.Keranjang(judul="Laporan Uji", peneliti="Peneliti Satu")
    k.tambah_tabel(
        "Regresi linear",
        "Koefisien",
        pd.DataFrame({"Variabel": ["x1", "x2"], "B": [0.24, -0.01]}),
        catatan="Sel biru signifikan",
        tanda_data="data#1",
    )
    k.tambah_teks("Regresi linear", "Cara membaca", "Koefisien B menunjukkan perubahan.")
    return k


# --------------------------------------------------------------------------- #
# Perjalanan pulang-pergi
# --------------------------------------------------------------------------- #


def test_simpan_lalu_buka_memulihkan_seluruh_isi(data, isi_keranjang):
    konfig = {"variabel": ["usia", "skor_kredit"], "target_numerik": "skor_kredit"}
    berkas = pr.simpan_proyek(data, "contoh_data_nasabah.csv", isi_keranjang, konfig)
    hasil = pr.buka_proyek(berkas)

    assert hasil.nama_data == "contoh_data_nasabah.csv"
    assert hasil.data.shape == data.shape
    assert list(hasil.data.columns) == list(data.columns)
    assert hasil.konfigurasi == konfig
    assert hasil.versi == pr.VERSI
    assert hasil.dibuat

    assert len(hasil.keranjang.item) == 2
    assert hasil.keranjang.judul == "Laporan Uji"
    assert hasil.keranjang.peneliti == "Peneliti Satu"
    tabel = hasil.keranjang.item[0]
    assert tabel.catatan == "Sel biru signifikan"
    assert tabel.tanda_data == "data#1"
    assert tabel.tabel.loc[0, "B"] == pytest.approx(0.24)


def test_tipe_data_dipertahankan_lewat_parquet(data, isi_keranjang):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", isi_keranjang))
    # Tanpa Parquet, kolom bilangan bulat kerap berubah menjadi pecahan lewat CSV.
    assert hasil.data["usia"].dtype == data["usia"].dtype


def test_csv_selalu_ikut_sebagai_cadangan(data):
    arsip = zipfile.ZipFile(io.BytesIO(pr.simpan_proyek(data, "d.csv")))
    nama = arsip.namelist()
    assert "data.csv" in nama
    assert "proyek.json" in nama
    # Proyek tetap terbuka di komputer tanpa engine Parquet.
    manifest = json.loads(arsip.read("proyek.json"))
    assert "data.csv" in manifest["berkas_data"]


def test_proyek_tanpa_keranjang_tetap_sah(data):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv"))
    assert hasil.keranjang.kosong()
    assert hasil.konfigurasi == {}


def test_data_kosong_ditolak():
    with pytest.raises(ValueError, match="Tidak ada data aktif"):
        pr.simpan_proyek(pd.DataFrame())


def test_ringkas_menyebut_isi_utama(data, isi_keranjang):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", isi_keranjang))
    ringkas = hasil.ringkas().set_index("Keterangan")["Isi"]
    assert "400" in ringkas["Ukuran data"]
    assert ringkas["Hasil tersimpan"] == "2 objek"


def test_nama_berkas_proyek_dibersihkan():
    assert pr.nama_berkas_proyek("contoh data/nasabah.csv") == "contoh_data_nasabah.nalardata"
    assert pr.nama_berkas_proyek("") == "nalardata.nalardata"


# --------------------------------------------------------------------------- #
# Penolakan berkas cacat
# --------------------------------------------------------------------------- #


def test_berkas_kosong_ditolak():
    with pytest.raises(ValueError, match="kosong"):
        pr.buka_proyek(b"")


def test_bukan_arsip_ditolak():
    with pytest.raises(ValueError, match="rusak atau bukan arsip"):
        pr.buka_proyek(b"ini bukan zip sama sekali")


def test_arsip_tanpa_manifest_ditolak():
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w") as arsip:
        arsip.writestr("data.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="proyek.json tidak ditemukan"):
        pr.buka_proyek(penampung.getvalue())


def test_format_asing_ditolak():
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w") as arsip:
        arsip.writestr("proyek.json", json.dumps({"format": "aplikasi-lain", "versi": 1}))
        arsip.writestr("data.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="tidak dikenali"):
        pr.buka_proyek(penampung.getvalue())


def test_versi_belum_didukung_ditolak_dengan_nomornya():
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w") as arsip:
        arsip.writestr("proyek.json", json.dumps({"format": pr.FORMAT, "versi": 99}))
        arsip.writestr("data.csv", "a,b\n1,2\n")
    with pytest.raises(ValueError, match="versi 99 belum didukung"):
        pr.buka_proyek(penampung.getvalue())


def test_versi_tidak_berupa_angka_ditolak():
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w") as arsip:
        arsip.writestr("proyek.json", json.dumps({"format": pr.FORMAT, "versi": "dua"}))
    with pytest.raises(ValueError, match="Nomor versi"):
        pr.buka_proyek(penampung.getvalue())


def test_proyek_tanpa_data_ditolak():
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w") as arsip:
        arsip.writestr("proyek.json", json.dumps({"format": pr.FORMAT, "versi": 1}))
    with pytest.raises(ValueError, match="tidak memuat data utama"):
        pr.buka_proyek(penampung.getvalue())


# --------------------------------------------------------------------------- #
# Batas keamanan arsip
# --------------------------------------------------------------------------- #


def test_arsip_dengan_terlalu_banyak_anggota_ditolak():
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        arsip.writestr("proyek.json", json.dumps({"format": pr.FORMAT, "versi": 1}))
        for i in range(pr.MAKS_ANGGOTA + 1):
            arsip.writestr(f"isi/{i}.txt", "x")
    with pytest.raises(ValueError, match="melebihi batas aman"):
        pr.buka_proyek(penampung.getvalue())


def test_arsip_yang_membengkak_saat_dibuka_ditolak(monkeypatch):
    """Arsip kecil yang mengembang menjadi raksasa ditolak sebelum dibaca.

    Batasnya diturunkan agar uji tetap cepat; yang diperiksa adalah rasio antara
    ukuran arsip dan ukuran isinya, bukan angka mutlaknya.
    """
    monkeypatch.setattr(pr, "MAKS_TERBUKA", 1_000_000)
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        arsip.writestr("proyek.json", json.dumps({"format": pr.FORMAT, "versi": 1}))
        # Nol yang sangat mudah dimampatkan: arsipnya kecil, isinya melewati batas.
        arsip.writestr("data.csv", b"0" * (pr.MAKS_TERBUKA + 1))
    padat = penampung.getvalue()
    # Arsipnya sendiri jauh lebih kecil daripada isinya — inilah bentuk zip bomb.
    assert len(padat) < pr.MAKS_TERBUKA // 100
    with pytest.raises(ValueError, match="membengkak"):
        pr.buka_proyek(padat)


def test_berkas_melebihi_batas_ukuran_ditolak(monkeypatch):
    monkeypatch.setattr(pr, "MAKS_BERKAS", 100)
    with pytest.raises(ValueError, match="melebihi batas aman"):
        pr.buka_proyek(b"x" * 200)


# --------------------------------------------------------------------------- #
# Ketahanan terhadap isi yang tidak lengkap
# --------------------------------------------------------------------------- #


def test_tabel_keranjang_yang_hilang_dilewati_tanpa_menggagalkan_proyek(data, isi_keranjang):
    berkas = pr.simpan_proyek(data, "d.csv", isi_keranjang)
    sumber = zipfile.ZipFile(io.BytesIO(berkas))

    # Susun ulang arsip tanpa berkas tabel keranjangnya.
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        for nama in sumber.namelist():
            if nama.startswith("keranjang/"):
                continue
            arsip.writestr(nama, sumber.read(nama))

    hasil = pr.buka_proyek(penampung.getvalue())
    # Tabelnya hilang, tetapi datanya dan item teks tetap pulih.
    assert not hasil.data.empty
    assert [i.jenis for i in hasil.keranjang.item] == ["tafsiran"]


def test_konfigurasi_rusak_tidak_menggagalkan_pembukaan(data):
    berkas = pr.simpan_proyek(data, "d.csv")
    sumber = zipfile.ZipFile(io.BytesIO(berkas))
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        for nama in sumber.namelist():
            arsip.writestr(
                nama, b"{bukan json" if nama == "konfigurasi.json" else sumber.read(nama)
            )
    assert pr.buka_proyek(penampung.getvalue()).konfigurasi == {}


@pytest.mark.parametrize("format_lama", sorted(pr.FORMAT_LAMA))
def test_proyek_dari_nama_aplikasi_lama_tetap_terbuka(data, format_lama):
    """Berkas yang disimpan sebelum aplikasi berganti nama tidak boleh ditolak.

    Aplikasi sudah dua kali berganti nama (Lentera MVA, MV Statlab, NalarData).
    Setiap nama format lama diuji, bukan hanya yang terakhir.
    """
    berkas = pr.simpan_proyek(data, "d.csv")
    sumber = zipfile.ZipFile(io.BytesIO(berkas))
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        for nama in sumber.namelist():
            isi = sumber.read(nama)
            if nama == "proyek.json":
                manifest = json.loads(isi)
                manifest["format"] = format_lama
                isi = json.dumps(manifest).encode("utf-8")
            arsip.writestr(nama, isi)

    hasil = pr.buka_proyek(penampung.getvalue())
    assert hasil.data.shape == data.shape


def test_ekstensi_berkas_mengikuti_nama_aplikasi():
    assert pr.EKSTENSI == "nalardata"
    assert pr.nama_berkas_proyek("data.csv").endswith(".nalardata")


# --------------------------------------------------------------------------- #
# Kamus variabel di dalam proyek
# --------------------------------------------------------------------------- #


@pytest.fixture
def isi_kamus(data) -> km.Kamus:
    k = km.Kamus.dari_data(data)
    k.tetapkan(
        "usia",
        nama_lengkap="Usia nasabah",
        definisi="Usia saat pengajuan kredit",
        satuan="tahun",
        skala=km.RASIO,
        peran="prediktor",
        kode_hilang=[99],
    )
    k.tetapkan("segmen_usaha", peran="kelompok")
    return k


def test_kamus_pulih_utuh_dari_berkas_proyek(data, isi_kamus):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", kamus=isi_kamus))
    usia = hasil.kamus["usia"]
    assert usia.nama_lengkap == "Usia nasabah"
    assert usia.definisi == "Usia saat pengajuan kredit"
    assert usia.satuan == "tahun"
    assert usia.skala == km.RASIO
    assert usia.peran == "prediktor"
    assert usia.kode_hilang == [99]
    assert usia.dikonfirmasi
    assert hasil.kamus["segmen_usaha"].peran == "kelompok"


def test_proyek_tanpa_kamus_tetap_sah(data):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv"))
    assert len(hasil.kamus) == 0


def test_kamus_rusak_tidak_menggagalkan_pembukaan(data, isi_kamus):
    """Satu keterangan variabel yang cacat tidak boleh menghanguskan seluruh proyek."""
    berkas = pr.simpan_proyek(data, "d.csv", kamus=isi_kamus)
    sumber = zipfile.ZipFile(io.BytesIO(berkas))
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        for nama in sumber.namelist():
            arsip.writestr(nama, b"{bukan json" if nama == "kamus.json" else sumber.read(nama))

    hasil = pr.buka_proyek(penampung.getvalue())
    assert not hasil.data.empty
    # Kamus hilang, namun data dan sisanya tetap pulih.
    assert len(hasil.kamus) == 0


def test_kamus_diselaraskan_dengan_data_saat_dibuka(data, isi_kamus):
    """Kamus yang memuat kolom yang tak ada di data akan menyesatkan penyaringan."""
    isi_kamus.variabel["kolom_hantu"] = km.Variabel(nama="kolom_hantu")
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", kamus=isi_kamus))
    assert "kolom_hantu" not in hasil.kamus
    assert hasil.kamus.kolom == list(data.columns)


def test_ringkas_menyebut_kamus(data, isi_kamus):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", kamus=isi_kamus))
    ringkas = hasil.ringkas().set_index("Keterangan")["Isi"]
    assert ringkas["Kamus variabel"].startswith(f"{len(data.columns)} kolom")
    assert "perlu diperiksa" in ringkas["Kamus variabel"]


# --------------------------------------------------------------------------- #
# Rancangan penelitian di dalam proyek
# --------------------------------------------------------------------------- #


@pytest.fixture
def isi_penelitian() -> pp.ProyekPenelitian:
    return pp.ProyekPenelitian(
        judul="Kualitas layanan dan kepuasan nasabah",
        bidang="Manajemen",
        pertanyaan=["Apakah kualitas layanan berhubungan dengan kepuasan?"],
        hipotesis=["H1: hubungan positif"],
        populasi="Nasabah cabang Jakarta",
        ukuran_populasi=2500,
        unit_analisis="Nasabah perorangan",
        desain="eksperimen",
        penugasan_acak=True,
        teknik_sampling="berlapis",
        target_sampel=200,
        praregistrasi=pp.Praregistrasi(
            hipotesis=["H1"], uji_direncanakan=["Regresi linear berganda"]
        ),
    )


def test_rancangan_penelitian_pulih_utuh(data, isi_penelitian):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", penelitian=isi_penelitian))
    pulang = hasil.penelitian
    assert pulang.judul == isi_penelitian.judul
    assert pulang.desain == "eksperimen"
    assert pulang.penugasan_acak
    assert pulang.teknik_sampling == "berlapis"
    assert pulang.ukuran_populasi == 2500
    assert pulang.boleh_sebab
    assert pulang.boleh_generalisasi


def test_praregistrasi_pulih_dengan_sidik_yang_sama(data, isi_penelitian):
    """Sidik yang berubah setelah pulang-pergi akan salah menuduh pengguna mengubah rencana."""
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", penelitian=isi_penelitian))
    assert hasil.penelitian.praregistrasi is not None
    assert hasil.penelitian.praregistrasi.sidik == isi_penelitian.praregistrasi.sidik


def test_proyek_tanpa_rancangan_tetap_sah(data):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv"))
    assert hasil.penelitian.kosong()
    # Bawaannya tetap yang paling berhati-hati.
    assert not hasil.penelitian.boleh_sebab
    assert not hasil.penelitian.boleh_generalisasi


def test_rancangan_rusak_tidak_menggagalkan_pembukaan(data, isi_penelitian):
    berkas = pr.simpan_proyek(data, "d.csv", penelitian=isi_penelitian)
    sumber = zipfile.ZipFile(io.BytesIO(berkas))
    penampung = io.BytesIO()
    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        for nama in sumber.namelist():
            arsip.writestr(
                nama, b"{bukan json" if nama == "penelitian.json" else sumber.read(nama)
            )

    hasil = pr.buka_proyek(penampung.getvalue())
    assert not hasil.data.empty
    assert hasil.penelitian.kosong()


def test_ringkas_menyebut_rancangan(data, isi_penelitian):
    hasil = pr.buka_proyek(pr.simpan_proyek(data, "d.csv", penelitian=isi_penelitian))
    ringkas = hasil.ringkas().set_index("Keterangan")["Isi"]
    assert ringkas["Rancangan penelitian"] == isi_penelitian.judul
