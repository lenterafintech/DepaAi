"""Uji kamus variabel.

Yang diuji adalah **keputusannya** — skala apa yang ditebak untuk bentuk data apa —
bukan sekadar tidak adanya galat. Kamus yang berjalan mulus namun menyebut skor Likert
sebagai rasio akan menyesatkan seluruh saran uji yang berdiri di atasnya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nalardata import kamus as km


@pytest.fixture(scope="module")
def acak() -> np.random.Generator:
    return np.random.default_rng(20260906)


# --------------------------------------------------------------------------- #
# Tebakan skala
# --------------------------------------------------------------------------- #


def test_likert_satu_sampai_lima_ditebak_ordinal():
    s = pd.Series([1, 2, 3, 4, 5, 3, 4, 2, 5, 1] * 10)
    skala, keyakinan, alasan = km.tebak_skala(s)
    assert skala == km.ORDINAL
    assert keyakinan == km.MUNGKIN
    assert "Likert" in alasan


def test_cacahan_mulai_dari_nol_ditebak_rasio_bukan_likert():
    """Pembeda satu angka: Likert mulai dari 1, cacahan lazim mulai dari 0."""
    s = pd.Series([0, 1, 2, 3, 0, 1, 4, 2, 0, 1] * 10)
    skala, _, alasan = km.tebak_skala(s)
    assert skala == km.RASIO
    assert "cacahan" in alasan


def test_teks_ditebak_nominal():
    s = pd.Series(["Perdagangan", "Jasa", "Manufaktur"] * 20)
    skala, keyakinan, _ = km.tebak_skala(s)
    assert skala == km.NOMINAL
    assert keyakinan == km.PASTI


def test_teks_bertingkat_khas_kuesioner_ditebak_ordinal():
    s = pd.Series(["Sangat tidak setuju", "Tidak setuju", "Netral", "Setuju"] * 10)
    skala, keyakinan, _ = km.tebak_skala(s)
    assert skala == km.ORDINAL
    assert keyakinan == km.MUNGKIN


def test_dua_nilai_ditebak_nominal(acak):
    s = pd.Series(acak.integers(0, 2, size=200))
    assert km.tebak_skala(s)[0] == km.NOMINAL


def test_pecahan_tanpa_negatif_ditebak_rasio_dengan_alasan_jujur(acak):
    """Interval dan rasio memang tidak dapat dibedakan dari data; alasannya harus mengaku."""
    s = pd.Series(acak.normal(500, 80, size=200).round(2))
    skala, keyakinan, alasan = km.tebak_skala(s)
    assert skala == km.RASIO
    assert keyakinan == km.TEBAKAN
    assert "tidak dapat" in alasan.lower()


def test_ada_nilai_negatif_ditebak_interval(acak):
    s = pd.Series(acak.normal(0, 1, size=200))
    assert km.tebak_skala(s)[0] == km.INTERVAL


def test_kolom_boolean_ditebak_nominal_dengan_pasti():
    skala, keyakinan, _ = km.tebak_skala(pd.Series([True, False, True, True]))
    assert (skala, keyakinan) == (km.NOMINAL, km.PASTI)


def test_kolom_tanggal_ditebak_interval():
    s = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    assert km.tebak_skala(s)[0] == km.INTERVAL


def test_kolom_kosong_tidak_menggagalkan_tebakan():
    skala, keyakinan, alasan = km.tebak_skala(pd.Series([np.nan] * 10))
    assert keyakinan == km.TEBAKAN
    assert "kosong" in alasan.lower()


def test_setiap_tebakan_menyertakan_alasan(acak):
    """Tebakan tanpa alasan tidak dapat dinilai pengguna, dan pengguna yang menilai."""
    contoh = [
        pd.Series([1, 2, 3, 4, 5] * 20),
        pd.Series(["a", "b", "c"] * 20),
        pd.Series(acak.normal(0, 1, 100)),
        pd.Series([True, False] * 50),
    ]
    for s in contoh:
        _, _, alasan = km.tebak_skala(s)
        assert alasan.strip()


# --------------------------------------------------------------------------- #
# Tebakan peran
# --------------------------------------------------------------------------- #


def test_kolom_id_unik_dikenali_sebagai_penanda_unit():
    s = pd.Series([f"N{i:03d}" for i in range(100)])
    assert km.tebak_peran("id_nasabah", s, km.NOMINAL) == "id"


def test_kolom_bernama_id_tetapi_berulang_bukan_penanda_unit():
    """Penanda unit harus benar-benar unik; namanya saja tidak cukup."""
    s = pd.Series(["A", "B", "A", "B"] * 10)
    assert km.tebak_peran("id_cabang", s, km.NOMINAL) == km.BELUM


def test_lama_dalam_tahun_bukan_penanda_waktu():
    """'lama_usaha_tahun' memuat kata tahun tetapi berisi durasi, bukan periode."""
    s = pd.Series([1, 3, 7, 12, 2, 5] * 10)
    assert km.tebak_peran("lama_usaha_tahun", s, km.RASIO) == km.BELUM


def test_tahun_kalender_dikenali_sebagai_penanda_waktu():
    s = pd.Series([2019, 2020, 2021, 2022] * 10)
    assert km.tebak_peran("tahun", s, km.INTERVAL) == "waktu"


def test_kolom_tanggal_dikenali_sebagai_penanda_waktu():
    s = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
    assert km.tebak_peran("dibuat", s, km.INTERVAL) == "waktu"


def test_peran_penelitian_tidak_pernah_ditebak(acak):
    """Outcome dan prediktor adalah keputusan penelitian, bukan sifat data."""
    s = pd.Series(acak.normal(600, 50, 100))
    assert km.tebak_peran("skor_kredit", s, km.RASIO) == km.BELUM
    assert km.tebak_peran("kepuasan", s, km.RASIO) == km.BELUM


# --------------------------------------------------------------------------- #
# Kode nilai hilang
# --------------------------------------------------------------------------- #


def test_kode_hilang_terpencil_diusulkan(acak):
    nilai = list(acak.integers(20, 60, size=100)) + [99, 99, 99]
    assert 99.0 in km.tebak_kode_hilang(pd.Series(nilai))


def test_angka_wajar_tidak_diusulkan_sebagai_kode_hilang(acak):
    """99 pada skor ujian 0-100 adalah nilai sungguhan, bukan kode."""
    nilai = list(acak.integers(0, 101, size=200))
    assert km.tebak_kode_hilang(pd.Series(nilai)) == []


def test_kolom_teks_tidak_menghasilkan_usulan_kode():
    assert km.tebak_kode_hilang(pd.Series(["a", "b", "99"])) == []


# --------------------------------------------------------------------------- #
# Kamus
# --------------------------------------------------------------------------- #


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"R{i:02d}" for i in range(30)],
            "usia": list(range(20, 50)),
            "kepuasan": [1, 2, 3, 4, 5] * 6,
            "wilayah": ["Jakarta", "Bandung", "Medan"] * 10,
        }
    )


def test_kamus_memuat_seluruh_kolom(data):
    k = km.Kamus.dari_data(data)
    assert k.kolom == list(data.columns)
    assert len(k) == 4


def test_penyaringan_memakai_kamus_bukan_dtype(data):
    """kepuasan bertipe angka, namun sebagai ordinal ia bukan variabel numerik."""
    k = km.Kamus.dari_data(data)
    assert "kepuasan" in k.kategorik()
    assert "kepuasan" not in k.numerik()
    assert "usia" in k.numerik()


def test_penanda_unit_dikeluarkan_dari_daftar_analisis(data):
    k = km.Kamus.dari_data(data)
    assert "id" not in k.numerik()
    assert "id" not in k.kategorik()


def test_tetapkan_menandai_sudah_dikonfirmasi(data):
    k = km.Kamus.dari_data(data)
    assert k["kepuasan"].perlu_diperiksa
    k.tetapkan("kepuasan", skala=km.ORDINAL, nama_lengkap="Kepuasan nasabah")
    assert k["kepuasan"].dikonfirmasi
    assert not k["kepuasan"].perlu_diperiksa
    assert k.judul("kepuasan") == "Kepuasan nasabah"


def test_tetapkan_menolak_ruas_asing(data):
    k = km.Kamus.dari_data(data)
    with pytest.raises(ValueError, match="tidak dikenal"):
        k.tetapkan("usia", skalaa=km.RASIO)


def test_skala_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        km.Variabel(nama="x", skala="kuantitatif")


def test_peran_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        km.Variabel(nama="x", peran="variabel bebas")


def test_kolom_asing_ditolak_dengan_pesan_yang_menyebut_namanya(data):
    with pytest.raises(KeyError, match="tidak ada dalam kamus"):
        km.Kamus.dari_data(data)["tidak_ada"]


def test_daftar_yang_perlu_diperiksa_menyusut_setelah_dikonfirmasi(data):
    k = km.Kamus.dari_data(data)
    semula = len(k.perlu_diperiksa())
    assert semula > 0
    k.tetapkan(k.perlu_diperiksa()[0])
    assert len(k.perlu_diperiksa()) == semula - 1


# --------------------------------------------------------------------------- #
# Penerapan ke data
# --------------------------------------------------------------------------- #


def test_kode_hilang_diterapkan_menjadi_kosong():
    df = pd.DataFrame({"usia": [30, 45, 99, 52]})
    k = km.Kamus.dari_data(df)
    k.tetapkan("usia", kode_hilang=[99])
    hasil = k.terapkan(df)
    assert hasil["usia"].isna().sum() == 1
    # Data asli tidak boleh ikut berubah.
    assert df["usia"].isna().sum() == 0


def test_label_nilai_diterapkan():
    df = pd.DataFrame({"jk": [1, 2, 1, 2]})
    k = km.Kamus.dari_data(df)
    k.tetapkan("jk", label_nilai={1: "Laki-laki", 2: "Perempuan"})
    assert k.terapkan(df)["jk"].tolist() == ["Laki-laki", "Perempuan", "Laki-laki", "Perempuan"]


def test_tanpa_kode_dan_label_data_tidak_berubah(data):
    k = km.Kamus.dari_data(data)
    pd.testing.assert_frame_equal(k.terapkan(data), data)


# --------------------------------------------------------------------------- #
# Penyelarasan dan penyimpanan
# --------------------------------------------------------------------------- #


def test_selaraskan_mempertahankan_keterangan_yang_sudah_ditulis(data):
    """Menambah kolom tidak boleh menghapus definisi operasional yang sudah diisi."""
    k = km.Kamus.dari_data(data)
    k.tetapkan("usia", definisi="Usia responden saat survei", satuan="tahun")

    lebih = data.assign(pendapatan=[5_000_000] * len(data)).drop(columns=["wilayah"])
    baru = k.selaraskan(lebih)

    assert baru["usia"].definisi == "Usia responden saat survei"
    assert baru["usia"].satuan == "tahun"
    assert "pendapatan" in baru
    assert "wilayah" not in baru


def test_perjalanan_pulang_pergi_lewat_dict(data):
    k = km.Kamus.dari_data(data)
    k.tetapkan("kepuasan", skala=km.ORDINAL, nama_lengkap="Kepuasan", kode_hilang=[9])
    pulang = km.Kamus.dari_dict(k.ke_dict())
    assert pulang["kepuasan"].skala == km.ORDINAL
    assert pulang["kepuasan"].nama_lengkap == "Kepuasan"
    assert pulang["kepuasan"].kode_hilang == [9]
    assert pulang["kepuasan"].dikonfirmasi


def test_isi_cacat_dilewati_tanpa_menggagalkan_pembukaan():
    """Berkas proyek yang rusak sebagian tetap terbuka; yang cacat dilewati."""
    isi = {
        "baik": {"nama": "baik", "skala": km.RASIO},
        "rusak": {"nama": "rusak", "skala": "entah"},
        "bukan_dict": "x",
    }
    k = km.Kamus.dari_dict(isi)
    assert k.kolom == ["baik"]


def test_dari_dict_kosong_menghasilkan_kamus_kosong():
    assert len(km.Kamus.dari_dict(None)) == 0
    assert len(km.Kamus.dari_dict({})) == 0


def test_label_kolom_spss_dipakai_sebagai_nama_lengkap(data):
    k = km.Kamus.dari_data(data, label_spss={"usia": "Usia responden"})
    assert k["usia"].nama_lengkap == "Usia responden"
    assert k.judul("usia") == "Usia responden"


def test_judul_kolom_asing_dikembalikan_apa_adanya(data):
    assert km.Kamus.dari_data(data).judul("tidak_ada") == "tidak_ada"


def test_ringkas_memuat_dasar_dugaan_setiap_kolom(data):
    ringkas = km.Kamus.dari_data(data).ringkas()
    assert len(ringkas) == 4
    assert (ringkas["Dasar dugaan"].str.strip() != "").all()
