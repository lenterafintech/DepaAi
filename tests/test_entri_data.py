"""Uji pembuatan dan pembersihan data hasil entri manual."""

from __future__ import annotations

import pandas as pd
import pytest

from mv_statlab import data_entry as de


def test_bakukan_nama():
    assert de.bakukan_nama("Skor Kredit") == "skor_kredit"
    assert de.bakukan_nama("  Rasio Utang / Pendapatan ") == "rasio_utang_pendapatan"
    assert de.bakukan_nama("KUAL-1") == "kual_1"


def test_validasi_menolak_nama_ganda_dan_kosong():
    ganda = [de.KolomBaru("usia"), de.KolomBaru("Usia")]
    assert any("terduplikasi" in m for m in de.validasi_kolom(ganda))
    assert any("tanpa nama" in m for m in de.validasi_kolom([de.KolomBaru("")]))
    assert de.validasi_kolom([]) == ["Belum ada kolom yang didefinisikan."]


def test_kategori_wajib_punya_pilihan():
    kurang = [de.KolomBaru("segmen", "Kategori", ["Mikro"])]
    assert any("minimal 2 pilihan" in m for m in de.validasi_kolom(kurang))
    cukup = [de.KolomBaru("segmen", "Kategori", ["Mikro", "Kecil"])]
    assert de.validasi_kolom(cukup) == []


def test_buat_kerangka_menghasilkan_tipe_yang_benar():
    kolom = [
        de.KolomBaru("Nama Responden", "Teks"),
        de.KolomBaru("Usia", "Angka bulat"),
        de.KolomBaru("Kepuasan", "Skala Likert 1–5"),
    ]
    df = de.buat_kerangka(kolom, 5)
    assert list(df.columns) == ["nama_responden", "usia", "kepuasan"]
    assert len(df) == 5
    assert df["usia"].dtype.name == "float64"  # sel kosong tampil kosong, bukan "None"
    assert df.isna().all().all()


def test_buat_kerangka_menolak_definisi_bermasalah():
    with pytest.raises(ValueError):
        de.buat_kerangka([de.KolomBaru("")], 3)


def test_kolom_kuesioner_menomori_butir():
    kolom = de.kolom_kuesioner({"Kualitas layanan": 3, "Harga": 2})
    assert [k.nama for k in kolom] == ["KUALIT1", "KUALIT2", "KUALIT3", "HARGA1", "HARGA2"]
    assert all(k.tipe == "Skala Likert 1–5" for k in kolom)


def test_rapikan_membuang_baris_kosong_dan_mengubah_angka():
    df = pd.DataFrame({"a": ["1", "2", None], "b": ["x", "y", None], "c": [None, None, None]})
    hasil = de.rapikan(df)
    assert len(hasil) == 2  # baris yang seluruhnya kosong dibuang
    assert pd.api.types.is_numeric_dtype(hasil["a"])
    assert not pd.api.types.is_numeric_dtype(hasil["b"])


def test_rapikan_menyimpan_angka_bulat_sebagai_bilangan_bulat():
    df = pd.DataFrame({"likert": [1.0, 4.0, 5.0], "rasio": [0.5, 1.25, 2.0]})
    hasil = de.rapikan(df)
    assert hasil["likert"].dtype.name == "Int64"
    assert hasil["rasio"].dtype.name == "float64"


def test_rapikan_tidak_memaksa_kolom_campuran_jadi_angka():
    df = pd.DataFrame({"a": ["1", "dua"]})
    assert not pd.api.types.is_numeric_dtype(de.rapikan(df)["a"])


def test_periksa_rentang_menandai_isian_di_luar_skala():
    kolom = [de.KolomBaru("kepuasan", "Skala Likert 1–5")]
    df = pd.DataFrame({"kepuasan": [1, 5, 9]})
    peringatan = de.periksa_rentang(df, kolom)
    assert peringatan and "1 isian di luar rentang" in peringatan[0]
    assert de.periksa_rentang(pd.DataFrame({"kepuasan": [1, 3, 5]}), kolom) == []


def test_ringkas_kelengkapan():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    ringkas = de.ringkas_kelengkapan(df)
    assert list(ringkas["Kolom"]) == ["a", "b"]
    assert list(ringkas["Terisi"]) == [2, 3]
    assert ringkas.loc[0, "% Terisi"] == pytest.approx(66.7)
    assert de.ringkas_kelengkapan(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# Variabel gabungan dari beberapa butir
# --------------------------------------------------------------------------- #


@pytest.fixture
def butir() -> pd.DataFrame:
    """Empat responden lengkap, satu responden yang hanya mengisi sebagian."""
    return pd.DataFrame(
        {
            "KUAL1": [1.0, 2.0, 3.0, 4.0, None],
            "KUAL2": [2.0, 3.0, 4.0, 5.0, 5.0],
            "KUAL3": [3.0, 3.0, 3.0, 4.0, None],
            "catatan": ["a", "b", "c", "d", "e"],
        }
    )


def test_rata_rata_menjaga_satuan_asli(butir):
    hasil = de.variabel_gabungan(butir, ["KUAL1", "KUAL2", "KUAL3"], "Kualitas")
    assert hasil.iloc[0] == pytest.approx(2.0)
    assert hasil.iloc[3] == pytest.approx(13 / 3)
    assert hasil.name == "kualitas"  # nama dibakukan


def test_jumlah_dan_skor_baku(butir):
    kolom = ["KUAL1", "KUAL2", "KUAL3"]
    jumlah = de.variabel_gabungan(butir, kolom, "K", "jumlah")
    assert jumlah.iloc[0] == pytest.approx(6.0)
    baku = de.variabel_gabungan(butir, kolom, "K", "z")
    # Skor baku berpusat di nol, sehingga responden terendah bernilai negatif.
    assert baku.iloc[0] < 0 < baku.iloc[3]


def test_responden_yang_kurang_lengkap_dibiarkan_kosong(butir):
    kolom = ["KUAL1", "KUAL2", "KUAL3"]
    # Responden kelima hanya mengisi 1 dari 3 butir.
    ketat = de.variabel_gabungan(butir, kolom, "K", minimal_terisi=3)
    assert pd.isna(ketat.iloc[4])
    longgar = de.variabel_gabungan(butir, kolom, "K", minimal_terisi=1)
    assert longgar.iloc[4] == pytest.approx(5.0)


def test_nama_variabel_dibakukan(butir):
    hasil = de.variabel_gabungan(butir, ["KUAL1", "KUAL2"], "Kualitas Layanan")
    assert hasil.name == "kualitas_layanan"


@pytest.mark.parametrize(
    "kolom, nama, cara, pesan",
    [
        (["KUAL1", "catatan"], "X", "rata", "bukan angka"),
        (["KUAL1"], "X", "rata", "minimal 2 butir"),
        (["KUAL1", "KUAL2"], "X", "median", "tidak dikenal"),
        (["KUAL1", "ZZ"], "X", "rata", "tidak ada dalam data"),
        (["KUAL1", "KUAL2"], "   ", "rata", "belum diisi"),
    ],
)
def test_masukan_cacat_ditolak_dengan_pesan_jelas(butir, kolom, nama, cara, pesan):
    with pytest.raises(ValueError, match=pesan):
        de.variabel_gabungan(butir, kolom, nama, cara)


def test_ringkasan_gabungan_melaporkan_kelengkapan(butir):
    kolom = ["KUAL1", "KUAL2", "KUAL3"]
    hasil = de.variabel_gabungan(butir, kolom, "Kualitas")
    ringkas = de.ringkas_gabungan(butir, kolom, hasil).set_index("Keterangan")["Nilai"]
    assert ringkas["Nama variabel"] == "kualitas"
    assert ringkas["Butir penyusun"] == "3 butir"
    assert ringkas["Terisi"] == "4 dari 5"


def test_katalog_cara_gabung_lengkap():
    for isi in de.CARA_GABUNG.values():
        assert {"nama", "catatan"} <= set(isi)
    assert de.CARA_GABUNG_BAWAAN in de.CARA_GABUNG
