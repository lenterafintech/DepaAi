"""Uji audit kualitas data dan HTMT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalardata import audit
from nalardata import reliability as rb

ROOT = Path(__file__).resolve().parents[1]


def _perkara(hasil: audit.HasilAudit) -> set[str]:
    return {t.perkara for t in hasil.temuan}


@pytest.fixture(scope="module")
def bersih() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


# --------------------------------------------------------------------------- #
# Audit tidak mengubah apa pun
# --------------------------------------------------------------------------- #


def test_audit_tidak_mengubah_data(bersih):
    sebelum = bersih.copy()
    audit.jalankan_audit(bersih)
    pd.testing.assert_frame_equal(bersih, sebelum)


def test_data_kosong_ditolak():
    with pytest.raises(ValueError, match="Tidak ada data"):
        audit.jalankan_audit(pd.DataFrame())


def test_data_contoh_tidak_punya_masalah_kritis(bersih):
    hasil = audit.jalankan_audit(bersih)
    assert hasil.kritis == []
    assert hasil.status() != audit.KRITIS
    assert hasil.ringkas()["Baris"] == 400


# --------------------------------------------------------------------------- #
# Tiap pemeriksaan menyala pada datanya sendiri
# --------------------------------------------------------------------------- #


def test_kolom_kosong_dan_konstan_dinilai_kritis():
    df = pd.DataFrame(
        {
            "kosong": [None] * 40,
            "tetap": [7] * 40,
            "wajar": list(range(40)),
        }
    )
    hasil = audit.jalankan_audit(df)
    assert {"Kolom kosong", "Kolom tidak beragam"} <= _perkara(hasil)
    assert hasil.status() == audit.KRITIS
    assert "belum siap dianalisis" in hasil.kesimpulan()


def test_nilai_hilang_bertingkat_menurut_proporsinya():
    n = 200
    df = pd.DataFrame(
        {
            "banyak_hilang": [None] * 60 + list(range(n - 60)),  # 30%
            "sedikit_hilang": [None] * 20 + list(range(n - 20)),  # 10%
            "hampir_penuh": [None] * 2 + list(range(n - 2)),  # 1%
        }
    )
    tingkat = {
        t.kolom: t.tingkat for t in audit.jalankan_audit(df).temuan if t.perkara == "Nilai hilang"
    }
    assert tingkat["banyak_hilang"] == audit.KRITIS
    assert tingkat["sedikit_hilang"] == audit.PERINGATAN
    assert tingkat["hampir_penuh"] == audit.CATATAN


def test_baris_kembar_terdeteksi():
    dasar = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = pd.concat([dasar, dasar], ignore_index=True)
    hasil = audit.jalankan_audit(df)
    baris = [t for t in hasil.temuan if t.perkara == "Baris kembar"][0]
    assert "3 baris identik" in baris.rincian


def test_angka_yang_tersimpan_sebagai_teks_terdeteksi():
    # Nilai unik dijaga di bawah ambang kategori agar yang diuji hanya perkara tipe.
    df = pd.DataFrame(
        {
            "murni_teks_angka": [str(i % 20) for i in range(60)],
            "campuran": [str(i % 16) for i in range(56)] + ["n/a", "-", "tidak ada", "?"],
        }
    )
    perkara: dict[str, set[str]] = {}
    for t in audit.jalankan_audit(df).temuan:
        perkara.setdefault(t.kolom, set()).add(t.perkara)
    assert "Angka tersimpan sebagai teks" in perkara["murni_teks_angka"]
    assert "Angka bercampur teks" in perkara["campuran"]


def test_nilai_tak_hingga_dinilai_kritis():
    df = pd.DataFrame({"rasio": [1.0, 2.0, np.inf, -np.inf] + [3.0] * 36})
    hasil = audit.jalankan_audit(df)
    temuan = [t for t in hasil.temuan if t.perkara == "Nilai tak hingga"][0]
    assert temuan.tingkat == audit.KRITIS
    assert "2 nilai" in temuan.rincian


def test_pencilan_dilaporkan_saat_melewati_ambang():
    acak = np.random.default_rng(5)
    wajar = pd.DataFrame({"x": acak.normal(size=200)})
    assert "Pencilan" not in _perkara(audit.jalankan_audit(wajar))

    banyak = pd.DataFrame({"x": list(acak.normal(size=180)) + [50.0] * 20})
    assert "Pencilan" in _perkara(audit.jalankan_audit(banyak))


def test_kategori_yang_tidak_seragam_terdeteksi():
    df = pd.DataFrame({"kota": ["Jakarta", "jakarta", " Jakarta ", "Bandung"] * 15})
    temuan = [
        t for t in audit.jalankan_audit(df).temuan if t.perkara == "Kategori tidak seragam"
    ]
    assert temuan and "menyusut" in temuan[0].rincian


def test_pengenal_dikenali_sebagai_kategori_terlalu_banyak():
    df = pd.DataFrame({"id": [f"ID{i:04d}" for i in range(120)]})
    assert "Kategori sangat banyak" in _perkara(audit.jalankan_audit(df))


def test_sampel_kecil_dan_baris_lengkap_sedikit():
    kecil = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert "Sampel kecil" in _perkara(audit.jalankan_audit(kecil))

    bolong = pd.DataFrame({"a": [1] * 100, "b": [None] * 70 + [2] * 30})
    hasil = audit.jalankan_audit(bolong)
    temuan = [t for t in hasil.temuan if t.perkara == "Baris lengkap sedikit"]
    assert temuan and temuan[0].tingkat == audit.KRITIS


def test_kemencengan_dilaporkan_sebagai_catatan():
    acak = np.random.default_rng(2)
    # Lognormal dipakai karena kemencengannya jauh melewati ambang 2,0; sebaran
    # eksponensial pada n=400 hanya mencapai 1,76 sehingga justru tidak dilaporkan.
    df = pd.DataFrame({"pendapatan": acak.lognormal(0, 1, 400)})
    temuan = [t for t in audit.jalankan_audit(df).temuan if t.perkara == "Sebaran menceng"]
    assert temuan and temuan[0].tingkat == audit.CATATAN
    assert "ke kanan" in temuan[0].rincian


def test_sebaran_wajar_tidak_dilaporkan_menceng():
    acak = np.random.default_rng(4)
    df = pd.DataFrame({"x": acak.normal(size=400)})
    assert "Sebaran menceng" not in _perkara(audit.jalankan_audit(df))


# --------------------------------------------------------------------------- #
# Pelaporan
# --------------------------------------------------------------------------- #


def test_tabel_diurutkan_dari_yang_paling_mendesak():
    df = pd.DataFrame(
        {
            "tetap": [1] * 50,  # kritis
            "kotor": ["A", "a"] * 25,  # peringatan
        }
    )
    tabel = audit.jalankan_audit(df).tabel()
    assert list(tabel["Tingkat"])[0] == "Kritis"
    assert set(tabel.columns) == {
        "Tingkat", "Perkara", "Kolom", "Rincian", "Dampak", "Saran"
    }


def test_tabel_kosong_tetap_punya_kolom():
    df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    hasil = audit.jalankan_audit(df)
    tabel = hasil.tabel()
    if hasil.temuan:
        assert not tabel.empty
    else:
        assert list(tabel.columns)[0] == "Tingkat"
        assert "siap dianalisis" in hasil.kesimpulan()


def test_setiap_temuan_menjelaskan_dampak_dan_saran(bersih):
    for temuan in audit.jalankan_audit(bersih).temuan:
        assert temuan.dampak.strip(), f"{temuan.perkara} tanpa dampak"
        assert temuan.saran.strip(), f"{temuan.perkara} tanpa saran"
        assert temuan.tingkat in audit.URUTAN


# --------------------------------------------------------------------------- #
# HTMT
# --------------------------------------------------------------------------- #


def _dua_konstruk(r_faktor: float, n: int = 600, muatan: float = 0.8) -> pd.DataFrame:
    acak = np.random.default_rng(7)
    f1 = acak.normal(size=n)
    f2 = r_faktor * f1 + np.sqrt(1 - r_faktor**2) * acak.normal(size=n)
    sisa = np.sqrt(1 - muatan**2)
    data = {f"a{i}": muatan * f1 + sisa * acak.normal(size=n) for i in range(1, 5)}
    data.update({f"b{i}": muatan * f2 + sisa * acak.normal(size=n) for i in range(1, 5)})
    return pd.DataFrame(data)


KONSTRUK = {"A": [f"a{i}" for i in range(1, 5)], "B": [f"b{i}" for i in range(1, 5)]}


def test_htmt_memulihkan_korelasi_konstruk_yang_ditanam():
    """Pada model kongenerik bermuatan seragam, HTMT menaksir korelasi faktornya."""
    for r in (0.30, 0.50, 0.70):
        nilai = rb.htmt(_dua_konstruk(r, n=4000), KONSTRUK).loc[0, "HTMT"]
        assert abs(nilai - r) < 0.05, f"r={r} menghasilkan HTMT={nilai}"


def test_htmt_meloloskan_konstruk_terpisah_dan_menolak_yang_menyatu():
    terpisah = rb.htmt(_dua_konstruk(0.35), KONSTRUK).loc[0]
    assert terpisah["HTMT"] < 0.85
    assert terpisah["Keputusan (0,85)"] == "Terpenuhi"

    menyatu = rb.htmt(_dua_konstruk(0.95), KONSTRUK).loc[0]
    assert menyatu["HTMT"] > 0.85
    assert menyatu["Keputusan (0,85)"] == "Tidak terpenuhi"


def test_htmt_melaporkan_kedua_ambang():
    tabel = rb.htmt(_dua_konstruk(0.87), KONSTRUK)
    baris = tabel.loc[0]
    assert baris["Keputusan (0,85)"] == "Tidak terpenuhi"
    # Ambang longgar 0,90 masih meloloskannya.
    assert baris["Keputusan (0,90)"] == "Terpenuhi"


def test_htmt_diurutkan_dari_pasangan_paling_bermasalah():
    df = _dua_konstruk(0.4)
    acak = np.random.default_rng(1)
    for i in range(1, 5):
        df[f"c{i}"] = 0.8 * df["a1"] + 0.2 * acak.normal(size=len(df))
    konstruk = dict(KONSTRUK, C=[f"c{i}" for i in range(1, 5)])
    tabel = rb.htmt(df, konstruk)
    assert len(tabel) == 3
    assert list(tabel["HTMT"]) == sorted(tabel["HTMT"], reverse=True)


@pytest.mark.parametrize(
    "konstruk, pesan",
    [
        ({"A": ["a1", "a2"]}, "minimal 2 konstruk"),
        ({"A": ["a1", "a2"], "B": ["zz1", "zz2"]}, "tidak ada dalam data"),
    ],
)
def test_htmt_menolak_masukan_cacat(konstruk, pesan):
    with pytest.raises(ValueError, match=pesan):
        rb.htmt(_dua_konstruk(0.5), konstruk)


# --------------------------------------------------------------------------- #
# Pemeriksaan lanjutan
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260906)


def _kamus_likert(df, butir):
    from nalardata import kamus as km

    k = km.Kamus.dari_data(df)
    for kolom in butir:
        k.tetapkan(kolom, skala=km.ORDINAL)
    return k


def _temuan(hasil, perkara: str):
    cocok = [t for t in hasil.temuan if t.perkara == perkara]
    return cocok[0] if cocok else None


def test_nilai_hilang_yang_berkaitan_dengan_jawaban_lain_ditandai(rng):
    """Nilai hilang acak boleh diabaikan; yang berpola menggeser kesimpulan."""
    n = 300
    skor = rng.normal(600, 60, n)
    df = pd.DataFrame(
        {
            "skor_kredit": skor,
            "usia": rng.normal(40, 9, n),
            "pendapatan": skor * 8000 + rng.normal(0, 3e5, n),
        }
    )
    df.loc[df["pendapatan"] > df["pendapatan"].quantile(0.75), "pendapatan"] = np.nan

    temuan = _temuan(audit.jalankan_audit(df), "Nilai hilang tidak acak")
    assert temuan is not None
    assert "skor_kredit" in temuan.rincian
    assert temuan.n_baris == 75


def test_nilai_hilang_yang_benar_benar_acak_tidak_ditandai(rng):
    n = 300
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    df.loc[rng.choice(n, 60, replace=False), "b"] = np.nan
    assert _temuan(audit.jalankan_audit(df), "Nilai hilang tidak acak") is None


def test_kolom_cacahan_yang_didominasi_nol_dilaporkan(rng):
    df = pd.DataFrame(
        {
            "komplain": np.r_[np.zeros(140, int), rng.integers(1, 6, 60)],
            "usia": rng.normal(40, 9, 200),
        }
    )
    temuan = _temuan(audit.jalankan_audit(df), "Nol berlebih")
    assert temuan is not None
    assert temuan.kolom == "komplain"
    assert "berinflasi nol" in temuan.saran


def test_kolom_pecahan_biasa_tidak_dianggap_nol_berlebih(rng):
    df = pd.DataFrame({"rasio": rng.normal(0.5, 0.2, 200)})
    assert _temuan(audit.jalankan_audit(df), "Nol berlebih") is None


def test_jawaban_seragam_pada_butir_ordinal_terdeteksi(rng):
    n = 200
    butir = [f"b{i}" for i in range(6)]
    df = pd.DataFrame({k: rng.integers(1, 6, n) for k in butir})
    df.loc[:19, butir] = 4  # dua puluh responden menjawab sama persis

    hasil = audit.jalankan_audit(df, _kamus_likert(df, butir))
    temuan = _temuan(hasil, "Jawaban seragam")
    assert temuan is not None
    assert temuan.n_baris >= 20
    assert "alpha Cronbach" in temuan.dampak
    assert {p.kode for p in temuan.penanganan} == {"hapus_baris", "tandai"}


def test_jawaban_seragam_dilewati_tanpa_kamus(rng):
    """Tanpa kamus, aplikasi tidak tahu butir mana yang ordinal — dan tidak menebak."""
    butir = [f"b{i}" for i in range(6)]
    df = pd.DataFrame({k: rng.integers(1, 6, 200) for k in butir})
    df.loc[:19, butir] = 4
    assert _temuan(audit.jalankan_audit(df), "Jawaban seragam") is None


def test_bias_metode_tunggal_terdeteksi_saat_satu_faktor_mendominasi(rng):
    n = 300
    dasar = rng.normal(0, 1, n)
    butir = [f"b{i}" for i in range(8)]
    df = pd.DataFrame(
        {k: np.clip((dasar * 1.6 + rng.normal(0, 0.4, n)).round() + 3, 1, 5) for k in butir}
    )
    temuan = _temuan(audit.jalankan_audit(df, _kamus_likert(df, butir)), "Indikasi bias metode tunggal")
    assert temuan is not None
    assert "Harman" in temuan.rincian
    assert "lemah dan hanya memberi isyarat" in temuan.saran


def test_butir_yang_benar_benar_berbeda_tidak_ditandai_bias_metode(rng):
    n = 300
    butir = [f"b{i}" for i in range(8)]
    df = pd.DataFrame({k: rng.integers(1, 6, n) for k in butir})
    hasil = audit.jalankan_audit(df, _kamus_likert(df, butir))
    assert _temuan(hasil, "Indikasi bias metode tunggal") is None


def test_pasangan_unit_dan_waktu_yang_berulang_dinilai_kritis():
    from nalardata import kamus as km

    df = pd.DataFrame(
        {
            "id_unit": ["A", "A", "B", "B", "A"],
            "tahun": [2021, 2022, 2021, 2022, 2022],
            "nilai": [1.0, 2, 3, 4, 5],
        }
    )
    k = km.Kamus.dari_data(df)
    k.tetapkan("id_unit", peran="id")
    k.tetapkan("tahun", peran="waktu")

    temuan = _temuan(audit.jalankan_audit(df, k), "Unit dan waktu berulang")
    assert temuan is not None
    assert temuan.tingkat == audit.KRITIS
    assert "galat baku terlalu kecil" in temuan.dampak


def test_panel_tidak_seimbang_dilaporkan_sebagai_catatan():
    from nalardata import kamus as km

    df = pd.DataFrame(
        {
            "id_unit": ["A", "A", "A", "B", "C"],
            "tahun": [2021, 2022, 2023, 2021, 2021],
            "nilai": [1.0, 2, 3, 4, 5],
        }
    )
    k = km.Kamus.dari_data(df)
    k.tetapkan("id_unit", peran="id")
    k.tetapkan("tahun", peran="waktu")

    temuan = _temuan(audit.jalankan_audit(df, k), "Panel tidak seimbang")
    assert temuan is not None
    assert temuan.tingkat == audit.CATATAN


def test_kolom_bernama_seperti_data_pribadi_ditandai(rng):
    df = pd.DataFrame(
        {
            "nama_responden": [f"R{i}" for i in range(50)],
            "no_hp": [f"08{i:09d}" for i in range(50)],
            "skor": rng.normal(0, 1, 50),
        }
    )
    temuan = _temuan(audit.jalankan_audit(df), "Kemungkinan data pribadi")
    assert temuan is not None
    assert "nama_responden" in temuan.kolom
    assert "hapus_kolom" in {p.kode for p in temuan.penanganan}


def test_kolom_analisis_biasa_tidak_disangka_data_pribadi(bersih):
    assert _temuan(audit.jalankan_audit(bersih), "Kemungkinan data pribadi") is None


# --------------------------------------------------------------------------- #
# Struktur temuan enam bagian
# --------------------------------------------------------------------------- #


def test_setiap_penanganan_menyebutkan_akibatnya(rng):
    """Akibat harus disebut sebelum tindakan dijalankan, bukan sesudahnya."""
    df = pd.DataFrame(
        {
            "nama": [f"R{i}" for i in range(60)],
            "email": [f"r{i}@contoh.id" for i in range(60)],
            "skor": rng.normal(0, 1, 60),
        }
    )
    for temuan in audit.jalankan_audit(df).temuan:
        for pilihan in temuan.penanganan:
            assert pilihan.label.strip()
            assert len(pilihan.akibat) > 15


def test_ringkas_baris_tidak_menumpahkan_seluruh_indeks():
    temuan = audit.Temuan(
        tingkat=audit.CATATAN, perkara="x", kolom="y", rincian="", dampak="", saran="",
        baris=list(range(80)),
    )
    ringkas = temuan.ringkas_baris()
    assert "80 baris" in ringkas
    assert len(ringkas) < 60


def test_ringkas_baris_menyebut_nomor_baris_yang_dilihat_pengguna():
    """Pengguna membaca nomor baris mulai dari satu, bukan dari nol."""
    temuan = audit.Temuan(
        tingkat=audit.CATATAN, perkara="x", kolom="y", rincian="", dampak="", saran="",
        baris=[0, 4],
    )
    assert temuan.ringkas_baris() == "baris 1, 5"


# --------------------------------------------------------------------------- #
# Penerapan penanganan
# --------------------------------------------------------------------------- #


def test_hapus_baris_mengembalikan_salinan_dan_catatannya(rng):
    n = 200
    butir = [f"b{i}" for i in range(6)]
    df = pd.DataFrame({k: rng.integers(1, 6, n) for k in butir})
    df.loc[:19, butir] = 4

    temuan = _temuan(audit.jalankan_audit(df, _kamus_likert(df, butir)), "Jawaban seragam")
    hasil, catatan = audit.terapkan(df, temuan, "hapus_baris")

    assert len(hasil) == len(df) - temuan.n_baris
    assert len(df) == n, "data asli tidak boleh ikut berubah"
    assert "dihapus" in catatan and str(len(hasil)) in catatan


def test_hapus_kolom_membuang_kolom_identitas(rng):
    df = pd.DataFrame(
        {
            "nama_responden": [f"R{i}" for i in range(50)],
            "no_hp": [f"08{i:09d}" for i in range(50)],
            "skor": rng.normal(0, 1, 50),
        }
    )
    temuan = _temuan(audit.jalankan_audit(df), "Kemungkinan data pribadi")
    hasil, catatan = audit.terapkan(df, temuan, "hapus_kolom")

    assert list(hasil.columns) == ["skor"]
    assert "nama_responden" in catatan
    assert df.shape[1] == 3, "data asli tidak boleh ikut berubah"


def test_tandai_tidak_mengubah_data_sama_sekali(rng):
    df = pd.DataFrame(
        {"nama": [f"R{i}" for i in range(50)], "skor": rng.normal(0, 1, 50)}
    )
    temuan = _temuan(audit.jalankan_audit(df), "Kemungkinan data pribadi")
    hasil, catatan = audit.terapkan(df, temuan, "tandai")
    pd.testing.assert_frame_equal(hasil, df)
    assert "tanpa mengubah data" in catatan


def test_penanganan_yang_tidak_ditawarkan_ditolak(rng):
    df = pd.DataFrame(
        {"nama": [f"R{i}" for i in range(50)], "skor": rng.normal(0, 1, 50)}
    )
    temuan = _temuan(audit.jalankan_audit(df), "Kemungkinan data pribadi")
    with pytest.raises(ValueError, match="tidak tersedia"):
        audit.terapkan(df, temuan, "isi_median")
