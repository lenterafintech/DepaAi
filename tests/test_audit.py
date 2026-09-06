"""Uji audit kualitas data dan HTMT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lentera_mva import audit
from lentera_mva import reliability as rb

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
