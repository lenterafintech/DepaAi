"""Uji Simulasi Sidang.

Yang diuji adalah apakah pertanyaannya benar-benar lahir dari analisis pengguna:
rancangan potong lintang harus memunculkan pertanyaan sebab-akibat, sampling
purposif harus memunculkan pertanyaan generalisasi, dan banyak uji harus
memunculkan pertanyaan kesalahan tipe I. Daftar pertanyaan umum yang sama untuk
semua orang sudah tersedia gratis di internet; yang tidak tersedia adalah
pertanyaan yang menyebut R² Anda sendiri.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nalardata import audit as ad
from nalardata import jejak as jk
from nalardata import kamus as km
from nalardata import narrative as nr
from nalardata import proyek_penelitian as pp
from nalardata import sidang as sd

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


@pytest.fixture(scope="module")
def konfig() -> nr.Konfigurasi:
    return nr.Konfigurasi(
        variabel=[
            "usia",
            "pendapatan_bulanan",
            "rasio_utang_pendapatan",
            "skor_kredit",
            "jumlah_keterlambatan",
        ],
        nama_data="contoh_data_nasabah.csv",
        target_numerik="skor_kredit",
        prediktor=["rasio_utang_pendapatan", "jumlah_keterlambatan"],
        kelompok="segmen_usaha",
    )


def _laporan(data, konfig, penelitian=None):
    return nr.analisis_dan_laporan(data, konfig, penelitian)[1]


def _jejak(nilai=(0.001, 0.02, 0.04, 0.3, 0.045)) -> jk.Jejak:
    j = jk.Jejak()
    for nomor, p in enumerate(nilai, start=1):
        j.catat_uji(f"Uji {nomor}", p=p)
    return j


def _teks(simulasi) -> str:
    return " ".join(p.pertanyaan + " " + p.jawaban for p in simulasi.pertanyaan)


# --------------------------------------------------------------------------- #
# Pertanyaan lahir dari analisisnya sendiri
# --------------------------------------------------------------------------- #


def test_potong_lintang_memunculkan_pertanyaan_sebab_akibat(data, konfig):
    proyek = pp.ProyekPenelitian(desain="potong_lintang", judul="X")
    simulasi = sd.susun(_laporan(data, konfig, proyek), proyek)
    assert any("sebab-akibat" in p.pertanyaan for p in simulasi.pertanyaan)


def test_eksperimen_acak_menanyakan_dasar_kesimpulan_sebabnya(data, konfig):
    """Pertanyaannya berbalik, bukan hilang: penguji tetap menuntut alasannya."""
    proyek = pp.ProyekPenelitian(
        desain="eksperimen", penugasan_acak=True, teknik_sampling="acak_sederhana", judul="X"
    )
    simulasi = sd.susun(_laporan(data, konfig, proyek), proyek)
    assert any("berani menyimpulkan sebab-akibat" in p.pertanyaan for p in simulasi.pertanyaan)


def test_sampling_purposif_memunculkan_pertanyaan_generalisasi(data, konfig):
    proyek = pp.ProyekPenelitian(teknik_sampling="purposif", judul="X")
    simulasi = sd.susun(_laporan(data, konfig, proyek), proyek)
    assert any("Kepada siapa kesimpulan ini berlaku" in p.pertanyaan for p in simulasi.pertanyaan)


def test_sampling_acak_tidak_memunculkan_pertanyaan_itu(data, konfig):
    proyek = pp.ProyekPenelitian(teknik_sampling="acak_sederhana", judul="X")
    simulasi = sd.susun(_laporan(data, konfig, proyek), proyek)
    assert not any("Kepada siapa" in p.pertanyaan for p in simulasi.pertanyaan)


def test_banyak_uji_memunculkan_pertanyaan_kesalahan_tipe_satu(data, konfig):
    simulasi = sd.susun(_laporan(data, konfig), jejak=_jejak())
    cocok = [p for p in simulasi.pertanyaan if "tipe I" in p.pertanyaan]
    assert cocok
    assert "Holm" in cocok[0].jawaban


def test_sedikit_uji_tidak_memunculkan_pertanyaan_itu(data, konfig):
    simulasi = sd.susun(_laporan(data, konfig), jejak=_jejak((0.01,)))
    assert not any("tipe I" in p.pertanyaan for p in simulasi.pertanyaan)


def test_penyimpangan_dari_praregistrasi_ditanyakan(data, konfig):
    proyek = pp.ProyekPenelitian(
        judul="X", praregistrasi=pp.Praregistrasi(uji_direncanakan=["Uji 1"])
    )
    simulasi = sd.susun(_laporan(data, konfig, proyek), proyek, _jejak())
    assert any("tidak ada pada rencana awal" in p.pertanyaan for p in simulasi.pertanyaan)


def test_pertanyaan_menyebut_angka_penggunanya_sendiri(data, konfig):
    """Pembeda dari daftar pertanyaan sidang mana pun: ia menyebut R² Anda."""
    simulasi = sd.susun(_laporan(data, konfig))
    cocok = [p for p in simulasi.pertanyaan if "R²" in p.pertanyaan]
    assert cocok
    assert "0,656" in cocok[0].pertanyaan


def test_pertanyaan_metode_menyebut_metode_utama_bukan_pemeriksaan(data, konfig):
    """Penguji menanyakan mengapa memakai regresi, bukan jarak Mahalanobis."""
    simulasi = sd.susun(_laporan(data, konfig))
    metode = [p for p in simulasi.pertanyaan if p.kategori == sd.METODE]
    assert metode
    assert any("Regresi" in p.pertanyaan or "MANOVA" in p.pertanyaan for p in metode)
    assert not any("Mahalanobis" in p.pertanyaan for p in metode)


def test_temuan_audit_ditanyakan(data, konfig):
    hasil = ad.jalankan_audit(data, km.Kamus.dari_data(data))
    simulasi = sd.susun(_laporan(data, konfig), audit=hasil)
    assert any(p.kategori == sd.DATA for p in simulasi.pertanyaan)


# --------------------------------------------------------------------------- #
# Mutu jawaban model
# --------------------------------------------------------------------------- #


def test_jawaban_tidak_signifikan_meluruskan_kekeliruan_yang_lazim(data, konfig):
    """'Tidak signifikan' bukan berarti 'terbukti tidak ada hubungan'."""
    simulasi = sd.susun(_laporan(data, konfig))
    cocok = [p for p in simulasi.pertanyaan if "tidak signifikan" in p.pertanyaan.lower()]
    if cocok:
        assert "bukan bahwa hubungannya terbukti tidak ada" in cocok[0].jawaban


def test_jawaban_r_kuadrat_menolak_ambang_mutlak(data, konfig):
    simulasi = sd.susun(_laporan(data, konfig))
    cocok = [p for p in simulasi.pertanyaan if "R²" in p.pertanyaan][0]
    assert "Tidak ada ambang mutlak" in cocok.jawaban


def test_setiap_pertanyaan_punya_jawaban_dan_kunci(data, konfig):
    proyek = pp.ProyekPenelitian(judul="X")
    simulasi = sd.susun(
        _laporan(data, konfig, proyek),
        proyek,
        _jejak(),
        ad.jalankan_audit(data, km.Kamus.dari_data(data)),
    )
    for butir in simulasi.pertanyaan:
        assert butir.pertanyaan.endswith("?")
        assert len(butir.jawaban) > 60
        assert butir.kunci


def test_jumlah_pertanyaan_dibatasi(data, konfig):
    """Lebih dari selusin berubah menjadi daftar yang dilewati, bukan latihan."""
    proyek = pp.ProyekPenelitian(judul="X")
    simulasi = sd.susun(
        _laporan(data, konfig, proyek),
        proyek,
        _jejak(),
        ad.jalankan_audit(data, km.Kamus.dari_data(data)),
    )
    assert len(simulasi.pertanyaan) <= sd.MAKS_PERTANYAAN


def test_pertanyaan_wajib_didahulukan(data, konfig):
    proyek = pp.ProyekPenelitian(judul="X")
    simulasi = sd.susun(_laporan(data, konfig, proyek), proyek, _jejak())
    bobot = [p.bobot for p in simulasi.pertanyaan]
    assert bobot == sorted(bobot, key=lambda b: b != sd.WAJIB)


def test_tanpa_ruang_proyek_yang_ditanyakan_justru_itu(data, konfig):
    simulasi = sd.susun(_laporan(data, konfig))
    assert any("rancangan penelitian anda" in p.pertanyaan.lower() for p in simulasi.pertanyaan)


# --------------------------------------------------------------------------- #
# Penilaian jawaban
# --------------------------------------------------------------------------- #


def test_jawaban_kosong_tidak_diluluskan():
    butir = sd.Pertanyaan(
        pertanyaan="Mengapa?", jawaban="karena", kategori=sd.HASIL, kunci=["hubungan"]
    )
    cukup, terlewat = butir.nilai("")
    assert not cukup
    assert terlewat == ["hubungan"]


def test_jawaban_yang_menyinggung_butir_kunci_diluluskan():
    butir = sd.Pertanyaan(
        pertanyaan="Mengapa?",
        jawaban="karena",
        kategori=sd.RANCANGAN,
        kunci=["hubungan", "arah", "perancu"],
    )
    cukup, terlewat = butir.nilai(
        "Karena desainnya potong lintang: yang terbaca hubungan, arah belum pasti, "
        "dan masih ada perancu."
    )
    assert cukup
    assert terlewat == []


def test_butir_yang_terlewat_disebutkan_agar_dapat_diperbaiki():
    butir = sd.Pertanyaan(
        pertanyaan="Mengapa?",
        jawaban="karena",
        kategori=sd.RANCANGAN,
        kunci=["hubungan", "arah", "perancu"],
    )
    _, terlewat = butir.nilai("Karena hanya menunjukkan hubungan.")
    assert "perancu" in terlewat


def test_kategori_dan_bobot_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        sd.Pertanyaan(pertanyaan="p?", jawaban="j", kategori="entah")
    with pytest.raises(ValueError, match="tidak dikenal"):
        sd.Pertanyaan(pertanyaan="p?", jawaban="j", kategori=sd.HASIL, bobot="penting")


# --------------------------------------------------------------------------- #
# Kesiapan
# --------------------------------------------------------------------------- #


def test_lencana_kesiapan_menuntut_seluruh_pertanyaan_wajib_dijawab(data, konfig):
    proyek = pp.ProyekPenelitian(judul="X")
    simulasi = sd.susun(_laporan(data, konfig, proyek), proyek, _jejak())

    lulus, total, siap = sd.kesiapan(simulasi, {})
    assert (lulus, siap) == (0, False)
    assert total == len(simulasi.wajib)

    # Menjawab dengan seluruh kata kunci pada tiap pertanyaan wajib.
    jawaban = {
        nomor: " ".join(butir.kunci)
        for nomor, butir in enumerate(simulasi.pertanyaan)
        if butir.bobot == sd.WAJIB
    }
    lulus, total, siap = sd.kesiapan(simulasi, jawaban)
    assert lulus == total and siap


def test_kesiapan_tanpa_pertanyaan_wajib_tidak_pernah_siap():
    assert sd.kesiapan(sd.Simulasi(), {}) == (0, 0, False)


def test_ringkas_menyandingkan_kategori_dan_bobot(data, konfig):
    simulasi = sd.susun(_laporan(data, konfig))
    tabel = simulasi.ringkas()
    assert list(tabel.columns) == ["Kategori", "Pertanyaan", "Bobot"]
    assert len(tabel) == len(simulasi.pertanyaan)
