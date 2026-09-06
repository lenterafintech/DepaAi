"""Uji penyusun naskah Bab III, IV, V, dan artikel IMRAD.

Yang diuji adalah apakah bahan yang sudah dikumpulkan aplikasi benar-benar sampai
ke bab yang tepat: rancangan penelitian ke Bab III, batas kesimpulan ke Bab V, dan
definisi operasional dari Kamus Variabel ke bagian yang hampir selalu ditanyakan
penguji.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nalardata import ekspor
from nalardata import kamus as km
from nalardata import narrative as nr
from nalardata import naskah as nk
from nalardata import proyek_penelitian as pp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


@pytest.fixture(scope="module")
def penelitian() -> pp.ProyekPenelitian:
    return pp.ProyekPenelitian(
        judul="Faktor yang berhubungan dengan skor kredit",
        populasi="Nasabah cabang Jakarta",
        ukuran_populasi=2500,
        unit_analisis="Nasabah perorangan",
        target_sampel=380,
        pertanyaan=["Apakah rasio utang berhubungan dengan skor kredit?"],
        teknik_sampling="purposif",
        desain="potong_lintang",
    )


@pytest.fixture(scope="module")
def kamus(data) -> km.Kamus:
    k = km.Kamus.dari_data(data)
    k.tetapkan(
        "skor_kredit",
        definisi="Skor kredit internal berskala 300-850",
        satuan="poin",
        peran="outcome",
    )
    return k


@pytest.fixture(scope="module")
def laporan(data, penelitian) -> nr.Laporan:
    konfig = nr.Konfigurasi(
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
    return nr.analisis_dan_laporan(data, konfig, penelitian)[1]


def _teks(dokumen) -> str:
    """Seluruh teks dokumen, termasuk butir daftar yang tersimpan di ruas ``poin``."""
    bagian = []
    for blok in dokumen.blok:
        bagian.append(blok.teks or "")
        bagian.extend(blok.poin or [])
    return " ".join(bagian)


# --------------------------------------------------------------------------- #
# Bab III
# --------------------------------------------------------------------------- #


def test_bab3_memuat_rancangan_populasi_dan_sampling(laporan, penelitian, kamus):
    teks = _teks(nk.susun(laporan, nk.BAB3, penelitian, kamus))
    assert "potong lintang" in teks
    assert "Nasabah cabang Jakarta" in teks
    assert "purposif" in teks
    assert "Nasabah perorangan" in teks


def test_bab3_menyebut_jumlah_populasi_dan_target_sampel(laporan, penelitian, kamus):
    teks = _teks(nk.susun(laporan, nk.BAB3, penelitian, kamus))
    assert "2.500" in teks
    assert "380" in teks


def test_bab3_memuat_tabel_definisi_operasional(laporan, penelitian, kamus):
    """Bagian yang hampir selalu ditanyakan penguji."""
    dokumen = nk.susun(laporan, nk.BAB3, penelitian, kamus)
    tabel = [b.tabel for b in dokumen.blok if b.jenis == "tabel" and b.tabel is not None]
    assert tabel
    assert "Definisi operasional" in tabel[0].columns
    assert "Skor kredit internal berskala 300-850" in tabel[0].values


def test_definisi_yang_belum_diisi_ditandai_bukan_dikosongkan(laporan, penelitian, kamus):
    dokumen = nk.susun(laporan, nk.BAB3, penelitian, kamus)
    tabel = [b.tabel for b in dokumen.blok if b.jenis == "tabel"][0]
    assert (tabel["Definisi operasional"] == "[belum diisi]").any()


def test_bab3_tanpa_kamus_menjelaskan_yang_hilang(laporan, penelitian):
    teks = _teks(nk.susun(laporan, nk.BAB3, penelitian, None))
    assert "belum diisi pada Kamus Variabel" in teks


def test_bab3_tanpa_ruang_proyek_meminta_diisi_dulu(laporan, kamus):
    teks = _teks(nk.susun(laporan, nk.BAB3, None, kamus))
    assert "belum diisi pada halaman Ruang Proyek" in teks


def test_bab3_menyebut_praregistrasi_bila_ada(laporan, kamus):
    proyek = pp.ProyekPenelitian(
        judul="X",
        praregistrasi=pp.Praregistrasi(uji_direncanakan=["Regresi linear berganda"]),
    )
    teks = _teks(nk.susun(laporan, nk.BAB3, proyek, kamus))
    assert "dicatat sebelum data diperiksa" in teks


# --------------------------------------------------------------------------- #
# Bab IV
# --------------------------------------------------------------------------- #


def test_bab4_memuat_hasil_seluruh_temuan(laporan, penelitian, kamus):
    dokumen = nk.susun(laporan, nk.BAB4, penelitian, kamus)
    teks = _teks(dokumen)
    for temuan in laporan.temuan:
        assert temuan.judul in teks


def test_bab4_memuat_seluruh_tabel_hasil(laporan, penelitian, kamus):
    dokumen = nk.susun(laporan, nk.BAB4, penelitian, kamus)
    tabel = [b for b in dokumen.blok if b.jenis == "tabel"]
    assert len(tabel) >= len(laporan.tabel)


def test_bab4_menyatakan_pembahasan_belum_termuat(laporan, penelitian, kamus):
    """Bagian yang dinilai penguji tidak boleh tampak sudah selesai."""
    teks = _teks(nk.susun(laporan, nk.BAB4, penelitian, kamus))
    assert "harus Anda tulis sendiri" in teks


# --------------------------------------------------------------------------- #
# Bab V
# --------------------------------------------------------------------------- #


def test_bab5_menjawab_pertanyaan_penelitian(laporan, penelitian, kamus):
    teks = _teks(nk.susun(laporan, nk.BAB5, penelitian, kamus))
    assert penelitian.pertanyaan[0] in teks


def test_bab5_memuat_seluruh_keterbatasan(laporan, penelitian, kamus):
    teks = _teks(nk.susun(laporan, nk.BAB5, penelitian, kamus))
    for batas in laporan.keterbatasan:
        assert batas in teks


def test_bab5_menegaskan_batas_berasal_dari_rancangan(laporan, penelitian, kamus):
    teks = _teks(nk.susun(laporan, nk.BAB5, penelitian, kamus))
    assert "Menghapusnya dari naskah tidak membuatnya tidak berlaku" in teks


def test_bab5_pada_potong_lintang_bebas_kosakata_sebab(laporan, penelitian, kamus):
    """Kunci kausalitas harus tetap berlaku sampai ke naskah."""
    from nalardata import pagar

    teks = _teks(nk.susun(laporan, nk.BAB5, penelitian, kamus))
    assert pagar.periksa_kausalitas(teks, penelitian) == []


# --------------------------------------------------------------------------- #
# Umum
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gaya", list(nk.GAYA))
def test_setiap_gaya_dibuka_peringatan_bukan_ditutup(gaya, laporan, penelitian, kamus):
    """Peringatan di akhir ditemukan setelah isinya terlanjur disalin."""
    dokumen = nk.susun(laporan, gaya, penelitian, kamus)
    jenis = [b.jenis for b in dokumen.blok]
    assert jenis.index("catatan") <= 3
    assert "harus Anda tulis sendiri" in dokumen.blok[jenis.index("catatan")].teks


@pytest.mark.parametrize("gaya", list(nk.GAYA))
def test_setiap_gaya_memakai_judul_penelitian(gaya, laporan, penelitian, kamus):
    dokumen = nk.susun(laporan, gaya, penelitian, kamus)
    assert penelitian.judul in dokumen.judul


@pytest.mark.parametrize("gaya", list(nk.GAYA))
@pytest.mark.parametrize("format_berkas", ["docx", "pdf", "html", "md"])
def test_setiap_gaya_dapat_diekspor(gaya, format_berkas, laporan, penelitian, kamus):
    dokumen = nk.susun(laporan, gaya, penelitian, kamus)
    berkas = ekspor.bangun(dokumen, format_berkas)
    assert len(berkas) > 500


def test_nama_berkas_menyebut_babnya(laporan, penelitian, kamus):
    dokumen = nk.susun(laporan, nk.BAB4, penelitian, kamus)
    assert "bab4" in ekspor.nama_berkas(dokumen, "docx")


def test_gaya_asing_ditolak(laporan):
    with pytest.raises(ValueError, match="tidak dikenal"):
        nk.susun(laporan, "bab6")


def test_imrad_memuat_abstrak_hasil_dan_diskusi(laporan, penelitian, kamus):
    teks = _teks(nk.susun(laporan, nk.IMRAD, penelitian, kamus))
    for bagian in ("Abstrak", "Metode", "Hasil", "Diskusi", "Keterbatasan"):
        assert bagian in teks
