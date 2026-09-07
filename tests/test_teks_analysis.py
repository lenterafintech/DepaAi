"""Uji analisis teks: statistik dasar, tokenisasi/pembersihan, dan frekuensi kata."""

from __future__ import annotations

import pandas as pd
import pytest

from nalardata import teks_analysis as tk


@pytest.fixture(scope="module")
def teks() -> pd.Series:
    return pd.Series(
        [
            "Penelitian ini menganalisis dampak pelatihan terhadap kinerja karyawan.",
            "Analisis data menunjukkan bahwa pelatihan meningkatkan kinerja secara signifikan.",
            "Kinerja karyawan yang mengikuti pelatihan lebih baik dibandingkan yang tidak.",
            "Pelatihan rutin sangat membantu meningkatkan kinerja tim kerja.",
            None,
            "   ",
        ]
    )


def test_statistik_dasar_menghitung_dokumen_valid_saja(teks):
    hasil = tk.statistik_dasar(teks)
    assert hasil.n_dokumen == 4  # baris kosong/None dibuang
    assert hasil.jumlah_kata > 0
    assert hasil.kosakata_unik > 0
    assert hasil.rata_kata_per_dokumen == pytest.approx(hasil.jumlah_kata / 4)


def test_statistik_dasar_seluruh_kosong_ditolak():
    with pytest.raises(ValueError):
        tk.statistik_dasar(pd.Series([None, "", "   "]))


def test_bersihkan_token_menyatukan_imbuhan(teks):
    """'pelatihan'/'melatih' dan 'kinerja'/'kerja' berbagi kata dasar setelah
    Sastrawi — inti dari kenapa stemming dipakai sebelum menghitung frekuensi."""
    token = tk.bersihkan_token(teks, hapus_stopword=True, stem=True)
    assert "latih" in token
    fd_kerja = token.count("kerja")
    assert fd_kerja >= 3  # 'kinerja' muncul di 3 dokumen, ikut jadi 'kerja'


def test_bersihkan_token_membuang_stopword(teks):
    token = tk.bersihkan_token(teks, hapus_stopword=True, stem=False)
    assert "yang" not in token
    assert "ini" not in token
    assert "terhadap" not in token


def test_bersihkan_token_stopword_dipertahankan_bila_dimatikan(teks):
    dengan = tk.bersihkan_token(teks, hapus_stopword=True, stem=False)
    tanpa = tk.bersihkan_token(teks, hapus_stopword=False, stem=False)
    assert len(tanpa) > len(dengan)
    assert "yang" in tanpa


def test_stopword_tambahan_ikut_dibuang(teks):
    token = tk.bersihkan_token(
        teks, hapus_stopword=True, stem=True, stopword_tambahan=["karyawan"]
    )
    assert "karyawan" not in token


def test_frekuensi_kata_terurut_menurun(teks):
    token = tk.bersihkan_token(teks)
    tabel = tk.frekuensi_kata(token, n_teratas=10)
    assert list(tabel["Frekuensi"]) == sorted(tabel["Frekuensi"], reverse=True)
    assert tabel["Persentase"].sum() <= 100.0 + 1e-6


def test_frekuensi_kata_daftar_kosong():
    tabel = tk.frekuensi_kata([], n_teratas=10)
    assert tabel.empty
    assert list(tabel.columns) == ["Kata", "Frekuensi", "Persentase"]


def test_frekuensi_kata_membatasi_n_teratas(teks):
    token = tk.bersihkan_token(teks, hapus_stopword=False, stem=False)
    tabel = tk.frekuensi_kata(token, n_teratas=3)
    assert len(tabel) <= 3
