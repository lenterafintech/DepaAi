"""Uji MANCOVA serta mesin CFA, analisis jalur, dan SEM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lentera_mva import mancova as mc
from lentera_mva import sem_analysis as sem

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def nasabah() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


@pytest.fixture(scope="module")
def kuesioner() -> pd.DataFrame:
    """Dua konstruk laten, empat butir masing-masing, dengan jalur A → B."""
    rng = np.random.default_rng(42)
    n = 400
    a = rng.normal(size=n)
    b = 0.55 * a + rng.normal(scale=0.8, size=n)
    data = {}
    for i in range(1, 5):
        data[f"A{i}"] = a + rng.normal(scale=0.45, size=n)
        data[f"B{i}"] = b + rng.normal(scale=0.45, size=n)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# MANCOVA
# --------------------------------------------------------------------------- #


def test_mancova_menghasilkan_uji_multivariat(nasabah):
    hasil = mc.run_mancova(
        nasabah,
        ["skor_kredit", "saldo_tabungan"],
        "segmen_usaha",
        ["usia", "lama_usaha_tahun"],
    )
    assert set(hasil.multivariate["Statistik"]) >= {"Wilks' lambda", "Pillai's trace"}
    assert hasil.n == len(nasabah)
    assert list(hasil.univariate["Variabel"]) == ["skor_kredit", "saldo_tabungan"]
    assert "Eta-squared parsial" in hasil.univariate.columns


def test_mancova_melaporkan_pengaruh_kovariat(nasabah):
    hasil = mc.run_mancova(
        nasabah, ["skor_kredit", "saldo_tabungan"], "segmen_usaha", ["lama_usaha_tahun"]
    )
    assert list(hasil.pengaruh_kovariat["Kovariat"]) == ["lama_usaha_tahun"]
    assert hasil.pengaruh_kovariat.loc[0, "Signifikan"] in {"Ya", "Tidak"}


def test_rata_terkoreksi_berbeda_dari_mentah(nasabah):
    hasil = mc.run_mancova(
        nasabah, ["skor_kredit", "saldo_tabungan"], "segmen_usaha", ["pendapatan_bulanan"]
    )
    banding = mc.bandingkan_rata(hasil)
    assert set(banding.columns) == {
        "Kelompok", "Variabel", "Rata-rata mentah", "Rata-rata terkoreksi", "Selisih",
    }
    # Kovariat yang berkaitan dengan kelompok menggeser rata-rata setelah dikoreksi.
    assert banding["Selisih"].abs().max() > 0


def test_mancova_menguji_homogenitas_kemiringan(nasabah):
    hasil = mc.run_mancova(
        nasabah, ["skor_kredit", "saldo_tabungan"], "segmen_usaha", ["usia"]
    )
    assert len(hasil.homogenitas_slope) == 2  # dua dependen × satu kovariat
    assert set(hasil.homogenitas_slope["Kesimpulan"]) <= {"Homogen", "Tidak homogen"}
    assert isinstance(hasil.slope_homogen(), bool)


def test_mancova_menolak_masukan_keliru(nasabah):
    with pytest.raises(ValueError):  # tanpa kovariat, seharusnya MANOVA
        mc.run_mancova(nasabah, ["skor_kredit", "saldo_tabungan"], "segmen_usaha", [])
    with pytest.raises(ValueError):  # dependen sekaligus kovariat
        mc.run_mancova(nasabah, ["skor_kredit", "usia"], "segmen_usaha", ["usia"])
    with pytest.raises(ValueError):  # hanya satu dependen
        mc.run_mancova(nasabah, ["skor_kredit"], "segmen_usaha", ["usia"])


# --------------------------------------------------------------------------- #
# Spesifikasi model
# --------------------------------------------------------------------------- #


def test_spesifikasi_cfa_dan_jalur():
    spec = sem.spesifikasi_cfa({"A": ["A1", "A2"], "B": ["B1", "B2"]})
    assert spec.splitlines() == ["A =~ A1 + A2", "B =~ B1 + B2"]
    assert sem.spesifikasi_jalur({"y": ["x1", "x2"]}) == "y ~ x1 + x2"
    gabung = sem.spesifikasi_sem({"A": ["A1", "A2"]}, {"y": ["A"]})
    assert "A =~ A1 + A2" in gabung and "y ~ A" in gabung


def test_spesifikasi_menolak_konstruk_kurang_butir():
    with pytest.raises(ValueError):
        sem.spesifikasi_cfa({"A": ["A1"]})
    with pytest.raises(ValueError):
        sem.spesifikasi_jalur({})


def test_pembacaan_variabel_dan_laten():
    spec = "A =~ A1 + A2\nB =~ B1 + B2\nB ~ A"
    assert sem._laten_dalam(spec) == ["A", "B"]
    assert sem._variabel_dalam(spec) == ["A1", "A2", "B1", "B2"]


# --------------------------------------------------------------------------- #
# CFA, jalur, dan SEM
# --------------------------------------------------------------------------- #


def test_cfa_memulihkan_struktur_dua_konstruk(kuesioner):
    konstruk = {"A": [f"A{i}" for i in range(1, 5)], "B": [f"B{i}" for i in range(1, 5)]}
    hasil = sem.jalankan(kuesioner, sem.spesifikasi_cfa(konstruk))
    muatan = hasil.muatan()
    assert len(muatan) == 8
    assert (muatan["Estimasi baku"].abs() > 0.5).all()  # seluruh butir memuat kuat
    assert set(hasil.laten) == {"A", "B"}
    assert hasil.jalur().empty  # CFA murni tidak punya jalur struktural


def test_indeks_kecocokan_lengkap_dan_berambang(kuesioner):
    konstruk = {"A": [f"A{i}" for i in range(1, 5)], "B": [f"B{i}" for i in range(1, 5)]}
    hasil = sem.jalankan(kuesioner, sem.spesifikasi_cfa(konstruk))
    tabel = sem.tabel_kecocokan(hasil)
    assert set(tabel["Indeks"]) == {"Chi-square", "chi2/df", "CFI", "TLI", "RMSEA", "GFI", "AGFI", "NFI"}
    assert set(tabel["Keputusan"]) <= {"Memenuhi", "Tidak memenuhi"}
    assert hasil.cocok()  # model yang benar spesifikasinya harus lolos


def test_reliabilitas_konstruk_dari_muatan(kuesioner):
    konstruk = {"A": [f"A{i}" for i in range(1, 5)], "B": [f"B{i}" for i in range(1, 5)]}
    hasil = sem.jalankan(kuesioner, sem.spesifikasi_cfa(konstruk))
    tabel = sem.reliabilitas_konstruk(hasil)
    assert list(tabel["Konstruk"]) == ["A", "B"]
    assert (tabel["CR"] > 0.7).all() and (tabel["AVE"] > 0.5).all()
    assert (tabel["Keputusan"] == "Memenuhi").all()


def test_sem_memulihkan_jalur_struktural(kuesioner):
    konstruk = {"A": [f"A{i}" for i in range(1, 5)], "B": [f"B{i}" for i in range(1, 5)]}
    hasil = sem.jalankan(kuesioner, sem.spesifikasi_sem(konstruk, {"B": ["A"]}))
    jalur = hasil.jalur()
    assert len(jalur) == 1
    baris = jalur.iloc[0]
    assert baris["Dari"] == "A" and baris["Ke"] == "B"
    assert baris["p-value"] < 0.001
    assert baris["Estimasi baku"] > 0.3  # arah dan besaran sesuai yang ditanam


def test_analisis_jalur_variabel_teramati(nasabah):
    spec = sem.spesifikasi_jalur(
        {
            "skor_kredit": ["pendapatan_bulanan", "rasio_utang_pendapatan"],
            "rasio_utang_pendapatan": ["pendapatan_bulanan"],
        }
    )
    hasil = sem.jalankan(nasabah, spec)
    assert hasil.laten == []
    assert len(hasil.jalur()) == 3
    efek = sem.efek_langsung_tidak_langsung(
        hasil, "pendapatan_bulanan", "rasio_utang_pendapatan", "skor_kredit"
    )
    # Efek tidak langsung adalah hasil kali kedua jalur pembentuknya.
    assert efek["Efek tidak langsung"] == pytest.approx(efek["a (X→M)"] * efek["b (M→Y)"])
    assert efek["Efek total"] == pytest.approx(
        efek["Efek langsung (X→Y)"] + efek["Efek tidak langsung"]
    )


def test_bootstrap_mediasi_memberi_interval(nasabah):
    spec = sem.spesifikasi_jalur(
        {
            "skor_kredit": ["pendapatan_bulanan", "rasio_utang_pendapatan"],
            "rasio_utang_pendapatan": ["pendapatan_bulanan"],
        }
    )
    tabel = sem.bootstrap_mediasi(
        nasabah, spec, "pendapatan_bulanan", "rasio_utang_pendapatan", "skor_kredit",
        n_boot=40, seed=3,
    )
    baris = tabel.iloc[0]
    assert baris["IK 95% Bawah"] < baris["Efek tidak langsung"] < baris["IK 95% Atas"]
    assert baris["Resample berhasil"] >= 20
    assert "Mediasi" in baris["Keputusan"] or baris["Keputusan"] == "Tidak ada mediasi"


def test_sem_menolak_variabel_yang_tidak_ada(kuesioner):
    with pytest.raises(ValueError):
        sem.jalankan(kuesioner, "A =~ A1 + tidak_ada")
    with pytest.raises(ValueError):
        sem.jalankan(kuesioner, "   ")


def test_catatan_chi_square_muncul_saat_signifikan(nasabah):
    spec = sem.spesifikasi_cfa(
        {"kapasitas": ["pendapatan_bulanan", "saldo_tabungan", "plafon_pinjaman"]}
    )
    hasil = sem.jalankan(nasabah, spec)
    catatan = sem.catatan_chi_square(hasil)
    p = float(hasil.statistik["chi2 p-value"])
    assert (catatan == "") == (p > 0.05)
