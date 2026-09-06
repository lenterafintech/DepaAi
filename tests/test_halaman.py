"""Uji asap seluruh halaman Streamlit: memastikan tidak ada galat saat dirender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
# Halaman yang sengaja tetap berguna tanpa data aktif. Beranda dan entri data
# justru tempat data dibuat; Laporan Hasil menampilkan hasil yang sudah disimpan,
# yang tetap sah dibaca sekalipun datanya sudah tidak dimuat lagi.
MANDIRI = {"beranda", "entri_data", "akun", "masuk", "laporan"}
PAGES = sorted(p for p in (ROOT / "views").glob("*.py") if p.stem not in MANDIRI)
SEMUA = sorted((ROOT / "views").glob("*.py"))
SAMPLE = ROOT / "data" / "contoh_data_nasabah.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


def _run(path: Path, sample: pd.DataFrame | None, paket: str = "profesional") -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=180)
    # Halaman diuji pada paket penuh; pembatasan paket diuji terpisah.
    app.session_state["paket_langganan"] = paket
    if sample is not None:
        app.session_state["dataset"] = sample
        app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    return app.run()


def test_beranda_tanpa_data():
    app = _run(ROOT / "views" / "beranda.py", None)
    assert not app.exception


def test_beranda_dengan_data(sample):
    app = _run(ROOT / "views" / "beranda.py", sample)
    assert not app.exception
    assert any("Pratinjau data" in md.value for md in app.markdown)


@pytest.mark.parametrize("page", SEMUA, ids=lambda p: p.stem)
def test_halaman_berjalan_dengan_data(page: Path, sample: pd.DataFrame):
    app = _run(page, sample)
    assert not app.exception, f"{page.name}: {app.exception}"
    assert not app.error, f"{page.name}: {[e.value for e in app.error]}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_halaman_meminta_data_saat_kosong(page: Path):
    app = _run(page, None)
    assert not app.exception
    assert any("Belum ada data" in w.value for w in app.warning)


def test_entri_data_berjalan_tanpa_data():
    """Halaman entri harus tetap dapat dipakai justru saat belum ada data."""
    app = _run(ROOT / "views" / "entri_data.py", None)
    assert not app.exception
    assert not app.error
    assert any("Tentukan kolom" in str(sub.value) for sub in app.subheader)


def test_halaman_terkunci_pada_paket_gratis(sample):
    """Metode di luar paket harus berhenti dengan ajakan naik paket, bukan galat."""
    app = _run(ROOT / "views" / "manova.py", sample, paket="gratis")
    assert not app.exception
    assert any("tidak termasuk dalam paket" in w.value for w in app.warning)


def test_halaman_terbuka_pada_paket_profesional(sample):
    app = _run(ROOT / "views" / "manova.py", sample, paket="profesional")
    assert not app.exception
    assert not any("tidak termasuk dalam paket" in w.value for w in app.warning)


def test_data_melebihi_batas_paket_ditolak(sample):
    """Data yang lebih besar dari batas paket dihentikan dengan pesan yang jelas."""
    app = _run(ROOT / "views" / "eksplorasi.py", sample, paket="gratis")
    assert not app.exception
    assert any("membatasi" in w.value for w in app.warning)


def test_halaman_akun_menampilkan_paket_aktif():
    app = _run(ROOT / "views" / "akun.py", None, paket="gratis")
    assert not app.exception
    assert any("Masa perkenalan" in i.value for i in app.info)


def test_halaman_masuk_menawarkan_pendaftaran():
    app = _run(ROOT / "views" / "masuk.py", None, paket="gratis")
    assert not app.exception
    assert any("Paket yang tersedia" in str(s.value) for s in app.subheader)


def test_laporan_hasil_tanpa_keranjang_menjelaskan_caranya():
    """Halaman Laporan Hasil harus berguna sekalipun belum ada yang disimpan."""
    app = _run(ROOT / "views" / "laporan.py", None)
    assert not app.exception
    pesan = " ".join(i.value for i in app.info)
    assert "Simpan ke laporan" in pesan


def test_laporan_hasil_menampilkan_isi_keranjang(sample):
    from lentera_mva import keranjang as kr

    isi = kr.Keranjang()
    isi.tambah_tabel("Regresi linear", "Koefisien regresi", sample.head(3))
    app = AppTest.from_file(str(ROOT / "views" / "laporan.py"), default_timeout=180)
    app.session_state["paket_langganan"] = "profesional"
    app.session_state["keranjang_hasil"] = isi
    app.run()
    assert not app.exception
    tajuk = " ".join(h.value for h in app.subheader)
    assert "Daftar isi" in tajuk and "Ekspor laporan" in tajuk
