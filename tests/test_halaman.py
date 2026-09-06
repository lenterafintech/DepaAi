"""Uji asap seluruh halaman Streamlit: memastikan tidak ada galat saat dirender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
# Beranda dan halaman entri sengaja tetap berguna tanpa data aktif: keduanya
# justru tempat data dibuat, jadi tidak ikut diuji sebagai halaman yang menuntut data.
MANDIRI = {"beranda", "entri_data", "akun"}
PAGES = sorted(p for p in (ROOT / "views").glob("*.py") if p.stem not in MANDIRI)
SEMUA = sorted((ROOT / "views").glob("*.py"))
SAMPLE = ROOT / "data" / "contoh_data_nasabah.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


def _run(path: Path, sample: pd.DataFrame | None, paket: str = "pro") -> AppTest:
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


def test_halaman_terbuka_pada_paket_pro(sample):
    app = _run(ROOT / "views" / "manova.py", sample, paket="pro")
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
    assert any("Mode uji coba" in i.value for i in app.info)
