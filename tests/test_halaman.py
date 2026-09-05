"""Uji asap seluruh halaman Streamlit: memastikan tidak ada galat saat dirender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p for p in (ROOT / "views").glob("*.py") if p.stem != "beranda")
SAMPLE = ROOT / "data" / "contoh_data_nasabah.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


def _run(path: Path, sample: pd.DataFrame | None) -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=180)
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


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_halaman_berjalan_dengan_data(page: Path, sample: pd.DataFrame):
    app = _run(page, sample)
    assert not app.exception, f"{page.name}: {app.exception}"
    assert not app.error, f"{page.name}: {[e.value for e in app.error]}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_halaman_meminta_data_saat_kosong(page: Path):
    app = _run(page, None)
    assert not app.exception
    assert any("Belum ada data" in w.value for w in app.warning)
