"""Pemuatan berkas data (CSV/TSV/Excel) dan ringkasan strukturnya."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES


class UnsupportedFileError(ValueError):
    """Format berkas tidak didukung."""


def _suffix(source: Any, filename: str | None) -> str:
    name = filename or getattr(source, "name", None) or str(source)
    return Path(str(name)).suffix.lower()


def excel_sheet_names(source: Any) -> list[str]:
    return pd.ExcelFile(source).sheet_names


def load_table(
    source: str | Path | BinaryIO,
    filename: str | None = None,
    sheet_name: str | int = 0,
    decimal: str = ".",
) -> pd.DataFrame:
    """Baca berkas tabular menjadi DataFrame.

    Pemisah kolom CSV dideteksi otomatis agar berkas Indonesia yang memakai
    titik koma tetap terbaca.
    """
    suffix = _suffix(source, filename)
    if suffix in EXCEL_SUFFIXES:
        df = pd.read_excel(source, sheet_name=sheet_name)
    elif suffix in CSV_SUFFIXES or suffix == "":
        df = pd.read_csv(source, sep=None, engine="python", decimal=decimal)
    else:
        raise UnsupportedFileError(
            f"Format '{suffix}' belum didukung. Gunakan {sorted(SUPPORTED_SUFFIXES)}."
        )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan tiap kolom: tipe, jumlah unik, missing, contoh nilai."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "Variabel": col,
                "Tipe": str(s.dtype),
                "Skala": "Numerik" if pd.api.types.is_numeric_dtype(s) else "Kategorik",
                "Terisi": int(s.notna().sum()),
                "Missing": int(s.isna().sum()),
                "% Missing": round(float(s.isna().mean() * 100), 2),
                "Nilai Unik": int(s.nunique(dropna=True)),
                "Contoh": ", ".join(map(str, s.dropna().unique()[:3])),
            }
        )
    return pd.DataFrame(rows)
