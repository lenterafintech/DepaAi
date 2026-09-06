"""Pemuatan berkas data (CSV/TSV/Excel/SPSS) dan ringkasan strukturnya."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
# Banyak peneliti di Indonesia menerima data dalam format SPSS, sehingga .sav
# didukung langsung agar tidak perlu dikonversi lebih dulu.
SPSS_SUFFIXES = {".sav", ".zsav"}
SUPPORTED_SUFFIXES = CSV_SUFFIXES | EXCEL_SUFFIXES | SPSS_SUFFIXES


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
    if suffix in SPSS_SUFFIXES:
        df = _baca_spss(source)
    elif suffix in EXCEL_SUFFIXES:
        df = pd.read_excel(source, sheet_name=sheet_name)
    elif suffix in CSV_SUFFIXES or suffix == "":
        df = pd.read_csv(source, sep=None, engine="python", decimal=decimal)
    else:
        raise UnsupportedFileError(
            f"Format '{suffix}' belum didukung. Gunakan {sorted(SUPPORTED_SUFFIXES)}."
        )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _baca_spss(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Baca berkas SPSS .sav beserta label nilainya.

    Label nilai SPSS dipakai apa adanya (misalnya 1 menjadi "Laki-laki") karena
    itulah yang dilihat peneliti di SPSS; tanpa itu, kolom kategori muncul sebagai
    angka tanpa makna dan mudah keliru diperlakukan sebagai variabel numerik.
    """
    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - hanya bila paket dicopot
        raise UnsupportedFileError(
            "Membaca berkas SPSS memerlukan paket pyreadstat. Pasang dengan "
            "'pip install pyreadstat', atau simpan data Anda sebagai CSV."
        ) from exc

    # pyreadstat menuntut jalur berkas, sedangkan unggahan Streamlit berupa aliran
    # dalam memori; aliran itu disalin dulu ke berkas sementara.
    if hasattr(source, "read"):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as sementara:
            sementara.write(source.read())
            jalur = sementara.name
        try:
            df, _ = pyreadstat.read_sav(jalur, apply_value_formats=True)
        finally:
            Path(jalur).unlink(missing_ok=True)
    else:
        df, _ = pyreadstat.read_sav(str(source), apply_value_formats=True)
    return df


def meta_spss(source: str | Path) -> pd.DataFrame:
    """Label variabel dari berkas SPSS, berguna sebagai kamus data."""
    import pyreadstat

    _, meta = pyreadstat.read_sav(str(source), metadataonly=True)
    label = meta.column_names_to_labels or {}
    return pd.DataFrame(
        {
            "Kolom": list(meta.column_names),
            "Label": [str(label.get(k) or "") for k in meta.column_names],
        }
    )


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
