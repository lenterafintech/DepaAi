"""Persiapan data: seleksi kolom, penanganan missing, encoding, dan penskalaan."""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_STRATEGIES = ("hapus baris", "rata-rata", "median", "modus", "isi nol")
SCALING_METHODS = ("tanpa penskalaan", "z-score", "min-max", "robust")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_levels: int = 50) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique(dropna=True) <= max_levels:
            out.append(c)
    return out


def handle_missing(df: pd.DataFrame, strategy: str = "hapus baris") -> pd.DataFrame:
    """Tangani nilai hilang. Kolom kategorik selalu diisi modus kecuali baris dihapus."""
    if strategy not in MISSING_STRATEGIES:
        raise ValueError(f"Strategi '{strategy}' tidak dikenal.")
    if strategy == "hapus baris":
        return df.dropna().reset_index(drop=True)

    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.notna().all():
            continue
        if pd.api.types.is_numeric_dtype(s):
            if strategy == "rata-rata":
                fill = s.mean()
            elif strategy == "median":
                fill = s.median()
            elif strategy == "modus":
                fill = s.mode().iloc[0] if not s.mode().empty else 0
            else:
                fill = 0
        else:
            fill = s.mode().iloc[0] if not s.mode().empty else ""
        out[col] = s.fillna(fill)
    return out


def scale(df: pd.DataFrame, method: str = "z-score") -> pd.DataFrame:
    """Penskalaan kolom numerik. Kolom dengan variasi nol dibiarkan nol."""
    if method not in SCALING_METHODS:
        raise ValueError(f"Metode penskalaan '{method}' tidak dikenal.")
    if method == "tanpa penskalaan":
        return df.copy()

    out = df.astype(float).copy()
    for col in out.columns:
        s = out[col]
        if method == "z-score":
            sd = s.std(ddof=1)
            out[col] = (s - s.mean()) / sd if sd > 0 else 0.0
        elif method == "min-max":
            rng = s.max() - s.min()
            out[col] = (s - s.min()) / rng if rng > 0 else 0.0
        else:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            out[col] = (s - s.median()) / iqr if iqr > 0 else 0.0
    return out


def encode_categorical(
    df: pd.DataFrame, columns: list[str], method: str = "one-hot"
) -> pd.DataFrame:
    """Ubah kolom kategorik menjadi numerik (one-hot dengan drop_first, atau ordinal)."""
    out = df.copy()
    if not columns:
        return out
    if method == "one-hot":
        return pd.get_dummies(out, columns=list(columns), drop_first=True, dtype=float)
    for col in columns:
        out[col] = pd.Categorical(out[col]).codes.astype(float)
    return out


def design_matrix(
    df: pd.DataFrame, predictors: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Bangun matriks prediktor; kolom kategorik di-dummy (drop_first).

    Mengembalikan matriks dan daftar kolom kategorik yang diperluas.
    """
    sub = df[list(predictors)].copy()
    cat_cols = [c for c in sub.columns if not pd.api.types.is_numeric_dtype(sub[c])]
    if cat_cols:
        sub = pd.get_dummies(sub, columns=cat_cols, drop_first=True, dtype=float)
    return sub.astype(float), cat_cols


def clean_subset(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ambil subset kolom dan buang baris yang tidak lengkap."""
    return df[list(columns)].replace([np.inf, -np.inf], np.nan).dropna()
