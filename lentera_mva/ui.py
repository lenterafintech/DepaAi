"""Pembantu antarmuka Streamlit yang dipakai bersama oleh seluruh halaman."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lentera_mva import preprocessing

DATA_KEY = "dataset"
NAME_KEY = "dataset_name"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "contoh_data_nasabah.csv"


def page_setup(title: str, icon: str = "📊") -> None:
    """Judul halaman. Konfigurasi global diatur sekali di app.py."""
    if not st.session_state.get("_page_configured"):
        st.set_page_config(page_title="Lentera MVA", page_icon="📊", layout="wide")
        st.session_state["_page_configured"] = True
    st.title(f"{icon} {title}")


def set_dataset(df: pd.DataFrame, name: str) -> None:
    st.session_state[DATA_KEY] = df
    st.session_state[NAME_KEY] = name


def get_dataset() -> pd.DataFrame | None:
    return st.session_state.get(DATA_KEY)


def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def require_dataset() -> pd.DataFrame:
    """Ambil data aktif; hentikan halaman dengan pesan bila belum ada."""
    df = get_dataset()
    if df is None:
        st.warning(
            "Belum ada data. Buka halaman **Beranda & Data** untuk mengunggah berkas "
            "CSV/Excel atau memuat contoh data."
        )
        if st.button("Muat contoh data nasabah", type="primary"):
            set_dataset(load_sample(), "contoh_data_nasabah.csv")
            st.rerun()
        st.stop()
    return df


def sidebar_info() -> None:
    df = get_dataset()
    with st.sidebar:
        st.markdown("### Data aktif")
        if df is None:
            st.caption("Belum ada data dimuat.")
            return
        st.caption(st.session_state.get(NAME_KEY, "data"))
        st.metric("Baris", f"{len(df):,}".replace(",", "."))
        st.metric("Kolom", df.shape[1])


def numeric_selector(
    df: pd.DataFrame,
    label: str = "Variabel numerik yang dianalisis",
    default_count: int = 6,
    min_selection: int = 2,
    key: str | None = None,
) -> list[str]:
    options = preprocessing.numeric_columns(df)
    if len(options) < min_selection:
        st.error(
            f"Data hanya punya {len(options)} kolom numerik, minimal {min_selection} dibutuhkan."
        )
        st.stop()
    default = options[: min(default_count, len(options))]
    selected = st.multiselect(label, options, default=default, key=key)
    if len(selected) < min_selection:
        st.info(f"Pilih minimal {min_selection} variabel untuk melanjutkan.")
        st.stop()
    return selected


def group_selector(
    df: pd.DataFrame,
    label: str = "Variabel kelompok",
    max_levels: int = 20,
    key: str | None = None,
) -> str:
    candidates = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= max_levels]
    # Kolom kategorik didahulukan karena lebih lazim berperan sebagai penanda kelompok.
    options = [c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])] + [
        c for c in candidates if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not options:
        st.error(
            f"Tidak ada kolom yang cocok sebagai variabel kelompok "
            f"(butuh 2 sampai {max_levels} kategori)."
        )
        st.stop()
    return st.selectbox(label, options, key=key)


def format_number(value: object) -> str:
    """Format angka desimal: notasi ilmiah untuk nilai sangat kecil/besar."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    number = float(value)
    magnitude = abs(number)
    if magnitude != 0 and (magnitude < 1e-4 or magnitude >= 1e7):
        return f"{number:.3e}"
    if number.is_integer():
        return f"{number:,.0f}"
    return f"{number:,.4f}"


def styled(df: pd.DataFrame):
    """Terapkan pemformatan angka pada seluruh kolom desimal."""
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    return df.style.format({c: format_number for c in float_cols}) if float_cols else df


def show_table(df: pd.DataFrame, filename: str, height: int | None = None) -> None:
    """Tampilkan tabel beserta tombol unduh CSV."""
    display = styled(df)
    if height is None:
        st.dataframe(display, width="stretch")
    else:
        st.dataframe(display, width="stretch", height=height)
    st.download_button(
        "Unduh tabel (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}_{abs(hash(tuple(df.columns))) % 10**6}",
    )


def interpretation(text: str) -> None:
    st.info(f"**Cara membaca:** {text}")


def method_note(title: str, body: str) -> None:
    with st.expander(f"Tentang {title}"):
        st.markdown(body)


def preprocessing_controls(key_prefix: str = "") -> tuple[str, str]:
    """Kontrol standar penanganan missing dan penskalaan."""
    col1, col2 = st.columns(2)
    with col1:
        missing = st.selectbox(
            "Penanganan nilai hilang",
            preprocessing.MISSING_STRATEGIES,
            key=f"{key_prefix}_missing",
        )
    with col2:
        scaling = st.selectbox(
            "Penskalaan variabel",
            preprocessing.SCALING_METHODS,
            index=1,
            key=f"{key_prefix}_scaling",
        )
    return missing, scaling


def prepare_numeric(
    df: pd.DataFrame, columns: list[str], missing: str, scaling: str
) -> pd.DataFrame:
    subset = preprocessing.handle_missing(df[columns], missing)
    return preprocessing.scale(subset, scaling)
