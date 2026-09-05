"""Halaman beranda: memuat data dan memeriksa strukturnya."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lentera_mva import descriptive, io_utils, ui

ui.page_setup("Beranda & Data", "🧭")

st.markdown(
    """
Aplikasi ini menyediakan alur analisis multivariat lengkap: dari pemeriksaan data,
uji asumsi, sampai pemodelan dan interpretasi hasil. Unggah data Anda, lalu pilih
metode pada menu di sisi kiri.
"""
)

METHODS = [
    ("🔍 Eksplorasi Data", "Deskriptif, distribusi, normalitas (univariat & Mardia), pencilan Mahalanobis"),
    ("🔗 Korelasi & Asumsi", "Pearson/Spearman/Kendall, korelasi parsial, KMO, Bartlett, VIF"),
    ("🧩 PCA", "Reduksi dimensi, scree plot, biplot, analisis paralel Horn"),
    ("🧠 Analisis Faktor", "EFA: principal/PAF/ML dengan rotasi varimax & promax"),
    ("🎯 Analisis Klaster", "K-Means, hierarki + dendrogram, DBSCAN, profil klaster"),
    ("📈 Regresi", "Regresi linear berganda & regresi logistik biner"),
    ("🧭 Analisis Diskriminan", "LDA/QDA, fungsi kanonik, Wilks' lambda, klasifikasi"),
    ("⚖️ MANOVA", "Uji beda vektor rata-rata antar kelompok + Hotelling's T²"),
    ("🔀 Korelasi Kanonik", "Hubungan antara dua gugus variabel sekaligus"),
]

st.subheader("Metode yang tersedia")
cols = st.columns(3)
for i, (name, desc) in enumerate(METHODS):
    with cols[i % 3]:
        st.markdown(f"**{name}**  \n{desc}")

st.divider()
st.subheader("1. Muat data")

tab_upload, tab_sample = st.tabs(["Unggah berkas", "Contoh data"])

with tab_upload:
    uploaded = st.file_uploader(
        "Berkas CSV, TSV, atau Excel", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"]
    )
    if uploaded is not None:
        sheet: str | int = 0
        if uploaded.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
            sheets = io_utils.excel_sheet_names(uploaded)
            sheet = st.selectbox("Pilih sheet", sheets)
        try:
            df = io_utils.load_table(uploaded, filename=uploaded.name, sheet_name=sheet)
        except Exception as exc:  # noqa: BLE001 - pesan gagal baca ditampilkan ke pengguna
            st.error(f"Gagal membaca berkas: {exc}")
        else:
            ui.set_dataset(df, uploaded.name)
            st.success(f"Berhasil memuat {len(df):,} baris dan {df.shape[1]} kolom.")

with tab_sample:
    st.markdown(
        "Contoh data nasabah sintetis (400 baris) berisi variabel demografi, kapasitas "
        "ekonomi, perilaku pembayaran, dan status gagal bayar — cocok untuk mencoba "
        "seluruh metode."
    )
    if st.button("Muat contoh data nasabah", type="primary"):
        ui.set_dataset(ui.load_sample(), "contoh_data_nasabah.csv")
        st.success("Contoh data dimuat.")

df = ui.get_dataset()
ui.sidebar_info()

if df is None:
    st.stop()

st.divider()
st.subheader("2. Periksa data")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baris", f"{len(df):,}".replace(",", "."))
c2.metric("Kolom", df.shape[1])
c3.metric("Kolom numerik", len(df.select_dtypes("number").columns))
c4.metric("Sel kosong", int(df.isna().sum().sum()))

st.markdown("**Pratinjau data**")
st.dataframe(df.head(50), width="stretch")

st.markdown("**Profil variabel**")
ui.show_table(io_utils.profile(df), "profil_variabel.csv")

numeric_df = df.select_dtypes("number")
if not numeric_df.empty:
    st.markdown("**Ringkasan statistik variabel numerik**")
    ui.show_table(descriptive.describe(numeric_df), "statistik_deskriptif.csv")

st.divider()
st.subheader("3. Pilih metode")
st.markdown(
    "Gunakan menu halaman di sisi kiri. Panduan singkat memilih metode:\n\n"
    "- Ingin **meringkas banyak variabel** menjadi sedikit dimensi → PCA atau Analisis Faktor\n"
    "- Ingin **mengelompokkan observasi** yang mirip → Analisis Klaster\n"
    "- Ingin **memprediksi nilai numerik** → Regresi Linear Berganda\n"
    "- Ingin **memprediksi dua kategori** (mis. gagal bayar) → Regresi Logistik\n"
    "- Ingin **memprediksi keanggotaan kelompok** dari banyak prediktor → Analisis Diskriminan\n"
    "- Ingin **membandingkan beberapa variabel dependen** antar kelompok → MANOVA\n"
    "- Ingin **menghubungkan dua gugus variabel** sekaligus → Korelasi Kanonik"
)

with st.expander("Catatan mutu data"):
    dup = int(df.duplicated().sum())
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.2]
    notes = []
    if dup:
        notes.append(f"Terdapat {dup} baris duplikat.")
    if const_cols:
        notes.append(f"Kolom tanpa variasi: {', '.join(const_cols)}.")
    if high_missing:
        notes.append(f"Kolom dengan missing > 20%: {', '.join(high_missing)}.")
    st.write("\n\n".join(f"- {n}" for n in notes) if notes else "Tidak ada masalah menonjol.")

if isinstance(df, pd.DataFrame) and df.empty:
    st.error("Data yang dimuat kosong.")
