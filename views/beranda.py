"""Halaman beranda: memuat data dan memeriksa strukturnya."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lentera_mva import descriptive, formatting, io_utils, ui

ui.page_setup(
    "Beranda & Data",
    "Lentera MVA",
    "Muat data Anda di sini, lalu pilih metode analisis atau langsung baca "
    "ringkasan kesimpulannya pada menu di sisi kiri.",
)

METODE = [
    ("Eksplorasi Data", "Deskriptif, distribusi, normalitas (univariat & Mardia), pencilan Mahalanobis"),
    ("Korelasi & Asumsi", "Pearson/Spearman/Kendall, korelasi parsial, KMO, Bartlett, VIF"),
    ("PCA", "Reduksi dimensi, scree plot, biplot, analisis paralel Horn"),
    ("Analisis Faktor", "EFA: principal/PAF/ML dengan rotasi varimax & promax"),
    ("Analisis Klaster", "K-Means, hierarki + dendrogram, DBSCAN, profil klaster"),
    ("Regresi", "Regresi linear berganda & regresi logistik biner"),
    ("Analisis Diskriminan", "LDA/QDA, fungsi kanonik, Wilks' lambda, klasifikasi"),
    ("MANOVA", "Uji beda vektor rata-rata antar kelompok + Hotelling's T²"),
    ("Korelasi Kanonik", "Hubungan antara dua gugus variabel sekaligus"),
]

RINGKASAN = [
    ("Ringkasan Eksekutif", "Untuk pimpinan dan pembaca non-statistik",
     "Kesimpulan utama, pendorong terkuat, dan rekomendasi tindakan — tanpa notasi statistik."),
    ("Ringkasan Akademik", "Untuk mahasiswa dan dosen",
     "Pelaporan bergaya jurnal, tabel APA, paragraf siap salin, dan rujukan ambang."),
    ("Ringkasan Profesional", "Untuk analis dan praktisi",
     "Metrik model, pemeriksaan asumsi, tindak lanjut teknis, dan risiko pemakaian."),
]

st.html(
    f"""
<style>
.mva-kartu {{display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px; margin: .4rem 0 1rem}}
.mva-kartu .k {{border: 1px solid {ui.WARNA['garis']}; border-radius: 10px; padding: 14px 16px;
  background: #fff}}
.mva-kartu .j {{font-weight: 650; font-size: .95rem; color: {ui.WARNA['tinta']}}}
.mva-kartu .u {{font-size: .7rem; letter-spacing: .08em; text-transform: uppercase;
  color: {ui.WARNA['aksen2']}; font-weight: 700; margin: .3rem 0 .45rem}}
.mva-kartu .d {{font-size: .84rem; line-height: 1.55; color: {ui.WARNA['tinta2']}}}
.mva-kartu.ringkas .k {{padding: 11px 13px}}
.mva-kartu.ringkas .j {{font-size: .88rem}}
.mva-kartu.ringkas .d {{font-size: .8rem}}
</style>
"""
)

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

# Angka ukuran data disampaikan sebagai satu baris keterangan, bukan empat kartu
# raksasa — ia konteks, bukan temuan utama halaman ini.
st.caption(
    f"{formatting.num(len(df))} baris · {df.shape[1]} kolom "
    f"({len(df.select_dtypes('number').columns)} numerik) · "
    f"{formatting.num(int(df.isna().sum().sum()))} sel kosong"
)

st.markdown("**Pratinjau data**")
st.dataframe(df.head(50), width="stretch", hide_index=True)

st.markdown("**Profil variabel**")
ui.show_table(io_utils.profile(df), "profil_variabel.csv")

numeric_df = df.select_dtypes("number")
if not numeric_df.empty:
    st.markdown("**Ringkasan statistik variabel numerik**")
    ui.show_table(descriptive.describe(numeric_df), "statistik_deskriptif.csv")

st.divider()
st.subheader("3. Baca ringkasan kesimpulan")
st.caption(
    "Ketiga halaman ini menjalankan seluruh metode sekaligus pada data di atas, lalu "
    "menuliskan kesimpulan yang sama untuk pembaca yang berbeda. Pilih sesuai kepada "
    "siapa hasilnya akan disampaikan."
)
st.html(
    '<div class="mva-kartu">'
    + "".join(
        f'<div class="k"><div class="j">{judul}</div>'
        f'<div class="u">{untuk}</div><div class="d">{isi}</div></div>'
        for judul, untuk, isi in RINGKASAN
    )
    + "</div>"
)

st.subheader("4. Atau pilih metode tertentu")
st.caption("Panduan singkat memilih metode, sesuai pertanyaan yang ingin dijawab:")
st.markdown(
    "- Meringkas **banyak variabel** menjadi sedikit dimensi → PCA atau Analisis Faktor\n"
    "- Mengelompokkan **observasi yang mirip** → Analisis Klaster\n"
    "- Memprediksi **nilai numerik** → Regresi Linear Berganda\n"
    "- Memprediksi **dua kategori** (mis. gagal bayar) → Regresi Logistik\n"
    "- Memprediksi **keanggotaan kelompok** dari banyak prediktor → Analisis Diskriminan\n"
    "- Membandingkan **beberapa variabel dependen** antar kelompok → MANOVA\n"
    "- Menghubungkan **dua gugus variabel** sekaligus → Korelasi Kanonik"
)
with st.expander("Daftar lengkap metode yang tersedia"):
    st.html(
        '<div class="mva-kartu ringkas">'
        + "".join(
            f'<div class="k"><div class="j">{judul}</div><div class="d">{isi}</div></div>'
            for judul, isi in METODE
        )
        + "</div>"
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
