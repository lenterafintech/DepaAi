"""Halaman beranda: memuat data dan memeriksa strukturnya."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import descriptive, formatting, io_utils, ui
from nalardata import proyek as pr

ui.page_setup(
    "Beranda & Data",
    "Data",
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

tab_upload, tab_sample, tab_proyek = st.tabs(
    ["Unggah berkas", "Contoh data", "Buka proyek"]
)

with tab_upload:
    uploaded = st.file_uploader(
        "Berkas CSV, TSV, Excel, atau SPSS",
        type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "sav", "zsav"],
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
        ui.set_dataset(ui.load_sample(), "contoh_data_nasabah.csv", contoh=True)
        st.success("Contoh data dimuat.")

with tab_proyek:
    st.markdown(
        "Berkas proyek `.nalardata` memuat data, kamus variabel, hasil yang Anda "
        "simpan di **Laporan Hasil**, dan pengaturan cakupan analisis sekaligus — "
        "sehingga pekerjaan dapat dilanjutkan pada sesi berikutnya."
    )
    berkas_proyek = st.file_uploader(
        "Berkas proyek", type=["nalardata", "zip"], key="unggah_proyek"
    )
    if berkas_proyek is not None:
        try:
            proyek_dibuka = pr.buka_proyek(berkas_proyek.getvalue())
        except ValueError as galat:
            st.error(str(galat), icon=":material/error:")
        else:
            ui.show_table(proyek_dibuka.ringkas(), "isi_proyek.csv")
            if st.button("Muat proyek ini", type="primary", key="muat_proyek"):
                ui.set_dataset(proyek_dibuka.data, proyek_dibuka.nama_data)
                st.session_state[ui.KERANJANG_KEY] = proyek_dibuka.keranjang
                if len(proyek_dibuka.kamus):
                    ui.set_kamus(proyek_dibuka.kamus)
                if not proyek_dibuka.penelitian.kosong():
                    ui.set_penelitian(proyek_dibuka.penelitian)
                if not proyek_dibuka.jejak.kosong():
                    ui.set_jejak(proyek_dibuka.jejak)
                for kunci, nilai in (proyek_dibuka.konfigurasi or {}).items():
                    # Kunci widget halaman laporan dipulihkan apa adanya.
                    st.session_state[kunci] = nilai
                st.success("Proyek dimuat. Data dan hasil tersimpan sudah pulih.")
                st.rerun()

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

st.divider()
st.subheader("Simpan proyek")
st.caption(
    "Menyimpan data aktif, hasil yang sudah masuk **Laporan Hasil**, dan pengaturan "
    "cakupan analisis ke dalam satu berkas. Aplikasi ini tidak menyimpan apa pun di "
    "server, jadi berkas inilah satu-satunya cara melanjutkan pekerjaan nanti."
)

isi_keranjang = ui.keranjang()
kunci_konfig = [k for k in st.session_state if str(k).startswith("kesimpulan_")]
konfigurasi = {k: st.session_state[k] for k in kunci_konfig}

try:
    berkas = pr.simpan_proyek(
        df,
        st.session_state.get(ui.NAME_KEY, "data"),
        isi_keranjang,
        konfigurasi,
        kamus=ui.kamus(),
        penelitian=ui.penelitian(),
        jejak=ui.jejak(),
    )
except ValueError as galat:
    st.info(str(galat))
else:
    kiri, kanan = st.columns([1, 2])
    kiri.download_button(
        "Unduh berkas proyek",
        berkas,
        file_name=pr.nama_berkas_proyek(st.session_state.get(ui.NAME_KEY, "data")),
        mime="application/zip",
        type="primary",
        width="stretch",
        key="unduh_proyek",
    )
    kanan.caption(
        f"Berisi {len(df):,} baris data".replace(",", ".")
        + f" · {len(isi_keranjang.item)} hasil tersimpan"
        + f" · kamus {len(ui.kamus())} variabel"
        + (f" · {len(konfigurasi)} pengaturan" if konfigurasi else "")
    )
