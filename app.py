"""NalarData - pendamping analisis dan pelaporan penelitian.

Jalankan dengan: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="NalarData", page_icon="📊", layout="wide")
st.session_state["_page_configured"] = True

# Menu disusun mengikuti tahapan penelitian, bukan daftar metode statistik.
# Pengguna yang belum menguasai statistik tahu sampai di mana penelitiannya,
# tetapi belum tentu tahu nama uji yang dicarinya - menyodorkan daftar metode
# sebagai pintu masuk justru meminta ia menjawab pertanyaan yang ia datangi
# aplikasi ini untuk menanyakannya.
PAGES = {
    "1 · Rencana": [
        st.Page("views/proyek.py", title="Ruang Proyek", icon=":material/flag:"),
    ],
    "2 · Data": [
        st.Page(
            "views/beranda.py",
            title="Beranda & Data",
            icon=":material/database:",
            default=True,
        ),
        st.Page("views/entri_data.py", title="Buat & Entri Data", icon=":material/edit_note:"),
        st.Page("views/kamus.py", title="Kamus Variabel", icon=":material/menu_book:"),
    ],
    "3 · Mutu Data": [
        st.Page("views/rapor_data.py", title="Rapor Data", icon=":material/fact_check:"),
        st.Page("views/eksplorasi.py", title="Eksplorasi Data", icon=":material/search_insights:"),
    ],
    "4 · Instrumen": [
        st.Page("views/reliabilitas.py", title="Reliabilitas & Validitas", icon=":material/rule:"),
    ],
    "5 · Pilih Metode": [
        st.Page("views/pemandu.py", title="Pemandu Uji", icon=":material/explore:"),
    ],
    "6 · Analisis — Beda & Hubungan": [
        st.Page("views/korelasi.py", title="Korelasi & Asumsi", icon=":material/linked_services:"),
        st.Page("views/nonparametrik.py", title="Uji Beda", icon=":material/stacked_line_chart:"),
        st.Page("views/manova.py", title="MANOVA", icon=":material/balance:"),
    ],
    "6 · Analisis — Pemodelan": [
        st.Page("views/regresi.py", title="Regresi", icon=":material/trending_up:"),
        st.Page("views/moderasi.py", title="Regresi Moderasi (MRA)", icon=":material/alt_route:"),
        st.Page("views/diskriminan.py", title="Analisis Diskriminan", icon=":material/rule_folder:"),
        st.Page("views/sem.py", title="CFA, Jalur & SEM", icon=":material/account_tree:"),
    ],
    "6 · Analisis — Reduksi & Kelompok": [
        st.Page("views/pca.py", title="PCA", icon=":material/compress:"),
        st.Page("views/analisis_faktor.py", title="Analisis Faktor", icon=":material/hub:"),
        st.Page("views/klaster.py", title="Analisis Klaster", icon=":material/scatter_plot:"),
        st.Page(
            "views/korelasi_kanonik.py",
            title="Korelasi Kanonik",
            icon=":material/compare_arrows:",
        ),
    ],
    "7 · Laporan": [
        st.Page("views/ringkasan_eksekutif.py", title="Laporan Umum", icon=":material/summarize:"),
        st.Page("views/ringkasan_akademik.py", title="Laporan Akademik", icon=":material/school:"),
        st.Page(
            "views/ringkasan_profesional.py",
            title="Laporan Profesional",
            icon=":material/engineering:",
        ),
        st.Page("views/laporan.py", title="Laporan Hasil", icon=":material/inventory_2:"),
        st.Page("views/sidang.py", title="Simulasi Sidang", icon=":material/quiz:"),
    ],
    "Akun": [
        st.Page("views/masuk.py", title="Masuk / Daftar", icon=":material/login:"),
        st.Page("views/akun.py", title="Akun & Langganan", icon=":material/workspace_premium:"),
    ],
}

# expanded=True agar seluruh halaman terlihat; tanpa ini Streamlit
# menyembunyikan halaman ke-11 dan seterusnya di balik "View more".
st.navigation(PAGES, expanded=True).run()
