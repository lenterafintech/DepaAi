"""Lentera MVA - aplikasi web analisis multivariat.

Jalankan dengan: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Lentera MVA", page_icon="📊", layout="wide")
st.session_state["_page_configured"] = True

PAGES = {
    "Data": [
        st.Page("views/beranda.py", title="Beranda & Data", icon=":material/database:", default=True),
        st.Page("views/entri_data.py", title="Buat & Entri Data", icon=":material/edit_note:"),
        st.Page("views/eksplorasi.py", title="Eksplorasi Data", icon=":material/search_insights:"),
        st.Page("views/korelasi.py", title="Korelasi & Asumsi", icon=":material/linked_services:"),
    ],
    "Instrumen": [
        st.Page("views/reliabilitas.py", title="Reliabilitas & Validitas", icon=":material/fact_check:"),
    ],
    "Reduksi Dimensi": [
        st.Page("views/pca.py", title="PCA", icon=":material/compress:"),
        st.Page("views/analisis_faktor.py", title="Analisis Faktor", icon=":material/hub:"),
    ],
    "Pengelompokan": [
        st.Page("views/klaster.py", title="Analisis Klaster", icon=":material/scatter_plot:"),
    ],
    "Pemodelan & Uji Beda": [
        st.Page("views/regresi.py", title="Regresi", icon=":material/trending_up:"),
        st.Page("views/diskriminan.py", title="Analisis Diskriminan", icon=":material/rule:"),
        st.Page("views/manova.py", title="MANOVA", icon=":material/balance:"),
        st.Page("views/korelasi_kanonik.py", title="Korelasi Kanonik", icon=":material/compare_arrows:"),
    ],
    "Akun": [
        st.Page("views/akun.py", title="Akun & Langganan", icon=":material/workspace_premium:"),
    ],
    "Ringkasan Kesimpulan": [
        st.Page("views/ringkasan_eksekutif.py", title="Ringkasan Eksekutif", icon=":material/summarize:"),
        st.Page("views/ringkasan_akademik.py", title="Ringkasan Akademik", icon=":material/school:"),
        st.Page("views/ringkasan_profesional.py", title="Ringkasan Profesional", icon=":material/engineering:"),
    ],
}

# expanded=True agar seluruh halaman terlihat; tanpa ini Streamlit
# menyembunyikan halaman ke-11 dan seterusnya di balik "View more".
st.navigation(PAGES, expanded=True).run()
