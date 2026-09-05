"""Lentera MVA - aplikasi web analisis multivariat.

Jalankan dengan: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Lentera MVA", page_icon="📊", layout="wide")
st.session_state["_page_configured"] = True

PAGES = {
    "Data": [
        st.Page("views/beranda.py", title="Beranda & Data", icon="🧭", default=True),
        st.Page("views/eksplorasi.py", title="Eksplorasi Data", icon="🔍"),
        st.Page("views/korelasi.py", title="Korelasi & Asumsi", icon="🔗"),
    ],
    "Reduksi Dimensi": [
        st.Page("views/pca.py", title="PCA", icon="🧩"),
        st.Page("views/analisis_faktor.py", title="Analisis Faktor", icon="🧠"),
    ],
    "Pengelompokan": [
        st.Page("views/klaster.py", title="Analisis Klaster", icon="🎯"),
    ],
    "Pemodelan & Uji Beda": [
        st.Page("views/regresi.py", title="Regresi", icon="📈"),
        st.Page("views/diskriminan.py", title="Analisis Diskriminan", icon="🧭"),
        st.Page("views/manova.py", title="MANOVA", icon="⚖️"),
        st.Page("views/korelasi_kanonik.py", title="Korelasi Kanonik", icon="🔀"),
    ],
}

st.navigation(PAGES).run()
