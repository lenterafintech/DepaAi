"""Ringkasan eksekutif: kesimpulan tanpa notasi statistik, langsung ke tindakan."""

from __future__ import annotations

import streamlit as st

from lentera_mva import kesimpulan_ui as kui
from lentera_mva import ui

analisis, laporan = kui.buka_ringkasan(
    "Ringkasan Eksekutif",
    "📋",
    "Ditulis untuk pimpinan dan pembaca non-statistik: bahasa sehari-hari, fokus pada "
    "apa artinya dan apa yang sebaiknya dilakukan.",
)

if laporan.pendorong:
    st.subheader("Peringkat pendorong")
    kiri, kanan = st.columns([1.15, 1])
    with kiri:
        kui.batang_pendorong(laporan)
    with kanan:
        if any(p.kinerja is not None for p in laporan.pendorong):
            st.plotly_chart(kui.matriks_prioritas(laporan), width="stretch")
    ui.interpretation(
        "Panjang batang menunjukkan seberapa kuat pengaruh sebuah faktor dibanding faktor "
        "terkuat. Pada matriks di sampingnya, faktor di kuadran kiri atas adalah yang "
        "penting tetapi kinerjanya masih di bawah rata-rata — di sanalah perbaikan paling "
        "terasa hasilnya."
    )

st.subheader("Apa yang ditemukan")
for temuan in laporan.temuan:
    with st.expander(f"**{temuan.judul}** — {temuan.ringkas}"):
        st.write(temuan.eksekutif)

st.subheader("Rekomendasi tindakan")
kui.daftar_bernomor([(r.judul, r.alasan, r.prioritas) for r in laporan.rekomendasi])

st.subheader("Batas kesimpulan")
kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

kui.analisis_yang_dilewati(laporan)
kui.unduhan(laporan, "eksekutif")
