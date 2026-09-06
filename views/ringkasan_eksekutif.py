"""Laporan untuk pembaca umum: kesimpulan tanpa notasi statistik, langsung ke tindakan."""

from __future__ import annotations

import streamlit as st

from nalardata import kesimpulan_ui as kui
from nalardata import ui

analisis, laporan, lengkap = kui.buka_ringkasan(
    "Laporan Umum",
    "Ditulis untuk pimpinan dan pembaca non-statistik: bahasa sehari-hari, fokus pada "
    "apa artinya dan apa yang sebaiknya dilakukan.",
    "ringkasan_eksekutif",
    "eksekutif",
)

if laporan.pendorong:
    st.subheader("Peringkat pendorong")
    kiri, kanan = st.columns([1.15, 1])
    with kiri:
        kui.batang_pendorong(laporan)
    with kanan:
        if any(p.kinerja is not None for p in laporan.pendorong):
            st.plotly_chart(kui.matriks_prioritas(laporan), width="stretch")
            st.caption(
                "Hanya faktor terkuat yang diberi nama pada grafik; arahkan kursor ke "
                "titik lain untuk melihat namanya."
            )
    ui.interpretation(
        "Panjang batang menunjukkan seberapa kuat pengaruh sebuah faktor dibanding faktor "
        "terkuat. Pada matriks di sampingnya, faktor di kuadran kiri atas adalah yang "
        "penting tetapi kinerjanya masih di bawah rata-rata — di sanalah perbaikan paling "
        "terasa hasilnya."
    )

st.subheader("Apa yang ditemukan")
if lengkap:
    # Laporan lengkap menguraikan tiap temuan apa adanya, tanpa perlu diklik.
    for temuan in laporan.temuan:
        st.markdown(f"**{temuan.judul}**")
        st.caption(f"Metode: {temuan.metode}")
        st.write(temuan.eksekutif)
else:
    for temuan in laporan.temuan:
        with st.expander(f"**{temuan.judul}** — {temuan.ringkas}"):
            st.write(temuan.eksekutif)

st.subheader("Rekomendasi tindakan")
kui.daftar_bernomor([(r.judul, r.alasan, r.prioritas) for r in laporan.rekomendasi])

st.subheader("Batas kesimpulan")
kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

if lengkap and laporan.tabel:
    st.subheader("Tabel hasil")
    st.caption(
        "Angka rinci di balik kesimpulan di atas. Bagian ini boleh dilewati bila Anda "
        "hanya memerlukan kesimpulannya."
    )
    for nomor, (judul_tabel, tabel, catatan) in laporan.tabel.items():
        st.markdown(f"**{nomor}.** {judul_tabel}")
        ui.show_table(tabel, f"{nomor.lower().replace(' ', '_')}_umum.csv")
        if catatan:
            st.caption(f"*Catatan.* {catatan}")

if lengkap:
    kui.analisis_yang_dilewati(laporan)

kui.unduhan(laporan, "eksekutif", lengkap)
