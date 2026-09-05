"""Ringkasan akademik: pelaporan bergaya jurnal, tabel APA, dan kalimat siap salin."""

from __future__ import annotations

from html import escape

import streamlit as st

from lentera_mva import kesimpulan_ui as kui

analisis, laporan = kui.buka_ringkasan(
    "Ringkasan Akademik",
    "Ditulis untuk mahasiswa dan dosen: statistik uji, derajat bebas, nilai p, ukuran "
    "efek, dan catatan asumsi — lengkap dengan tabel serta paragraf siap salin.",
)

st.subheader("Ikhtisar hasil")
st.html(
    '<div class="mva-quote"><div class="qh">Abstrak temuan</div><div class="qb">'
    + escape(" ".join(t.ringkas for t in laporan.temuan))
    + "</div></div>"
)

st.subheader("Temuan dan pelaporan statistik")
for temuan in laporan.temuan:
    st.markdown(f"**{temuan.judul}**")
    st.caption(f"Metode: {temuan.metode}")
    st.write(temuan.akademik)

if laporan.tabel:
    st.subheader("Tabel hasil")
    for nomor, (judul, tabel, catatan) in laporan.tabel.items():
        st.markdown(f"**{nomor}.** {judul}")
        st.dataframe(tabel, width="stretch", hide_index=True)
        if catatan:
            st.caption(f"*Catatan.* {catatan}")

st.subheader("Kalimat siap salin")
st.caption(
    "Paragraf berikut mengikuti konvensi pelaporan statistik dan dapat langsung disalin "
    "ke naskah. Sesuaikan nama variabel dengan istilah pada penelitian Anda."
)
for paragraf in laporan.paragraf:
    st.markdown(f"**{paragraf.bagian}**")
    st.code(paragraf.teks, language=None, wrap_lines=True)

st.subheader("Keterbatasan dan saran penelitian lanjutan")
kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

st.subheader("Rujukan ambang yang dipakai")
st.caption(
    "Daftar ini memuat rujukan ambang statistik yang dipakai aplikasi, bukan rujukan "
    "teoretis penelitian Anda. Sesuaikan gaya sitasi dengan pedoman institusi."
)
for rujukan in laporan.rujukan:
    st.markdown(f"- {rujukan}")

kui.analisis_yang_dilewati(laporan)
kui.unduhan(laporan, "akademik")
