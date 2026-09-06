"""Laporan akademik: pelaporan bergaya jurnal, tabel APA, dan kalimat siap salin."""

from __future__ import annotations

from html import escape

import streamlit as st

from nalardata import kesimpulan_ui as kui
from nalardata import sumber as sm
from nalardata import ui

analisis, laporan, lengkap = kui.buka_ringkasan(
    "Laporan Akademik",
    "Ditulis untuk mahasiswa, dosen, dan pengajar: statistik uji, derajat bebas, nilai "
    "p, ukuran efek, dan catatan asumsi.",
    "ringkasan_akademik",
    "akademik",
)

st.subheader("Ikhtisar hasil")
st.html(
    '<div class="mva-quote"><div class="qh">Abstrak temuan</div><div class="qb">'
    + escape(" ".join(t.ringkas for t in laporan.temuan))
    + "</div></div>"
)

st.subheader("Temuan dan pelaporan statistik")
# Indeks sel dibangun sekali, lalu dipakai seluruh paragraf pada halaman ini.
indeks_sumber = sm.indeks(laporan.tabel)

if lengkap:
    for nomor, temuan in enumerate(laporan.temuan):
        st.markdown(f"**{temuan.judul}**")
        st.caption(f"Metode: {temuan.metode}")
        st.write(temuan.akademik)
        ui.sumber_angka(temuan.akademik, indeks_sumber, f"temuan_{nomor}")
else:
    # Ringkasan cukup memuat inti tiap temuan; uraian lengkapnya ada di laporan lengkap.
    kui.daftar_bernomor([(t.judul, t.ringkas, None) for t in laporan.temuan])

if lengkap and laporan.tabel:
    st.subheader("Tabel hasil")
    for nomor, (judul, tabel, catatan) in laporan.tabel.items():
        st.markdown(f"**{nomor}.** {judul}")
        ui.show_table(tabel, f"{nomor.lower().replace(' ', '_')}_akademik.csv")
        if catatan:
            st.caption(f"*Catatan.* {catatan}")

if lengkap and laporan.paragraf:
    st.subheader("Kalimat siap salin")
    st.caption(
        "Paragraf berikut mengikuti konvensi pelaporan statistik dan dapat langsung "
        "disalin ke naskah. Sesuaikan nama variabel dengan istilah pada penelitian Anda."
    )
    for nomor, paragraf in enumerate(laporan.paragraf):
        st.markdown(f"**{paragraf.bagian}**")
        st.code(paragraf.teks, language=None, wrap_lines=True)
        ui.sumber_angka(paragraf.teks, indeks_sumber, f"paragraf_{nomor}")

st.subheader("Keterbatasan dan saran penelitian lanjutan")
kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

if lengkap:
    st.subheader("Rujukan ambang yang dipakai")
    st.caption(
        "Daftar ini memuat rujukan ambang statistik yang dipakai aplikasi, bukan rujukan "
        "teoretis penelitian Anda. Sesuaikan gaya sitasi dengan pedoman institusi."
    )
    for rujukan in laporan.rujukan:
        st.markdown(f"- {rujukan}")

    kui.analisis_yang_dilewati(laporan)

kui.unduhan(laporan, "akademik", lengkap)
