"""Kesesuaian Hasil: membuktikan angkanya, bukan mengklaimnya.

Pengguna Indonesia tidak akan memercayai perangkat baru sebelum angkanya cocok
dengan SPSS atau R. Halaman ini menunjukkan perbandingannya secara terbuka — dan
menyebut apa adanya metode yang belum punya acuan.
"""

from __future__ import annotations

import streamlit as st

from nalardata import formatting, ui, validasi as vl

ui.butuh_fitur("dasar", "Kesesuaian Hasil")
ui.page_setup(
    "Kesesuaian Hasil",
    "Mutu",
    "Perbandingan hasil NalarData dengan keluaran R pada dataset acuan yang sama. "
    "Termasuk daftar metode yang belum divalidasi.",
)
ui.sidebar_info()

st.info(
    "Yang dibandingkan adalah **fungsi aplikasi ini sendiri**, bukan pustaka di "
    "baliknya. Membandingkan scipy dengan R hanya menguji scipy; yang perlu diuji "
    "adalah lapisan yang ditulis di sini, karena di lapisan itulah kekeliruan "
    "penulisan kode terjadi.",
    icon=":material/info:",
)

with st.spinner("Menjalankan ulang seluruh pembandingan…"):
    rinci = vl.jalankan()
    ringkas = vl.ringkas(rinci)
    cakupan = vl.cakupan()

kolom = st.columns(3)
kolom[0].metric("Metode divalidasi", formatting.num(cakupan["Metode divalidasi"]))
kolom[1].metric("Belum divalidasi", formatting.num(cakupan["Belum divalidasi"]))
kolom[2].metric("Besaran diperiksa", formatting.num(cakupan["Besaran diperiksa"]))

berbeda = int((rinci["Status"] == vl.BERBEDA).sum())
gagal = int((rinci["Status"] == vl.GAGAL).sum())
if gagal:
    st.error(
        f"{formatting.num(gagal)} pembandingan tidak dapat dijalankan.",
        icon=":material/error:",
    )
elif berbeda:
    st.warning(
        f"{formatting.num(berbeda)} besaran berbeda dari acuannya. Rincian dan "
        "sebabnya ada pada tabel di bawah.",
        icon=":material/warning:",
    )
else:
    st.success(
        "Seluruh besaran yang diperiksa cocok dengan acuannya. Selisih yang ada "
        "berasal dari pembulatan pada angka acuan yang diterbitkan.",
        icon=":material/check_circle:",
    )

st.caption(
    f"Cakupannya {cakupan['Metode divalidasi']} dari "
    f"{cakupan['Metode divalidasi'] + cakupan['Belum divalidasi']} metode. Sisanya "
    "belum punya acuan, dan itu disebut apa adanya di tabel — daftar yang "
    "menyembunyikan lubangnya sendiri tidak dapat dipercaya untuk hal lain mana pun."
)

ui.judul_bagian("Status per metode", kicker="Kesesuaian")
ui.show_table(
    ringkas,
    "kesesuaian_metode.csv",
    bagian="Mutu hasil",
    judul="Kesesuaian hasil NalarData dengan perangkat pembanding",
    catatan="Metode berstatus 'Belum divalidasi' belum punya acuan terbit yang datanya dapat disertakan.",
)

ui.judul_bagian(
    "Rincian per besaran",
    "Angka yang dibandingkan satu per satu, beserta selisihnya.",
    kicker="Kesesuaian",
)
tampil = rinci.copy()
for kolom_angka in ("NalarData", "Acuan", "Selisih"):
    tampil[kolom_angka] = [formatting.num_auto(v) for v in tampil[kolom_angka]]
ui.show_table(tampil, "kesesuaian_rincian.csv")

ui.judul_bagian("Yang belum divalidasi", kicker="Kesesuaian")
st.caption("Beserta sebab mengapa acuannya belum tersedia.")
for metode, alasan in vl.BELUM_DIVALIDASI.items():
    st.markdown(f"- **{metode}** — {alasan}")

ui.interpretation(
    "Acuan di sini **beku**: nilainya diambil dari keluaran R yang terdokumentasi "
    "pada dataset klasik yang ikut disertakan, bukan dari R yang dijalankan langsung. "
    "Menjalankan R secara langsung akan lebih baik, tetapi menuntut R terpasang di "
    "setiap tempat aplikasi ini dijalankan — dan acuan beku yang jujur lebih berguna "
    "daripada acuan hidup yang tidak pernah tersedia."
)
