"""Simulasi Sidang: berlatih mempertahankan analisis Anda sendiri.

Pertanyaan di sini tidak dapat disusun tanpa melihat hasil Anda — ia menanyakan R²
Anda, kelompok Anda yang gagal uji normalitas, dan uji Anda yang di luar rencana.
Lencana kesiapan hanya diberikan setelah Anda menjawab sendiri; aplikasi boleh
menyusun kalimat, ia tidak boleh menggantikan pemahaman.
"""

from __future__ import annotations

import streamlit as st

from nalardata import audit as ad
from nalardata import formatting, kamus as km, kesimpulan_ui as kui, sidang as sd, ui

JAWABAN = "sidang_jawaban"

ui.butuh_fitur("sidang", "Simulasi Sidang")
ui.page_setup(
    "Simulasi Sidang",
    "Laporan",
    "Pertanyaan penguji yang disusun dari analisis Anda sendiri, beserta jawaban "
    "model yang diturunkan dari angka Anda — bukan dari contoh buku.",
)
df = ui.require_dataset()
ui.sidebar_info()
kui.pasang_gaya()

analisis, laporan = kui.siapkan_laporan(df)
penelitian = ui.penelitian()
jejak = ui.jejak()
hasil_audit = ad.jalankan_audit(df, ui.kamus())

simulasi = sd.susun(laporan, penelitian, jejak, hasil_audit)
jawaban = st.session_state.setdefault(JAWABAN, {})

if simulasi.kosong():
    ui.keadaan_kosong(
        "Belum ada yang dapat ditanyakan",
        "Jalankan analisis lebih dulu pada halaman Laporan Akademik, lalu kembali "
        "ke sini.",
        ikon="?",
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Kesiapan
# --------------------------------------------------------------------------- #

lulus, total, siap = sd.kesiapan(simulasi, jawaban)

kolom = st.columns(3)
kolom[0].metric("Pertanyaan", formatting.num(len(simulasi.pertanyaan)))
kolom[1].metric("Wajib dijawab", formatting.num(total))
kolom[2].metric("Sudah Anda jawab", formatting.num(lulus))

if siap:
    st.success(
        "**Siap Sidang.** Seluruh pertanyaan wajib sudah Anda jawab sendiri. "
        "Perlu diingat, penguji sungguhan dapat menanyakan apa saja — termasuk hal "
        "yang tidak terbaca dari data. Lulus simulasi ini bukan jaminan lulus sidang.",
        icon=":material/verified:",
    )
else:
    st.info(
        f"Lencana **Siap Sidang** diberikan setelah {formatting.num(total)} pertanyaan "
        "wajib Anda jawab sendiri. Membaca jawaban model tidak menggantikannya — yang "
        "dinilai penguji adalah kemampuan Anda menjelaskan, bukan isi berkasnya.",
        icon=":material/school:",
    )

st.caption(
    "Pemeriksaan jawaban di bawah bersifat kasar: aplikasi hanya melihat apakah "
    "gagasan kuncinya Anda sebut, bukan apakah kalimatnya benar. Menilai pemahaman "
    "bukan wewenang aplikasi."
)

# --------------------------------------------------------------------------- #
# Pertanyaan
# --------------------------------------------------------------------------- #

for nomor, butir in enumerate(simulasi.pertanyaan):
    cukup, terlewat = butir.nilai(jawaban.get(nomor, ""))
    tanda = "✓" if cukup else ("●" if butir.bobot == sd.WAJIB else "○")
    label = f"{tanda}  **{nomor + 1}. {butir.pertanyaan}**"

    with st.expander(label, expanded=not cukup and nomor == lulus):
        st.caption(
            f"{sd.LABEL_KATEGORI[butir.kategori]}"
            + (" · wajib dijawab" if butir.bobot == sd.WAJIB else " · mungkin ditanyakan")
        )

        isi = st.text_area(
            "Jawaban Anda",
            value=jawaban.get(nomor, ""),
            key=f"sidang_jawab_{nomor}",
            height=110,
            placeholder="Tuliskan dengan kalimat Anda sendiri, seperti saat menjawab penguji.",
        )
        if isi != jawaban.get(nomor, ""):
            jawaban[nomor] = isi
            st.rerun()

        if isi.strip():
            if cukup:
                st.success("Gagasan kuncinya sudah Anda sebut.", icon=":material/check:")
            else:
                st.warning(
                    "Belum menyinggung: " + ", ".join(f"**{k}**" for k in terlewat),
                    icon=":material/lightbulb:",
                )

        with st.popover("Lihat jawaban model", width="stretch"):
            st.write(butir.jawaban)

# --------------------------------------------------------------------------- #
# Daftar pertanyaan
# --------------------------------------------------------------------------- #

st.divider()
ui.judul_bagian(
    "Seluruh pertanyaan",
    "Dapat diunduh untuk dilatih di luar aplikasi.",
    kicker="Sidang",
)
ui.show_table(
    simulasi.ringkas(),
    "pertanyaan_sidang.csv",
    bagian="Persiapan sidang",
    judul="Pertanyaan penguji yang mungkin muncul",
    catatan="Disusun dari rancangan, asumsi, dan hasil analisis ini sendiri.",
)

if st.button("Kosongkan jawaban", key="sidang_kosongkan"):
    st.session_state[JAWABAN] = {}
    st.rerun()
