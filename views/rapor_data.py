"""Rapor Data: pemeriksaan mutu sebelum satu uji pun dijalankan.

Setiap temuan disajikan dalam enam bagian — masalahnya, apa yang terdampak,
akibatnya terhadap analisis, pilihan tindakan, apa yang akan berubah bila tindakan
itu dipilih, dan cara membatalkannya. Aplikasi tidak pernah memperbaiki data sendiri:
memperbaiki diam-diam dapat mengubah kesimpulan tanpa disadari siapa pun.
"""

from __future__ import annotations

import streamlit as st

from nalardata import audit as ad
from nalardata import formatting, ui

RIWAYAT = "rapor_riwayat_data"

ui.butuh_fitur("dasar")
ui.page_setup(
    "Rapor Data",
    "Data",
    "Pemeriksaan mutu menyeluruh sebelum analisis dijalankan. Aplikasi melaporkan "
    "dan menawarkan tindakan; Anda yang memutuskan, dan setiap tindakan dapat "
    "dibatalkan.",
)
df = ui.require_dataset()
ui.sidebar_info()

kamus = ui.kamus()
hasil = ad.jalankan_audit(df, kamus)

# --------------------------------------------------------------------------- #
# Kesimpulan
# --------------------------------------------------------------------------- #

status = hasil.status()
if status == ad.KRITIS:
    st.error(hasil.kesimpulan(), icon=":material/error:")
elif status == ad.PERINGATAN:
    st.warning(hasil.kesimpulan(), icon=":material/warning:")
else:
    st.success(hasil.kesimpulan(), icon=":material/check_circle:")

kolom = st.columns(4)
kolom[0].metric("Baris", formatting.num(hasil.n_baris))
kolom[1].metric("Kolom", formatting.num(hasil.n_kolom))
kolom[2].metric("Masalah kritis", formatting.num(len(hasil.kritis)))
kolom[3].metric("Perlu dicermati", formatting.num(len(hasil.peringatan)))

riwayat = st.session_state.setdefault(RIWAYAT, [])
if riwayat:
    kiri, kanan = st.columns([3, 1])
    kiri.info(
        f"{formatting.num(len(riwayat))} tindakan sudah diterapkan pada data aktif. "
        "Berkas asli Anda tidak tersentuh.",
        icon=":material/history:",
    )
    if kanan.button("Batalkan terakhir", key="rapor_undo", width="stretch"):
        sebelum, catatan = riwayat.pop()
        ui.set_dataset(sebelum, st.session_state.get(ui.NAME_KEY, "data"))
        ui.jejak().catat_perubahan(
            f"Membatalkan: {catatan}", halaman="Rapor Data"
        )
        st.rerun()

if not hasil.temuan:
    ui.keadaan_kosong(
        "Tidak ada masalah yang ditemukan",
        "Seluruh pemeriksaan mutu terlewati. Data siap dianalisis — lanjutkan ke "
        "Pemandu Uji untuk memilih metodenya.",
        ikon="✓",
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Temuan
# --------------------------------------------------------------------------- #

ui.judul_bagian(
    "Temuan",
    "Diurutkan dari yang paling mendesak. Aplikasi tidak mengubah apa pun sebelum "
    "Anda memilih tindakan.",
    kicker="Rapor",
)

IKON = {ad.KRITIS: "🔴", ad.PERINGATAN: "🟠", ad.CATATAN: "🔵"}

for nomor, temuan in enumerate(
    sorted(hasil.temuan, key=lambda t: (ad.URUTAN[t.tingkat], t.perkara))
):
    judul = f"{IKON[temuan.tingkat]}  **{temuan.perkara}** — {temuan.kolom}"
    with st.expander(judul, expanded=temuan.tingkat == ad.KRITIS):
        st.markdown(f"**Apa yang ditemukan.** {temuan.rincian}")
        if temuan.baris:
            st.caption(f"Terdampak: {temuan.ringkas_baris()}")
        st.markdown(f"**Akibatnya pada analisis.** {temuan.dampak}")
        st.markdown(f"**Yang sebaiknya dilakukan.** {temuan.saran}")

        if not temuan.penanganan:
            st.caption(
                "Tidak ada tindakan otomatis untuk temuan ini — perbaikannya menuntut "
                "keputusan yang hanya dapat Anda ambil sendiri."
            )
            continue

        st.markdown("**Pilihan tindakan**")
        for pilihan in temuan.penanganan:
            kiri, kanan = st.columns([3, 1])
            kiri.markdown(f"{pilihan.label} — {pilihan.akibat}")
            kunci = f"rapor_{nomor}_{pilihan.kode}"
            if kanan.button("Terapkan", key=kunci, width="stretch"):
                try:
                    baru, catatan = ad.terapkan(df, temuan, pilihan.kode)
                except ValueError as galat:
                    st.error(str(galat), icon=":material/error:")
                else:
                    riwayat.append((df.copy(), catatan))
                    if pilihan.kode != "tandai":
                        ui.set_dataset(baru, st.session_state.get(ui.NAME_KEY, "data"))
                    ui.jejak().catat_perubahan(catatan, halaman="Rapor Data")
                    st.rerun()

# --------------------------------------------------------------------------- #
# Tabel lengkap
# --------------------------------------------------------------------------- #

ui.judul_bagian("Seluruh temuan dalam satu tabel", kicker="Rapor")
ui.show_table(
    hasil.tabel(),
    "rapor_data.csv",
    bagian="Mutu data",
    judul="Rapor kualitas data",
    catatan="Diurutkan dari temuan yang paling mendesak.",
)

ui.interpretation(
    "Rapor ini memeriksa **bentuk** data. Ia tidak dapat mengetahui apakah "
    "responden menjawab jujur, apakah pertanyaannya dipahami, atau apakah "
    "datanya benar-benar berasal dari populasi yang Anda maksud. Temuan yang "
    "tidak Anda tindaklanjuti sebaiknya tetap disebutkan pada bagian keterbatasan."
)
