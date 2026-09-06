"""Laporan Hasil: mengumpulkan analisis yang benar-benar dijalankan pengguna."""

from __future__ import annotations

import streamlit as st

from mv_statlab import ekspor, formatting, ui

ui.butuh_fitur("keranjang")
ui.page_setup(
    "Laporan Hasil",
    "Laporan Analisis",
    "Berisi hasil yang Anda simpan sendiri dari halaman-halaman metode. Berbeda dari "
    "Laporan Umum, Akademik, dan Profesional yang disusun otomatis, halaman ini memuat "
    "persis analisis yang Anda jalankan.",
)
ui.sidebar_info()

isi = ui.keranjang()

if isi.kosong():
    ui.keadaan_kosong(
        "Belum ada hasil yang disimpan",
        "Jalankan analisis di halaman mana pun — Regresi, CFA, Uji Non-parametrik, "
        "dan lainnya — lalu tekan tombol Simpan ke laporan di bawah tabel hasilnya.",
        ikon="＋",
    )
    st.stop()

ringkas = isi.ringkas()
kolom = st.columns(4)
for slot, (label, nilai) in zip(kolom, ringkas.items()):
    slot.metric(label, formatting.num(nilai))

basi = isi.basi(ui.tanda_data())
if basi:
    st.warning(
        f"{formatting.num(len(basi))} hasil disimpan dari data yang berbeda dengan yang "
        "sedang aktif. Angkanya tetap utuh seperti saat disimpan, namun periksa lagi "
        "sebelum dilaporkan bersama hasil yang baru.",
        icon=":material/history:",
    )

# --------------------------------------------------------------------------- #
# Identitas laporan
# --------------------------------------------------------------------------- #

with st.expander("Judul dan penyusun", expanded=False):
    kiri, kanan = st.columns(2)
    isi.judul = kiri.text_input("Judul laporan", value=isi.judul, key="laporan_judul")
    isi.peneliti = kanan.text_input(
        "Nama penyusun",
        value=isi.peneliti,
        key="laporan_peneliti",
        placeholder="opsional — dicantumkan pada halaman muka",
    )

ui.judul_bagian("Daftar isi", kicker="Isi laporan")
ui.show_table(isi.daftar_isi(), "daftar_isi_laporan.csv")

# --------------------------------------------------------------------------- #
# Isi per bagian
# --------------------------------------------------------------------------- #

ui.judul_bagian(
    "Rincian per bagian",
    "Setiap hasil dapat dihapus sendiri bila tidak jadi dipakai.",
    kicker="Isi laporan",
)
tanda_sekarang = ui.tanda_data()

for nama, butir in isi.per_bagian().items():
    with st.expander(f"**{nama}** — {len(butir)} objek", expanded=False):
        for item in butir:
            atas, tombol = st.columns([5, 1])
            with atas:
                label = item.judul
                if item.basi(tanda_sekarang):
                    label += "  ·  ⚠︎ data berbeda"
                st.markdown(f"**{label}**")
                st.caption(f"Disimpan {item.waktu}")
            if tombol.button("Hapus", key=f"hapus_{item.sidik}"):
                isi.hapus(item.sidik)
                st.rerun()

            if item.jenis == "tabel" and item.tabel is not None:
                st.dataframe(ui.styled(item.tabel), width="stretch", hide_index=True)
                if item.catatan:
                    st.caption(f"*Catatan.* {item.catatan}")
            else:
                st.write(item.teks)
            st.divider()

        if st.button(f"Hapus seluruh bagian '{nama}'", key=f"hapus_bagian_{nama}"):
            isi.hapus_bagian(nama)
            st.rerun()

# --------------------------------------------------------------------------- #
# Ekspor
# --------------------------------------------------------------------------- #

ui.judul_bagian(
    "Ekspor laporan",
    "Seluruh format disusun dari isi yang sama, sehingga angkanya identik.",
    kicker="Unduh",
)

if not ui.paket_aktif().punya("unduh_laporan"):
    st.info(
        "Ekspor laporan tersedia mulai paket Mahasiswa & Pengajar. Isi di atas tetap "
        "dapat dibaca dan disalin.",
        icon=":material/lock:",
    )
else:
    st.caption(
        "Seluruh format disusun dari isi yang sama, sehingga Word, PDF, Excel, "
        "PowerPoint, HTML, dan Markdown memuat angka yang identik."
    )
    kiri, kanan = st.columns(2)
    kode = kiri.selectbox(
        "Format berkas",
        list(ekspor.FORMAT),
        format_func=lambda k: ekspor.FORMAT[k].nama,
        key="laporan_format",
    )
    kiri.caption(ekspor.FORMAT[kode].keterangan)

    if kode in {"py", "r"}:
        kanan.info(
            "Sintaks Python dan R dibangkitkan dari konfigurasi analisis pada halaman "
            "Laporan Umum, Akademik, atau Profesional — bukan dari keranjang ini.",
            icon=":material/info:",
        )
    else:
        try:
            berkas = ekspor.bangun(isi, kode)
        except Exception as galat:  # noqa: BLE001 - kegagalan ekspor tidak menghentikan halaman
            st.error(f"Berkas gagal disusun: {galat}", icon=":material/error:")
        else:
            nama_berkas = ekspor.nama_berkas(isi, kode)
            kanan.download_button(
                f"Unduh {ekspor.FORMAT[kode].nama}",
                berkas,
                file_name=nama_berkas,
                mime=ekspor.FORMAT[kode].mime,
                type="primary",
                width="stretch",
                key=f"unduh_laporan_{kode}",
            )
            kanan.caption(f"`{nama_berkas}`")

st.divider()
if st.button("Kosongkan laporan", key="kosongkan_laporan"):
    isi.kosongkan()
    st.rerun()
st.caption(
    "Isi laporan tersimpan selama sesi berlangsung. Menutup peramban atau keluar dari "
    "akun akan mengosongkannya, jadi unduh berkasnya bila ingin disimpan."
)
