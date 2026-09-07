"""Muat Data: unggah berkas, coba contoh data, atau buka proyek — lalu simpan proyek.

Dipindah dari Beranda lama. Tiga cabang yang memanggil ``ui.set_dataset(...)``
(unggah, contoh, proyek) diikuti ``st.rerun()`` agar kartu status data aktif di
puncak aplikasi langsung menampilkan data yang baru dimuat pada giliran render
berikutnya — sebelumnya hanya cabang "buka proyek" yang memanggilnya, sehingga
kartu status dan sidebar sempat menampilkan dua keadaan berbeda dalam satu render.
"""

from __future__ import annotations

import streamlit as st

from nalardata import formatting, io_utils, ui
from nalardata import proyek as pr


def render(df, kamus, penelitian, key_prefix: str = "") -> None:
    """``key_prefix`` membedakan kunci widget bila fungsi ini dirender dua kali dalam satu
    giliran — Beranda merendernya langsung sebelum data ada, dan tab Data selalu
    menyediakannya juga; tanpa kunci berbeda, keduanya akan bentrok pada giliran itu."""
    tab_upload, tab_sample, tab_proyek = st.tabs(
        ["Unggah berkas", "Contoh data", "Buka proyek"]
    )

    with tab_upload:
        uploaded = st.file_uploader(
            "Berkas CSV, TSV, Excel, atau SPSS",
            type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "sav", "zsav"],
            key=f"{key_prefix}unggah_data",
        )
        if uploaded is not None:
            sheet: str | int = 0
            if uploaded.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                sheets = io_utils.excel_sheet_names(uploaded)
                sheet = st.selectbox("Pilih sheet", sheets, key=f"{key_prefix}pilih_sheet")
            try:
                baru = io_utils.load_table(uploaded, filename=uploaded.name, sheet_name=sheet)
            except Exception as exc:  # noqa: BLE001 - pesan gagal baca ditampilkan ke pengguna
                st.error(f"Gagal membaca berkas: {exc}")
            else:
                ui.set_dataset(baru, uploaded.name)
                st.success(f"Berhasil memuat {len(baru):,} baris dan {baru.shape[1]} kolom.")
                st.rerun()

    with tab_sample:
        st.markdown(
            "Contoh data nasabah sintetis (400 baris) berisi variabel demografi, kapasitas "
            "ekonomi, perilaku pembayaran, dan status gagal bayar — cocok untuk mencoba "
            "seluruh metode."
        )
        if st.button("Muat contoh data nasabah", type="primary", key=f"{key_prefix}muat_contoh"):
            ui.set_dataset(ui.load_sample(), "contoh_data_nasabah.csv", contoh=True)
            st.success("Contoh data dimuat.")
            st.rerun()

    with tab_proyek:
        st.markdown(
            "Berkas proyek `.nalardata` memuat data, kamus variabel, hasil yang tersimpan "
            "di **Laporan**, dan pengaturan cakupan analisis sekaligus — sehingga pekerjaan "
            "dapat dilanjutkan pada sesi berikutnya."
        )
        berkas_proyek = st.file_uploader(
            "Berkas proyek", type=["nalardata", "zip"], key=f"{key_prefix}unggah_proyek"
        )
        if berkas_proyek is not None:
            try:
                proyek_dibuka = pr.buka_proyek(berkas_proyek.getvalue())
            except ValueError as galat:
                st.error(str(galat), icon=":material/error:")
            else:
                ui.show_table(proyek_dibuka.ringkas(), "isi_proyek.csv")
                if st.button(
                    "Muat proyek ini", type="primary", key=f"{key_prefix}muat_proyek"
                ):
                    ui.set_dataset(proyek_dibuka.data, proyek_dibuka.nama_data)
                    st.session_state[ui.KERANJANG_KEY] = proyek_dibuka.keranjang
                    if len(proyek_dibuka.kamus):
                        ui.set_kamus(proyek_dibuka.kamus)
                    if not proyek_dibuka.penelitian.kosong():
                        ui.set_penelitian(proyek_dibuka.penelitian)
                    if not proyek_dibuka.jejak.kosong():
                        ui.set_jejak(proyek_dibuka.jejak)
                    for kunci, nilai in (proyek_dibuka.konfigurasi or {}).items():
                        # Kunci widget halaman laporan dipulihkan apa adanya.
                        st.session_state[kunci] = nilai
                    st.success("Proyek dimuat. Data dan hasil tersimpan sudah pulih.")
                    st.rerun()

    if df is None:
        return

    st.divider()
    st.markdown("**Pratinjau data**")
    st.html(
        "<div>"
        + ui.pil(f"{formatting.num(len(df))} baris", "info")
        + ui.pil(f"{df.shape[1]} kolom", "info")
        + ui.pil(f"{len(df.select_dtypes('number').columns)} numerik", "netral")
        + ui.pil(f"{formatting.num(int(df.isna().sum().sum()))} sel kosong", "netral")
        + "</div>"
    )
    st.dataframe(df.head(50), width="stretch", hide_index=True)
    with st.expander("Profil variabel"):
        ui.show_table(io_utils.profile(df), "profil_variabel.csv")

    st.divider()
    st.markdown("**Simpan proyek**")
    st.caption(
        "Menyimpan data aktif, hasil yang sudah tersimpan di Laporan, dan pengaturan "
        "cakupan analisis ke dalam satu berkas. Aplikasi ini tidak menyimpan apa pun di "
        "server, jadi berkas inilah satu-satunya cara melanjutkan pekerjaan nanti."
    )
    isi_keranjang = ui.keranjang()
    kunci_konfig = [k for k in st.session_state if str(k).startswith("kesimpulan_")]
    konfigurasi = {k: st.session_state[k] for k in kunci_konfig}

    try:
        berkas = pr.simpan_proyek(
            df,
            st.session_state.get(ui.NAME_KEY, "data"),
            isi_keranjang,
            konfigurasi,
            kamus=kamus,
            penelitian=penelitian,
            jejak=ui.jejak(),
        )
    except ValueError as galat:
        st.info(str(galat))
    else:
        kiri, kanan = st.columns([1, 2])
        kiri.download_button(
            "Unduh berkas proyek",
            berkas,
            file_name=pr.nama_berkas_proyek(st.session_state.get(ui.NAME_KEY, "data")),
            mime="application/zip",
            type="primary",
            width="stretch",
            key=f"{key_prefix}unduh_proyek",
        )
        kanan.caption(
            f"Berisi {len(df):,} baris data".replace(",", ".")
            + f" · {len(isi_keranjang.item)} hasil tersimpan"
            + f" · kamus {len(kamus)} variabel"
            + (f" · {len(konfigurasi)} pengaturan" if konfigurasi else "")
        )
