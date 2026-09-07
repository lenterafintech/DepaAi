"""Penanganan Data: menerapkan imputasi, penskalaan, dan encoding secara eksplisit.

Melengkapi Rapor Data, yang hanya bisa menandai atau menghapus baris/kolom.
``nalardata/preprocessing.py`` sudah punya imputasi, penskalaan, dan encoding
kategorik — tapi selama ini hanya dipakai diam-diam di dalam tiap modul
statistik (lewat ``clean_subset``, yaitu penghapusan baris tak lengkap) atau
lewat kontrol yang hanya terpasang di halaman Analisis Klaster. Di sini
pengguna dapat menerapkannya secara sadar sebelum lanjut ke Analisis.

Riwayat dan pembatalan memakai kunci session state yang sama dengan
``rapor_data.RIWAYAT`` — supaya tombol "Batalkan terakhir" di Rapor Data juga
membatalkan aksi dari sini, tanpa perlu mekanisme undo kedua.
"""

from __future__ import annotations

import streamlit as st

from nalardata import formatting, preprocessing as pp, ui
from views import rapor_data


def render(df, kamus, penelitian) -> None:
    if not ui.butuh_fitur("dasar"):
        return

    numerik = pp.numeric_columns(df)
    kategorik = [c for c in df.columns if c not in numerik]

    st.caption(
        "Tindakan di sini mengubah data aktif untuk seluruh tab berikutnya. "
        "Setiap tindakan tercatat di riwayat dan dapat dibatalkan lewat tombol "
        "“Batalkan terakhir” di Rapor Data."
    )

    with st.expander("Nilai hilang & penskalaan", expanded=True):
        if not numerik:
            st.caption("Tidak ada kolom numerik pada data ini.")
        else:
            kolom_pilih = st.multiselect(
                "Kolom numerik", numerik, default=numerik, key="pen_num_kolom"
            )
            missing, scaling = ui.preprocessing_controls("pen_data")

            if kolom_pilih:
                kosong = int(df[kolom_pilih].isna().sum().sum())
                st.caption(
                    f"Pratinjau: {formatting.num(kosong)} sel kosong akan ditangani "
                    f"({missing}), lalu kolom terpilih diskalakan ({scaling})."
                )
                pratinjau = pp.scale(pp.handle_missing(df[kolom_pilih], missing), scaling)
                st.dataframe(pratinjau.head(10), width="stretch", hide_index=True)

                if st.button(
                    "Terapkan pada data aktif", key="pen_num_terapkan", type="primary"
                ):
                    if missing == "hapus baris":
                        baru = df.dropna(subset=kolom_pilih).reset_index(drop=True)
                        baru[kolom_pilih] = pp.scale(baru[kolom_pilih], scaling)
                    else:
                        baru = df.copy()
                        baru[kolom_pilih] = pp.scale(
                            pp.handle_missing(df[kolom_pilih], missing), scaling
                        )
                    _terapkan(
                        df,
                        baru,
                        f"Nilai hilang ({missing}) + penskalaan ({scaling}) pada "
                        f"{len(kolom_pilih)} kolom numerik",
                    )

    if kategorik:
        with st.expander("Encoding kategorik"):
            kolom_kat = st.multiselect("Kolom kategorik", kategorik, key="pen_kat_kolom")
            metode = st.radio(
                "Metode", ["one-hot", "ordinal"], horizontal=True, key="pen_kat_metode"
            )
            if kolom_kat:
                st.caption(
                    "One-hot membuat kolom biner baru per kategori (kategori pertama "
                    "jadi acuan); ordinal mengubah tiap kategori jadi satu angka kode."
                )
                if st.button("Terapkan encoding", key="pen_kat_terapkan", type="primary"):
                    baru = pp.encode_categorical(df, kolom_kat, metode)
                    _terapkan(
                        df,
                        baru,
                        f"Encoding {metode} pada {len(kolom_kat)} kolom kategorik",
                    )


def _terapkan(sebelum, baru, catatan: str) -> None:
    riwayat = st.session_state.setdefault(rapor_data.RIWAYAT, [])
    riwayat.append((sebelum.copy(), catatan))
    ui.set_dataset(baru, st.session_state.get(ui.NAME_KEY, "data"))
    ui.jejak().catat_perubahan(catatan, halaman="Penanganan Data")
    st.rerun()
