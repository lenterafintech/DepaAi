"""Halaman analisis teks: statistik dasar, tokenisasi, dan frekuensi kata."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import formatting, plots, teks_analysis as tk, ui


def render(df, kamus, penelitian) -> None:
    if not ui.butuh_fitur("lanjutan"):
        return

    ui.method_note(
        "analisis teks",
        "Cakupan halaman ini sengaja dibatasi pada yang **deskriptif**: statistik "
        "teks, tokenisasi, dan frekuensi kata bersih dari imbuhan — bukan pengkodean "
        "tematik atau topic modeling penuh. Kata diafiks lewat Sastrawi supaya "
        "'menganalisis', 'dianalisis', dan 'analisis' dihitung sebagai kata yang sama, "
        "dan stopword bahasa Indonesia (kata sambung, kata depan, dsb.) dibuang "
        "sebelum dihitung.",
    )

    dipandu = ui.konfigurasi_pemandu()
    ui.banner_dipandu(dipandu)

    kandidat = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if not kandidat:
        st.error(
            "Tidak ada kolom bertipe teks pada data ini. Analisis teks memerlukan "
            "kolom berisi kalimat atau paragraf, bukan kategori pendek."
        )
        return

    kolom = st.selectbox(
        "Kolom teks yang dianalisis",
        kandidat,
        index=ui.indeks_pilihan(kandidat, dipandu.get("outcome")),
        key="teks_kolom",
        help="Pilih kolom berisi jawaban terbuka, komentar, atau esai — bukan "
        "kolom kategori pendek seperti jenis kelamin atau wilayah.",
    )

    teks = df[kolom]
    try:
        statistik = tk.statistik_dasar(teks)
    except ValueError as exc:
        st.error(str(exc))
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dokumen", formatting.num(statistik.n_dokumen))
    m2.metric("Total kata", formatting.num(statistik.jumlah_kata))
    m3.metric("Kosakata unik", formatting.num(statistik.kosakata_unik))
    m4.metric("Rata-rata kata/dokumen", formatting.num(statistik.rata_kata_per_dokumen, 1))

    kiri, kanan = st.columns(2)
    hapus_stopword = kiri.checkbox(
        "Buang stopword bahasa Indonesia", value=True, key="teks_stopword"
    )
    stem = kanan.checkbox(
        "Satukan imbuhan lewat stemming (Sastrawi)", value=True, key="teks_stem"
    )
    stopword_tambahan_teks = st.text_input(
        "Stopword tambahan (opsional, pisahkan dengan koma)",
        key="teks_stopword_tambahan",
        help="Kata yang ingin ikut dibuang tapi belum termasuk daftar stopword bawaan, "
        "misalnya nama produk atau singkatan yang sering muncul di seluruh dokumen.",
    )
    stopword_tambahan = [s for s in stopword_tambahan_teks.split(",") if s.strip()]

    with st.spinner("Memproses teks..."):
        token = tk.bersihkan_token(
            teks, hapus_stopword=hapus_stopword, stem=stem, stopword_tambahan=stopword_tambahan
        )

    if not token:
        st.warning(
            "Tidak ada kata tersisa setelah dibersihkan — mungkin seluruh isi "
            "kolom ini adalah stopword, atau coba matikan pembuangan stopword."
        )
        return

    n_teratas = st.slider("Jumlah kata tersering yang ditampilkan", 5, 50, 20, key="teks_n")
    tabel = tk.frekuensi_kata(token, n_teratas=n_teratas)

    st.plotly_chart(plots.word_frequency_bar(tabel, n_teratas), width="stretch")
    ui.show_table(
        tabel,
        "teks_frekuensi_kata.csv",
        bagian="Analisis teks",
        judul=f"Kata tersering pada kolom '{kolom}'",
    )
    ui.interpretation(
        "Frekuensi di sini dihitung dari kata yang sudah disatukan bentuk dasarnya "
        "(bila stemming diaktifkan) — 'pelatihan' dan 'melatih' tercatat sebagai "
        "kata yang sama. Kata yang paling sering muncul menunjukkan tema yang "
        "paling menonjol menurut responden sendiri, tetapi tetap perlu dibaca "
        "dalam konteks kalimat aslinya sebelum disimpulkan sebagai temuan."
    )

    st.caption(
        f"n = {formatting.num(statistik.n_dokumen)} dokumen, "
        f"{formatting.num(len(token))} kata tersisa setelah dibersihkan dari "
        f"{formatting.num(statistik.jumlah_kata)} kata mentah."
    )
