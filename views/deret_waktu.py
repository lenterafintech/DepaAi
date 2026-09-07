"""Halaman deret waktu: uji stasioneritas (ADF) dan peramalan ARIMA."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import arima_analysis as ar
from nalardata import formatting, plots, preprocessing, ui


def render(df, kamus, penelitian) -> None:
    if not ui.butuh_fitur("lanjutan"):
        return

    ui.method_note(
        "deret waktu (ARIMA)",
        "ARIMA meramalkan sebuah deret dari polanya sendiri di masa lalu — bukan dari "
        "variabel lain seperti regresi. Data diuji stasioner lebih dulu (Augmented "
        "Dickey-Fuller): rata-rata dan ragamnya harus stabil sepanjang waktu sebelum "
        "polanya dapat diestimasi dengan andal. Bila belum stasioner, deret di-"
        "differencing otomatis sampai stasioner atau batas differencing tercapai.",
    )

    numerik = preprocessing.numeric_columns(df)
    if not numerik:
        st.error("Tidak ada kolom numerik yang dapat diramalkan.")
        return

    kol1, kol2 = st.columns(2)
    kandidat_waktu = [c for c in df.columns if c not in numerik]
    urutan = kol1.selectbox(
        "Kolom urutan waktu (opsional — bila kosong, urutan baris data dipakai apa adanya)",
        [None] + kandidat_waktu,
        format_func=lambda k: "— urutan baris —" if k is None else k,
        key="arima_waktu",
        help="Kolom tanggal/periode untuk mengurutkan data. Data yang belum terurut "
        "waktu akan menghasilkan pola yang keliru.",
    )
    nilai = kol2.selectbox("Kolom nilai yang diramalkan", numerik, key="arima_nilai")

    kerja = df.sort_values(urutan) if urutan else df
    series = pd.to_numeric(kerja[nilai], errors="coerce").dropna().reset_index(drop=True)
    if len(series) < 15:
        st.error(
            f"Perlu sekurang-kurangnya 15 titik waktu; data ini hanya berisi "
            f"{len(series)} nilai yang valid."
        )
        return

    try:
        stasioneritas = ar.uji_stasioneritas(series)
    except ValueError as exc:
        st.error(str(exc))
        return

    if stasioneritas.stasioner:
        st.success(
            f"Deret sudah stasioner. {stasioneritas.rincian}",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            f"Deret belum sepenuhnya stasioner. {stasioneritas.rincian}",
            icon=":material/warning:",
        )

    with st.spinner("Mencari order ARIMA terbaik (p, d, q)..."):
        try:
            hasil = ar.fit_arima(series)
        except ValueError as exc:
            st.error(str(exc))
            return

    p, d, q = hasil.order
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Order (p, d, q)", f"({p}, {d}, {q})")
    m2.metric("AIC", formatting.num(hasil.aic, 1))
    m3.metric("BIC", formatting.num(hasil.bic, 1))
    m4.metric("n", formatting.num(hasil.n))

    if hasil.residual_bersih():
        st.success(
            "Ljung-Box tidak signifikan: residual tidak menyisakan autokorelasi yang "
            "berarti — pola pada deret sudah tertangkap model.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            "Ljung-Box signifikan: residual masih menyisakan autokorelasi. Order "
            "ARIMA ini mungkin belum menangkap seluruh pola pada deret.",
            icon=":material/warning:",
        )

    tab_koef, tab_diag, tab_ramal = st.tabs(["Koefisien", "Diagnostik Residual", "Ramalan"])

    with tab_koef:
        ui.show_table(
            hasil.ringkasan_koefisien(),
            "arima_koefisien.csv",
            bagian="Deret waktu (ARIMA)",
            judul=f"Koefisien ARIMA{hasil.order}",
        )
        ui.interpretation(
            "Koefisien AR (ar.L…) menunjukkan seberapa besar nilai masa lalu ikut "
            "menentukan nilai sekarang; koefisien MA (ma.L…) menunjukkan seberapa "
            "besar galat peramalan masa lalu ikut terbawa. sigma2 adalah ragam "
            "residual, bukan koefisien yang ditafsirkan arah pengaruhnya."
        )

    with tab_diag:
        ui.show_table(
            hasil.uji_ljung_box(),
            "arima_ljung_box.csv",
            bagian="Deret waktu (ARIMA)",
            judul="Uji Ljung-Box pada residual",
        )
        ui.interpretation(
            "H0 Ljung-Box: tidak ada autokorelasi tersisa pada residual. p ≥ 0,05 "
            "berarti model sudah cukup menangkap pola deret; p < 0,05 menandakan "
            "masih ada pola yang terlewat, dan order ARIMA sebaiknya ditinjau ulang."
        )

    with tab_ramal:
        langkah = st.slider("Jumlah langkah ke depan yang diramalkan", 1, 30, 10, key="arima_langkah")
        ramalan = hasil.ramalkan(langkah)
        st.plotly_chart(
            plots.time_series_forecast(series, ramalan, nilai), width="stretch"
        )
        ui.show_table(
            ramalan,
            "arima_ramalan.csv",
            bagian="Deret waktu (ARIMA)",
            judul=f"Ramalan {langkah} langkah ke depan",
        )
        ui.interpretation(
            "Pita di sekitar garis ramalan adalah interval kepercayaan 95% — semakin "
            "jauh langkah peramalannya, semakin lebar pitanya, karena ketidakpastian "
            "menumpuk pada tiap langkah."
        )

    st.caption(
        f"Order dipilih otomatis lewat pencarian AIC terkecil pada rentang p, q "
        f"hingga {ar.MAKS_P} — bukan dibaca manual dari plot ACF/PACF."
    )
