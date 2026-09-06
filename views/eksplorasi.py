"""Halaman eksplorasi data: deskriptif, distribusi, normalitas, dan pencilan."""

from __future__ import annotations

import streamlit as st

from lentera_mva import descriptive, plots, preprocessing, ui

ui.butuh_fitur("dasar")
ui.page_setup(
    "Eksplorasi Data",
    "Data",
    "Statistik deskriptif, bentuk sebaran, uji normalitas, dan deteksi pencilan — "
    "pemeriksaan yang perlu dilakukan sebelum metode lain dijalankan.",
)
df = ui.require_dataset()
ui.sidebar_info()

tab_desc, tab_dist, tab_norm, tab_out = st.tabs(
    ["Statistik Deskriptif", "Distribusi & Hubungan", "Uji Normalitas", "Deteksi Pencilan"]
)

with tab_desc:
    st.subheader("Statistik deskriptif variabel numerik")
    ui.show_table(descriptive.describe(df), "deskriptif.csv")
    ui.interpretation(
        "Skewness di luar rentang ±2 dan kurtosis di luar ±7 menandakan distribusi "
        "yang jauh dari normal. CV (%) besar berarti variasi data relatif tinggi "
        "terhadap rata-ratanya."
    )

    cat_cols = [c for c in df.columns if not str(df[c].dtype).startswith(("int", "float"))]
    cat_cols += [c for c in df.select_dtypes("number").columns if df[c].nunique() <= 10]
    if cat_cols:
        st.subheader("Tabel frekuensi variabel kategorik")
        col = st.selectbox("Variabel", sorted(set(cat_cols)))
        ui.show_table(descriptive.frequency_table(df[col]), f"frekuensi_{col}.csv")

with tab_dist:
    st.subheader("Distribusi satu variabel")
    numeric_cols = preprocessing.numeric_columns(df)
    var = st.selectbox("Variabel numerik", numeric_cols)
    st.plotly_chart(plots.distribution_plot(df[var]), width="stretch")

    group_options = [None] + [
        c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 12
    ]
    grp = st.selectbox(
        "Bandingkan antar kelompok (opsional)",
        group_options,
        format_func=lambda v: "Tanpa pengelompokan" if v is None else v,
    )
    if grp is not None:
        st.plotly_chart(plots.box_by_group(df, var, grp), width="stretch")

    st.subheader("Matriks sebar antar variabel")
    selected = st.multiselect(
        "Variabel (maksimal 6 agar tetap terbaca)",
        numeric_cols,
        default=numeric_cols[: min(4, len(numeric_cols))],
        max_selections=6,
    )
    if len(selected) >= 2:
        color = df[grp] if grp is not None else None
        subset = preprocessing.clean_subset(df, selected)
        color = color.loc[subset.index] if color is not None else None
        st.plotly_chart(plots.scatter_matrix(subset, color), width="stretch")

with tab_norm:
    st.subheader("Normalitas univariat")
    ui.show_table(descriptive.normality_tests(df), "uji_normalitas.csv")
    ui.interpretation(
        "p-value < 0,05 berarti data variabel tersebut menyimpang dari distribusi normal. "
        "Shapiro-Wilk paling andal untuk n kecil, sedangkan D'Agostino cocok untuk n besar."
    )

    st.subheader("Normalitas multivariat (uji Mardia)")
    selected = ui.numeric_selector(df, "Variabel yang diuji bersama", key="mardia_vars")
    try:
        result = descriptive.mardia_test(df[selected])
    except ValueError as exc:
        st.error(str(exc))
    else:
        ui.show_table(result.to_frame(), "uji_mardia.csv")
        if result.multivariate_normal:
            st.success("Asumsi normalitas multivariat terpenuhi (p > 0,05 pada kedua uji).")
        else:
            st.warning(
                "Data menyimpang dari normal multivariat. Pertimbangkan transformasi "
                "(log/akar), penanganan pencilan, atau metode yang robust."
            )

with tab_out:
    st.subheader("Pencilan univariat (aturan IQR)")
    ui.show_table(descriptive.univariate_outliers(df), "pencilan_univariat.csv")

    st.subheader("Pencilan multivariat (jarak Mahalanobis)")
    selected = ui.numeric_selector(df, "Variabel pembentuk jarak", key="maha_vars")
    alpha = st.select_slider(
        "Tingkat signifikansi cutoff chi-square", options=[0.05, 0.01, 0.005, 0.001], value=0.001
    )
    try:
        table = descriptive.mahalanobis_outliers(df[selected], alpha=alpha)
    except ValueError as exc:
        st.error(str(exc))
    else:
        n_out = int((table["Pencilan"] == "Ya").sum())
        st.metric("Jumlah pencilan multivariat", n_out)
        ui.show_table(
            table.sort_values("Mahalanobis D2", ascending=False), "pencilan_mahalanobis.csv"
        )
        ui.interpretation(
            "Observasi ditandai pencilan bila jarak Mahalanobis melebihi nilai kritis "
            "chi-square pada derajat bebas sebesar jumlah variabel. Pencilan multivariat "
            "dapat sangat memengaruhi hasil PCA, regresi, dan analisis klaster."
        )
