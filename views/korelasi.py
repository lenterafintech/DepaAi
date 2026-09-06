"""Halaman korelasi antar variabel dan uji kelayakan data multivariat."""

from __future__ import annotations

import streamlit as st

from mv_statlab import assumptions, correlation, plots, ui

ui.butuh_fitur("dasar")
ui.page_setup(
    "Korelasi & Uji Asumsi",
    "Data",
    "Kekuatan hubungan antarvariabel beserta pemeriksaan kelayakan data: KMO, "
    "Bartlett, multikolinearitas, dan homogenitas.",
)
df = ui.require_dataset()
ui.sidebar_info()

selected = ui.numeric_selector(df, "Variabel yang dianalisis", default_count=8, key="corr_vars")
subset = df[selected]

tab_corr, tab_partial, tab_adequacy, tab_multicol, tab_homog = st.tabs(
    ["Matriks Korelasi", "Korelasi Parsial", "Kelayakan Faktor", "Multikolinearitas", "Homogenitas"]
)

with tab_corr:
    method = st.radio(
        "Metode korelasi",
        correlation.METHODS,
        horizontal=True,
        format_func=lambda m: {
            "pearson": "Pearson (linear, data numerik normal)",
            "spearman": "Spearman (peringkat, tahan pencilan)",
            "kendall": "Kendall (peringkat, sampel kecil)",
        }[m],
    )
    result = correlation.correlation_matrix(subset, method=method)
    st.plotly_chart(
        plots.correlation_heatmap(result.matrix, f"Matriks Korelasi ({method.title()})"),
        width="stretch",
    )
    st.caption(f"n = {result.n} observasi lengkap")

    alpha = st.slider("Tingkat signifikansi (α)", 0.01, 0.10, 0.05, 0.01)
    st.subheader("Pasangan variabel terurut berdasarkan kekuatan hubungan")
    ui.show_table(result.significant_pairs(alpha), "korelasi_pasangan.csv")
    ui.interpretation(
        "Nilai r mendekati ±1 berarti hubungan makin kuat; tanda negatif berarti arah "
        "berlawanan. Kolom 'Signifikan' menunjukkan apakah hubungan tersebut kemungkinan "
        "besar bukan kebetulan pada sampel ini."
    )

with tab_partial:
    st.subheader("Korelasi parsial")
    st.markdown(
        "Korelasi parsial mengukur hubungan dua variabel **setelah mengontrol** "
        "seluruh variabel lain yang dipilih."
    )
    try:
        partial = correlation.partial_correlation(subset)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.plotly_chart(
            plots.correlation_heatmap(partial, "Matriks Korelasi Parsial"), width="stretch"
        )
        ui.show_table(partial.reset_index(names="Variabel"), "korelasi_parsial.csv")
        ui.interpretation(
            "Bila korelasi biasa tinggi tetapi korelasi parsialnya kecil, hubungan itu "
            "sebagian besar dijelaskan oleh variabel lain (hubungan semu)."
        )

with tab_adequacy:
    st.subheader("Kelayakan data untuk analisis faktor / PCA")
    try:
        kmo_result = assumptions.kmo(subset)
        bartlett = assumptions.bartlett_sphericity(subset)
    except ValueError as exc:
        st.error(str(exc))
    else:
        c1, c2 = st.columns(2)
        c1.metric("KMO keseluruhan", f"{kmo_result.overall:.3f}", kmo_result.interpretation)
        c2.metric("Bartlett p-value", f"{bartlett.p_value:.4g}")
        ui.show_table(bartlett.to_frame(), "uji_bartlett.csv")
        st.markdown("**Measure of Sampling Adequacy (MSA) per variabel**")
        ui.show_table(kmo_result.to_frame(), "kmo_per_variabel.csv")
        ui.interpretation(
            "KMO ≥ 0,5 dan Bartlett signifikan (p < 0,05) adalah syarat minimum agar data "
            "layak difaktorkan. Variabel dengan MSA < 0,5 sebaiknya dipertimbangkan untuk "
            "dikeluarkan lalu analisis diulang."
        )

with tab_multicol:
    st.subheader("Variance Inflation Factor (VIF)")
    try:
        table = assumptions.vif(subset)
    except ValueError as exc:
        st.error(str(exc))
    else:
        ui.show_table(
            table,
            "vif.csv",
            bagian="Multikolinearitas",
            judul="Nilai VIF",
        )
        worst = table.loc[table["VIF"].idxmax()]
        if worst["VIF"] >= 10:
            st.warning(
                f"Variabel '{worst['Variabel']}' memiliki VIF {worst['VIF']:.2f} "
                "(multikolinearitas tinggi). Pertimbangkan membuang salah satu variabel "
                "yang saling tumpang tindih atau gunakan PCA sebagai prediktor."
            )
        else:
            st.success("Tidak ada indikasi multikolinearitas serius (semua VIF < 10).")

with tab_homog:
    st.subheader("Homogenitas ragam & matriks kovarians antar kelompok")
    group = ui.group_selector(df, "Variabel kelompok", key="homog_group")
    try:
        box = assumptions.box_m(subset, df.loc[subset.index, group])
    except ValueError as exc:
        st.error(str(exc))
    else:
        ui.show_table(box.to_frame(), "box_m.csv")
    levene = assumptions.levene_by_variable(subset, df.loc[subset.index, group])
    if not levene.empty:
        st.markdown("**Uji Levene per variabel**")
        ui.show_table(levene, "levene.csv")
    ui.interpretation(
        "Box's M dan Levene menguji apakah sebaran antar kelompok setara. Asumsi ini "
        "dipakai oleh MANOVA dan analisis diskriminan linear; bila dilanggar, gunakan "
        "diskriminan kuadratik (QDA) atau statistik Pillai's trace yang lebih tahan."
    )
