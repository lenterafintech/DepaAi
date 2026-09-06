"""Halaman analisis faktor eksploratori (EFA)."""

from __future__ import annotations

import streamlit as st

from lentera_mva import assumptions, factor_analysis, pca_analysis, plots, ui

ui.butuh_fitur("reduksi")
ui.page_setup(
    "Analisis Faktor Eksploratori",
    "Reduksi Dimensi",
    "Menemukan faktor laten yang menjelaskan pola korelasi antarvariabel, lalu "
    "memutarnya agar tiap faktor lebih mudah ditafsirkan.",
)
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "analisis faktor",
    "Analisis faktor mencari **variabel laten** (faktor) yang tidak terukur langsung "
    "namun menjelaskan pola korelasi antar variabel teramati. Berbeda dari PCA yang "
    "murni meringkas varians, analisis faktor memodelkan varians bersama (common "
    "variance) dan menyisakan varians unik tiap variabel.",
)

selected = ui.numeric_selector(df, "Variabel indikator", default_count=8, key="fa_vars")
subset = df[selected].dropna()

c1, c2, c3 = st.columns(3)
with c1:
    n_factors = st.slider("Jumlah faktor", 1, len(selected), min(2, len(selected)))
with c2:
    method = st.selectbox(
        "Metode ekstraksi",
        factor_analysis.EXTRACTION_METHODS,
        index=1,
        format_func=lambda m: {
            "principal": "Principal component",
            "paf": "Principal axis factoring",
            "ml": "Maximum likelihood",
        }[m],
    )
with c3:
    rotation = st.selectbox("Rotasi", factor_analysis.ROTATIONS, index=1)

try:
    result = factor_analysis.run_factor_analysis(
        subset, n_factors=n_factors, method=method, rotation=rotation
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    kmo_result = assumptions.kmo(subset)
    bartlett = assumptions.bartlett_sphericity(subset)
    m1, m2, m3 = st.columns(3)
    m1.metric("KMO", f"{kmo_result.overall:.3f}", kmo_result.interpretation)
    m2.metric("Bartlett p-value", f"{bartlett.p_value:.4g}")
    m3.metric(
        "Varians dijelaskan", f"{result.variance['% Kumulatif'].iloc[-1]:.1f}%"
    )
    if kmo_result.overall < 0.5 or bartlett.p_value >= 0.05:
        st.warning(
            "Syarat kelayakan belum terpenuhi (KMO < 0,5 atau Bartlett tidak signifikan). "
            "Hasil faktor perlu ditafsirkan dengan hati-hati."
        )
except ValueError:
    pass

tab_load, tab_var, tab_scores, tab_retain = st.tabs(
    ["Matriks Faktor", "Varians Dijelaskan", "Skor Faktor", "Penentuan Jumlah Faktor"]
)

with tab_load:
    st.plotly_chart(
        plots.loadings_heatmap(result.loadings, f"Muatan Faktor ({result.rotation})"),
        width="stretch",
    )
    threshold = st.slider("Ambang muatan yang dianggap berarti", 0.3, 0.8, 0.5, 0.05)
    display = result.loadings
    ui.show_table(display.reset_index(names="Variabel"), "faktor_muatan.csv")

    st.markdown("**Pemetaan variabel ke faktor dominan**")
    ui.show_table(result.dominant_loadings(threshold), "faktor_pemetaan.csv")

    st.markdown("**Komunalitas dan keunikan**")
    comm = (
        result.communalities
        .to_frame()
        .join(result.uniquenesses)
        .reset_index(names="Variabel")
    )
    ui.show_table(comm, "faktor_komunalitas.csv")

    if result.factor_correlation is not None:
        st.markdown("**Korelasi antar faktor** (rotasi promax bersifat oblique)")
        ui.show_table(
            result.factor_correlation.reset_index(names="Faktor"),
            "faktor_korelasi.csv",
        )
    ui.interpretation(
        f"Variabel dengan |muatan| ≥ {threshold:.2f} dianggap membentuk faktor tersebut. "
        "Variabel yang memuat tinggi pada lebih dari satu faktor (cross-loading) atau "
        "berkomunalitas rendah sebaiknya ditinjau ulang. Beri nama faktor sesuai makna "
        "bersama variabel penyusunnya."
    )

with tab_var:
    ui.show_table(result.variance, "faktor_varians.csv")
    st.plotly_chart(plots.scree_plot(result.eigenvalues), width="stretch")

with tab_scores:
    ui.show_table(
        result.scores.reset_index(names="Indeks"), "faktor_skor.csv", height=320
    )
    if result.n_factors >= 2:
        cx, cy = st.columns(2)
        x = cx.selectbox("Sumbu X", result.scores.columns, index=0)
        y = cy.selectbox("Sumbu Y", result.scores.columns, index=1)
        color_options = [None] + [
            c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 12
        ]
        color_col = st.selectbox(
            "Warnai berdasarkan",
            color_options,
            format_func=lambda v: "Tanpa pewarnaan" if v is None else v,
        )
        color = df.loc[result.scores.index, color_col] if color_col else None
        st.plotly_chart(
            plots.scatter_2d(result.scores, x, y, color, "Sebaran Skor Faktor"),
            width="stretch",
        )

with tab_retain:
    st.markdown(
        "Tiga pendekatan lazim menentukan jumlah faktor: kriteria eigenvalue > 1, "
        "titik siku pada scree plot, dan analisis paralel Horn (paling akurat)."
    )
    kaiser = int((result.eigenvalues > 1).sum())
    st.metric("Saran kriteria Kaiser (eigenvalue > 1)", kaiser)
    if st.button("Jalankan analisis paralel Horn"):
        with st.spinner("Menyimulasikan data acak..."):
            parallel = pca_analysis.parallel_analysis(subset)
        ui.show_table(parallel, "faktor_analisis_paralel.csv")
        st.success(
            f"Analisis paralel menyarankan {int((parallel['Dipertahankan'] == 'Ya').sum())} faktor."
        )
