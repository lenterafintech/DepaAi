"""Halaman Principal Component Analysis."""

from __future__ import annotations

import streamlit as st

from lentera_mva import assumptions, pca_analysis, plots, ui

ui.page_setup("Principal Component Analysis (PCA)", "🧩")
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "PCA",
    "PCA meringkas banyak variabel yang saling berkorelasi menjadi sedikit komponen "
    "baru (principal component) yang saling ortogonal. Komponen pertama menyerap "
    "varians terbesar, komponen berikutnya menyerap sisa varians. Berguna untuk "
    "reduksi dimensi, membuat indeks komposit, dan mengatasi multikolinearitas.",
)

selected = ui.numeric_selector(df, "Variabel yang dimasukkan ke PCA", default_count=8, key="pca_vars")
subset = df[selected].dropna()

c1, c2 = st.columns(2)
with c1:
    standardize = st.toggle(
        "Standardisasi variabel (matriks korelasi)",
        value=True,
        help="Wajib bila satuan antar variabel berbeda, misalnya rupiah vs tahun.",
    )
with c2:
    n_components = st.slider(
        "Jumlah komponen yang dipertahankan", 1, len(selected), min(2, len(selected))
    )

try:
    result = pca_analysis.run_pca(subset, n_components=n_components, standardize=standardize)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Komponen (eigenvalue > 1)", result.kaiser_components)
m2.metric("Komponen untuk 80% varians", result.components_needed(0.80))
m3.metric(
    "Varians dijelaskan komponen terpilih",
    f"{result.cumulative_ratio[n_components - 1] * 100:.1f}%",
)

tab_var, tab_load, tab_score, tab_check = st.tabs(
    ["Varians & Scree", "Muatan Komponen", "Skor & Biplot", "Kelayakan"]
)

with tab_var:
    st.plotly_chart(plots.scree_plot(result.eigenvalues), width="stretch")
    st.plotly_chart(plots.variance_plot(result.variance_table()), width="stretch")
    ui.show_table(result.variance_table(), "pca_varians.csv")
    ui.interpretation(
        "Pilih jumlah komponen dari titik siku (elbow) pada scree plot, dari kriteria "
        "eigenvalue > 1, atau dari target varians kumulatif (umumnya 70-80%)."
    )

    st.subheader("Analisis paralel Horn")
    if st.button("Jalankan analisis paralel (200 simulasi)"):
        with st.spinner("Menyimulasikan data acak..."):
            parallel = pca_analysis.parallel_analysis(subset)
        ui.show_table(parallel, "pca_analisis_paralel.csv")
        keep = int((parallel["Dipertahankan"] == "Ya").sum())
        st.success(f"Analisis paralel menyarankan mempertahankan {keep} komponen.")

with tab_load:
    st.plotly_chart(
        plots.loadings_heatmap(result.loadings, "Muatan Komponen (Component Loadings)"),
        width="stretch",
    )
    ui.show_table(result.loadings.reset_index(names="Variabel"), "pca_muatan.csv")
    st.markdown("**Komunalitas** — proporsi varians tiap variabel yang terserap komponen terpilih")
    ui.show_table(
        result.communalities.to_frame().reset_index(names="Variabel"),
        "pca_komunalitas.csv",
    )
    ui.interpretation(
        "Muatan |≥ 0,5| menandakan variabel tersebut mewakili komponen bersangkutan. "
        "Komunalitas rendah (< 0,3) berarti variabel kurang terwakili oleh komponen terpilih."
    )

with tab_score:
    if result.n_components >= 2:
        color_options = [None] + [
            c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 12
        ]
        color_col = st.selectbox(
            "Warnai berdasarkan",
            color_options,
            format_func=lambda v: "Tanpa pewarnaan" if v is None else v,
        )
        color = df.loc[result.scores.index, color_col] if color_col else None
        cx, cy = st.columns(2)
        x = cx.selectbox("Sumbu X", result.scores.columns, index=0)
        y = cy.selectbox("Sumbu Y", result.scores.columns, index=1)
        st.plotly_chart(plots.biplot(result.scores, result.loadings, x, y, color), width="stretch")
    else:
        st.info("Pilih minimal 2 komponen untuk menampilkan biplot.")

    st.subheader("Skor komponen tiap observasi")
    ui.show_table(result.scores.reset_index(names="Indeks"), "pca_skor.csv", height=320)
    ui.interpretation(
        "Skor komponen dapat dipakai sebagai variabel baru pada analisis lanjutan "
        "(misalnya regresi atau klaster) untuk menghindari multikolinearitas."
    )

with tab_check:
    try:
        kmo_result = assumptions.kmo(subset)
        bartlett = assumptions.bartlett_sphericity(subset)
    except ValueError as exc:
        st.error(str(exc))
    else:
        c1, c2 = st.columns(2)
        c1.metric("KMO", f"{kmo_result.overall:.3f}", kmo_result.interpretation)
        c2.metric("Bartlett p-value", f"{bartlett.p_value:.4g}")
        if kmo_result.overall < 0.5 or bartlett.p_value >= 0.05:
            st.warning(
                "Data kurang layak direduksi: pastikan variabel memang saling berkorelasi."
            )
        else:
            st.success("Data layak untuk reduksi dimensi.")
        ui.show_table(kmo_result.to_frame(), "pca_kmo.csv")
