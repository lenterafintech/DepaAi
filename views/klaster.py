"""Halaman analisis klaster: K-Means, hierarki, dan DBSCAN."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lentera_mva import clustering, pca_analysis, plots, ui

ui.page_setup("Analisis Klaster", "🎯")
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "analisis klaster",
    "Analisis klaster mengelompokkan observasi yang mirip berdasarkan banyak variabel "
    "sekaligus, tanpa label kelompok yang sudah diketahui. K-Means cepat untuk data "
    "besar dan klaster berbentuk bulat; hierarki memberi dendrogram yang menunjukkan "
    "struktur penggabungan; DBSCAN menemukan klaster berbentuk bebas sekaligus "
    "menandai pencilan.",
)

selected = ui.numeric_selector(df, "Variabel pembentuk klaster", default_count=6, key="cl_vars")
missing, scaling = ui.preprocessing_controls("cluster")
data = ui.prepare_numeric(df, selected, missing, scaling)
if scaling == "tanpa penskalaan":
    st.warning(
        "Tanpa penskalaan, variabel bersatuan besar (misalnya rupiah) akan mendominasi "
        "jarak antar observasi. Penskalaan z-score umumnya disarankan."
    )

algorithm = st.radio(
    "Algoritma", ["K-Means", "Hierarki (Agglomerative)", "DBSCAN"], horizontal=True
)

st.divider()

if algorithm == "K-Means":
    st.subheader("Penentuan jumlah klaster")
    kmax = st.slider("Uji k hingga", 3, min(12, max(3, len(data) - 1)), 8)
    with st.spinner("Menghitung metrik untuk tiap k..."):
        diagnostics = clustering.kmeans_diagnostics(data, 2, kmax)
    st.plotly_chart(plots.elbow_plot(diagnostics), width="stretch")
    ui.show_table(diagnostics, "kmeans_diagnostik.csv")
    best_k = int(diagnostics.loc[diagnostics["Silhouette"].idxmax(), "k"])
    st.caption(f"Silhouette tertinggi tercapai pada k = {best_k}.")

    k = st.slider("Jumlah klaster yang dipakai", 2, kmax, best_k)
    result = clustering.run_kmeans(data, k)

elif algorithm == "Hierarki (Agglomerative)":
    c1, c2, c3 = st.columns(3)
    linkage_method = c1.selectbox("Metode linkage", clustering.LINKAGE_METHODS)
    metric = c2.selectbox(
        "Ukuran jarak",
        clustering.DISTANCE_METRICS,
        disabled=linkage_method == "ward",
        help="Metode ward selalu memakai jarak euclidean.",
    )
    k = c3.slider("Jumlah klaster", 2, 10, 3)
    result = clustering.run_hierarchical(data, k, linkage_method, metric)
    if result.linkage_matrix is not None:
        st.plotly_chart(plots.dendrogram(result.linkage_matrix), width="stretch")

else:
    c1, c2 = st.columns(2)
    eps = c1.slider("eps (radius ketetanggaan)", 0.1, 5.0, 1.0, 0.1)
    min_samples = c2.slider("min_samples (minimal anggota inti)", 2, 25, 5)
    result = clustering.run_dbscan(data, eps=eps, min_samples=min_samples)
    noise = int((result.labels == -1).sum())
    st.caption(
        f"{result.n_clusters} klaster terbentuk, {noise} observasi ditandai sebagai "
        "noise/pencilan (label -1)."
    )
    if result.n_clusters == 0:
        st.warning("Belum ada klaster terbentuk. Perbesar eps atau kecilkan min_samples.")

st.divider()
st.subheader("Hasil pengelompokan")

m1, m2, m3 = st.columns(3)
m1.metric("Jumlah klaster", result.n_clusters)
m2.metric(
    "Silhouette", f"{result.silhouette:.3f}" if result.silhouette is not None else "-"
)
m3.metric("Observasi terklaster", int((result.labels >= 0).sum()))

tab_profile, tab_viz, tab_quality, tab_data = st.tabs(
    ["Profil Klaster", "Visualisasi", "Kualitas", "Data Berlabel"]
)

with tab_profile:
    ui.show_table(result.sizes(), "klaster_ukuran.csv")
    st.markdown("**Karakteristik tiap klaster (nilai asli, bukan hasil penskalaan)**")
    profile = clustering.profile_clusters(df.loc[result.labels.index, selected], result.labels)
    ui.show_table(profile, "klaster_profil.csv")
    ui.interpretation(
        "Bandingkan rata-rata tiap klaster terhadap rata-rata total untuk menamai "
        "klaster. Kolom F dan p-value menunjukkan variabel mana yang paling membedakan "
        "klaster; variabel dengan p ≥ 0,05 tidak berkontribusi membedakan kelompok."
    )
    if result.centers is not None:
        st.markdown("**Centroid pada skala analisis**")
        ui.show_table(result.centers.reset_index(names="Klaster"), "klaster_centroid.csv")

with tab_viz:
    if len(selected) >= 2:
        pca_result = pca_analysis.run_pca(data, n_components=2, standardize=True)
        scores = pca_result.scores.loc[result.labels.index]
        st.plotly_chart(
            plots.scatter_2d(
                scores,
                "PC1",
                "PC2",
                result.labels,
                "Sebaran Klaster pada Dua Komponen Utama",
            ),
            width="stretch",
        )
        st.caption(
            f"Dua komponen ini menjelaskan "
            f"{pca_result.cumulative_ratio[1] * 100:.1f}% varians data."
        )
    cx, cy = st.columns(2)
    xvar = cx.selectbox("Sumbu X (variabel asli)", selected, index=0)
    yvar = cy.selectbox("Sumbu Y (variabel asli)", selected, index=min(1, len(selected) - 1))
    raw = df.loc[result.labels.index, [xvar, yvar]]
    st.plotly_chart(
        plots.scatter_2d(raw, xvar, yvar, result.labels, "Sebaran Klaster pada Variabel Asli"),
        width="stretch",
    )

with tab_quality:
    ui.show_table(result.quality_table(), "klaster_kualitas.csv")
    if result.n_clusters >= 2:
        detail = clustering.silhouette_detail(data, result.labels)
        st.plotly_chart(plots.silhouette_plot(detail), width="stretch")
    ui.interpretation(
        "Silhouette berkisar -1 sampai 1: nilai > 0,5 menunjukkan klaster yang terpisah "
        "baik, sedangkan nilai negatif berarti observasi kemungkinan salah kelompok."
    )

with tab_data:
    labeled = df.loc[result.labels.index].copy()
    labeled.insert(0, "Klaster", result.labels.to_numpy())
    ui.show_table(labeled, "data_dengan_klaster.csv", height=420)
    if st.button("Simpan label klaster ke data aktif"):
        updated = df.copy()
        updated["klaster"] = pd.Series(result.labels, index=result.labels.index)
        ui.set_dataset(updated, st.session_state.get(ui.NAME_KEY, "data") + " (+klaster)")
        st.success(
            "Kolom 'klaster' ditambahkan ke data aktif dan siap dipakai pada halaman "
            "lain, misalnya MANOVA atau analisis diskriminan."
        )
