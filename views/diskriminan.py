"""Halaman analisis diskriminan (LDA/QDA)."""

from __future__ import annotations

import streamlit as st

from lentera_mva import assumptions, discriminant, plots, preprocessing, ui

ui.butuh_fitur("uji_beda")
ui.page_setup(
    "Analisis Diskriminan",
    "Pemodelan & Uji Beda",
    "Mencari kombinasi variabel yang paling membedakan kelompok, sekaligus "
    "menguji seberapa tepat keanggotaan kelompok dapat diprediksi.",
)
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "analisis diskriminan",
    "Analisis diskriminan mencari kombinasi linear variabel prediktor yang paling "
    "membedakan kelompok yang sudah diketahui, lalu memakainya untuk mengklasifikasikan "
    "observasi. LDA mengasumsikan matriks kovarians antar kelompok sama; bila asumsi "
    "itu dilanggar, QDA lebih tepat.",
)

group = ui.group_selector(df, "Variabel kelompok (dependen)", key="da_group")
numeric_cols = [c for c in preprocessing.numeric_columns(df) if c != group]
predictors = st.multiselect(
    "Variabel prediktor numerik",
    numeric_cols,
    default=numeric_cols[: min(5, len(numeric_cols))],
    key="da_pred",
)
kind = st.radio(
    "Jenis analisis",
    ["linear", "kuadratik"],
    horizontal=True,
    format_func=lambda k: "Linear (LDA)" if k == "linear" else "Kuadratik (QDA)",
)

if len(predictors) < 2:
    st.info("Pilih minimal 2 variabel prediktor.")
    st.stop()

try:
    result = discriminant.run_discriminant(df, group, predictors, kind=kind)
except Exception as exc:  # noqa: BLE001 - kesalahan model ditampilkan ke pengguna
    st.error(f"Analisis gagal: {exc}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Akurasi (data latih)", f"{result.accuracy * 100:.2f}%")
m2.metric("Akurasi validasi silang", f"{result.cv_accuracy * 100:.2f}%")
m3.metric("Jumlah kelompok", len(result.classes))

tab_func, tab_class, tab_group, tab_assum = st.tabs(
    ["Fungsi Diskriminan", "Hasil Klasifikasi", "Profil Kelompok", "Asumsi"]
)

with tab_func:
    if result.eigenvalues is not None:
        st.markdown("**Ringkasan fungsi kanonik**")
        ui.show_table(result.eigenvalues, "diskriminan_eigenvalue.csv")
        st.markdown("**Uji signifikansi fungsi (Wilks' lambda)**")
        ui.show_table(result.wilks, "diskriminan_wilks.csv")
        ui.interpretation(
            "Fungsi dengan p < 0,05 berarti benar-benar membedakan kelompok. Korelasi "
            "kanonik menunjukkan kekuatan hubungan antara fungsi tersebut dan "
            "keanggotaan kelompok."
        )
    if result.coefficients is not None:
        st.markdown("**Koefisien fungsi diskriminan**")
        ui.show_table(
            result.coefficients.reset_index(names="Variabel"),
            "diskriminan_koefisien.csv",
        )
        ui.interpretation(
            "Semakin besar nilai mutlak koefisien, semakin besar sumbangan variabel "
            "tersebut dalam memisahkan kelompok pada fungsi bersangkutan."
        )
    if result.scores is not None and result.scores.shape[1] >= 2:
        st.plotly_chart(
            plots.scatter_2d(
                result.scores,
                result.scores.columns[0],
                result.scores.columns[1],
                result.labels,
                "Sebaran Observasi pada Dua Fungsi Diskriminan",
            ),
            width="stretch",
        )
    elif result.scores is not None:
        st.plotly_chart(
            plots.box_by_group(
                result.scores.join(result.labels), result.scores.columns[0], group
            ),
            width="stretch",
        )
    if kind == "kuadratik":
        st.info(
            "QDA tidak menghasilkan fungsi kanonik karena batas antar kelompoknya "
            "berbentuk kuadratik, bukan garis lurus."
        )

with tab_class:
    st.plotly_chart(plots.confusion_heatmap(result.confusion), width="stretch")
    ui.show_table(result.confusion.reset_index(names="Aktual"), "diskriminan_konfusi.csv")
    ui.show_table(result.summary(), "diskriminan_ringkasan.csv")
    ui.interpretation(
        "Akurasi validasi silang lebih jujur menggambarkan performa pada data baru "
        "dibanding akurasi data latih. Selisih yang besar antara keduanya menandakan "
        "model terlalu menyesuaikan data (overfitting)."
    )

    st.subheader("Prediksi tiap observasi")
    detail = df.loc[result.labels.index].copy()
    detail.insert(0, "Prediksi", result.predictions.to_numpy())
    detail.insert(0, "Aktual", result.labels.to_numpy())
    detail.insert(0, "Tepat", (result.labels == result.predictions).map({True: "Ya", False: "Tidak"}).to_numpy())
    ui.show_table(detail, "diskriminan_prediksi.csv", height=360)

with tab_group:
    st.markdown("**Rata-rata prediktor tiap kelompok (centroid)**")
    ui.show_table(result.group_means, "diskriminan_centroid.csv")
    var = st.selectbox("Lihat distribusi variabel", predictors)
    st.plotly_chart(
        plots.box_by_group(df.loc[result.labels.index], var, group), width="stretch"
    )

with tab_assum:
    subset = df.loc[result.labels.index, predictors]
    try:
        box = assumptions.box_m(subset, result.labels)
    except ValueError as exc:
        st.error(str(exc))
    else:
        ui.show_table(box.to_frame(), "diskriminan_boxm.csv")
        if not box.homogeneous and kind == "linear":
            st.warning(
                "Matriks kovarians antar kelompok tidak homogen (Box's M signifikan). "
                "Analisis diskriminan kuadratik (QDA) lebih sesuai untuk data ini."
            )
    levene = assumptions.levene_by_variable(subset, result.labels)
    if not levene.empty:
        st.markdown("**Uji Levene per variabel**")
        ui.show_table(levene, "diskriminan_levene.csv")
    st.markdown("**Multikolinearitas antar prediktor**")
    try:
        ui.show_table(assumptions.vif(subset), "diskriminan_vif.csv")
    except ValueError as exc:
        st.error(str(exc))
