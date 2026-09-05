"""Halaman regresi linear berganda dan regresi logistik biner."""

from __future__ import annotations

import streamlit as st

from lentera_mva import plots, preprocessing, regression, ui

ui.page_setup(
    "Analisis Regresi",
    "Pemodelan & Uji Beda",
    "Memodelkan satu variabel hasil dari beberapa faktor penjelas, lengkap "
    "dengan pemeriksaan asumsi klasiknya.",
)
df = ui.require_dataset()
ui.sidebar_info()

tab_linear, tab_logistic = st.tabs(["Regresi Linear Berganda", "Regresi Logistik Biner"])

with tab_linear:
    ui.method_note(
        "regresi linear berganda",
        "Memodelkan variabel dependen numerik sebagai fungsi linear dari beberapa "
        "prediktor. Menghasilkan koefisien pengaruh tiap prediktor, ukuran kecocokan "
        "model (R²), serta pemeriksaan asumsi klasik.",
    )
    numeric_cols = preprocessing.numeric_columns(df)
    if len(numeric_cols) < 2:
        st.error("Regresi memerlukan minimal 2 kolom numerik.")
        st.stop()

    y = st.selectbox("Variabel dependen (Y)", numeric_cols, key="lin_y")
    candidates = [c for c in df.columns if c != y and df[c].nunique(dropna=True) > 1]
    predictors = st.multiselect(
        "Variabel prediktor (X) — kolom kategorik otomatis diubah jadi dummy",
        candidates,
        default=[c for c in numeric_cols if c != y][:3],
        key="lin_x",
    )

    if not predictors:
        st.info("Pilih minimal satu prediktor.")
    else:
        try:
            result = regression.linear_regression(df, y, predictors)
        except Exception as exc:  # noqa: BLE001 - kesalahan model ditampilkan ke pengguna
            st.error(f"Model gagal diestimasi: {exc}")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R²", f"{result.model.rsquared:.4f}")
            m2.metric("Adjusted R²", f"{result.model.rsquared_adj:.4f}")
            m3.metric("F", f"{result.model.fvalue:.2f}")
            m4.metric("p-value model", f"{result.model.f_pvalue:.4g}")

            st.subheader("Koefisien")
            ui.show_table(result.coefficients, "regresi_koefisien.csv")
            st.code(result.equation(), language="text")
            ui.interpretation(
                "Koefisien B menunjukkan perubahan rata-rata Y saat prediktor naik satu "
                "satuan, dengan prediktor lain tetap. Beta (baku) membuat pengaruh antar "
                "prediktor dapat dibandingkan langsung karena satuannya diseragamkan."
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ukuran kecocokan model**")
                st.dataframe(ui.styled(result.fit), width="stretch", hide_index=True)
            with c2:
                st.markdown("**Tabel ANOVA**")
                st.dataframe(ui.styled(result.anova), width="stretch", hide_index=True)

            st.subheader("Uji asumsi klasik")
            ui.show_table(result.diagnostics, "regresi_asumsi.csv")
            if not result.vif.empty:
                st.markdown("**Multikolinearitas (VIF)**")
                ui.show_table(result.vif, "regresi_vif.csv")
            st.plotly_chart(
                plots.residual_plots(result.fitted, result.residuals), width="stretch"
            )
            failed = result.diagnostics.loc[
                result.diagnostics["Kesimpulan"] != "Terpenuhi", "Asumsi"
            ].tolist()
            if failed:
                st.warning(
                    "Asumsi yang perlu diperiksa: " + "; ".join(failed) + ". "
                    "Pertimbangkan transformasi variabel, penanganan pencilan, atau "
                    "standard error yang robust."
                )
            else:
                st.success("Seluruh asumsi klasik yang diuji terpenuhi.")

            with st.expander("Seleksi variabel otomatis (stepwise)"):
                st.markdown(
                    "Seleksi maju-mundur berbasis p-value untuk menyaring prediktor "
                    "yang berkontribusi."
                )
                if st.button("Jalankan stepwise"):
                    chosen, log = regression.stepwise_selection(df, y, predictors)
                    if log.empty:
                        st.info("Tidak ada prediktor yang memenuhi ambang p < 0,05.")
                    else:
                        ui.show_table(log, "regresi_stepwise.csv")
                    st.success(
                        "Prediktor terpilih: " + (", ".join(chosen) if chosen else "(tidak ada)")
                    )

            with st.expander("Ringkasan statsmodels lengkap"):
                st.code(result.model.summary().as_text(), language="text")

with tab_logistic:
    ui.method_note(
        "regresi logistik biner",
        "Memodelkan peluang terjadinya satu dari dua kategori (misalnya gagal bayar "
        "vs lancar). Hasil utamanya odds ratio: berapa kali lipat peluang kejadian "
        "berubah saat prediktor naik satu satuan.",
    )
    binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if not binary_cols:
        st.error(
            "Tidak ada kolom dengan tepat 2 kategori pada data ini. Regresi logistik "
            "biner membutuhkan variabel dependen dua kategori."
        )
    else:
        y = st.selectbox("Variabel dependen biner (Y)", binary_cols, key="log_y")
        levels = sorted(df[y].dropna().unique().tolist(), key=str)
        positive = st.selectbox(
            "Kategori yang dianggap 'kejadian' (positif)", levels, index=len(levels) - 1
        )
        candidates = [c for c in df.columns if c != y and df[c].nunique(dropna=True) > 1]
        predictors = st.multiselect(
            "Variabel prediktor (X)",
            candidates,
            default=[c for c in preprocessing.numeric_columns(df) if c != y][:3],
            key="log_x",
        )
        threshold = st.slider("Ambang klasifikasi", 0.05, 0.95, 0.5, 0.05)

        if not predictors:
            st.info("Pilih minimal satu prediktor.")
        else:
            try:
                result = regression.logistic_regression(
                    df, y, predictors, positive_class=positive, threshold=threshold
                )
            except Exception as exc:  # noqa: BLE001 - kesalahan model ditampilkan ke pengguna
                st.error(f"Model gagal diestimasi: {exc}")
            else:
                perf = dict(zip(result.performance["Metrik"], result.performance["Nilai"]))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AUC", f"{result.auc:.4f}")
                m2.metric("Akurasi", f"{perf['Akurasi']:.4f}")
                m3.metric("Recall", f"{perf['Recall (Sensitivitas)']:.4f}")
                m4.metric("Presisi", f"{perf['Presisi']:.4f}")

                st.subheader("Koefisien dan odds ratio")
                ui.show_table(result.coefficients, "logistik_koefisien.csv")
                ui.interpretation(
                    f"Kategori positif = '{result.classes[1]}'. Odds ratio > 1 berarti "
                    "prediktor menaikkan peluang kejadian, < 1 berarti menurunkan. "
                    "Selang kepercayaan yang melewati angka 1 menandakan pengaruh tidak "
                    "signifikan secara statistik."
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Kecocokan model**")
                    st.dataframe(ui.styled(result.fit), width="stretch", hide_index=True)
                with c2:
                    st.markdown("**Performa klasifikasi**")
                    st.dataframe(ui.styled(result.performance), width="stretch", hide_index=True)

                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(plots.roc_plot(result.roc, result.auc), width="stretch")
                with c4:
                    st.plotly_chart(
                        plots.confusion_heatmap(result.confusion), width="stretch"
                    )

                st.subheader("Probabilitas prediksi")
                pred_df = df.loc[result.probabilities.index].copy()
                pred_df.insert(0, "Probabilitas", result.probabilities.to_numpy())
                ui.show_table(
                    pred_df.sort_values("Probabilitas", ascending=False),
                    "logistik_prediksi.csv",
                    height=360,
                )

                with st.expander("Ringkasan statsmodels lengkap"):
                    st.code(result.model.summary().as_text(), language="text")
