"""Halaman analisis korelasi kanonik."""

from __future__ import annotations

import streamlit as st

from lentera_mva import cca, plots, preprocessing, ui

ui.page_setup("Korelasi Kanonik", "🔀")
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "korelasi kanonik",
    "Korelasi kanonik mengukur hubungan antara **dua gugus variabel** sekaligus, "
    "misalnya gugus kapasitas ekonomi (pendapatan, tabungan, plafon) dengan gugus "
    "perilaku pembayaran (skor kredit, keterlambatan, riwayat lunas). Metode ini "
    "membentuk pasangan variat kanonik yang korelasinya semaksimal mungkin.",
)

numeric_cols = preprocessing.numeric_columns(df)
if len(numeric_cols) < 4:
    st.error("Korelasi kanonik memerlukan minimal 4 variabel numerik (2 di tiap gugus).")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    x_vars = st.multiselect(
        "Gugus X", numeric_cols, default=numeric_cols[: min(3, len(numeric_cols))], key="cca_x"
    )
with c2:
    remaining = [c for c in numeric_cols if c not in x_vars]
    y_vars = st.multiselect(
        "Gugus Y", remaining, default=remaining[: min(3, len(remaining))], key="cca_y"
    )

if not x_vars or not y_vars:
    st.info("Isi kedua gugus variabel untuk melanjutkan.")
    st.stop()

try:
    result = cca.run_cca(df, x_vars, y_vars)
except Exception as exc:  # noqa: BLE001 - kesalahan model ditampilkan ke pengguna
    st.error(f"Analisis gagal: {exc}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Korelasi kanonik tertinggi", f"{result.correlations[0]:.4f}")
m2.metric("R² kanonik pertama", f"{result.correlations[0] ** 2:.4f}")
m3.metric("Observasi terpakai", result.n)

tab_sum, tab_load, tab_scores, tab_red = st.tabs(
    ["Ringkasan & Signifikansi", "Muatan Kanonik", "Skor Variat", "Redundansi"]
)

with tab_sum:
    ui.show_table(result.summary(), "cca_ringkasan.csv")
    st.markdown("**Uji signifikansi fungsi kanonik (Wilks' lambda)**")
    ui.show_table(result.significance, "cca_signifikansi.csv")
    ui.interpretation(
        "Baris pertama menguji apakah seluruh fungsi kanonik secara bersama signifikan. "
        "Baris berikutnya menguji sisa fungsi setelah fungsi sebelumnya dikeluarkan. "
        "Hanya fungsi dengan p < 0,05 yang layak ditafsirkan."
    )

with tab_load:
    st.markdown("**Muatan kanonik gugus X** — korelasi variabel asli dengan variat kanoniknya")
    st.plotly_chart(
        plots.loadings_heatmap(result.x_loadings, "Muatan Kanonik Gugus X"), width="stretch"
    )
    ui.show_table(result.x_loadings.reset_index(names="Variabel"), "cca_muatan_x.csv")

    st.markdown("**Muatan kanonik gugus Y**")
    st.plotly_chart(
        plots.loadings_heatmap(result.y_loadings, "Muatan Kanonik Gugus Y"), width="stretch"
    )
    ui.show_table(result.y_loadings.reset_index(names="Variabel"), "cca_muatan_y.csv")
    ui.interpretation(
        "Variabel dengan |muatan| ≥ 0,3 dianggap berkontribusi pada variat kanonik. "
        "Bandingkan pola muatan di kedua gugus untuk memahami hubungan apa yang "
        "sebenarnya ditangkap oleh pasangan variat tersebut."
    )

    with st.expander("Bobot kanonik (koefisien terstandar)"):
        ui.show_table(result.x_weights.reset_index(names="Variabel"), "cca_bobot_x.csv")
        ui.show_table(result.y_weights.reset_index(names="Variabel"), "cca_bobot_y.csv")

with tab_scores:
    func = st.selectbox("Fungsi kanonik", result.x_scores.columns)
    scores = result.x_scores[[func]].rename(columns={func: f"Variat X ({func})"})
    scores[f"Variat Y ({func})"] = result.y_scores[func]
    color_options = [None] + [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 12]
    color_col = st.selectbox(
        "Warnai berdasarkan",
        color_options,
        format_func=lambda v: "Tanpa pewarnaan" if v is None else v,
    )
    color = df.loc[scores.index, color_col] if color_col else None
    st.plotly_chart(
        plots.scatter_2d(
            scores,
            f"Variat X ({func})",
            f"Variat Y ({func})",
            color,
            f"Hubungan Variat Kanonik {func} (r = {result.correlations[list(result.x_scores.columns).index(func)]:.3f})",
        ),
        width="stretch",
    )
    ui.show_table(scores.reset_index(names="Indeks"), "cca_skor.csv", height=320)

with tab_red:
    ui.show_table(result.redundancy, "cca_redundansi.csv")
    ui.interpretation(
        "Indeks redundansi menunjukkan berapa persen varians satu gugus yang dapat "
        "dijelaskan oleh variat kanonik gugus lawannya. Korelasi kanonik tinggi belum "
        "tentu bermakna praktis bila redundansinya kecil."
    )
