"""Halaman MANOVA satu jalur."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import assumptions, descriptive, mancova, manova, plots, preprocessing, ui


def render(df, kamus, penelitian) -> None:
    if not ui.butuh_fitur("uji_beda"):
        return

    ui.method_note(
        "MANOVA",
        "MANOVA menguji apakah beberapa variabel dependen **sekaligus** berbeda antar "
        "kelompok. Dibanding menjalankan banyak ANOVA terpisah, MANOVA memperhitungkan "
        "korelasi antar variabel dependen dan menjaga tingkat kesalahan tipe I.",
    )

    dipandu = ui.konfigurasi_pemandu()
    ui.banner_dipandu(dipandu)

    mode = st.radio(
        "Desain",
        ["Antar kelompok (MANOVA)", "Pengukuran berulang (dalam-subjek)"],
        horizontal=True,
        key="manova_mode",
    )
    st.divider()
    if mode == "Antar kelompok (MANOVA)":
        _render_antar_kelompok(df, dipandu)
    else:
        _render_berulang(df)


def _render_antar_kelompok(df, dipandu: dict) -> None:
    factor = ui.group_selector(
        df, "Variabel faktor (kelompok)", key="manova_factor", default=dipandu.get("kelompok")
    )
    numeric_cols = [c for c in preprocessing.numeric_columns(df) if c != factor]
    default_dv = [c for c in dipandu.get("prediktor", []) if c in numeric_cols] or numeric_cols[
        : min(3, len(numeric_cols))
    ]
    dependents = st.multiselect(
        "Variabel dependen numerik (minimal 2)",
        numeric_cols,
        default=default_dv,
        key="manova_dv",
    )

    if len(dependents) < 2:
        st.info("Pilih minimal 2 variabel dependen.")
        return

    try:
        result = manova.run_manova(df, dependents, factor)
    except Exception as exc:  # noqa: BLE001 - kesalahan model ditampilkan ke pengguna
        st.error(f"MANOVA gagal dijalankan: {exc}")
        return

    tab_main, tab_uni, tab_cov, tab_desc, tab_assum = st.tabs(
        ["Uji Multivariat", "ANOVA Lanjutan", "MANCOVA", "Deskriptif Kelompok", "Asumsi"]
    )

    with tab_main:
        st.subheader("Statistik uji multivariat")
        ui.show_table(
                result.multivariate,
                "manova_multivariat.csv",
                bagian="MANOVA",
                judul="Uji multivariat",
            )
        if result.multivariate["p-value"].min() < 0.05:
            st.success(result.conclusion())
        else:
            st.info(result.conclusion())
        ui.interpretation(
            "Empat statistik dilaporkan: Wilks' lambda (paling umum), Pillai's trace "
            "(paling tahan pelanggaran asumsi), Hotelling-Lawley, dan Roy's greatest root. "
            "Bila keempatnya sepakat, kesimpulan makin meyakinkan."
        )

        if df[factor].nunique() == 2:
            st.subheader("Hotelling's T² (perbandingan dua kelompok)")
            try:
                ui.show_table(
                    manova.hotelling_t2(df, dependents, factor), "hotelling_t2.csv"
                )
            except ValueError as exc:
                st.error(str(exc))

    with tab_uni:
        st.subheader("ANOVA univariat per variabel dependen")
        ui.show_table(result.univariate, "manova_univariat.csv")
        ui.interpretation(
            "Uji lanjutan ini menunjukkan variabel dependen mana yang benar-benar berbeda "
            "antar kelompok. Eta-squared adalah ukuran besarnya efek: 0,01 kecil, 0,06 "
            "sedang, 0,14 besar."
        )
        var = st.selectbox("Visualisasikan variabel", dependents)
        st.plotly_chart(plots.box_by_group(df, var, factor), width="stretch")

    with tab_cov:
        st.caption(
            "MANCOVA menjawab pertanyaan yang tidak dapat dijawab MANOVA: apakah kelompok "
            "tetap berbeda setelah pengaruh variabel lain disingkirkan. Kovariat lazimnya "
            "variabel yang sudah berbeda sejak awal antar kelompok — misalnya usia."
        )
        kandidat_kovariat = [c for c in numeric_cols if c not in dependents]
        kovariat = st.multiselect(
            "Kovariat yang dikendalikan",
            kandidat_kovariat,
            default=kandidat_kovariat[:1],
            key="mancova_kovariat",
        )
        if not kovariat:
            st.info("Pilih minimal satu kovariat. Tanpa kovariat, gunakan tab Uji Multivariat.")
        else:
            try:
                hasil_cov = mancova.run_mancova(df, dependents, factor, kovariat)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.subheader("Uji multivariat setelah kovariat dikendalikan")
                ui.show_table(
                hasil_cov.multivariate,
                "mancova_multivariat.csv",
                bagian="MANCOVA",
                judul="Uji multivariat terkoreksi",
            )
                if hasil_cov.signifikan():
                    st.success(hasil_cov.conclusion(), icon=":material/check_circle:")
                else:
                    st.info(hasil_cov.conclusion(), icon=":material/info:")

                st.subheader("Pengaruh kovariat")
                ui.show_table(hasil_cov.pengaruh_kovariat, "mancova_kovariat.csv")
                ui.interpretation(
                    "Kovariat yang signifikan memang perlu dikendalikan; bila tidak "
                    "signifikan, ia hanya mengurangi derajat bebas tanpa memperbaiki apa "
                    "pun — pertimbangkan mengeluarkannya dari model."
                )

                st.subheader("ANCOVA univariat")
                ui.show_table(hasil_cov.univariate, "mancova_univariat.csv")

                st.subheader("Rata-rata terkoreksi")
                ui.show_table(mancova.bandingkan_rata(hasil_cov), "mancova_rata.csv")
                ui.interpretation(
                    "Rata-rata terkoreksi adalah rata-rata tiap kelompok seandainya seluruh "
                    "kelompok memiliki nilai kovariat yang sama. Selisih besar terhadap "
                    "rata-rata mentah menandakan perbedaan antar kelompok sebagian berasal "
                    "dari kovariat, bukan dari kelompoknya sendiri."
                )

                st.subheader("Asumsi kemiringan regresi homogen")
                ui.show_table(hasil_cov.homogenitas_slope, "mancova_slope.csv")
                if not hasil_cov.slope_homogen():
                    st.warning(
                        "Kemiringan regresi kovariat berbeda antar kelompok, sehingga "
                        "asumsi MANCOVA dilanggar. Artinya pengaruh kovariat sendiri "
                        "bergantung pada kelompok — pertimbangkan analisis moderasi "
                        "alih-alih memaksakan MANCOVA."
                    )

    with tab_desc:
        ui.show_table(result.group_sizes, "manova_ukuran_kelompok.csv")
        st.markdown("**Rata-rata tiap variabel dependen per kelompok**")
        ui.show_table(result.group_means, "manova_rata_rata.csv")
        smallest = int(result.group_sizes["N"].min())
        if smallest <= len(dependents):
            st.warning(
                f"Kelompok terkecil hanya berisi {smallest} observasi, sementara ada "
                f"{len(dependents)} variabel dependen. Tambah data atau kurangi variabel "
                "dependen agar hasil stabil."
            )

    with tab_assum:
        subset = df[[*dependents, factor]].dropna()
        st.subheader("Normalitas multivariat (Mardia)")
        try:
            mardia = descriptive.mardia_test(subset[dependents])
        except ValueError as exc:
            st.error(str(exc))
        else:
            ui.show_table(mardia.to_frame(), "manova_mardia.csv")

        st.subheader("Homogenitas matriks kovarians (Box's M)")
        try:
            box = assumptions.box_m(subset[dependents], subset[factor])
        except ValueError as exc:
            st.error(str(exc))
        else:
            ui.show_table(box.to_frame(), "manova_boxm.csv")
            if not box.homogeneous:
                st.warning(
                    "Matriks kovarians tidak homogen. Gunakan Pillai's trace sebagai acuan "
                    "utama karena paling tahan terhadap pelanggaran asumsi ini."
                )

        st.subheader("Homogenitas ragam per variabel (Levene)")
        levene = assumptions.levene_by_variable(subset[dependents], subset[factor])
        if not levene.empty:
            ui.show_table(levene, "manova_levene.csv")


def _render_berulang(df) -> None:
    ui.method_note(
        "ANOVA pengukuran berulang",
        "Membandingkan lebih dari dua pengukuran pada **subjek yang sama** — misalnya "
        "skor sebelum, sesudah, dan tindak lanjut pelatihan. Karena pengukurannya "
        "berkorelasi (subjeknya sama), sphericity — ragam selisih antar semua pasangan "
        "kondisi yang seragam — harus diperiksa lebih dulu; bila dilanggar, derajat "
        "bebasnya dikoreksi (Greenhouse-Geisser atau Huynh-Feldt) sebelum nilai p dibaca.",
    )
    numeric_cols = preprocessing.numeric_columns(df)
    kondisi = st.multiselect(
        "Kolom kondisi/waktu (minimal 2, subjek yang sama pada tiap kolom)",
        numeric_cols,
        default=numeric_cols[: min(3, len(numeric_cols))],
        key="rm_kondisi",
    )
    if len(kondisi) < 2:
        st.info("Pilih minimal 2 kolom yang mewakili pengukuran berulang pada subjek yang sama.")
        return

    try:
        hasil = manova.run_repeated_measures(df, kondisi)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("Rata-rata tiap kondisi")
    ui.show_table(hasil.ringkasan, "rm_ringkasan.csv")

    st.subheader("Uji sphericity (Mauchly)")
    if not hasil.mauchly.berlaku:
        st.info("Hanya 2 kondisi: sphericity otomatis terpenuhi, tidak perlu diuji.")
    else:
        tabel_mauchly = pd.DataFrame(
            [
                {
                    "Mauchly's W": hasil.mauchly.w,
                    "Chi-square": hasil.mauchly.chi_square,
                    "df": hasil.mauchly.df,
                    "p-value": hasil.mauchly.p_value,
                    "Sphericity": "Terpenuhi" if hasil.mauchly.terpenuhi else "Dilanggar",
                }
            ]
        )
        ui.show_table(tabel_mauchly, "rm_mauchly.csv", bagian="MANOVA", judul="Uji sphericity Mauchly")

    st.subheader("ANOVA pengukuran berulang")
    st.caption("Baris pertama tanpa koreksi; gunakan baris terkoreksi bila sphericity dilanggar.")
    ui.show_table(hasil.anova.reset_index().rename(columns={"index": "Sumber"}), "rm_anova.csv")
    ui.show_table(
        hasil.terkoreksi,
        "rm_terkoreksi.csv",
        bagian="MANOVA",
        judul="ANOVA pengukuran berulang (terkoreksi)",
    )

    if hasil.mauchly.terpenuhi:
        st.success(hasil.kesimpulan(), icon=":material/check_circle:")
    else:
        st.warning(hasil.kesimpulan(), icon=":material/warning:")

    for catatan in hasil.catatan:
        st.caption(catatan)

    ui.interpretation(
        "Greenhouse-Geisser lebih konservatif (epsilon lebih kecil, cenderung lebih "
        "aman dipakai) dibanding Huynh-Feldt; laporkan salah satunya secara konsisten "
        "beserta nilai epsilonnya, bukan memilih yang kebetulan signifikan."
    )
