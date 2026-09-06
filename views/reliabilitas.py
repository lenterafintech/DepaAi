"""Halaman reliabilitas dan validitas instrumen pengukuran."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nalardata import formatting, plots, preprocessing, reliability as rb, ui

ui.butuh_fitur("instrumen")
ui.page_setup(
    "Reliabilitas & Validitas",
    "Instrumen",
    "Memeriksa apakah butir-butir kuesioner benar-benar mengukur konstruk yang sama, "
    "dan apakah antar konstruk cukup terbedakan — prasyarat sebelum CFA dan SEM.",
)
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "reliabilitas dan validitas",
    "**Alpha Cronbach** mengukur konsistensi internal butir; nilai ≥ 0,70 lazim "
    "diterima. **Omega McDonald** dan **composite reliability (CR)** menghitung hal "
    "serupa dari muatan faktor, sehingga tidak mengasumsikan seluruh butir sama "
    "kuatnya. **AVE** adalah rata-rata ragam butir yang terserap konstruk; ≥ 0,50 "
    "berarti konstruk menjelaskan lebih banyak daripada galat pengukurannya. "
    "**Fornell-Larcker** menuntut akar AVE tiap konstruk melampaui korelasinya "
    "dengan konstruk lain — bila tidak, dua konstruk sebetulnya mengukur hal sama.",
)

numerik = preprocessing.numeric_columns(df)
if len(numerik) < 2:
    st.error("Halaman ini memerlukan minimal 2 kolom numerik.")
    st.stop()

tab_satu, tab_banyak = st.tabs(["Satu konstruk", "Beberapa konstruk"])

# --------------------------------------------------------------------------- #
# Satu konstruk: alpha dan rincian butir
# --------------------------------------------------------------------------- #

with tab_satu:
    tebakan = rb.tebak_konstruk(numerik)
    bawaan = next(iter(tebakan.values()), numerik[: min(4, len(numerik))])
    butir = st.multiselect(
        "Butir yang membentuk satu konstruk",
        numerik,
        default=[b for b in bawaan if b in numerik],
        key="rel_butir",
    )
    if len(butir) < 2:
        st.info("Pilih minimal 2 butir.")
    else:
        try:
            hasil = rb.alpha_cronbach(df[butir])
        except ValueError as exc:
            st.error(str(exc))
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Alpha Cronbach", f"{hasil.alpha:.3f}".replace(".", ","))
            m2.metric(
                "Spearman-Brown",
                formatting.num(hasil.spearman_brown, 3),
                help=(
                    "Reliabilitas belah-dua: butir dibelah ganjil-genap, korelasi "
                    "antar belahan dikoreksi ke panjang tes penuh. Nilainya lazim "
                    "berdekatan dengan alpha."
                ),
            )
            m3.metric("Jumlah butir", hasil.n_item)
            m4.metric("Kesimpulan", hasil.interpretasi())
            if np.isfinite(hasil.korelasi_belahan):
                st.caption(
                    "Korelasi mentah antar kedua belahan "
                    f"{formatting.num(hasil.korelasi_belahan, 3)}; setelah koreksi "
                    f"Spearman-Brown menjadi {formatting.num(hasil.spearman_brown, 3)} "
                    "karena tiap belahan hanya separuh panjang instrumen."
                )

            muatan = rb.muatan_faktor_tunggal(df[butir])
            cr, ave = rb.cr_ave(muatan)
            n1, n2, n3 = st.columns(3)
            n1.metric("Omega McDonald", f"{rb.omega_mcdonald(muatan):.3f}".replace(".", ","))
            n2.metric("Composite reliability", f"{cr:.3f}".replace(".", ","))
            n3.metric("AVE", f"{ave:.3f}".replace(".", ","))

            st.subheader("Statistik per butir")
            tabel = hasil.item.copy()
            tabel["Muatan faktor"] = muatan.reindex(tabel["Butir"]).to_numpy()
            tabel["Keputusan"] = [
                "Pertahankan"
                if (r >= 0.3 and abs(l) >= rb.AMBANG_MUATAN and a <= hasil.alpha)
                else "Tinjau ulang"
                for r, l, a in zip(
                    tabel["Korelasi item-total"],
                    tabel["Muatan faktor"],
                    tabel["Alpha jika dibuang"],
                )
            ]
            ui.show_table(
            tabel,
            "reliabilitas_butir.csv",
            bagian="Reliabilitas",
            judul="Statistik per butir",
        )
            ui.interpretation(
                "Butir yang sehat memiliki korelasi item-total ≥ 0,30 dan muatan ≥ 0,50, "
                "serta tidak menaikkan alpha bila dibuang. Kolom 'Alpha jika dibuang' "
                "yang lebih tinggi daripada alpha keseluruhan menandakan butir tersebut "
                "justru mengganggu konsistensi konstruk."
            )

            bermasalah = hasil.butir_bermasalah()
            if bermasalah:
                st.warning(
                    "Perlu ditinjau: " + ", ".join(bermasalah) + ". Pertimbangkan "
                    "memperbaiki kalimat butirnya atau mengeluarkannya dari konstruk."
                )

            st.plotly_chart(
                plots.loadings_heatmap(
                    muatan.to_frame("Faktor"), "Muatan Butir pada Konstruk"
                ),
                width="stretch",
            )

# --------------------------------------------------------------------------- #
# Beberapa konstruk: CR, AVE, dan validitas diskriminan
# --------------------------------------------------------------------------- #

with tab_banyak:
    st.caption(
        "Kelompokkan butir menjadi beberapa konstruk. Bila butir dinamai berpola "
        "(KUAL1, KUAL2, …), pengelompokan awal ditebak otomatis dari awalannya."
    )
    tebakan = rb.tebak_konstruk(numerik)
    if not tebakan:
        tebakan = {"konstruk1": numerik[: min(3, len(numerik))]}
    if "rel_konstruk" not in st.session_state:
        st.session_state["rel_konstruk"] = list(tebakan)

    dipakai = st.multiselect(
        "Konstruk yang dianalisis",
        list(tebakan),
        default=[k for k in st.session_state["rel_konstruk"] if k in tebakan],
        key="rel_pilih_konstruk",
    )

    konstruk: dict[str, list[str]] = {}
    for nama in dipakai:
        pilihan = st.multiselect(
            f"Butir konstruk {nama}",
            numerik,
            default=[b for b in tebakan[nama] if b in numerik],
            key=f"rel_butir_{nama}",
        )
        if len(pilihan) >= 2:
            konstruk[nama] = pilihan

    if len(konstruk) < 2:
        st.info("Pilih minimal 2 konstruk yang masing-masing berisi minimal 2 butir.")
    else:
        try:
            hasil = rb.analisis_konstruk(df, konstruk)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.subheader("Reliabilitas dan validitas konvergen")
            ui.show_table(rb.tabel_konstruk(hasil), "reliabilitas_konstruk.csv")
            ui.interpretation(
                "Konstruk dinyatakan memenuhi bila alpha dan CR ≥ 0,70 serta AVE ≥ 0,50. "
                "AVE di bawah 0,50 berarti galat pengukuran lebih besar daripada ragam "
                "yang benar-benar dijelaskan konstruk."
            )

            st.subheader("Validitas diskriminan (Fornell-Larcker)")
            matriks = rb.fornell_larcker(df, hasil)
            tampil = matriks.reset_index().rename(columns={"index": "Konstruk"})
            ui.show_table(tampil, "fornell_larcker.csv")
            st.caption(
                "Angka pada diagonal adalah akar AVE; di bawahnya korelasi antar "
                "konstruk. Validitas diskriminan terpenuhi bila diagonal melampaui "
                "seluruh korelasi pada baris dan kolomnya."
            )
            periksa = rb.periksa_diskriminan(df, hasil)
            ui.show_table(periksa, "diskriminan_konstruk.csv")

            gagal = periksa[periksa["Keputusan"] == "Tidak terpenuhi"]
            if not gagal.empty:
                st.warning(
                    "Pasangan berikut belum terbedakan: "
                    + ", ".join(gagal["Pasangan"])
                    + ". Kedua konstruk kemungkinan mengukur hal yang sama — "
                    "pertimbangkan menggabungkannya atau merevisi butirnya."
                )

            st.subheader("Validitas diskriminan (HTMT)")
            st.caption(
                "Rasio Heterotrait-Monotrait membandingkan korelasi antar butir dari "
                "konstruk berbeda dengan korelasi antar butir di dalam konstruk yang "
                "sama. Henseler dkk. (2015) melaporkannya lebih peka daripada "
                "Fornell-Larcker; keduanya disediakan karena penelaah berbeda meminta "
                "yang berbeda."
            )
            try:
                tabel_htmt = rb.htmt(df, konstruk)
            except ValueError as galat_htmt:
                st.info(str(galat_htmt))
            else:
                ui.show_table(
                    tabel_htmt,
                    "htmt.csv",
                    bagian="Reliabilitas",
                    judul="Validitas diskriminan (HTMT)",
                )
                lewat = tabel_htmt[tabel_htmt["Keputusan (0,85)"] == "Tidak terpenuhi"]
                if lewat.empty:
                    st.success(
                        "Seluruh pasangan konstruk berada di bawah 0,85.",
                        icon=":material/check_circle:",
                    )
                else:
                    longgar = lewat[lewat["Keputusan (0,90)"] == "Terpenuhi"]
                    pesan = "Pasangan melewati ambang ketat 0,85: " + ", ".join(
                        f"{b['Konstruk A']}–{b['Konstruk B']}" for _, b in lewat.iterrows()
                    )
                    if not longgar.empty:
                        pesan += (
                            ". Sebagian masih di bawah ambang longgar 0,90, yang dapat "
                            "diterima bila kedua konstruk memang berdekatan maknanya."
                        )
                    st.warning(pesan, icon=":material/warning:")

            st.subheader("Skor konstruk")
            skor = rb.skor_konstruk(df, hasil)
            st.caption(
                "Rata-rata butir tiap konstruk. Skor ini dapat disimpan ke data aktif "
                "untuk dipakai pada regresi, analisis jalur, atau MANOVA."
            )
            ui.show_table(skor.head(50), "skor_konstruk.csv", height=320)
            if st.button("Simpan skor konstruk ke data aktif", key="simpan_skor"):
                diperbarui = df.copy()
                for nama in skor.columns:
                    diperbarui[f"skor_{nama.lower()}"] = skor[nama]
                ui.set_dataset(
                    diperbarui, st.session_state.get(ui.NAME_KEY, "data") + " (+skor)"
                )
                st.success(
                    "Skor konstruk ditambahkan sebagai kolom baru pada data aktif."
                )
