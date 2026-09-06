"""Halaman uji non-parametrik: alternatif ketika asumsi normalitas tidak terpenuhi."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import formatting, nonparametrik as npar, parametrik as par
from nalardata import preprocessing, ui

ui.butuh_fitur("nonparametrik")
ui.page_setup(
    "Uji Non-parametrik",
    "Pemodelan & Uji Beda",
    "Uji beda dan uji kaitan yang tidak menuntut sebaran normal — dipakai pada data "
    "ordinal, sampel kecil, atau sebaran yang jelas menceng.",
)
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "Kapan uji non-parametrik dipakai",
    "Uji-t, ANOVA, dan korelasi Pearson menuntut data yang sebarannya mendekati "
    "normal. Bila tuntutan itu tidak terpenuhi, nilai p-nya menyesatkan. Uji di "
    "halaman ini bekerja pada **peringkat**, bukan nilai mentahnya, sehingga tetap "
    "sah tanpa asumsi sebaran. Harganya: uji peringkat sedikit kurang peka ketika "
    "data sebenarnya normal, dan kesimpulannya berbicara tentang median atau "
    "kecenderungan, bukan rata-rata.",
)

numerik = preprocessing.numeric_columns(df)
kategorik = [c for c in df.columns if df[c].nunique(dropna=True) <= 20]

if not numerik:
    st.error("Halaman ini memerlukan minimal satu kolom numerik.")
    st.stop()

# Penetapan variabel dari Pemandu Uji, bila pengguna tiba lewat tombol
# "Konfirmasi dan siapkan halamannya". Kosong bila ia membuka halaman ini sendiri.
dipandu = ui.konfigurasi_pemandu()
if dipandu.get("metode"):
    st.success(
        f"Disiapkan dari Pemandu Uji: **{dipandu['metode']}**. Pilihan di bawah sudah "
        "terisi sesuai variabel yang Anda tentukan di sana, dan tetap dapat diubah.",
        icon=":material/explore:",
    )

perlu, alasan = npar.perlu_nonparametrik(df, numerik[: min(6, len(numerik))])
(st.warning if perlu else st.info)(
    alasan,
    icon=":material/rule:" if perlu else ":material/check_circle:",
)


def lapor(hasil: npar.HasilUji, kunci: str) -> None:
    """Pelaporan seragam untuk seluruh uji: angka, ukuran efek, dan kalimat siap salin."""
    for catatan in hasil.catatan:
        st.warning(catatan, icon=":material/info:")

    k1, k2, k3, k4 = st.columns(4)
    desimal = 0 if float(hasil.statistik).is_integer() else 3
    k1.metric(hasil.label_statistik, formatting.num(hasil.statistik, desimal))
    k2.metric("Nilai p", formatting.num(hasil.p_value, 4))
    k3.metric(hasil.efek_nama or "Ukuran efek", formatting.num(hasil.efek_nilai, 3))
    k4.metric("n", formatting.num(hasil.n))

    if hasil.signifikan:
        st.success(
            f"Perbedaan bermakna secara statistik (p < 0,05). Ukuran efek tergolong "
            f"**{hasil.efek_tafsir}**.",
            icon=":material/trending_up:",
        )
    else:
        st.info(
            "Tidak ada bukti perbedaan yang bermakna pada taraf 5%. Ini bukan bukti "
            "bahwa keduanya sama, melainkan bahwa data yang ada belum cukup "
            "menunjukkan perbedaan.",
            icon=":material/horizontal_rule:",
        )

    if hasil.keterangan:
        st.caption(hasil.keterangan)
    if hasil.tabel is not None and not hasil.tabel.empty:
        ui.show_table(hasil.tabel, f"{kunci}.csv")

    st.markdown("**Kalimat siap salin**")
    st.code(hasil.ringkas(), language=None, wrap_lines=True)
    if hasil.padanan != "—":
        st.caption(f"Padanan parametriknya: {hasil.padanan}.")


tab_bebas, tab_banyak, tab_pasangan, tab_kategorik, tab_korelasi = st.tabs(
    [
        "Dua kelompok bebas",
        "Lebih dari dua kelompok",
        "Sampel berpasangan",
        "Data kategorik",
        "Korelasi peringkat",
    ]
)

# --------------------------------------------------------------------------- #
# Dua kelompok bebas
# --------------------------------------------------------------------------- #

with tab_bebas:
    st.caption(
        "Membandingkan dua kelompok yang berisi orang atau unit berbeda — misalnya "
        "nasabah lancar dan nasabah menunggak."
    )
    dua = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if not dua:
        st.info("Tidak ada kolom yang berisi tepat dua kategori.")
    else:
        kol1, kol2, kol3 = st.columns(3)
        nilai = kol1.selectbox(
            "Variabel yang dibandingkan",
            numerik,
            index=ui.indeks_pilihan(numerik, dipandu.get("outcome")),
            key="np_mw_nilai",
        )
        grup = kol2.selectbox(
            "Penanda kelompok",
            dua,
            index=ui.indeks_pilihan(dua, dipandu.get("kelompok")),
            key="np_mw_grup",
        )
        pilihan_dua = ["t_bebas", "t_welch", "mann_whitney", "ks2", "mood"]
        peta_dua = {
            "Uji-t sampel bebas": "t_bebas",
            "Uji-t Welch": "t_welch",
            "Mann-Whitney U": "mann_whitney",
        }
        jenis = kol3.selectbox(
            "Uji",
            pilihan_dua,
            index=ui.indeks_pilihan(
                pilihan_dua, peta_dua.get(dipandu.get("metode", "")), bawaan=2
            ),
            format_func=lambda k: {
                "t_bebas": "Uji-t sampel bebas (ragam sama)",
                "t_welch": "Uji-t Welch (ragam boleh berbeda)",
                "mann_whitney": "Mann-Whitney U (letak sebaran)",
                "ks2": "Kolmogorov-Smirnov (bentuk sebaran)",
                "mood": "Median Mood (posisi terhadap median)",
            }[k],
            key="np_mw_jenis",
        )
        tingkat = sorted(df[grup].dropna().unique())
        a = df.loc[df[grup] == tingkat[0], nilai]
        b = df.loc[df[grup] == tingkat[1], nilai]
        try:
            if jenis in {"t_bebas", "t_welch"}:
                uji = par.uji_t_bebas(
                    df[nilai], df[grup].astype(str), ragam_sama=jenis == "t_bebas"
                )
            else:
                uji = {
                    "mann_whitney": npar.mann_whitney,
                    "ks2": npar.kolmogorov_smirnov_2,
                    "mood": npar.mood_median,
                }[jenis](a, b, str(tingkat[0]), str(tingkat[1]))
        except ValueError as exc:
            st.error(str(exc))
        else:
            lapor(uji, f"uji_{jenis}")

# --------------------------------------------------------------------------- #
# Lebih dari dua kelompok
# --------------------------------------------------------------------------- #

with tab_banyak:
    st.caption(
        "Membandingkan tiga kelompok bebas atau lebih sekaligus, lalu menunjuk "
        "pasangan mana yang benar-benar berbeda."
    )
    banyak = [c for c in kategorik if 3 <= df[c].nunique(dropna=True) <= 12]
    if not banyak:
        st.info("Tidak ada kolom berisi 3 sampai 12 kategori.")
    else:
        kol1, kol2, kol3 = st.columns(3)
        nilai_k = kol1.selectbox(
            "Variabel yang dibandingkan",
            numerik,
            index=ui.indeks_pilihan(numerik, dipandu.get("outcome")),
            key="np_kw_nilai",
        )
        grup_k = kol2.selectbox(
            "Penanda kelompok",
            banyak,
            index=ui.indeks_pilihan(banyak, dipandu.get("kelompok")),
            key="np_kw_grup",
        )
        pilihan_banyak = ["anova", "welch_anova", "kruskal"]
        peta_banyak = {
            "One-Way ANOVA": "anova",
            "Welch ANOVA": "welch_anova",
            "Kruskal-Wallis": "kruskal",
        }
        jenis_k = kol3.selectbox(
            "Uji",
            pilihan_banyak,
            index=ui.indeks_pilihan(
                pilihan_banyak, peta_banyak.get(dipandu.get("metode", "")), bawaan=2
            ),
            format_func=lambda k: {
                "anova": "One-Way ANOVA (ragam sama)",
                "welch_anova": "Welch ANOVA (ragam boleh berbeda)",
                "kruskal": "Kruskal-Wallis (tanpa asumsi normalitas)",
            }[k],
            key="np_kw_jenis",
        )
        try:
            uji_k = {
                "anova": par.anova_satu_arah,
                "welch_anova": par.welch_anova,
                "kruskal": npar.kruskal_wallis,
            }[jenis_k](df[nilai_k], df[grup_k])
        except ValueError as exc:
            st.error(str(exc))
        else:
            lapor(uji_k, f"uji_{jenis_k}")

            # Uji lanjutan mengikuti uji utamanya: Tukey mengandaikan ragam sama,
            # Games-Howell tidak, dan Dunn bekerja atas peringkat.
            if jenis_k == "anova":
                st.subheader("Uji lanjutan Tukey HSD")
                ui.show_table(par.tukey(df[nilai_k], df[grup_k]), "tukey.csv")
                ui.interpretation(
                    "Tukey HSD membandingkan seluruh pasangan sekaligus sambil menjaga "
                    "peluang salah tolak keseluruhan tetap 5 persen. Selang yang tidak "
                    "melewati nol berarti pasangan itu berbeda."
                )
            elif jenis_k == "welch_anova":
                st.subheader("Uji lanjutan Games-Howell")
                ui.show_table(par.games_howell(df[nilai_k], df[grup_k]), "games_howell.csv")
                ui.interpretation(
                    "Games-Howell dipakai berpasangan dengan Welch ANOVA karena ia juga "
                    "tidak menuntut ragam antar kelompok sama."
                )
            else:
                st.subheader("Uji lanjutan Dunn")
                koreksi = st.radio(
                    "Koreksi pembandingan ganda",
                    ["holm", "bonferroni", "tanpa"],
                    format_func=lambda k: {
                        "holm": "Holm (disarankan)",
                        "bonferroni": "Bonferroni (paling ketat)",
                        "tanpa": "Tanpa koreksi",
                    }[k],
                    horizontal=True,
                    key="np_dunn_koreksi",
                )
                if koreksi == "tanpa":
                    st.warning(
                        "Tanpa koreksi, membandingkan banyak pasangan sekaligus menaikkan "
                        "peluang menemukan perbedaan yang sebenarnya tidak ada.",
                        icon=":material/warning:",
                    )
                ui.show_table(npar.dunn(df[nilai_k], df[grup_k], koreksi), "dunn.csv")

# --------------------------------------------------------------------------- #
# Sampel berpasangan
# --------------------------------------------------------------------------- #

with tab_pasangan:
    st.caption(
        "Dua atau lebih pengukuran pada **subjek yang sama** — misalnya nilai sebelum "
        "dan sesudah pelatihan pada orang yang sama."
    )
    if len(numerik) < 2:
        st.info("Diperlukan minimal 2 kolom numerik.")
    else:
        mode = st.radio(
            "Jumlah pengukuran",
            ["dua", "banyak"],
            format_func=lambda k: "Dua pengukuran" if k == "dua" else "Tiga pengukuran atau lebih",
            horizontal=True,
            key="np_pasangan_mode",
        )
        if mode == "dua":
            kol1, kol2, kol3 = st.columns(3)
            p1 = kol1.selectbox("Pengukuran pertama", numerik, key="np_w_a")
            p2 = kol2.selectbox(
                "Pengukuran kedua",
                [c for c in numerik if c != p1],
                key="np_w_b",
            )
            jenis_p = kol3.selectbox(
                "Uji",
                ["wilcoxon", "tanda"],
                format_func=lambda k: {
                    "wilcoxon": "Wilcoxon peringkat bertanda",
                    "tanda": "Uji tanda (hanya arah perubahan)",
                }[k],
                key="np_w_jenis",
            )
            try:
                uji_p = (npar.wilcoxon if jenis_p == "wilcoxon" else npar.uji_tanda)(
                    df[p1], df[p2], p1, p2
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                lapor(uji_p, f"uji_{jenis_p}")
        else:
            kolom_f = st.multiselect(
                "Pengukuran berulang (minimal 3 kolom)",
                numerik,
                default=numerik[: min(3, len(numerik))],
                key="np_friedman",
            )
            if len(kolom_f) < 3:
                st.info("Pilih minimal 3 kolom.")
            else:
                try:
                    uji_f = npar.friedman(df, kolom_f)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    lapor(uji_f, "uji_friedman")

# --------------------------------------------------------------------------- #
# Data kategorik
# --------------------------------------------------------------------------- #

with tab_kategorik:
    st.caption(
        "Menguji apakah dua variabel kategorik saling berkaitan — misalnya apakah "
        "wilayah berkaitan dengan status gagal bayar."
    )
    if len(kategorik) < 2:
        st.info("Diperlukan minimal 2 kolom kategorik.")
    else:
        kol1, kol2 = st.columns(2)
        ka = kol1.selectbox("Variabel baris", kategorik, key="np_chi_a")
        kb = kol2.selectbox(
            "Variabel kolom", [c for c in kategorik if c != ka], key="np_chi_b"
        )
        try:
            uji_c = npar.chi_square(df[ka], df[kb], ka, kb)
        except ValueError as exc:
            st.error(str(exc))
        else:
            lapor(uji_c, "uji_chi_square")
            try:
                uji_f2 = npar.fisher_eksak(df[ka], df[kb], ka, kb)
            except ValueError:
                pass  # bukan tabel 2x2, uji Fisher memang tidak berlaku
            else:
                st.subheader("Uji eksak Fisher")
                st.caption(
                    "Tabel 2×2 dapat diuji secara eksak, sehingga hasilnya sah "
                    "sekalipun frekuensinya kecil."
                )
                lapor(uji_f2, "uji_fisher")

# --------------------------------------------------------------------------- #
# Korelasi peringkat
# --------------------------------------------------------------------------- #

with tab_korelasi:
    st.caption(
        "Mengukur keeratan hubungan searah tanpa menuntut hubungan yang lurus maupun "
        "sebaran normal."
    )
    metode = st.radio(
        "Metode",
        ["spearman", "kendall"],
        format_func=lambda k: "Spearman rho" if k == "spearman" else "Kendall tau",
        horizontal=True,
        key="np_kor_metode",
    )
    pilihan = st.multiselect(
        "Variabel",
        numerik,
        default=numerik[: min(5, len(numerik))],
        key="np_kor_var",
    )
    if len(pilihan) < 2:
        st.info("Pilih minimal 2 variabel.")
    else:
        koefisien, nilai_p = npar.matriks_peringkat(df, pilihan, metode)
        st.subheader("Matriks korelasi")
        ui.show_table(koefisien.round(3).reset_index(names="Variabel"), "korelasi_peringkat.csv")
        st.subheader("Pasangan terkuat")
        baris = []
        for i, a in enumerate(pilihan):
            for b in pilihan[i + 1 :]:
                baris.append(
                    {
                        "Variabel A": a,
                        "Variabel B": b,
                        "Koefisien": float(koefisien.loc[a, b]),
                        "p-value": float(nilai_p.loc[a, b]),
                        "Kekuatan": npar._tafsir_r(float(koefisien.loc[a, b])),
                        "Keputusan": (
                            "Bermakna" if float(nilai_p.loc[a, b]) < 0.05 else "Tidak bermakna"
                        ),
                    }
                )
        tabel = pd.DataFrame(baris).sort_values("Koefisien", key=abs, ascending=False)
        ui.show_table(tabel.reset_index(drop=True), "pasangan_peringkat.csv")

ui.interpretation(
    "Uji non-parametrik menyimpulkan tentang median dan kecenderungan peringkat, "
    "bukan rata-rata. Saat melaporkannya, sebutkan uji yang dipakai, statistik uji, "
    "nilai p, dan ukuran efeknya — nilai p saja tidak memberi tahu seberapa besar "
    "perbedaannya."
)
