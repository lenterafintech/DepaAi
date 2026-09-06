"""Halaman regresi moderasi (MRA)."""

from __future__ import annotations

import streamlit as st

from nalardata import formatting, moderation as mo, plots, preprocessing, ui

ui.butuh_fitur("regresi")
ui.page_setup(
    "Regresi Moderasi (MRA)",
    "Pemodelan & Uji Beda",
    "Menguji apakah pengaruh sebuah faktor berubah menurut tingkat faktor lain — "
    "misalnya apakah pengaruh pendapatan terhadap skor kredit berbeda pada nasabah "
    "berpengalaman dan nasabah baru.",
)
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "regresi moderasi",
    "Model yang diuji adalah Y = X + M + (X × M). Bila koefisien suku interaksi "
    "signifikan, pengaruh X terhadap Y berbeda pada tingkat M yang berbeda. "
    "**Kemiringan sederhana** menunjukkan besar pengaruh X pada moderator rendah, "
    "rata-rata, dan tinggi. **Johnson-Neyman** melangkah lebih jauh: ia menghitung "
    "pada nilai moderator berapa tepatnya pengaruh X mulai atau berhenti signifikan, "
    "tanpa bergantung pada pilihan titik ±1 SD yang sifatnya sembarang.",
)

numerik = preprocessing.numeric_columns(df)
if len(numerik) < 3:
    st.error("Regresi moderasi memerlukan minimal 3 kolom numerik.")
    st.stop()

kol1, kol2, kol3 = st.columns(3)
y = kol1.selectbox("Variabel terikat (Y)", numerik, key="mra_y")
x = kol2.selectbox(
    "Prediktor (X)", [c for c in numerik if c != y], key="mra_x"
)
m = kol3.selectbox(
    "Moderator (M)", [c for c in numerik if c not in {y, x}], key="mra_m"
)
kontrol = st.multiselect(
    "Variabel kontrol (opsional)",
    [c for c in numerik if c not in {y, x, m}],
    key="mra_kontrol",
)
pusatkan = st.checkbox(
    "Pusatkan variabel (mean centering)",
    value=True,
    help=(
        "Dianjurkan. Tanpa pemusatan, koefisien X bermakna 'pengaruh saat M bernilai "
        "nol' — nilai yang sering tidak pernah ada dalam data."
    ),
    key="mra_pusat",
)

try:
    hasil = mo.regresi_moderasi(df, y, x, m, kontrol, pusatkan)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

perubahan = hasil.uji_perubahan()
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² model", formatting.num(hasil.model.rsquared, 3))
m2.metric("ΔR² interaksi", formatting.num(perubahan["Delta R2"], 3))
m3.metric("F perubahan", formatting.num(perubahan["F"], 2))
m4.metric("p interaksi", formatting.pval_ringkas(perubahan["p-value"]))

if hasil.signifikan():
    st.success(hasil.kesimpulan(), icon=":material/check_circle:")
else:
    st.info(hasil.kesimpulan(), icon=":material/info:")

tab_koef, tab_slope, tab_jn = st.tabs(
    ["Koefisien", "Kemiringan sederhana", "Johnson-Neyman"]
)

with tab_koef:
    ui.show_table(
            hasil.koefisien,
            "moderasi_koefisien.csv",
            bagian="Regresi moderasi",
            judul="Koefisien model moderasi",
        )
    ui.interpretation(
        f"Baris **{hasil.nama_interaksi}** adalah suku interaksinya. Bila signifikan, "
        f"pengaruh {x} terhadap {y} bergantung pada tingkat {m}. ΔR² menunjukkan "
        "berapa banyak tambahan keragaman yang dijelaskan oleh moderasi itu — "
        "signifikan secara statistik belum tentu besar secara praktis."
    )
    if pusatkan:
        st.caption(
            f"Variabel dipusatkan, sehingga koefisien {x} dibaca sebagai pengaruh "
            f"{x} saat {m} berada pada rata-ratanya."
        )

with tab_slope:
    slopes = mo.simple_slopes(hasil)
    ui.show_table(slopes, "moderasi_simple_slopes.csv")
    st.plotly_chart(
        plots.moderation_plot(mo.data_plot_slopes(hasil), x, y), width="stretch"
    )
    ui.interpretation(
        "Tiap garis adalah hubungan X–Y pada satu tingkat moderator. Garis yang "
        "berbeda kemiringannya menandakan moderasi; garis yang sejajar berarti "
        "pengaruh X seragam. Perhatikan juga kolom signifikansi: kemiringan bisa saja "
        "nyata pada moderator tinggi tetapi tidak pada moderator rendah."
    )

with tab_jn:
    tabel = mo.rentang_signifikan(hasil)
    ui.show_table(tabel, "moderasi_johnson_neyman.csv")
    jn = mo.johnson_neyman(hasil)
    if jn["ada_batas"]:
        batas = ", ".join(formatting.num(b, 3) for b in jn["batas"])
        st.caption(
            f"Nilai {m} tempat pengaruh {x} berganti status signifikansi: {batas}. "
            f"Rentang nilai {m} yang teramati kira-kira "
            f"{formatting.num(hasil.rata_m - 3 * hasil.sd_m, 2)} sampai "
            f"{formatting.num(hasil.rata_m + 3 * hasil.sd_m, 2)}."
        )
    else:
        st.caption(
            "Tidak ada titik potong nyata: status signifikansi pengaruh X tidak "
            "berubah pada rentang nilai moderator mana pun."
        )
    ui.interpretation(
        "Batas Johnson-Neyman di luar rentang data yang teramati sebaiknya diabaikan "
        "— ia hasil ekstrapolasi matematis, bukan temuan tentang responden Anda."
    )

st.caption(
    f"n = {formatting.num(hasil.n)} observasi lengkap · model dibandingkan dengan "
    "model tanpa suku interaksi untuk memperoleh ΔR²."
)
