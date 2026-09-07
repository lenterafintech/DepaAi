"""Halaman regresi data panel: pengaruh tetap, pengaruh acak, dan uji Hausman."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from nalardata import formatting, panel_analysis as pnl, preprocessing, ui


def render(df, kamus, penelitian) -> None:
    if not ui.butuh_fitur("lanjutan"):
        return

    ui.method_note(
        "regresi data panel",
        "Data panel mengamati entitas yang sama (mis. perusahaan, individu, wilayah) "
        "berulang kali sepanjang waktu, sehingga pengamatan pada entitas yang sama "
        "cenderung lebih mirip satu sama lain — melanggar independensi yang dituntut "
        "OLS biasa. **Pengaruh tetap** memberi tiap entitas intersepnya sendiri; "
        "**pengaruh acak** memperlakukan efek entitas sebagai bagian dari galat, "
        "lebih efisien bila layak dipakai. Uji Hausman memutuskan mana yang sah "
        "dipakai pada data Anda.",
    )

    dipandu = ui.konfigurasi_pemandu()
    ui.banner_dipandu(dipandu)

    numerik = preprocessing.numeric_columns(df)
    if len(numerik) < 2:
        st.error(
            "Regresi panel memerlukan sekurang-kurangnya dua kolom numerik "
            "(satu hasil, satu prediktor)."
        )
        return

    kol1, kol2 = st.columns(2)
    # Entitas panel harus muncul berulang (nunique < jumlah baris) — kolom yang
    # setiap nilainya unik (mis. nomor identitas) bukan entitas panel sama sekali,
    # melainkan penanda baris.
    kandidat_entitas = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) < len(df)]
    if not kandidat_entitas:
        st.error(
            "Tidak ada kolom yang cocok sebagai entitas panel — dibutuhkan kolom "
            "yang nilainya berulang (unit yang sama diamati lebih dari sekali)."
        )
        return
    entitas = kol1.selectbox(
        "Kolom entitas (identitas, mis. perusahaan/individu)",
        kandidat_entitas,
        index=ui.indeks_pilihan(kandidat_entitas, dipandu.get("kelompok")),
        key="panel_entitas",
        help="Kolom yang menandai unit yang diamati berulang, misalnya kode perusahaan.",
    )
    kandidat_waktu = [c for c in df.columns if c != entitas]
    waktu = kol2.selectbox(
        "Kolom waktu (opsional)",
        [None] + kandidat_waktu,
        format_func=lambda k: "— tidak dipakai —" if k is None else k,
        key="panel_waktu",
        help="Menambahkan efek waktu (mis. dummy tahun) pada kedua model bila diisi.",
    )

    kandidat_y = [c for c in numerik if c != entitas]
    y = st.selectbox(
        "Variabel hasil (Y)",
        kandidat_y,
        index=ui.indeks_pilihan(kandidat_y, dipandu.get("outcome")),
        key="panel_y",
    )
    kandidat_x = [c for c in numerik if c not in {y, entitas, waktu}]
    default_x = [c for c in dipandu.get("prediktor", []) if c in kandidat_x] or kandidat_x[
        : min(3, len(kandidat_x))
    ]
    prediktor = st.multiselect(
        "Prediktor (X)",
        kandidat_x,
        default=default_x,
        key="panel_x",
    )
    if not prediktor:
        st.info("Pilih sekurang-kurangnya satu prediktor.")
        return

    try:
        hasil = pnl.regresi_panel(df, y, prediktor, entitas, waktu)
    except ValueError as exc:
        st.error(str(exc))
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Entitas", formatting.num(hasil.n_entitas))
    m2.metric("Observasi", formatting.num(hasil.n_observasi))
    m3.metric("R² within (pengaruh tetap)", formatting.num(hasil.r2_within, 3))

    h = hasil.hausman()
    if np.isfinite(h["p-value"]) and h["p-value"] < 0.05:
        st.success(hasil.kesimpulan(), icon=":material/check_circle:")
    else:
        st.info(hasil.kesimpulan(), icon=":material/info:")

    tab_fe, tab_re, tab_hausman = st.tabs(
        ["Pengaruh Tetap (FE)", "Pengaruh Acak (RE)", "Uji Hausman"]
    )

    with tab_fe:
        ui.show_table(
            hasil.koefisien_fe,
            "panel_koefisien_fe.csv",
            bagian="Regresi panel",
            judul="Koefisien pengaruh tetap (fixed effects)",
        )
        ui.interpretation(
            "Koefisien di sini mengendalikan seluruh karakteristik entitas yang tidak "
            "berubah sepanjang waktu, termasuk yang tidak terukur — cocok bila Anda "
            "menduga entitas yang berbeda punya faktor tersembunyi yang ikut "
            "memengaruhi hasil."
        )

    with tab_re:
        ui.show_table(
            hasil.koefisien_re,
            "panel_koefisien_re.csv",
            bagian="Regresi panel",
            judul="Koefisien pengaruh acak (random effects)",
        )
        st.caption(
            "Diestimasi lewat model campuran (intersep acak per entitas) sebagai "
            "pengganti RE-GLS klasik, karena paket khusus data panel (linearmodels) "
            "tidak dipasang di aplikasi ini. Koefisiennya tetap dapat dibaca dan "
            "dibandingkan lewat uji Hausman."
        )
        ui.interpretation(
            "Pengaruh acak lebih efisien (galat baku lebih kecil) daripada pengaruh "
            "tetap — tetapi hanya sah dipakai bila uji Hausman TIDAK signifikan."
        )

    with tab_hausman:
        tabel = pd.DataFrame(
            [
                {
                    "Statistik": formatting.num(h["statistik"], 3),
                    "Derajat bebas": h["df"],
                    "p-value": formatting.pval_ringkas(h["p-value"]),
                }
            ]
        )
        ui.show_table(
            tabel,
            "panel_hausman.csv",
            bagian="Regresi panel",
            judul="Uji Hausman",
        )
        ui.interpretation(
            "H0 uji Hausman: efek entitas tidak berkorelasi dengan prediktor, sehingga "
            "pengaruh acak konsisten dan boleh dipakai. p < 0,05 menolak H0 — pakai "
            "pengaruh tetap. Uji ini kadang tidak dapat dihitung (selisih matriks "
            "ragam tidak dapat dibalik); pada kondisi itu pengaruh tetap adalah "
            "pilihan yang lebih aman."
        )

    st.caption(
        f"n = {formatting.num(hasil.n_observasi)} observasi pada "
        f"{formatting.num(hasil.n_entitas)} entitas"
        + (f", dengan efek waktu pada '{waktu}'." if hasil.efek_waktu else ".")
    )
