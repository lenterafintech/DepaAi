"""Halaman CFA, analisis jalur, dan SEM."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lentera_mva import formatting, plots, preprocessing, reliability as rb
from lentera_mva import sem_analysis as sem
from lentera_mva import ui

ui.butuh_fitur("sem")
ui.page_setup(
    "CFA, Jalur & SEM",
    "Model Struktural",
    "Menguji model yang Anda rancang: apakah butir benar mengukur konstruknya (CFA), "
    "bagaimana antar variabel saling memengaruhi (analisis jalur), dan keduanya "
    "sekaligus (SEM).",
)
df = ui.require_dataset()
ui.sidebar_info()

ui.method_note(
    "CFA, analisis jalur, dan SEM",
    "Ketiganya diestimasi pada data yang distandardisasi, dengan fungsi tujuan yang "
    "dapat Anda pilih sendiri. "
    "**CFA** menguji model pengukuran: apakah butir-butir benar memuat pada konstruk "
    "yang diniatkan. **Analisis jalur** menguji hubungan antar variabel teramati, "
    "termasuk efek tidak langsung lewat variabel perantara. **SEM** menggabungkan "
    "keduanya. Kecocokan model dinilai dari beberapa indeks sekaligus, bukan satu "
    "angka — uji chi-square hampir selalu signifikan pada sampel besar, sehingga "
    "indeks pendamping seperti CFI, TLI, dan RMSEA ikut dilaporkan.",
)

numerik = preprocessing.numeric_columns(df)
if len(numerik) < 3:
    st.error("Halaman ini memerlukan minimal 3 kolom numerik.")
    st.stop()

# --------------------------------------------------------------------------- #
# Pilihan estimator
# --------------------------------------------------------------------------- #

saran, alasan = sem.saran_estimator(df, numerik)
with st.expander(
    f"Metode estimasi — saran untuk data ini: {sem.ESTIMATOR[saran]['nama']}",
    expanded=False,
):
    st.caption(alasan)
    kode_est = list(sem.ESTIMATOR)
    estimator = st.radio(
        "Fungsi tujuan estimasi",
        kode_est,
        index=kode_est.index(saran),
        format_func=lambda k: sem.ESTIMATOR[k]["nama"]
        + (" — disarankan" if k == saran else ""),
        key="sem_estimator",
    )
    st.markdown(f"**Dipakai bila:** {sem.ESTIMATOR[estimator]['kapan']}")
    st.caption(sem.ESTIMATOR[estimator]["catatan"])
    st.caption(
        "Estimator menentukan asumsi yang dituntut dari data, bukan model yang diuji. "
        "Sebutkan estimator yang dipakai saat melaporkan hasil, karena penelaah "
        "menanyakannya."
    )


def tampilkan_hasil(hasil: sem.HasilSEM, kunci: str) -> None:
    """Bagian pelaporan yang sama untuk CFA, jalur, maupun SEM."""
    for catatan in hasil.catatan:
        st.warning(catatan)

    fit = sem.tabel_kecocokan(hasil)
    lolos = int((fit["Keputusan"] == "Memenuhi").sum())
    st.caption(
        f"Estimator: **{hasil.nama_estimator}** · {hasil.n} observasi dipakai"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Chi-square/df", formatting.num(float(hasil.statistik["chi2"]) / max(float(hasil.statistik["DoF"]), 1), 3))
    m2.metric("CFI", formatting.num(float(hasil.statistik["CFI"]), 3))
    m3.metric("RMSEA", formatting.num(float(hasil.statistik["RMSEA"]), 3))
    m4.metric("SRMR", formatting.num(float(hasil.statistik.get("SRMR", float("nan"))), 3))
    m5.metric("Indeks memenuhi", f"{lolos}/{len(fit)}")

    st.subheader("Kecocokan model")
    ui.show_table(fit, f"kecocokan_{kunci}.csv")
    catatan = sem.catatan_chi_square(hasil)
    if catatan:
        st.warning(catatan, icon=":material/edit_note:")

    if not hasil.muatan().empty:
        st.subheader("Model pengukuran")
        muatan = hasil.muatan().rename(columns={"Ke": "Butir", "Dari": "Konstruk"})
        ui.show_table(
            muatan[
                ["Konstruk", "Butir", "Estimasi", "Estimasi baku", "Std. Error", "z", "p-value", "Signifikan"]
            ],
            f"muatan_{kunci}.csv",
        )
        st.caption(
            "Butir pertama tiap konstruk difiksasi sebagai acuan skala, sehingga tidak "
            "memiliki nilai z dan p — itu konsekuensi identifikasi model, bukan "
            "kegagalan uji."
        )
        reliabilitas = sem.reliabilitas_konstruk(hasil)
        if not reliabilitas.empty:
            st.subheader("Reliabilitas konstruk")
            ui.show_table(reliabilitas, f"reliabilitas_{kunci}.csv")
            ui.interpretation(
                "CR ≥ 0,70 dan AVE ≥ 0,50 menandakan konstruk cukup andal dan lebih "
                "banyak menjelaskan daripada galat pengukurannya. Kolom 'Butir arah "
                "terbalik' menghitung butir bermuatan negatif — lazim pada pernyataan "
                "yang dirumuskan negatif, dan sudah diperhitungkan dalam CR."
            )

    if not hasil.jalur().empty:
        st.subheader("Model struktural")
        jalur = hasil.jalur()
        ui.show_table(
            jalur[["Dari", "Ke", "Estimasi", "Estimasi baku", "Std. Error", "z", "p-value", "Signifikan"]],
            f"jalur_{kunci}.csv",
        )
        st.plotly_chart(plots.path_diagram(jalur, hasil.laten), width="stretch")
        ui.interpretation(
            "Angka pada panah adalah koefisien baku: perubahan variabel tujuan dalam "
            "satuan simpangan baku untuk setiap kenaikan satu simpangan baku variabel "
            "asal. Panah abu-abu menandakan jalur yang belum terbukti signifikan."
        )


tab_cfa, tab_jalur, tab_sem, tab_sintaks = st.tabs(
    ["CFA (model pengukuran)", "Analisis jalur", "SEM penuh", "Sintaks sendiri"]
)

# --------------------------------------------------------------------------- #
# CFA
# --------------------------------------------------------------------------- #

with tab_cfa:
    st.caption(
        "Kelompokkan butir menjadi konstruk. Bila butir dinamai berpola (KUAL1, "
        "KUAL2, …), pengelompokan awal ditebak dari awalannya."
    )
    tebakan = rb.tebak_konstruk(numerik) or {"konstruk1": numerik[:3]}
    dipakai = st.multiselect(
        "Konstruk yang diuji",
        list(tebakan),
        default=list(tebakan)[: min(2, len(tebakan))],
        key="cfa_konstruk",
    )
    konstruk_cfa: dict[str, list[str]] = {}
    for nama in dipakai:
        butir = st.multiselect(
            f"Butir konstruk {nama}",
            numerik,
            default=[b for b in tebakan[nama] if b in numerik],
            key=f"cfa_butir_{nama}",
        )
        if len(butir) >= 2:
            konstruk_cfa[nama] = butir

    if not konstruk_cfa:
        st.info("Pilih minimal satu konstruk dengan minimal 2 butir.")
    else:
        spesifikasi = sem.spesifikasi_cfa(konstruk_cfa)
        with st.expander("Spesifikasi model"):
            st.code(spesifikasi, language="text")
        try:
            hasil = sem.jalankan(df, spesifikasi, estimator=estimator)
        except Exception as exc:  # noqa: BLE001 - kegagalan estimasi ditampilkan apa adanya
            st.error(f"Model gagal diestimasi: {exc}")
        else:
            tampilkan_hasil(hasil, "cfa")

# --------------------------------------------------------------------------- #
# Analisis jalur
# --------------------------------------------------------------------------- #

with tab_jalur:
    st.caption(
        "Susun hubungan antar variabel teramati: pilih variabel terikat, lalu "
        "prediktornya. Variabel yang menjadi terikat sekaligus prediktor berperan "
        "sebagai perantara (mediator)."
    )
    jumlah = st.number_input("Jumlah persamaan", 1, 4, 2, key="jalur_jumlah")
    jalur_spec: dict[str, list[str]] = {}
    for i in range(int(jumlah)):
        kiri, kanan = st.columns([1, 2])
        terikat = kiri.selectbox(
            f"Variabel terikat {i + 1}", numerik, index=min(i, len(numerik) - 1),
            key=f"jalur_y_{i}",
        )
        penjelas = kanan.multiselect(
            f"Prediktor untuk {terikat}",
            [c for c in numerik if c != terikat],
            default=[c for c in numerik if c != terikat][: 2 - i if i < 2 else 1],
            key=f"jalur_x_{i}",
        )
        if penjelas:
            jalur_spec[terikat] = penjelas

    if not jalur_spec:
        st.info("Tentukan minimal satu persamaan lengkap.")
    else:
        spesifikasi = sem.spesifikasi_jalur(jalur_spec)
        with st.expander("Spesifikasi model"):
            st.code(spesifikasi, language="text")
        try:
            hasil = sem.jalankan(df, spesifikasi, estimator=estimator)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Model gagal diestimasi: {exc}")
        else:
            tampilkan_hasil(hasil, "jalur")

            st.subheader("Uji mediasi")
            st.caption(
                "Pilih rangkaian X → M → Y untuk menguji apakah pengaruh X terhadap Y "
                "mengalir lewat M. Interval kepercayaan dihitung dengan bootstrap "
                "karena sebaran hasil kali dua koefisien tidak normal."
            )
            med1, med2, med3 = st.columns(3)
            x_med = med1.selectbox("X", numerik, key="med_x")
            m_med = med2.selectbox(
                "M (perantara)", [c for c in numerik if c != x_med], key="med_m"
            )
            y_med = med3.selectbox(
                "Y", [c for c in numerik if c not in {x_med, m_med}], key="med_y"
            )
            efek = sem.efek_langsung_tidak_langsung(hasil, x_med, m_med, y_med)
            ui.show_table(pd.DataFrame([efek]), "mediasi_efek.csv")
            n_boot = st.slider("Jumlah resample bootstrap", 100, 1000, 200, 100, key="med_boot")
            if st.button("Jalankan bootstrap", key="med_jalan"):
                with st.spinner("Menjalankan bootstrap…"):
                    try:
                        tabel = sem.bootstrap_mediasi(
                            df, spesifikasi, x_med, m_med, y_med, n_boot=int(n_boot)
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        ui.show_table(tabel, "mediasi_bootstrap.csv")
                        ui.interpretation(
                            "Efek tidak langsung dinyatakan signifikan bila interval "
                            "kepercayaan tidak memuat nol. VAF adalah bagian efek total "
                            "yang mengalir lewat perantara: 20–80% lazim disebut mediasi "
                            "parsial, di atas 80% mendekati mediasi penuh."
                        )

# --------------------------------------------------------------------------- #
# SEM penuh
# --------------------------------------------------------------------------- #

with tab_sem:
    st.caption(
        "Gabungkan model pengukuran dan struktural. Konstruk laten yang didefinisikan "
        "di sini dapat langsung dipakai sebagai variabel pada persamaan struktural."
    )
    tebakan = rb.tebak_konstruk(numerik) or {"konstruk1": numerik[:3]}
    dipakai = st.multiselect(
        "Konstruk laten",
        list(tebakan),
        default=list(tebakan)[: min(2, len(tebakan))],
        key="sem_konstruk",
    )
    konstruk_sem: dict[str, list[str]] = {}
    for nama in dipakai:
        butir = st.multiselect(
            f"Butir konstruk {nama}",
            numerik,
            default=[b for b in tebakan[nama] if b in numerik],
            key=f"sem_butir_{nama}",
        )
        if len(butir) >= 2:
            konstruk_sem[nama] = butir

    if len(konstruk_sem) < 2:
        st.info("Definisikan minimal 2 konstruk laten untuk menyusun jalur antar keduanya.")
    else:
        nama_konstruk = list(konstruk_sem)
        kiri, kanan = st.columns([1, 2])
        terikat = kiri.selectbox("Konstruk terikat", nama_konstruk, index=len(nama_konstruk) - 1, key="sem_y")
        penjelas = kanan.multiselect(
            f"Konstruk yang memengaruhi {terikat}",
            [k for k in nama_konstruk if k != terikat],
            default=[k for k in nama_konstruk if k != terikat],
            key="sem_x",
        )
        if not penjelas:
            st.info("Pilih minimal satu konstruk penjelas.")
        else:
            spesifikasi = sem.spesifikasi_sem(konstruk_sem, {terikat: penjelas})
            with st.expander("Spesifikasi model"):
                st.code(spesifikasi, language="text")
            try:
                hasil = sem.jalankan(df, spesifikasi, estimator=estimator)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Model gagal diestimasi: {exc}")
            else:
                tampilkan_hasil(hasil, "sem")


# --------------------------------------------------------------------------- #
# Sintaks yang diketik sendiri
# --------------------------------------------------------------------------- #

with tab_sintaks:
    st.caption(
        "Tiga tab sebelumnya menyusun model lewat pilihan. Di sini Anda menuliskan "
        "modelnya langsung — berguna untuk model yang tidak tertampung oleh pilihan "
        "itu: kovarians residual, konstruk bertingkat, atau model yang sudah Anda "
        "tulis untuk lavaan maupun AMOS."
    )

    with st.expander("Tata tulis yang dikenali", expanded=False):
        for operator, arti in sem.OPERATOR.items():
            st.markdown(f"- `{operator}` — {arti}")
        st.caption(
            "Baris yang diawali `#` diperlakukan sebagai catatan. Nama variabel harus "
            "sama persis dengan nama kolom pada data, tanpa spasi."
        )
        st.markdown("**Kolom yang tersedia pada data ini**")
        st.code(", ".join(map(str, df.columns)), language=None, wrap_lines=True)

    spesifikasi_bebas = st.text_area(
        "Spesifikasi model",
        value=st.session_state.get("sem_sintaks_isi", sem.CONTOH_SINTAKS),
        height=240,
        key="sem_sintaks_isi",
        help="Gunakan tata tulis gaya lavaan; periksa daftar operator di atas.",
    )

    masalah = sem.periksa_spesifikasi(spesifikasi_bebas, df)
    if masalah:
        st.warning(
            "Spesifikasi belum dapat dijalankan:\n\n"
            + "\n".join(f"- {m}" for m in masalah),
            icon=":material/rule:",
        )
    else:
        st.success(
            "Sintaks terbaca dengan benar.", icon=":material/check_circle:"
        )
        if st.button("Jalankan model", type="primary", key="sem_sintaks_jalan"):
            st.session_state["sem_sintaks_jalankan"] = True

        if st.session_state.get("sem_sintaks_jalankan"):
            try:
                hasil = sem.jalankan(df, spesifikasi_bebas, estimator=estimator)
            except Exception as exc:  # noqa: BLE001 - kegagalan estimasi apa adanya
                st.error(
                    f"Model gagal diestimasi: {exc}\n\n"
                    "Sintaksnya sendiri sudah benar, jadi kegagalan ini biasanya "
                    "berarti model tidak teridentifikasi — misalnya jalur terlalu "
                    "banyak untuk jumlah variabel yang ada — atau datanya tidak "
                    "mencukupi."
                )
            else:
                tampilkan_hasil(hasil, "sintaks")
