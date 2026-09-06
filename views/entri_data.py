"""Halaman entri data: membuat tabel baru dari nol atau menyunting data aktif."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import data_entry as de
from nalardata import formatting, reliability as rb, ui

ui.butuh_fitur("entri_data")
ui.page_setup(
    "Buat & Entri Data",
    "Data",
    "Susun tabel data langsung di aplikasi — tentukan kolomnya, isi barisnya, lalu "
    "pakai sebagai data aktif tanpa perlu berkas CSV.",
)
ui.sidebar_info()

RANCANGAN = "entri_rancangan"  # daftar KolomBaru yang sedang disusun
DRAFT = "entri_draft"  # isi tabel yang sedang diketik

if RANCANGAN not in st.session_state:
    st.session_state[RANCANGAN] = [
        de.KolomBaru("responden", "Teks"),
        de.KolomBaru("usia", "Angka bulat"),
    ]


def config_kolom(kolom: list[de.KolomBaru]) -> dict:
    """Terjemahkan definisi kolom menjadi pengaturan editor tabel Streamlit."""
    config = {}
    for k in kolom:
        nama = de.bakukan_nama(k.nama)
        rentang = de.RENTANG_LIKERT.get(k.tipe)
        if rentang:
            config[nama] = st.column_config.NumberColumn(
                nama, min_value=rentang[0], max_value=rentang[1], step=1,
                help=f"Skala {rentang[0]} sampai {rentang[1]}",
            )
        elif k.tipe in ("Kategori", "Ya/Tidak"):
            config[nama] = st.column_config.SelectboxColumn(nama, options=k.daftar_pilihan())
        elif k.tipe == "Angka bulat":
            config[nama] = st.column_config.NumberColumn(nama, step=1)
        elif k.tipe == "Angka desimal":
            config[nama] = st.column_config.NumberColumn(nama, format="%.4f")
        else:
            config[nama] = st.column_config.TextColumn(nama)
    return config


tab_baru, tab_sunting, tab_gabung = st.tabs(
    ["Buat data baru", "Sunting data aktif", "Variabel gabungan"]
)

# --------------------------------------------------------------------------- #
# Membuat data baru
# --------------------------------------------------------------------------- #

with tab_baru:
    st.subheader("1. Tentukan kolom")
    st.caption(
        "Tiap baris di bawah menjadi satu kolom pada tabel Anda. Nama kolom otomatis "
        "dibakukan menjadi huruf kecil tanpa spasi agar aman dipakai di seluruh analisis."
    )

    rancangan = pd.DataFrame(
        {
            "Nama kolom": [k.nama for k in st.session_state[RANCANGAN]],
            "Tipe": [k.tipe for k in st.session_state[RANCANGAN]],
            "Pilihan (pisahkan dengan koma)": [
                ", ".join(k.pilihan) for k in st.session_state[RANCANGAN]
            ],
        }
    )
    rancangan_baru = st.data_editor(
        rancangan,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="editor_rancangan",
        column_config={
            "Nama kolom": st.column_config.TextColumn(required=True),
            "Tipe": st.column_config.SelectboxColumn(options=list(de.TIPE_KOLOM)),
            "Pilihan (pisahkan dengan koma)": st.column_config.TextColumn(
                help="Hanya untuk tipe Kategori, misalnya: Mikro, Kecil, Menengah"
            ),
        },
    )
    kolom = [
        de.KolomBaru(
            nama=str(baris["Nama kolom"] or ""),
            tipe=str(baris["Tipe"] or "Angka desimal"),
            pilihan=[p.strip() for p in str(baris["Pilihan (pisahkan dengan koma)"] or "").split(",")],
        )
        for _, baris in rancangan_baru.iterrows()
    ]
    st.session_state[RANCANGAN] = kolom

    with st.expander("Tambahkan butir kuesioner sekaligus"):
        st.caption(
            "Menyusun butir bernomor untuk tiap konstruk — pola penamaan yang dipakai "
            "pada analisis faktor konfirmatori dan SEM."
        )
        kol1, kol2, kol3 = st.columns([2, 1, 1.4])
        nama_konstruk = kol1.text_input(
            "Nama konstruk", placeholder="Kualitas layanan", key="konstruk_nama"
        )
        jumlah_butir = kol2.number_input("Jumlah butir", 1, 30, 4, key="konstruk_butir")
        skala = kol3.selectbox(
            "Skala", ["Skala Likert 1–5", "Skala Likert 1–7"], key="konstruk_skala"
        )
        if st.button("Tambahkan butir", key="tambah_butir"):
            if not nama_konstruk.strip():
                st.warning("Isi nama konstruknya lebih dulu.")
            else:
                st.session_state[RANCANGAN] = kolom + de.kolom_kuesioner(
                    {nama_konstruk: int(jumlah_butir)}, skala
                )
                st.rerun()

    masalah = de.validasi_kolom(kolom)
    if masalah:
        for pesan in masalah:
            st.warning(pesan)
        st.stop()

    st.subheader("2. Isi data")
    jumlah_baris = st.number_input(
        "Jumlah baris awal (baris masih dapat ditambah langsung di tabel)",
        1, 1000, 10, key="entri_baris",
    )
    tanda = (tuple((de.bakukan_nama(k.nama), k.tipe) for k in kolom), int(jumlah_baris))
    if st.session_state.get("entri_tanda") != tanda:
        kerangka = de.buat_kerangka(kolom, int(jumlah_baris))
        lama = st.session_state.get(DRAFT)
        if lama is not None and not lama.empty:
            # Isian yang sudah diketik dipertahankan untuk kolom yang namanya sama.
            bersama = [c for c in kerangka.columns if c in lama.columns]
            for c in bersama:
                kerangka.loc[: len(lama) - 1, c] = lama[c].to_numpy()[: len(kerangka)]
        st.session_state[DRAFT] = kerangka
        st.session_state["entri_tanda"] = tanda

    draft = st.data_editor(
        st.session_state[DRAFT],
        num_rows="dynamic",
        width="stretch",
        hide_index=False,
        key="editor_isi",
        column_config=config_kolom(kolom),
    )

    rapi = de.rapikan(draft)
    for peringatan in de.periksa_rentang(rapi, kolom):
        st.warning(peringatan)

    if rapi.empty:
        st.info("Tabel masih kosong. Isi minimal satu baris untuk dapat disimpan.")
    else:
        kelengkapan = de.ringkas_kelengkapan(rapi)
        kosong = int(kelengkapan["Kosong"].sum())
        st.caption(
            f"{formatting.num(len(rapi))} baris terisi · {rapi.shape[1]} kolom · "
            f"{formatting.num(kosong)} sel masih kosong"
        )
        with st.expander("Kelengkapan per kolom"):
            ui.show_table(kelengkapan, "kelengkapan_entri.csv")

        st.subheader("3. Simpan")
        nama_data = st.text_input("Nama data", "data_entri.csv", key="entri_nama")
        simpan, unduh = st.columns(2)
        if simpan.button("Jadikan data aktif", type="primary", width="stretch"):
            ui.set_dataset(rapi, nama_data or "data_entri.csv")
            st.success(
                f"Data aktif diganti dengan {formatting.num(len(rapi))} baris entri. "
                "Seluruh halaman analisis kini memakai data ini."
            )
        unduh.download_button(
            "Unduh sebagai CSV",
            rapi.to_csv(index=False).encode("utf-8"),
            file_name=nama_data or "data_entri.csv",
            mime="text/csv",
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# Menyunting data aktif
# --------------------------------------------------------------------------- #

with tab_sunting:
    aktif = ui.get_dataset()
    if aktif is None:
        st.info(
            "Belum ada data aktif. Buat tabel baru pada tab sebelah, atau muat berkas "
            "lewat halaman Beranda & Data."
        )
    else:
        st.caption(
            f"Menyunting **{st.session_state.get(ui.NAME_KEY, 'data')}** — "
            f"{formatting.num(len(aktif))} baris, {aktif.shape[1]} kolom. Baris dapat "
            "ditambah atau dihapus langsung di tabel."
        )
        disunting = st.data_editor(
            aktif,
            num_rows="dynamic",
            width="stretch",
            height=460,
            key="editor_aktif",
        )
        hasil = de.rapikan(disunting)
        selisih = len(hasil) - len(aktif)
        keterangan = (
            f"{formatting.num(abs(selisih))} baris {'ditambah' if selisih > 0 else 'dihapus'}"
            if selisih
            else "jumlah baris tidak berubah"
        )
        st.caption(f"Hasil suntingan: {formatting.num(len(hasil))} baris ({keterangan}).")
        if st.button("Simpan perubahan ke data aktif", type="primary", key="simpan_sunting"):
            ui.set_dataset(hasil, st.session_state.get(ui.NAME_KEY, "data"))
            st.success("Perubahan disimpan. Halaman analisis akan memakai data terbaru.")


# --------------------------------------------------------------------------- #
# Variabel gabungan dari beberapa butir
# --------------------------------------------------------------------------- #

with tab_gabung:
    st.caption(
        "Penelitian kuesioner biasanya mengukur satu konstruk lewat beberapa butir. "
        "Sebelum dianalisis, butir-butir itu diringkas menjadi satu variabel — "
        "misalnya KUAL1 sampai KUAL4 menjadi variabel *Kualitas*."
    )

    aktif_g = ui.get_dataset()
    if aktif_g is None or aktif_g.empty:
        st.info(
            "Belum ada data aktif. Muat data di halaman *Beranda & Data*, atau buat "
            "data baru pada tab pertama."
        )
    else:
        angka_g = [
            k
            for k in aktif_g.columns
            if pd.to_numeric(aktif_g[k], errors="coerce").notna().any()
        ]
        if len(angka_g) < 2:
            st.warning("Diperlukan minimal 2 kolom berisi angka.")
        else:
            tebakan = rb.tebak_konstruk(angka_g)
            if tebakan:
                st.caption(
                    "Kelompok butir yang terdeteksi dari pola penamaan: "
                    + ", ".join(f"**{n}** ({len(b)} butir)" for n, b in tebakan.items())
                )

            pilihan_awal = next(iter(tebakan.values()), angka_g[:2])
            nama_awal = next(iter(tebakan), "")

            butir = st.multiselect(
                "Butir penyusun konstruk",
                angka_g,
                default=[b for b in pilihan_awal if b in angka_g],
                key="gabung_butir",
            )
            kol1, kol2 = st.columns(2)
            nama_baru = kol1.text_input(
                "Nama variabel baru",
                value=nama_awal,
                key="gabung_nama",
                placeholder="misalnya: kualitas",
            )
            cara = kol2.selectbox(
                "Cara meringkas",
                list(de.CARA_GABUNG),
                format_func=lambda k: de.CARA_GABUNG[k]["nama"],
                key="gabung_cara",
            )
            st.caption(de.CARA_GABUNG[cara]["catatan"])

            minimal = st.slider(
                "Butir minimal yang harus terisi agar responden tetap diberi skor",
                1,
                max(len(butir), 1),
                max(len(butir), 1),
                key="gabung_minimal",
                help=(
                    "Responden yang mengisi lebih sedikit dari ini dibiarkan kosong, "
                    "bukan dihitung dari sisa butir yang ada."
                ),
            )

            try:
                gabungan = de.variabel_gabungan(
                    aktif_g, butir, nama_baru, cara, minimal_terisi=minimal
                )
            except ValueError as galat:
                st.info(str(galat))
            else:
                if str(gabungan.name) in aktif_g.columns:
                    st.warning(
                        f"Kolom **{gabungan.name}** sudah ada dan akan ditimpa.",
                        icon=":material/warning:",
                    )
                ui.show_table(
                    de.ringkas_gabungan(aktif_g, butir, gabungan), "variabel_gabungan.csv"
                )

                kosong = int(gabungan.isna().sum())
                if kosong:
                    st.caption(
                        f"{formatting.num(kosong)} responden tidak diberi skor karena "
                        f"butir terisinya kurang dari {minimal}."
                    )

                if st.button(
                    f"Tambahkan '{gabungan.name}' ke data aktif",
                    type="primary",
                    key="gabung_simpan",
                ):
                    diperbarui = aktif_g.copy()
                    diperbarui[str(gabungan.name)] = gabungan
                    ui.set_dataset(
                        diperbarui, st.session_state.get(ui.NAME_KEY, "data")
                    )
                    st.success(
                        f"Variabel **{gabungan.name}** ditambahkan. Seluruh halaman "
                        "analisis kini dapat memakainya."
                    )
                    st.rerun()
