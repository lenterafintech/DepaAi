"""Kamus variabel: memberi tahu aplikasi apa arti setiap kolom.

Halaman ini memperbaiki kelemahan yang paling sering menjatuhkan penelitian
kuantitatif — salah menetapkan skala pengukuran. Aplikasi menebak dari bentuk data
dan menyebutkan seberapa yakin ia; pengguna mengonfirmasi dari maksudnya.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import formatting, kamus as km, ui

# --------------------------------------------------------------------------- #
# Kartu konfirmasi per kolom — bahasa awam, bukan istilah statistik
# --------------------------------------------------------------------------- #

# Interval dan rasio sengaja digabung jadi satu pilihan ("angka yang wajar
# dirata-ratakan"): keduanya sama-sama diperlakukan sebagai angka oleh hampir
# semua uji, dan pengguna awam tidak perlu — dan biasanya tidak bisa — membedakan
# keduanya. Ordinal dan nominal jauh lebih menentukan, jadi keduanya tetap terpisah.
_SKALA_PILIHAN = (km.NOMINAL, km.ORDINAL, km.RASIO)
_SKALA_LABEL = {
    km.NOMINAL: "Kategori tanpa urutan",
    km.ORDINAL: "Tingkatan atau skor berjenjang",
    km.RASIO: "Angka yang wajar dirata-ratakan",
}
_SKALA_CONTOH = {
    km.NOMINAL: "misalnya jenis kelamin, kota asal, jurusan",
    km.ORDINAL: "misalnya sangat tidak setuju s.d. sangat setuju, rendah–sedang–tinggi",
    km.RASIO: "misalnya umur, pendapatan, skor ujian, jumlah anak",
}

# Peran yang paling sering dipakai muncul langsung; sisanya (kovariat, mediator,
# moderator, indikator, waktu) disembunyikan di balik "Peran lain" karena
# istilahnya sendiri menuntut penjelasan yang tidak muat dalam satu kartu.
_PERAN_UMUM = (km.BELUM, "outcome", "prediktor", "kelompok", "id", "tidak dipakai")
_PERAN_LANJUT = ("kovariat", "mediator", "moderator", "indikator", "waktu")
_PERAN_LAINNYA = "__lainnya__"
_PERAN_LABEL_AWAM = {
    km.BELUM: "Belum tahu — tentukan nanti saat memilih metode",
    "outcome": "Hasil yang ingin saya jelaskan atau bandingkan",
    "prediktor": "Diduga memengaruhi atau membedakan hasil di atas",
    "kelompok": "Membagi responden menjadi beberapa kelompok",
    "id": "Sekadar identitas responden, bukan untuk dianalisis",
    "tidak dipakai": "Tidak dipakai dalam analisis",
}


def _kartu_konfirmasi(df: pd.DataFrame, kamus: km.Kamus, nama: str) -> None:
    v = kamus[nama]
    with st.container(border=True):
        st.markdown(f"**{v.judul}**  ·  `{nama}`")

        if nama in df.columns:
            contoh = [str(c) for c in df[nama].dropna().unique()[:5]]
            if contoh:
                st.caption("Contoh isi: " + ", ".join(contoh))

        st.caption(
            f"Dugaan aplikasi ({km.LABEL_KEYAKINAN[v.keyakinan].lower()}): {v.alasan}"
        )

        bucket_sekarang = v.skala if v.skala in _SKALA_PILIHAN else km.RASIO
        skala_baru = st.radio(
            "Kolom ini sebenarnya...",
            _SKALA_PILIHAN,
            index=_SKALA_PILIHAN.index(bucket_sekarang),
            format_func=lambda k: _SKALA_LABEL[k],
            captions=[_SKALA_CONTOH[k] for k in _SKALA_PILIHAN],
            key=f"kartu_skala_{nama}",
            horizontal=True,
        )

        with st.expander("Dipakai sebagai apa dalam analisis? (boleh dilewati)"):
            urutan_peran = _PERAN_UMUM + (_PERAN_LAINNYA,)
            peran_sekarang = v.peran if v.peran in _PERAN_UMUM else _PERAN_LAINNYA
            pilihan_peran = st.radio(
                "Peran variabel",
                urutan_peran,
                index=urutan_peran.index(peran_sekarang),
                format_func=lambda k: (
                    "Peran lain (kovariat, mediator, moderator, dst.)"
                    if k == _PERAN_LAINNYA
                    else _PERAN_LABEL_AWAM[k]
                ),
                key=f"kartu_peran_{nama}",
                label_visibility="collapsed",
            )
            peran_baru = pilihan_peran
            if pilihan_peran == _PERAN_LAINNYA:
                indeks_lanjut = (
                    _PERAN_LANJUT.index(v.peran) if v.peran in _PERAN_LANJUT else 0
                )
                peran_baru = st.selectbox(
                    "Pilih peran",
                    _PERAN_LANJUT,
                    index=indeks_lanjut,
                    format_func=lambda k: km.LABEL_PERAN[k],
                    key=f"kartu_peran_lanjut_{nama}",
                )

        if st.button(
            "Sesuai, tandai sudah diperiksa",
            key=f"kartu_konfirmasi_{nama}",
            type="primary",
        ):
            kamus.tetapkan(nama, skala=skala_baru, peran=peran_baru, dikonfirmasi=True)
            ui.set_kamus(kamus)
            st.rerun()


def render(df, kamus, penelitian) -> None:
    if not ui.butuh_fitur("kamus"):
        return

    st.info(
        "Aplikasi membaca **bentuk** data, bukan **maksud** penelitian. Angka 1 sampai 5 "
        "bisa berarti skor Likert, jumlah anak, atau kode wilayah — ketiganya menuntut uji "
        "yang berbeda dan tidak satu pun dapat dibedakan dari angkanya saja.",
        icon=":material/info:",
    )

    perlu = kamus.perlu_diperiksa()
    dikonfirmasi = sum(1 for v in kamus if v.dikonfirmasi)
    kolom_ukur = st.columns(4)
    kolom_ukur[0].metric("Jumlah kolom", formatting.num(len(kamus)))
    kolom_ukur[1].metric("Anda konfirmasi", formatting.num(dikonfirmasi))
    kolom_ukur[2].metric("Perlu diperiksa", formatting.num(len(perlu)))
    kolom_ukur[3].metric(
        "Masuk analisis", formatting.num(len(kamus.numerik()) + len(kamus.kategorik()))
    )
    st.caption(
        "Kolom yang tidak perlu diperiksa adalah yang skalanya terbaca pasti dari data — "
        "kolom teks, misalnya. Kolom berisi angka hampir selalu perlu Anda periksa."
    )

    # --------------------------------------------------------------------------- #
    # Konfirmasi per kolom — cara utama, bahasa awam
    # --------------------------------------------------------------------------- #

    if perlu:
        ui.judul_bagian(
            f"{formatting.num(len(perlu))} kolom masih memakai tebakan aplikasi",
            "Saran metode pada Pemandu Uji bergantung pada jawaban di sini — kolom "
            "Likert yang tercatat sebagai angka biasa akan mengantar ke uji yang keliru.",
            kicker="Konfirmasi",
        )
        for nama in perlu:
            _kartu_konfirmasi(df, kamus, nama)
    else:
        st.success("Seluruh kolom sudah Anda konfirmasi.", icon=":material/check_circle:")

    # --------------------------------------------------------------------------- #
    # Penyuntingan massal — jalan pintas untuk yang sudah paham istilahnya
    # --------------------------------------------------------------------------- #

    with st.expander(
        "Sunting cepat semua kolom sekaligus (tabel)", expanded=not perlu
    ):
        st.caption(
            "Ubah langsung di tabel. Baris yang Anda sunting otomatis ditandai sudah "
            "dikonfirmasi. Cara ini mengasumsikan Anda sudah paham istilah skala "
            "pengukuran (nominal/ordinal/interval/rasio) — bila belum, pakai kartu di "
            "atas."
        )

        with st.expander("Apa beda keempat skala itu?", expanded=False):
            for kode, keterangan in km.LABEL_SKALA.items():
                st.markdown(f"- **{kode.capitalize()}** — {keterangan.split('—', 1)[1].strip()}")
            st.caption(
                "Interval dan rasio tidak dapat dibedakan dari data. Keduanya sama-sama "
                "diperlakukan sebagai angka oleh hampir semua uji, jadi salah memilih di antara "
                "keduanya jarang berakibat fatal — berbeda dengan salah memilih antara ordinal "
                "dan rasio."
            )

        sunting = pd.DataFrame(
            [
                {
                    "Kolom": v.nama,
                    "Nama lengkap": v.nama_lengkap,
                    "Skala": v.skala,
                    "Peran": v.peran,
                    "Satuan": v.satuan,
                    "Definisi operasional": v.definisi,
                    "Sudah diperiksa": v.dikonfirmasi,
                }
                for v in kamus
            ]
        )

        hasil = st.data_editor(
            sunting,
            width="stretch",
            hide_index=True,
            disabled=["Kolom"],
            key="editor_kamus",
            column_config={
                "Kolom": st.column_config.TextColumn("Kolom", help="Nama kolom pada berkas data"),
                "Nama lengkap": st.column_config.TextColumn(
                    "Nama lengkap", help="Nama yang dipakai pada laporan, misalnya 'Kepuasan nasabah'"
                ),
                "Skala": st.column_config.SelectboxColumn("Skala", options=list(km.SKALA), required=True),
                "Peran": st.column_config.SelectboxColumn("Peran", options=list(km.PERAN), required=True),
                "Satuan": st.column_config.TextColumn("Satuan", help="rupiah, tahun, kali, persen"),
                "Definisi operasional": st.column_config.TextColumn(
                    "Definisi operasional", help="Bagaimana variabel ini diukur — masuk ke Bab III"
                ),
                "Sudah diperiksa": st.column_config.CheckboxColumn("Sudah diperiksa"),
            },
        )

        berubah = 0
        for _, baris in hasil.iterrows():
            nama = str(baris["Kolom"])
            if nama not in kamus:
                continue
            lama = kamus[nama]
            usulan = {
                "nama_lengkap": str(baris["Nama lengkap"] or ""),
                "skala": str(baris["Skala"]),
                "peran": str(baris["Peran"]),
                "satuan": str(baris["Satuan"] or ""),
                "definisi": str(baris["Definisi operasional"] or ""),
                "dikonfirmasi": bool(baris["Sudah diperiksa"]),
            }
            if any(getattr(lama, ruas) != nilai for ruas, nilai in usulan.items()):
                # Menyunting isi berarti memeriksanya; mencentang kotak juga.
                if usulan == {**usulan, "dikonfirmasi": lama.dikonfirmasi} and not lama.dikonfirmasi:
                    usulan["dikonfirmasi"] = True
                kamus.tetapkan(nama, **usulan)
                berubah += 1

        if berubah:
            ui.set_kamus(kamus)

    ui.interpretation(
        "**Ordinal atau rasio** adalah pilihan yang paling menentukan. Skor Likert yang "
        "diperlakukan sebagai rasio akan dihitung rata-ratanya seolah jarak antara "
        "'setuju' dan 'sangat setuju' sama dengan jarak antara 'netral' dan 'setuju' — "
        "padahal belum tentu. Pemandu uji membaca kolom ini untuk memutuskan antara uji "
        "parametrik dan non-parametrik."
    )

    # --------------------------------------------------------------------------- #
    # Kode nilai hilang
    # --------------------------------------------------------------------------- #

    usulan_kode = km.usulan_kode_hilang(df, kamus)
    sudah_ada = {v.nama: v.kode_hilang for v in kamus if v.kode_hilang}

    if usulan_kode or sudah_ada:
        ui.judul_bagian(
            "Kode nilai hilang",
            "Angka seperti 99 atau 999 kerap dipakai untuk menandai jawaban kosong. "
            "Bila tidak dikenali, angka itu ikut dihitung sebagai nilai sungguhan.",
            kicker="Kamus",
        )

    for nama, angka in usulan_kode.items():
        daftar = ", ".join(formatting.num(a, 0) for a in angka)
        kiri, kanan = st.columns([4, 1])
        kiri.markdown(
            f"**{kamus.judul(nama)}** memuat {daftar} yang terpencil jauh dari nilai lain. "
            "Bila itu kode jawaban kosong, tandai agar tidak ikut dihitung."
        )
        if kanan.button("Tandai kosong", key=f"kode_{nama}", width="stretch"):
            kamus.tetapkan(nama, kode_hilang=list(angka))
            ui.set_kamus(kamus)
            st.rerun()

    for nama, angka in sudah_ada.items():
        kiri, kanan = st.columns([4, 1])
        kiri.markdown(
            f"**{kamus.judul(nama)}** — {', '.join(formatting.num(a, 0) for a in angka)} "
            "diperlakukan sebagai nilai kosong."
        )
        if kanan.button("Batalkan", key=f"batal_kode_{nama}", width="stretch"):
            kamus.tetapkan(nama, kode_hilang=[])
            ui.set_kamus(kamus)
            st.rerun()

    if sudah_ada:
        terapkan = kamus.terapkan(df)
        selisih = int(terapkan.isna().sum().sum() - df.isna().sum().sum())
        st.caption(
            f"Bila diterapkan, {formatting.num(selisih)} nilai akan menjadi kosong. Data "
            "asli tetap utuh — penerapan terjadi saat analisis dijalankan, bukan sekarang."
        )

    # --------------------------------------------------------------------------- #
    # Dasar dugaan
    # --------------------------------------------------------------------------- #

    ui.judul_bagian(
        "Dasar dugaan aplikasi",
        "Alasan di balik setiap tebakan, agar Anda dapat menilainya sendiri.",
        kicker="Kamus",
    )
    ui.show_table(
        kamus.ringkas(),
        "kamus_variabel.csv",
        bagian="Kamus variabel",
        judul="Kamus variabel",
        catatan="Skala dan peran yang dipakai seluruh analisis pada laporan ini.",
    )
