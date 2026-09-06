"""Kamus variabel: memberi tahu aplikasi apa arti setiap kolom.

Halaman ini memperbaiki kelemahan yang paling sering menjatuhkan penelitian
kuantitatif — salah menetapkan skala pengukuran. Aplikasi menebak dari bentuk data
dan menyebutkan seberapa yakin ia; pengguna mengonfirmasi dari maksudnya.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import formatting, kamus as km, ui

ui.butuh_fitur("kamus")
ui.page_setup(
    "Kamus Variabel",
    "Data",
    "Skala pengukuran dan peran setiap kolom. Aplikasi menebak dari bentuk data; "
    "hanya Anda yang tahu maksudnya, jadi tebakan itu perlu Anda periksa.",
)
df = ui.require_dataset()
ui.sidebar_info()

kamus = ui.kamus()

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

if perlu:
    st.warning(
        f"{formatting.num(len(perlu))} kolom masih memakai tebakan aplikasi: "
        + ", ".join(f"**{n}**" for n in perlu[:8])
        + ("," + " dan lainnya." if len(perlu) > 8 else "."),
        icon=":material/help:",
    )
else:
    st.success("Seluruh kolom sudah Anda konfirmasi.", icon=":material/check_circle:")

# --------------------------------------------------------------------------- #
# Penyuntingan massal
# --------------------------------------------------------------------------- #

ui.judul_bagian(
    "Skala dan peran",
    "Ubah langsung di tabel. Baris yang Anda sunting otomatis ditandai sudah "
    "dikonfirmasi.",
    kicker="Kamus",
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
