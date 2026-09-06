"""Ruang Proyek: pertanyaan penelitian, rancangan, dan ukuran sampel.

Halaman ini berguna **sebelum data ada**. Ia menetapkan dua hal yang tidak dapat
disimpulkan dari angka mana pun: apakah kesimpulan boleh berbunyi sebab-akibat, dan
apakah kesimpulan boleh diberlakukan pada populasi.
"""

from __future__ import annotations

import streamlit as st

from nalardata import formatting, proyek_penelitian as pp, sampel as sp, ui

ui.butuh_fitur("kamus")
ui.page_setup(
    "Ruang Proyek",
    "Rencana",
    "Pertanyaan penelitian, rancangan, dan ukuran sampel. Diisi sebelum data "
    "dikumpulkan — dan menentukan apa yang boleh disimpulkan nanti.",
)
ui.sidebar_info()

proyek = ui.penelitian()

kurang = proyek.kekurangan()
if kurang:
    st.info(
        "Bagian yang belum diisi: " + ", ".join(f"**{k}**" for k in kurang) + ". "
        "Halaman lain tetap dapat dipakai, namun batas kesimpulan akan memakai "
        "anggapan yang paling berhati-hati.",
        icon=":material/edit_note:",
    )

tab_dasar, tab_desain, tab_sampel, tab_pra = st.tabs(
    ["Pertanyaan penelitian", "Rancangan", "Ukuran sampel", "Praregistrasi"]
)

# --------------------------------------------------------------------------- #
# Pertanyaan penelitian
# --------------------------------------------------------------------------- #

with tab_dasar:
    kiri, kanan = st.columns(2)
    judul = kiri.text_input("Judul penelitian", value=proyek.judul, key="pp_judul")
    bidang = kanan.text_input(
        "Bidang", value=proyek.bidang, key="pp_bidang", placeholder="Manajemen, pendidikan, kesehatan"
    )

    pertanyaan = st.text_area(
        "Pertanyaan penelitian",
        value="\n".join(proyek.pertanyaan),
        key="pp_pertanyaan",
        height=110,
        help="Satu pertanyaan per baris.",
        placeholder="Apakah kualitas layanan berhubungan dengan kepuasan nasabah?",
    )
    hipotesis = st.text_area(
        "Hipotesis",
        value="\n".join(proyek.hipotesis),
        key="pp_hipotesis",
        height=110,
        help="Satu hipotesis per baris.",
        placeholder="H1: Kualitas layanan berhubungan positif dengan kepuasan nasabah",
    )

    kiri, tengah, kanan = st.columns(3)
    populasi = kiri.text_input(
        "Populasi", value=proyek.populasi, key="pp_populasi",
        placeholder="Nasabah cabang Jakarta",
    )
    unit = tengah.text_input(
        "Unit analisis", value=proyek.unit_analisis, key="pp_unit",
        placeholder="Nasabah perorangan",
    )
    ukuran_populasi = kanan.number_input(
        "Ukuran populasi (bila diketahui)",
        min_value=0,
        value=int(proyek.ukuran_populasi or 0),
        step=10,
        key="pp_n_populasi",
        help="Nol berarti belum diketahui.",
    )

# --------------------------------------------------------------------------- #
# Rancangan
# --------------------------------------------------------------------------- #

with tab_desain:
    kiri, kanan = st.columns(2)
    kode_desain = kiri.selectbox(
        "Desain penelitian",
        list(pp.DESAIN),
        index=list(pp.DESAIN).index(proyek.desain),
        format_func=lambda k: pp.DESAIN[k].nama,
        key="pp_desain",
    )
    kiri.caption(pp.DESAIN[kode_desain].keterangan)

    kode_sampling = kanan.selectbox(
        "Teknik pengambilan sampel",
        list(pp.SAMPLING),
        index=list(pp.SAMPLING).index(proyek.teknik_sampling),
        format_func=lambda k: pp.SAMPLING[k].nama,
        key="pp_sampling",
    )
    kanan.caption(pp.SAMPLING[kode_sampling].alasan)

    kiri, kanan = st.columns(2)
    acak = kiri.checkbox(
        "Subjek ditempatkan ke kelompok secara acak",
        value=proyek.penugasan_acak,
        key="pp_acak",
        disabled=not pp.DESAIN[kode_desain].perlu_penugasan_acak,
        help=(
            "Hanya berlaku pada rancangan eksperimen. Penugasan acak adalah satu-satunya "
            "hal yang membuka kesimpulan sebab-akibat."
        ),
    )
    kode_sumber = kanan.selectbox(
        "Sumber data",
        list(pp.SUMBER_DATA),
        index=list(pp.SUMBER_DATA).index(proyek.sumber_data),
        format_func=lambda k: pp.SUMBER_DATA[k],
        key="pp_sumber",
    )

    # Pratinjau langsung: pengguna melihat akibat pilihannya sebelum menyimpan.
    pratinjau = pp.ProyekPenelitian(
        desain=kode_desain,
        penugasan_acak=acak,
        teknik_sampling=kode_sampling,
        sumber_data=kode_sumber,
    )

    ui.judul_bagian(
        "Akibat pilihan ini pada kesimpulan",
        "Ditentukan rancangan, bukan hasil statistiknya.",
        kicker="Batas",
    )
    kiri, kanan = st.columns(2)
    with kiri:
        if pratinjau.boleh_sebab:
            st.success("Bahasa sebab-akibat **dibolehkan**", icon=":material/lock_open:")
        else:
            st.warning("Bahasa sebab-akibat **dikunci**", icon=":material/lock:")
        st.caption(pratinjau.alasan_sebab)
    with kanan:
        if pratinjau.boleh_generalisasi:
            st.success("Generalisasi ke populasi **dibolehkan**", icon=":material/lock_open:")
        else:
            st.warning("Generalisasi ke populasi **dikunci**", icon=":material/lock:")
        st.caption(pratinjau.alasan_generalisasi)

    st.markdown("**Kalimat batas yang akan ikut ke laporan**")
    for baris in pratinjau.batas_kesimpulan():
        st.markdown(f"- {baris}")

# --------------------------------------------------------------------------- #
# Ukuran sampel
# --------------------------------------------------------------------------- #

with tab_sampel:
    st.caption(
        "Slovin dan daya uji menjawab pertanyaan yang berbeda. Slovin bertanya "
        "'seberapa teliti perkiraan saya terhadap populasi'; daya uji bertanya "
        "'seberapa besar peluang saya menemukan pengaruh bila pengaruh itu memang ada'. "
        "Keduanya ditampilkan berdampingan agar tidak saling menggantikan."
    )

    kiri, tengah, kanan = st.columns(3)
    galat = kiri.select_slider(
        "Galat yang ditoleransi (Slovin & Cochran)",
        options=[0.01, 0.03, 0.05, 0.10],
        value=0.05,
        format_func=lambda x: f"{x:.0%}",
        key="pp_galat",
    )
    efek = tengah.selectbox(
        "Besar pengaruh yang hendak dideteksi",
        ["kecil", "sedang", "besar"],
        index=1,
        key="pp_efek",
        help="Menurut patokan Cohen (1988). Makin kecil pengaruhnya, makin besar sampelnya.",
    )
    kembali = kanan.slider(
        "Perkiraan kuesioner yang kembali terisi",
        min_value=0.3,
        max_value=1.0,
        value=0.8,
        step=0.05,
        format="%.0f%%",
        key="pp_kembali",
    )

    kiri, tengah, kanan = st.columns(3)
    n_kelompok = kiri.number_input(
        "Jumlah kelompok yang dibandingkan", min_value=2, max_value=20, value=3, key="pp_kelompok"
    )
    n_prediktor = tengah.number_input(
        "Jumlah prediktor pada regresi", min_value=1, max_value=50, value=3, key="pp_prediktor"
    )
    kanan.metric("Populasi", formatting.num(ukuran_populasi) if ukuran_populasi else "belum diisi")

    hitungan = []
    if ukuran_populasi:
        hitungan.append(sp.slovin(int(ukuran_populasi), galat))
    hitungan.append(sp.cochran(0.5, galat, populasi=int(ukuran_populasi) or None))
    hitungan.append(sp.daya_uji_t(sp.EFEK_D[efek]))
    hitungan.append(sp.daya_uji_anova(int(n_kelompok), sp.EFEK_F[efek]))
    hitungan.append(sp.daya_uji_regresi(int(n_prediktor), sp.EFEK_F2[efek]))
    hitungan.append(sp.daya_uji_korelasi(sp.EFEK_R[efek]))

    tabel = sp.bandingkan(hitungan, tingkat_kembali=kembali)
    ui.show_table(
        tabel,
        "ukuran_sampel.csv",
        bagian="Rencana penelitian",
        judul="Perbandingan ukuran sampel",
        catatan="Dihitung pada alfa 0,05 dan daya uji 0,80.",
    )

    terbesar = int(tabel["Ukuran sampel"].max())
    sebar = sp.dengan_cadangan(terbesar, kembali)
    st.success(
        f"Agar seluruh rencana analisis tercukupi, targetkan **{formatting.num(terbesar)} "
        f"responden** — dan sebarkan sekitar **{formatting.num(sebar)} kuesioner** "
        f"dengan perkiraan pengembalian {kembali:.0%}.",
        icon=":material/groups:",
    )

    if not ukuran_populasi:
        st.caption(
            "Slovin tidak ditampilkan karena ukuran populasi belum diisi pada tab "
            "*Pertanyaan penelitian*."
        )

    target = st.number_input(
        "Target sampel yang Anda tetapkan",
        min_value=0,
        value=int(proyek.target_sampel or terbesar),
        step=10,
        key="pp_target",
    )

    ui.interpretation(
        "Slovin berhenti di sekitar 400 berapa pun besar populasinya, sehingga untuk "
        "populasi besar ia hampir selalu memberi angka yang lebih kecil daripada daya "
        "uji. Bila keduanya berbeda jauh, ikuti yang lebih besar — sampel yang cukup "
        "untuk menaksir proporsi belum tentu cukup untuk menemukan pengaruh."
    )

# --------------------------------------------------------------------------- #
# Praregistrasi
# --------------------------------------------------------------------------- #

with tab_pra:
    st.caption(
        "Catat hipotesis dan uji yang direncanakan **sebelum** melihat data. Nanti "
        "aplikasi membandingkannya dengan yang benar-benar dijalankan. Menyimpang dari "
        "rencana adalah hal biasa dalam penelitian; yang tidak biasa adalah menyimpang "
        "tanpa menyebutkannya."
    )

    lama = proyek.praregistrasi
    pra_hipotesis = st.text_area(
        "Hipotesis yang akan diuji",
        value="\n".join(lama.hipotesis) if lama else "\n".join(proyek.hipotesis),
        key="pp_pra_hipotesis",
        height=110,
    )
    pra_uji = st.text_area(
        "Uji yang direncanakan",
        value="\n".join(lama.uji_direncanakan) if lama else "",
        key="pp_pra_uji",
        height=110,
        placeholder="Regresi linear berganda\nUji beda dua kelompok bebas",
    )
    pra_catatan = st.text_input(
        "Catatan", value=lama.catatan if lama else "", key="pp_pra_catatan"
    )

    if lama and not lama.kosong():
        st.info(
            f"Praregistrasi dicatat **{lama.waktu}** dengan sidik `{lama.sidik}`.",
            icon=":material/verified:",
        )

# --------------------------------------------------------------------------- #
# Simpan
# --------------------------------------------------------------------------- #

st.divider()


def _baris(teks: str) -> list[str]:
    return [b.strip() for b in teks.splitlines() if b.strip()]


if st.button("Simpan rancangan", type="primary", key="pp_simpan"):
    isi_pra = _baris(pra_hipotesis) or _baris(pra_uji)
    baru = pp.ProyekPenelitian(
        judul=judul,
        bidang=bidang,
        pertanyaan=_baris(pertanyaan),
        hipotesis=_baris(hipotesis),
        populasi=populasi,
        ukuran_populasi=int(ukuran_populasi) or None,
        unit_analisis=unit,
        desain=kode_desain,
        penugasan_acak=acak,
        teknik_sampling=kode_sampling,
        sumber_data=kode_sumber,
        target_sampel=int(target) or None,
        dibuat=proyek.dibuat,
        praregistrasi=(
            pp.Praregistrasi(
                waktu=lama.waktu if lama else "",
                hipotesis=_baris(pra_hipotesis),
                uji_direncanakan=_baris(pra_uji),
                catatan=pra_catatan,
            )
            if isi_pra
            else None
        ),
    )
    ui.set_penelitian(baru)
    st.success("Rancangan tersimpan untuk sesi ini. Unduh berkas proyek agar tidak hilang.")
    st.rerun()

ui.judul_bagian("Ringkasan rancangan", kicker="Rencana")
ui.show_table(
    proyek.ringkas(),
    "rancangan_penelitian.csv",
    bagian="Rencana penelitian",
    judul="Rancangan penelitian",
)
