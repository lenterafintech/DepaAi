"""Pemandu Uji: memilih metode dengan melihat data Anda, bukan diagram alur.

Halaman ini menjawab pertanyaan yang paling sering diajukan penulis skripsi — "uji
apa yang harus saya pakai" — dengan memeriksa data yang sungguh ada, lalu menyebutkan
mengapa alternatifnya tidak dipilih.
"""

from __future__ import annotations

import streamlit as st

from nalardata import audit as ad
from nalardata import formatting, kamus as km, pagar, pemandu as pmd, ui

ui.butuh_fitur("pemandu")
ui.page_setup(
    "Pemandu Uji",
    "Rencana",
    "Ceritakan tujuan penelitian Anda; aplikasi memeriksa data lalu menyarankan "
    "metodenya beserta alasannya — termasuk alasan menolak alternatifnya.",
)
df = ui.require_dataset()
ui.sidebar_info()

kamus = ui.kamus()
penelitian = ui.penelitian()

st.info(
    "Pemandu membaca **bentuk** data, bukan **maksud** penelitian. Ia tidak tahu "
    "apakah pengamatan Anda benar-benar saling bebas, apakah variabelnya benar-benar "
    "mengukur yang Anda maksud, atau apakah pertanyaan penelitiannya sudah tepat. "
    "Saran di bawah adalah titik awal yang berdasar, bukan keputusan akhir.",
    icon=":material/info:",
)

# --------------------------------------------------------------------------- #
# Audit lebih dulu: memilih uji di atas data yang cacat tidak ada gunanya
# --------------------------------------------------------------------------- #

hasil_audit = ad.jalankan_audit(df, kamus)
kritis = [t for t in hasil_audit.temuan if t.tingkat == ad.KRITIS]
if kritis:
    st.error(
        f"{formatting.num(len(kritis))} masalah kritis ditemukan pada data. Memilih uji "
        "di atas data yang cacat tidak ada gunanya — perbaiki lebih dulu di halaman "
        "**Eksplorasi Data**.",
        icon=":material/error:",
    )
    for temuan in kritis[:5]:
        st.markdown(f"- **{temuan.kolom or 'Data'}** — {temuan.rincian} {temuan.saran}")

belum = kamus.perlu_diperiksa()
if belum:
    st.warning(
        f"{formatting.num(len(belum))} kolom masih memakai tebakan skala aplikasi. "
        "Saran di bawah bergantung pada skala itu — kolom Likert yang tercatat sebagai "
        "rasio akan mengantar Anda ke uji parametrik yang keliru. Periksa di "
        "**Kamus Variabel**.",
        icon=":material/help:",
    )

# --------------------------------------------------------------------------- #
# Tujuan penelitian
# --------------------------------------------------------------------------- #

ui.judul_bagian(
    "Apa yang ingin Anda ketahui?",
    "Pilih dengan pertanyaannya, bukan dengan istilah statistiknya.",
    kicker="Langkah 1",
)

tujuan = st.radio(
    "Tujuan penelitian",
    list(pmd.TUJUAN),
    format_func=lambda k: f"{pmd.TUJUAN[k]} — {pmd.PERTANYAAN_TUJUAN[k]}",
    key="pemandu_tujuan",
    label_visibility="collapsed",
)

# --------------------------------------------------------------------------- #
# Variabel
# --------------------------------------------------------------------------- #

ui.judul_bagian("Variabel mana yang terlibat?", kicker="Langkah 2")

semua = list(df.columns)
numerik = kamus.numerik()
kategorik = kamus.kategorik()


def _label(nama: str) -> str:
    butir = kamus.variabel.get(nama)
    return f"{kamus.judul(nama)} ({butir.skala})" if butir else nama


outcome = prediktor = kelompok = None
berpasangan = False

if tujuan == "membandingkan":
    berpasangan = st.toggle(
        "Pengukuran berulang pada unit yang sama (berpasangan)",
        key="pemandu_berpasangan",
        help=(
            "Satu-satunya hal yang tidak dapat dibaca dari data. Sebelum dan sesudah "
            "pelatihan pada orang yang sama adalah berpasangan; kelompok kontrol dan "
            "kelompok perlakuan yang berisi orang berbeda adalah bebas."
        ),
    )
    if berpasangan:
        pilihan = st.multiselect(
            "Kolom pengukuran berulang", numerik, key="pemandu_ulang", format_func=_label
        )
        outcome = pilihan[0] if pilihan else None
        prediktor = pilihan[1:]
    else:
        kiri, kanan = st.columns(2)
        outcome = kiri.selectbox(
            "Variabel yang dibandingkan",
            [None] + semua,
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_outcome_beda",
        )
        kelompok = kanan.selectbox(
            "Penanda kelompok",
            [None] + kategorik,
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_kelompok",
        )

elif tujuan in {"menghubungkan"}:
    pilihan = st.multiselect(
        "Dua variabel yang ingin dihubungkan",
        semua,
        max_selections=2,
        key="pemandu_hubungan",
        format_func=_label,
    )
    outcome = pilihan[0] if pilihan else None
    prediktor = pilihan[1:]

elif tujuan in {"memperkirakan_nilai", "memperkirakan_kategori"}:
    kandidat = numerik if tujuan == "memperkirakan_nilai" else semua
    kiri, kanan = st.columns([1, 2])
    outcome = kiri.selectbox(
        "Yang ingin diperkirakan",
        [None] + kandidat,
        format_func=lambda k: "— pilih —" if k is None else _label(k),
        key="pemandu_outcome_reg",
    )
    prediktor = kanan.multiselect(
        "Variabel penjelas",
        [k for k in numerik if k != outcome],
        key="pemandu_prediktor",
        format_func=_label,
    )

else:
    label = {
        "meringkas": "Variabel yang ingin diringkas",
        "mengelompokkan": "Variabel dasar pengelompokan",
        "menguji_model": "Indikator penyusun konstruk",
        "mutu_instrumen": "Butir penyusun satu konstruk",
    }[tujuan]
    prediktor = st.multiselect(label, semua, key="pemandu_banyak", format_func=_label)

# --------------------------------------------------------------------------- #
# Saran
# --------------------------------------------------------------------------- #

rekomendasi = pmd.sarankan(
    df,
    kamus,
    tujuan,
    outcome=outcome,
    prediktor=prediktor,
    kelompok=kelompok,
    berpasangan=berpasangan,
)

ui.judul_bagian("Yang disarankan", kicker="Langkah 3")

if rekomendasi.belum_terjawab:
    ui.keadaan_kosong(
        "Masih ada yang perlu Anda tentukan",
        " ".join(rekomendasi.belum_terjawab),
        ikon="?",
    )
    st.stop()

for catatan in rekomendasi.catatan:
    st.warning(catatan, icon=":material/warning:")

if not rekomendasi.berhasil:
    st.stop()

utama = rekomendasi.utama
st.success(f"**{utama.metode}**", icon=":material/check_circle:")
st.markdown(utama.alasan)

if utama.peringatan:
    st.info(utama.peringatan, icon=":material/tune:")

label_eksploratori = pagar.label_eksploratori(utama.metode, penelitian)
if label_eksploratori:
    st.caption(f":material/science: {label_eksploratori}")

kiri, kanan = st.columns([3, 2])

with kiri:
    st.markdown("**Syarat metode ini pada data Anda**")
    ikon = {pmd.TERPENUHI: "✓", pmd.DILANGGAR: "✕", pmd.TIDAK_DIUJI: "–"}
    for syarat in utama.syarat:
        st.markdown(f"{ikon[syarat.status]} **{syarat.nama}** — {syarat.rincian}")

    dilanggar = [s for s in utama.syarat if s.dilanggar]
    if dilanggar:
        st.warning(
            "Syarat yang tidak terpenuhi wajib disebutkan pada laporan, bukan "
            "dihilangkan dari naskah.",
            icon=":material/gavel:",
        )

with kanan:
    if utama.lanjutan:
        st.markdown("**Langkah berikutnya**")
        st.markdown(utama.lanjutan)
    if utama.pembanding:
        st.markdown("**Padanan di perangkat lain**")
        st.caption(utama.pembanding)

# --------------------------------------------------------------------------- #
# Konfirmasi
# --------------------------------------------------------------------------- #

st.divider()
kiri, kanan = st.columns([2, 3])
if not utama.tersedia:
    st.warning(
        f"**{utama.metode}** adalah metode yang paling tepat untuk data Anda, tetapi "
        "belum tersedia di aplikasi ini. Pertimbangkan alternatif di bawah, atau "
        "jalankan metode ini di SPSS maupun R.",
        icon=":material/build:",
    )

if kiri.button(
    "Konfirmasi dan siapkan halamannya",
    type="primary",
    key="pemandu_konfirmasi",
    disabled=not utama.tersedia,
):
    ui.jejak().catat_keputusan(
        f"Memilih {utama.metode}",
        halaman="Pemandu Uji",
        rincian=utama.alasan,
    )
    ui.set_konfigurasi_pemandu(utama.konfig)
    kanan.success(
        f"Tercatat, dan variabel yang Anda pilih sudah disiapkan. Buka halaman "
        f"**{utama.halaman}**; pilihannya sudah terisi.",
        icon=":material/task_alt:",
    )
else:
    kanan.caption(
        "Aplikasi tidak menjalankan uji apa pun sebelum Anda menekan tombol ini. "
        "Mengenali nama kolom bukan alasan yang cukup untuk menyimpulkan."
    )

# --------------------------------------------------------------------------- #
# Alternatif
# --------------------------------------------------------------------------- #

if rekomendasi.alternatif:
    ui.judul_bagian(
        "Mengapa bukan yang lain",
        "Bagian ini yang paling berguna saat penguji bertanya.",
        kicker="Alternatif",
    )
    for alternatif in rekomendasi.alternatif:
        label = f"**{alternatif.metode}** — tidak dipilih"
        if not alternatif.tersedia:
            label += "  ·  belum tersedia di aplikasi ini"
        with st.expander(label):
            st.markdown(alternatif.ditolak_karena)
            if alternatif.peringatan:
                st.caption(alternatif.peringatan)

    ui.show_table(
        rekomendasi.ringkas(),
        "saran_metode.csv",
        bagian="Pemandu uji",
        judul=f"Saran metode untuk tujuan: {pmd.TUJUAN[tujuan]}",
        catatan="Metode utama beserta alternatif yang tidak dipilih dan alasannya.",
    )

ui.interpretation(
    "Saran ini berdasar pada **bentuk** data: skala variabel, jumlah kelompok, "
    "sebaran, dan keseragaman ragam. Ia tidak menilai apakah pertanyaan penelitian "
    "Anda sudah tepat, apakah variabelnya sahih, atau apakah pengamatannya benar-benar "
    "saling bebas. Bila teori Anda menuntut metode lain, teori yang menang — asalkan "
    "alasannya Anda sebutkan."
)
