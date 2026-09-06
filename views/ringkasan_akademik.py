"""Laporan akademik: pelaporan bergaya jurnal, tabel APA, dan kalimat siap salin."""

from __future__ import annotations

from html import escape

import streamlit as st

from nalardata import ekspor, kesimpulan_ui as kui
from nalardata import naskah as nk
from nalardata import sumber as sm
from nalardata import ui

analisis, laporan, lengkap = kui.buka_ringkasan(
    "Laporan Akademik",
    "Ditulis untuk mahasiswa, dosen, dan pengajar: statistik uji, derajat bebas, nilai "
    "p, ukuran efek, dan catatan asumsi.",
    "ringkasan_akademik",
    "akademik",
)

st.subheader("Ikhtisar hasil")
st.html(
    '<div class="mva-quote"><div class="qh">Abstrak temuan</div><div class="qb">'
    + escape(" ".join(t.ringkas for t in laporan.temuan))
    + "</div></div>"
)

st.subheader("Temuan dan pelaporan statistik")
# Indeks sel dibangun sekali, lalu dipakai seluruh paragraf pada halaman ini.
indeks_sumber = sm.indeks(laporan.tabel)

if lengkap:
    for nomor, temuan in enumerate(laporan.temuan):
        st.markdown(f"**{temuan.judul}**")
        st.caption(f"Metode: {temuan.metode}")
        st.write(temuan.akademik)
        ui.sumber_angka(temuan.akademik, indeks_sumber, f"temuan_{nomor}")
else:
    # Ringkasan cukup memuat inti tiap temuan; uraian lengkapnya ada di laporan lengkap.
    kui.daftar_bernomor([(t.judul, t.ringkas, None) for t in laporan.temuan])

if lengkap and laporan.tabel:
    st.subheader("Tabel hasil")
    for nomor, (judul, tabel, catatan) in laporan.tabel.items():
        st.markdown(f"**{nomor}.** {judul}")
        ui.show_table(tabel, f"{nomor.lower().replace(' ', '_')}_akademik.csv")
        if catatan:
            st.caption(f"*Catatan.* {catatan}")

if lengkap and laporan.paragraf:
    st.subheader("Kalimat siap salin")
    st.caption(
        "Paragraf berikut mengikuti konvensi pelaporan statistik dan dapat langsung "
        "disalin ke naskah. Sesuaikan nama variabel dengan istilah pada penelitian Anda."
    )
    for nomor, paragraf in enumerate(laporan.paragraf):
        st.markdown(f"**{paragraf.bagian}**")
        st.code(paragraf.teks, language=None, wrap_lines=True)
        ui.sumber_angka(paragraf.teks, indeks_sumber, f"paragraf_{nomor}")

st.subheader("Keterbatasan dan saran penelitian lanjutan")
kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

if lengkap:
    st.subheader("Rujukan ambang yang dipakai")
    st.caption(
        "Daftar ini memuat rujukan ambang statistik yang dipakai aplikasi, bukan rujukan "
        "teoretis penelitian Anda. Sesuaikan gaya sitasi dengan pedoman institusi."
    )
    for rujukan in laporan.rujukan:
        st.markdown(f"- {rujukan}")

    kui.analisis_yang_dilewati(laporan)

# --------------------------------------------------------------------------- #
# Naskah skripsi
# --------------------------------------------------------------------------- #

ui.judul_bagian(
    "Kerangka naskah skripsi",
    "Bahan yang sudah dikumpulkan aplikasi disusun ulang mengikuti urutan bab, "
    "bukan urutan analisis.",
    kicker="Naskah",
)

if not ui.paket_aktif().punya("unduh_laporan"):
    st.info(
        "Unduhan naskah tersedia mulai paket Mahasiswa & Pengajar.",
        icon=":material/lock:",
    )
else:
    st.caption(
        "Yang dihasilkan **kerangka berisi**, bukan naskah jadi. Pembahasan teoretis "
        "dan kaitan dengan penelitian terdahulu harus Anda tulis sendiri — bagian "
        "itulah yang dinilai penguji."
    )
    kiri, tengah, kanan = st.columns([2, 1.4, 1.4])
    gaya = kiri.selectbox(
        "Bagian naskah",
        list(nk.GAYA),
        format_func=lambda k: nk.GAYA[k],
        key="naskah_gaya",
    )
    format_naskah = tengah.selectbox(
        "Format berkas",
        ["docx", "pdf", "html", "md"],
        format_func=lambda k: ekspor.FORMAT[k].nama,
        key="naskah_format",
    )

    try:
        dokumen = nk.susun(laporan, gaya, ui.penelitian(), ui.kamus())
        berkas = ekspor.bangun(dokumen, format_naskah)
    except Exception as galat:  # noqa: BLE001 - kegagalan ekspor tidak menghentikan halaman
        st.error(f"Naskah gagal disusun: {galat}", icon=":material/error:")
    else:
        nama = ekspor.nama_berkas(dokumen, format_naskah)
        kanan.download_button(
            f"Unduh {nk.GAYA[gaya].split('—')[0].strip()}",
            berkas,
            file_name=nama,
            mime=ekspor.FORMAT[format_naskah].mime,
            type="primary",
            width="stretch",
            key=f"unduh_naskah_{gaya}_{format_naskah}",
        )
        kanan.caption(f"`{nama}`")

        with st.expander("Pratinjau isi", expanded=False):
            for blok in dokumen.blok[:24]:
                if blok.jenis == "subjudul":
                    st.markdown(f"**{blok.teks}**")
                elif blok.jenis == "catatan":
                    st.caption(blok.teks)
                elif blok.jenis == "poin" and blok.poin:
                    for butir in blok.poin[:6]:
                        st.markdown(f"- {butir}")
                elif blok.jenis == "tabel" and blok.tabel is not None:
                    st.dataframe(blok.tabel, width="stretch", hide_index=True)
                elif blok.teks and blok.jenis != "judul":
                    st.write(blok.teks)

kui.unduhan(laporan, "akademik", lengkap)
