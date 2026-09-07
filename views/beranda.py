"""Beranda: penjelasan aplikasi dan titik mulai, atau dasbor ringkas bila data sudah ada.

Sebelum ada data, halaman ini adalah pintu masuk satu-satunya yang dibutuhkan pengguna
baru — penjelasan singkat lalu langsung tiga cara memuat data (dipakai ulang dari
``muat_data.render``, bukan disalin, sehingga hanya ada satu tempat berat tombol unggah
bekerja). Sesudah ada data, halaman berganti jadi dasbor dengan **satu tombol tindakan
berikutnya** yang selalu benar — pengguna tidak perlu menebak dari delapan tab.
"""

from __future__ import annotations

import streamlit as st

from nalardata import audit as ad
from nalardata import formatting, ui
from views import muat_data

LANGKAH = [
    ("Muat data", "Unggah berkas Anda, coba data contoh, atau lanjutkan proyek lama."),
    ("Periksa mutu", "Rapor Data memeriksa nilai kosong, pencilan, dan kejanggalan lain."),
    ("Pilih metode", "Pemandu menyarankan uji yang tepat, atau pilih sendiri secara manual."),
    ("Baca laporan", "Interpretasi dan statistik deskriptif disusun otomatis dari yang Anda jalankan."),
]

METODE = [
    ("Uji beda & korelasi", "Uji-t, ANOVA, non-parametrik, Pearson/Spearman/Kendall, chi-square"),
    ("Regresi", "Regresi linear berganda, regresi logistik biner, regresi moderasi (MRA)"),
    ("MANOVA & Diskriminan", "Uji beda multivariat, klasifikasi kelompok, korelasi kanonik"),
    ("CFA, Jalur & SEM", "Model struktural, muatan faktor, kecocokan model"),
    ("PCA & Analisis Faktor", "Reduksi dimensi, faktor laten, rotasi varimax/promax"),
    ("Analisis Klaster", "K-Means, hierarki + dendrogram, DBSCAN"),
    ("Reliabilitas & Validitas", "Alpha, omega, composite reliability, AVE, HTMT"),
]

PETA = [
    ("Belum tahu harus mulai dari mana", "Beranda — ikuti tombol tindakan berikutnya"),
    ("Menentukan sebab-akibat/populasi berlaku ke siapa", "Rencana"),
    ("Memuat atau mengganti data, entri manual, kamus variabel", "Data"),
    ("Memeriksa nilai kosong, pencilan, normalitas", "Mutu Data"),
    ("Menjalankan uji statistik", "Analisis"),
    ("Membaca hasil dengan interpretasinya, mengunduh naskah", "Laporan"),
    ("Berlatih menjawab pertanyaan penguji/atasan", "Simulasi Sidang"),
]


def render(df, kamus, penelitian) -> None:
    if df is None:
        _sebelum_data(kamus, penelitian)
    else:
        _dashboard(df, kamus, penelitian)
    _panduan_penggunaan()


def _sebelum_data(kamus, penelitian) -> None:
    st.html(
        '<div class="mva-head"><h1>Analisis multivariat, tanpa harus jadi ahli statistik</h1>'
        '<p class="desc">Unggah data Anda. NalarData memeriksa mutunya, membantu memilih '
        "metode, menjalankan analisis, dan menyusun laporan yang mudah dipahami — "
        "lengkap dengan interpretasinya, bukan sekadar angka.</p><hr></div>"
    )
    st.caption(
        "🔒 Data diproses hanya untuk sesi ini dan tidak dikirim ke pihak lain. "
        "Simpan berkas proyek bila ingin melanjutkan pekerjaan nanti."
    )
    muat_data.render(None, kamus, penelitian, key_prefix="beranda_")

    st.divider()
    ui.judul_bagian("Alur kerja empat langkah", kicker="Cara kerja")
    ui.langkah_grid(LANGKAH)


def _dashboard(df, kamus, penelitian) -> None:
    hasil_audit = ad.jalankan_audit(df, kamus)
    status = hasil_audit.status()
    sudah_dijalankan = ui.keranjang().bagian()

    ui.judul_bagian(
        f"Data aktif: {st.session_state.get(ui.NAME_KEY, 'data')}",
        kicker="Dasbor",
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ukuran", f"{formatting.num(len(df))} × {df.shape[1]}")
    k2.metric(
        "Rapor Data",
        {"kritis": "Ada temuan kritis", "peringatan": "Perlu dicermati", "baik": "Bersih"}[status],
    )
    k3.metric("Metode dijalankan", len(sudah_dijalankan))
    k4.metric("Rencana penelitian", "Belum diisi" if penelitian.kosong() else "Terisi")

    if status == "kritis":
        st.error(hasil_audit.kesimpulan(), icon=":material/error:")
    elif status == "peringatan":
        st.warning(hasil_audit.kesimpulan(), icon=":material/warning:")

    st.divider()
    if status == "kritis" or (status == "peringatan" and not sudah_dijalankan):
        st.info(
            "**Langkah berikutnya:** buka tab **Mutu Data** di atas untuk "
            "memeriksa dan menindaklanjuti temuannya.",
            icon=":material/arrow_forward:",
        )
    elif not sudah_dijalankan:
        st.info(
            "**Langkah berikutnya:** buka tab **Analisis** di atas, lalu pilih "
            "**Dipandu aplikasi** agar NalarData menyarankan metode yang cocok "
            "dengan data Anda.",
            icon=":material/arrow_forward:",
        )
    else:
        st.info(
            f"**Langkah berikutnya:** buka tab **Laporan** di atas — "
            f"{len(sudah_dijalankan)} metode sudah punya hasil yang siap dibaca "
            "lengkap dengan interpretasinya.",
            icon=":material/arrow_forward:",
        )


def _panduan_penggunaan() -> None:
    st.divider()
    with st.expander("📖 Panduan penggunaan aplikasi"):
        ui.judul_bagian(
            "Delapan tab, satu alur",
            "Tab di atas disusun mengikuti urutan penelitian sungguhan — dari kiri ke "
            "kanan adalah urutan yang disarankan, tetapi tidak ada yang mengunci Anda "
            "untuk mengikutinya persis.",
            kicker="Panduan",
        )
        st.markdown(
            "- **Rencana** — opsional, tetapi lebih baik diisi sebelum melihat data. "
            "Menentukan apakah kesimpulan nanti boleh berbunyi sebab-akibat.\n"
            "- **Data** — memuat data (unggah/contoh/proyek), entri manual, dan kamus "
            "variabel (skala & peran tiap kolom).\n"
            "- **Mutu Data** — Rapor Data (nilai kosong, duplikat, pencilan) dan "
            "Eksplorasi (deskriptif, distribusi, normalitas).\n"
            "- **Analisis** — pilih **Dipandu aplikasi** bila belum yakin uji apa "
            "yang tepat, atau **Pilih metode manual** bila sudah tahu.\n"
            "- **Laporan** — otomatis tersusun dari metode yang benar-benar Anda "
            "jalankan, dengan interpretasi dan statistik deskriptif, dalam tiga ragam "
            "bahasa (Umum/Akademik/Profesional).\n"
            "- **Simulasi Sidang** — langkah terakhir, berlatih menjawab pertanyaan "
            "dari analisis Anda sendiri.\n"
            "- **Akun** — paket langganan yang berlaku."
        )

        st.markdown("**Kalau Anda ingin ...**")
        st.dataframe(
            {"Yang ingin dilakukan": [p[0] for p in PETA], "Buka tab": [p[1] for p in PETA]},
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Metode yang tersedia di tab Analisis**")
        st.dataframe(
            {"Kelompok": [m[0] for m in METODE], "Mencakup": [m[1] for m in METODE]},
            width="stretch",
            hide_index=True,
        )
