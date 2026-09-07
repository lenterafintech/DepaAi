"""Ringkasan Otomatis: satu baterai analisis, tiga ragam bahasa.

Menggantikan tiga halaman terpisah (Laporan Umum/Akademik/Profesional) yang
sebelumnya membingungkan karena tampak seperti tiga laporan berbeda, padahal
ketiganya menjalankan analisis yang **sama persis** dan hanya berbeda cara
menuliskannya. Register di sini hanya mengganti bahasa penyajian; angkanya
selalu identik lintas register karena berasal dari satu ``nr.Laporan`` yang
sama — lihat ``tests/test_sumber.py`` untuk uji yang menjaga hal ini.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from nalardata import ekspor, kesimpulan_ui as kui
from nalardata import naskah as nk
from nalardata import narrative as nr
from nalardata import sumber as sm
from nalardata import ui

REGISTER = {
    "eksekutif": ("Umum", "ringkasan_eksekutif", "Untuk pimpinan dan pembaca non-statistik"),
    "akademik": ("Akademik", "ringkasan_akademik", "Untuk mahasiswa, dosen, dan pengajar"),
    "profesional": ("Profesional", "ringkasan_profesional", "Untuk analis dan praktisi"),
}


def render(df) -> None:
    kui.pasang_gaya()
    paket = ui.paket_aktif()
    tersedia = {k: v for k, v in REGISTER.items() if paket.punya(v[1])}
    if not tersedia:
        st.info("Ringkasan otomatis belum termasuk paket Anda.", icon=":material/lock:")
        return

    register = st.radio(
        "Ditulis untuk",
        list(tersedia),
        format_func=lambda k: f"{tersedia[k][0]} — {tersedia[k][2]}",
        horizontal=True,
        key="ringkasan_register",
    )
    terkunci = [v[0] for k, v in REGISTER.items() if k not in tersedia]
    if terkunci:
        st.caption(f"Register {', '.join(terkunci)} tersedia pada paket yang lebih tinggi.")

    hasil = kui.siapkan_laporan(df)
    if hasil is None:
        return
    analisis, laporan = hasil
    lengkap = kui.pilih_kedalaman(register)
    kui.kartu_headline(laporan)
    st.subheader("Status pemeriksaan")
    kui.kartu_lampu(laporan)

    if register == "eksekutif":
        _render_eksekutif(analisis, laporan, lengkap)
    elif register == "akademik":
        _render_akademik(analisis, laporan, lengkap)
    else:
        _render_profesional(analisis, laporan, lengkap)


def _render_eksekutif(analisis: nr.Analisis, laporan: nr.Laporan, lengkap: bool) -> None:
    if laporan.pendorong:
        st.subheader("Peringkat pendorong")
        kiri, kanan = st.columns([1.15, 1])
        with kiri:
            kui.batang_pendorong(laporan)
        with kanan:
            if any(p.kinerja is not None for p in laporan.pendorong):
                st.plotly_chart(kui.matriks_prioritas(laporan), width="stretch")
                st.caption(
                    "Hanya faktor terkuat yang diberi nama pada grafik; arahkan kursor ke "
                    "titik lain untuk melihat namanya."
                )
        ui.interpretation(
            "Panjang batang menunjukkan seberapa kuat pengaruh sebuah faktor dibanding "
            "faktor terkuat. Pada matriks di sampingnya, faktor di kuadran kiri atas "
            "adalah yang penting tetapi kinerjanya masih di bawah rata-rata — di sanalah "
            "perbaikan paling terasa hasilnya."
        )

    st.subheader("Apa yang ditemukan")
    if lengkap:
        for temuan in laporan.temuan:
            st.markdown(f"**{temuan.judul}**")
            st.caption(f"Metode: {temuan.metode}")
            st.write(temuan.eksekutif)
    else:
        for temuan in laporan.temuan:
            with st.expander(f"**{temuan.judul}** — {temuan.ringkas}"):
                st.write(temuan.eksekutif)

    st.subheader("Rekomendasi tindakan")
    kui.daftar_bernomor([(r.judul, r.alasan, r.prioritas) for r in laporan.rekomendasi])

    st.subheader("Batas kesimpulan")
    kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

    if lengkap and laporan.tabel:
        st.subheader("Tabel hasil")
        st.caption(
            "Angka rinci di balik kesimpulan di atas. Bagian ini boleh dilewati bila "
            "Anda hanya memerlukan kesimpulannya."
        )
        for nomor, (judul_tabel, tabel, catatan) in laporan.tabel.items():
            st.markdown(f"**{nomor}.** {judul_tabel}")
            ui.show_table(tabel, f"{nomor.lower().replace(' ', '_')}_umum.csv")
            if catatan:
                st.caption(f"*Catatan.* {catatan}")

    if lengkap:
        kui.analisis_yang_dilewati(laporan)

    kui.unduhan(laporan, "eksekutif", lengkap)


def _render_akademik(analisis: nr.Analisis, laporan: nr.Laporan, lengkap: bool) -> None:
    st.subheader("Ikhtisar hasil")
    st.html(
        '<div class="mva-quote"><div class="qh">Abstrak temuan</div><div class="qb">'
        + escape(" ".join(t.ringkas for t in laporan.temuan))
        + "</div></div>"
    )

    st.subheader("Temuan dan pelaporan statistik")
    indeks_sumber = sm.indeks(laporan.tabel)

    if lengkap:
        for nomor, temuan in enumerate(laporan.temuan):
            st.markdown(f"**{temuan.judul}**")
            st.caption(f"Metode: {temuan.metode}")
            st.write(temuan.akademik)
            ui.sumber_angka(temuan.akademik, indeks_sumber, f"temuan_{nomor}")
    else:
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
            "Daftar ini memuat rujukan ambang statistik yang dipakai aplikasi, bukan "
            "rujukan teoretis penelitian Anda. Sesuaikan gaya sitasi dengan pedoman "
            "institusi."
        )
        for rujukan in laporan.rujukan:
            st.markdown(f"- {rujukan}")
        kui.analisis_yang_dilewati(laporan)

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


def _render_profesional(analisis: nr.Analisis, laporan: nr.Laporan, lengkap: bool) -> None:
    st.subheader("Metrik kunci")
    metrik = st.columns(4)
    slot = 0
    if analisis.regresi is not None:
        metrik[slot].metric("R² model", nr.num(analisis.regresi.model.rsquared, 3))
        slot += 1
    if analisis.logistik is not None:
        metrik[slot].metric("AUC", nr.num(analisis.logistik.auc, 3))
        slot += 1
    if analisis.klaster is not None:
        metrik[slot].metric("Segmen", analisis.klaster.n_clusters)
        slot += 1
    if analisis.diskriminan is not None and slot < 4:
        diskriminan = analisis.diskriminan
        punya_cv = bool(pd.notna(diskriminan.cv_accuracy))
        metrik[slot].metric(
            "Akurasi CV" if punya_cv else "Akurasi latih",
            nr.pct((diskriminan.cv_accuracy if punya_cv else diskriminan.accuracy) * 100),
            help=None if punya_cv else "Validasi silang gagal: ada kelompok beranggota < 2.",
        )
        slot += 1
    if analisis.vif is not None and not analisis.vif.empty and slot < 4:
        metrik[slot].metric("VIF maks", nr.num(float(analisis.vif["VIF"].max())))

    if laporan.pendorong:
        st.subheader("Kontribusi fitur")
        kui.batang_pendorong(laporan)

    st.subheader("Temuan teknis")
    if lengkap:
        for temuan in laporan.temuan:
            st.markdown(f"**{temuan.judul}**")
            st.caption(f"Metode: {temuan.metode}")
            st.write(temuan.profesional)
    else:
        for temuan in laporan.temuan:
            with st.expander(f"**{temuan.judul}** — {temuan.metode}"):
                st.write(temuan.profesional)

    asumsi = nr.tabel_asumsi(analisis)
    if not asumsi.empty:
        st.subheader("Ringkasan pemeriksaan asumsi")
        ui.show_table(asumsi, "pemeriksaan_asumsi.csv")

    st.subheader("Tindak lanjut yang disarankan")
    kui.daftar_bernomor([(r.judul, r.alasan, r.prioritas) for r in laporan.rekomendasi])

    st.subheader("Risiko dan batas pemakaian")
    kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

    if lengkap and laporan.tabel:
        st.subheader("Tabel hasil")
        for nomor, (judul, tabel, catatan) in laporan.tabel.items():
            st.markdown(f"**{nomor}.** {judul}")
            ui.show_table(tabel, f"{nomor.lower().replace(' ', '_')}_profesional.csv")
            if catatan:
                st.caption(f"*Catatan.* {catatan}")

    if lengkap:
        kui.analisis_yang_dilewati(laporan)

    kui.unduhan(laporan, "profesional", lengkap)
