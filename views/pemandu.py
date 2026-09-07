"""Pemandu Uji: memilih metode dengan melihat data Anda, bukan diagram alur.

Halaman ini menjawab pertanyaan yang paling sering diajukan penulis skripsi — "uji
apa yang harus saya pakai" — dengan memeriksa data yang sungguh ada, lalu menyebutkan
mengapa alternatifnya tidak dipilih.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nalardata import audit as ad
from nalardata import formatting, kamus as km, pemandu as pmd, ui


def _prasi_tunggal(kandidat: list[str], kamus: km.Kamus, *peran: str) -> str | None:
    """Kolom pertama di antara ``kandidat`` yang perannya sudah dikonfirmasi di Kamus Variabel."""
    for nama in kamus.dengan_peran(*peran):
        if nama in kandidat:
            return nama
    return None


def _prasi_banyak(kandidat: list[str], kamus: km.Kamus, *peran: str) -> list[str]:
    return [n for n in kamus.dengan_peran(*peran) if n in kandidat]


def _status_metode(utama: pmd.Saran) -> tuple[str, str, str]:
    """(jenis_pil, ikon, label) untuk status visual kartu.

    Ikon+teks selalu menyertai warna (baik=hijau/perhatian=kuning/kritis=merah) —
    warna tidak pernah jadi satu-satunya penanda, sama seperti simbol ✓/✕/– yang
    sudah dipakai di baris syarat.
    """
    if not utama.tersedia:
        return "kritis", "✕", "Belum bisa dijalankan di aplikasi ini"
    if any(s.dilanggar for s in utama.syarat):
        return "perhatian", "!", "Layak dipakai dengan catatan"
    return "baik", "✓", "Layak dijalankan"


def _ringkas_variabel(konfig: dict, kamus: km.Kamus) -> list[str]:
    """Variabel yang dipakai metode ini, dalam bahasa sehari-hari — bukan
    outcome/prediktor/kelompok istilah statistik yang belum tentu dikenal."""
    baris = []
    if konfig.get("outcome"):
        baris.append(f"Variabel utama: **{kamus.judul(konfig['outcome'])}**")
    if konfig.get("kelompok"):
        baris.append(f"Kelompok/entitas: **{kamus.judul(konfig['kelompok'])}**")
    if konfig.get("prediktor"):
        nama = ", ".join(kamus.judul(p) for p in konfig["prediktor"])
        baris.append(f"Variabel lain: **{nama}**")
    if konfig.get("berpasangan"):
        baris.append("Pengukuran berpasangan (unit yang sama, diukur berulang).")
    return baris


def _simpan_peran(kamus: km.Kamus, nama: str | None, peran: str) -> None:
    """Menuliskan peran yang baru saja dipilih pengguna kembali ke Kamus Variabel.

    Pengguna awam tidak pernah ditanya "peran" secara abstrak di Kamus Variabel —
    pertanyaan itu hanya masuk akal dalam konteks pertanyaan penelitian yang
    sedang dijawab di sini. ``dikonfirmasi`` sengaja dipertahankan apa adanya:
    menuliskan peran tidak boleh diam-diam membungkam peringatan "skala kolom
    ini masih tebakan", karena keduanya adalah hal yang berbeda.
    """
    if nama and nama in kamus and kamus[nama].peran != peran:
        kamus.tetapkan(nama, peran=peran, dikonfirmasi=kamus[nama].dikonfirmasi)


def render(df, kamus, penelitian) -> None:
    if not ui.butuh_fitur("pemandu"):
        return

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

    # Satu baris ringkas menggantikan dua kotak info besar yang sebelumnya
    # mendorong Langkah 1 turun layar (poin 9: kepadatan layar) — penjelasan
    # lengkap tetap ada, dipindah ke expander collapsed di bawahnya. Peringatan
    # skala DI SINI sengaja hanya berupa hitungan ringkas seluruh data; peringatan
    # yang menyebut NAMA variabel muncul lebih spesifik dekat kartu hasil, hanya
    # bila variabel yang benar-benar dipakai belum dikonfirmasi (lihat di bawah).
    belum = kamus.perlu_diperiksa()
    kiri_ringkas, kanan_ringkas = st.columns([5, 2])
    kiri_ringkas.caption(
        "Pemandu membaca **bentuk** data, bukan **maksud** penelitian — lihat batas "
        "kemampuannya di bawah."
    )
    if belum:
        with kanan_ringkas:
            st.html(ui.pil(f"{formatting.num(len(belum))} skala belum dikonfirmasi", "perhatian"))

    with st.expander("Batas kemampuan Pemandu"):
        st.markdown(
            "Pemandu membaca **bentuk** data, bukan **maksud** penelitian. Ia tidak tahu "
            "apakah pengamatan Anda benar-benar saling bebas, apakah variabelnya benar-benar "
            "mengukur yang Anda maksud, atau apakah pertanyaan penelitiannya sudah tepat. "
            "Saran di bawah adalah titik awal yang berdasar, bukan keputusan akhir."
        )
        if belum:
            st.markdown(
                f"{formatting.num(len(belum))} kolom pada seluruh data masih memakai "
                "tebakan skala aplikasi — kolom Likert yang tercatat sebagai rasio akan "
                "mengantar Anda ke uji parametrik yang keliru. Periksa kapan saja di tab "
                "**Kamus Variabel**."
            )

    if penelitian is None or penelitian.kosong() or not penelitian.lengkap():
        st.info(
            "Rencana penelitian belum lengkap. Pemandu tetap dapat dipakai penuh — "
            "yang berubah hanya bahasa rekomendasi (boleh atau tidaknya kata "
            "'pengaruh'/'menyebabkan') dan batas kesimpulan yang ditampilkan di kartu "
            "hasil. Isi di tab **Rencana** kapan saja, termasuk setelah melihat saran "
            "di sini.",
            icon=":material/assignment:",
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

    ui.judul_bagian(
        "Variabel mana yang terlibat?",
        "Kolom yang perannya sudah Anda konfirmasi di Kamus Variabel otomatis "
        "terisi di sini — periksa dan ubah bila keliru.",
        kicker="Langkah 2",
    )

    # Kolom berperan "id" (dikenali otomatis dari kamus, mis. nomor responden) tidak
    # pernah masuk akal sebagai outcome/prediktor/kelompok — itu sekadar identitas,
    # bukan sesuatu yang diukur. Kamus.numerik() sudah menerapkan aturan yang sama;
    # di sini disamakan untuk seluruh daftar kandidat, termasuk yang kategorik.
    semua = [c for c in df.columns if not (c in kamus and kamus[c].peran == "id")]
    numerik = kamus.numerik()
    kategorik = kamus.kategorik()


    def _label(nama: str) -> str:
        butir = kamus.variabel.get(nama)
        return f"{kamus.judul(nama)} ({butir.skala})" if butir else nama


    def _indeks(kandidat: list[str], nilai: str | None) -> int:
        return ([None] + kandidat).index(nilai) if nilai in kandidat else 0


    outcome = prediktor = kelompok = None
    berpasangan = False
    kovariat: list[str] = []

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
                "Kolom pengukuran berulang",
                numerik,
                key="pemandu_ulang",
                format_func=_label,
                help="Misalnya skor sebelum dan sesudah pelatihan, diukur pada orang yang sama.",
            )
            outcome = pilihan[0] if pilihan else None
            prediktor = pilihan[1:]
        else:
            kiri, kanan = st.columns(2)
            outcome = kiri.selectbox(
                "Variabel yang dibandingkan",
                [None] + semua,
                index=_indeks(semua, _prasi_tunggal(semua, kamus, "outcome")),
                format_func=lambda k: "— pilih —" if k is None else _label(k),
                key="pemandu_outcome_beda",
                help="Angka atau kategori yang ingin Anda lihat bedanya antar kelompok, misalnya skor ujian.",
            )
            kelompok = kanan.selectbox(
                "Penanda kelompok",
                [None] + kategorik,
                index=_indeks(kategorik, _prasi_tunggal(kategorik, kamus, "kelompok")),
                format_func=lambda k: "— pilih —" if k is None else _label(k),
                key="pemandu_kelompok",
                help="Kolom yang membagi responden menjadi beberapa kelompok, misalnya jenis kelamin atau kelas perlakuan.",
            )

    elif tujuan == "membandingkan_banyak_outcome":
        kiri, kanan = st.columns([2, 3])
        kelompok = kiri.selectbox(
            "Penanda kelompok",
            [None] + kategorik,
            index=_indeks(kategorik, _prasi_tunggal(kategorik, kamus, "kelompok")),
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_kelompok_manova",
            help="Kolom yang membagi responden menjadi beberapa kelompok.",
        )
        prediktor = kanan.multiselect(
            "Variabel hasil (dependen) yang dibandingkan sekaligus",
            numerik,
            key="pemandu_outcome_manova",
            format_func=_label,
            help="Pilih sekurang-kurangnya dua ukuran hasil yang diuji bersama, "
            "misalnya nilai ujian dan skor motivasi.",
        )
        kandidat_kovariat = [k for k in numerik if k not in {kelompok, *prediktor}]
        kovariat = st.multiselect(
            "Kovariat (opsional) — mengubah saran menjadi MANCOVA",
            kandidat_kovariat,
            key="pemandu_kovariat_manova",
            format_func=_label,
            help="Variabel numerik yang ingin dikendalikan sebelum kelompok "
            "dibandingkan, misalnya usia sebagai kovariat saat membandingkan "
            "skor tes antar kelas. Kosongkan bila tidak ada.",
        )

    elif tujuan in {"menguji_mediasi", "menguji_moderasi"}:
        if tujuan == "menguji_mediasi":
            label_m = "Variabel perantara (mediator, M)"
            bantuan_m = "Variabel yang Anda duga menjadi jalur perantara antara X dan Y."
        else:
            label_m = "Variabel moderator (M)"
            bantuan_m = "Variabel yang Anda duga mengubah kekuatan pengaruh X terhadap Y."

        kiri, tengah, kanan = st.columns(3)
        outcome = kiri.selectbox(
            "Variabel hasil (Y)",
            [None] + numerik,
            index=_indeks(numerik, _prasi_tunggal(numerik, kamus, "outcome")),
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_y_medmod",
            help="Angka yang menjadi hasil akhir yang ingin dijelaskan.",
        )
        kandidat_x = [k for k in numerik if k != outcome]
        x = tengah.selectbox(
            "Variabel bebas (X)",
            [None] + kandidat_x,
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_x_medmod",
            help="Variabel yang diduga memengaruhi Y.",
        )
        kandidat_m = [k for k in numerik if k not in {outcome, x}]
        m = kanan.selectbox(
            label_m,
            [None] + kandidat_m,
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_m_medmod",
            help=bantuan_m,
        )
        prediktor = [v for v in (x, m) if v]

    elif tujuan in {"menghubungkan"}:
        pilihan = st.multiselect(
            "Dua variabel yang ingin dihubungkan",
            semua,
            max_selections=2,
            key="pemandu_hubungan",
            format_func=_label,
            help="Urutan tidak penting — Pemandu hanya menguji apakah keduanya bergerak bersamaan.",
        )
        outcome = pilihan[0] if pilihan else None
        prediktor = pilihan[1:]

    elif tujuan in {"memperkirakan_nilai", "memperkirakan_kategori"}:
        kandidat = numerik if tujuan == "memperkirakan_nilai" else semua
        kiri, kanan = st.columns([1, 2])
        outcome = kiri.selectbox(
            "Yang ingin diperkirakan",
            [None] + kandidat,
            index=_indeks(kandidat, _prasi_tunggal(kandidat, kamus, "outcome")),
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_outcome_reg",
            help="Angka atau kategori yang nilainya ingin Anda jelaskan atau perkirakan.",
        )
        kandidat_prediktor = [k for k in numerik if k != outcome]
        prediktor = kanan.multiselect(
            "Variabel penjelas",
            kandidat_prediktor,
            default=_prasi_banyak(kandidat_prediktor, kamus, "prediktor", "kovariat"),
            key="pemandu_prediktor",
            format_func=_label,
            help="Variabel yang Anda duga ikut menentukan naik-turunnya nilai di atas.",
        )

    elif tujuan == "menganalisis_panel":
        # Kolom entitas panel (mis. kode perusahaan) sengaja diambil dari SELURUH
        # kolom, bukan ``semua`` — kolom berperan "id" yang dikecualikan di tempat
        # lain justru lazim menjadi penanda entitas di sini. Entitas harus muncul
        # berulang (nunique < jumlah baris); kolom bernilai unik per baris bukan
        # entitas panel sama sekali.
        kandidat_entitas = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) < len(df)]
        kelompok = st.selectbox(
            "Kolom entitas (unit yang diamati berulang, mis. kode perusahaan)",
            [None] + kandidat_entitas,
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_entitas_panel",
            help="Kolom yang menandai unit yang sama diamati berulang sepanjang waktu.",
        )
        kandidat_y = [k for k in numerik if k != kelompok]
        kiri, kanan = st.columns([1, 2])
        outcome = kiri.selectbox(
            "Variabel hasil (Y)",
            [None] + kandidat_y,
            index=_indeks(kandidat_y, _prasi_tunggal(kandidat_y, kamus, "outcome")),
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_outcome_panel",
        )
        kandidat_x = [k for k in numerik if k not in {outcome, kelompok}]
        prediktor = kanan.multiselect(
            "Prediktor (X)",
            kandidat_x,
            key="pemandu_prediktor_panel",
            format_func=_label,
        )

    elif tujuan == "meramalkan_waktu":
        outcome = st.selectbox(
            "Variabel yang diramalkan",
            [None] + numerik,
            index=_indeks(numerik, _prasi_tunggal(numerik, kamus, "outcome")),
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_outcome_arima",
            help="Deret angka yang ingin diramalkan nilainya di masa depan.",
        )

    elif tujuan == "menganalisis_teks":
        kandidat_teks = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        outcome = st.selectbox(
            "Kolom teks yang dianalisis",
            [None] + kandidat_teks,
            format_func=lambda k: "— pilih —" if k is None else _label(k),
            key="pemandu_outcome_teks",
            help="Kolom berisi kalimat atau paragraf, misalnya jawaban terbuka atau komentar.",
        )

    else:
        label, bantuan, peran_disukai = {
            "meringkas": (
                "Variabel yang ingin diringkas",
                "Belasan butir kuesioner yang Anda duga sebenarnya mengukur beberapa hal saja.",
                ("indikator", "prediktor"),
            ),
            "mengelompokkan": (
                "Variabel dasar pengelompokan",
                "Variabel angka yang dipakai untuk menemukan kemiripan antar responden.",
                ("prediktor",),
            ),
            "menguji_model": (
                "Indikator penyusun konstruk",
                "Seluruh butir yang menyusun konstruk-konstruk dalam model Anda, digabung dari semua konstruk.",
                ("indikator",),
            ),
            "mutu_instrumen": (
                "Butir penyusun satu konstruk",
                "Butir kuesioner yang dirancang untuk mengukur satu hal yang sama.",
                ("indikator",),
            ),
        }[tujuan]
        prediktor = st.multiselect(
            label,
            semua,
            default=_prasi_banyak(semua, kamus, *peran_disukai),
            key="pemandu_banyak",
            format_func=_label,
            help=bantuan,
        )

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
        penelitian=penelitian,
        kovariat=kovariat,
    )

    ui.judul_bagian("Yang disarankan", kicker="Langkah 3")

    if rekomendasi.belum_terjawab:
        ui.keadaan_kosong(
            "Masih ada yang perlu Anda tentukan",
            " ".join(rekomendasi.belum_terjawab),
            ikon="?",
        )
        return

    for catatan in rekomendasi.catatan:
        st.warning(catatan, icon=":material/warning:")

    if not rekomendasi.berhasil:
        return

    if rekomendasi.perlu_konfirmasi:
        nama_variabel = ", ".join(f"**{kamus.judul(n)}**" for n in rekomendasi.perlu_konfirmasi)
        st.warning(
            f"Rekomendasi sementara — skala {nama_variabel} belum dikonfirmasi. "
            "Saran di bawah memakai tebakan aplikasi; periksa di tab **Kamus Variabel** "
            "untuk memastikan saran ini tetap benar.",
            icon=":material/rule:",
        )

    # Susunan kartu di bawah ini mengikuti urutan yang diminta: metode utama →
    # alasan → variabel dipakai → bukti dari data → asumsi terpenuhi/dilanggar →
    # hal yang perlu dicermati → hal yang tidak dapat diperiksa aplikasi →
    # (alternatif ada di bagiannya sendiri, di bawah) → ukuran efek → padanan
    # SPSS/R → batas kesimpulan dari Rencana.
    utama = rekomendasi.utama
    jenis_status, ikon_status, label_status = _status_metode(utama)
    st.success(f"**{utama.metode}**", icon=":material/check_circle:")
    st.html(ui.pil(f"{ikon_status} {label_status}", jenis_status))
    st.markdown(utama.alasan)

    if utama.status_bukti:
        st.caption(f":material/science: {utama.status_bukti}")

    kiri, kanan = st.columns([3, 2])

    with kiri:
        variabel = _ringkas_variabel(utama.konfig, kamus)
        if variabel:
            st.markdown("**Variabel yang dipakai**")
            for baris in variabel:
                st.markdown(f"- {baris}")

        bukti = [s for s in utama.syarat if s.terpenuhi]
        if bukti:
            st.markdown("**Bukti dari data**")
            for syarat in bukti:
                st.markdown(f"- {syarat.rincian}")

        st.markdown("**Asumsi terpenuhi/dilanggar**")
        ikon = {pmd.TERPENUHI: "✓", pmd.DILANGGAR: "✕", pmd.TIDAK_DIUJI: "–"}
        for syarat in utama.syarat:
            st.markdown(f"{ikon[syarat.status]} **{syarat.nama}** — {syarat.rincian}")

        dilanggar = [s for s in utama.syarat if s.dilanggar]
        if dilanggar or utama.peringatan:
            st.markdown("**Hal yang perlu dicermati**")
            if utama.peringatan:
                st.info(utama.peringatan, icon=":material/tune:")
            if dilanggar:
                st.warning(
                    "Syarat yang tidak terpenuhi wajib disebutkan pada laporan, bukan "
                    "dihilangkan dari naskah.",
                    icon=":material/gavel:",
                )

        if utama.tidak_dapat_diperiksa:
            st.markdown("**Hal yang tidak dapat diperiksa aplikasi**")
            for hal in utama.tidak_dapat_diperiksa:
                st.markdown(f"- {hal}")

    with kanan:
        if utama.lanjutan:
            st.markdown("**Langkah berikutnya**")
            st.markdown(utama.lanjutan)
        if utama.ukuran_efek:
            st.markdown("**Ukuran efek yang dilaporkan**")
            st.caption(utama.ukuran_efek)
        if utama.pembanding:
            st.markdown("**Padanan SPSS/R/perangkat lain**")
            st.caption(utama.pembanding)
        if utama.keterbatasan_desain:
            st.markdown("**Batas kesimpulan dari Rencana**")
            for batas in utama.keterbatasan_desain:
                st.caption(f"- {batas}")

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
        _simpan_peran(kamus, outcome, "outcome")
        _simpan_peran(kamus, kelompok, "kelompok")
        for nama in prediktor or []:
            _simpan_peran(kamus, nama, "prediktor")
        ui.set_kamus(kamus)

        ui.jejak().catat_keputusan(
            f"Memilih {utama.metode}",
            halaman="Pemandu Uji",
            rincian=utama.alasan,
        )
        ui.set_konfigurasi_pemandu(utama.konfig)
        kanan.success(
            "Tercatat. Panel metodenya tampil tepat di bawah ini, dengan variabel "
            "yang sudah terisi sesuai pilihan Anda.",
            icon=":material/task_alt:",
        )
    else:
        kanan.caption(
            "Aplikasi tidak menjalankan uji apa pun sebelum Anda menekan tombol ini. "
            "Mengenali nama kolom bukan alasan yang cukup untuk menyimpulkan."
        )

    st.caption(
        "Ada pertanyaan penelitian lain? Pilih tujuan yang berbeda di Langkah 1 dan "
        "jalankan Pemandu lagi — hasil yang sudah dikonfirmasi tidak hilang, semuanya "
        "terkumpul otomatis di tab Laporan."
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
