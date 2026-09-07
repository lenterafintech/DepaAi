"""NalarData - pendamping analisis dan pelaporan penelitian.

Jalankan dengan: streamlit run app.py

Satu halaman, delapan tab — menggantikan arsitektur ``st.navigation`` multi-halaman
lama (9 kelompok sidebar, tiga di antaranya bernama sama "6 · Analisis — ...").
Pengguna baru sempat bingung harus mulai dari mana di antara dua puluh dua halaman;
di sini ia hanya maju dari tab ke tab, seperti pada aplikasi rujukan yang diminta.
"""

from __future__ import annotations

import streamlit as st

from nalardata import pemandu as pmd, ui
from views import (
    akun,
    analisis_faktor,
    beranda,
    deret_waktu,
    diskriminan,
    eksplorasi,
    entri_data,
    kesesuaian,
    klaster,
    korelasi,
    korelasi_kanonik,
    laporan as laporan_hasil,
    manova,
    moderasi,
    muat_data,
    nonparametrik,
    panel,
    pca,
    penanganan_data,
    proyek,
    rapor_data,
    regresi,
    reliabilitas,
    ringkasan,
    sem,
    sidang,
    teks,
)
from views import kamus as kamus_page
from views import pemandu as pemandu_page

ui.siapkan_aplikasi()
ui.kepala_aplikasi()
ui.strip_status()

df = ui.get_dataset()
kamus = ui.kamus()
penelitian = ui.penelitian()
df_ok = ui.dataset_valid(df)

# Kelompok metode manual mengikuti nama kelompok sidebar lama, agar pengguna yang
# sudah terbiasa tidak kehilangan pengelompokan yang dikenalnya. Label metode yang
# juga dapat direkomendasikan Pemandu disamakan persis dengan
# ``nalardata.pemandu.METODE_TERSEDIA`` supaya serah-terima dari mode Dipandu dapat
# langsung menunjuk panel yang benar.
KELOMPOK_METODE: dict[str, list[tuple[str, object]]] = {
    "Uji Beda & Hubungan": [
        ("Korelasi & Asumsi", korelasi.render),
        ("Uji Beda", nonparametrik.render),
        ("MANOVA", manova.render),
    ],
    "Pemodelan": [
        ("Regresi", regresi.render),
        ("Regresi Moderasi (MRA)", moderasi.render),
        ("Analisis Diskriminan", diskriminan.render),
        ("CFA, Jalur & SEM", sem.render),
    ],
    "Reduksi & Kelompok": [
        ("PCA", pca.render),
        ("Analisis Faktor", analisis_faktor.render),
        ("Analisis Klaster", klaster.render),
        ("Korelasi Kanonik", korelasi_kanonik.render),
    ],
    "Instrumen": [
        ("Reliabilitas & Validitas", reliabilitas.render),
    ],
    "Data Lanjutan": [
        ("Regresi Panel", panel.render),
        ("Deret Waktu (ARIMA)", deret_waktu.render),
        ("Analisis Teks", teks.render),
    ],
}
RENDER_METODE = {nama: fn for grup in KELOMPOK_METODE.values() for nama, fn in grup}


def _ringkasan_keputusan_pemandu(dipandu: dict, kamus) -> str:
    """Kalimat ringkas metode + variabel + alasan satu kalimat, ditampilkan di
    atas panel metode setelah konfirmasi (poin 10) — supaya pengguna yang baru
    saja menekan tombol konfirmasi tidak perlu menggulir ke atas untuk
    mengingat apa yang baru saja ia pilih dan mengapa."""
    variabel = []
    if dipandu.get("outcome"):
        variabel.append(kamus.judul(dipandu["outcome"]))
    if dipandu.get("kelompok"):
        variabel.append(kamus.judul(dipandu["kelompok"]))
    for nama in dipandu.get("prediktor") or []:
        variabel.append(kamus.judul(nama))
    daftar_variabel = ", ".join(variabel) if variabel else "-"

    alasan = (dipandu.get("alasan") or "").strip()
    kalimat_pertama = alasan.split(". ")[0].strip() if alasan else ""
    if kalimat_pertama and not kalimat_pertama.endswith("."):
        kalimat_pertama += "."

    ringkasan = f"**{dipandu['metode']}** pada variabel {daftar_variabel}."
    if kalimat_pertama:
        ringkasan += f" {kalimat_pertama}"
    return ringkasan


def _kembali_ke_pemandu() -> None:
    st.session_state["analisis_mode"] = "Dipandu aplikasi"


def _tab_analisis() -> None:
    mode = st.radio(
        "Cara memilih metode",
        ["Dipandu aplikasi", "Pilih metode sendiri"],
        horizontal=True,
        key="analisis_mode",
    )
    st.divider()
    if not df_ok:
        ui.pesan_data_diperlukan(df)
        return

    if mode == "Dipandu aplikasi":
        pemandu_page.render(df, kamus, penelitian)
        # Dibaca SETELAH render Pemandu, bukan sebelumnya: tombol konfirmasi di
        # dalam pemandu_page.render() mengubah konfigurasi ini pada giliran
        # render yang sama saat diklik — membacanya lebih awal akan memakai
        # nilai basi sebelum konfirmasi, dan panel metode gagal langsung
        # terbuka pada klik yang sama.
        dipandu = ui.konfigurasi_pemandu()
        halaman = pmd.METODE_TERSEDIA.get(dipandu.get("metode", ""), "")
        if halaman in RENDER_METODE:
            st.divider()
            ui.judul_bagian(f"Panel metode: {halaman}", kicker="Analisis")
            st.info(
                _ringkasan_keputusan_pemandu(dipandu, kamus), icon=":material/task_alt:"
            )
            RENDER_METODE[halaman](df, kamus, penelitian)
    else:
        # Cabang ini tidak merender Pemandu, sehingga konfigurasi tidak dapat
        # berubah pada giliran ini — aman dibaca di sini, dan TETAP ada
        # (tidak pernah dihapus) meski pengguna berpindah ke mode manual.
        dipandu = ui.konfigurasi_pemandu()
        if dipandu.get("metode"):
            # Session state milik widget ("analisis_mode") tidak boleh diubah
            # setelah widget itu diinstansiasi pada giliran render yang sama
            # (radio-nya sudah dirender di atas) — perubahan lewat on_click
            # dieksekusi di awal giliran BERIKUTNYA, sebelum widget mana pun
            # diinstansiasi ulang, sehingga tidak melanggar batasan itu.
            st.button(
                f":material/arrow_back: Kembali ke Pemandu — metode terakhir: "
                f"{dipandu['metode']}",
                key="kembali_ke_pemandu",
                on_click=_kembali_ke_pemandu,
            )

        grup = st.selectbox("Kelompok metode", list(KELOMPOK_METODE), key="analisis_grup")
        opsi = [nama for nama, _ in KELOMPOK_METODE[grup]]
        pilihan = st.selectbox("Metode", opsi, key="analisis_metode")
        st.divider()
        dict(KELOMPOK_METODE[grup])[pilihan](df, kamus, penelitian)


def _tab_data() -> None:
    tab_muat, tab_entri, tab_kamus = st.tabs(
        ["Muat Data", "Entri Manual", "Kamus Variabel"]
    )
    with tab_muat:
        muat_data.render(df, kamus, penelitian, key_prefix="data_")
    with tab_entri:
        entri_data.render(df, kamus, penelitian)
    with tab_kamus:
        if not df_ok:
            ui.pesan_data_diperlukan(df)
        else:
            kamus_page.render(df, kamus, penelitian)


def _tab_mutu_data() -> None:
    tab_rapor, tab_eksplorasi, tab_penanganan = st.tabs(
        ["Rapor Data", "Eksplorasi", "Penanganan Data"]
    )
    with tab_rapor:
        if not df_ok:
            ui.pesan_data_diperlukan(df)
        else:
            rapor_data.render(df, kamus, penelitian)
    with tab_eksplorasi:
        if not df_ok:
            ui.pesan_data_diperlukan(df)
        else:
            eksplorasi.render(df, kamus, penelitian)
    with tab_penanganan:
        if not df_ok:
            ui.pesan_data_diperlukan(df)
        else:
            penanganan_data.render(df, kamus, penelitian)


def _tab_laporan() -> None:
    tab_narasi, tab_hasil, tab_mutu = st.tabs(
        ["Ringkasan Otomatis", "Hasil yang Anda Jalankan", "Kesesuaian Hasil"]
    )
    with tab_narasi:
        if not df_ok:
            ui.pesan_data_diperlukan(df)
        else:
            ringkasan.render(df)
    with tab_hasil:
        laporan_hasil.render()
    with tab_mutu:
        kesesuaian.render()


def _tab_simulasi_sidang() -> None:
    if not df_ok:
        ui.pesan_data_diperlukan(df)
        return
    sidang.render(df, kamus, penelitian)


(
    tab_beranda,
    tab_rencana,
    tab_data,
    tab_mutu_data,
    tab_analisis,
    tab_laporan,
    tab_sidang,
    tab_akun,
) = st.tabs(
    [
        ":material/home: Beranda",
        ":material/assignment: Rencana",
        ":material/folder: Data",
        ":material/fact_check: Mutu Data",
        ":material/explore: Analisis",
        ":material/description: Laporan",
        ":material/school: Simulasi Sidang",
        ":material/person: Akun",
    ]
)

with tab_beranda:
    beranda.render(df, kamus, penelitian)
with tab_rencana:
    proyek.render(df, kamus, penelitian)
with tab_data:
    _tab_data()
with tab_mutu_data:
    _tab_mutu_data()
with tab_analisis:
    _tab_analisis()
with tab_laporan:
    _tab_laporan()
with tab_sidang:
    _tab_simulasi_sidang()
with tab_akun:
    akun.render()
