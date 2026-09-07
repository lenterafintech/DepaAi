"""Uji asap aplikasi satu halaman: memastikan tidak ada galat pada tiap kombinasi tab.

Sejak arsitektur ``st.navigation`` multi-halaman diganti satu ``app.py`` dengan
delapan ``st.tabs()``, seluruh tab dirender pada setiap giliran skrip yang sama —
Streamlit hanya menyembunyikan visualnya, bukan melewati kodenya. Karena itu uji
di sini menjalankan ``app.py`` secara utuh (bukan per berkas ``views/*.py`` seperti
sebelumnya) dan memeriksa keluaran gabungan seluruh tab pada satu giliran render.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
SAMPLE = ROOT / "data" / "contoh_data_nasabah.csv"

KELOMPOK_METODE = {
    "Uji Beda & Hubungan": ["Korelasi & Asumsi", "Uji Beda", "MANOVA"],
    "Pemodelan": ["Regresi", "Regresi Moderasi (MRA)", "Analisis Diskriminan", "CFA, Jalur & SEM"],
    "Reduksi & Kelompok": ["PCA", "Analisis Faktor", "Analisis Klaster", "Korelasi Kanonik"],
    "Instrumen": ["Reliabilitas & Validitas"],
}


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


@pytest.fixture(scope="module")
def sample_survei() -> pd.DataFrame:
    """Data kuesioner sintetis (butir bernama berpola) untuk menguji CFA/PLS-SEM."""
    import numpy as np

    rng = np.random.default_rng(9)
    n = 200
    kual = rng.normal(0, 1, n)
    puas = 0.5 * kual + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "KUAL1": kual * 0.8 + rng.normal(0, 0.5, n),
            "KUAL2": kual * 0.75 + rng.normal(0, 0.5, n),
            "KUAL3": kual * 0.7 + rng.normal(0, 0.5, n),
            "PUAS1": puas * 0.8 + rng.normal(0, 0.5, n),
            "PUAS2": puas * 0.75 + rng.normal(0, 0.5, n),
            "PUAS3": puas * 0.7 + rng.normal(0, 0.5, n),
        }
    )


def _run(sample: pd.DataFrame | None, paket: str = "profesional", **tambahan) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=300)
    # Aplikasi diuji pada paket penuh secara bawaan; pembatasan paket diuji terpisah.
    app.session_state["paket_langganan"] = paket
    if sample is not None:
        app.session_state["dataset"] = sample
        app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    for kunci, nilai in tambahan.items():
        app.session_state[kunci] = nilai
    return app.run()


def _html(app: AppTest) -> str:
    """Seluruh isi st.html pada satu giliran render, digabung menjadi satu teks."""
    return " ".join(str(e.body) for e in app.get("html"))


def _teks(app: AppTest) -> str:
    return " ".join(md.value for md in app.markdown)


# --------------------------------------------------------------------------- #
# Asap dasar: aplikasi berjalan tanpa galat, dengan dan tanpa data
# --------------------------------------------------------------------------- #


def test_app_berjalan_tanpa_data():
    app = _run(None)
    assert not app.exception
    assert not app.error


def test_app_berjalan_dengan_data(sample):
    app = _run(sample)
    assert not app.exception
    assert not app.error
    assert "Pratinjau data" in _teks(app)


@pytest.mark.parametrize(
    "grup,metode",
    [(g, m) for g, ms in KELOMPOK_METODE.items() for m in ms],
)
def test_setiap_metode_manual_berjalan(sample, grup: str, metode: str):
    """Setiap metode yang dapat dipilih manual harus dapat dirender tanpa galat."""
    app = _run(sample)
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    app.selectbox(key="analisis_grup").set_value(grup).run()
    app.selectbox(key="analisis_metode").set_value(metode).run()
    assert not app.exception, f"{grup}/{metode}: {app.exception}"
    assert not app.error, f"{grup}/{metode}: {[e.value for e in app.error]}"


def test_manova_pengukuran_berulang_berjalan(sample):
    """Mode ANOVA pengukuran berulang (Mauchly + koreksi) harus berjalan tanpa galat."""
    app = _run(sample)
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    app.selectbox(key="analisis_grup").set_value("Uji Beda & Hubungan").run()
    app.selectbox(key="analisis_metode").set_value("MANOVA").run()
    app.radio(key="manova_mode").set_value("Pengukuran berulang (dalam-subjek)").run()
    ms = app.multiselect(key="rm_kondisi")
    ms.set_value(ms.options[:3]).run()
    assert not app.exception
    assert not app.error


def test_pls_sem_berjalan_sampai_bootstrap(sample_survei):
    """Tab PLS-SEM: pemilihan 2 konstruk, jalur, dan bootstrap harus berjalan tanpa galat."""
    app = _run(sample_survei)
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    app.selectbox(key="analisis_grup").set_value("Pemodelan").run()
    app.selectbox(key="analisis_metode").set_value("CFA, Jalur & SEM").run()
    ms = app.multiselect(key="pls_konstruk")
    ms.set_value(list(ms.options)).run()
    assert not app.exception
    app.button(key="pls_jalan_boot").click().run()
    assert not app.exception
    assert not app.error


def test_beranda_sebelum_data_menawarkan_cara_memulai():
    app = _run(None)
    assert not app.exception
    assert "tanpa harus jadi ahli statistik" in _html(app)


def test_beranda_sesudah_data_menampilkan_dasbor(sample):
    app = _run(sample)
    assert not app.exception
    assert "Dasbor" in _html(app)
    assert any("Langkah berikutnya" in i.value for i in app.info)


def test_entri_data_berjalan_tanpa_data():
    """Entri manual justru tempat data dibuat, harus tetap berguna tanpa data aktif."""
    app = _run(None)
    assert not app.exception
    assert not app.error
    assert any("Tentukan kolom" in str(sub.value) for sub in app.subheader)


def test_ruang_proyek_berguna_sebelum_data_ada():
    """Rencana mendahului data; tab ini tidak boleh menuntut unggahan lebih dulu."""
    app = _run(None)
    assert not app.exception
    assert "sebab-akibat" in _teks(app)


def test_kesesuaian_hasil_tidak_bergantung_data():
    app = _run(None)
    assert not app.exception
    teks = _teks(app)
    assert "CFA / SEM" in teks or "belum" in teks.lower()


# --------------------------------------------------------------------------- #
# Keadaan kosong: tab yang butuh data harus mengarahkan, bukan galat
# --------------------------------------------------------------------------- #


def test_mutu_data_meminta_data_saat_kosong():
    app = _run(None)
    assert not app.exception
    assert "Perlu data terlebih dahulu" in _html(app)


def test_analisis_manual_meminta_data_saat_kosong():
    app = _run(None)
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    assert not app.exception
    assert "Perlu data terlebih dahulu" in _html(app)


def test_laporan_ringkasan_meminta_data_saat_kosong():
    app = _run(None)
    assert not app.exception
    assert "Perlu data terlebih dahulu" in _html(app)


def test_simulasi_sidang_meminta_data_saat_kosong():
    app = _run(None)
    assert not app.exception
    assert "Perlu data terlebih dahulu" in _html(app)


# --------------------------------------------------------------------------- #
# Paket dan batas ukuran
# --------------------------------------------------------------------------- #


def test_metode_terkunci_pada_paket_gratis(sample):
    """Metode di luar paket harus berhenti dengan ajakan naik paket, bukan galat."""
    app = _run(sample, paket="gratis")
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    app.selectbox(key="analisis_grup").set_value("Uji Beda & Hubungan").run()
    app.selectbox(key="analisis_metode").set_value("MANOVA").run()
    assert not app.exception
    assert any("tidak termasuk dalam paket" in w.value for w in app.warning)


def test_metode_terbuka_pada_paket_profesional(sample):
    app = _run(sample, paket="profesional")
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    app.selectbox(key="analisis_grup").set_value("Uji Beda & Hubungan").run()
    app.selectbox(key="analisis_metode").set_value("MANOVA").run()
    assert not app.exception
    assert not any("tidak termasuk dalam paket" in w.value for w in app.warning)


def test_data_melebihi_batas_paket_ditolak(sample):
    """Data yang lebih besar dari batas paket dihentikan dengan pesan yang jelas.

    Contoh bawaan sengaja muat pada paket Gratis, jadi datanya digandakan sampai
    melewati batas — yang diuji adalah penegakannya, bukan ukuran contohnya.
    """
    from nalardata import langganan as lg

    ulang = lg.PAKET["gratis"].maks_baris // len(sample) + 2
    besar = pd.concat([sample] * ulang, ignore_index=True)
    app = _run(besar, paket="gratis", dataset_name="data_saya.csv")
    assert not app.exception
    assert any("membatasi" in w.value for w in app.warning)


def test_contoh_data_tetap_terbuka_pada_paket_gratis(sample):
    """Onboarding tidak boleh terbentur dinding berbayar.

    Contoh data bawaan lebih besar daripada batas paket Gratis. Analisis tetap
    harus berjalan atasnya, karena tombol "Muat contoh data" adalah jalan masuk
    pertama pengguna baru ke aplikasi.
    """
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.session_state["paket_langganan"] = "gratis"
    app.session_state["dataset"] = sample
    app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    app.session_state["data_adalah_contoh"] = True
    app.run()
    assert not app.exception
    assert not any("membatasi" in w.value for w in app.warning)


def test_data_pengguna_yang_terlalu_besar_tetap_dibatasi(sample):
    """Pengecualian ukuran hanya berlaku bagi contoh bawaan, bukan data pengguna."""
    besar = pd.concat([sample] * 4, ignore_index=True)
    app = _run(besar, paket="gratis", dataset_name="data_saya.csv")
    assert not app.exception
    assert any("membatasi" in w.value for w in app.warning)


def test_akun_menampilkan_paket_aktif():
    app = _run(None, paket="gratis")
    assert not app.exception
    assert any("Masa perkenalan" in i.value for i in app.info)


def test_akun_menawarkan_pendaftaran():
    app = _run(None, paket="gratis")
    assert not app.exception
    assert any("Ganti paket" in str(s.value) for s in app.subheader)
    assert any("Buat akun" in b.label for b in app.button)


# --------------------------------------------------------------------------- #
# Hasil yang Anda Jalankan (bekas Laporan Hasil, kini otomatis)
# --------------------------------------------------------------------------- #


def test_hasil_dijalankan_tanpa_isi_menjelaskan_caranya():
    """Keadaan kosong harus mengarahkan langkah berikutnya, bukan sekadar memberi tahu."""
    app = _run(None)
    assert not app.exception
    isi = _html(app)
    assert "mva-kosong" in isi
    assert "tab Analisis" in isi


def test_metode_yang_dijalankan_otomatis_tercatat_di_laporan(sample):
    """Menjalankan sebuah metode manual harus otomatis muncul di 'Hasil yang Anda Jalankan'."""
    app = _run(sample)
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    app.selectbox(key="analisis_grup").set_value("Reduksi & Kelompok").run()
    app.selectbox(key="analisis_metode").set_value("PCA").run()
    assert not app.exception
    isi = _html(app)
    assert "Daftar isi" in isi and "Ekspor laporan" in isi
    assert "mva-bagian" in isi


def test_bilah_status_menyebut_data_aktif(sample):
    """Bilah status di puncak aplikasi mencegah keliru menganalisis data yang salah."""
    app = _run(sample)
    isi = _html(app)
    assert "mva-strip" in isi
    assert "contoh_data_nasabah.csv" in isi
    assert "400" in isi  # jumlah baris ikut ditampilkan


def test_token_warna_mengikuti_tema():
    """Palet terang dan gelap harus punya kunci yang sama persis."""
    from nalardata import ui

    assert set(ui.WARNA) == set(ui.WARNA_GELAP)
    gaya = ui._gaya()
    # Setiap token palet wajib tertulis sebagai custom property tersendiri —
    # boleh ada custom property tambahan (mis. token bayangan) di luar palet,
    # tetapi tidak boleh ada token palet yang hilang atau menyatu satu baris.
    for nama in ui.WARNA:
        assert f"\n  --{nama}: " in gaya, nama
    assert gaya.count("\n  --") >= len(ui.WARNA)
    assert "prefers-reduced-motion" in gaya


def test_toolbar_bawaan_streamlit_disembunyikan():
    """Menu Deploy/Fork bawaan Streamlit Cloud tidak relevan bagi produk publik."""
    from nalardata import ui

    gaya = ui._gaya()
    assert 'stToolbar' in gaya and "display: none" in gaya


# --------------------------------------------------------------------------- #
# Teks tafsiran
# --------------------------------------------------------------------------- #


def test_tafsiran_merender_markdown_bukan_bintang_harfiah():
    """st.html tidak memproses markdown, sehingga ** perlu diubah sendiri."""
    from nalardata import ui

    assert ui._markdown_ringkas("**tebal**") == "<b>tebal</b>"
    assert ui._markdown_ringkas("*miring*") == "<i>miring</i>"
    assert ui._markdown_ringkas("`kode`") == "<code>kode</code>"


def test_tafsiran_mengamankan_tanda_kurung_sudut():
    """Nama variabel seperti <NA> tidak boleh berubah menjadi tag."""
    from nalardata import ui

    assert ui._markdown_ringkas("nilai <NA> pada kolom") == "nilai &lt;NA&gt; pada kolom"
    assert "<script>" not in ui._markdown_ringkas("<script>alert(1)</script>")


# --------------------------------------------------------------------------- #
# Kunci kausalitas pada Ringkasan Otomatis
# --------------------------------------------------------------------------- #


def _jalankan_ringkasan(sample, desain: str, acak: bool = False):
    from nalardata import proyek_penelitian as pp

    app = AppTest.from_file(str(APP), default_timeout=300)
    app.session_state["paket_langganan"] = "profesional"
    app.session_state["dataset"] = sample
    app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    app.session_state["proyek_penelitian"] = pp.ProyekPenelitian(
        desain=desain, penugasan_acak=acak
    )
    app.run()
    return app


def _teks_laporan(app: AppTest) -> str:
    """Seluruh teks pada laporan yang benar-benar disusun aplikasi.

    Diambil dari objek laporannya, bukan dari tangkapan layar: teks temuan sebagian
    dirender lewat ``st.html`` sehingga tidak muncul pada ``app.markdown``, dan
    memeriksa yang muncul saja akan meloloskan justru bagian yang dibaca penguji.
    """
    lap = app.session_state["kesimpulan_laporan"]
    bagian = [lap.headline, lap.subheadline, lap.pendorong_sumber]
    bagian += [t.judul + t.ringkas + t.eksekutif + t.akademik + t.profesional for t in lap.temuan]
    bagian += [r.judul + r.alasan for r in lap.rekomendasi]
    bagian += [p.teks for p in lap.paragraf]
    bagian += [l.catatan for l in lap.lampu]
    bagian += [d.catatan for d in lap.pendorong]
    bagian += list(lap.keterbatasan)
    return " ".join(bagian)


def test_ringkasan_menghormati_kunci_kausalitas(sample):
    """Rancangan potong lintang tidak boleh menghasilkan bahasa sebab-akibat."""
    from nalardata import pagar
    from nalardata import proyek_penelitian as pp

    app = _jalankan_ringkasan(sample, "potong_lintang")
    assert not app.exception
    lintang = pp.ProyekPenelitian(desain="potong_lintang")
    assert pagar.periksa_kausalitas(_teks_laporan(app), lintang) == []


def test_rancangan_eksperimen_membuka_bahasa_sebab(sample):
    """Rancangan ikut menandai cache; bila tidak, laporan lama dipakai ulang."""
    app = _jalankan_ringkasan(sample, "eksperimen", acak=True)
    assert not app.exception
    assert "berpengaruh" in _teks_laporan(app)


def test_batas_rancangan_ikut_ke_laporan(sample):
    app = _jalankan_ringkasan(sample, "potong_lintang")
    assert any(
        "bukan sebab-akibat" in k
        for k in app.session_state["kesimpulan_laporan"].keterbatasan
    )


def test_ringkasan_akademik_menawarkan_kerangka_naskah(sample):
    app = _run(sample)
    app.radio(key="ringkasan_register").set_value("akademik").run()
    assert not app.exception
    caption = [c.value for c in app.caption]
    assert any("bukan naskah jadi" in c for c in caption)
    assert any("harus Anda tulis sendiri" in c for c in caption)
    assert any("naskah_bab" in c for c in caption), "nama berkas naskah harus tampil"


# --------------------------------------------------------------------------- #
# Rapor Data
# --------------------------------------------------------------------------- #


def test_rapor_data_menampilkan_temuan_dan_pilihan_tindakan(sample):
    kotor = sample.copy()
    kotor["nama_responden"] = [f"R{i}" for i in range(len(kotor))]

    app = _run(kotor)
    assert not app.exception
    teks = _teks(app)
    assert "Apa yang ditemukan" in teks
    assert "Akibatnya pada analisis" in teks
    assert "Yang sebaiknya dilakukan" in teks
    assert "Pilihan tindakan" in teks


def test_rapor_data_tidak_mengubah_data_saat_hanya_dibuka(sample):
    """Aplikasi melaporkan; pengguna yang memutuskan."""
    app = _run(sample)
    assert not app.exception
    assert app.session_state["dataset"].shape == sample.shape


# --------------------------------------------------------------------------- #
# Penanganan Data
# --------------------------------------------------------------------------- #


def test_penanganan_data_tidak_mengubah_data_saat_hanya_dibuka(sample):
    app = _run(sample)
    assert not app.exception
    assert app.session_state["dataset"].shape == sample.shape


def test_penanganan_data_menerapkan_imputasi_dan_penskalaan(sample):
    from nalardata import preprocessing as pp

    kotor = sample.copy()
    numerik = pp.numeric_columns(kotor)
    kolom = numerik[0]
    kotor.loc[0, kolom] = None

    app = _run(kotor)
    app.selectbox(key="pen_data_missing").set_value("rata-rata").run()
    app.button(key="pen_num_terapkan").click().run()
    assert not app.exception

    hasil = app.session_state["dataset"]
    assert hasil[numerik].isna().sum().sum() == 0
    assert hasil.shape[0] == kotor.shape[0]

    riwayat = app.session_state["rapor_riwayat_data"]
    assert len(riwayat) == 1
    assert "penskalaan" in riwayat[0][1]


def test_penanganan_data_batalkan_lewat_tombol_rapor_data(sample):
    """Riwayat dipakai bersama Rapor Data: satu tombol Batalkan untuk keduanya."""
    from nalardata import preprocessing as pp

    kotor = sample.copy()
    numerik = pp.numeric_columns(kotor)
    kotor.loc[0, numerik[0]] = None

    app = _run(kotor)
    app.button(key="pen_num_terapkan").click().run()
    assert app.session_state["dataset"][numerik[0]].isna().sum() == 0

    app.button(key="rapor_undo").click().run()
    assert not app.exception
    pd.testing.assert_frame_equal(
        app.session_state["dataset"].reset_index(drop=True),
        kotor.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Struktur tab (menggantikan uji navigasi st.navigation lama)
# --------------------------------------------------------------------------- #


def _daftar_tab() -> list[str]:
    """Label tab yang benar-benar dilewatkan ke ``st.tabs`` di app.py."""
    pohon = ast.parse(APP.read_text())
    for simpul in ast.walk(pohon):
        if (
            isinstance(simpul, ast.Call)
            and isinstance(simpul.func, ast.Attribute)
            and simpul.func.attr == "tabs"
            and simpul.args
            and isinstance(simpul.args[0], ast.List)
        ):
            label = [el.value for el in simpul.args[0].elts if isinstance(el, ast.Constant)]
            if len(label) >= 6:  # tab utama, bukan sub-tab kelompok metode 2-3 label
                return label
    return []


def test_tab_utama_berjumlah_delapan_tanpa_duplikat():
    """Sembilan kelompok sidebar lama (tiga di antaranya bernama sama) diringkas jadi delapan tab."""
    label = _daftar_tab()
    assert len(label) == 8
    assert len(set(label)) == 8, "tidak boleh ada judul tab yang berulang"


def test_urutan_tab_mengikuti_tahapan_penelitian():
    """Tab adalah perjalanan, bukan daftar metode."""
    label = " ".join(_daftar_tab())
    for kata in ("Beranda", "Rencana", "Data", "Mutu Data", "Analisis", "Laporan", "Sidang", "Akun"):
        assert kata in label, kata


def test_setiap_berkas_views_dipakai_app():
    """Berkas ``views/*.py`` yang tidak diimpor ``app.py`` adalah sisa yang terlupa."""
    sumber = APP.read_text()
    yatim = [
        p.stem
        for p in (ROOT / "views").glob("*.py")
        if p.stem != "__init__" and p.stem not in sumber
    ]
    assert yatim == [], f"berkas views yatim (tidak dipakai app.py): {yatim}"


def test_halaman_terkunci_memakai_nama_metode_bukan_keterangan_fiturnya(sample):
    """Keterangan fitur adalah kalimat, bukan nama bagian."""
    app = _run(sample, paket="gratis")
    assert not app.exception
    tajuk = _teks(app) + " ".join(h.value for h in getattr(app, "header", []))
    assert "latihan menjawab pertanyaan penguji" not in tajuk.split("—")[0][:80]
    assert any("Mahasiswa" in c.value for c in app.caption)


# --------------------------------------------------------------------------- #
# Serah-terima Pemandu ke panel metode
# --------------------------------------------------------------------------- #


def test_panel_metode_terisi_dari_pemandu(sample):
    """Pengguna tidak boleh diminta memilih ulang variabel yang baru saja ia sebut."""
    app = _run(
        sample,
        pemandu_konfigurasi={
            "metode": "One-Way ANOVA",
            "outcome": "skor_kredit",
            "kelompok": "segmen_usaha",
            "prediktor": [],
            "berpasangan": False,
        },
    )
    assert not app.exception
    assert any("Disiapkan dari Pemandu Uji" in s.value for s in app.success)


def test_panel_metode_tanpa_pemandu_tidak_mengisi_apa_apa(sample):
    """Mengisi pilihan orang yang tidak memintanya justru membingungkan."""
    app = _run(sample)
    app.radio(key="analisis_mode").set_value("Pilih metode sendiri").run()
    app.selectbox(key="analisis_grup").set_value("Uji Beda & Hubungan").run()
    app.selectbox(key="analisis_metode").set_value("Uji Beda").run()
    assert not app.exception
    assert not any("Disiapkan dari Pemandu Uji" in s.value for s in app.success)


def test_kolom_pemandu_yang_sudah_tidak_ada_tidak_menggagalkan_app(sample):
    """Data dapat berganti setelah pemandu dijalankan."""
    app = _run(
        sample,
        pemandu_konfigurasi={
            "metode": "One-Way ANOVA",
            "outcome": "kolom_yang_sudah_dihapus",
            "kelompok": "juga_tidak_ada",
            "prediktor": [],
            "berpasangan": False,
        },
    )
    assert not app.exception


def test_pemandu_konfirmasi_menampilkan_panel_metode_di_bawahnya(sample):
    """Konfirmasi di mode Dipandu harus langsung menampilkan panel metode — tanpa
    perlu pengguna berpindah tab atau halaman sendiri (lihat keluhan navigasi
    yang mendorong perombakan alur ini)."""
    app = _run(sample)
    app.radio(key="pemandu_tujuan").set_value("membandingkan").run()
    app.selectbox(key="pemandu_outcome_beda").set_value("skor_kredit (rasio)").run()
    app.selectbox(key="pemandu_kelompok").set_value("segmen_usaha (nominal)").run()
    app.button(key="pemandu_konfirmasi").click().run()
    assert not app.exception
    assert any("tampil tepat di bawah ini" in s.value for s in app.success)
    assert any("Disiapkan dari Pemandu Uji" in s.value for s in app.success)


def test_pemandu_langkah_2_terisi_dari_peran_kamus(sample):
    """Kolom yang perannya sudah dikonfirmasi di Kamus Variabel tidak perlu dipilih ulang."""
    from nalardata import kamus as km

    kamus = km.Kamus.dari_data(sample)
    kamus.tetapkan("skor_kredit", peran="outcome", dikonfirmasi=True)
    kamus.tetapkan("segmen_usaha", peran="kelompok", dikonfirmasi=True)

    app = _run(sample, kamus_variabel=kamus)
    app.radio(key="pemandu_tujuan").set_value("membandingkan").run()
    assert not app.exception
    assert app.selectbox(key="pemandu_outcome_beda").value == "skor_kredit"
    assert app.selectbox(key="pemandu_kelompok").value == "segmen_usaha"


def test_kamus_kartu_konfirmasi_menandai_kolom(sample):
    """Kartu bahasa awam adalah cara utama menandai kolom sudah diperiksa."""
    from nalardata import kamus as km

    kamus = km.Kamus.dari_data(sample)
    perlu = kamus.perlu_diperiksa()
    assert perlu, "data contoh harus punya kolom yang perlu diperiksa agar uji ini berarti"
    target = perlu[0]

    app = _run(sample, kamus_variabel=kamus)
    assert not app.exception
    app.button(key=f"kartu_konfirmasi_{target}").click().run()
    assert not app.exception
    assert app.session_state["kamus_variabel"][target].dikonfirmasi


def test_kesesuaian_hasil_menyebut_yang_belum_divalidasi():
    """Daftar yang menyembunyikan lubangnya sendiri tidak dapat dipercaya."""
    app = _run(None)
    assert not app.exception
    teks = _teks(app)
    assert any("CFA / SEM" in md.value for md in app.markdown) or "belum" in teks.lower()
