"""Uji asap seluruh halaman Streamlit: memastikan tidak ada galat saat dirender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
# Halaman yang sengaja tetap berguna tanpa data aktif. Beranda dan entri data
# justru tempat data dibuat; Laporan Hasil menampilkan hasil yang sudah disimpan,
# yang tetap sah dibaca sekalipun datanya sudah tidak dimuat lagi. Ruang Proyek
# adalah tahap sebelum data dikumpulkan, sehingga meminta data di sana justru
# membalik urutan penelitian.
MANDIRI = {"beranda", "entri_data", "akun", "masuk", "laporan", "proyek"}
PAGES = sorted(p for p in (ROOT / "views").glob("*.py") if p.stem not in MANDIRI)
SEMUA = sorted((ROOT / "views").glob("*.py"))
SAMPLE = ROOT / "data" / "contoh_data_nasabah.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


def _run(path: Path, sample: pd.DataFrame | None, paket: str = "profesional") -> AppTest:
    app = AppTest.from_file(str(path), default_timeout=180)
    # Halaman diuji pada paket penuh; pembatasan paket diuji terpisah.
    app.session_state["paket_langganan"] = paket
    if sample is not None:
        app.session_state["dataset"] = sample
        app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    return app.run()


def test_beranda_tanpa_data():
    app = _run(ROOT / "views" / "beranda.py", None)
    assert not app.exception


def test_beranda_dengan_data(sample):
    app = _run(ROOT / "views" / "beranda.py", sample)
    assert not app.exception
    assert any("Pratinjau data" in md.value for md in app.markdown)


@pytest.mark.parametrize("page", SEMUA, ids=lambda p: p.stem)
def test_halaman_berjalan_dengan_data(page: Path, sample: pd.DataFrame):
    app = _run(page, sample)
    assert not app.exception, f"{page.name}: {app.exception}"
    assert not app.error, f"{page.name}: {[e.value for e in app.error]}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_halaman_meminta_data_saat_kosong(page: Path):
    app = _run(page, None)
    assert not app.exception
    assert any("Belum ada data" in w.value for w in app.warning)


def test_entri_data_berjalan_tanpa_data():
    """Halaman entri harus tetap dapat dipakai justru saat belum ada data."""
    app = _run(ROOT / "views" / "entri_data.py", None)
    assert not app.exception
    assert not app.error
    assert any("Tentukan kolom" in str(sub.value) for sub in app.subheader)


def test_halaman_terkunci_pada_paket_gratis(sample):
    """Metode di luar paket harus berhenti dengan ajakan naik paket, bukan galat."""
    app = _run(ROOT / "views" / "manova.py", sample, paket="gratis")
    assert not app.exception
    assert any("tidak termasuk dalam paket" in w.value for w in app.warning)


def test_halaman_terbuka_pada_paket_profesional(sample):
    app = _run(ROOT / "views" / "manova.py", sample, paket="profesional")
    assert not app.exception
    assert not any("tidak termasuk dalam paket" in w.value for w in app.warning)


def test_data_melebihi_batas_paket_ditolak(sample):
    """Data yang lebih besar dari batas paket dihentikan dengan pesan yang jelas."""
    app = _run(ROOT / "views" / "eksplorasi.py", sample, paket="gratis")
    assert not app.exception
    assert any("membatasi" in w.value for w in app.warning)


def test_halaman_akun_menampilkan_paket_aktif():
    app = _run(ROOT / "views" / "akun.py", None, paket="gratis")
    assert not app.exception
    assert any("Masa perkenalan" in i.value for i in app.info)


def test_halaman_masuk_menawarkan_pendaftaran():
    app = _run(ROOT / "views" / "masuk.py", None, paket="gratis")
    assert not app.exception
    assert any("Paket yang tersedia" in str(s.value) for s in app.subheader)


def _html(app) -> str:
    """Seluruh isi st.html pada halaman, digabung menjadi satu teks."""
    return " ".join(str(e.body) for e in app.get("html"))


def test_laporan_hasil_tanpa_keranjang_menjelaskan_caranya():
    """Keadaan kosong harus mengarahkan langkah berikutnya, bukan sekadar memberi tahu."""
    app = _run(ROOT / "views" / "laporan.py", None)
    assert not app.exception
    isi = _html(app)
    assert "mva-kosong" in isi  # memakai komponen keadaan kosong, bukan kotak bawaan
    assert "Simpan ke laporan" in isi


def test_laporan_hasil_menampilkan_isi_keranjang(sample):
    from nalardata import keranjang as kr

    isi = kr.Keranjang()
    isi.tambah_tabel("Regresi linear", "Koefisien regresi", sample.head(3))
    app = AppTest.from_file(str(ROOT / "views" / "laporan.py"), default_timeout=180)
    app.session_state["paket_langganan"] = "profesional"
    app.session_state["keranjang_hasil"] = isi
    app.run()
    assert not app.exception
    isi = _html(app)
    assert "Daftar isi" in isi and "Ekspor laporan" in isi
    # Judul bagian memakai komponen bersama, bukan subheader polos.
    assert "mva-bagian" in isi


def test_bilah_status_menyebut_data_aktif(sample):
    """Bilah status di atas halaman mencegah keliru menganalisis data yang salah."""
    app = _run(ROOT / "views" / "eksplorasi.py", sample)
    isi = _html(app)
    assert "mva-strip" in isi
    assert "contoh_data_nasabah.csv" in isi
    assert "400" in isi  # jumlah baris ikut ditampilkan


def test_token_warna_mengikuti_tema():
    """Palet terang dan gelap harus punya kunci yang sama persis."""
    from nalardata import ui

    assert set(ui.WARNA) == set(ui.WARNA_GELAP)
    gaya = ui._gaya()
    # Token benar-benar tertulis sebagai custom property, bukan menyatu satu baris.
    assert gaya.count("\n  --") == len(ui.WARNA)
    assert "prefers-reduced-motion" in gaya


def test_contoh_data_tetap_terbuka_pada_paket_gratis(sample):
    """Onboarding tidak boleh terbentur dinding berbayar.

    Contoh data bawaan lebih besar daripada batas paket Gratis. Halaman analisis
    tetap harus berjalan atasnya, karena tombol "Muat contoh data" adalah jalan
    masuk pertama pengguna baru ke aplikasi.
    """
    app = AppTest.from_file(str(ROOT / "views" / "eksplorasi.py"), default_timeout=180)
    app.session_state["paket_langganan"] = "gratis"
    app.session_state["dataset"] = sample
    app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    app.session_state["data_adalah_contoh"] = True
    app.run()
    assert not app.exception
    assert not any("membatasi" in w.value for w in app.warning)


def test_data_pengguna_tetap_dibatasi_paket(sample):
    """Pengecualian hanya berlaku bagi contoh bawaan, bukan data pengguna."""
    app = AppTest.from_file(str(ROOT / "views" / "eksplorasi.py"), default_timeout=180)
    app.session_state["paket_langganan"] = "gratis"
    app.session_state["dataset"] = sample
    app.session_state["dataset_name"] = "data_saya.csv"
    app.run()
    assert not app.exception
    assert any("membatasi" in w.value for w in app.warning)


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


def test_ruang_proyek_berguna_sebelum_data_ada():
    """Tahap 0 mendahului data; halaman ini tidak boleh menuntut unggahan lebih dulu."""
    app = _run(ROOT / "views" / "proyek.py", None)
    assert not app.exception
    assert not any("Belum ada data" in w.value for w in app.warning)
    teks = " ".join(md.value for md in app.markdown)
    assert "sebab-akibat" in teks


# --------------------------------------------------------------------------- #
# Kunci kausalitas pada halaman laporan
# --------------------------------------------------------------------------- #


def _jalankan_laporan(sample, desain: str, acak: bool = False):
    from nalardata import proyek_penelitian as pp

    app = AppTest.from_file(str(ROOT / "views" / "ringkasan_akademik.py"), default_timeout=300)
    app.session_state["paket_langganan"] = "profesional"
    app.session_state["dataset"] = sample
    app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    app.session_state["proyek_penelitian"] = pp.ProyekPenelitian(
        desain=desain, penugasan_acak=acak
    )
    app.run()
    return app


def _teks_laporan(app) -> str:
    """Seluruh teks pada laporan yang benar-benar disusun halaman itu.

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


def test_halaman_laporan_menghormati_kunci_kausalitas(sample):
    """Rancangan potong lintang tidak boleh menghasilkan bahasa sebab-akibat."""
    from nalardata import pagar
    from nalardata import proyek_penelitian as pp

    app = _jalankan_laporan(sample, "potong_lintang")
    assert not app.exception
    lintang = pp.ProyekPenelitian(desain="potong_lintang")
    assert pagar.periksa_kausalitas(_teks_laporan(app), lintang) == []


def test_rancangan_eksperimen_membuka_bahasa_sebab_di_halaman(sample):
    """Rancangan ikut menandai cache; bila tidak, laporan lama dipakai ulang."""
    app = _jalankan_laporan(sample, "eksperimen", acak=True)
    assert not app.exception
    assert "berpengaruh" in _teks_laporan(app)


def test_batas_rancangan_ikut_ke_halaman(sample):
    app = _jalankan_laporan(sample, "potong_lintang")
    assert any(
        "bukan sebab-akibat" in k
        for k in app.session_state["kesimpulan_laporan"].keterbatasan
    )
