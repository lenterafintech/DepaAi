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
# membalik urutan penelitian. Kesesuaian Hasil membandingkan aplikasi dengan R
# di atas dataset acuan bawaan, bukan di atas data pengguna.
MANDIRI = {"beranda", "entri_data", "akun", "masuk", "laporan", "proyek", "kesesuaian"}
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
    """Data yang lebih besar dari batas paket dihentikan dengan pesan yang jelas.

    Contoh bawaan sengaja muat pada paket Gratis, jadi datanya digandakan sampai
    melewati batas - yang diuji adalah penegakannya, bukan ukuran contohnya.
    """
    import pandas as pd

    from nalardata import langganan as lg

    ulang = lg.PAKET["gratis"].maks_baris // len(sample) + 2
    besar = pd.concat([sample] * ulang, ignore_index=True)
    app = _run(ROOT / "views" / "eksplorasi.py", besar, paket="gratis")
    app.session_state["dataset_name"] = "data_saya.csv"
    app.run()
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


def test_data_pengguna_yang_terlalu_besar_tetap_dibatasi(sample):
    """Pengecualian hanya berlaku bagi contoh bawaan, bukan data pengguna.

    Contoh bawaan kini muat pada paket Gratis, jadi datanya digandakan agar
    yang diuji benar-benar penegakan batasnya.
    """
    import pandas as pd

    besar = pd.concat([sample] * 4, ignore_index=True)
    app = AppTest.from_file(str(ROOT / "views" / "eksplorasi.py"), default_timeout=180)
    app.session_state["paket_langganan"] = "gratis"
    app.session_state["dataset"] = besar
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


# --------------------------------------------------------------------------- #
# Rapor Data
# --------------------------------------------------------------------------- #


def test_rapor_data_menampilkan_temuan_dan_pilihan_tindakan(sample):
    import pandas as pd

    kotor = sample.copy()
    kotor["nama_responden"] = [f"R{i}" for i in range(len(kotor))]

    app = _run(ROOT / "views" / "rapor_data.py", kotor)
    assert not app.exception
    teks = " ".join(md.value for md in app.markdown)
    assert "Apa yang ditemukan" in teks
    assert "Akibatnya pada analisis" in teks
    assert "Yang sebaiknya dilakukan" in teks
    assert "Pilihan tindakan" in teks


def test_rapor_data_tidak_mengubah_data_saat_hanya_dibuka(sample):
    """Aplikasi melaporkan; pengguna yang memutuskan."""
    app = _run(ROOT / "views" / "rapor_data.py", sample)
    assert not app.exception
    assert app.session_state["dataset"].shape == sample.shape


# --------------------------------------------------------------------------- #
# Navigasi
# --------------------------------------------------------------------------- #


def _jalur_terdaftar() -> list[str]:
    """Jalur halaman yang benar-benar dilewatkan ke st.navigation.

    Dibaca dari pohon sintaksis, bukan dari ekspresi reguler: sebagian
    ``st.Page`` ditulis berbaris-baris sehingga pencocokan teks meleset justru
    pada halaman yang paling panjang keterangannya.
    """
    import ast

    pohon = ast.parse((ROOT / "app.py").read_text())
    jalur = []
    for simpul in ast.walk(pohon):
        if (
            isinstance(simpul, ast.Call)
            and isinstance(simpul.func, ast.Attribute)
            and simpul.func.attr == "Page"
            and simpul.args
            and isinstance(simpul.args[0], ast.Constant)
        ):
            jalur.append(str(simpul.args[0].value))
    return jalur


def test_setiap_halaman_terdaftar_pada_navigasi():
    """Halaman yang tidak terdaftar tidak akan pernah ditemukan pengguna."""
    terdaftar = {Path(j).name for j in _jalur_terdaftar()}
    tersedia = {p.name for p in (ROOT / "views").glob("*.py")}
    assert tersedia - terdaftar == set(), "halaman ada tetapi tidak masuk menu"
    assert terdaftar - tersedia == set(), "menu menunjuk halaman yang tidak ada"


def test_urutan_menu_mengikuti_tahapan_penelitian():
    """Menu adalah perjalanan, bukan daftar metode.

    Pengguna yang belum menguasai statistik tahu sampai di mana penelitiannya,
    tetapi belum tentu tahu nama uji yang dicarinya.
    """
    import ast

    pohon = ast.parse((ROOT / "app.py").read_text())
    grup = []
    for simpul in ast.walk(pohon):
        if isinstance(simpul, ast.Dict):
            grup = [k.value for k in simpul.keys if isinstance(k, ast.Constant)]
            break

    bernomor = [g for g in grup if g[0].isdigit()]
    assert bernomor == sorted(bernomor), "tahapan harus tampil berurutan"
    for kata in ("Rencana", "Data", "Pilih Metode", "Analisis", "Laporan"):
        assert any(kata in g for g in grup), kata


def test_halaman_bawaan_hanya_satu():
    """Dua halaman bawaan membuat Streamlit menolak menjalankan aplikasi."""
    import ast

    pohon = ast.parse((ROOT / "app.py").read_text())
    bawaan = 0
    for simpul in ast.walk(pohon):
        if isinstance(simpul, ast.Call) and getattr(simpul.func, "attr", "") == "Page":
            for kata_kunci in simpul.keywords:
                if kata_kunci.arg == "default" and getattr(kata_kunci.value, "value", False):
                    bawaan += 1
    assert bawaan == 1


def test_halaman_terkunci_memakai_nama_halaman_bukan_keterangan_fiturnya(sample):
    """Keterangan fitur adalah kalimat, bukan nama halaman."""
    app = _run(ROOT / "views" / "sidang.py", sample, paket="gratis")
    assert not app.exception
    tajuk = " ".join(md.value for md in app.markdown) + " ".join(
        h.value for h in getattr(app, "header", [])
    )
    assert "latihan menjawab pertanyaan penguji" not in tajuk.split("—")[0][:80]
    assert any("Mahasiswa" in c.value for c in app.caption)


def test_halaman_akademik_menawarkan_kerangka_naskah(sample):
    """Tajuk bagian dirender lewat st.html, jadi yang diperiksa isinya."""
    app = _run(ROOT / "views" / "ringkasan_akademik.py", sample)
    assert not app.exception
    caption = [c.value for c in app.caption]
    assert any("bukan naskah jadi" in c for c in caption)
    assert any("harus Anda tulis sendiri" in c for c in caption)
    assert any("naskah_bab" in c for c in caption), "nama berkas naskah harus tampil"


# --------------------------------------------------------------------------- #
# Serah-terima Pemandu ke halaman metode
# --------------------------------------------------------------------------- #


def test_halaman_uji_beda_terisi_dari_pemandu(sample):
    """Pengguna tidak boleh diminta memilih ulang variabel yang baru saja ia sebut."""
    app = AppTest.from_file(str(ROOT / "views" / "nonparametrik.py"), default_timeout=180)
    app.session_state["paket_langganan"] = "profesional"
    app.session_state["dataset"] = sample
    app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    app.session_state["pemandu_konfigurasi"] = {
        "metode": "One-Way ANOVA",
        "outcome": "skor_kredit",
        "kelompok": "segmen_usaha",
        "prediktor": [],
        "berpasangan": False,
    }
    app.run()
    assert not app.exception
    assert any("Disiapkan dari Pemandu Uji" in s.value for s in app.success)


def test_halaman_uji_beda_tanpa_pemandu_tidak_mengisi_apa_apa(sample):
    """Mengisi pilihan orang yang tidak memintanya justru membingungkan."""
    app = _run(ROOT / "views" / "nonparametrik.py", sample)
    assert not app.exception
    assert not any("Disiapkan dari Pemandu Uji" in s.value for s in app.success)


def test_kolom_pemandu_yang_sudah_tidak_ada_tidak_menggagalkan_halaman(sample):
    """Data dapat berganti setelah pemandu dijalankan."""
    app = AppTest.from_file(str(ROOT / "views" / "nonparametrik.py"), default_timeout=180)
    app.session_state["paket_langganan"] = "profesional"
    app.session_state["dataset"] = sample
    app.session_state["dataset_name"] = "contoh_data_nasabah.csv"
    app.session_state["pemandu_konfigurasi"] = {
        "metode": "One-Way ANOVA",
        "outcome": "kolom_yang_sudah_dihapus",
        "kelompok": "juga_tidak_ada",
        "prediktor": [],
        "berpasangan": False,
    }
    app.run()
    assert not app.exception


def test_halaman_kesesuaian_menyebut_yang_belum_divalidasi():
    """Daftar yang menyembunyikan lubangnya sendiri tidak dapat dipercaya."""
    app = _run(ROOT / "views" / "kesesuaian.py", None)
    assert not app.exception
    teks = " ".join(md.value for md in app.markdown)
    assert "Belum divalidasi" in str(app.session_state) or "belum" in teks.lower()
    assert any("CFA / SEM" in md.value for md in app.markdown)
