"""Uji penyusun kesimpulan naratif tiga register pembaca."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nalardata import narrative as nr
from nalardata.report_html import laporan_html

ROOT = Path(__file__).resolve().parents[1]
NUMERIK = [
    "usia",
    "lama_usaha_tahun",
    "pendapatan_bulanan",
    "saldo_tabungan",
    "rasio_utang_pendapatan",
    "skor_kredit",
    "jumlah_keterlambatan",
    "riwayat_pinjaman_lunas",
]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


@pytest.fixture(scope="module")
def konfig_lengkap() -> nr.Konfigurasi:
    return nr.Konfigurasi(
        variabel=NUMERIK,
        nama_data="contoh_data_nasabah.csv",
        target_numerik="skor_kredit",
        prediktor=[
            "rasio_utang_pendapatan",
            "jumlah_keterlambatan",
            "pendapatan_bulanan",
            "lama_usaha_tahun",
        ],
        target_biner="gagal_bayar",
        kelompok="segmen_usaha",
        gugus_x=["pendapatan_bulanan", "saldo_tabungan"],
        gugus_y=["skor_kredit", "jumlah_keterlambatan"],
    )


@pytest.fixture(scope="module")
def hasil(data, konfig_lengkap) -> tuple[nr.Analisis, nr.Laporan]:
    return nr.analisis_dan_laporan(data, konfig_lengkap)


def test_format_angka_gaya_indonesia():
    assert nr.num(1234.5678, 2) == "1.234,57"
    assert nr.num(0.5, 3) == "0,500"
    assert nr.num(400) == "400"
    assert nr.num(float("nan")) == "-"
    assert nr.pval(0.0001) == "p < 0,001"
    assert nr.pval(0.023) == "p = 0,023"
    assert nr.bintang(0.0005) == "***"
    assert nr.bintang(0.2) == ""


def test_daftar_frasa():
    assert nr._daftar([]) == "tidak ada"
    assert nr._daftar(["a"]) == "a"
    assert nr._daftar(["a", "b"]) == "a dan b"
    assert nr._daftar(["a", "b", "c"]) == "a, b, dan c"
    assert "2 variabel lainnya" in nr._daftar(["a", "b", "c", "d", "e"], maksimal=3)


def test_seluruh_analisis_berjalan(hasil):
    analisis, _ = hasil
    assert analisis.gagal == {}
    for atribut in (
        "normalitas",
        "mardia",
        "korelasi",
        "vif",
        "kmo",
        "pca",
        "klaster",
        "regresi",
        "logistik",
        "manova",
        "diskriminan",
        "kanonik",
        "uji_beda",
    ):
        assert getattr(analisis, atribut) is not None, atribut
    # Moderasi, reliabilitas, dan CFA menuntut konfigurasi tambahan (moderator,
    # atau butir kuesioner bernama berpola) yang tidak dipenuhi fixture ini.
    assert analisis.moderasi is None
    assert analisis.reliabilitas is None
    assert analisis.sem is None


def test_laporan_memuat_seluruh_metode(hasil):
    _, laporan = hasil
    assert laporan.dilewati == []
    assert len(laporan.temuan) == 11
    assert laporan.headline and laporan.subheadline
    assert laporan.lampu and laporan.rekomendasi and laporan.keterbatasan
    assert laporan.tabel and laporan.paragraf and laporan.rujukan


def test_tiga_register_pembaca_berbeda(hasil):
    _, laporan = hasil
    for temuan in laporan.temuan:
        teks = {pembaca: temuan.teks(pembaca) for pembaca in nr.AUDIENCES}
        assert len(set(teks.values())) == 3, temuan.judul
        for isi in teks.values():
            assert len(isi) > 80
        # Narasi untuk pembaca awam tidak boleh memuat notasi statistik teknis.
        assert "p < 0," not in teks["eksekutif"]
        assert "χ²" not in teks["eksekutif"]


def test_register_akademik_memuat_statistik_uji(hasil):
    _, laporan = hasil
    gabungan = " ".join(t.akademik for t in laporan.temuan)
    for penanda in ("p < 0,001", "R²", "Wilks", "eigenvalue"):
        assert penanda in gabungan


def test_pendorong_terurut_dan_ternormalkan(hasil):
    _, laporan = hasil
    assert laporan.pendorong
    kekuatan = [p.kekuatan for p in laporan.pendorong]
    assert kekuatan == sorted(kekuatan, reverse=True)
    assert kekuatan[0] == pytest.approx(1.0)
    assert all(0 <= p.kekuatan <= 1 for p in laporan.pendorong)


def test_lampu_status_valid(hasil):
    _, laporan = hasil
    assert all(l.status in {"baik", "perhatian", "kritis"} for l in laporan.lampu)
    assert {"Kecukupan data", "Multikolinearitas"} <= {l.label for l in laporan.lampu}


def test_tabel_korelasi_bergaya_apa(hasil):
    analisis, _ = hasil
    tabel = nr.tabel_deskriptif_korelasi(analisis)
    assert list(tabel.columns[:3]) == ["Variabel", "M", "SD"]
    assert tabel.iloc[0]["1"] == "—"
    assert tabel.iloc[0]["2"] == ""  # segitiga atas dikosongkan
    assert any("*" in str(v) for v in tabel.iloc[1:]["1"])


def test_markdown_dan_html_untuk_setiap_pembaca(hasil):
    _, laporan = hasil
    for pembaca in nr.AUDIENCES:
        markdown = laporan.markdown(pembaca)
        assert laporan.headline in markdown
        assert "Rekomendasi tindakan" in markdown
        html = laporan_html(laporan, pembaca)
        assert html.startswith("<!doctype html>")
        assert "Status pemeriksaan" in html
        assert escape(nr.AUDIENCE_LABELS[pembaca]) in html
    assert "Kalimat siap salin" in laporan.markdown("akademik")
    assert "Kalimat siap salin" not in laporan.markdown("eksekutif")


def test_konfigurasi_minimal_tetap_menghasilkan_laporan(data):
    konfig = nr.Konfigurasi(variabel=NUMERIK[:3], nama_data="uji.csv")
    analisis, laporan = nr.analisis_dan_laporan(data, konfig)
    assert analisis.regresi is None and analisis.manova is None
    assert len(laporan.temuan) >= 4
    assert laporan.headline
    assert laporan.markdown("eksekutif")


def test_analisis_gagal_dicatat_bukan_dilempar(data):
    rusak = data.copy()
    rusak["konstan"] = 1.0
    konfig = nr.Konfigurasi(
        variabel=["konstan", "usia", "skor_kredit"],
        nama_data="uji.csv",
        kelompok="segmen_usaha",
    )
    analisis = nr.jalankan_analisis(rusak, konfig)
    laporan = nr.susun_laporan(analisis)
    assert laporan.headline  # laporan tetap tersusun meski sebagian metode gagal
    assert len(laporan.dilewati) == len(set(laporan.dilewati))


def test_teks_tidak_memuat_placeholder_kosong(hasil):
    _, laporan = hasil
    for temuan in laporan.temuan:
        for pembaca in nr.AUDIENCES:
            teks = temuan.teks(pembaca)
            # "nan" sebagai kata utuh menandakan angka gagal dihitung; "bulanan" tidak.
            assert not re.search(r"\bnan\b", teks, flags=re.IGNORECASE), teks
            assert "  " not in teks
            assert "None" not in teks
            assert "  ." not in teks and " ." not in teks


def test_diskriminan_tanpa_validasi_silang(data):
    """Kelompok dengan anggota tunggal membuat validasi silang gagal; narasi harus jujur."""
    langka = data.copy()
    langka["kelompok_uji"] = "besar"
    langka.loc[langka.index[0], "kelompok_uji"] = "langka"
    konfig = nr.Konfigurasi(
        variabel=NUMERIK[:4],
        nama_data="uji.csv",
        kelompok="kelompok_uji",
    )
    analisis = nr.jalankan_analisis(langka, konfig)
    assert analisis.diskriminan is not None
    assert pd.isna(analisis.diskriminan.cv_accuracy)

    temuan = nr.temuan_diskriminan(analisis)
    for pembaca in nr.AUDIENCES:
        teks = temuan.teks(pembaca)
        assert "-%" not in teks
        assert "validasi silang" in teks.lower()
    assert "-%" not in temuan.ringkas


def test_laporan_html_gabungan_memuat_ketiga_pembaca(hasil):
    from nalardata.report_html import laporan_html_semua

    _, laporan = hasil
    html = laporan_html_semua(laporan)
    assert html.count('class="panel"') == len(nr.AUDIENCES)
    for pembaca in nr.AUDIENCES:
        assert f'data-panel="{pembaca}"' in html
    # Bagian khas akademik hanya muncul di panelnya sendiri.
    assert html.count("Kalimat siap salin") == 1


# --------------------------------------------------------------------------- #
# Kunci kausalitas pada laporan sungguhan
# --------------------------------------------------------------------------- #


def _seluruh_teks(lap) -> str:
    """Setiap ruas teks laporan, tanpa kecuali.

    Memeriksa sebagian saja akan meloloskan kebocoran di ruas yang terlupa — dan
    ruas yang terlupa itulah yang justru akan dibaca penguji.
    """
    bagian = [lap.headline, lap.subheadline, lap.pendorong_sumber]
    bagian += [t.judul + t.ringkas + t.eksekutif + t.akademik + t.profesional for t in lap.temuan]
    bagian += [r.judul + r.alasan for r in lap.rekomendasi]
    bagian += [p.teks for p in lap.paragraf]
    bagian += [l.catatan for l in lap.lampu]
    bagian += list(lap.keterbatasan)
    bagian += [d.catatan for d in lap.pendorong]
    return " ".join(bagian)


def test_laporan_potong_lintang_tidak_memuat_satu_pun_ungkapan_sebab(data, konfig_lengkap):
    """Invarian utama kunci kausalitas, diuji pada laporan yang benar-benar disusun."""
    from nalardata import pagar
    from nalardata import proyek_penelitian as pp

    lintang = pp.ProyekPenelitian(desain="potong_lintang", teknik_sampling="purposif")
    _, lap = nr.analisis_dan_laporan(data, konfig_lengkap, lintang)
    assert pagar.periksa_kausalitas(_seluruh_teks(lap), lintang) == []


def test_laporan_eksperimen_acak_boleh_memakai_bahasa_sebab(data, konfig_lengkap):
    from nalardata import proyek_penelitian as pp

    eksperimen = pp.ProyekPenelitian(
        desain="eksperimen", penugasan_acak=True, teknik_sampling="acak_sederhana"
    )
    _, lap = nr.analisis_dan_laporan(data, konfig_lengkap, eksperimen)
    assert "berpengaruh" in _seluruh_teks(lap)


def test_tanpa_rancangan_kunci_tetap_menutup(data, konfig_lengkap):
    """Pengguna yang melewati Ruang Proyek tetap terlindungi."""
    from nalardata import pagar

    _, lap = nr.analisis_dan_laporan(data, konfig_lengkap, None)
    assert pagar.periksa_kausalitas(_seluruh_teks(lap), None) == []
    assert any("belum diisi" in k for k in lap.keterbatasan)


def test_batas_rancangan_muncul_paling_depan_pada_keterbatasan(data, konfig_lengkap):
    from nalardata import proyek_penelitian as pp

    lintang = pp.ProyekPenelitian(desain="potong_lintang", teknik_sampling="insidental")
    _, lap = nr.analisis_dan_laporan(data, konfig_lengkap, lintang)
    assert "bukan sebab-akibat" in lap.keterbatasan[0]
    assert any("bukan bagi seluruh populasi" in k for k in lap.keterbatasan)


def test_keterbatasan_tidak_terduplikasi(data, konfig_lengkap):
    from nalardata import proyek_penelitian as pp

    _, lap = nr.analisis_dan_laporan(
        data, konfig_lengkap, pp.ProyekPenelitian(desain="potong_lintang")
    )
    assert len(lap.keterbatasan) == len(set(lap.keterbatasan))


def test_setiap_rujukan_menyebut_apa_yang_dirujuknya():
    """Rujukan tanpa keterangan tidak dapat ditelusuri pembaca.

    Aplikasi mengutip ambang statistik dari kepustakaan; tiap butir wajib
    menyebutkan ambang mana yang bersandar padanya, bukan sekadar nama penulis.
    """
    for rujukan in nr.RUJUKAN:
        assert "—" in rujukan, rujukan
        penulis, keterangan = rujukan.split("—", 1)
        assert len(keterangan.strip()) > 15
        assert any(tahun in penulis for tahun in ("19", "20"))


def test_rujukan_memuat_dasar_ambang_yang_paling_sering_dipakai():
    """Ambang yang paling banyak dipakai pengguna kuesioner harus punya dasarnya."""
    gabungan = " ".join(nr.RUJUKAN)
    for kata in ("HTMT", "omega", "Fornell-Larcker", "KMO", "efek"):
        assert kata in gabungan


# --------------------------------------------------------------------------- #
# Fase B — pembangun temuan_* yang melengkapi baterai otomatis
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def data_survei() -> pd.DataFrame:
    """Data kuesioner sintetis dengan dua konstruk laten yang benar-benar ditanam.

    Butir bernama KUALn/PUASn (berpola nomor) agar ``reliability.tebak_konstruk``
    mengelompokkannya secara otomatis, persis seperti yang dibaca ``jalankan_analisis``.
    """
    rng = np.random.default_rng(42)
    n = 200
    kual = rng.normal(0, 1, n)
    puas = rng.normal(0, 1, n)
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


def test_uji_beda_dua_kelompok_terpasang_otomatis(data):
    konfig = nr.Konfigurasi(
        variabel=NUMERIK,
        nama_data="uji.csv",
        target_numerik="skor_kredit",
        kelompok="gagal_bayar",
    )
    analisis = nr.jalankan_analisis(data, konfig)
    assert analisis.uji_beda is not None
    assert analisis.uji_beda.n_kelompok == 2

    temuan = nr.temuan_uji_beda(analisis)
    for pembaca in nr.AUDIENCES:
        teks = temuan.teks(pembaca)
        assert len(teks) > 40
        assert not re.search(r"\bnan\b", teks, flags=re.IGNORECASE)


def test_uji_beda_tiga_kelompok_memilih_anova_atau_kruskal(data):
    konfig = nr.Konfigurasi(
        variabel=NUMERIK,
        nama_data="uji.csv",
        target_numerik="skor_kredit",
        kelompok="segmen_usaha",
    )
    analisis = nr.jalankan_analisis(data, konfig)
    assert analisis.uji_beda is not None
    assert analisis.uji_beda.n_kelompok == 3
    assert analisis.uji_beda.uji.kode in {"anova", "kruskal"}


def test_moderasi_terpasang_saat_moderator_diisi(data):
    konfig = nr.Konfigurasi(
        variabel=["skor_kredit", "pendapatan_bulanan", "lama_usaha_tahun"],
        nama_data="uji.csv",
        target_numerik="skor_kredit",
        prediktor=["pendapatan_bulanan", "lama_usaha_tahun"],
        moderator="lama_usaha_tahun",
    )
    analisis = nr.jalankan_analisis(data, konfig)
    assert analisis.moderasi is not None
    assert analisis.moderasi.x == "pendapatan_bulanan"
    assert analisis.moderasi.m == "lama_usaha_tahun"

    temuan = nr.temuan_moderasi(analisis)
    assert "pengaruh" not in temuan.judul.lower()
    assert "terhadap" not in temuan.judul.lower()
    for pembaca in nr.AUDIENCES:
        assert len(temuan.teks(pembaca)) > 40


def test_moderasi_tanpa_moderator_tidak_terpasang(data):
    konfig = nr.Konfigurasi(
        variabel=["skor_kredit", "pendapatan_bulanan", "lama_usaha_tahun"],
        nama_data="uji.csv",
        target_numerik="skor_kredit",
        prediktor=["pendapatan_bulanan", "lama_usaha_tahun"],
    )
    analisis = nr.jalankan_analisis(data, konfig)
    assert analisis.moderasi is None
    with pytest.raises(ValueError):
        nr.temuan_moderasi(analisis)


def test_reliabilitas_dan_cfa_terpasang_otomatis_dari_pola_nama_butir(data_survei):
    konfig = nr.Konfigurasi(variabel=list(data_survei.columns), nama_data="survei.csv")
    analisis = nr.jalankan_analisis(data_survei, konfig)
    assert analisis.gagal == {}
    assert analisis.reliabilitas is not None
    assert {h.nama for h in analisis.reliabilitas} == {"KUAL", "PUAS"}
    assert all(h.memenuhi() for h in analisis.reliabilitas)  # konstruk ditanam kuat
    assert analisis.sem is not None
    assert analisis.sem.cocok()

    temuan_rel = nr.temuan_reliabilitas(analisis)
    temuan_sem = nr.temuan_sem(analisis)
    for temuan in (temuan_rel, temuan_sem):
        for pembaca in nr.AUDIENCES:
            teks = temuan.teks(pembaca)
            assert len(teks) > 40
            assert not re.search(r"\bnan\b", teks, flags=re.IGNORECASE)
            assert "None" not in teks


def test_reliabilitas_tanpa_pola_nama_butir_tidak_terpasang(data):
    """Data non-kuesioner (tanpa butir bernomor) tidak boleh salah dikira survei."""
    konfig = nr.Konfigurasi(variabel=NUMERIK, nama_data="uji.csv")
    analisis = nr.jalankan_analisis(data, konfig)
    assert analisis.reliabilitas is None
    assert analisis.sem is None


def test_temuan_baru_menghormati_kunci_kausalitas(data, data_survei):
    """Uji beda, moderasi, reliabilitas, dan CFA tidak boleh lolos satu pun ungkapan
    sebab-akibat pada rancangan potong lintang — termasuk pada judulnya, yang tidak
    ikut diproses ulang oleh ``kunci_kesimpulan`` sehingga harus aman sejak ditulis."""
    from nalardata import pagar
    from nalardata import proyek_penelitian as pp

    lintang = pp.ProyekPenelitian(desain="potong_lintang")

    konfig = nr.Konfigurasi(
        variabel=["skor_kredit", "pendapatan_bulanan", "lama_usaha_tahun", "segmen_usaha"],
        nama_data="uji.csv",
        target_numerik="skor_kredit",
        prediktor=["pendapatan_bulanan", "lama_usaha_tahun"],
        moderator="lama_usaha_tahun",
        kelompok="segmen_usaha",
    )
    _, lap = nr.analisis_dan_laporan(data, konfig, lintang)
    assert pagar.periksa_kausalitas(_seluruh_teks(lap), lintang) == []
    assert any("Uji beda" in m or "Moderasi" in t.judul for m, t in zip(lap.metode_terpakai, lap.temuan))

    konfig_survei = nr.Konfigurasi(variabel=list(data_survei.columns), nama_data="survei.csv")
    _, lap_survei = nr.analisis_dan_laporan(data_survei, konfig_survei, lintang)
    assert pagar.periksa_kausalitas(_seluruh_teks(lap_survei), lintang) == []
