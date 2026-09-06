"""Uji penyusun kesimpulan naratif tiga register pembaca."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

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
    ):
        assert getattr(analisis, atribut) is not None, atribut


def test_laporan_memuat_seluruh_metode(hasil):
    _, laporan = hasil
    assert laporan.dilewati == []
    assert len(laporan.temuan) == 10
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
