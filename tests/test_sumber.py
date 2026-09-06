"""Uji penelusuran angka narasi kembali ke sel tabelnya.

Penelusuran ini bukan hiasan: ia satu-satunya cara pembimbing memeriksa paragraf
yang disusun otomatis, dan satu-satunya cara penulisnya mempertahankan angka di
sidang. Penelusuran yang meloloskan angka tak bersumber lebih berbahaya daripada
tidak ada penelusuran, karena ia menumbuhkan rasa aman yang keliru.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nalardata import narrative as nr
from nalardata import sumber as sm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "contoh_data_nasabah.csv")


@pytest.fixture(scope="module")
def laporan(data) -> nr.Laporan:
    konfig = nr.Konfigurasi(
        variabel=[
            "usia",
            "pendapatan_bulanan",
            "rasio_utang_pendapatan",
            "skor_kredit",
            "jumlah_keterlambatan",
        ],
        nama_data="contoh_data_nasabah.csv",
        target_numerik="skor_kredit",
        prediktor=["rasio_utang_pendapatan", "jumlah_keterlambatan"],
        kelompok="segmen_usaha",
    )
    return nr.analisis_dan_laporan(data, konfig)[1]


@pytest.fixture(scope="module")
def indeks(laporan) -> list[sm.Sel]:
    return sm.indeks(laporan.tabel)


# --------------------------------------------------------------------------- #
# Penguraian angka
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "teks, harapan",
    [
        ("0,656", 0.656),
        ("-269,2985", -269.2985),
        ("1.234,56", 1234.56),
        ("400", 400.0),
        ("bukan angka", None),
    ],
)
def test_angka_bergaya_indonesia_terbaca(teks, harapan):
    assert sm.ke_angka(teks) == harapan


def test_sel_mengambil_angka_sesudah_tanda_sama_dengan():
    """'χ²(10) = 718,01' memuat derajat bebas dan statistik uji sekaligus."""
    assert sm._nilai("χ²(10) = 718,01") == pytest.approx(718.01)
    assert sm._nilai("χ²(30) = 130,08") == pytest.approx(130.08)


def test_sel_kosong_diabaikan():
    for kosong in ("—", "-", "", "nan", None):
        assert sm._nilai(kosong) is None


# --------------------------------------------------------------------------- #
# Indeks
# --------------------------------------------------------------------------- #


def test_indeks_mencakup_seluruh_tabel(indeks, laporan):
    assert len(indeks) > 50
    assert {s.tabel for s in indeks} == set(laporan.tabel)


def test_setiap_sel_menyebut_letaknya(indeks):
    for sel in indeks[:20]:
        letak = sel.letak()
        assert sel.tabel in letak and sel.kolom in letak


def test_indeks_tabel_kosong_menghasilkan_daftar_kosong():
    assert sm.indeks({}) == []
    assert sm.indeks(None) == []


def test_isi_tabel_cacat_dilewati_tanpa_galat():
    assert sm.indeks({"Tabel 1": "bukan tuple"}) == []
    assert sm.indeks({"Tabel 1": ("judul", pd.DataFrame(), "")}) == []


# --------------------------------------------------------------------------- #
# Pencocokan
# --------------------------------------------------------------------------- #


def test_kalimat_yang_membulatkan_tetap_cocok_dengan_tabel(indeks):
    """Kalimat menyebut t = -14,90 sementara tabel menuliskan -14,899."""
    daftar = sm.cari(-14.90, 2, indeks)
    assert any(s.kolom == "t" for s in daftar)


def test_persentase_ditelusuri_ke_proporsinya(indeks):
    """Narasi menyebut 65,6% untuk R² yang tersimpan sebagai 0,656."""
    assert sm.cari(65.6, 1, indeks)


def test_angka_yang_tidak_ada_tidak_dicocokkan_paksa(indeks):
    assert sm.cari(123456.789, 3, indeks) == []


# --------------------------------------------------------------------------- #
# Penelusuran kalimat
# --------------------------------------------------------------------------- #


def test_bilangan_bulat_kecil_tidak_ditelusuri(indeks):
    """'2 dari 3 kelompok' akan cocok dengan hampir semua tabel dan hanya derau."""
    petunjuk = sm.telusuri("Normalitas ditolak pada 2 dari 3 kelompok.", indeks)
    assert petunjuk == []


def test_tahun_sitasi_tidak_ditelusuri(indeks):
    """'Hair dkk. (2019)' bukan hasil analisis dan tidak punya sel asal."""
    assert sm.telusuri("Ambang mengikuti Hair dkk. (2019).", indeks) == []


def test_tahun_di_luar_sitasi_tetap_ditelusuri(indeks):
    """Angka 2019 sebagai nilai data bukan tahun terbitan."""
    assert sm.telusuri("Nilai tertinggi mencapai 2019 unit.", indeks)


def test_paragraf_regresi_dapat_ditelusuri_seluruhnya(laporan, indeks):
    """Invarian utama: koefisien yang dikutip narasi wajib ada di tabel.

    Tanpa ini, pembimbing yang menerima 'B = -269,2985' tidak punya cara
    memeriksanya selain percaya.
    """
    temuan = [t for t in laporan.temuan if "Regresi" in t.metode]
    assert temuan, "laporan uji harus memuat temuan regresi"

    petunjuk = sm.telusuri(temuan[0].akademik, indeks)
    berdesimal = [p for p in petunjuk if "," in p.tampil]
    bersumber = [p for p in berdesimal if p.bersumber]
    assert len(bersumber) / len(berdesimal) > 0.9


def test_statistik_model_punya_selnya_sendiri(laporan, indeks):
    """R² dan F sebelumnya hanya ada pada catatan kaki, yang bukan sel."""
    judul = [j for j, _, _ in laporan.tabel.values()]
    assert any("Ringkasan model" in j for j in judul)
    assert sm.cari(0.656, 3, indeks), "R² harus dapat ditunjuk sebagai sel"


@pytest.mark.parametrize(
    "kata", ["normalitas", "komponen utama", "klaster", "diskriminan"]
)
def test_metode_yang_dinarasikan_punya_tabelnya(laporan, kata):
    """Metode tanpa tabel menghasilkan paragraf yang tidak dapat diperiksa siapa pun."""
    judul = " ".join(j for j, _, _ in laporan.tabel.values()).lower()
    assert kata in judul


def test_tabel_yang_gagal_disusun_tercatat_bukan_hilang(data):
    """Tabel klaster pernah absen tanpa satu pun tanda di layar."""
    konfig = nr.Konfigurasi(
        variabel=["usia", "skor_kredit", "pendapatan_bulanan"],
        nama_data="d.csv",
    )
    analisis = nr.jalankan_analisis(data, konfig)
    nr.susun_tabel(analisis)
    for pesan in analisis.gagal.values():
        assert "Tabel gagal disusun" not in pesan, analisis.gagal


# --------------------------------------------------------------------------- #
# Tampilan
# --------------------------------------------------------------------------- #


def test_ringkas_menyandingkan_angka_dan_letaknya(laporan, indeks):
    temuan = [t for t in laporan.temuan if "Regresi" in t.metode][0]
    tabel = sm.ringkas(temuan.akademik, indeks)
    assert list(tabel.columns) == ["Angka", "Sumber"]
    assert len(tabel) > 5
    assert tabel["Sumber"].str.contains("Tabel").any()


def test_angka_tanpa_sumber_dinyatakan_terbuka(indeks):
    """Angka yang tidak ada di tabel harus terlihat, bukan disembunyikan."""
    tabel = sm.ringkas("Nilainya mencapai 987654,321 satuan.", indeks)
    assert (tabel["Sumber"] == "tidak ditemukan di tabel mana pun").all()


def test_kalimat_tanpa_angka_menghasilkan_tabel_kosong(indeks):
    assert sm.ringkas("Tidak ada angka di sini.", indeks).empty
