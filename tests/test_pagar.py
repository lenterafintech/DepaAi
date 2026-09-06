"""Uji pagar metodologis.

Pagar diuji sebagai perilaku: kalimat mana yang tertutup pada rancangan mana, dan
apakah hasil pelurusannya benar-benar lolos pemeriksaannya sendiri. Pagar yang
mengaku memblokir tetapi meloloskan satu ungkapan lebih berbahaya daripada tidak
ada pagar sama sekali, karena ia menumbuhkan rasa aman yang keliru.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nalardata import pagar
from nalardata import proyek_penelitian as pp


@pytest.fixture
def lintang() -> pp.ProyekPenelitian:
    return pp.ProyekPenelitian(desain="potong_lintang")


@pytest.fixture
def eksperimen() -> pp.ProyekPenelitian:
    return pp.ProyekPenelitian(desain="eksperimen", penugasan_acak=True)


# --------------------------------------------------------------------------- #
# Kunci kausalitas — pemilihan kata di hulu
# --------------------------------------------------------------------------- #


def test_kata_hubungan_mengikuti_rancangan(lintang, eksperimen):
    assert pagar.kata_hubungan(lintang) == "berhubungan dengan"
    assert pagar.kata_hubungan(eksperimen) == "berpengaruh terhadap"


def test_kata_benda_mengikuti_rancangan(lintang, eksperimen):
    assert pagar.kata_benda_hubungan(lintang) == "hubungan"
    assert pagar.kata_benda_hubungan(eksperimen) == "pengaruh"


def test_tanpa_rancangan_dipakai_kosakata_paling_berhati_hati():
    """Pengguna yang melewati Ruang Proyek tetap harus terlindungi."""
    assert pagar.kata_hubungan(None) == "berhubungan dengan"
    assert pagar.kata_benda_hubungan(None) == "hubungan"


# --------------------------------------------------------------------------- #
# Kunci kausalitas — pemeriksaan
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kalimat",
    [
        "Kualitas layanan berpengaruh terhadap kepuasan.",
        "Harga memengaruhi keputusan pembelian.",
        "Kenaikan suku bunga menyebabkan penurunan permintaan.",
        "Pelatihan meningkatkan kinerja karyawan.",
        "Promosi berdampak pada penjualan.",
    ],
)
def test_ungkapan_sebab_ditemukan_pada_potong_lintang(kalimat, lintang):
    assert pagar.periksa_kausalitas(kalimat, lintang)


def test_kalimat_asosiatif_lolos(lintang):
    assert pagar.periksa_kausalitas(
        "Kualitas layanan berhubungan positif dengan kepuasan.", lintang
    ) == []


def test_ungkapan_sebab_dibolehkan_pada_eksperimen_acak(eksperimen):
    assert pagar.periksa_kausalitas(
        "Pelatihan meningkatkan kinerja karyawan.", eksperimen
    ) == []


def test_temuan_tidak_dilaporkan_berlapis(lintang):
    """'berpengaruh terhadap' tidak boleh dilaporkan sekaligus sebagai 'pengaruh terhadap'."""
    temuan = pagar.periksa_kausalitas("A berpengaruh terhadap B.", lintang)
    assert temuan == ["berpengaruh terhadap"]


# --------------------------------------------------------------------------- #
# Kunci kausalitas — pelurusan
# --------------------------------------------------------------------------- #


def test_pelurusan_menghasilkan_kalimat_yang_lolos_pemeriksaan(lintang):
    """Invarian terpenting: keluaran pelurusan tidak boleh menyisakan satu pun ungkapan."""
    kalimat = (
        "Kualitas layanan berpengaruh terhadap kepuasan, harga memengaruhi keputusan, "
        "dan pelatihan meningkatkan kinerja serta menurunkan keluhan."
    )
    lurus = pagar.luruskan_kausalitas(kalimat, lintang)
    assert pagar.periksa_kausalitas(lurus, lintang) == []


def test_pelurusan_mempertahankan_huruf_besar_di_awal_kalimat(lintang):
    lurus = pagar.luruskan_kausalitas("Memengaruhi kepuasan secara nyata.", lintang)
    assert lurus.startswith("Berhubungan dengan")


def test_pelurusan_tidak_mengubah_apa_pun_pada_eksperimen(eksperimen):
    kalimat = "Pelatihan meningkatkan kinerja."
    assert pagar.luruskan_kausalitas(kalimat, eksperimen) == kalimat


def test_ungkapan_panjang_diproses_sebelum_yang_pendek(lintang):
    """Urutan padanan salah akan menghasilkan 'berhubungan dengan terhadap'."""
    lurus = pagar.luruskan_kausalitas("A berpengaruh terhadap B.", lintang)
    assert lurus == "A berhubungan dengan B."


# --------------------------------------------------------------------------- #
# Gerbang asumsi
# --------------------------------------------------------------------------- #


def test_tanpa_pelanggaran_gerbang_diam():
    assert pagar.gerbang_asumsi([]) == ""


def test_satu_pelanggaran_disebut_lengkap():
    pesan = pagar.gerbang_asumsi(
        [pagar.Pelanggaran("Normalitas", "Ditolak pada 2 kelompok.", "Pakai uji non-parametrik.")]
    )
    assert "Normalitas" in pesan
    assert "non-parametrik" in pesan


def test_banyak_pelanggaran_disebut_jumlah_dan_namanya():
    pesan = pagar.gerbang_asumsi(
        [
            pagar.Pelanggaran("Normalitas", "a", "b"),
            pagar.Pelanggaran("Homogenitas", "a", "b"),
            pagar.Pelanggaran("Linearitas", "a", "b"),
        ]
    )
    assert "3 asumsi" in pesan
    assert "Homogenitas" in pesan


def test_setiap_asumsi_baku_punya_akibat_yang_dijelaskan():
    for asumsi in pagar.AKIBAT_ASUMSI:
        assert len(pagar.akibat_pelanggaran(asumsi)) > 40


def test_asumsi_asing_tetap_mendapat_kalimat():
    assert pagar.akibat_pelanggaran("sesuatu yang belum terdaftar").strip()


# --------------------------------------------------------------------------- #
# Bukan nilai-p saja
# --------------------------------------------------------------------------- #


def test_tabel_hanya_berisi_p_ditolak():
    tabel = pd.DataFrame({"Variabel": ["x"], "t": [2.5], "Sig.": [0.02]})
    assert not pagar.cukup_dari_nilai_p(tabel)
    assert "ukuran efek" in pagar.peringatan_nilai_p(tabel)


@pytest.mark.parametrize("kolom", ["Cohen's d", "Eta kuadrat", "R²", "Odds ratio"])
def test_tabel_berukuran_efek_diterima(kolom):
    tabel = pd.DataFrame({"Variabel": ["x"], "Sig.": [0.02], kolom: [0.4]})
    assert pagar.cukup_dari_nilai_p(tabel)
    assert pagar.peringatan_nilai_p(tabel) == ""


def test_tabel_berselang_kepercayaan_diterima():
    tabel = pd.DataFrame(
        {"Variabel": ["x"], "Sig.": [0.02], "CI bawah": [0.1], "CI atas": [0.6]}
    )
    assert pagar.cukup_dari_nilai_p(tabel)


def test_tabel_kosong_ditolak():
    assert not pagar.cukup_dari_nilai_p(pd.DataFrame())


# --------------------------------------------------------------------------- #
# Penanda eksploratori
# --------------------------------------------------------------------------- #


def test_tanpa_praregistrasi_semua_analisis_eksploratori(lintang):
    assert pagar.eksploratori("Regresi linear berganda", lintang)
    assert "belum ada praregistrasi" in pagar.label_eksploratori("Regresi", lintang).lower()


def test_uji_yang_direncanakan_tidak_ditandai():
    proyek = pp.ProyekPenelitian(
        praregistrasi=pp.Praregistrasi(uji_direncanakan=["Regresi linear berganda"])
    )
    assert not pagar.eksploratori("Regresi linear berganda", proyek)
    assert pagar.label_eksploratori("Regresi linear berganda", proyek) == ""


def test_uji_di_luar_rencana_ditandai_dengan_namanya():
    proyek = pp.ProyekPenelitian(
        praregistrasi=pp.Praregistrasi(uji_direncanakan=["Regresi linear berganda"])
    )
    label = pagar.label_eksploratori("Analisis klaster", proyek)
    assert "Analisis klaster" in label
    assert "menyembunyikannya tidak" in label


def test_penamaan_yang_sedikit_berbeda_tetap_dianggap_terencana():
    """Pengguna menulis rencananya dengan kata-katanya sendiri, bukan istilah aplikasi."""
    proyek = pp.ProyekPenelitian(
        praregistrasi=pp.Praregistrasi(uji_direncanakan=["regresi berganda"])
    )
    assert not pagar.eksploratori("Regresi berganda", proyek)


# --------------------------------------------------------------------------- #
# Tinjauan manusia
# --------------------------------------------------------------------------- #


def test_sem_pada_sampel_kecil_ditandai():
    alasan = pagar.perlu_tinjauan_manusia(120, "SEM")
    assert "200" in alasan


def test_sem_pada_sampel_besar_tidak_ditandai():
    assert pagar.perlu_tinjauan_manusia(500, "SEM") == ""


def test_parameter_terlalu_banyak_untuk_sampelnya_ditandai():
    alasan = pagar.perlu_tinjauan_manusia(100, "CFA", parameter=40)
    assert "lima responden per parameter" in alasan


def test_banyak_asumsi_dilanggar_sekaligus_ditandai():
    pelanggaran = [pagar.Pelanggaran(f"A{i}", "a", "b") for i in range(3)]
    assert "tidak terpenuhi sekaligus" in pagar.perlu_tinjauan_manusia(
        400, "Regresi", pelanggaran
    )


def test_sampel_sangat_kecil_selalu_ditandai():
    assert "tidak stabil" in pagar.perlu_tinjauan_manusia(25, "Korelasi")


def test_analisis_wajar_tidak_ditandai():
    assert pagar.perlu_tinjauan_manusia(400, "Regresi linear", [], parameter=5) == ""


# --------------------------------------------------------------------------- #
# Keutuhan daftar kosakata
# --------------------------------------------------------------------------- #


def test_kosakata_memuat_ungkapan_yang_lazim_dipakai_skripsi():
    daftar = [asal for asal, _ in pagar.PADANAN]
    for kata in (
        "berpengaruh terhadap",
        "berpengaruh",
        "dipengaruhi",
        "memengaruhi",
        "menyebabkan",
        "meningkatkan",
        "pengaruh",
        "dampak",
    ):
        assert kata in daftar


def test_ungkapan_panjang_terdaftar_sebelum_bagiannya_yang_pendek():
    """Urutan yang salah membuat pelurusan memotong ungkapan di tengah.

    Bila "berpengaruh" diproses sebelum "berpengaruh terhadap", hasilnya menjadi
    "berhubungan terhadap" — kalimat rusak yang tetap lolos pemeriksaan.
    """
    daftar = [asal for asal, _ in pagar.PADANAN]
    for i, pendek in enumerate(daftar):
        for panjang in daftar[i + 1 :]:
            # Pelurusan mencocokkan dengan batas kata, jadi "pengaruh" tidak
            # pernah mengenai "mempengaruhi". Yang berbahaya hanyalah ungkapan
            # pendek yang dapat cocok di awal kata pada ungkapan yang lebih panjang.
            berbahaya = panjang.startswith(pendek) or f" {pendek}" in panjang
            assert not berbahaya, (
                f"'{panjang}' memuat '{pendek}' pada batas kata "
                "tetapi terdaftar sesudahnya"
            )


def test_setiap_padanan_bebas_dari_kosakata_sebab(lintang):
    """Padanan yang masih memuat ungkapan sebab akan lolos memeriksa dirinya sendiri."""
    for _, ganti in pagar.PADANAN:
        assert pagar.periksa_kausalitas(ganti, lintang) == []


def test_penjelasan_kunci_tidak_memakai_kosakata_yang_dikuncinya(lintang):
    """Kalimat yang menerangkan mengapa sebab-akibat dikunci tidak boleh memakainya."""
    for desain in pp.DESAIN.values():
        assert pagar.periksa_kausalitas(desain.alasan, lintang) == []
    for cara in pp.SAMPLING.values():
        assert pagar.periksa_kausalitas(cara.alasan, lintang) == []
