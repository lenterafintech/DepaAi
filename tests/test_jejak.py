"""Uji jejak keputusan dan jejak uji.

Yang diuji adalah angka dan keputusannya: berapa uji tercatat, kapan peringatan uji
berganda menyala, dan berapa temuan yang kehilangan signifikansi setelah dikoreksi.
Peringatan yang menyala di saat yang keliru akan diabaikan pengguna, dan pagar yang
diabaikan sama dengan tidak ada.
"""

from __future__ import annotations

import pytest

from nalardata import jejak as jk
from nalardata import proyek_penelitian as pp


@pytest.fixture
def jejak_terisi() -> jk.Jejak:
    j = jk.Jejak()
    for nomor, p in enumerate([0.001, 0.02, 0.04, 0.30, 0.60, 0.045], start=1):
        j.catat_uji(f"Uji {nomor}", halaman="Regresi", p=p)
    return j


# --------------------------------------------------------------------------- #
# Pencatatan
# --------------------------------------------------------------------------- #


def test_jejak_baru_kosong():
    assert jk.Jejak().kosong()


def test_langkah_dicatat_beserta_waktunya():
    langkah = jk.Jejak().catat_keputusan("Memilih Kruskal-Wallis", halaman="Pemandu")
    assert langkah.waktu
    assert langkah.jenis == jk.KEPUTUSAN


def test_langkah_yang_sama_persis_tidak_dicatat_dua_kali():
    """Streamlit merender ulang halaman; render ulang bukan uji tambahan."""
    j = jk.Jejak()
    j.catat_uji("Regresi linear", halaman="Regresi", p=0.01)
    j.catat_uji("Regresi linear", halaman="Regresi", p=0.01)
    assert len(j.uji()) == 1


def test_uji_sama_dengan_nilai_p_berbeda_dicatat_terpisah():
    """Mengganti variabel menghasilkan nilai p lain, dan itu memang uji baru."""
    j = jk.Jejak()
    j.catat_uji("Regresi linear", halaman="Regresi", p=0.01)
    j.catat_uji("Regresi linear", halaman="Regresi", p=0.20)
    assert len(j.uji()) == 2


def test_jenis_langkah_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        jk.Langkah(jenis="entah", ringkas="x")


def test_hitungan_memisahkan_jenis_langkah(jejak_terisi):
    jejak_terisi.catat_perubahan("Menghapus 3 baris duplikat", halaman="Rapor Data")
    jejak_terisi.catat_keputusan("Memakai galat baku robust", halaman="Regresi")
    hitung = jejak_terisi.hitungan()
    assert hitung["Uji dijalankan"] == 6
    assert hitung["Hasil signifikan"] == 4
    assert hitung["Perubahan data"] == 1


def test_kosongkan_menghapus_seluruh_langkah(jejak_terisi):
    jejak_terisi.kosongkan()
    assert jejak_terisi.kosong()


# --------------------------------------------------------------------------- #
# Uji berganda
# --------------------------------------------------------------------------- #


def test_peringatan_diam_selama_uji_masih_sedikit():
    j = jk.Jejak()
    for nomor in range(3):
        j.catat_uji(f"Uji {nomor}", p=0.01 * (nomor + 1))
    assert j.peringatan_uji_berganda() == ""


def test_peringatan_menyala_setelah_melewati_ambang(jejak_terisi):
    pesan = jejak_terisi.peringatan_uji_berganda()
    assert "6 uji" in pesan
    assert "26%" in pesan


def test_peluang_kebetulan_naik_bersama_jumlah_uji():
    j = jk.Jejak()
    for nomor in range(20):
        j.catat_uji(f"Uji {nomor}", p=0.5)
    # 1 - 0,95^20 sudah mendekati dua pertiga.
    assert "64%" in j.peringatan_uji_berganda()


def test_koreksi_holm_mencabut_signifikansi_temuan_lemah(jejak_terisi):
    tabel = jejak_terisi.koreksi_berganda("holm")
    assert len(tabel) == 6
    assert tabel.loc[0, "Signifikan sesudah"]
    assert jejak_terisi.berubah_setelah_koreksi("holm") == 3


def test_bonferroni_lebih_ketat_daripada_benjamini_hochberg(jejak_terisi):
    ketat = jejak_terisi.berubah_setelah_koreksi("bonferroni")
    longgar = jejak_terisi.berubah_setelah_koreksi("fdr_bh")
    assert ketat >= longgar


def test_metode_koreksi_asing_ditolak(jejak_terisi):
    with pytest.raises(ValueError, match="tidak dikenal"):
        jejak_terisi.koreksi_berganda("sidak")


def test_koreksi_tanpa_uji_ditolak_dengan_pesan_jelas():
    with pytest.raises(ValueError, match="Belum ada uji"):
        jk.Jejak().koreksi_berganda()


def test_berubah_setelah_koreksi_nol_bila_belum_ada_uji():
    assert jk.Jejak().berubah_setelah_koreksi() == 0


def test_setiap_metode_koreksi_dijelaskan():
    for keterangan in jk.METODE_KOREKSI.values():
        assert "—" in keterangan


# --------------------------------------------------------------------------- #
# Perbandingan dengan praregistrasi
# --------------------------------------------------------------------------- #


def test_tanpa_praregistrasi_semua_dianggap_di_luar_rencana(jejak_terisi):
    banding = jejak_terisi.banding_rencana(None)
    assert not banding.ada_praregistrasi
    assert len(banding.di_luar_rencana) == 6
    assert "eksploratori" in banding.kalimat()


def test_uji_terencana_dan_tidak_terencana_dipisahkan(jejak_terisi):
    pra = pp.Praregistrasi(uji_direncanakan=["Uji 1", "Uji 2"])
    banding = jejak_terisi.banding_rencana(pra)
    assert set(banding.sesuai_rencana) == {"Uji 1", "Uji 2"}
    assert len(banding.di_luar_rencana) == 4


def test_uji_yang_direncanakan_tetapi_belum_dijalankan_disebut():
    j = jk.Jejak()
    j.catat_uji("Regresi linear berganda", p=0.01)
    pra = pp.Praregistrasi(
        uji_direncanakan=["Regresi linear berganda", "Analisis klaster"]
    )
    banding = j.banding_rencana(pra)
    assert banding.direncanakan_belum_jalan == ["Analisis klaster"]
    assert "belum dijalankan" in banding.kalimat()


def test_penamaan_yang_sedikit_berbeda_tetap_dianggap_cocok():
    j = jk.Jejak()
    j.catat_uji("Regresi linear berganda", p=0.01)
    banding = j.banding_rencana(pp.Praregistrasi(uji_direncanakan=["regresi linear"]))
    assert banding.sesuai_rencana == ["Regresi linear berganda"]


def test_kalimat_banding_siap_masuk_keterbatasan(jejak_terisi):
    kalimat = jejak_terisi.banding_rencana(
        pp.Praregistrasi(uji_direncanakan=["Uji 1"])
    ).kalimat()
    assert kalimat.endswith(".")
    assert "di luar rencana" in kalimat


# --------------------------------------------------------------------------- #
# Tampilan dan penyimpanan
# --------------------------------------------------------------------------- #


def test_ringkas_memuat_seluruh_langkah(jejak_terisi):
    tabel = jejak_terisi.ringkas()
    assert len(tabel) == 6
    assert list(tabel.columns) == ["Waktu", "Jenis", "Halaman", "Langkah", "p", "Keterangan"]


def test_perjalanan_pulang_pergi_lewat_dict(jejak_terisi):
    jejak_terisi.catat_perubahan("Mengisi nilai hilang dengan median")
    pulang = jk.Jejak.dari_dict(jejak_terisi.ke_dict())
    assert len(pulang.langkah) == len(jejak_terisi.langkah)
    assert pulang.nilai_p() == jejak_terisi.nilai_p()
    assert pulang.hitungan() == jejak_terisi.hitungan()


def test_langkah_cacat_dilewati_tanpa_menggagalkan_pembukaan():
    pulang = jk.Jejak.dari_dict(
        {
            "langkah": [
                {"jenis": "uji", "ringkas": "Baik", "p": 0.01},
                {"jenis": "entah", "ringkas": "Rusak"},
                "bukan dict",
                {"jenis": "uji", "ringkas": "p tak terbaca", "p": "banyak"},
            ]
        }
    )
    assert [l.ringkas for l in pulang.langkah] == ["Baik"]


def test_dari_dict_kosong_menghasilkan_jejak_kosong():
    assert jk.Jejak.dari_dict(None).kosong()
    assert jk.Jejak.dari_dict({}).kosong()
