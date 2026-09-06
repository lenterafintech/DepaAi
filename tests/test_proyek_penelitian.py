"""Uji Ruang Proyek: rancangan penelitian dan dua kunci pengaman yang lahir darinya.

Yang diuji adalah **keputusannya** — desain mana yang membuka bahasa sebab-akibat,
teknik sampling mana yang membolehkan generalisasi — karena di situlah letak
kekeliruan yang paling sering lolos dari pemeriksaan.
"""

from __future__ import annotations

import pytest

from nalardata import proyek_penelitian as pp


# --------------------------------------------------------------------------- #
# Kunci kausalitas
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "desain",
    ["potong_lintang", "kuasi_eksperimen", "longitudinal", "panel", "deret_waktu", "studi_kasus"],
)
def test_desain_tanpa_penugasan_acak_menutup_bahasa_sebab(desain):
    """Hanya eksperimen berpenugasan acak yang membuka kesimpulan sebab-akibat."""
    assert not pp.ProyekPenelitian(desain=desain).boleh_sebab


def test_eksperimen_dengan_penugasan_acak_membuka_bahasa_sebab():
    assert pp.ProyekPenelitian(desain="eksperimen", penugasan_acak=True).boleh_sebab


def test_eksperimen_tanpa_penugasan_acak_tetap_tertutup():
    """Menyebut penelitian 'eksperimen' saja tidak cukup; penugasannya harus acak."""
    proyek = pp.ProyekPenelitian(desain="eksperimen", penugasan_acak=False)
    assert not proyek.boleh_sebab
    assert "tidak ditempatkan secara acak" in proyek.alasan_sebab


def test_setiap_desain_menjelaskan_alasannya():
    for kode in pp.DESAIN:
        proyek = pp.ProyekPenelitian(desain=kode)
        assert proyek.alasan_sebab.strip()
        assert len(proyek.alasan_sebab) > 40


def test_longitudinal_mengakui_lebih_kuat_daripada_potong_lintang():
    """Keduanya sama-sama tertutup, tetapi alasannya tidak boleh disamaratakan."""
    alasan = pp.ProyekPenelitian(desain="longitudinal").alasan_sebab
    assert "lebih meyakinkan" in alasan


# --------------------------------------------------------------------------- #
# Kunci generalisasi
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "teknik", ["acak_sederhana", "sistematis", "berlapis", "gugus", "jenuh"]
)
def test_sampling_probabilitas_membolehkan_generalisasi(teknik):
    assert pp.ProyekPenelitian(teknik_sampling=teknik).boleh_generalisasi


@pytest.mark.parametrize("teknik", ["purposif", "insidental", "kuota", "bola_salju"])
def test_sampling_non_probabilitas_menutup_generalisasi(teknik):
    proyek = pp.ProyekPenelitian(teknik_sampling=teknik)
    assert not proyek.boleh_generalisasi
    assert "bukan bagi seluruh populasi" in proyek.alasan_generalisasi


def test_besar_sampel_tidak_menggantikan_cara_pengambilannya():
    """Kekeliruan 'n besar berarti representatif' dijawab langsung di sini."""
    proyek = pp.ProyekPenelitian(teknik_sampling="insidental", target_sampel=100_000)
    assert not proyek.boleh_generalisasi
    assert "berapa pun banyaknya responden" in proyek.alasan_generalisasi


def test_sampel_jenuh_mengingatkan_uji_signifikansi_kurang_bermakna():
    alasan = pp.SAMPLING["jenuh"].alasan
    assert "sensus" in pp.SAMPLING["jenuh"].nama.lower()
    assert "kurang bermakna" in alasan


def test_sampling_gugus_mengingatkan_penyesuaian_galat_baku():
    assert "galat bakunya perlu disesuaikan" in pp.SAMPLING["gugus"].alasan


# --------------------------------------------------------------------------- #
# Batas kesimpulan
# --------------------------------------------------------------------------- #


def test_batas_kesimpulan_menyebut_hubungan_bukan_sebab():
    batas = pp.ProyekPenelitian(desain="potong_lintang").batas_kesimpulan()
    assert any("bukan sebab-akibat" in b for b in batas)


def test_batas_kesimpulan_pada_eksperimen_acak_menyatakan_sebab_didukung():
    batas = pp.ProyekPenelitian(
        desain="eksperimen", penugasan_acak=True, teknik_sampling="acak_sederhana"
    ).batas_kesimpulan()
    assert any("mendukung kesimpulan sebab-akibat" in b for b in batas)


def test_data_sekunder_menambah_satu_batas():
    primer = pp.ProyekPenelitian(sumber_data="primer").batas_kesimpulan()
    sekunder = pp.ProyekPenelitian(sumber_data="sekunder").batas_kesimpulan()
    assert len(sekunder) == len(primer) + 1
    assert any("pihak lain" in b for b in sekunder)


def test_batas_kesimpulan_selalu_ada_walau_proyek_belum_diisi():
    """Laporan tidak boleh terbit tanpa batas, sekalipun penggunanya melewati Tahap 0."""
    assert len(pp.ProyekPenelitian().batas_kesimpulan()) >= 2


# --------------------------------------------------------------------------- #
# Kelengkapan
# --------------------------------------------------------------------------- #


def test_proyek_kosong_menyebutkan_apa_yang_kurang():
    kurang = pp.ProyekPenelitian().kekurangan()
    assert "Judul penelitian" in kurang
    assert "Pertanyaan penelitian" in kurang
    assert "Populasi penelitian" in kurang
    assert "Unit analisis" in kurang


def test_proyek_terisi_dinyatakan_lengkap():
    proyek = pp.ProyekPenelitian(
        judul="Pengaruh kualitas layanan terhadap kepuasan",
        pertanyaan=["Apakah kualitas layanan berhubungan dengan kepuasan?"],
        populasi="Nasabah cabang Jakarta",
        unit_analisis="Nasabah perorangan",
    )
    assert proyek.lengkap()
    assert proyek.kekurangan() == []


def test_proyek_baru_dinyatakan_kosong():
    assert pp.ProyekPenelitian().kosong()
    assert not pp.ProyekPenelitian(judul="Ada isinya").kosong()


def test_ringkas_menyebut_kedua_kunci():
    ringkas = pp.ProyekPenelitian(
        desain="potong_lintang", teknik_sampling="purposif"
    ).ringkas().set_index("Keterangan")["Isi"]
    assert ringkas["Kesimpulan sebab-akibat"] == "Tidak boleh"
    assert ringkas["Generalisasi ke populasi"] == "Tidak boleh"


# --------------------------------------------------------------------------- #
# Nilai yang tidak sah
# --------------------------------------------------------------------------- #


def test_desain_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        pp.ProyekPenelitian(desain="korelasional")


def test_teknik_sampling_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        pp.ProyekPenelitian(teknik_sampling="random")


def test_sumber_data_asing_ditolak():
    with pytest.raises(ValueError, match="tidak dikenal"):
        pp.ProyekPenelitian(sumber_data="tersier")


# --------------------------------------------------------------------------- #
# Praregistrasi
# --------------------------------------------------------------------------- #


def test_praregistrasi_mencatat_waktu_sendiri():
    assert pp.Praregistrasi(hipotesis=["H1"]).waktu


def test_sidik_berubah_ketika_isinya_berubah():
    """Sidik dipakai agar rencana yang diubah diam-diam tetap terlihat."""
    satu = pp.Praregistrasi(hipotesis=["H1"], uji_direncanakan=["Regresi"])
    dua = pp.Praregistrasi(hipotesis=["H1", "H2"], uji_direncanakan=["Regresi"])
    assert satu.sidik != dua.sidik


def test_sidik_tidak_berubah_karena_waktu_pencatatan():
    satu = pp.Praregistrasi(hipotesis=["H1"], waktu="01-01-2026 08:00")
    dua = pp.Praregistrasi(hipotesis=["H1"], waktu="02-02-2026 09:00")
    assert satu.sidik == dua.sidik


def test_praregistrasi_tanpa_isi_dinyatakan_kosong():
    assert pp.Praregistrasi().kosong()
    assert not pp.Praregistrasi(hipotesis=["H1"]).kosong()


# --------------------------------------------------------------------------- #
# Penyimpanan
# --------------------------------------------------------------------------- #


def test_perjalanan_pulang_pergi_lewat_dict():
    asal = pp.ProyekPenelitian(
        judul="Kualitas layanan dan kepuasan",
        bidang="Manajemen pemasaran",
        pertanyaan=["Apakah ada hubungan?", "Seberapa kuat?"],
        hipotesis=["H1: ada hubungan positif"],
        populasi="Nasabah cabang Jakarta",
        ukuran_populasi=2500,
        unit_analisis="Nasabah perorangan",
        desain="eksperimen",
        penugasan_acak=True,
        teknik_sampling="berlapis",
        sumber_data="primer",
        target_sampel=200,
        praregistrasi=pp.Praregistrasi(
            hipotesis=["H1"], uji_direncanakan=["Regresi linear berganda"]
        ),
    )
    pulang = pp.ProyekPenelitian.dari_dict(asal.ke_dict())

    assert pulang.judul == asal.judul
    assert pulang.pertanyaan == asal.pertanyaan
    assert pulang.ukuran_populasi == 2500
    assert pulang.desain == "eksperimen"
    assert pulang.penugasan_acak
    assert pulang.boleh_sebab
    assert pulang.boleh_generalisasi
    assert pulang.praregistrasi is not None
    assert pulang.praregistrasi.sidik == asal.praregistrasi.sidik


def test_nilai_cacat_kembali_ke_bawaan_tanpa_menggagalkan():
    """Berkas proyek rusak sebagian tidak boleh menghanguskan seluruh rancangan."""
    pulang = pp.ProyekPenelitian.dari_dict(
        {
            "judul": "Ada",
            "desain": "entah",
            "teknik_sampling": "entah",
            "sumber_data": "entah",
            "pertanyaan": "bukan daftar",
            "ukuran_populasi": "banyak",
        }
    )
    assert pulang.judul == "Ada"
    assert pulang.desain == "potong_lintang"
    assert pulang.teknik_sampling == "purposif"
    assert pulang.sumber_data == "primer"
    assert pulang.pertanyaan == []
    assert pulang.ukuran_populasi is None


def test_dari_dict_kosong_menghasilkan_proyek_bawaan():
    assert pp.ProyekPenelitian.dari_dict(None).kosong()
    assert pp.ProyekPenelitian.dari_dict({}).kosong()


# --------------------------------------------------------------------------- #
# Kosakata sebab-akibat
# --------------------------------------------------------------------------- #


def test_kata_sebab_memuat_ungkapan_yang_lazim_dipakai_skripsi():
    for kata in ("berpengaruh terhadap", "memengaruhi", "menyebabkan", "meningkatkan"):
        assert kata in pp.KATA_SEBAB
