"""Uji Pemilih Metode Terpandu.

Yang diuji adalah **keputusannya**, bukan sekadar tidak adanya galat. Data dibuat
sengaja untuk tiap cabang: dua kelompok normal harus menghasilkan uji-t, dua kelompok
menceng harus menghasilkan Mann-Whitney, ragam yang tidak seragam harus menghasilkan
Welch. Pemandu yang berjalan mulus namun menyarankan ANOVA di atas data yang menceng
lebih berbahaya daripada tidak ada pemandu, karena penggunanya justru tidak akan
memeriksa ulang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nalardata import kamus as km
from nalardata import pemandu as pmd

N = 120


@pytest.fixture(scope="module")
def acak() -> np.random.Generator:
    return np.random.default_rng(20260906)


def _kamus(df: pd.DataFrame, **skala) -> km.Kamus:
    k = km.Kamus.dari_data(df)
    for kolom, nilai in skala.items():
        k.tetapkan(kolom, skala=nilai)
    return k


def _sarankan(df: pd.DataFrame, skala: dict | None = None, **kw) -> pmd.Rekomendasi:
    return pmd.sarankan(df, _kamus(df, **(skala or {})), **kw)


def _dua_kelompok(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"y": np.r_[a, b], "g": ["A"] * len(a) + ["B"] * len(b)})


def _tiga_kelompok(a, b, c) -> pd.DataFrame:
    return pd.DataFrame(
        {"y": np.r_[a, b, c], "g": ["A"] * len(a) + ["B"] * len(b) + ["C"] * len(c)}
    )


# --------------------------------------------------------------------------- #
# Membandingkan dua kelompok bebas
# --------------------------------------------------------------------------- #


def test_dua_kelompok_normal_ragam_sama_menghasilkan_uji_t(acak):
    df = _dua_kelompok(acak.normal(50, 8, N), acak.normal(54, 8, N))
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "Uji-t sampel bebas"


def test_dua_kelompok_ragam_berbeda_menghasilkan_welch(acak):
    df = _dua_kelompok(acak.normal(50, 3, N), acak.normal(54, 18, N))
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "Uji-t Welch"
    ditolak = [a for a in hasil.alternatif if "ragam sama" in a.metode]
    assert ditolak and "Levene" in ditolak[0].ditolak_karena


def test_dua_kelompok_menceng_menghasilkan_mann_whitney(acak):
    df = _dua_kelompok(acak.lognormal(2, 1, N), acak.lognormal(2.4, 1, N))
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "Mann-Whitney U"
    assert "Shapiro" in hasil.alternatif[0].ditolak_karena


# --------------------------------------------------------------------------- #
# Membandingkan lebih dari dua kelompok
# --------------------------------------------------------------------------- #


def test_tiga_kelompok_normal_menghasilkan_anova(acak):
    df = _tiga_kelompok(
        acak.normal(50, 8, N), acak.normal(54, 8, N), acak.normal(48, 8, N)
    )
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "One-Way ANOVA"
    assert "Tukey" in hasil.utama.lanjutan


def test_tiga_kelompok_menceng_menghasilkan_kruskal_wallis(acak):
    df = _tiga_kelompok(
        acak.lognormal(2, 1, N), acak.lognormal(2.3, 1, N), acak.lognormal(1.9, 1, N)
    )
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "Kruskal-Wallis"
    assert "Dunn" in hasil.utama.lanjutan


def test_tiga_kelompok_ragam_berbeda_menghasilkan_welch_anova(acak):
    df = _tiga_kelompok(
        acak.normal(50, 2, N), acak.normal(54, 14, N), acak.normal(48, 25, N)
    )
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "Welch ANOVA"
    assert "Games-Howell" in hasil.utama.lanjutan


def test_alasan_menolak_anova_menyebut_angka_ujinya(acak):
    """'Tidak normal' saja tidak mengajari; yang mengajari adalah angkanya."""
    df = _tiga_kelompok(
        acak.lognormal(2, 1, N), acak.lognormal(2.3, 1, N), acak.lognormal(1.9, 1, N)
    )
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    ditolak = [a for a in hasil.alternatif if a.metode == "One-Way ANOVA"][0]
    assert "3 dari 3 kelompok" in ditolak.ditolak_karena


# --------------------------------------------------------------------------- #
# Skala dibaca dari kamus, bukan dari dtype
# --------------------------------------------------------------------------- #


def test_skor_likert_normal_tetap_menghasilkan_uji_non_parametrik(acak):
    """Inti kamus variabel: angka 1-5 bertipe int tetapi ordinal."""
    df = _dua_kelompok(acak.integers(1, 6, N), acak.integers(1, 6, N))
    hasil = _sarankan(
        df, skala={"y": km.ORDINAL}, tujuan="membandingkan", outcome="y", kelompok="g"
    )
    assert hasil.utama.metode == "Mann-Whitney U"
    assert "ordinal" in hasil.utama.alasan


def test_kolom_yang_sama_sebagai_rasio_menghasilkan_uji_parametrik(acak):
    """Kolom yang sama, keputusan berbeda — perbedaannya hanya pada kamus."""
    nilai = np.r_[acak.normal(50, 8, N), acak.normal(52, 8, N)]
    df = pd.DataFrame({"y": nilai, "g": ["A"] * N + ["B"] * N})
    assert _sarankan(
        df, skala={"y": km.RASIO}, tujuan="membandingkan", outcome="y", kelompok="g"
    ).utama.metode == "Uji-t sampel bebas"
    assert _sarankan(
        df, skala={"y": km.ORDINAL}, tujuan="membandingkan", outcome="y", kelompok="g"
    ).utama.metode == "Mann-Whitney U"


def test_outcome_kategorik_menghasilkan_chi_square(acak):
    df = pd.DataFrame(
        {"y": acak.choice(["ya", "tidak"], 2 * N), "g": acak.choice(list("ABC"), 2 * N)}
    )
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "Chi-square"
    assert "Cramér" in hasil.utama.lanjutan
    assert "rata-ratanya tidak bermakna" in hasil.alternatif[0].ditolak_karena


def test_sel_berfrekuensi_kecil_menghasilkan_fisher():
    df = pd.DataFrame(
        {"y": ["ya"] * 8 + ["tidak"] * 2, "g": ["A"] * 5 + ["B"] * 5}
    )
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.utama.metode == "Uji eksak Fisher"
    assert "frekuensi harapan" in hasil.alternatif[0].ditolak_karena.lower()


# --------------------------------------------------------------------------- #
# Berpasangan
# --------------------------------------------------------------------------- #


def test_dua_pengukuran_berpasangan_normal_menghasilkan_uji_t_berpasangan(acak):
    df = pd.DataFrame({"sebelum": acak.normal(50, 8, N), "sesudah": acak.normal(54, 8, N)})
    hasil = _sarankan(
        df, tujuan="membandingkan", outcome="sebelum", prediktor=["sesudah"], berpasangan=True
    )
    assert hasil.utama.metode == "Uji-t berpasangan"


def test_dua_pengukuran_berpasangan_menceng_menghasilkan_wilcoxon(acak):
    df = pd.DataFrame(
        {"sebelum": acak.lognormal(2, 1, N), "sesudah": acak.lognormal(2.3, 1, N)}
    )
    hasil = _sarankan(
        df, tujuan="membandingkan", outcome="sebelum", prediktor=["sesudah"], berpasangan=True
    )
    assert hasil.utama.metode == "Wilcoxon signed-rank"


def test_tiga_pengukuran_berpasangan_menghasilkan_friedman(acak):
    df = pd.DataFrame({f"waktu{i}": acak.normal(50, 8, N) for i in range(3)})
    hasil = _sarankan(
        df,
        tujuan="membandingkan",
        outcome="waktu0",
        prediktor=["waktu1", "waktu2"],
        berpasangan=True,
    )
    assert hasil.utama.metode == "Friedman"
    assert "Kendall" in hasil.utama.lanjutan


def test_berpasangan_menghasilkan_uji_berbeda_dari_bebas(acak):
    """Satu-satunya hal yang ditanyakan dan tak dapat dihitung memang harus berpengaruh."""
    nilai = pd.DataFrame({"a": acak.normal(50, 8, N), "b": acak.normal(54, 8, N)})
    berpasangan = _sarankan(
        nilai, tujuan="membandingkan", outcome="a", prediktor=["b"], berpasangan=True
    )
    panjang = _dua_kelompok(nilai["a"].to_numpy(), nilai["b"].to_numpy())
    bebas = _sarankan(panjang, tujuan="membandingkan", outcome="y", kelompok="g")
    assert berpasangan.utama.metode != bebas.utama.metode


# --------------------------------------------------------------------------- #
# Hubungan
# --------------------------------------------------------------------------- #


def test_dua_variabel_normal_menghasilkan_pearson(acak):
    x = acak.normal(0, 1, 300)
    df = pd.DataFrame({"x": x * 10 + 50, "y": x * 6 + acak.normal(0, 4, 300) + 30})
    hasil = _sarankan(df, tujuan="menghubungkan", outcome="x", prediktor=["y"])
    assert hasil.utama.metode == "Korelasi Pearson"


def test_variabel_menceng_menghasilkan_spearman(acak):
    df = pd.DataFrame({"x": acak.lognormal(2, 1, 300), "y": acak.lognormal(2, 1, 300)})
    hasil = _sarankan(df, tujuan="menghubungkan", outcome="x", prediktor=["y"])
    assert hasil.utama.metode == "Korelasi Spearman"
    assert any(a.metode == "Korelasi Kendall tau" for a in hasil.alternatif)


def test_variabel_ordinal_menghasilkan_spearman(acak):
    df = pd.DataFrame({"x": acak.normal(50, 8, 300), "y": acak.integers(1, 6, 300)})
    hasil = _sarankan(
        df, skala={"y": km.ORDINAL}, tujuan="menghubungkan", outcome="x", prediktor=["y"]
    )
    assert hasil.utama.metode == "Korelasi Spearman"
    assert "ordinal" in hasil.utama.alasan


def test_dua_variabel_kategorik_menghasilkan_chi_square(acak):
    df = pd.DataFrame(
        {"x": acak.choice(list("PQ"), 300), "y": acak.choice(list("XYZ"), 300)}
    )
    hasil = _sarankan(df, tujuan="menghubungkan", outcome="x", prediktor=["y"])
    assert hasil.utama.metode in {"Chi-square", "Uji eksak Fisher"}
    assert "Pearson" in hasil.alternatif[0].metode


# --------------------------------------------------------------------------- #
# Regresi dan klasifikasi
# --------------------------------------------------------------------------- #


def test_outcome_angka_menghasilkan_regresi_linear(acak):
    x1, x2 = acak.normal(0, 1, 300), acak.normal(0, 1, 300)
    df = pd.DataFrame({"y": 3 * x1 - 2 * x2 + acak.normal(0, 1, 300), "x1": x1, "x2": x2})
    hasil = _sarankan(df, tujuan="memperkirakan_nilai", outcome="y", prediktor=["x1", "x2"])
    assert hasil.utama.metode == "Regresi linear berganda"
    assert "asumsi klasik" in hasil.utama.lanjutan


def test_multikolinearitas_dilaporkan_sebagai_syarat_dilanggar(acak):
    x1 = acak.normal(0, 1, 300)
    df = pd.DataFrame(
        {"y": x1 + acak.normal(0, 1, 300), "x1": x1, "x2": x1 + acak.normal(0, 0.01, 300)}
    )
    hasil = _sarankan(df, tujuan="memperkirakan_nilai", outcome="y", prediktor=["x1", "x2"])
    kolinear = [s for s in hasil.utama.syarat if s.nama == "Multikolinearitas"][0]
    assert kolinear.dilanggar
    assert any("Kurangi prediktor" in a.metode for a in hasil.alternatif)


def test_outcome_dua_kategori_menghasilkan_regresi_logistik(acak):
    x = acak.normal(0, 1, 300)
    df = pd.DataFrame({"y": (x > 0).astype(int), "x": x, "z": acak.normal(0, 1, 300)})
    hasil = _sarankan(
        df, tujuan="memperkirakan_kategori", outcome="y", prediktor=["x", "z"]
    )
    assert hasil.utama.metode == "Regresi logistik biner"
    assert "odds ratio" in hasil.utama.lanjutan


def test_outcome_banyak_kategori_menghasilkan_diskriminan(acak):
    df = pd.DataFrame(
        {
            "y": acak.choice(list("ABCD"), 300),
            "x": acak.normal(0, 1, 300),
            "z": acak.normal(0, 1, 300),
        }
    )
    hasil = _sarankan(
        df, tujuan="memperkirakan_kategori", outcome="y", prediktor=["x", "z"]
    )
    assert hasil.utama.metode == "Analisis diskriminan"


def test_kelas_sangat_timpang_ditandai(acak):
    y = np.r_[np.ones(290), np.zeros(10)]
    df = pd.DataFrame({"y": y, "x": acak.normal(0, 1, 300), "z": acak.normal(0, 1, 300)})
    hasil = _sarankan(
        df, tujuan="memperkirakan_kategori", outcome="y", prediktor=["x", "z"]
    )
    seimbang = [s for s in hasil.utama.syarat if s.nama == "Keseimbangan kelas"][0]
    assert seimbang.dilanggar
    assert "kelas minoritas" in seimbang.rincian


# --------------------------------------------------------------------------- #
# Tujuan lain
# --------------------------------------------------------------------------- #


def test_meringkas_menghasilkan_efa_dengan_pca_sebagai_alternatif(acak):
    dasar = acak.normal(0, 1, 300)
    df = pd.DataFrame({f"b{i}": dasar + acak.normal(0, 0.6, 300) for i in range(6)})
    hasil = _sarankan(df, tujuan="meringkas", prediktor=list(df.columns))
    assert "EFA" in hasil.utama.metode
    assert any("PCA" in a.metode for a in hasil.alternatif)
    assert any(s.nama == "KMO" for s in hasil.utama.syarat)


def test_mengelompokkan_menghasilkan_klaster(acak):
    df = pd.DataFrame({"x": acak.normal(0, 1, 300), "y": acak.normal(0, 1, 300)})
    hasil = _sarankan(df, tujuan="mengelompokkan", prediktor=["x", "y"])
    assert hasil.utama.metode == "Analisis klaster"
    assert "jawaban benar tunggal" in hasil.utama.lanjutan


def test_menguji_model_menghasilkan_sem_dengan_saran_estimator(acak):
    dasar = acak.normal(0, 1, 300)
    df = pd.DataFrame({f"b{i}": dasar + acak.normal(0, 0.6, 300) for i in range(6)})
    hasil = _sarankan(df, tujuan="menguji_model", prediktor=list(df.columns))
    assert "SEM" in hasil.utama.metode
    assert hasil.utama.peringatan


def test_sampel_kecil_untuk_sem_ditandai_dilanggar(acak):
    dasar = acak.normal(0, 1, 80)
    df = pd.DataFrame({f"b{i}": dasar + acak.normal(0, 0.6, 80) for i in range(6)})
    hasil = _sarankan(df, tujuan="menguji_model", prediktor=list(df.columns))
    ukuran = [s for s in hasil.utama.syarat if s.nama == "Ukuran sampel"][0]
    assert ukuran.dilanggar
    assert "200" in ukuran.rincian


def test_mutu_instrumen_menghasilkan_reliabilitas(acak):
    dasar = acak.normal(0, 1, 300)
    df = pd.DataFrame({f"b{i}": dasar + acak.normal(0, 0.6, 300) for i in range(5)})
    hasil = _sarankan(df, tujuan="mutu_instrumen", prediktor=list(df.columns))
    assert "reliabilitas" in hasil.utama.metode.lower()
    assert "omega" in hasil.utama.lanjutan


# --------------------------------------------------------------------------- #
# Invarian
# --------------------------------------------------------------------------- #


@pytest.fixture
def semua_rekomendasi(acak) -> list[pmd.Rekomendasi]:
    hasil = []
    df2 = _dua_kelompok(acak.normal(50, 8, N), acak.normal(54, 8, N))
    hasil.append(_sarankan(df2, tujuan="membandingkan", outcome="y", kelompok="g"))
    df3 = _tiga_kelompok(
        acak.lognormal(2, 1, N), acak.lognormal(2.3, 1, N), acak.lognormal(1.9, 1, N)
    )
    hasil.append(_sarankan(df3, tujuan="membandingkan", outcome="y", kelompok="g"))

    x = acak.normal(0, 1, 300)
    reg = pd.DataFrame({"y": x + acak.normal(0, 1, 300), "x": x, "z": acak.normal(0, 1, 300)})
    hasil.append(_sarankan(reg, tujuan="memperkirakan_nilai", outcome="y", prediktor=["x", "z"]))
    hasil.append(_sarankan(reg, tujuan="menghubungkan", outcome="y", prediktor=["x"]))

    log = pd.DataFrame({"y": (x > 0).astype(int), "x": x, "z": acak.normal(0, 1, 300)})
    hasil.append(_sarankan(log, tujuan="memperkirakan_kategori", outcome="y", prediktor=["x", "z"]))

    dasar = acak.normal(0, 1, 300)
    butir = pd.DataFrame({f"b{i}": dasar + acak.normal(0, 0.6, 300) for i in range(6)})
    for tujuan in ("meringkas", "menguji_model", "mutu_instrumen"):
        hasil.append(_sarankan(butir, tujuan=tujuan, prediktor=list(butir.columns)))
    hasil.append(_sarankan(butir, tujuan="mengelompokkan", prediktor=list(butir.columns)))
    return hasil


def test_setiap_saran_utama_menjelaskan_alasannya(semua_rekomendasi):
    for rekomendasi in semua_rekomendasi:
        assert rekomendasi.utama is not None
        assert len(rekomendasi.utama.alasan) > 30


def test_setiap_saran_utama_menunjuk_halaman_metodenya(semua_rekomendasi):
    for rekomendasi in semua_rekomendasi:
        assert rekomendasi.utama.halaman.strip()


def test_setiap_alternatif_menjelaskan_mengapa_tidak_dipilih(semua_rekomendasi):
    """Bagian inilah yang mengajari; alternatif tanpa alasan hanya daftar nama."""
    for rekomendasi in semua_rekomendasi:
        for alternatif in rekomendasi.alternatif:
            assert len(alternatif.ditolak_karena) > 30, alternatif.metode


def test_setiap_rekomendasi_menawarkan_sekurangnya_satu_alternatif(semua_rekomendasi):
    for rekomendasi in semua_rekomendasi:
        assert rekomendasi.alternatif


def test_ringkas_menyandingkan_utama_dan_alternatif(semua_rekomendasi):
    tabel = semua_rekomendasi[0].ringkas()
    assert (tabel["Status"] == "Disarankan").sum() == 1
    assert (tabel["Status"] == "Tidak dipilih").sum() >= 1
    assert (tabel["Alasan"].str.strip() != "").all()


# --------------------------------------------------------------------------- #
# Masukan yang belum lengkap
# --------------------------------------------------------------------------- #


def test_tanpa_outcome_pemandu_bertanya_bukan_menebak(acak):
    df = _dua_kelompok(acak.normal(50, 8, N), acak.normal(54, 8, N))
    hasil = _sarankan(df, tujuan="membandingkan", kelompok="g")
    assert not hasil.berhasil
    assert hasil.belum_terjawab


def test_tanpa_kelompok_pemandu_bertanya(acak):
    df = _dua_kelompok(acak.normal(50, 8, N), acak.normal(54, 8, N))
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y")
    assert not hasil.berhasil
    assert any("kelompok" in p.lower() for p in hasil.belum_terjawab)


def test_kelompok_tunggal_dilaporkan_sebagai_catatan():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "g": ["A"] * 4})
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert not hasil.berhasil
    assert any("1 kelompok" in c for c in hasil.catatan)


def test_kolom_yang_tidak_ada_diabaikan_bukan_menggagalkan(acak):
    df = _dua_kelompok(acak.normal(50, 8, N), acak.normal(54, 8, N))
    hasil = _sarankan(df, tujuan="membandingkan", outcome="tidak_ada", kelompok="g")
    assert not hasil.berhasil
    assert hasil.belum_terjawab


def test_tujuan_asing_ditolak(acak):
    df = _dua_kelompok(acak.normal(50, 8, N), acak.normal(54, 8, N))
    with pytest.raises(ValueError, match="tidak dikenal"):
        _sarankan(df, tujuan="mencari kebenaran")


def test_setiap_tujuan_punya_pertanyaan_sehari_hari():
    """Pengguna memilih tujuan lewat pertanyaannya, bukan lewat istilah statistik."""
    assert set(pmd.TUJUAN) == set(pmd.PERTANYAAN_TUJUAN)
    for pertanyaan in pmd.PERTANYAAN_TUJUAN.values():
        assert pertanyaan.endswith("?")


# --------------------------------------------------------------------------- #
# Pemeriksaan data
# --------------------------------------------------------------------------- #


def test_normalitas_diuji_per_kelompok_bukan_gabungan():
    """Dua kelompok normal berjarak jauh tampak dwipuncak bila digabung.

    Normalitas yang dituntut uji-t adalah normalitas di dalam kelompok. Menguji
    data gabungan akan menolak normalitas yang sebenarnya terpenuhi, lalu
    mengantar pengguna ke uji non-parametrik tanpa alasan.
    """
    # Dibangun dari kuantil sebaran normal, bukan dari undian acak: Shapiro-Wilk
    # menolak sekitar lima persen sampel yang sungguh normal, dan uji ini menyoal
    # logika penggabungan, bukan keberuntungan undian.
    from scipy import stats as _stats

    baku = _stats.norm.ppf(np.linspace(0.01, 0.99, 60))
    df = _dua_kelompok(10 + baku, 60 + baku)
    assert pmd.periksa_normalitas(df, "y", "g").terpenuhi
    assert pmd.periksa_normalitas(df, "y", None).dilanggar


def test_kelompok_terlalu_kecil_ditandai():
    df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6], "g": ["A"] * 5 + ["B"]})
    syarat = pmd.periksa_ukuran_kelompok(df, "g")
    assert syarat.dilanggar
    assert "1 pengamatan" in syarat.rincian


def test_kelompok_kecil_tetapi_memadai_disebut_apa_adanya(acak):
    df = _dua_kelompok(acak.normal(50, 8, 20), acak.normal(54, 8, 20))
    syarat = pmd.periksa_ukuran_kelompok(df, "g")
    assert syarat.terpenuhi
    assert "non-parametrik lebih aman" in syarat.rincian


# --------------------------------------------------------------------------- #
# Ketersediaan metode
# --------------------------------------------------------------------------- #


def test_setiap_metode_yang_disarankan_benar_benar_dapat_dijalankan(semua_rekomendasi):
    """Pagar struktural, bukan dokumentasi.

    Pemandu pernah menyarankan uji-t sampel bebas, uji-t Welch, One-Way ANOVA,
    dan Welch ANOVA — empat uji yang sama sekali belum ada di aplikasi. Pengguna
    yang menuruti sarannya tiba di halaman yang tidak dapat menjalankannya, dan
    tidak ada satu pun uji yang menangkapnya sampai validasi lintas software
    dijalankan.
    """
    for rekomendasi in semua_rekomendasi:
        assert rekomendasi.utama.tersedia, rekomendasi.utama.metode


def test_alternatif_yang_belum_ada_ditandai_bukan_disembunyikan(acak):
    """Metode terbaik yang belum tersedia tetap layak disebut, asalkan ditandai."""
    df = pd.DataFrame({f"waktu{i}": acak.normal(50, 8, N) for i in range(3)})
    hasil = _sarankan(
        df,
        tujuan="membandingkan",
        outcome="waktu0",
        prediktor=["waktu1", "waktu2"],
        berpasangan=True,
    )
    belum = [a for a in hasil.alternatif if not a.tersedia]
    assert belum, "ANOVA ukur ulang belum tersedia dan harus ditandai"
    assert "belum tersedia" in belum[0].ditolak_karena.lower()


def test_halaman_pada_saran_cocok_dengan_daftar_metode(semua_rekomendasi):
    """Halaman yang keliru mengantar pengguna ke tempat yang salah."""
    for rekomendasi in semua_rekomendasi:
        utama = rekomendasi.utama
        assert utama.halaman == pmd.METODE_TERSEDIA[utama.metode], utama.metode


def test_rekomendasi_membawa_penetapan_variabelnya(acak):
    """Pengguna tidak boleh diminta memilih ulang variabel yang baru saja ia sebut."""
    df = _dua_kelompok(acak.normal(50, 8, N), acak.normal(54, 8, N))
    hasil = _sarankan(df, tujuan="membandingkan", outcome="y", kelompok="g")
    assert hasil.konfig["outcome"] == "y"
    assert hasil.konfig["kelompok"] == "g"
    assert hasil.utama.konfig["metode"] == hasil.utama.metode
