"""Penyusun kesimpulan naratif dari hasil analisis multivariat.

Alurnya dua tahap. ``jalankan_analisis`` menghitung seluruh hasil statistik satu
kali dan menyimpannya dalam :class:`Analisis`; ``susun_laporan`` kemudian
menerjemahkan hasil itu menjadi :class:`Laporan` — headline, lampu status,
peringkat pendorong, temuan, rekomendasi, keterbatasan, tabel bergaya APA, dan
paragraf siap salin.

Setiap temuan ditulis dalam tiga register pembaca:

- ``eksekutif``: pimpinan dan pembaca non-statistik — bahasa sehari-hari, fokus
  pada makna dan tindak lanjut, tanpa notasi statistik.
- ``akademik``: mahasiswa dan dosen — pelaporan bergaya jurnal lengkap dengan
  statistik uji, derajat bebas, p-value, ukuran efek, dan catatan asumsi.
- ``profesional``: analis dan praktisi — implikasi kerja, keterbatasan model,
  serta langkah teknis berikutnya.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from nalardata.formatting import bintang, num, num_auto, pct, pval  # noqa: F401
from nalardata import (
    assumptions,
    cca,
    clustering,
    correlation,
    descriptive,
    discriminant,
    manova,
    pca_analysis,
    preprocessing,
    regression,
)

AUDIENCES = ("eksekutif", "akademik", "profesional")
AUDIENCE_LABELS = {
    "eksekutif": "Umum & Eksekutif",
    "akademik": "Mahasiswa, Dosen & Pengajar",
    "profesional": "Profesional & Analis",
}
# Kedalaman dokumen, terpisah dari registernya. Ringkasan dan laporan lengkap adalah
# dua dokumen berbeda, bukan potongan panjang-pendek dari naskah yang sama.
KEDALAMAN = {
    False: "Ringkasan",
    True: "Laporan Lengkap",
}
STATUS_LABELS = {"baik": "Memadai", "perhatian": "Perlu dicermati", "kritis": "Bermasalah"}


def _efek_r2(r2: float) -> str:
    if r2 >= 0.5:
        return "besar"
    if r2 >= 0.25:
        return "sedang"
    if r2 >= 0.05:
        return "kecil"
    return "sangat kecil"


def _efek_eta(eta: float) -> str:
    if eta >= 0.14:
        return "besar"
    if eta >= 0.06:
        return "sedang"
    return "kecil"


def tabel_markdown(df: pd.DataFrame) -> str:
    """Ubah tabel menjadi Markdown tanpa bergantung pada paket tambahan."""
    kolom = [str(c) for c in df.columns]
    baris = ["| " + " | ".join(kolom) + " |", "|" + "|".join("---" for _ in kolom) + "|"]
    for nilai in df.itertuples(index=False):
        baris.append("| " + " | ".join(str(v).replace("|", "\\|") for v in nilai) + " |")
    return "\n".join(baris)


def _daftar(items: list[str], maksimal: int = 3) -> str:
    """Rangkai daftar menjadi frasa: "a", "a dan b", "a, b, dan c"."""
    semua = [str(i) for i in items]
    if not semua:
        return "tidak ada"
    dipakai = semua if len(semua) == maksimal + 1 else semua[:maksimal]
    sisa = len(semua) - len(dipakai)
    if sisa > 0:
        return ", ".join(dipakai) + f", dan {sisa} variabel lainnya"
    if len(dipakai) == 1:
        return dipakai[0]
    if len(dipakai) == 2:
        return f"{dipakai[0]} dan {dipakai[1]}"
    return ", ".join(dipakai[:-1]) + f", dan {dipakai[-1]}"


# --------------------------------------------------------------------------- #
# Struktur data
# --------------------------------------------------------------------------- #


@dataclass
class Konfigurasi:
    """Pilihan variabel yang menentukan analisis mana saja yang dijalankan."""

    variabel: list[str]
    nama_data: str = "data"
    target_numerik: str | None = None
    prediktor: list[str] = field(default_factory=list)
    target_biner: str | None = None
    prediktor_biner: list[str] = field(default_factory=list)
    kelompok: str | None = None
    gugus_x: list[str] = field(default_factory=list)
    gugus_y: list[str] = field(default_factory=list)


@dataclass
class Analisis:
    """Kumpulan hasil mentah seluruh metode yang berhasil dijalankan."""

    df: pd.DataFrame
    konfig: Konfigurasi
    normalitas: pd.DataFrame | None = None
    mardia: descriptive.MardiaResult | None = None
    pencilan: pd.DataFrame | None = None
    korelasi: correlation.CorrelationResult | None = None
    vif: pd.DataFrame | None = None
    kmo: assumptions.KMOResult | None = None
    bartlett: assumptions.BartlettResult | None = None
    pca: pca_analysis.PCAResult | None = None
    klaster: clustering.ClusterResult | None = None
    profil_klaster: pd.DataFrame | None = None
    regresi: regression.LinearRegressionResult | None = None
    logistik: regression.LogisticRegressionResult | None = None
    manova: manova.ManovaResult | None = None
    box_m: assumptions.BoxMResult | None = None
    diskriminan: discriminant.DiscriminantResult | None = None
    kanonik: cca.CCAResult | None = None
    gagal: dict[str, str] = field(default_factory=dict)


@dataclass
class Temuan:
    """Satu temuan analisis dalam tiga register pembaca."""

    judul: str
    metode: str
    ringkas: str
    eksekutif: str
    akademik: str
    profesional: str

    def teks(self, pembaca: str) -> str:
        if pembaca not in AUDIENCES:
            raise ValueError(f"Pembaca '{pembaca}' tidak dikenal. Pilih dari {AUDIENCES}.")
        return getattr(self, pembaca)


@dataclass
class Lampu:
    """Indikator status ringkas untuk dasbor kesimpulan."""

    label: str
    nilai: str
    status: str  # baik | perhatian | kritis
    catatan: str

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)


@dataclass
class Pendorong:
    """Satu variabel penjelas beserta kekuatan pengaruhnya."""

    nama: str
    kekuatan: float  # 0..1, sudah dinormalkan terhadap pendorong terkuat
    nilai: float
    satuan: str  # "beta" | "odds ratio" | "muatan"
    arah: str  # naik | turun
    p_value: float
    kinerja: float | None = None  # persentil rata-rata variabel (0..100)
    catatan: str = ""

    @property
    def signifikan(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)


@dataclass
class Rekomendasi:
    judul: str
    alasan: str
    prioritas: str  # tinggi | sedang | rendah


@dataclass
class Paragraf:
    """Paragraf siap salin untuk naskah akademik."""

    bagian: str
    teks: str


@dataclass
class Laporan:
    dataset: str
    n_baris: int
    n_kolom: int
    tanggal: str
    metode_terpakai: list[str] = field(default_factory=list)
    headline: str = ""
    subheadline: str = ""
    lampu: list[Lampu] = field(default_factory=list)
    pendorong: list[Pendorong] = field(default_factory=list)
    pendorong_sumber: str = ""
    temuan: list[Temuan] = field(default_factory=list)
    rekomendasi: list[Rekomendasi] = field(default_factory=list)
    keterbatasan: list[str] = field(default_factory=list)
    tabel: dict[str, tuple[str, pd.DataFrame, str]] = field(default_factory=dict)
    paragraf: list[Paragraf] = field(default_factory=list)
    rujukan: list[str] = field(default_factory=list)
    dilewati: list[str] = field(default_factory=list)
    # Konfigurasi disimpan bersama laporan agar sintaks Python/R yang dapat
    # dijalankan ulang bisa dibangkitkan tanpa membawa datanya.
    konfig: "Konfigurasi | None" = None

    def poin_kunci(self) -> list[str]:
        return [t.ringkas for t in self.temuan]

    def markdown(self, pembaca: str) -> str:
        judul = AUDIENCE_LABELS[pembaca]
        baris = [
            f"# Kesimpulan Analisis Multivariat — {judul}",
            "",
            f"Sumber data: **{self.dataset}** ({num(self.n_baris)} baris, "
            f"{self.n_kolom} kolom) · Tanggal analisis: {self.tanggal}",
            "",
            f"> **{self.headline}**",
            "",
            self.subheadline,
            "",
            "## Status pemeriksaan",
            "",
        ]
        baris += [
            f"- **{l.label}** — {l.nilai} ({l.status_label}). {l.catatan}" for l in self.lampu
        ]
        baris += ["", "## Poin kunci", ""]
        baris += [f"{i}. {poin}" for i, poin in enumerate(self.poin_kunci(), start=1)]

        if self.pendorong:
            baris += ["", f"## Peringkat pendorong ({self.pendorong_sumber})", ""]
            for i, p in enumerate(self.pendorong, start=1):
                baris.append(
                    f"{i}. **{p.nama}** — {p.satuan} {num(p.nilai, 3)} "
                    f"({p.arah}, {pval(p.p_value)}). {p.catatan}"
                )

        baris += ["", "## Temuan rinci", ""]
        for temuan in self.temuan:
            baris += [
                f"### {temuan.judul}",
                "",
                f"*Metode: {temuan.metode}*",
                "",
                temuan.teks(pembaca),
                "",
            ]

        if pembaca == "akademik" and self.tabel:
            baris += ["## Tabel hasil", ""]
            for nomor, (judul_tabel, tabel, catatan) in self.tabel.items():
                baris += [
                    f"**{nomor}.** {judul_tabel}",
                    "",
                    tabel_markdown(tabel),
                    "",
                    f"*Catatan.* {catatan}" if catatan else "",
                    "",
                ]

        if pembaca == "akademik" and self.paragraf:
            baris += ["## Kalimat siap salin", ""]
            for p in self.paragraf:
                baris += [f"**{p.bagian}**", "", p.teks, ""]

        if self.rekomendasi:
            baris += ["## Rekomendasi tindakan", ""]
            for i, r in enumerate(self.rekomendasi, start=1):
                baris.append(f"{i}. **{r.judul}** (prioritas {r.prioritas}) — {r.alasan}")
            baris.append("")

        if self.keterbatasan:
            baris += ["## Batas kesimpulan", ""]
            baris += [f"- {k}" for k in self.keterbatasan]
            baris.append("")

        if pembaca == "akademik" and self.rujukan:
            baris += ["## Rujukan ambang yang dipakai", ""]
            baris += [f"- {r}" for r in self.rujukan]
            baris.append("")

        if self.dilewati:
            baris += ["## Analisis yang tidak dapat dijalankan", ""]
            baris += [f"- {catatan}" for catatan in self.dilewati]
            baris.append("")

        baris += [
            "---",
            "",
            "Disusun otomatis oleh NalarData dari data yang Anda unggah. Kesimpulan "
            "statistik menunjukkan pola dalam data, bukan bukti sebab-akibat; keputusan "
            "akhir tetap memerlukan pertimbangan konteks dan keahlian bidang.",
        ]
        return "\n".join(b for b in baris if b is not None)

    def to_frame(self, pembaca: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Temuan": [t.judul for t in self.temuan],
                "Metode": [t.metode for t in self.temuan],
                "Kesimpulan": [t.teks(pembaca) for t in self.temuan],
            }
        )


# --------------------------------------------------------------------------- #
# Tahap 1 — menjalankan seluruh analisis
# --------------------------------------------------------------------------- #


def jalankan_analisis(df: pd.DataFrame, konfig: Konfigurasi) -> Analisis:
    """Hitung seluruh analisis yang mungkin; kegagalan dicatat, bukan dilempar."""
    hasil = Analisis(df=df, konfig=konfig)
    variabel = konfig.variabel
    subset = preprocessing.clean_subset(df, variabel)

    langkah = {
        "Uji normalitas": lambda: setattr(hasil, "normalitas", descriptive.normality_tests(subset)),
        "Uji Mardia": lambda: setattr(hasil, "mardia", descriptive.mardia_test(subset)),
        "Deteksi pencilan": lambda: setattr(
            hasil, "pencilan", descriptive.mahalanobis_outliers(subset)
        ),
        "Korelasi": lambda: setattr(hasil, "korelasi", correlation.correlation_matrix(subset)),
        "VIF": lambda: setattr(hasil, "vif", assumptions.vif(subset)),
        "KMO": lambda: setattr(hasil, "kmo", assumptions.kmo(subset)),
        "Bartlett": lambda: setattr(
            hasil, "bartlett", assumptions.bartlett_sphericity(subset)
        ),
        "PCA": lambda: setattr(hasil, "pca", pca_analysis.run_pca(subset, standardize=True)),
        "Analisis klaster": lambda: _jalankan_klaster(hasil, subset),
    }
    if konfig.target_numerik and konfig.prediktor:
        langkah["Regresi linear"] = lambda: setattr(
            hasil,
            "regresi",
            regression.linear_regression(df, konfig.target_numerik, konfig.prediktor),
        )
    if konfig.target_biner and (konfig.prediktor_biner or konfig.prediktor):
        dipakai = konfig.prediktor_biner or konfig.prediktor
        prediktor_biner = [p for p in dipakai if p != konfig.target_biner]
        langkah["Regresi logistik"] = lambda: setattr(
            hasil,
            "logistik",
            regression.logistic_regression(df, konfig.target_biner, prediktor_biner),
        )
    if konfig.kelompok:
        dv = [v for v in variabel if v != konfig.kelompok]
        langkah["MANOVA"] = lambda: setattr(hasil, "manova", manova.run_manova(df, dv, konfig.kelompok))
        langkah["Box's M"] = lambda: setattr(
            hasil, "box_m", assumptions.box_m(df[dv], df[konfig.kelompok])
        )
        langkah["Analisis diskriminan"] = lambda: setattr(
            hasil, "diskriminan", discriminant.run_discriminant(df, konfig.kelompok, dv)
        )
    if konfig.gugus_x and konfig.gugus_y:
        langkah["Korelasi kanonik"] = lambda: setattr(
            hasil, "kanonik", cca.run_cca(df, konfig.gugus_x, konfig.gugus_y)
        )

    for nama, jalankan in langkah.items():
        try:
            jalankan()
        except Exception as exc:  # noqa: BLE001 - alasan kegagalan dilaporkan ke pengguna
            hasil.gagal[nama] = str(exc)
    return hasil


def _jalankan_klaster(hasil: Analisis, subset: pd.DataFrame) -> None:
    data = preprocessing.scale(subset, "z-score")
    diagnostik = clustering.kmeans_diagnostics(data, 2, min(6, max(3, len(data) // 20)))
    k = int(diagnostik.loc[diagnostik["Silhouette"].idxmax(), "k"])
    hasil.klaster = clustering.run_kmeans(data, k)
    hasil.profil_klaster = clustering.profile_clusters(subset, hasil.klaster.labels)


# --------------------------------------------------------------------------- #
# Tahap 2a — temuan per metode
# --------------------------------------------------------------------------- #


def temuan_kualitas_data(a: Analisis) -> Temuan:
    df = a.df
    n, k = df.shape
    numerik = preprocessing.numeric_columns(df)
    kategorik = [c for c in df.columns if c not in numerik]
    missing_pct = float(df.isna().mean().mean() * 100)
    duplikat = int(df.duplicated().sum())
    kolom_bermasalah = [c for c in df.columns if df[c].isna().mean() > 0.2]
    rasio = n / max(len(a.konfig.variabel), 1)
    n_pencilan = (
        int((a.pencilan["Pencilan"] == "Ya").sum()) if a.pencilan is not None else 0
    )

    status = "siap dianalisis"
    if missing_pct > 10 or duplikat > n * 0.05:
        status = "perlu dibersihkan lebih dulu"

    eksekutif = (
        f"Data yang dianalisis berisi {num(n)} catatan dengan {k} keterangan pada tiap "
        f"catatan. Kelengkapan isian mencapai {pct(100 - missing_pct)} dan ditemukan "
        f"{num(duplikat)} catatan kembar, sehingga secara umum data {status}. "
        f"Ada {num(n_pencilan)} catatan yang polanya menyimpang jauh dari kebanyakan — "
        "jumlah sekecil ini wajar, tetapi perlu dicek apakah memang kasus istimewa atau "
        "salah input."
    )
    if kolom_bermasalah:
        eksekutif += (
            f" Perlu diperhatikan, kolom {_daftar(kolom_bermasalah)} banyak yang kosong "
            "sehingga kesimpulan dari kolom tersebut kurang bisa diandalkan."
        )

    akademik = (
        f"Analisis dilakukan terhadap N = {num(n)} observasi dengan {k} variabel "
        f"({len(numerik)} numerik, {len(kategorik)} kategorik). Proporsi data hilang "
        f"rata-rata {pct(missing_pct, 2)} dan terdapat {num(duplikat)} baris duplikat. "
        f"Rasio observasi terhadap variabel yang dianalisis sebesar {num(rasio, 1)} : 1, "
        + (
            "memenuhi aturan praktis minimal 10 observasi per variabel (Hair dkk., 2019)."
            if rasio >= 10
            else "di bawah aturan praktis minimal 10 observasi per variabel (Hair dkk., "
            "2019) sehingga stabilitas estimasi perlu dilaporkan sebagai keterbatasan."
        )
        + f" Deteksi pencilan multivariat dengan jarak Mahalanobis (α = 0,001) "
        f"mengidentifikasi {num(n_pencilan)} kasus ekstrem."
    )

    profesional = (
        f"Basis data: {num(n)} record, {k} field ({len(numerik)} numerik). "
        f"Kelengkapan {pct(100 - missing_pct)}, duplikat {num(duplikat)} record, "
        f"pencilan multivariat {num(n_pencilan)} record. "
    )
    if kolom_bermasalah:
        profesional += (
            f"Field {_daftar(kolom_bermasalah)} memiliki missing > 20% — putuskan antara "
            "imputasi berbasis model atau mengeluarkannya dari fitur. "
        )
    profesional += (
        "Sebelum hasil dipakai untuk keputusan operasional, pastikan definisi tiap field "
        "konsisten dengan sistem sumber, periode datanya seragam, dan proses pengambilan "
        "datanya dapat diulang."
    )

    return Temuan(
        judul="Kesiapan dan mutu data",
        metode="Profil data, jarak Mahalanobis",
        ringkas=(
            f"Data {num(n)} baris × {k} kolom, kelengkapan {pct(100 - missing_pct)}, "
            f"{num(n_pencilan)} pencilan multivariat; {status}."
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_normalitas(a: Analisis) -> Temuan:
    uji = a.normalitas
    mardia = a.mardia
    if uji is None or mardia is None:
        raise ValueError("Hasil uji normalitas tidak tersedia.")
    tidak_normal = uji.loc[uji["Kesimpulan"] == "Tidak normal", "Variabel"].tolist()
    jumlah = len(a.konfig.variabel)
    normal_multivariat = mardia.multivariate_normal

    eksekutif = (
        f"Dari {jumlah} ukuran yang diperiksa, {len(tidak_normal)} memiliki sebaran yang "
        "miring atau berekor panjang — artinya ada sekelompok kecil catatan dengan nilai "
        "jauh di atas atau di bawah kebanyakan. "
        + (
            "Pola seperti ini wajar pada data keuangan, tetapi membuat nilai rata-rata "
            "kurang mewakili keadaan sesungguhnya, sehingga sebaiknya juga melihat nilai "
            "tengah (median) saat menyusun laporan."
            if tidak_normal
            else "Sebarannya cukup rapi sehingga rata-rata sudah cukup mewakili keadaan data."
        )
    )

    akademik = (
        f"Uji normalitas univariat (Shapiro-Wilk) menunjukkan {len(tidak_normal)} dari "
        f"{jumlah} variabel menyimpang dari distribusi normal"
        + (f": {_daftar(tidak_normal, 6)}. " if tidak_normal else ". ")
        + f"Uji Mardia menghasilkan koefisien skewness multivariat {num(mardia.skewness, 3)} "
        f"(χ² = {num(mardia.skew_chi2)}; df = {mardia.skew_df}; {pval(mardia.skew_p)}) dan "
        f"kurtosis multivariat {num(mardia.kurtosis, 3)} (z = {num(mardia.kurt_z)}; "
        f"{pval(mardia.kurt_p)}), "
        + (
            "sehingga asumsi normalitas multivariat terpenuhi."
            if normal_multivariat
            else "sehingga asumsi normalitas multivariat tidak terpenuhi. Konsekuensinya, "
            "hasil uji parametrik perlu dilaporkan disertai catatan, atau data "
            "ditransformasi (logaritma/akar) dan analisis diulang."
        )
    )

    profesional = (
        f"{len(tidak_normal)} fitur menyimpang dari normal; normalitas multivariat "
        + ("terpenuhi" if normal_multivariat else "tidak terpenuhi")
        + ". Untuk pemodelan hal ini bukan penghalang — teknik berbasis jarak dan regresi "
        "tetap berjalan — namun uji signifikansi menjadi lebih sensitif terhadap pencilan "
        "dan interval kepercayaan cenderung terlalu sempit. Langkah praktis: transformasi "
        "log pada variabel bersatuan rupiah, winsorisasi pencilan ekstrem, atau gunakan "
        "estimator dan standard error yang robust."
    )

    return Temuan(
        judul="Sebaran data dan asumsi normalitas",
        metode="Shapiro-Wilk, Kolmogorov-Smirnov, uji Mardia",
        ringkas=(
            f"{len(tidak_normal)} dari {jumlah} variabel tidak normal; normalitas "
            f"multivariat {'terpenuhi' if normal_multivariat else 'tidak terpenuhi'}."
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_korelasi(a: Analisis) -> Temuan:
    hasil = a.korelasi
    if hasil is None:
        raise ValueError("Hasil korelasi tidak tersedia.")
    pasangan = hasil.significant_pairs()
    signifikan = pasangan[pasangan["Signifikan"] == "Ya"]
    teratas = signifikan.head(3)
    vif_max = float(a.vif["VIF"].max()) if a.vif is not None else np.nan
    multikol = (
        a.vif.loc[a.vif["VIF"] >= 10, "Variabel"].tolist() if a.vif is not None else []
    )

    def _frase(baris) -> str:
        arah = "searah" if baris["r"] > 0 else "berlawanan arah"
        return f"{baris['Variabel 1']}–{baris['Variabel 2']} ({arah}, r = {num(baris['r'], 3)})"

    daftar = "; ".join(_frase(b) for _, b in teratas.iterrows()) or "tidak ada"

    eksekutif = (
        f"Dari {len(pasangan)} pasangan ukuran yang dibandingkan, {len(signifikan)} "
        f"pasangan benar-benar bergerak bersamaan. Hubungan paling menonjol: {daftar}. "
    )
    if not teratas.empty:
        b = teratas.iloc[0]
        eksekutif += (
            f"Artinya, ketika {b['Variabel 1']} naik, {b['Variabel 2']} cenderung "
            + ("ikut naik. " if b["r"] > 0 else "justru turun. ")
        )
    eksekutif += (
        "Perlu diingat, bergerak bersamaan belum tentu saling menyebabkan — bisa saja "
        "keduanya dipengaruhi hal ketiga yang sama."
    )

    akademik = (
        f"Analisis korelasi Pearson (n = {num(hasil.n)}) menemukan {len(signifikan)} dari "
        f"{len(pasangan)} pasangan variabel berkorelasi signifikan pada α = 0,05. "
        f"Korelasi terkuat: {daftar}. "
    )
    if not teratas.empty:
        b = teratas.iloc[0]
        akademik += (
            f"Pasangan terkuat berbagi ragam sebesar r² = {num(b['r'] ** 2, 3)} "
            f"({pct(b['r'] ** 2 * 100)}). "
        )
    akademik += (
        f"Pemeriksaan multikolinearitas menghasilkan VIF maksimum {num(vif_max)}"
        + (
            f" pada {_daftar(multikol)} (VIF ≥ 10), sehingga variabel tersebut sebaiknya "
            "tidak dimasukkan bersamaan ke dalam satu model."
            if multikol
            else ", masih di bawah ambang 10, sehingga tidak terdapat masalah "
            "multikolinearitas (Hair dkk., 2019)."
        )
    )

    profesional = (
        f"{len(signifikan)} pasangan berkorelasi signifikan; VIF maksimum {num(vif_max)}. "
    )
    profesional += (
        f"Fitur {_daftar(multikol)} saling menggantikan informasi — pilih salah satu, "
        "gabungkan menjadi indeks, atau pakai skor komponen utama agar koefisien model "
        "tetap stabil dan dapat ditafsirkan. "
        if multikol
        else "Tidak ada fitur yang saling menggantikan secara berlebihan, sehingga "
        "seluruhnya dapat dipakai bersama dalam satu model. "
    )
    profesional += (
        "Peta korelasi ini juga menjadi dasar memilih kandidat prediktor sebelum "
        "pemodelan dan menandai fitur yang perlu dipantau bila sumber datanya berubah."
    )

    return Temuan(
        judul="Keterkaitan antar variabel",
        metode="Korelasi Pearson dengan uji signifikansi dan VIF",
        ringkas=(
            f"{len(signifikan)} pasangan variabel berkorelasi signifikan"
            + (f"; multikolinearitas pada {_daftar(multikol)}." if multikol else "; tanpa multikolinearitas.")
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_pca(a: Analisis) -> Temuan:
    hasil = a.pca
    if hasil is None:
        raise ValueError("Hasil PCA tidak tersedia.")
    jumlah = len(hasil.variables)
    k80 = hasil.components_needed(0.80)
    var_pc1 = hasil.explained_ratio[0] * 100
    dominan = hasil.loadings["PC1"].abs().sort_values(ascending=False).head(3).index.tolist()
    retensi = hasil.cumulative_ratio[k80 - 1] * 100

    eksekutif = (
        f"Sebanyak {jumlah} ukuran yang dianalisis sebenarnya dapat diringkas menjadi "
        f"{k80} dimensi utama tanpa kehilangan banyak informasi (masih menyimpan "
        f"{pct(retensi)} isi data). Dimensi pertama saja menjelaskan {pct(var_pc1)} "
        f"keragaman data dan paling kuat dibentuk oleh {_daftar(dominan)}. "
        "Artinya banyak ukuran yang menceritakan hal serupa, sehingga laporan rutin bisa "
        "disederhanakan menjadi beberapa indikator utama saja."
    )

    akademik = (
        f"Analisis komponen utama pada {jumlah} variabel "
        + (
            f"layak dilakukan (KMO = {num(a.kmo.overall, 3)}, kategori "
            f"{a.kmo.interpretation.lower()}"
            if a.kmo is not None
            else "dilakukan tanpa uji KMO"
        )
        + (
            f"; Bartlett's test of sphericity χ² = {num(a.bartlett.chi_square)}, "
            f"df = {a.bartlett.df}, {pval(a.bartlett.p_value)}). "
            if a.bartlett is not None
            else "). "
        )
        + f"Kriteria Kaiser mempertahankan {hasil.kaiser_components} komponen dengan "
        f"eigenvalue > 1, sedangkan kriteria varians kumulatif 80% memerlukan {k80} "
        f"komponen. Komponen pertama memiliki eigenvalue {num(hasil.eigenvalues[0], 3)} "
        f"dan menjelaskan {pct(var_pc1)} varians total, dimuat terutama oleh "
        f"{_daftar(dominan)}. Komunalitas berkisar {num(hasil.communalities.min(), 3)} "
        f"sampai {num(hasil.communalities.max(), 3)}."
    )

    profesional = (
        f"Reduksi dimensi memangkas {jumlah} fitur menjadi {k80} komponen dengan retensi "
        f"informasi {pct(retensi)}. Komponen pertama ({pct(var_pc1)} varians) "
        f"merepresentasikan {_daftar(dominan)} dan dapat dipakai sebagai indeks komposit. "
        "Skor komponen bersifat ortogonal sehingga aman menjadi input model yang sensitif "
        "terhadap multikolinearitas. Catatan operasional: muatan dan parameter penskalaan "
        "bergantung pada data pelatihan, jadi simpan keduanya agar skor pada data baru "
        "tetap sebanding."
    )

    return Temuan(
        judul="Struktur dimensi data",
        metode="Principal Component Analysis (matriks korelasi)",
        ringkas=(
            f"{jumlah} variabel dapat diringkas menjadi {k80} komponen "
            f"({pct(retensi)} informasi dipertahankan)."
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_klaster(a: Analisis) -> Temuan:
    hasil = a.klaster
    profil = a.profil_klaster
    if hasil is None or profil is None:
        raise ValueError("Hasil analisis klaster tidak tersedia.")
    ukuran = hasil.sizes()
    pembeda = profil.loc[profil["Pembeda"] == "Signifikan", "Variabel"].tolist()
    terbesar = ukuran.loc[ukuran["Anggota"].idxmax()]
    mutu = clustering._silhouette_label(hasil.silhouette).lower()

    sorotan = []
    for kolom in [c for c in profil.columns if c.startswith("Klaster ")][:3]:
        selisih = (profil[kolom] - profil["Rata-rata Total"]).abs()
        idx = selisih.idxmax()
        arah = "di atas" if profil.loc[idx, kolom] > profil.loc[idx, "Rata-rata Total"] else "di bawah"
        sorotan.append(f"{kolom} menonjol pada {profil.loc[idx, 'Variabel']} ({arah} rata-rata)")

    eksekutif = (
        f"Data terbagi secara alami menjadi {hasil.n_clusters} kelompok dengan karakter "
        f"berbeda. Kelompok terbesar mencakup {num(int(terbesar['Anggota']))} catatan "
        f"({pct(float(terbesar['Persen (%)']))} dari total). "
        + (f"Ciri khas tiap kelompok: {'; '.join(sorotan)}. " if sorotan else "")
        + f"Ketegasan pemisahan kelompok tergolong {mutu}. "
        "Pengelompokan ini berguna untuk menyusun perlakuan atau penawaran yang berbeda "
        "sesuai karakter tiap kelompok, alih-alih satu pendekatan untuk semua."
    )

    akademik = (
        f"Analisis klaster K-Means pada data terstandardisasi (z-score) menghasilkan "
        f"solusi {hasil.n_clusters} klaster sebagai yang terbaik berdasarkan koefisien "
        f"silhouette ({num(hasil.silhouette, 3)}), didukung indeks Calinski-Harabasz "
        f"{num(hasil.calinski_harabasz)} dan Davies-Bouldin {num(hasil.davies_bouldin, 3)}. "
        f"Ukuran klaster berkisar {num(int(ukuran['Anggota'].min()))} sampai "
        f"{num(int(ukuran['Anggota'].max()))} observasi. Uji ANOVA satu jalur menunjukkan "
        f"{len(pembeda)} variabel berbeda signifikan antar klaster (α = 0,05)"
        + (f": {_daftar(pembeda, 6)}. " if pembeda else ". ")
        + "K-Means mengasumsikan klaster berbentuk sferis dengan ragam serupa, sehingga "
        "solusi ini perlu dikonfirmasi dengan metode hierarki atau divalidasi pada "
        "sampel terpisah."
    )

    profesional = (
        f"Segmentasi menghasilkan {hasil.n_clusters} segmen (silhouette "
        f"{num(hasil.silhouette, 3)}, kategori {mutu}); segmen terbesar "
        f"{pct(float(terbesar['Persen (%)']))} dari basis. Variabel pembeda utama: "
        f"{_daftar(pembeda, 5)}. Sebelum dipakai operasional, kunci dua hal: parameter "
        "penskalaan dan centroid harus disimpan agar catatan baru masuk ke segmen yang "
        "konsisten, dan stabilitas segmen perlu diuji ulang berkala karena komposisi data "
        "berubah seiring waktu."
    )

    return Temuan(
        judul="Segmentasi observasi",
        metode=f"K-Means (k = {hasil.n_clusters}, dipilih dari silhouette tertinggi)",
        ringkas=(
            f"Terbentuk {hasil.n_clusters} kelompok alami dengan ketegasan pemisahan "
            f"{mutu}; {len(pembeda)} variabel menjadi pembeda utama."
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_regresi(a: Analisis) -> Temuan:
    hasil = a.regresi
    if hasil is None:
        raise ValueError("Hasil regresi linear tidak tersedia.")
    target = hasil.y_name
    model = hasil.model
    koef = hasil.coefficients[hasil.coefficients["Variabel"] != "const"]
    signifikan = koef[koef["Signifikan"] == "Ya"].copy()
    signifikan["abs_beta"] = signifikan["Beta (baku)"].abs()
    signifikan = signifikan.sort_values("abs_beta", ascending=False)
    tidak_signifikan = koef.loc[koef["Signifikan"] == "Tidak", "Variabel"].tolist()
    asumsi_gagal = hasil.diagnostics.loc[
        hasil.diagnostics["Kesimpulan"] != "Terpenuhi", "Asumsi"
    ].tolist()

    daftar_awam = "; ".join(
        f"{b['Variabel']} ({'menaikkan' if b['B'] > 0 else 'menurunkan'} {target})"
        for _, b in signifikan.head(3).iterrows()
    )

    eksekutif = (
        f"Faktor-faktor yang dianalisis mampu menjelaskan {pct(model.rsquared * 100)} "
        f"naik-turunnya {target}; sisanya {pct((1 - model.rsquared) * 100)} dipengaruhi "
        "hal lain yang belum tercatat dalam data ini. "
        + (
            f"Faktor yang terbukti berpengaruh: {daftar_awam}. "
            if not signifikan.empty
            else "Sayangnya tidak ada faktor yang terbukti berpengaruh secara meyakinkan. "
        )
        + (
            f"Sementara {_daftar(tidak_signifikan)} belum menunjukkan pengaruh yang "
            "meyakinkan, sehingga belum layak dijadikan dasar keputusan. "
            if tidak_signifikan
            else ""
        )
        + "Angka ini menggambarkan kecenderungan umum, bukan ramalan pasti untuk kasus "
        "per kasus."
    )

    rincian = "; ".join(
        f"{b['Variabel']} (B = {num_auto(b['B'])}; SE = {num_auto(b['Std. Error'])}; "
        f"β = {num(b['Beta (baku)'], 3)}; t = {num(b['t'])}; {pval(b['p-value'])})"
        for _, b in signifikan.iterrows()
    ) or "tidak ada prediktor yang signifikan"

    akademik = (
        f"Regresi linear berganda dengan {target} sebagai variabel terikat menghasilkan "
        f"model yang signifikan, F({num(int(model.df_model))}, {num(int(model.df_resid))}) "
        f"= {num(model.fvalue)}, {pval(model.f_pvalue)}, dengan R² = "
        f"{num(model.rsquared, 3)} (R² adjusted = {num(model.rsquared_adj, 3)}), tergolong "
        f"efek {_efek_r2(model.rsquared)}. Prediktor signifikan: {rincian}. "
        f"Persamaan regresi: {hasil.equation()}. "
        + (
            f"Uji asumsi klasik menemukan pelanggaran pada {_daftar(asumsi_gagal)}, "
            "sehingga galat baku sebaiknya dilaporkan dengan koreksi robust. "
            if asumsi_gagal
            else "Seluruh asumsi klasik yang diuji (normalitas residual, homoskedastisitas, "
            "dan non-autokorelasi) terpenuhi. "
        )
        + (
            f"VIF maksimum {num(hasil.vif['VIF'].max())} menunjukkan "
            + (
                "adanya multikolinearitas."
                if hasil.vif["VIF"].max() >= 10
                else "tidak ada multikolinearitas."
            )
            if not hasil.vif.empty
            else ""
        )
    )

    profesional = (
        f"Model menjelaskan {pct(model.rsquared * 100)} varians {target} (adj. R² "
        f"{num(model.rsquared_adj, 3)}; RMSE {num(np.sqrt(model.mse_resid))}). "
        + (
            f"Pengungkit terbesar berdasarkan koefisien baku: "
            f"{_daftar(signifikan['Variabel'].tolist())}. "
            if not signifikan.empty
            else "Tidak ada prediktor signifikan — kumpulkan variabel penjelas tambahan. "
        )
        + (f"Kandidat untuk dipangkas: {_daftar(tidak_signifikan)}. " if tidak_signifikan else "")
        + (
            f"Perhatian: {_daftar(asumsi_gagal)} tidak terpenuhi, sehingga interval "
            "kepercayaan cenderung terlalu sempit. "
            if asumsi_gagal
            else ""
        )
        + "Sebelum dipakai memprediksi data baru, uji model pada sampel yang tidak dipakai "
        "melatih dan pantau pergeseran distribusi prediktor dari waktu ke waktu."
    )

    return Temuan(
        judul=f"Faktor penentu {target}",
        metode="Regresi linear berganda (OLS)",
        ringkas=(
            f"Model menjelaskan {pct(model.rsquared * 100)} variasi {target}; "
            + (
                f"pendorong utama {_daftar(signifikan['Variabel'].tolist(), 2)}."
                if not signifikan.empty
                else "tidak ada prediktor signifikan."
            )
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_logistik(a: Analisis) -> Temuan:
    hasil = a.logistik
    if hasil is None:
        raise ValueError("Hasil regresi logistik tidak tersedia.")
    target = hasil.y_name
    koef = hasil.coefficients[hasil.coefficients["Variabel"] != "const"]
    signifikan = koef[koef["Signifikan"] == "Ya"].copy()
    signifikan["kekuatan"] = (signifikan["Odds Ratio"] - 1).abs()
    signifikan = signifikan.sort_values("kekuatan", ascending=False)
    perf = dict(zip(hasil.performance["Metrik"], hasil.performance["Nilai"]))
    fit = dict(zip(hasil.fit["Metrik"], hasil.fit["Nilai"]))
    positif = hasil.classes[1]

    mutu_auc = (
        "sangat baik"
        if hasil.auc >= 0.9
        else "baik"
        if hasil.auc >= 0.8
        else "cukup"
        if hasil.auc >= 0.7
        else "lemah"
    )

    daftar = "; ".join(
        _efek_odds(a, str(b["Variabel"]), float(b["B"]), str(positif))
        for _, b in signifikan.head(3).iterrows()
    )

    eksekutif = (
        f"Model ini menaksir kemungkinan terjadinya '{positif}' dengan ketepatan "
        f"{pct(perf['Akurasi'] * 100)} dan kemampuan membedakan yang tergolong {mutu_auc}. "
        + (f"Pendorong utamanya: {daftar}. " if daftar else "Belum ada pendorong yang meyakinkan. ")
        + f"Dari seluruh kasus '{positif}' yang sebenarnya, model berhasil menangkap "
        f"{pct(perf['Recall (Sensitivitas)'] * 100)}. "
        "Hasil ini cocok dipakai sebagai penyaring awal untuk menentukan siapa yang perlu "
        "diperiksa lebih teliti, bukan sebagai vonis akhir."
    )

    rincian = "; ".join(
        f"{b['Variabel']} (B = {num_auto(b['B'])}; Wald z = {num(b['Wald z'])}; "
        f"{pval(b['p-value'])}; OR = {num(b['Odds Ratio'], 3)}; IK 95% "
        f"[{num(b['OR Bawah'], 3)}; {num(b['OR Atas'], 3)}])"
        for _, b in signifikan.iterrows()
    ) or "tidak ada prediktor yang signifikan"

    akademik = (
        f"Regresi logistik biner dengan '{positif}' sebagai kategori acuan positif "
        f"menghasilkan model yang signifikan, χ² = "
        f"{num(fit['Likelihood Ratio Chi-square'])}, {pval(fit['p-value (model)'])}, "
        f"dengan Nagelkerke R² = {num(fit['Nagelkerke R2'], 3)} dan McFadden R² = "
        f"{num(fit['McFadden R2'], 3)}. Prediktor signifikan: {rincian}. Kemampuan "
        f"diskriminasi tergolong {mutu_auc} (AUC = {num(hasil.auc, 3)}) dengan akurasi "
        f"klasifikasi {pct(perf['Akurasi'] * 100)}, sensitivitas "
        f"{pct(perf['Recall (Sensitivitas)'] * 100)}, dan spesifisitas "
        f"{pct(perf['Spesifisitas'] * 100)} pada ambang 0,50. Odds ratio ditafsirkan "
        "sebagai perubahan rasio peluang per satu satuan kenaikan prediktor, dengan "
        "prediktor lain dikontrol."
    )

    profesional = (
        f"Pengklasifikasi mencapai AUC {num(hasil.auc, 3)} ({mutu_auc}), akurasi "
        f"{pct(perf['Akurasi'] * 100)}, presisi {pct(perf['Presisi'] * 100)}, recall "
        f"{pct(perf['Recall (Sensitivitas)'] * 100)} pada ambang 0,50. "
        + (
            f"Fitur berpengaruh terbesar: {_daftar(signifikan['Variabel'].tolist())}. "
            if not signifikan.empty
            else ""
        )
        + "Ambang keputusan sebaiknya tidak dibiarkan di 0,50 melainkan disetel menurut "
        "biaya relatif kesalahan: turunkan ambang bila melewatkan kasus positif jauh lebih "
        "mahal daripada memeriksa kasus negatif. Sebelum produksi, validasi pada periode "
        "data berbeda serta pantau pergeseran populasi dan kalibrasi probabilitasnya."
    )

    return Temuan(
        judul=f"Prediksi {target}",
        metode="Regresi logistik biner",
        ringkas=(
            f"Model prediksi '{positif}' berkinerja {mutu_auc} (AUC {num(hasil.auc, 3)}, "
            f"akurasi {pct(perf['Akurasi'] * 100)})."
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_perbedaan_kelompok(a: Analisis) -> Temuan:
    hasil = a.manova
    if hasil is None:
        raise ValueError("Hasil MANOVA tidak tersedia.")
    kelompok = hasil.factor
    wilks = hasil.multivariate.set_index("Statistik").loc["Wilks' lambda"]
    pillai = hasil.multivariate.set_index("Statistik").loc["Pillai's trace"]
    signifikan = float(wilks["p-value"]) < 0.05
    univariat = hasil.univariate.sort_values("Eta-squared", ascending=False)
    berbeda = univariat.loc[univariat["Signifikan"] == "Ya", "Variabel"].tolist()
    n_kelompok = len(hasil.group_sizes)
    teratas = univariat.iloc[0] if not univariat.empty else None

    catatan_box = ""
    if a.box_m is not None:
        catatan_box = (
            f"Box's M = {num(a.box_m.statistic)} (χ² = {num(a.box_m.chi_square)}; "
            f"df = {a.box_m.df}; {pval(a.box_m.p_value)}), sehingga homogenitas matriks "
            "kovarians "
            + (
                "terpenuhi. "
                if a.box_m.homogeneous
                else "tidak terpenuhi; Pillai's trace dijadikan acuan utama karena paling "
                "tahan terhadap pelanggaran asumsi ini. "
            )
        )

    eksekutif = (
        f"Kelompok {kelompok} yang terdiri atas {n_kelompok} kategori "
        + (
            "memang berbeda nyata satu sama lain bila dilihat dari keseluruhan ukuran yang "
            "dianalisis. "
            if signifikan
            else "ternyata tidak berbeda secara meyakinkan pada ukuran-ukuran yang dianalisis. "
        )
        + (
            f"Perbedaan paling terasa pada {teratas['Variabel']}. "
            if teratas is not None and signifikan
            else ""
        )
        + (
            f"Secara rinci, {len(berbeda)} ukuran benar-benar berbeda antar kelompok. "
            if signifikan
            else ""
        )
        + "Temuan ini menjadi dasar memperlakukan tiap kelompok secara berbeda — sejauh "
        "perbedaannya memang nyata dan cukup besar."
    )

    rincian_univariat = (
        "; ".join(
            f"{b['Variabel']} (F = {num(b['F'])}; {pval(b['p-value'])}; η² = "
            f"{num(b['Eta-squared'], 3)}; efek {_efek_eta(b['Eta-squared'])})"
            for _, b in univariat[univariat["Signifikan"] == "Ya"].head(4).iterrows()
        )
        if berbeda
        else "tidak ada variabel yang berbeda signifikan"
    )

    akademik = (
        f"MANOVA satu jalur dengan {kelompok} sebagai faktor ({n_kelompok} kelompok) dan "
        f"{len(hasil.dependents)} variabel dependen menghasilkan Wilks' lambda = "
        f"{num(wilks['Nilai'], 4)}, F({num(int(wilks['df Hipotesis']))}, "
        f"{num(int(wilks['df Galat']))}) = {num(wilks['F'])}, {pval(wilks['p-value'])}; "
        f"Pillai's trace = {num(pillai['Nilai'], 4)}, {pval(pillai['p-value'])}. "
        + (
            "Perbedaan vektor rata-rata antar kelompok terbukti signifikan. "
            if signifikan
            else "Tidak terdapat perbedaan vektor rata-rata yang signifikan antar kelompok. "
        )
        + catatan_box
        + f"Uji lanjutan ANOVA univariat: {rincian_univariat}."
    )

    profesional = (
        f"Uji beda multivariat antar {n_kelompok} kelompok {kelompok}: "
        + (
            f"signifikan (Wilks' lambda {num(wilks['Nilai'], 4)}; {pval(wilks['p-value'])}). "
            if signifikan
            else f"tidak signifikan ({pval(wilks['p-value'])}). "
        )
        + (
            f"Variabel paling membedakan: {_daftar(berbeda, 4)} — inilah yang layak dipakai "
            "sebagai penanda kelompok pada sistem penyaringan atau pelaporan. "
            if berbeda
            else "Tidak ada variabel yang layak dipakai sebagai penanda pembeda kelompok. "
        )
        + "Perlu diingat, perbedaan yang signifikan secara statistik belum tentu besar "
        "secara praktis — periksa ukuran efek (η²) dan selisih rata-rata sesungguhnya "
        "sebelum mengubah kebijakan."
    )

    return Temuan(
        judul=f"Perbedaan antar kelompok {kelompok}",
        metode="MANOVA satu jalur dengan uji lanjutan ANOVA",
        ringkas=(
            f"Kelompok {kelompok} "
            + (
                f"berbeda signifikan pada {len(berbeda)} variabel."
                if signifikan
                else "tidak berbeda signifikan pada variabel yang dianalisis."
            )
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_diskriminan(a: Analisis) -> Temuan:
    hasil = a.diskriminan
    if hasil is None:
        raise ValueError("Hasil analisis diskriminan tidak tersedia.")
    kelompok = hasil.labels.name
    n_kelompok = len(hasil.classes)
    baseline = float(hasil.labels.value_counts(normalize=True).max())
    # Validasi silang gagal bila ada kelompok dengan anggota terlalu sedikit;
    # dalam keadaan itu akurasi data latih dipakai, disertai catatan terbuka.
    punya_cv = bool(np.isfinite(hasil.cv_accuracy))
    akurasi = float(hasil.cv_accuracy if punya_cv else hasil.accuracy)
    dasar = "validasi silang" if punya_cv else "data latih"
    catatan_cv = (
        ""
        if punya_cv
        else " Validasi silang tidak dapat dihitung karena ada kelompok yang anggotanya "
        "terlalu sedikit, sehingga angka ini berasal dari data yang sama dengan data "
        "pelatihan dan cenderung terlalu optimistis."
    )
    peningkatan = (akurasi - baseline) * 100
    fungsi_signifikan = (
        int((hasil.wilks["Signifikan"] == "Ya").sum()) if hasil.wilks is not None else 0
    )
    kontributor = (
        hasil.coefficients.iloc[:, 0].abs().sort_values(ascending=False).head(3).index.tolist()
        if hasil.coefficients is not None
        else []
    )

    eksekutif = (
        f"Berbekal ukuran-ukuran yang tersedia, kelompok {kelompok} suatu catatan dapat "
        f"ditebak dengan ketepatan {pct(akurasi * 100)}. Sebagai pembanding, "
        f"menebak dengan selalu memilih kelompok terbesar hanya benar {pct(baseline * 100)}, "
        + (
            f"jadi ukuran-ukuran ini memang membantu (lebih tepat {num(peningkatan, 1)} poin). "
            if peningkatan > 0
            else "jadi ukuran-ukuran ini belum banyak membantu. "
        )
        + (f"Ukuran paling menentukan: {_daftar(kontributor)}. " if kontributor else "")
        + "Artinya kelompok-kelompok tersebut punya ciri yang benar-benar dapat dikenali "
        "dari data."
        + catatan_cv
    )

    rincian_fungsi = ""
    if hasil.wilks is not None and not hasil.wilks.empty:
        baris = hasil.wilks.iloc[0]
        lam = float(baris["Wilks' Lambda"])
        rincian_fungsi = (
            f" (fungsi pertama: Wilks' lambda = {num(lam, 4)}; χ² = "
            f"{num(baris['Chi-square'])}; df = {int(baris['df'])}; {pval(baris['p-value'])})"
        )

    akademik = (
        f"Analisis diskriminan linear terhadap {n_kelompok} kelompok {kelompok} "
        f"menghasilkan {fungsi_signifikan} fungsi diskriminan yang signifikan"
        + rincian_fungsi
        + f". Ketepatan klasifikasi mencapai {pct(hasil.accuracy * 100)} pada data latih"
        + (
            f" dan {pct(hasil.cv_accuracy * 100)} pada validasi silang 5-lipat"
            if punya_cv
            else " (validasi silang tidak dapat dihitung karena terdapat kelompok dengan "
            "anggota kurang dari dua)"
        )
        + f", dibandingkan proporsi kelompok terbesar {pct(baseline * 100)} sebagai patokan "
        "minimal (proportional chance criterion). "
        + (f"Kontributor terbesar pada fungsi pertama: {_daftar(kontributor)}." if kontributor else "")
    )

    profesional = (
        f"Klasifikasi keanggotaan {kelompok} tercapai pada {pct(akurasi * 100)} "
        f"({dasar}) versus baseline {pct(baseline * 100)}. "
        + (f"Fitur penentu: {_daftar(kontributor)}. " if kontributor else "")
    )
    if punya_cv:
        selisih = abs(hasil.accuracy - hasil.cv_accuracy) * 100
        profesional += (
            f"Selisih akurasi data latih dan validasi silang {num(selisih, 1)} poin — "
            + (
                "cukup lebar, indikasi model terlalu menyesuaikan data latih. "
                if selisih > 10
                else "kecil, model cukup stabil. "
            )
        )
    else:
        profesional += (
            "Validasi silang gagal dijalankan karena ada kelompok dengan anggota kurang "
            "dari dua — gabungkan kategori kecil atau kumpulkan lebih banyak data sebelum "
            "angka ini dipercaya. "
        )
    profesional += (
        "Aturan klasifikasi ini dapat dipakai menandai catatan baru, dengan peninjauan "
        "berkala saat komposisi kelompok berubah."
    )

    return Temuan(
        judul=f"Ciri pembeda kelompok {kelompok}",
        metode="Analisis diskriminan linear (LDA) dengan validasi silang",
        ringkas=(
            f"Keanggotaan {kelompok} dapat ditebak dengan ketepatan {pct(akurasi * 100)} "
            f"({dasar}; patokan {pct(baseline * 100)})."
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


def temuan_kanonik(a: Analisis) -> Temuan:
    hasil = a.kanonik
    if hasil is None:
        raise ValueError("Hasil korelasi kanonik tidak tersedia.")
    r1 = float(hasil.correlations[0])
    sig = hasil.significance.iloc[0]
    lam = float(sig["Wilks' Lambda"])
    n_signifikan = int((hasil.significance["Signifikan"] == "Ya").sum())
    redundansi = float(hasil.redundancy["Redundansi Y|X (%)"].iloc[0])
    muatan_x = hasil.x_loadings.iloc[:, 0].abs().sort_values(ascending=False).head(2).index.tolist()
    muatan_y = hasil.y_loadings.iloc[:, 0].abs().sort_values(ascending=False).head(2).index.tolist()
    kekuatan = correlation.strength_label(r1).lower()

    eksekutif = (
        f"Dua kelompok ukuran yang dibandingkan saling berkaitan dengan kekuatan "
        f"{kekuatan} (nilai keterkaitan {num(r1, 3)} dari maksimal 1). Kaitan itu terutama "
        f"dijembatani oleh {_daftar(muatan_x, 2)} di satu sisi dan {_daftar(muatan_y, 2)} "
        "di sisi lain. Dengan kata lain, kedua kelompok ukuran ini menceritakan sebagian "
        "cerita yang sama, sehingga pelaporannya tidak perlu sepenuhnya terpisah."
    )

    akademik = (
        f"Analisis korelasi kanonik antara gugus X ({len(a.konfig.gugus_x)} variabel) dan "
        f"gugus Y ({len(a.konfig.gugus_y)} variabel) pada n = {num(hasil.n)} menghasilkan "
        f"{n_signifikan} fungsi kanonik yang signifikan. Fungsi pertama memiliki korelasi "
        f"kanonik Rc = {num(r1, 3)} (Rc² = {num(r1**2, 3)}); Wilks' lambda = {num(lam, 4)}; "
        f"χ² = {num(sig['Chi-square'])}; df = {int(sig['df'])}; {pval(sig['p-value'])}. "
        f"Muatan kanonik tertinggi pada gugus X dimiliki {_daftar(muatan_x, 2)}, pada gugus "
        f"Y dimiliki {_daftar(muatan_y, 2)}. Indeks redundansi menunjukkan {pct(redundansi)} "
        "varians gugus Y dapat dijelaskan oleh variat kanonik gugus X — angka ini perlu "
        "dilaporkan bersama Rc karena korelasi kanonik tinggi tidak otomatis berarti daya "
        "jelas yang besar."
    )

    profesional = (
        f"Keterkaitan antar dua gugus variabel: Rc {num(r1, 3)} ({kekuatan}), redundansi "
        f"{pct(redundansi)}. Penghubung utamanya {_daftar(muatan_x, 2)} terhadap "
        f"{_daftar(muatan_y, 2)}. "
        + (
            "Redundansi rendah berarti kaitan tersebut belum cukup kuat untuk saling "
            "menggantikan; tetap kumpulkan kedua gugus data. "
            if redundansi < 20
            else "Redundansi cukup tinggi sehingga sebagian pengukuran dapat disederhanakan "
            "tanpa banyak kehilangan informasi. "
        )
        + "Gunakan temuan ini untuk merampingkan pengumpulan data dan memilih indikator "
        "yang benar-benar menambah informasi baru."
    )

    return Temuan(
        judul="Hubungan antar dua gugus variabel",
        metode="Analisis korelasi kanonik",
        ringkas=(
            f"Kedua gugus variabel berkaitan {kekuatan} (Rc = {num(r1, 3)}) dengan "
            f"redundansi {pct(redundansi)}."
        ),
        eksekutif=eksekutif,
        akademik=akademik,
        profesional=profesional,
    )


# --------------------------------------------------------------------------- #
# Tahap 2b — lampu status, pendorong, rekomendasi, keterbatasan
# --------------------------------------------------------------------------- #


def susun_lampu(a: Analisis) -> list[Lampu]:
    lampu: list[Lampu] = []
    n = len(a.df)
    rasio = n / max(len(a.konfig.variabel), 1)
    lampu.append(
        Lampu(
            label="Kecukupan data",
            nilai=f"N = {num(n)} · {num(rasio, 1)} obs/variabel",
            status="baik" if rasio >= 10 else "perhatian" if rasio >= 5 else "kritis",
            catatan=(
                "Memenuhi aturan praktis minimal 10 observasi per variabel."
                if rasio >= 10
                else "Di bawah aturan praktis 10 observasi per variabel; estimasi kurang stabil."
            ),
        )
    )

    if a.mardia is not None:
        normal = a.mardia.multivariate_normal
        lampu.append(
            Lampu(
                label="Asumsi normalitas",
                nilai=f"Mardia {pval(a.mardia.skew_p)}",
                status="baik" if normal else "perhatian",
                catatan=(
                    "Normalitas multivariat terpenuhi."
                    if normal
                    else "Menyimpang dari normal; uji signifikansi lebih peka terhadap pencilan."
                ),
            )
        )

    if a.kmo is not None:
        lampu.append(
            Lampu(
                label="Kelayakan reduksi dimensi",
                nilai=f"KMO = {num(a.kmo.overall, 3)}",
                status=(
                    "baik"
                    if a.kmo.overall >= 0.7
                    else "perhatian"
                    if a.kmo.overall >= 0.5
                    else "kritis"
                ),
                catatan=f"Kategori {a.kmo.interpretation.lower()}.",
            )
        )

    if a.vif is not None and not a.vif.empty:
        vif_max = float(a.vif["VIF"].max())
        lampu.append(
            Lampu(
                label="Multikolinearitas",
                nilai=f"VIF maks = {num(vif_max)}",
                status="baik" if vif_max < 5 else "perhatian" if vif_max < 10 else "kritis",
                catatan=(
                    "Antar variabel tidak saling menggantikan."
                    if vif_max < 10
                    else "Beberapa variabel memuat informasi yang sama."
                ),
            )
        )

    if a.regresi is not None:
        r2 = float(a.regresi.model.rsquared)
        lampu.append(
            Lampu(
                label="Daya jelas model",
                nilai=f"R² = {num(r2, 3)}",
                status="baik" if r2 >= 0.5 else "perhatian" if r2 >= 0.25 else "kritis",
                catatan=(
                    f"{pct((1 - r2) * 100)} variasi masih berasal dari faktor di luar model."
                ),
            )
        )

    if a.logistik is not None:
        auc = float(a.logistik.auc)
        lampu.append(
            Lampu(
                label="Ketajaman prediksi",
                nilai=f"AUC = {num(auc, 3)}",
                status="baik" if auc >= 0.8 else "perhatian" if auc >= 0.7 else "kritis",
                catatan=(
                    "Model memisahkan dua kelompok dengan baik."
                    if auc >= 0.8
                    else "Kemampuan memisahkan masih terbatas."
                ),
            )
        )

    if a.klaster is not None and a.klaster.silhouette is not None:
        sil = float(a.klaster.silhouette)
        lampu.append(
            Lampu(
                label="Ketegasan segmentasi",
                nilai=f"Silhouette = {num(sil, 3)}",
                status="baik" if sil >= 0.5 else "perhatian" if sil >= 0.25 else "kritis",
                catatan=clustering._silhouette_label(sil) + ".",
            )
        )

    return lampu


def _catatan_efek(a: Analisis, prediktor: str, koefisien: float) -> str:
    """Terjemahkan koefisien menjadi perubahan nyata per satu simpangan baku.

    Koefisien mentah sering bernilai sangat kecil bila satuan prediktornya besar
    (misalnya rupiah), sehingga sulit dibaca. Menyatakannya per satu simpangan baku
    membuat besaran pengaruhnya tetap terbaca berapa pun satuannya.
    """
    target = a.regresi.y_name if a.regresi is not None else "variabel terikat"
    if prediktor in a.df.columns and pd.api.types.is_numeric_dtype(a.df[prediktor]):
        sd = float(a.df[prediktor].std(ddof=1))
        if np.isfinite(sd) and sd > 0:
            perubahan = koefisien * sd
            arah = "naik" if perubahan > 0 else "turun"
            return (
                f"Kenaikan {prediktor} sebesar {num(sd, 2)} satuan (satu simpangan baku) "
                f"membuat {target} {arah} sekitar {num(abs(perubahan), 2)} satuan."
            )
    return (
        f"Setiap kenaikan satu satuan mengubah {target} sebesar "
        f"{num_auto(koefisien)} satuan."
    )


def _efek_odds(a: Analisis, prediktor: str, koefisien: float, kelas: str) -> str:
    """Nyatakan pengaruh sebuah prediktor sebagai perubahan peluang per simpangan baku.

    Odds ratio per satu satuan bisa berangka ekstrem bila satuan prediktornya kecil
    (misalnya rasio 0–1), sehingga menyesatkan bila dibaca apa adanya oleh pembaca awam.
    """
    if prediktor in a.df.columns and pd.api.types.is_numeric_dtype(a.df[prediktor]):
        sd = float(a.df[prediktor].std(ddof=1))
        if np.isfinite(sd) and sd > 0:
            rasio = float(np.exp(koefisien * sd))
            if rasio >= 1:
                return (
                    f"kenaikan {prediktor} sebesar {num(sd, 2)} satuan (satu simpangan baku) "
                    f"menaikkan peluang '{kelas}' sekitar {pct((rasio - 1) * 100)}"
                )
            return (
                f"kenaikan {prediktor} sebesar {num(sd, 2)} satuan (satu simpangan baku) "
                f"menurunkan peluang '{kelas}' sekitar {pct((1 - rasio) * 100)}"
            )
    return f"{prediktor} mengubah peluang '{kelas}' secara signifikan"


def _kapital(teks: str) -> str:
    return teks[:1].upper() + teks[1:] if teks else teks


def susun_pendorong(a: Analisis) -> tuple[list[Pendorong], str]:
    """Peringkat variabel penjelas; sumbernya regresi, logistik, atau PCA."""

    def _kinerja(nama: str) -> float | None:
        if nama not in a.df.columns or not pd.api.types.is_numeric_dtype(a.df[nama]):
            return None
        kolom = a.df[nama].dropna()
        if kolom.empty:
            return None
        return float((kolom <= kolom.mean()).mean() * 100)

    if a.regresi is not None:
        koef = a.regresi.coefficients[a.regresi.coefficients["Variabel"] != "const"].copy()
        koef["abs_beta"] = koef["Beta (baku)"].abs()
        koef = koef.dropna(subset=["abs_beta"]).sort_values("abs_beta", ascending=False)
        if not koef.empty:
            puncak = float(koef["abs_beta"].iloc[0]) or 1.0
            pendorong = [
                Pendorong(
                    nama=str(b["Variabel"]),
                    kekuatan=float(b["abs_beta"] / puncak),
                    nilai=float(b["Beta (baku)"]),
                    satuan="beta",
                    arah="naik" if b["B"] > 0 else "turun",
                    p_value=float(b["p-value"]),
                    kinerja=_kinerja(str(b["Variabel"])),
                    catatan=(
                        _catatan_efek(a, str(b["Variabel"]), float(b["B"]))
                        if float(b["p-value"]) < 0.05
                        else "Belum terbukti berpengaruh; jangan dijadikan dasar keputusan."
                    ),
                )
                for _, b in koef.head(8).iterrows()
            ]
            return pendorong, f"koefisien baku regresi terhadap {a.regresi.y_name}"

    if a.logistik is not None:
        koef = a.logistik.coefficients[a.logistik.coefficients["Variabel"] != "const"].copy()
        koef["kekuatan"] = koef["Odds Ratio"].apply(lambda v: abs(np.log(max(float(v), 1e-9))))
        koef = koef.sort_values("kekuatan", ascending=False)
        if not koef.empty:
            puncak = float(koef["kekuatan"].iloc[0]) or 1.0
            pendorong = [
                Pendorong(
                    nama=str(b["Variabel"]),
                    kekuatan=float(b["kekuatan"] / puncak),
                    nilai=float(b["Odds Ratio"]),
                    satuan="odds ratio",
                    arah="naik" if float(b["Odds Ratio"]) > 1 else "turun",
                    p_value=float(b["p-value"]),
                    kinerja=_kinerja(str(b["Variabel"])),
                    catatan=(
                        _kapital(
                            _efek_odds(
                                a, str(b["Variabel"]), float(b["B"]), str(a.logistik.classes[1])
                            )
                        )
                        + "."
                        if float(b["p-value"]) < 0.05
                        else "Belum terbukti berpengaruh."
                    ),
                )
                for _, b in koef.head(8).iterrows()
            ]
            return pendorong, f"odds ratio regresi logistik terhadap {a.logistik.y_name}"

    if a.pca is not None:
        muatan = a.pca.loadings["PC1"]
        urut = muatan.abs().sort_values(ascending=False)
        puncak = float(urut.iloc[0]) or 1.0
        pendorong = [
            Pendorong(
                nama=str(nama),
                kekuatan=float(abs(muatan[nama]) / puncak),
                nilai=float(muatan[nama]),
                satuan="muatan",
                arah="naik" if muatan[nama] > 0 else "turun",
                p_value=float("nan"),
                kinerja=_kinerja(str(nama)),
                catatan="Kontribusi terhadap dimensi utama data.",
            )
            for nama in urut.head(8).index
        ]
        return pendorong, "muatan komponen utama pertama"

    return [], ""


def susun_rekomendasi(a: Analisis) -> list[Rekomendasi]:
    rekomendasi: list[Rekomendasi] = []

    if a.regresi is not None:
        koef = a.regresi.coefficients[a.regresi.coefficients["Variabel"] != "const"]
        signifikan = koef[koef["Signifikan"] == "Ya"].copy()
        if not signifikan.empty:
            signifikan["abs_beta"] = signifikan["Beta (baku)"].abs()
            utama = signifikan.sort_values("abs_beta", ascending=False).iloc[0]
            rekomendasi.append(
                Rekomendasi(
                    judul=f"Fokuskan upaya pada {utama['Variabel']}",
                    alasan=(
                        f"Variabel ini punya pengaruh baku terbesar terhadap "
                        f"{a.regresi.y_name} (β = {num(utama['Beta (baku)'], 3)}, "
                        f"{pval(utama['p-value'])}). Perubahan di sini paling terasa "
                        "hasilnya dibanding variabel lain."
                    ),
                    prioritas="tinggi",
                )
            )
        tidak_signifikan = koef.loc[koef["Signifikan"] == "Tidak", "Variabel"].tolist()
        if tidak_signifikan:
            rekomendasi.append(
                Rekomendasi(
                    judul=f"Hentikan pengambilan keputusan berdasarkan {_daftar(tidak_signifikan)}",
                    alasan=(
                        "Variabel ini tidak menunjukkan pengaruh yang meyakinkan setelah "
                        "variabel lain diperhitungkan, sehingga sumber daya yang dicurahkan "
                        "ke sana kemungkinan besar tidak berbalas."
                    ),
                    prioritas="sedang",
                )
            )
        r2 = float(a.regresi.model.rsquared)
        if r2 < 0.5:
            rekomendasi.append(
                Rekomendasi(
                    judul="Lengkapi data dengan faktor penjelas baru",
                    alasan=(
                        f"{pct((1 - r2) * 100)} variasi {a.regresi.y_name} belum terjelaskan "
                        "oleh variabel yang tersedia. Tambahkan variabel perilaku, waktu, "
                        "atau konteks eksternal pada pengumpulan data berikutnya."
                    ),
                    prioritas="tinggi" if r2 < 0.25 else "sedang",
                )
            )

    if a.logistik is not None and np.isfinite(a.logistik.auc):
        if a.logistik.auc >= 0.7:
            rekomendasi.append(
                Rekomendasi(
                    judul="Pakai model sebagai penyaring awal, dengan ambang yang disetel",
                    alasan=(
                        f"AUC {num(a.logistik.auc, 3)} cukup untuk memprioritaskan "
                        "pemeriksaan. Setel ambang menurut biaya relatif kesalahan, bukan "
                        "dibiarkan di 0,50."
                    ),
                    prioritas="tinggi",
                )
            )
        else:
            rekomendasi.append(
                Rekomendasi(
                    judul="Jangan gunakan model prediksi ini untuk keputusan otomatis",
                    alasan=(
                        f"AUC {num(a.logistik.auc, 3)} masih lemah; kesalahan klasifikasi "
                        "terlalu sering untuk dipakai tanpa pemeriksaan manusia."
                    ),
                    prioritas="tinggi",
                )
            )

    if a.klaster is not None and a.klaster.silhouette is not None:
        if a.klaster.silhouette >= 0.25:
            rekomendasi.append(
                Rekomendasi(
                    judul=f"Bedakan perlakuan untuk {a.klaster.n_clusters} segmen yang terbentuk",
                    alasan=(
                        "Segmen memiliki profil yang berbeda nyata, sehingga satu pendekatan "
                        "seragam untuk semua kemungkinan besar tidak optimal."
                    ),
                    prioritas="sedang",
                )
            )
        else:
            rekomendasi.append(
                Rekomendasi(
                    judul="Tunda penggunaan segmentasi untuk kebijakan",
                    alasan=(
                        f"Silhouette {num(a.klaster.silhouette, 3)} menunjukkan batas antar "
                        "segmen kabur; kelompok yang terbentuk belum cukup tegas untuk "
                        "dijadikan dasar perlakuan berbeda."
                    ),
                    prioritas="sedang",
                )
            )

    if a.vif is not None and not a.vif.empty and float(a.vif["VIF"].max()) >= 10:
        bermasalah = a.vif.loc[a.vif["VIF"] >= 10, "Variabel"].tolist()
        rekomendasi.append(
            Rekomendasi(
                judul=f"Rampingkan variabel yang tumpang tindih ({_daftar(bermasalah)})",
                alasan=(
                    "Variabel-variabel ini memuat informasi yang hampir sama sehingga "
                    "membuat koefisien model tidak stabil. Pilih salah satu atau gabungkan "
                    "menjadi satu indeks."
                ),
                prioritas="tinggi",
            )
        )

    if a.pca is not None:
        k80 = a.pca.components_needed(0.80)
        if k80 < len(a.pca.variables) / 2:
            rekomendasi.append(
                Rekomendasi(
                    judul=f"Sederhanakan pelaporan menjadi {k80} indikator utama",
                    alasan=(
                        f"{len(a.pca.variables)} ukuran yang dilaporkan sekarang sebagian "
                        f"besar menceritakan hal serupa; {k80} dimensi sudah menyimpan "
                        f"{pct(a.pca.cumulative_ratio[k80 - 1] * 100)} informasinya."
                    ),
                    prioritas="rendah",
                )
            )

    missing = float(a.df.isna().mean().mean() * 100)
    if missing > 5:
        rekomendasi.append(
            Rekomendasi(
                judul="Perbaiki kelengkapan pencatatan data",
                alasan=(
                    f"Rata-rata {pct(missing)} isian kosong. Data yang hilang tidak acak "
                    "dapat menggeser seluruh kesimpulan di atas."
                ),
                prioritas="tinggi",
            )
        )

    rekomendasi.append(
        Rekomendasi(
            judul="Validasi ulang pada periode data berikutnya",
            alasan=(
                "Seluruh kesimpulan di atas berlaku untuk potret data saat ini. Ulangi "
                "analisis pada periode berikutnya untuk memastikan polanya bertahan "
                "sebelum dijadikan kebijakan tetap."
            ),
            prioritas="sedang",
        )
    )
    return rekomendasi


def susun_keterbatasan(a: Analisis) -> list[str]:
    batas = [
        "Analisis ini bersifat observasional dan potong lintang. Yang ditemukan adalah "
        "pola hubungan, bukan bukti sebab-akibat; kalimat “A menyebabkan B” belum "
        "didukung oleh desain data ini.",
    ]
    if a.regresi is not None:
        sisa = (1 - float(a.regresi.model.rsquared)) * 100
        batas.append(
            f"Sebesar {pct(sisa)} variasi {a.regresi.y_name} tidak terjelaskan oleh model, "
            "sehingga keputusan besar sebaiknya tidak digantungkan pada model ini saja."
        )
        gagal = a.regresi.diagnostics.loc[
            a.regresi.diagnostics["Kesimpulan"] != "Terpenuhi", "Asumsi"
        ].tolist()
        if gagal:
            batas.append(
                f"Asumsi {_daftar(gagal)} tidak terpenuhi, sehingga interval kepercayaan "
                "dan nilai p perlu dibaca dengan hati-hati."
            )
    if a.mardia is not None and not a.mardia.multivariate_normal:
        batas.append(
            "Data menyimpang dari normalitas multivariat; uji parametrik tetap dijalankan "
            "karena cukup tahan (robust) pada ukuran sampel ini, namun hal ini perlu "
            "dinyatakan terbuka dalam pelaporan."
        )
    rasio = len(a.df) / max(len(a.konfig.variabel), 1)
    if rasio < 10:
        batas.append(
            f"Rasio observasi terhadap variabel hanya {num(rasio, 1)} : 1, di bawah aturan "
            "praktis 10 : 1, sehingga hasil dapat berubah pada sampel lain."
        )
    if a.pencilan is not None:
        n_out = int((a.pencilan["Pencilan"] == "Ya").sum())
        if n_out:
            batas.append(
                f"Terdapat {num(n_out)} pencilan multivariat yang tetap disertakan dalam "
                "analisis. Hasil dapat berubah bila kasus-kasus tersebut dikeluarkan atau "
                "ditangani secara khusus."
            )
    if a.klaster is not None:
        batas.append(
            "Solusi klaster bergantung pada variabel yang dipilih dan penskalaannya; "
            "kelompok yang terbentuk adalah konstruksi statistik, bukan kategori yang "
            "sudah ada di lapangan."
        )
    batas.append(
        "Data mewakili catatan yang berhasil terkumpul. Pihak atau kasus yang tidak "
        "tercatat tidak terwakili, sehingga generalisasi ke populasi yang lebih luas "
        "perlu kehati-hatian."
    )
    return batas


# --------------------------------------------------------------------------- #
# Tahap 2c — tabel bergaya APA dan paragraf siap salin
# --------------------------------------------------------------------------- #


def tabel_deskriptif_korelasi(a: Analisis) -> pd.DataFrame:
    """Tabel 1 bergaya APA: M, SD, dan matriks korelasi segitiga bawah + bintang."""
    if a.korelasi is None:
        raise ValueError("Hasil korelasi tidak tersedia.")
    variabel = list(a.korelasi.matrix.columns)
    subset = preprocessing.clean_subset(a.df, variabel)
    baris = []
    for i, nama in enumerate(variabel):
        data = {"Variabel": f"{i + 1}. {nama}"}
        data["M"] = num(float(subset[nama].mean()), 3)
        data["SD"] = num(float(subset[nama].std(ddof=1)), 3)
        for j, lain in enumerate(variabel):
            kolom = str(j + 1)
            if j > i:
                data[kolom] = ""
            elif j == i:
                data[kolom] = "—"
            else:
                r = float(a.korelasi.matrix.loc[nama, lain])
                p = float(a.korelasi.p_values.loc[nama, lain])
                data[kolom] = f"{num(r, 3)}{bintang(p)}"
        baris.append(data)
    return pd.DataFrame(baris)


def tabel_regresi(a: Analisis) -> pd.DataFrame:
    if a.regresi is None:
        raise ValueError("Hasil regresi tidak tersedia.")
    koef = a.regresi.coefficients
    return pd.DataFrame(
        {
            "Prediktor": koef["Variabel"],
            "B": [num_auto(v) for v in koef["B"]],
            "SE": [num_auto(v) for v in koef["Std. Error"]],
            "β": [num(v, 3) if np.isfinite(v) else "—" for v in koef["Beta (baku)"]],
            "t": [num(v, 3) for v in koef["t"]],
            "p": [
                "< 0,001" if v < 0.001 else num(v, 3) for v in koef["p-value"]
            ],
            "IK 95%": [
                f"[{num(low, 3)}; {num(high, 3)}]"
                for low, high in zip(koef.iloc[:, 6], koef.iloc[:, 7])
            ],
        }
    )


def tabel_asumsi(a: Analisis) -> pd.DataFrame:
    baris = []
    if a.mardia is not None:
        baris.append(
            {
                "Asumsi": "Normalitas multivariat (Mardia skewness)",
                "Statistik": f"χ² = {num(a.mardia.skew_chi2)}",
                "p": "< 0,001" if a.mardia.skew_p < 0.001 else num(a.mardia.skew_p, 3),
                "Keputusan": "Terpenuhi" if a.mardia.skew_p > 0.05 else "Tidak terpenuhi",
            }
        )
    if a.kmo is not None:
        baris.append(
            {
                "Asumsi": "Kecukupan sampel (KMO)",
                "Statistik": num(a.kmo.overall, 3),
                "p": "—",
                "Keputusan": "Layak" if a.kmo.overall >= 0.5 else "Tidak layak",
            }
        )
    if a.bartlett is not None:
        baris.append(
            {
                "Asumsi": "Bartlett's test of sphericity",
                "Statistik": f"χ²({a.bartlett.df}) = {num(a.bartlett.chi_square)}",
                "p": "< 0,001" if a.bartlett.p_value < 0.001 else num(a.bartlett.p_value, 3),
                "Keputusan": "Layak difaktorkan" if a.bartlett.adequate else "Tidak layak",
            }
        )
    if a.vif is not None and not a.vif.empty:
        vif_max = float(a.vif["VIF"].max())
        baris.append(
            {
                "Asumsi": "Multikolinearitas (VIF maksimum)",
                "Statistik": num(vif_max),
                "p": "—",
                "Keputusan": "Aman" if vif_max < 10 else "Bermasalah",
            }
        )
    if a.box_m is not None:
        baris.append(
            {
                "Asumsi": "Homogenitas matriks kovarians (Box's M)",
                "Statistik": f"χ²({a.box_m.df}) = {num(a.box_m.chi_square)}",
                "p": "< 0,001" if a.box_m.p_value < 0.001 else num(a.box_m.p_value, 3),
                "Keputusan": "Terpenuhi" if a.box_m.homogeneous else "Tidak terpenuhi",
            }
        )
    if a.regresi is not None:
        for _, b in a.regresi.diagnostics.iterrows():
            p = b["p-value"]
            baris.append(
                {
                    "Asumsi": b["Asumsi"],
                    "Statistik": num(b["Statistik"], 3),
                    "p": "—" if not np.isfinite(p) else ("< 0,001" if p < 0.001 else num(p, 3)),
                    "Keputusan": b["Kesimpulan"],
                }
            )
    return pd.DataFrame(baris)


def tabel_manova(a: Analisis) -> pd.DataFrame:
    if a.manova is None:
        raise ValueError("Hasil MANOVA tidak tersedia.")
    tabel = a.manova.multivariate
    return pd.DataFrame(
        {
            "Statistik uji": tabel["Statistik"],
            "Nilai": [num(v, 4) for v in tabel["Nilai"]],
            "F": [num(v, 3) for v in tabel["F"]],
            "df hipotesis": [num(int(v)) for v in tabel["df Hipotesis"]],
            "df galat": [num(int(v)) for v in tabel["df Galat"]],
            "p": ["< 0,001" if v < 0.001 else num(v, 3) for v in tabel["p-value"]],
        }
    )


def tabel_univariat(a: Analisis) -> pd.DataFrame:
    if a.manova is None:
        raise ValueError("Hasil MANOVA tidak tersedia.")
    tabel = a.manova.univariate
    return pd.DataFrame(
        {
            "Variabel dependen": tabel["Variabel"],
            "F": [num(v, 3) for v in tabel["F"]],
            "p": ["< 0,001" if v < 0.001 else num(v, 3) for v in tabel["p-value"]],
            "η²": [num(v, 3) for v in tabel["Eta-squared"]],
            "Besaran efek": [_efek_eta(v) for v in tabel["Eta-squared"]],
        }
    )


def susun_tabel(a: Analisis) -> dict[str, tuple[str, pd.DataFrame, str]]:
    tabel: dict[str, tuple[str, pd.DataFrame, str]] = {}
    nomor = 1
    try:
        tabel[f"Tabel {nomor}"] = (
            f"Statistik deskriptif dan matriks korelasi antarvariabel (N = {num(a.korelasi.n)})",
            tabel_deskriptif_korelasi(a),
            "Korelasi Pearson dua arah. * p < 0,05; ** p < 0,01; *** p < 0,001.",
        )
        nomor += 1
    except Exception:  # noqa: BLE001 - tabel opsional
        pass
    if a.regresi is not None:
        model = a.regresi.model
        tabel[f"Tabel {nomor}"] = (
            f"Hasil regresi linear berganda terhadap {a.regresi.y_name}",
            tabel_regresi(a),
            f"R² = {num(model.rsquared, 3)}; R² adjusted = {num(model.rsquared_adj, 3)}; "
            f"F({num(int(model.df_model))}, {num(int(model.df_resid))}) = "
            f"{num(model.fvalue)}; {pval(model.f_pvalue)}. β adalah koefisien terstandardisasi.",
        )
        nomor += 1
    if a.manova is not None:
        tabel[f"Tabel {nomor}"] = (
            f"Hasil uji MANOVA satu jalur berdasarkan {a.manova.factor}",
            tabel_manova(a),
            "Empat statistik uji multivariat dilaporkan sekaligus; Pillai's trace paling "
            "tahan terhadap pelanggaran asumsi homogenitas kovarians.",
        )
        nomor += 1
        tabel[f"Tabel {nomor}"] = (
            "Uji lanjutan ANOVA univariat per variabel dependen",
            tabel_univariat(a),
            "Ambang besaran efek η²: 0,01 kecil; 0,06 sedang; 0,14 besar (Cohen, 1988).",
        )
        nomor += 1
    asumsi = tabel_asumsi(a)
    if not asumsi.empty:
        tabel[f"Tabel {nomor}"] = (
            "Ringkasan pemeriksaan asumsi statistik",
            asumsi,
            "Ambang mengikuti Hair dkk. (2019) dan Tabachnick & Fidell (2019).",
        )
    return tabel


def susun_paragraf(a: Analisis) -> list[Paragraf]:
    """Paragraf bergaya naskah, siap disalin ke bab metode/hasil/pembahasan."""
    paragraf: list[Paragraf] = []
    n = len(preprocessing.clean_subset(a.df, a.konfig.variabel))
    metode = ["statistik deskriptif"]
    if a.korelasi is not None:
        metode.append("analisis korelasi Pearson")
    if a.pca is not None:
        metode.append("analisis komponen utama")
    if a.klaster is not None:
        metode.append("analisis klaster K-Means")
    if a.regresi is not None:
        metode.append("regresi linear berganda")
    if a.logistik is not None:
        metode.append("regresi logistik biner")
    if a.manova is not None:
        metode.append("MANOVA satu jalur")
    if a.diskriminan is not None:
        metode.append("analisis diskriminan")
    if a.kanonik is not None:
        metode.append("analisis korelasi kanonik")

    paragraf.append(
        Paragraf(
            bagian="Metode",
            teks=(
                f"Data yang dianalisis terdiri atas {num(n)} observasi lengkap dengan "
                f"{len(a.konfig.variabel)} variabel. Analisis dilakukan menggunakan "
                f"{_daftar(metode, len(metode))}. Sebelum pengujian utama, dilakukan "
                "pemeriksaan prasyarat berupa uji normalitas univariat (Shapiro-Wilk), "
                "normalitas multivariat (uji Mardia), deteksi pencilan multivariat "
                "melalui jarak Mahalanobis, serta pemeriksaan multikolinearitas dengan "
                "variance inflation factor. Taraf signifikansi yang digunakan adalah "
                "α = 0,05."
            ),
        )
    )

    if a.mardia is not None:
        paragraf.append(
            Paragraf(
                bagian="Hasil — pemeriksaan prasyarat",
                teks=(
                    f"Uji Mardia menunjukkan koefisien skewness multivariat sebesar "
                    f"{num(a.mardia.skewness, 3)} (χ² = {num(a.mardia.skew_chi2)}; df = "
                    f"{a.mardia.skew_df}; {pval(a.mardia.skew_p)}) dan koefisien kurtosis "
                    f"multivariat sebesar {num(a.mardia.kurtosis, 3)} (z = "
                    f"{num(a.mardia.kurt_z)}; {pval(a.mardia.kurt_p)}), sehingga asumsi "
                    "normalitas multivariat "
                    + ("terpenuhi. " if a.mardia.multivariate_normal else "tidak terpenuhi. ")
                    + (
                        f"Pemeriksaan multikolinearitas menghasilkan VIF maksimum sebesar "
                        f"{num(float(a.vif['VIF'].max()))}, "
                        + (
                            "masih di bawah ambang 10 sehingga tidak terdapat gejala "
                            "multikolinearitas."
                            if float(a.vif["VIF"].max()) < 10
                            else "melampaui ambang 10 sehingga terdapat gejala "
                            "multikolinearitas yang perlu ditangani."
                        )
                        if a.vif is not None and not a.vif.empty
                        else ""
                    )
                ),
            )
        )

    if a.regresi is not None:
        model = a.regresi.model
        koef = a.regresi.coefficients[a.regresi.coefficients["Variabel"] != "const"]
        signifikan = koef[koef["Signifikan"] == "Ya"]
        kalimat = " ".join(
            f"{b['Variabel']} berpengaruh "
            + ("positif" if b["B"] > 0 else "negatif")
            + f" dan signifikan terhadap {a.regresi.y_name} (β = "
            f"{num(b['Beta (baku)'], 3)}; t = {num(b['t'])}; {pval(b['p-value'])})."
            for _, b in signifikan.iterrows()
        )
        paragraf.append(
            Paragraf(
                bagian="Hasil — regresi berganda",
                teks=(
                    f"Model regresi berganda yang diuji signifikan secara statistik, "
                    f"F({num(int(model.df_model))}, {num(int(model.df_resid))}) = "
                    f"{num(model.fvalue)}; {pval(model.f_pvalue)}, dengan koefisien "
                    f"determinasi R² = {num(model.rsquared, 3)} (R² adjusted = "
                    f"{num(model.rsquared_adj, 3)}). Artinya, "
                    f"{pct(model.rsquared * 100)} variasi {a.regresi.y_name} dapat "
                    f"dijelaskan oleh variabel prediktor yang dimasukkan ke dalam model. "
                    + (kalimat if kalimat else "Tidak terdapat prediktor yang signifikan.")
                ),
            )
        )

    if a.manova is not None:
        wilks = a.manova.multivariate.set_index("Statistik").loc["Wilks' lambda"]
        univariat = a.manova.univariate[a.manova.univariate["Signifikan"] == "Ya"]
        lanjutan = " ".join(
            f"Perbedaan signifikan ditemukan pada {b['Variabel']} (F = {num(b['F'])}; "
            f"{pval(b['p-value'])}; η² = {num(b['Eta-squared'], 3)})."
            for _, b in univariat.head(4).iterrows()
        )
        paragraf.append(
            Paragraf(
                bagian="Hasil — uji beda multivariat",
                teks=(
                    f"Hasil MANOVA satu jalur menunjukkan Wilks' lambda sebesar "
                    f"{num(wilks['Nilai'], 4)} dengan F({num(int(wilks['df Hipotesis']))}, "
                    f"{num(int(wilks['df Galat']))}) = {num(wilks['F'])}; "
                    f"{pval(wilks['p-value'])}. "
                    + (
                        f"Dengan demikian terdapat perbedaan yang signifikan pada "
                        f"kombinasi variabel dependen berdasarkan {a.manova.factor}. "
                        if float(wilks["p-value"]) < 0.05
                        else f"Dengan demikian tidak terdapat perbedaan yang signifikan "
                        f"berdasarkan {a.manova.factor}. "
                    )
                    + lanjutan
                ),
            )
        )

    if a.logistik is not None:
        fit = dict(zip(a.logistik.fit["Metrik"], a.logistik.fit["Nilai"]))
        koef = a.logistik.coefficients[a.logistik.coefficients["Variabel"] != "const"]
        signifikan = koef[koef["Signifikan"] == "Ya"]
        kalimat = " ".join(
            f"{b['Variabel']} berpengaruh signifikan dengan odds ratio sebesar "
            f"{num(b['Odds Ratio'], 3)} (IK 95% [{num(b['OR Bawah'], 3)}; "
            f"{num(b['OR Atas'], 3)}]; {pval(b['p-value'])})."
            for _, b in signifikan.iterrows()
        )
        paragraf.append(
            Paragraf(
                bagian="Hasil — regresi logistik",
                teks=(
                    f"Model regresi logistik biner signifikan secara keseluruhan "
                    f"(χ² = {num(fit['Likelihood Ratio Chi-square'])}; "
                    f"{pval(fit['p-value (model)'])}) dengan Nagelkerke R² sebesar "
                    f"{num(fit['Nagelkerke R2'], 3)}. Model mampu mengklasifikasikan "
                    f"{pct(float(a.logistik.performance.iloc[0]['Nilai']) * 100)} kasus "
                    f"secara tepat dengan nilai AUC {num(a.logistik.auc, 3)}. " + kalimat
                ),
            )
        )

    pembuka = "Temuan penelitian ini "
    if a.regresi is not None:
        koef = a.regresi.coefficients[a.regresi.coefficients["Variabel"] != "const"]
        signifikan = koef[koef["Signifikan"] == "Ya"].copy()
        if not signifikan.empty:
            signifikan["abs_beta"] = signifikan["Beta (baku)"].abs()
            utama = signifikan.sort_values("abs_beta", ascending=False).iloc[0]
            pembuka += (
                f"menegaskan peran {utama['Variabel']} sebagai penentu utama "
                f"{a.regresi.y_name}. Pengaruhnya bertanda "
                + ("positif" if utama["B"] > 0 else "negatif")
                + f" dan merupakan yang terkuat di antara seluruh prediktor "
                f"(β = {num(utama['Beta (baku)'], 3)}). "
            )
        else:
            pembuka += (
                f"belum menemukan prediktor yang secara konsisten menjelaskan "
                f"{a.regresi.y_name}. "
            )
        pembuka += (
            f"Namun demikian, model hanya menjelaskan {pct(float(a.regresi.model.rsquared) * 100)} "
            "varians variabel terikat, sehingga sebagian besar variasinya masih ditentukan "
            "oleh faktor di luar model. Hal ini membuka ruang bagi penelitian lanjutan "
            "untuk memasukkan variabel anteseden yang belum terukur."
        )
    else:
        pembuka += (
            "menunjukkan adanya struktur yang konsisten pada data, baik dalam bentuk "
            "keterkaitan antarvariabel maupun pengelompokan observasi. Pola ini perlu "
            "diuji lebih lanjut dengan desain penelitian yang memungkinkan penarikan "
            "kesimpulan kausal."
        )
    paragraf.append(Paragraf(bagian="Pembahasan — kalimat pembuka", teks=pembuka))

    return paragraf


RUJUKAN = [
    "Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). "
    "Lawrence Erlbaum Associates. — ambang besaran efek.",
    "Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019). Multivariate data "
    "analysis (8th ed.). Cengage. — ambang VIF, KMO, rasio sampel per variabel.",
    "Kaiser, H. F. (1974). An index of factorial simplicity. Psychometrika, 39(1), 31–36. "
    "— tafsir nilai KMO.",
    "Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis with "
    "applications. Biometrika, 57(3), 519–530. — uji normalitas multivariat.",
    "Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and "
    "validation of cluster analysis. Journal of Computational and Applied Mathematics, "
    "20, 53–65. — tafsir nilai silhouette.",
    "Tabachnick, B. G., & Fidell, L. S. (2019). Using multivariate statistics (7th ed.). "
    "Pearson. — prosedur pemeriksaan asumsi multivariat.",
]


# --------------------------------------------------------------------------- #
# Tahap 2d — headline dan perakit laporan
# --------------------------------------------------------------------------- #


def susun_headline(a: Analisis, temuan: list[Temuan]) -> tuple[str, str]:
    if a.regresi is not None:
        koef = a.regresi.coefficients[a.regresi.coefficients["Variabel"] != "const"]
        signifikan = koef[koef["Signifikan"] == "Ya"].copy()
        r2 = float(a.regresi.model.rsquared)
        if not signifikan.empty:
            signifikan["abs_beta"] = signifikan["Beta (baku)"].abs()
            utama = signifikan.sort_values("abs_beta", ascending=False).iloc[0]
            arah = "menaikkan" if utama["B"] > 0 else "menurunkan"
            headline = (
                f"{utama['Variabel']} adalah penentu terkuat {a.regresi.y_name} — "
                f"dan arahnya {arah}."
            )
            sub = (
                f"Model menjelaskan {pct(r2 * 100)} naik-turunnya {a.regresi.y_name}; "
                f"{pct((1 - r2) * 100)} sisanya berasal dari faktor yang belum terukur "
                "dalam data ini."
            )
            return headline, sub
        return (
            f"Belum ada faktor yang terbukti menentukan {a.regresi.y_name}.",
            f"Model hanya menjelaskan {pct(r2 * 100)} variasinya, sehingga data yang ada "
            "belum cukup untuk menjadi dasar keputusan.",
        )

    if a.logistik is not None:
        mutu = "layak dipakai sebagai penyaring awal" if a.logistik.auc >= 0.7 else "belum layak dipakai"
        return (
            f"Model prediksi {a.logistik.y_name} {mutu}.",
            f"Kemampuan membedakan dua kelompok berada pada AUC {num(a.logistik.auc, 3)} "
            f"dengan ketepatan {pct(float(a.logistik.performance.iloc[0]['Nilai']) * 100)}.",
        )

    if a.klaster is not None and a.pca is not None:
        k80 = a.pca.components_needed(0.80)
        return (
            f"Data terbagi menjadi {a.klaster.n_clusters} kelompok dengan karakter berbeda.",
            f"Selain itu, {len(a.pca.variables)} ukuran yang dianalisis sebenarnya dapat "
            f"diringkas menjadi {k80} dimensi utama tanpa kehilangan banyak informasi.",
        )

    if temuan:
        return temuan[0].ringkas, temuan[0].eksekutif[:220]
    return "Analisis selesai dijalankan.", ""


def susun_laporan(a: Analisis) -> Laporan:
    """Terjemahkan hasil analisis menjadi laporan tiga register pembaca."""
    laporan = Laporan(
        dataset=a.konfig.nama_data,
        n_baris=len(a.df),
        n_kolom=a.df.shape[1],
        tanggal=date.today().strftime("%d-%m-%Y"),
        konfig=a.konfig,
    )

    pembangun = [
        ("Mutu data", temuan_kualitas_data),
        ("Normalitas", temuan_normalitas),
        ("Korelasi", temuan_korelasi),
        ("PCA", temuan_pca),
        ("Analisis klaster", temuan_klaster),
        ("Regresi linear", temuan_regresi),
        ("Regresi logistik", temuan_logistik),
        ("MANOVA", temuan_perbedaan_kelompok),
        ("Analisis diskriminan", temuan_diskriminan),
        ("Korelasi kanonik", temuan_kanonik),
    ]
    for nama, bangun in pembangun:
        try:
            laporan.temuan.append(bangun(a))
            laporan.metode_terpakai.append(nama)
        except Exception as exc:  # noqa: BLE001 - metode yang gagal dilaporkan apa adanya
            alasan = a.gagal.get(nama, str(exc))
            if nama in a.gagal or "tidak tersedia" not in str(exc):
                laporan.dilewati.append(f"{nama}: {alasan}")

    laporan.lampu = susun_lampu(a)
    laporan.pendorong, laporan.pendorong_sumber = susun_pendorong(a)
    laporan.rekomendasi = susun_rekomendasi(a)
    laporan.keterbatasan = susun_keterbatasan(a)
    laporan.tabel = susun_tabel(a)
    laporan.paragraf = susun_paragraf(a)
    laporan.rujukan = list(RUJUKAN)
    laporan.headline, laporan.subheadline = susun_headline(a, laporan.temuan)
    return laporan


def analisis_dan_laporan(df: pd.DataFrame, konfig: Konfigurasi) -> tuple[Analisis, Laporan]:
    """Jalan pintas: hitung analisis lalu susun laporannya."""
    analisis = jalankan_analisis(df, konfig)
    return analisis, susun_laporan(analisis)
