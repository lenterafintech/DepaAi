"""Uji non-parametrik: alternatif ketika asumsi normalitas tidak terpenuhi.

Uji-t, ANOVA, dan korelasi Pearson menuntut sebaran yang mendekati normal. Ketika
tuntutan itu tidak terpenuhi — data ordinal, sampel kecil, atau sebaran yang jelas
menceng — kesimpulannya menjadi rapuh. Uji di modul ini bekerja pada peringkat,
bukan nilai mentahnya, sehingga sah dipakai tanpa asumsi sebaran tertentu.

Tiap uji dilaporkan lengkap: statistik uji, nilai p, ukuran efek, dan padanan
parametriknya, karena penelaah menanyakan ketiganya.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# Padanan parametrik tiap uji, ditampilkan agar pembaca tahu uji mana yang
# digantikan dan mengapa.
PADANAN = {
    "mann_whitney": "Uji-t sampel bebas",
    "wilcoxon": "Uji-t sampel berpasangan",
    "kruskal": "ANOVA satu arah",
    "friedman": "ANOVA pengukuran berulang",
    "chi2": "—",
    "fisher": "—",
    "spearman": "Korelasi Pearson",
    "kendall": "Korelasi Pearson",
    "ks2": "Uji-t sampel bebas",
    "tanda": "Uji-t sampel berpasangan",
    "mood": "Uji Levene",
}


@dataclass
class HasilUji:
    """Satu uji non-parametrik beserta pelaporannya."""

    nama: str
    kode: str
    statistik: float
    p_value: float
    n: int
    label_statistik: str = "Statistik"
    efek_nama: str = ""
    efek_nilai: float = float("nan")
    efek_tafsir: str = ""
    keterangan: str = ""
    tabel: pd.DataFrame | None = None
    catatan: list[str] = field(default_factory=list)

    @property
    def signifikan(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)

    @property
    def padanan(self) -> str:
        return PADANAN.get(self.kode, "—")

    def ringkas(self) -> str:
        """Kalimat pelaporan siap salin."""
        from nalardata.formatting import num, pval

        # Statistik yang berupa cacahan ditulis tanpa desimal agar tidak
        # terbaca seolah-olah hasil pengukuran.
        desimal = 0 if float(self.statistik).is_integer() else 3
        bagian = (
            f"{self.label_statistik} = {num(self.statistik, desimal)}, "
            f"{pval(self.p_value)}"
        )
        if np.isfinite(self.efek_nilai):
            bagian += f", {self.efek_nama} = {num(self.efek_nilai, 3)}"
        putusan = "berbeda bermakna" if self.signifikan else "tidak berbeda bermakna"
        return f"{self.nama}: {bagian} — {putusan} pada taraf 5%."


# --------------------------------------------------------------------------- #
# Penafsir ukuran efek
# --------------------------------------------------------------------------- #


def _tafsir_r(nilai: float) -> str:
    """Ambang Cohen (1988) untuk ukuran efek berbasis korelasi."""
    besar = abs(nilai)
    if not np.isfinite(besar):
        return "tidak dapat dihitung"
    if besar < 0.10:
        return "dapat diabaikan"
    if besar < 0.30:
        return "kecil"
    if besar < 0.50:
        return "sedang"
    return "besar"


def _tafsir_eta(nilai: float) -> str:
    """Ambang epsilon²/eta² (Cohen, 1988)."""
    if not np.isfinite(nilai):
        return "tidak dapat dihitung"
    if nilai < 0.01:
        return "dapat diabaikan"
    if nilai < 0.06:
        return "kecil"
    if nilai < 0.14:
        return "sedang"
    return "besar"


def _tafsir_w(nilai: float) -> str:
    """Ambang Kendall's W sebagai derajat kesepakatan peringkat."""
    if not np.isfinite(nilai):
        return "tidak dapat dihitung"
    if nilai < 0.1:
        return "kesepakatan sangat lemah"
    if nilai < 0.3:
        return "kesepakatan lemah"
    if nilai < 0.5:
        return "kesepakatan sedang"
    return "kesepakatan kuat"


def _bersih(*deret: pd.Series) -> list[np.ndarray]:
    """Buang baris yang salah satu nilainya hilang, sejajar antar deret."""
    bingkai = pd.concat([pd.Series(d).reset_index(drop=True) for d in deret], axis=1)
    bingkai = bingkai.apply(pd.to_numeric, errors="coerce").dropna()
    return [bingkai.iloc[:, i].to_numpy(dtype=float) for i in range(bingkai.shape[1])]


def _deskriptif(kelompok: dict[str, np.ndarray]) -> pd.DataFrame:
    """Ringkasan tiap kelompok: n, median, kuartil, dan rata-rata peringkat."""
    semua = np.concatenate(list(kelompok.values()))
    peringkat = stats.rankdata(semua)
    baris = []
    mulai = 0
    for nama, nilai in kelompok.items():
        selesai = mulai + len(nilai)
        baris.append(
            {
                "Kelompok": nama,
                "n": len(nilai),
                "Median": float(np.median(nilai)) if len(nilai) else float("nan"),
                "Kuartil 1": float(np.percentile(nilai, 25)) if len(nilai) else float("nan"),
                "Kuartil 3": float(np.percentile(nilai, 75)) if len(nilai) else float("nan"),
                "Rata-rata peringkat": float(np.mean(peringkat[mulai:selesai]))
                if len(nilai)
                else float("nan"),
            }
        )
        mulai = selesai
    return pd.DataFrame(baris)


# --------------------------------------------------------------------------- #
# Dua kelompok bebas
# --------------------------------------------------------------------------- #


def mann_whitney(a: pd.Series, b: pd.Series, nama_a: str = "A", nama_b: str = "B") -> HasilUji:
    """Uji Mann-Whitney U: dua kelompok bebas, padanan uji-t sampel bebas."""
    x, = _bersih(a)
    y, = _bersih(b)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Tiap kelompok memerlukan minimal 2 pengamatan.")

    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    # Rank-biserial correlation: selisih peluang satu kelompok melampaui kelompok lain.
    rb = 2 * u / (len(x) * len(y)) - 1
    catatan = []
    if len(x) < 20 or len(y) < 20:
        catatan.append(
            "Salah satu kelompok berisi kurang dari 20 pengamatan; nilai p dihitung "
            "dengan hampiran normal dan sebaiknya ditafsirkan hati-hati."
        )
    return HasilUji(
        nama=f"Mann-Whitney U ({nama_a} vs {nama_b})",
        kode="mann_whitney",
        statistik=float(u),
        p_value=float(p),
        n=len(x) + len(y),
        label_statistik="U",
        efek_nama="r rank-biserial",
        efek_nilai=float(rb),
        efek_tafsir=_tafsir_r(rb),
        keterangan=(
            f"Median {nama_a} = {np.median(x):.4g}; median {nama_b} = {np.median(y):.4g}."
        ),
        tabel=_deskriptif({nama_a: x, nama_b: y}),
        catatan=catatan,
    )


def kolmogorov_smirnov_2(
    a: pd.Series, b: pd.Series, nama_a: str = "A", nama_b: str = "B"
) -> HasilUji:
    """Uji KS dua sampel: menguji kesamaan bentuk sebaran, bukan sekadar pusatnya."""
    x, = _bersih(a)
    y, = _bersih(b)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Tiap kelompok memerlukan minimal 2 pengamatan.")
    d, p = stats.ks_2samp(x, y)
    return HasilUji(
        nama=f"Kolmogorov-Smirnov dua sampel ({nama_a} vs {nama_b})",
        kode="ks2",
        statistik=float(d),
        p_value=float(p),
        n=len(x) + len(y),
        label_statistik="D",
        efek_nama="D",
        efek_nilai=float(d),
        efek_tafsir=_tafsir_r(d),
        keterangan=(
            "Menguji apakah kedua sebaran berbeda secara keseluruhan — letak, sebaran, "
            "maupun bentuknya."
        ),
        tabel=_deskriptif({nama_a: x, nama_b: y}),
    )


def mood_median(
    a: pd.Series, b: pd.Series, nama_a: str = "A", nama_b: str = "B"
) -> HasilUji:
    """Uji median Mood: membandingkan proporsi yang berada di atas median gabungan."""
    x, = _bersih(a)
    y, = _bersih(b)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Tiap kelompok memerlukan minimal 2 pengamatan.")
    stat, p, median, tabel = stats.median_test(x, y)
    isi = pd.DataFrame(
        tabel,
        index=["Di atas median", "Di bawah / sama dengan median"],
        columns=[nama_a, nama_b],
    ).reset_index(names="Posisi")
    return HasilUji(
        nama=f"Uji median Mood ({nama_a} vs {nama_b})",
        kode="mood",
        statistik=float(stat),
        p_value=float(p),
        n=len(x) + len(y),
        label_statistik="chi-square",
        keterangan=f"Median gabungan = {median:.4g}.",
        tabel=isi,
    )


# --------------------------------------------------------------------------- #
# Lebih dari dua kelompok bebas
# --------------------------------------------------------------------------- #


def kruskal_wallis(nilai: pd.Series, kelompok: pd.Series) -> HasilUji:
    """Uji Kruskal-Wallis: tiga kelompok bebas atau lebih, padanan ANOVA satu arah."""
    bingkai = pd.DataFrame({"nilai": pd.to_numeric(nilai, errors="coerce"), "grup": kelompok})
    bingkai = bingkai.dropna()
    kelompok_isi = {
        str(nama): bagian["nilai"].to_numpy(dtype=float)
        for nama, bagian in bingkai.groupby("grup", observed=True)
        if len(bagian) >= 2
    }
    if len(kelompok_isi) < 2:
        raise ValueError("Diperlukan minimal 2 kelompok yang masing-masing berisi 2 data.")

    h, p = stats.kruskal(*kelompok_isi.values())
    n = int(sum(len(v) for v in kelompok_isi.values()))
    k = len(kelompok_isi)
    # Epsilon kuadrat: proporsi keragaman peringkat yang dijelaskan kelompok.
    epsilon2 = (h - k + 1) / (n - k) if n > k else float("nan")
    return HasilUji(
        nama=f"Kruskal-Wallis ({k} kelompok)",
        kode="kruskal",
        statistik=float(h),
        p_value=float(p),
        n=n,
        label_statistik="H",
        efek_nama="epsilon kuadrat",
        efek_nilai=float(epsilon2),
        efek_tafsir=_tafsir_eta(epsilon2),
        keterangan=f"Derajat bebas = {k - 1}.",
        tabel=_deskriptif(kelompok_isi),
        catatan=(
            ["Uji ini hanya menyatakan ada perbedaan, belum menunjuk kelompok mana; "
             "lihat uji lanjutan Dunn."]
            if p < 0.05
            else []
        ),
    )


def dunn(nilai: pd.Series, kelompok: pd.Series, koreksi: str = "holm") -> pd.DataFrame:
    """Uji lanjutan Dunn: pasangan kelompok mana yang berbeda, dengan koreksi ganda.

    Nilai p mentah dikoreksi karena membandingkan banyak pasangan sekaligus
    menaikkan peluang temuan palsu; Holm dipakai sebagai bawaan karena lebih kuat
    daripada Bonferroni tanpa mengorbankan kendali galat jenis I.
    """
    bingkai = pd.DataFrame({"nilai": pd.to_numeric(nilai, errors="coerce"), "grup": kelompok})
    bingkai = bingkai.dropna()
    bingkai["peringkat"] = stats.rankdata(bingkai["nilai"])
    ringkas = bingkai.groupby("grup", observed=True)["peringkat"].agg(["mean", "count"])
    ringkas = ringkas[ringkas["count"] >= 1]
    n = int(ringkas["count"].sum())
    if len(ringkas) < 2:
        raise ValueError("Diperlukan minimal 2 kelompok.")

    # Koreksi angka kembar pada peringkat.
    _, jumlah = np.unique(bingkai["nilai"], return_counts=True)
    ikat = float(np.sum(jumlah**3 - jumlah))
    sigma_dasar = (n * (n + 1) / 12) - ikat / (12 * (n - 1)) if n > 1 else float("nan")

    nama = list(ringkas.index)
    baris = []
    for i, a in enumerate(nama):
        for b in nama[i + 1 :]:
            beda = ringkas.loc[a, "mean"] - ringkas.loc[b, "mean"]
            galat = np.sqrt(sigma_dasar * (1 / ringkas.loc[a, "count"] + 1 / ringkas.loc[b, "count"]))
            z = beda / galat if galat else float("nan")
            baris.append(
                {
                    "Kelompok A": str(a),
                    "Kelompok B": str(b),
                    "Selisih rata-rata peringkat": float(beda),
                    "z": float(z),
                    "p mentah": float(2 * stats.norm.sf(abs(z))),
                }
            )

    hasil = pd.DataFrame(baris)
    hasil[f"p ({koreksi})"] = _koreksi_ganda(hasil["p mentah"].to_numpy(), koreksi)
    hasil["Keputusan"] = np.where(
        hasil[f"p ({koreksi})"] < 0.05, "Berbeda bermakna", "Tidak berbeda"
    )
    return hasil.sort_values(f"p ({koreksi})").reset_index(drop=True)


def _koreksi_ganda(p: np.ndarray, metode: str = "holm") -> np.ndarray:
    """Koreksi nilai p untuk pembandingan ganda (Holm, Bonferroni, atau tanpa koreksi)."""
    m = len(p)
    if metode == "bonferroni":
        return np.minimum(p * m, 1.0)
    if metode == "tanpa":
        return p
    # Holm-Bonferroni: ambang menyesuaikan urutan nilai p, lalu dijaga tetap monoton.
    urut = np.argsort(p)
    disesuaikan = np.empty(m, dtype=float)
    berjalan = 0.0
    for pangkat, indeks in enumerate(urut):
        nilai = (m - pangkat) * p[indeks]
        berjalan = max(berjalan, nilai)
        disesuaikan[indeks] = min(berjalan, 1.0)
    return disesuaikan


# --------------------------------------------------------------------------- #
# Sampel berpasangan
# --------------------------------------------------------------------------- #


def wilcoxon(a: pd.Series, b: pd.Series, nama_a: str = "Sebelum", nama_b: str = "Sesudah") -> HasilUji:
    """Uji Wilcoxon peringkat bertanda: dua pengukuran pada subjek yang sama."""
    x, y = _bersih(a, b)
    if len(x) < 5:
        raise ValueError("Uji Wilcoxon memerlukan minimal 5 pasangan lengkap.")
    beda = x - y
    if np.all(beda == 0):
        raise ValueError("Seluruh pasangan bernilai sama, tidak ada yang dapat diuji.")

    w, p = stats.wilcoxon(x, y)
    n = len(x)
    # Ukuran efek r = z / akar(n), dengan z dari hampiran normal statistik W.
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sigma if sigma else float("nan")
    r = z / np.sqrt(n) if n else float("nan")
    naik = int(np.sum(beda > 0))
    turun = int(np.sum(beda < 0))
    return HasilUji(
        nama=f"Wilcoxon peringkat bertanda ({nama_a} vs {nama_b})",
        kode="wilcoxon",
        statistik=float(w),
        p_value=float(p),
        n=n,
        label_statistik="W",
        efek_nama="r",
        efek_nilai=float(abs(r)),
        efek_tafsir=_tafsir_r(r),
        keterangan=(
            f"{naik} pasangan naik, {turun} turun, {n - naik - turun} tidak berubah. "
            f"Median selisih = {np.median(beda):.4g}."
        ),
        tabel=pd.DataFrame(
            {
                "Pengukuran": [nama_a, nama_b, "Selisih"],
                "n": [n, n, n],
                "Median": [float(np.median(x)), float(np.median(y)), float(np.median(beda))],
                "Kuartil 1": [
                    float(np.percentile(x, 25)),
                    float(np.percentile(y, 25)),
                    float(np.percentile(beda, 25)),
                ],
                "Kuartil 3": [
                    float(np.percentile(x, 75)),
                    float(np.percentile(y, 75)),
                    float(np.percentile(beda, 75)),
                ],
            }
        ),
    )


def uji_tanda(a: pd.Series, b: pd.Series, nama_a: str = "Sebelum", nama_b: str = "Sesudah") -> HasilUji:
    """Uji tanda: hanya arah perubahan yang dihitung, besarnya diabaikan."""
    x, y = _bersih(a, b)
    beda = x - y
    naik = int(np.sum(beda > 0))
    turun = int(np.sum(beda < 0))
    pasangan = naik + turun
    if pasangan == 0:
        raise ValueError("Tidak ada pasangan yang berubah, uji tanda tidak dapat dijalankan.")
    p = float(stats.binomtest(naik, pasangan, 0.5).pvalue)
    proporsi = naik / pasangan
    return HasilUji(
        nama=f"Uji tanda ({nama_a} vs {nama_b})",
        kode="tanda",
        statistik=float(naik),
        p_value=p,
        n=pasangan,
        label_statistik="Jumlah naik",
        efek_nama="Proporsi naik",
        efek_nilai=float(proporsi),
        efek_tafsir=_tafsir_r(proporsi - 0.5),
        keterangan=(
            f"{naik} naik, {turun} turun, {len(beda) - pasangan} tidak berubah "
            "(pasangan tanpa perubahan tidak diikutkan)."
        ),
    )


def friedman(data: pd.DataFrame, kolom: list[str]) -> HasilUji:
    """Uji Friedman: tiga pengukuran berulang atau lebih pada subjek yang sama."""
    if len(kolom) < 3:
        raise ValueError("Uji Friedman memerlukan minimal 3 pengukuran.")
    kerja = data[kolom].apply(pd.to_numeric, errors="coerce").dropna()
    if len(kerja) < 3:
        raise ValueError("Diperlukan minimal 3 subjek dengan data lengkap.")

    chi2, p = stats.friedmanchisquare(*[kerja[k].to_numpy(dtype=float) for k in kolom])
    n, k = len(kerja), len(kolom)
    # Kendall's W: chi-square dinormalkan menjadi skala 0–1.
    w = chi2 / (n * (k - 1)) if n and k > 1 else float("nan")
    peringkat = kerja.rank(axis=1)
    return HasilUji(
        nama=f"Friedman ({k} pengukuran berulang)",
        kode="friedman",
        statistik=float(chi2),
        p_value=float(p),
        n=n,
        label_statistik="chi-square",
        efek_nama="Kendall's W",
        efek_nilai=float(w),
        efek_tafsir=_tafsir_w(w),
        keterangan=f"Derajat bebas = {k - 1}; {n} subjek berdata lengkap.",
        tabel=pd.DataFrame(
            {
                "Pengukuran": kolom,
                "Median": [float(kerja[k_].median()) for k_ in kolom],
                "Rata-rata peringkat": [float(peringkat[k_].mean()) for k_ in kolom],
            }
        ),
    )


# --------------------------------------------------------------------------- #
# Data kategorik
# --------------------------------------------------------------------------- #


def chi_square(a: pd.Series, b: pd.Series, nama_a: str = "Baris", nama_b: str = "Kolom") -> HasilUji:
    """Uji chi-square kebebasan dua variabel kategorik, dengan Cramér's V."""
    bingkai = pd.DataFrame({nama_a: a, nama_b: b}).dropna()
    if bingkai.empty:
        raise ValueError("Tidak ada baris lengkap untuk diuji.")
    silang = pd.crosstab(bingkai[nama_a], bingkai[nama_b])
    if silang.shape[0] < 2 or silang.shape[1] < 2:
        raise ValueError("Kedua variabel harus memiliki minimal 2 kategori.")

    chi2, p, db, harapan = stats.chi2_contingency(silang)
    n = int(silang.to_numpy().sum())
    # Cramér's V dengan koreksi Bergsma untuk tabel kecil dihindari agar tetap
    # sebanding dengan angka yang dilaporkan perangkat lain.
    v = np.sqrt(chi2 / (n * (min(silang.shape) - 1))) if n else float("nan")
    kecil = int(np.sum(harapan < 5))
    catatan = []
    if kecil:
        catatan.append(
            f"{kecil} sel memiliki frekuensi harapan di bawah 5 "
            f"({kecil / harapan.size * 100:.0f}% dari seluruh sel). Chi-square menjadi "
            "kurang dapat diandalkan; pertimbangkan menggabungkan kategori atau "
            "memakai uji eksak Fisher."
        )
    return HasilUji(
        nama=f"Chi-square kebebasan ({nama_a} × {nama_b})",
        kode="chi2",
        statistik=float(chi2),
        p_value=float(p),
        n=n,
        label_statistik="chi-square",
        efek_nama="Cramér's V",
        efek_nilai=float(v),
        efek_tafsir=_tafsir_r(v),
        keterangan=f"Derajat bebas = {db}; tabel {silang.shape[0]}×{silang.shape[1]}.",
        tabel=silang.reset_index(),
        catatan=catatan,
    )


def fisher_eksak(a: pd.Series, b: pd.Series, nama_a: str = "Baris", nama_b: str = "Kolom") -> HasilUji:
    """Uji eksak Fisher untuk tabel 2×2, sah pada frekuensi kecil sekalipun."""
    bingkai = pd.DataFrame({nama_a: a, nama_b: b}).dropna()
    silang = pd.crosstab(bingkai[nama_a], bingkai[nama_b])
    if silang.shape != (2, 2):
        raise ValueError(
            f"Uji eksak Fisher hanya untuk tabel 2×2; tabel yang terbentuk "
            f"{silang.shape[0]}×{silang.shape[1]}."
        )
    odds, p = stats.fisher_exact(silang.to_numpy())
    return HasilUji(
        nama=f"Uji eksak Fisher ({nama_a} × {nama_b})",
        kode="fisher",
        statistik=float(odds),
        p_value=float(p),
        n=int(silang.to_numpy().sum()),
        label_statistik="Odds ratio",
        efek_nama="Odds ratio",
        efek_nilai=float(odds),
        efek_tafsir=(
            "tidak ada kaitan"
            if not np.isfinite(odds) or abs(odds - 1) < 0.1
            else ("kaitan positif" if odds > 1 else "kaitan negatif")
        ),
        keterangan="Nilai p dihitung secara eksak, bukan hampiran.",
        tabel=silang.reset_index(),
    )


# --------------------------------------------------------------------------- #
# Korelasi peringkat
# --------------------------------------------------------------------------- #


def korelasi_peringkat(a: pd.Series, b: pd.Series, metode: str = "spearman",
                       nama_a: str = "X", nama_b: str = "Y") -> HasilUji:
    """Korelasi Spearman atau Kendall: kaitan monoton tanpa asumsi normalitas."""
    x, y = _bersih(a, b)
    if len(x) < 3:
        raise ValueError("Diperlukan minimal 3 pasangan lengkap.")
    if metode == "kendall":
        koef, p = stats.kendalltau(x, y)
        nama = f"Kendall's tau ({nama_a} — {nama_b})"
        label = "tau"
    else:
        koef, p = stats.spearmanr(x, y)
        nama = f"Spearman rho ({nama_a} — {nama_b})"
        label = "rho"
    return HasilUji(
        nama=nama,
        kode="spearman" if metode != "kendall" else "kendall",
        statistik=float(koef),
        p_value=float(p),
        n=len(x),
        label_statistik=label,
        efek_nama=label,
        efek_nilai=float(koef),
        efek_tafsir=_tafsir_r(koef),
        keterangan=(
            "Mengukur keeratan hubungan yang searah (monoton), bukan harus lurus "
            "seperti korelasi Pearson."
        ),
    )


def matriks_peringkat(df: pd.DataFrame, kolom: list[str], metode: str = "spearman") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriks korelasi peringkat beserta matriks nilai p-nya."""
    kerja = df[kolom].apply(pd.to_numeric, errors="coerce").dropna()
    koefisien = kerja.corr(method="spearman" if metode != "kendall" else "kendall")
    p = pd.DataFrame(np.nan, index=kolom, columns=kolom, dtype=float)
    for i, a in enumerate(kolom):
        p.loc[a, a] = 0.0
        for b in kolom[i + 1 :]:
            uji = korelasi_peringkat(kerja[a], kerja[b], metode, a, b)
            p.loc[a, b] = p.loc[b, a] = uji.p_value
    return koefisien, p


# --------------------------------------------------------------------------- #
# Saran uji
# --------------------------------------------------------------------------- #


def perlu_nonparametrik(df: pd.DataFrame, kolom: list[str]) -> tuple[bool, str]:
    """Apakah data ini lebih tepat diuji tanpa asumsi normalitas?"""
    ada = [k for k in kolom if k in df.columns]
    kerja = df[ada].apply(pd.to_numeric, errors="coerce").dropna()
    if kerja.empty or len(kerja) < 3:
        return True, "Data terlalu sedikit untuk mengandalkan asumsi normalitas."

    menceng = kerja.skew().abs()
    tak_normal = []
    for k in kerja.columns:
        nilai = kerja[k].to_numpy(dtype=float)
        if 3 <= len(nilai) <= 5000:
            if float(stats.shapiro(nilai).pvalue) < 0.05:
                tak_normal.append(k)
    if len(kerja) < 30:
        return True, (
            f"Hanya {len(kerja)} baris lengkap. Pada sampel sekecil ini uji parametrik "
            "bertumpu penuh pada asumsi normalitas yang sulit diperiksa."
        )
    if tak_normal:
        return True, (
            f"{len(tak_normal)} dari {len(kerja.columns)} variabel menyimpang dari "
            f"normal menurut Shapiro-Wilk ({', '.join(tak_normal[:4])}"
            f"{' dan lainnya' if len(tak_normal) > 4 else ''})."
        )
    if float(menceng.max()) > 1:
        return True, (
            f"Kemencengan tertinggi {float(menceng.max()):.2f} melebihi 1, sebaran "
            "jelas tidak simetris."
        )
    return False, (
        "Sebaran variabel yang dipilih cukup mendekati normal; uji parametrik masih "
        "sah dipakai. Uji di halaman ini tetap dapat dijalankan sebagai pembanding."
    )
