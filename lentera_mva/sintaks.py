"""Bangkitkan sintaks yang dapat dijalankan ulang di Python dan R.

Aplikasi statistik yang dipakai di lingkungan akademik dituntut *reproducible*:
pembaca harus dapat menjalankan ulang analisis yang sama dan memperoleh angka yang
sama. Modul ini menuliskan langkah analisis yang benar-benar dipilih pengguna
menjadi berkas skrip — Python memakai pustaka yang persis sama dengan yang
dipakai aplikasi, sehingga angkanya identik; R memakai padanan terdekat sehingga
hasilnya dapat diperiksa silang.

Skrip sengaja tidak memuat datanya, hanya jalur berkasnya. Data pengguna tetap
berada di tangan pengguna.
"""

from __future__ import annotations

from datetime import date

from lentera_mva.narrative import Konfigurasi

# Padanan pustaka R untuk tiap langkah, ditulis apa adanya agar pembaca tahu
# mana yang identik dan mana yang sekadar sepadan.
CATATAN_R = (
    "Skrip R memakai padanan terdekat, bukan pustaka yang sama persis. Statistik "
    "uji dan nilai p umumnya sama; perbedaan kecil dapat muncul pada rotasi faktor, "
    "penanganan nilai hilang, dan pemilihan tipe jumlah kuadrat (SS)."
)


def _kutip(teks: object) -> str:
    return '"' + str(teks).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _daftar_py(nilai: list[str]) -> str:
    return "[" + ", ".join(_kutip(v) for v in nilai) + "]"


def _daftar_r(nilai: list[str]) -> str:
    return "c(" + ", ".join(_kutip(v) for v in nilai) + ")"


def _kepala(konfig: Konfigurasi, bahasa: str, komentar: str) -> list[str]:
    judul = "Python" if bahasa == "py" else "R"
    return [
        f"{komentar} Sintaks {judul} — Lentera MVA",
        f"{komentar} Dibangkitkan otomatis pada {date.today().strftime('%d-%m-%Y')}",
        f"{komentar} Sumber data: {konfig.nama_data}",
        f"{komentar}",
        f"{komentar} Skrip ini menjalankan ulang analisis yang Anda pilih di aplikasi.",
        f"{komentar} Sesuaikan JALUR_DATA di bawah dengan letak berkas data Anda.",
        "",
    ]


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #


def sintaks_python(konfig: Konfigurasi) -> str:
    """Skrip Python yang memakai pustaka yang sama dengan aplikasi."""
    b: list[str] = _kepala(konfig, "py", "#")
    b += [
        '"""Jalankan dengan: python sintaks_analisis.py',
        "",
        "Kebutuhan pustaka:",
        "    pip install pandas numpy scipy scikit-learn statsmodels",
        '"""',
        "",
        "import numpy as np",
        "import pandas as pd",
        "from scipy import stats",
        "",
        f"JALUR_DATA = {_kutip(konfig.nama_data)}",
        f"VARIABEL = {_daftar_py(konfig.variabel)}",
        "",
        "df = pd.read_csv(JALUR_DATA)",
        "X = df[VARIABEL].dropna()",
        "print(f'Data terbaca: {len(df)} baris, {df.shape[1]} kolom')",
        "",
        "# ------------------------------------------------------------------ #",
        "# Statistik deskriptif dan uji normalitas (Shapiro-Wilk)",
        "# ------------------------------------------------------------------ #",
        "print(X.describe().T)",
        "for kolom in VARIABEL:",
        "    nilai = df[kolom].dropna()",
        "    if 3 <= len(nilai) <= 5000:",
        "        w, p = stats.shapiro(nilai)",
        "        print(f'{kolom}: W = {w:.4f}, p = {p:.4f}')",
        "",
        "# ------------------------------------------------------------------ #",
        "# Matriks korelasi Pearson beserta nilai p",
        "# ------------------------------------------------------------------ #",
        "korelasi = X.corr()",
        "print(korelasi.round(3))",
        "for i, a in enumerate(VARIABEL):",
        "    for b in VARIABEL[i + 1:]:",
        "        r, p = stats.pearsonr(X[a], X[b])",
        "        print(f'{a} - {b}: r = {r:.3f}, p = {p:.4f}')",
        "",
        "# ------------------------------------------------------------------ #",
        "# Kelayakan reduksi dimensi: Bartlett dan KMO",
        "# ------------------------------------------------------------------ #",
        "n, p_var = X.shape",
        "det = np.linalg.det(korelasi.values)",
        "chi2 = -((n - 1) - (2 * p_var + 5) / 6) * np.log(det)",
        "db = p_var * (p_var - 1) / 2",
        "print(f'Bartlett: chi2 = {chi2:.2f}, db = {db:.0f}, "
        "p = {stats.chi2.sf(chi2, db):.4g}')",
        "",
        "invers = np.linalg.pinv(korelasi.values)",
        "d = np.sqrt(np.diag(invers))",
        "parsial = -invers / np.outer(d, d)",
        "np.fill_diagonal(parsial, 0)",
        "r2 = np.square(korelasi.values); np.fill_diagonal(r2, 0)",
        "print(f'KMO = {r2.sum() / (r2.sum() + np.square(parsial).sum()):.3f}')",
        "",
        "# ------------------------------------------------------------------ #",
        "# PCA pada data terstandar",
        "# ------------------------------------------------------------------ #",
        "from sklearn.decomposition import PCA",
        "from sklearn.preprocessing import StandardScaler",
        "",
        "Xs = StandardScaler().fit_transform(X)",
        "pca = PCA().fit(Xs)",
        "for i, (nilai_eigen, rasio) in enumerate("
        "zip(pca.explained_variance_, pca.explained_variance_ratio_), start=1):",
        "    print(f'PC{i}: eigenvalue = {nilai_eigen:.3f}, "
        "varians = {rasio * 100:.1f}%')",
        "",
        "# ------------------------------------------------------------------ #",
        "# Analisis klaster K-Means (k dipilih lewat nilai siluet tertinggi)",
        "# ------------------------------------------------------------------ #",
        "from sklearn.cluster import KMeans",
        "from sklearn.metrics import silhouette_score",
        "",
        "for k in range(2, 7):",
        "    label = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)",
        "    print(f'k = {k}: siluet = {silhouette_score(Xs, label):.3f}')",
    ]

    if konfig.target_numerik and konfig.prediktor:
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# Regresi linear berganda (OLS)",
            "# ------------------------------------------------------------------ #",
            "import statsmodels.api as sm",
            "from statsmodels.stats.outliers_influence import variance_inflation_factor",
            "",
            f"TARGET = {_kutip(konfig.target_numerik)}",
            f"PREDIKTOR = {_daftar_py(konfig.prediktor)}",
            "",
            "kerja = df[[TARGET] + PREDIKTOR].dropna()",
            "Xr = sm.add_constant(kerja[PREDIKTOR])",
            "model = sm.OLS(kerja[TARGET], Xr).fit()",
            "print(model.summary())",
            "",
            "# Multikolinearitas",
            "for i, nama in enumerate(Xr.columns):",
            "    if nama != 'const':",
            "        print(f'VIF {nama} = "
            "{variance_inflation_factor(Xr.values, i):.2f}')",
            "",
            "# Durbin-Watson dan Breusch-Pagan",
            "from statsmodels.stats.stattools import durbin_watson",
            "from statsmodels.stats.diagnostic import het_breuschpagan",
            "print(f'Durbin-Watson = {durbin_watson(model.resid):.3f}')",
            "lm, lm_p, f, f_p = het_breuschpagan(model.resid, Xr)",
            "print(f'Breusch-Pagan: LM = {lm:.3f}, p = {lm_p:.4f}')",
        ]

    if konfig.target_biner and (konfig.prediktor_biner or konfig.prediktor):
        prediktor = konfig.prediktor_biner or konfig.prediktor
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# Regresi logistik biner",
            "# ------------------------------------------------------------------ #",
            "import statsmodels.api as sm",
            "",
            f"TARGET_BINER = {_kutip(konfig.target_biner)}",
            f"PREDIKTOR_BINER = {_daftar_py(list(prediktor))}",
            "",
            "kerja_b = df[[TARGET_BINER] + PREDIKTOR_BINER].dropna()",
            "y = pd.Categorical(kerja_b[TARGET_BINER]).codes",
            "Xb = sm.add_constant(kerja_b[PREDIKTOR_BINER])",
            "logit = sm.Logit(y, Xb).fit()",
            "print(logit.summary())",
            "print('Odds ratio:')",
            "print(np.exp(logit.params).round(4))",
        ]

    if konfig.kelompok:
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# MANOVA dan analisis diskriminan",
            "# ------------------------------------------------------------------ #",
            "from statsmodels.multivariate.manova import MANOVA",
            "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
            "from sklearn.model_selection import cross_val_score",
            "",
            f"KELOMPOK = {_kutip(konfig.kelompok)}",
            "kerja_m = df[VARIABEL + [KELOMPOK]].dropna()",
            "rumus = ' + '.join(VARIABEL) + ' ~ C(' + KELOMPOK + ')'",
            "print(MANOVA.from_formula(rumus, data=kerja_m).mv_test())",
            "",
            "lda = LinearDiscriminantAnalysis()",
            "skor = cross_val_score(lda, kerja_m[VARIABEL], kerja_m[KELOMPOK], cv=5)",
            "print(f'Ketepatan klasifikasi (validasi silang) = {skor.mean() * 100:.1f}%')",
        ]

    if konfig.gugus_x and konfig.gugus_y:
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# Korelasi kanonik",
            "# ------------------------------------------------------------------ #",
            "from sklearn.cross_decomposition import CCA",
            "",
            f"GUGUS_X = {_daftar_py(konfig.gugus_x)}",
            f"GUGUS_Y = {_daftar_py(konfig.gugus_y)}",
            "kerja_c = df[GUGUS_X + GUGUS_Y].dropna()",
            "n_dim = min(len(GUGUS_X), len(GUGUS_Y))",
            "cca = CCA(n_components=n_dim).fit(kerja_c[GUGUS_X], kerja_c[GUGUS_Y])",
            "u, v = cca.transform(kerja_c[GUGUS_X], kerja_c[GUGUS_Y])",
            "for i in range(n_dim):",
            "    print(f'Fungsi {i + 1}: r kanonik = "
            "{np.corrcoef(u[:, i], v[:, i])[0, 1]:.3f}')",
        ]

    b += [
        "",
        "# ------------------------------------------------------------------ #",
        "# Catatan",
        "# ------------------------------------------------------------------ #",
        "# Skrip ini memakai pustaka yang sama dengan aplikasi Lentera MVA",
        "# (pandas, numpy, scipy, scikit-learn, statsmodels), sehingga angkanya",
        "# identik dengan yang ditampilkan di layar.",
        "",
    ]
    return "\n".join(b)


# --------------------------------------------------------------------------- #
# R
# --------------------------------------------------------------------------- #


def sintaks_r(konfig: Konfigurasi) -> str:
    """Skrip R sebagai pemeriksaan silang di luar Python."""
    b: list[str] = _kepala(konfig, "r", "#")
    b += [
        "# Kebutuhan paket:",
        '#   install.packages(c("psych", "car", "MASS", "candisc", "CCA", "lmtest"))',
        "",
        f"# {CATATAN_R}",
        "",
        f"JALUR_DATA <- {_kutip(konfig.nama_data)}",
        f"VARIABEL   <- {_daftar_r(konfig.variabel)}",
        "",
        "df <- read.csv(JALUR_DATA, stringsAsFactors = TRUE)",
        "X  <- na.omit(df[, VARIABEL])",
        "cat(sprintf('Data terbaca: %d baris, %d kolom\\n', nrow(df), ncol(df)))",
        "",
        "# ------------------------------------------------------------------ #",
        "# Deskriptif dan uji normalitas",
        "# ------------------------------------------------------------------ #",
        "summary(X)",
        "for (kolom in VARIABEL) {",
        "  nilai <- na.omit(df[[kolom]])",
        "  if (length(nilai) >= 3 && length(nilai) <= 5000) {",
        "    uji <- shapiro.test(nilai)",
        "    cat(sprintf('%s: W = %.4f, p = %.4f\\n', kolom, uji$statistic, uji$p.value))",
        "  }",
        "}",
        "",
        "# ------------------------------------------------------------------ #",
        "# Korelasi, Bartlett, dan KMO",
        "# ------------------------------------------------------------------ #",
        "library(psych)",
        "print(round(cor(X), 3))",
        "print(corr.test(X))",
        "print(cortest.bartlett(cor(X), n = nrow(X)))",
        "print(KMO(X))",
        "",
        "# ------------------------------------------------------------------ #",
        "# PCA pada data terstandar",
        "# ------------------------------------------------------------------ #",
        "pca <- prcomp(X, center = TRUE, scale. = TRUE)",
        "print(summary(pca))",
        "cat('Eigenvalue:\\n'); print(round(pca$sdev^2, 3))",
        "",
        "# ------------------------------------------------------------------ #",
        "# Analisis klaster K-Means",
        "# ------------------------------------------------------------------ #",
        "Xs <- scale(X)",
        "set.seed(42)",
        "for (k in 2:6) {",
        "  km <- kmeans(Xs, centers = k, nstart = 10)",
        "  cat(sprintf('k = %d: rasio between/total = %.3f\\n', k,",
        "              km$betweenss / km$totss))",
        "}",
        "# Nilai siluet: library(cluster); summary(silhouette(km$cluster, dist(Xs)))",
    ]

    if konfig.target_numerik and konfig.prediktor:
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# Regresi linear berganda",
            "# ------------------------------------------------------------------ #",
            "library(car); library(lmtest)",
            f"TARGET    <- {_kutip(konfig.target_numerik)}",
            f"PREDIKTOR <- {_daftar_r(konfig.prediktor)}",
            "",
            "rumus <- as.formula(paste(TARGET, '~', paste(PREDIKTOR, collapse = ' + ')))",
            "model <- lm(rumus, data = df)",
            "print(summary(model))",
            "cat('VIF:\\n'); print(vif(model))",
            "print(durbinWatsonTest(model))",
            "print(bptest(model))  # Breusch-Pagan",
        ]

    if konfig.target_biner and (konfig.prediktor_biner or konfig.prediktor):
        prediktor = konfig.prediktor_biner or konfig.prediktor
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# Regresi logistik biner",
            "# ------------------------------------------------------------------ #",
            f"TARGET_BINER    <- {_kutip(konfig.target_biner)}",
            f"PREDIKTOR_BINER <- {_daftar_r(list(prediktor))}",
            "",
            "rumus_b <- as.formula(paste(TARGET_BINER, '~',",
            "                            paste(PREDIKTOR_BINER, collapse = ' + ')))",
            "logit <- glm(rumus_b, data = df, family = binomial())",
            "print(summary(logit))",
            "cat('Odds ratio:\\n'); print(round(exp(coef(logit)), 4))",
        ]

    if konfig.kelompok:
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# MANOVA dan analisis diskriminan",
            "# ------------------------------------------------------------------ #",
            "library(MASS)",
            f"KELOMPOK <- {_kutip(konfig.kelompok)}",
            "rumus_m <- as.formula(paste('cbind(', paste(VARIABEL, collapse = ', '),",
            "                            ') ~', KELOMPOK))",
            "fit <- manova(rumus_m, data = df)",
            "print(summary(fit, test = 'Wilks'))",
            "print(summary(fit, test = 'Pillai'))",
            "print(summary.aov(fit))",
            "",
            "rumus_d <- as.formula(paste(KELOMPOK, '~', paste(VARIABEL, collapse = ' + ')))",
            "lda_fit <- lda(rumus_d, data = df, CV = TRUE)",
            "tabel <- table(df[[KELOMPOK]], lda_fit$class)",
            "cat(sprintf('Ketepatan klasifikasi = %.1f%%\\n',",
            "            100 * sum(diag(tabel)) / sum(tabel)))",
            "# Uji Box's M: library(biotools); boxM(df[, VARIABEL], df[[KELOMPOK]])",
        ]

    if konfig.gugus_x and konfig.gugus_y:
        b += [
            "",
            "# ------------------------------------------------------------------ #",
            "# Korelasi kanonik",
            "# ------------------------------------------------------------------ #",
            "library(CCA)",
            f"GUGUS_X <- {_daftar_r(konfig.gugus_x)}",
            f"GUGUS_Y <- {_daftar_r(konfig.gugus_y)}",
            "kerja <- na.omit(df[, c(GUGUS_X, GUGUS_Y)])",
            "hasil <- cc(kerja[, GUGUS_X], kerja[, GUGUS_Y])",
            "cat('Korelasi kanonik:\\n'); print(round(hasil$cor, 3))",
        ]

    b += ["", f"# {CATATAN_R}", ""]
    return "\n".join(b)


def bangkitkan(konfig: Konfigurasi | None, bahasa: str) -> str:
    """Skrip untuk bahasa yang diminta; konfigurasi kosong menghasilkan penjelasan."""
    if konfig is None:
        return (
            "# Sintaks tidak dapat dibangkitkan: konfigurasi analisis tidak tersimpan\n"
            "# pada laporan ini. Jalankan ulang analisis dari halaman ringkasan.\n"
        )
    if bahasa == "py":
        return sintaks_python(konfig)
    if bahasa == "r":
        return sintaks_r(konfig)
    raise ValueError(f"Bahasa '{bahasa}' tidak dikenal. Pilih 'py' atau 'r'.")
