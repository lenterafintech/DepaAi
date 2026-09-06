"""Grafik statis untuk disisipkan ke berkas laporan.

Grafik di layar dibuat dengan Plotly, tetapi Plotly memerlukan paket tambahan
(`kaleido`) untuk menghasilkan gambar. Berkas Word, PDF, dan PowerPoint hanya
menerima gambar, sehingga grafik untuk laporan digambar ulang memakai matplotlib —
menghindari satu dependensi tanpa mengorbankan apa pun, karena media cetak memang
tidak punya interaksi yang hilang.

Warnanya mengikuti palet yang sama dengan grafik di layar (``plots.py``), yang sudah
lolos pemeriksaan keterbacaan bagi pembaca dengan buta warna.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from nalardata.plots import ACUAN, GARIS, TINTA, TINTA_REDUP

# Kutub warna untuk besaran yang punya arah. Bukan warna seri: biru berarti
# menaikkan, oranye menurunkan, abu-abu berarti belum terbukti.
KUTUB_NAIK = "#2a78d6"
KUTUB_TURUN = "#eb6834"
NETRAL = "#9aa5b8"

LEBAR = 9.0  # inci, pas untuk halaman A4 berorientasi tegak
DPI = 200  # cukup tajam untuk dicetak


def _siapkan():
    """Muat matplotlib dengan backend tanpa layar; kegagalan dilaporkan jelas."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - hanya bila paket dicopot
        raise RuntimeError(
            "Menyisipkan grafik ke laporan memerlukan matplotlib. Pasang dengan "
            "'pip install matplotlib', atau ekspor laporan tanpa grafik."
        ) from exc
    matplotlib.use("Agg")  # tanpa ini matplotlib mencari layar dan gagal di server
    import matplotlib.pyplot as plt

    return plt


def _sumbu_indonesia(sumbu, arah: str = "x") -> None:
    """Pemisah desimal pada sumbu dibuat koma agar seragam dengan label lainnya."""
    from matplotlib.ticker import FuncFormatter

    format_koma = FuncFormatter(lambda nilai, _: f"{nilai:g}".replace(".", ","))
    if arah == "x":
        sumbu.xaxis.set_major_formatter(format_koma)
    else:
        sumbu.yaxis.set_major_formatter(format_koma)


def _rapikan(sumbu, plt) -> None:
    """Sumbu dan garis bantu dibuat tenang agar datanya yang menonjol."""
    for sisi in ("top", "right"):
        sumbu.spines[sisi].set_visible(False)
    for sisi in ("left", "bottom"):
        sumbu.spines[sisi].set_color(GARIS)
        sumbu.spines[sisi].set_linewidth(0.8)
    sumbu.tick_params(colors=TINTA_REDUP, labelsize=9, length=0)
    sumbu.set_axisbelow(True)


def _ke_png(gambar, plt) -> bytes:
    penampung = io.BytesIO()
    gambar.savefig(penampung, format="png", dpi=DPI, bbox_inches="tight",
                   facecolor="white")
    plt.close(gambar)
    return penampung.getvalue()


# --------------------------------------------------------------------------- #
# Peringkat pendorong
# --------------------------------------------------------------------------- #


def peringkat_pendorong(pendorong: list, judul: str = "Peringkat pendorong") -> bytes:
    """Batang mendatar: besaran pengaruh tiap faktor, diurutkan dari yang terkuat.

    Arah pengaruh dibawa warna sekaligus tanda angkanya, sehingga pembaca yang
    tidak membedakan warna tetap dapat membacanya dari label.
    """
    if not pendorong:
        raise ValueError("Tidak ada pendorong yang dapat digambar.")

    plt = _siapkan()
    dipakai = list(pendorong)[:10][::-1]  # matplotlib menggambar dari bawah ke atas
    nama = [p.nama for p in dipakai]
    nilai = [abs(float(p.nilai)) for p in dipakai]
    warna = [
        NETRAL if not p.signifikan else (KUTUB_TURUN if p.arah == "turun" else KUTUB_NAIK)
        for p in dipakai
    ]

    tinggi = max(2.2, 0.42 * len(dipakai) + 1.1)
    gambar, sumbu = plt.subplots(figsize=(LEBAR, tinggi))
    posisi = np.arange(len(dipakai))
    # height < 1 menyisakan celah antar batang, sehingga tepinya tidak menyatu.
    batang = sumbu.barh(posisi, nilai, height=0.62, color=warna)

    sumbu.set_yticks(posisi)
    sumbu.set_yticklabels(nama, fontsize=9.5, color=TINTA)
    sumbu.set_xlabel("Kekuatan pengaruh (nilai mutlak)", fontsize=9, color=TINTA_REDUP)
    sumbu.grid(axis="x", color=GARIS, linewidth=0.8)
    sumbu.grid(axis="y", visible=False)
    _rapikan(sumbu, plt)

    # Label langsung pada tiap batang: nilai bertanda, jadi arahnya terbaca tanpa warna.
    batas = max(nilai) if nilai else 1.0
    for bar, p in zip(batang, dipakai):
        tanda = "-" if p.arah == "turun" else "+"
        keterangan = f"{tanda}{abs(float(p.nilai)):.3f}".replace(".", ",")
        if not p.signifikan:
            keterangan += " (belum terbukti)"
        sumbu.text(
            bar.get_width() + batas * 0.015,
            bar.get_y() + bar.get_height() / 2,
            keterangan,
            va="center",
            fontsize=8.5,
            color=TINTA_REDUP,
        )
    sumbu.set_xlim(0, batas * 1.28)
    _sumbu_indonesia(sumbu, "x")

    sumbu.set_title(judul, fontsize=12, color=TINTA, fontweight="bold", loc="left", pad=12)
    return _ke_png(gambar, plt)


# --------------------------------------------------------------------------- #
# Matriks korelasi
# --------------------------------------------------------------------------- #


def peta_korelasi(matriks: pd.DataFrame, judul: str = "Matriks korelasi") -> bytes:
    """Peta panas korelasi dengan skala divergen dan titik tengah netral.

    Korelasi punya nol yang bermakna dan dua arah, sehingga skalanya divergen —
    bukan satu warna terang ke gelap yang akan menyamarkan tanda.
    """
    if matriks is None or matriks.empty:
        raise ValueError("Matriks korelasi kosong.")

    plt = _siapkan()
    from matplotlib.colors import LinearSegmentedColormap

    skala = LinearSegmentedColormap.from_list(
        "nalardata_divergen", [KUTUB_TURUN, "#eef0f2", KUTUB_NAIK]
    )

    nama = [str(k) for k in matriks.columns]
    nilai = matriks.to_numpy(dtype=float)
    sisi = max(4.2, min(LEBAR, 0.62 * len(nama) + 2.4))
    gambar, sumbu = plt.subplots(figsize=(sisi, sisi * 0.86))

    # vmin/vmax dipaku pada -1 dan 1 agar titik tengah netral benar-benar di nol.
    peta = sumbu.imshow(nilai, cmap=skala, vmin=-1, vmax=1)

    sumbu.set_xticks(np.arange(len(nama)))
    sumbu.set_yticks(np.arange(len(nama)))
    sumbu.set_xticklabels(nama, rotation=45, ha="right", fontsize=8.5, color=TINTA)
    sumbu.set_yticklabels(nama, fontsize=8.5, color=TINTA)
    sumbu.tick_params(length=0)
    for sisi_bingkai in sumbu.spines.values():
        sisi_bingkai.set_visible(False)

    # Garis pemisah tipis antar sel agar warna yang bersebelahan tidak menyatu.
    sumbu.set_xticks(np.arange(len(nama) + 1) - 0.5, minor=True)
    sumbu.set_yticks(np.arange(len(nama) + 1) - 0.5, minor=True)
    sumbu.grid(which="minor", color="white", linewidth=1.6)
    sumbu.tick_params(which="minor", length=0)

    # Angka dicetak di tiap sel: matriks laporan selalu kecil, dan tanpa angka
    # pembaca hanya dapat menerka besarannya dari warna.
    if len(nama) <= 12:
        for i in range(len(nama)):
            for j in range(len(nama)):
                if not np.isfinite(nilai[i, j]):
                    continue
                sumbu.text(
                    j,
                    i,
                    f"{nilai[i, j]:.2f}".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    # Tinta gelap di atas warna muda, putih di atas warna pekat.
                    color="white" if abs(nilai[i, j]) > 0.62 else TINTA,
                )

    bilah = gambar.colorbar(peta, ax=sumbu, shrink=0.78, pad=0.02)
    bilah.set_label("Korelasi", fontsize=9, color=TINTA_REDUP)
    bilah.ax.tick_params(colors=TINTA_REDUP, labelsize=8, length=0)
    _sumbu_indonesia(bilah.ax, "y")
    bilah.outline.set_visible(False)

    sumbu.set_title(judul, fontsize=12, color=TINTA, fontweight="bold", loc="left", pad=14)
    return _ke_png(gambar, plt)


# --------------------------------------------------------------------------- #
# Status pemeriksaan
# --------------------------------------------------------------------------- #


def status_pemeriksaan(lampu: list, judul: str = "Status pemeriksaan") -> bytes:
    """Deretan status berbentuk batang bertingkat, bukan grafik sesungguhnya.

    Yang dibawa hanyalah tiga keadaan, sehingga bentuk paling jujur adalah daftar
    berlabel — bukan grafik yang memberi kesan ada besaran yang dibandingkan.
    """
    if not lampu:
        raise ValueError("Tidak ada pemeriksaan yang dapat digambar.")

    plt = _siapkan()
    warna_status = {"baik": "#1b6f4a", "perhatian": "#96690b", "kritis": ACUAN}

    dipakai = list(lampu)[::-1]
    tinggi = max(1.8, 0.46 * len(dipakai) + 0.9)
    gambar, sumbu = plt.subplots(figsize=(LEBAR, tinggi))
    posisi = np.arange(len(dipakai))

    for y, l in zip(posisi, dipakai):
        warna = warna_status.get(l.status, NETRAL)
        sumbu.barh([y], [1], height=0.6, color=warna)
        sumbu.text(
            1.03,
            y,
            f"{l.status_label} — {l.nilai}",
            va="center",
            fontsize=9,
            color=TINTA,
        )

    sumbu.set_yticks(posisi)
    sumbu.set_yticklabels([l.label for l in dipakai], fontsize=9.5, color=TINTA)
    sumbu.set_xlim(0, 4.2)
    sumbu.set_xticks([])
    for sisi in sumbu.spines.values():
        sisi.set_visible(False)
    sumbu.tick_params(length=0)
    sumbu.set_title(judul, fontsize=12, color=TINTA, fontweight="bold", loc="left", pad=12)
    return _ke_png(gambar, plt)


# --------------------------------------------------------------------------- #
# Kumpulan grafik untuk satu laporan
# --------------------------------------------------------------------------- #


def grafik_laporan(laporan) -> list[tuple[str, bytes]]:
    """Grafik yang layak masuk laporan, beserta judulnya.

    Kegagalan satu grafik tidak menggagalkan ekspor: laporan tetap berguna tanpa
    gambar, sedangkan berkas yang gagal disusun sama sekali tidak berguna.
    """
    hasil: list[tuple[str, bytes]] = []

    if getattr(laporan, "lampu", None):
        try:
            hasil.append(("Status pemeriksaan", status_pemeriksaan(laporan.lampu)))
        except Exception:  # noqa: BLE001 - grafik gagal, laporan tetap jalan
            pass

    if getattr(laporan, "pendorong", None):
        try:
            hasil.append(
                (
                    "Peringkat pendorong",
                    peringkat_pendorong(laporan.pendorong),
                )
            )
        except Exception:  # noqa: BLE001
            pass

    return hasil
