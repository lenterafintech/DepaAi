"""Pagar metodologis: aturan yang tidak boleh dilewati aplikasi maupun penggunanya.

Aplikasi yang membuat analisis menjadi mudah dapat berubah menjadi mesin skripsi.
Modul ini mengumpulkan pagarnya di satu tempat, bukan menyebarnya ke tiap halaman,
karena pagar yang dipasang belakangan di dua puluh halaman pasti terlewat di salah
satunya.

Empat pagar yang bekerja secara mekanis ada di sini:

* **Kunci kausalitas** — kosakata sebab-akibat hanya terbuka bila rancangan
  penelitian mendukung. Bekerja dua arah: memilih kata saat kalimat disusun, dan
  memeriksa kalimat yang terlanjur tersusun.
* **Gerbang asumsi** — pelanggaran asumsi harus tampil bersama hasil, tidak boleh
  disimpan di tab lain yang bisa dilewati.
* **Bukan nilai-p saja** — kesimpulan menuntut ukuran efek, bukan hanya signifikansi.
* **Penanda eksploratori** — analisis di luar praregistrasi ditandai apa adanya.

Yang tidak ada di sini adalah pagar yang menuntut penilaian manusia. Untuk itu
tersedia ``perlu_tinjauan_manusia`` yang menandai, bukan memutuskan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from nalardata import proyek_penelitian as pp

# --------------------------------------------------------------------------- #
# Kunci kausalitas
# --------------------------------------------------------------------------- #

# Padanan asosiatif untuk tiap ungkapan sebab-akibat. Urutannya penting: ungkapan
# yang lebih panjang harus diproses lebih dulu agar tidak terpotong oleh yang pendek
# ("berpengaruh terhadap" sebelum "pengaruh").
PADANAN: tuple[tuple[str, str], ...] = (
    ("berpengaruh secara signifikan terhadap", "berhubungan secara signifikan dengan"),
    ("berpengaruh signifikan terhadap", "berhubungan signifikan dengan"),
    ("berpengaruh terhadap", "berhubungan dengan"),
    ("berpengaruh pada", "berhubungan dengan"),
    ("terbukti berpengaruh", "terbukti berhubungan"),
    ("tidak berpengaruh", "tidak berhubungan"),
    ("berpengaruh", "berhubungan"),
    ("dipengaruhi oleh", "berkaitan dengan"),
    ("dipengaruhi", "berkaitan dengan"),
    ("terpengaruh oleh", "berkaitan dengan"),
    ("terpengaruh", "berkaitan"),
    ("berdampak pada", "berkaitan dengan"),
    ("berdampak", "berkaitan"),
    ("pengaruh terhadap", "hubungan dengan"),
    ("pengaruhnya", "hubungannya"),
    ("pengaruh", "hubungan"),
    ("dampak dari", "kaitan dengan"),
    ("dampaknya", "kaitannya"),
    ("dampak", "kaitan"),
    ("akibat dari", "kaitan dengan"),
    ("disebabkan oleh", "berkaitan dengan"),
    ("disebabkan", "berkaitan dengan"),
    ("menyebabkan", "berkaitan dengan"),
    ("mengakibatkan", "berkaitan dengan"),
    ("berakibat", "berkaitan"),
    ("penyebab", "faktor yang berkaitan"),
    ("mempengaruhi", "berhubungan dengan"),
    ("memengaruhi", "berhubungan dengan"),
    ("meningkatkan", "berkaitan dengan lebih tingginya"),
    ("menurunkan", "berkaitan dengan lebih rendahnya"),
    ("menaikkan", "berkaitan dengan lebih tingginya"),
)

# Kata kerja yang dipakai saat kalimat disusun, dipilih menurut kuncinya.
KATA_KERJA = {
    True: "berpengaruh terhadap",
    False: "berhubungan dengan",
}
KATA_BENDA = {
    True: "pengaruh",
    False: "hubungan",
}


def kata_hubungan(penelitian: pp.ProyekPenelitian | None) -> str:
    """Kata kerja yang boleh dipakai narasi untuk menghubungkan dua variabel.

    Dipakai **saat menyusun** kalimat. Menyusun dengan kata yang benar sejak awal
    jauh lebih aman daripada memperbaikinya belakangan.
    """
    return KATA_KERJA[bool(penelitian and penelitian.boleh_sebab)]


def kata_benda_hubungan(penelitian: pp.ProyekPenelitian | None) -> str:
    """Kata benda padanannya: 'pengaruh' bila kunci terbuka, 'hubungan' bila tidak."""
    return KATA_BENDA[bool(penelitian and penelitian.boleh_sebab)]


def periksa_kausalitas(teks: str, penelitian: pp.ProyekPenelitian | None) -> list[str]:
    """Daftar ungkapan sebab-akibat yang muncul padahal rancangan tidak mendukungnya.

    Diperiksa memakai daftar padanan yang sama dengan yang dipakai untuk meluruskan,
    bukan daftar terpisah. Dua daftar yang berdiri sendiri pasti menyimpang cepat atau
    lambat, dan pemeriksa yang meloloskan satu ungkapan lebih berbahaya daripada tidak
    ada pemeriksa sama sekali karena ia menumbuhkan rasa aman yang keliru.

    Dipakai sebagai invarian pada uji: kalimat yang dihasilkan aplikasi sendiri tidak
    boleh memuat satu pun ungkapan ini pada rancangan potong lintang.
    """
    if penelitian is not None and penelitian.boleh_sebab:
        return []

    rendah = teks.lower()
    temuan = [
        asal for asal, _ in PADANAN if re.search(rf"\b{re.escape(asal)}", rendah)
    ]
    # Ungkapan panjang sudah mencakup yang pendek; sisakan yang paling menjelaskan.
    return [k for k in temuan if not any(k != lain and k in lain for lain in temuan)]


def luruskan_kausalitas(teks: str, penelitian: pp.ProyekPenelitian | None) -> str:
    """Ganti kosakata sebab-akibat dengan padanan asosiatifnya.

    Jaring pengaman terakhir bagi kalimat yang lolos dari pemilihan kata di hulu.
    Huruf besar di awal ungkapan dipertahankan agar kalimat tidak rusak.
    """
    if penelitian is not None and penelitian.boleh_sebab:
        return teks

    hasil = teks
    for asal, ganti in PADANAN:
        hasil = re.sub(
            rf"\b{re.escape(asal)}",
            lambda m, g=ganti: g.capitalize() if m.group(0)[0].isupper() else g,
            hasil,
            flags=re.IGNORECASE,
        )
    return hasil


# --------------------------------------------------------------------------- #
# Gerbang asumsi
# --------------------------------------------------------------------------- #


@dataclass
class Pelanggaran:
    """Satu asumsi yang tidak terpenuhi, beserta akibatnya pada kesimpulan."""

    asumsi: str
    rincian: str
    akibat: str

    def kalimat(self) -> str:
        return f"{self.asumsi}: {self.rincian} {self.akibat}"


# Akibat pelanggaran per asumsi. Disatukan agar kalimatnya seragam di seluruh
# aplikasi, dan agar dapat ditinjau sekaligus.
AKIBAT_ASUMSI = {
    "normalitas": (
        "Nilai p pada uji parametrik menjadi kurang dapat dipercaya, terutama pada "
        "sampel kecil. Pertimbangkan uji non-parametrik."
    ),
    "homogenitas": (
        "Ragam antar kelompok tidak sama, sehingga uji-t dan ANOVA baku dapat "
        "menyesatkan. Pertimbangkan Welch."
    ),
    "multikolinearitas": (
        "Koefisien tiap prediktor menjadi tidak stabil dan tandanya dapat berbalik, "
        "meskipun kecocokan model keseluruhan tetap baik."
    ),
    "heteroskedastisitas": (
        "Galat baku menjadi bias, sehingga nilai p dan selang kepercayaan tidak "
        "dapat dipercaya. Pakai galat baku robust."
    ),
    "autokorelasi": (
        "Galat baku terlalu kecil, sehingga hasil tampak lebih signifikan daripada "
        "seharusnya."
    ),
    "linearitas": (
        "Hubungan yang sebenarnya tidak lurus tidak akan tertangkap model linear."
    ),
}


def gerbang_asumsi(pelanggaran: list[Pelanggaran]) -> str:
    """Kalimat peringatan yang wajib tampil berdampingan dengan hasil.

    Mengembalikan teks kosong bila tidak ada pelanggaran, sehingga pemanggilnya
    dapat menampilkan tanpa memeriksa lebih dulu.
    """
    if not pelanggaran:
        return ""
    if len(pelanggaran) == 1:
        return f"Satu asumsi tidak terpenuhi. {pelanggaran[0].kalimat()}"
    daftar = "; ".join(p.asumsi for p in pelanggaran)
    return (
        f"{len(pelanggaran)} asumsi tidak terpenuhi ({daftar}). Hasil di bawah tetap "
        "ditampilkan apa adanya, namun batas ini wajib disebutkan pada laporan."
    )


def akibat_pelanggaran(asumsi: str) -> str:
    """Akibat baku sebuah pelanggaran asumsi; asumsi asing dijawab dengan kalimat umum."""
    return AKIBAT_ASUMSI.get(
        asumsi.strip().lower(),
        "Kesimpulan dari uji ini perlu dibaca dengan hati-hati.",
    )


# --------------------------------------------------------------------------- #
# Bukan nilai-p saja
# --------------------------------------------------------------------------- #

# Nama kolom yang dianggap memuat ukuran efek atau selang kepercayaan.
PENANDA_EFEK = (
    "efek", "effect", "cohen", "eta", "omega", "epsilon", "r2", "r²",
    "r kuadrat", "adj", "odds", "or", "rr", "hedges", "cramer", "cramér",
    "phi", "rank-biserial", "kendall", "w", "f²", "f2", "d",
)
PENANDA_SELANG = ("ci", "selang", "batas bawah", "batas atas", "lower", "upper", "ll", "ul")


def cukup_dari_nilai_p(tabel: pd.DataFrame) -> bool:
    """Apakah tabel hasil memuat lebih daripada sekadar nilai p.

    Kesimpulan yang hanya bersandar pada signifikansi menyembunyikan besar
    pengaruhnya — dan pengaruh sangat kecil pun menjadi signifikan bila sampelnya
    besar.
    """
    if tabel is None or tabel.empty:
        return False
    nama = [str(k).strip().lower() for k in tabel.columns]
    punya_efek = any(any(t == n or t in n.split() or t in n for t in PENANDA_EFEK) for n in nama)
    punya_selang = any(any(t in n for t in PENANDA_SELANG) for n in nama)
    return punya_efek or punya_selang


def peringatan_nilai_p(tabel: pd.DataFrame) -> str:
    """Peringatan bila tabel hasil hanya memuat signifikansi."""
    if cukup_dari_nilai_p(tabel):
        return ""
    return (
        "Tabel ini hanya memuat signifikansi. Nilai p menyatakan seberapa mungkin "
        "hasil sebesar ini muncul bila tidak ada hubungan sama sekali — bukan "
        "seberapa besar hubungannya. Sertakan ukuran efek sebelum menyimpulkan."
    )


# --------------------------------------------------------------------------- #
# Penanda eksploratori
# --------------------------------------------------------------------------- #


def eksploratori(uji: str, penelitian: pp.ProyekPenelitian | None) -> bool:
    """Apakah sebuah uji berada di luar rencana yang dipraregistrasikan.

    Tanpa praregistrasi, seluruh analisis dianggap eksploratori — itulah keadaan
    yang jujur, bukan sebaliknya.
    """
    pra = penelitian.praregistrasi if penelitian else None
    if pra is None or not pra.uji_direncanakan:
        return True
    nama = uji.strip().lower()
    return not any(
        nama in rencana.strip().lower() or rencana.strip().lower() in nama
        for rencana in pra.uji_direncanakan
    )


def label_eksploratori(uji: str, penelitian: pp.ProyekPenelitian | None) -> str:
    """Kalimat penanda untuk analisis di luar rencana; kosong bila terencana."""
    if not eksploratori(uji, penelitian):
        return ""
    pra = penelitian.praregistrasi if penelitian else None
    if pra is None or not pra.uji_direncanakan:
        return (
            "Analisis eksploratori — belum ada praregistrasi, sehingga hasil ini "
            "sebaiknya dilaporkan sebagai penjajakan dan diuji ulang pada data lain."
        )
    return (
        f"Analisis eksploratori — '{uji}' tidak termasuk uji yang direncanakan pada "
        "praregistrasi. Sebutkan hal ini pada laporan; menyimpang dari rencana adalah "
        "hal biasa, menyembunyikannya tidak."
    )


# --------------------------------------------------------------------------- #
# Tinjauan manusia
# --------------------------------------------------------------------------- #


def perlu_tinjauan_manusia(
    n: int,
    metode: str,
    pelanggaran: list[Pelanggaran] | None = None,
    parameter: int | None = None,
) -> str:
    """Alasan mengapa hasil ini sebaiknya diperiksa statistisi; kosong bila tidak perlu.

    Menandai, bukan memutuskan. Aplikasi tidak berwenang menyatakan sebuah analisis
    salah — ia hanya menunjukkan keadaan yang membuat kesalahan menjadi mungkin.
    """
    alasan = []
    metode_rendah = metode.strip().lower()
    rumit = any(k in metode_rendah for k in ("sem", "cfa", "jalur", "mediasi", "multilevel"))

    if rumit and n < 200:
        alasan.append(
            f"Model {metode} dijalankan pada {n} responden, sedangkan model laten "
            "lazimnya menuntut sekurang-kurangnya 200."
        )
    if parameter and n < 5 * parameter:
        alasan.append(
            f"Ada {parameter} parameter yang ditaksir dari {n} responden — di bawah "
            "patokan lima responden per parameter."
        )
    if pelanggaran and len(pelanggaran) >= 3:
        alasan.append(
            f"{len(pelanggaran)} asumsi tidak terpenuhi sekaligus, sehingga perbaikan "
            "satu per satu belum tentu memadai."
        )
    if n < 30:
        alasan.append(
            f"Hanya {n} pengamatan yang dianalisis, sehingga hampir semua uji "
            "menjadi tidak stabil."
        )
    return " ".join(alasan)
