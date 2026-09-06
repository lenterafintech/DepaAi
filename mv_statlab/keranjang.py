"""Keranjang hasil: mengumpulkan analisis yang benar-benar dijalankan pengguna.

Halaman ringkasan menyusun kesimpulan dari rangkaian metode yang dipilih mesin
narasi. Yang tidak tertampung di sana adalah analisis yang dijalankan pengguna
sendiri di halaman metode — regresi dengan prediktor pilihannya, CFA dengan model
yang ia tulis, uji non-parametrik tertentu. Modul ini menampung hasil-hasil itu
sehingga ikut terbawa ke berkas ekspor.

Modul ini sengaja tidak menyentuh Streamlit agar dapat diuji tanpa menjalankan
aplikasi; penyimpanannya di sesi diurus ``mv_statlab.ui``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# Jenis isi yang dapat ditampung. Grafik belum termasuk: menyisipkannya ke Word dan
# PDF menuntut perender gambar tersendiri, jadi ditunda sampai alur ini terbukti.
JENIS = {
    "tabel": "Tabel hasil",
    "tafsiran": "Cara membaca",
    "catatan": "Catatan",
}

# Pemisah ruas saat menghitung sidik jari. Memakai unit separator (U+001F) karena
# lambang itu tidak muncul pada judul, teks, maupun isi tabel.
PEMISAH = ""


def _sidik(bagian: str, judul: str, jenis: str, teks: str, tabel: pd.DataFrame | None) -> str:
    """Sidik jari isi, dipakai untuk mengenali item yang sama persis.

    Pengguna kerap menekan ulang tombol yang sama atau kembali ke halaman yang
    sudah dibuka; tanpa sidik jari, keranjang akan dipenuhi salinan yang identik.
    """
    bagian_isi = [str(bagian), str(judul), str(jenis), str(teks)]
    if tabel is not None:
        # to_csv memberi wakilan yang stabil: urutan kolom, nilai, dan tipe ikut
        # menentukan hasilnya, sementara alamat objek tidak.
        bagian_isi.append(tabel.to_csv(index=False))
    # Pemisah eksplisit agar ruas yang bersebelahan tidak menyatu; tanpa itu
    # ("ab", "c") dan ("a", "bc") akan menghasilkan sidik yang sama.
    return hashlib.sha1(PEMISAH.join(bagian_isi).encode("utf-8")).hexdigest()[:16]


@dataclass
class Item:
    """Satu hasil yang disimpan pengguna."""

    bagian: str  # nama metode, misalnya "Regresi linear"
    judul: str
    jenis: str = "tabel"
    tabel: pd.DataFrame | None = None
    teks: str = ""
    catatan: str = ""
    tanda_data: str = ""  # identitas data saat hasil ini ditangkap
    waktu: str = field(default_factory=lambda: datetime.now().strftime("%d-%m-%Y %H:%M"))
    sidik: str = ""

    def __post_init__(self) -> None:
        if self.jenis not in JENIS:
            raise ValueError(
                f"Jenis '{self.jenis}' tidak dikenal. Pilih dari: {', '.join(JENIS)}."
            )
        if self.jenis == "tabel" and self.tabel is None:
            raise ValueError("Item bertipe tabel memerlukan DataFrame.")
        if self.jenis != "tabel" and not str(self.teks).strip():
            raise ValueError(f"Item bertipe {self.jenis} memerlukan teks.")
        if self.tabel is not None:
            # Salinan, bukan rujukan: data aktif dapat berubah setelah hasil
            # ditangkap, dan hasil yang sudah disimpan tidak boleh ikut berubah.
            self.tabel = self.tabel.copy()
        if not self.sidik:
            self.sidik = _sidik(self.bagian, self.judul, self.jenis, self.teks, self.tabel)

    def basi(self, tanda_sekarang: str) -> bool:
        """Apakah item ini berasal dari data yang sudah tidak aktif lagi?"""
        return bool(self.tanda_data) and bool(tanda_sekarang) and self.tanda_data != tanda_sekarang


@dataclass
class Keranjang:
    """Kumpulan hasil yang siap disusun menjadi laporan."""

    item: list[Item] = field(default_factory=list)
    judul: str = "Laporan Hasil Analisis Statistik"
    peneliti: str = ""

    # ----------------------------------------------------------------- #
    # Perubahan isi
    # ----------------------------------------------------------------- #

    def tambah(self, item: Item) -> bool:
        """Simpan satu hasil. Mengembalikan False bila isinya sudah ada."""
        if any(a.sidik == item.sidik for a in self.item):
            return False
        self.item.append(item)
        return True

    def tambah_tabel(
        self,
        bagian: str,
        judul: str,
        tabel: pd.DataFrame,
        catatan: str = "",
        tanda_data: str = "",
    ) -> bool:
        return self.tambah(
            Item(
                bagian=bagian,
                judul=judul,
                jenis="tabel",
                tabel=tabel,
                catatan=catatan,
                tanda_data=tanda_data,
            )
        )

    def tambah_teks(
        self,
        bagian: str,
        judul: str,
        teks: str,
        jenis: str = "tafsiran",
        tanda_data: str = "",
    ) -> bool:
        return self.tambah(
            Item(
                bagian=bagian,
                judul=judul,
                jenis=jenis,
                teks=teks,
                tanda_data=tanda_data,
            )
        )

    def hapus(self, sidik: str) -> bool:
        sebelum = len(self.item)
        self.item = [i for i in self.item if i.sidik != sidik]
        return len(self.item) < sebelum

    def hapus_bagian(self, bagian: str) -> int:
        sebelum = len(self.item)
        self.item = [i for i in self.item if i.bagian != bagian]
        return sebelum - len(self.item)

    def kosongkan(self) -> None:
        self.item.clear()

    # ----------------------------------------------------------------- #
    # Pembacaan
    # ----------------------------------------------------------------- #

    def kosong(self) -> bool:
        return not self.item

    def bagian(self) -> list[str]:
        """Nama bagian menurut urutan hasil pertama kali disimpan."""
        urut: list[str] = []
        for i in self.item:
            if i.bagian not in urut:
                urut.append(i.bagian)
        return urut

    def per_bagian(self) -> dict[str, list[Item]]:
        kelompok: dict[str, list[Item]] = {}
        for nama in self.bagian():
            kelompok[nama] = [i for i in self.item if i.bagian == nama]
        return kelompok

    def basi(self, tanda_sekarang: str) -> list[Item]:
        return [i for i in self.item if i.basi(tanda_sekarang)]

    def ringkas(self) -> dict[str, int]:
        """Hitungan isi keranjang, untuk ditampilkan sebagai metrik."""
        return {
            "Bagian": len(self.bagian()),
            "Tabel": sum(1 for i in self.item if i.jenis == "tabel"),
            "Tafsiran": sum(1 for i in self.item if i.jenis == "tafsiran"),
            "Objek": len(self.item),
        }

    def daftar_isi(self) -> pd.DataFrame:
        """Daftar isi laporan, sejajar dengan yang ditampilkan di halaman Laporan."""
        baris = []
        for nomor, (nama, isi) in enumerate(self.per_bagian().items(), start=1):
            baris.append(
                {
                    "#": nomor,
                    "Bagian": nama,
                    "Objek": " · ".join(dict.fromkeys(i.judul for i in isi)),
                    "Jumlah": len(isi),
                }
            )
        return pd.DataFrame(baris)
