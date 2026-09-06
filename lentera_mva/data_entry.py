"""Pembuatan dan penyuntingan data langsung di dalam aplikasi.

Berisi logika murni (tanpa Streamlit) untuk menyusun kerangka data kosong dari
definisi kolom, membangkitkan kolom kuesioner, serta membersihkan hasil entri
sebelum dipakai sebagai data aktif.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Tipe kolom yang dapat dipilih pengguna beserta dtype pandas-nya.
# Kolom angka memakai float64 dan kolom teks memakai object, karena keduanya
# menampilkan sel kosong sebagai sel benar-benar kosong pada editor tabel —
# dtype nullable (Int64/string) justru menuliskan "None" di setiap sel kosong.
TIPE_KOLOM: dict[str, str] = {
    "Angka desimal": "float64",
    "Angka bulat": "float64",
    "Skala Likert 1–5": "float64",
    "Skala Likert 1–7": "float64",
    "Teks": "object",
    "Kategori": "object",
    "Ya/Tidak": "object",
}
RENTANG_LIKERT = {"Skala Likert 1–5": (1, 5), "Skala Likert 1–7": (1, 7)}
PILIHAN_BINER = ["Ya", "Tidak"]


@dataclass
class KolomBaru:
    """Definisi satu kolom yang akan dibuat."""

    nama: str
    tipe: str = "Angka desimal"
    pilihan: list[str] = field(default_factory=list)

    def dtype(self) -> str:
        return TIPE_KOLOM.get(self.tipe, "float64")

    def daftar_pilihan(self) -> list[str]:
        if self.tipe == "Ya/Tidak":
            return list(PILIHAN_BINER)
        return [p for p in self.pilihan if str(p).strip()]


def bakukan_nama(nama: str) -> str:
    """Ubah judul kolom menjadi nama yang aman: huruf kecil, tanpa spasi ganda."""
    bersih = re.sub(r"[^\w\s-]", "", str(nama).strip(), flags=re.UNICODE)
    bersih = re.sub(r"[\s-]+", "_", bersih).strip("_").lower()
    return bersih


def validasi_kolom(kolom: list[KolomBaru]) -> list[str]:
    """Kembalikan daftar masalah yang membuat definisi kolom belum bisa dipakai."""
    masalah: list[str] = []
    nama = [bakukan_nama(k.nama) for k in kolom]
    if not kolom:
        masalah.append("Belum ada kolom yang didefinisikan.")
    if any(not n for n in nama):
        masalah.append("Ada kolom tanpa nama.")
    ganda = sorted({n for n in nama if n and nama.count(n) > 1})
    if ganda:
        masalah.append(f"Nama kolom terduplikasi: {', '.join(ganda)}.")
    for k in kolom:
        if k.tipe == "Kategori" and len(k.daftar_pilihan()) < 2:
            masalah.append(f"Kolom '{k.nama}' bertipe Kategori perlu minimal 2 pilihan.")
    return masalah


def buat_kerangka(kolom: list[KolomBaru], n_baris: int = 10) -> pd.DataFrame:
    """Bangun tabel kosong siap diisi sesuai definisi kolom."""
    if masalah := validasi_kolom(kolom):
        raise ValueError(masalah[0])
    data: dict[str, pd.Series] = {}
    for k in kolom:
        nama = bakukan_nama(k.nama)
        dtype = k.dtype()
        kosong = np.nan if dtype == "float64" else None
        data[nama] = pd.Series([kosong] * n_baris, dtype=dtype)
    return pd.DataFrame(data)


def kolom_kuesioner(
    konstruk: dict[str, int], skala: str = "Skala Likert 1–5"
) -> list[KolomBaru]:
    """Susun kolom butir kuesioner: KUAL1, KUAL2, … untuk tiap konstruk.

    Pola penamaan ini yang lazim dipakai pada analisis faktor konfirmatori dan SEM,
    sehingga butir satu konstruk mudah dikenali dan dikelompokkan.
    """
    kolom: list[KolomBaru] = []
    for nama, jumlah in konstruk.items():
        awalan = bakukan_nama(nama).upper()[:6] or "ITEM"
        kolom += [KolomBaru(f"{awalan}{i}", skala) for i in range(1, int(jumlah) + 1)]
    return kolom


def rapikan(df: pd.DataFrame) -> pd.DataFrame:
    """Buang baris kosong dan ubah kolom angka bertipe teks menjadi numerik.

    Editor tabel mengembalikan sel kosong sebagai None dan kadang angka sebagai
    teks; keduanya perlu dirapikan sebelum data dipakai untuk analisis.
    """
    hasil = df.copy()
    hasil = hasil.dropna(how="all")
    for kolom_nama in hasil.columns:
        if pd.api.types.is_numeric_dtype(hasil[kolom_nama]):
            continue
        angka = pd.to_numeric(hasil[kolom_nama], errors="coerce")
        terisi = hasil[kolom_nama].notna().sum()
        # Kolom dianggap numerik hanya bila seluruh isinya memang angka.
        if terisi and angka.notna().sum() == terisi:
            hasil[kolom_nama] = angka
    for kolom_nama in hasil.columns:
        # Angka yang seluruhnya bulat (misalnya skala Likert) disimpan sebagai
        # bilangan bulat agar tidak tampil sebagai 4,0 pada tabel dan laporan.
        if pd.api.types.is_float_dtype(hasil[kolom_nama]):
            nilai = hasil[kolom_nama].dropna()
            if not nilai.empty and (nilai % 1 == 0).all():
                hasil[kolom_nama] = hasil[kolom_nama].astype("Int64")
    return hasil.reset_index(drop=True)


def ringkas_kelengkapan(df: pd.DataFrame) -> pd.DataFrame:
    """Tabel kelengkapan isian per kolom, untuk memandu proses entri."""
    if df.empty:
        return pd.DataFrame(columns=["Kolom", "Terisi", "Kosong", "% Terisi"])
    terisi = df.notna().sum()
    return pd.DataFrame(
        {
            "Kolom": df.columns,
            "Terisi": terisi.to_numpy(),
            "Kosong": (len(df) - terisi).to_numpy(),
            "% Terisi": np.round(terisi.to_numpy() / len(df) * 100, 1),
        }
    )


def periksa_rentang(df: pd.DataFrame, kolom: list[KolomBaru]) -> list[str]:
    """Peringatan bila isian keluar dari rentang skala yang ditetapkan."""
    peringatan: list[str] = []
    for k in kolom:
        rentang = RENTANG_LIKERT.get(k.tipe)
        nama = bakukan_nama(k.nama)
        if not rentang or nama not in df.columns:
            continue
        nilai = pd.to_numeric(df[nama], errors="coerce").dropna()
        di_luar = nilai[(nilai < rentang[0]) | (nilai > rentang[1])]
        if not di_luar.empty:
            peringatan.append(
                f"Kolom {nama}: {len(di_luar)} isian di luar rentang "
                f"{rentang[0]}–{rentang[1]}."
            )
    return peringatan
