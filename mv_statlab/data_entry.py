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


# --------------------------------------------------------------------------- #
# Variabel gabungan dari beberapa butir
# --------------------------------------------------------------------------- #

# Cara meringkas beberapa butir menjadi satu variabel konstruk.
CARA_GABUNG = {
    "rata": {
        "nama": "Rata-rata butir",
        "catatan": (
            "Satuannya tetap sama dengan butir aslinya (misalnya tetap pada skala "
            "1–5), sehingga paling mudah ditafsirkan. Pilihan paling lazim."
        ),
    },
    "jumlah": {
        "nama": "Jumlah butir",
        "catatan": (
            "Skor total seperti pada tes prestasi. Satuannya bergantung pada "
            "banyaknya butir, jadi tidak sebanding antar konstruk yang butirnya "
            "berbeda jumlah."
        ),
    },
    "z": {
        "nama": "Rata-rata skor baku (z)",
        "catatan": (
            "Tiap butir dibakukan lebih dulu, sehingga butir bersatuan atau berskala "
            "berbeda memberi bobot yang setara."
        ),
    },
}
CARA_GABUNG_BAWAAN = "rata"


def variabel_gabungan(
    df: pd.DataFrame,
    butir: list[str],
    nama: str,
    cara: str = CARA_GABUNG_BAWAAN,
    minimal_terisi: int | None = None,
) -> pd.Series:
    """Ringkas beberapa butir menjadi satu variabel konstruk.

    ``minimal_terisi`` menetapkan berapa butir minimal yang harus terisi agar satu
    responden tetap diberi skor; responden yang kurang dari itu bernilai kosong,
    bukan dihitung dari sisa butir yang ada. Tanpa aturan ini, responden yang hanya
    mengisi satu butir akan tampak setara dengan yang mengisi seluruhnya.
    """
    if cara not in CARA_GABUNG:
        raise ValueError(
            f"Cara '{cara}' tidak dikenal. Pilih dari: {', '.join(CARA_GABUNG)}."
        )
    if len(butir) < 2:
        raise ValueError("Variabel gabungan memerlukan minimal 2 butir.")

    hilang = [b for b in butir if b not in df.columns]
    if hilang:
        raise ValueError(f"Butir tidak ada dalam data: {', '.join(hilang)}.")

    bersih = bakukan_nama(nama)
    if not bersih:
        raise ValueError("Nama variabel baru belum diisi.")

    angka = df[butir].apply(pd.to_numeric, errors="coerce")
    bukan_angka = [b for b in butir if angka[b].isna().all() and df[b].notna().any()]
    if bukan_angka:
        raise ValueError(
            f"Butir berikut bukan angka sehingga tidak dapat digabung: "
            f"{', '.join(bukan_angka)}."
        )

    if cara == "z":
        simpangan = angka.std(ddof=1).replace(0, np.nan)
        angka = (angka - angka.mean()) / simpangan

    terisi = angka.notna().sum(axis=1)
    batas = len(butir) if minimal_terisi is None else int(minimal_terisi)
    batas = max(1, min(batas, len(butir)))

    hasil = angka.sum(axis=1) if cara == "jumlah" else angka.mean(axis=1)
    hasil = hasil.where(terisi >= batas)
    return pd.Series(hasil, index=df.index, name=bersih, dtype="float64")


def ringkas_gabungan(df: pd.DataFrame, butir: list[str], hasil: pd.Series) -> pd.DataFrame:
    """Ringkasan singkat variabel gabungan untuk ditampilkan sesudah dibuat."""
    return pd.DataFrame(
        [
            {"Keterangan": "Nama variabel", "Nilai": str(hasil.name)},
            {"Keterangan": "Butir penyusun", "Nilai": f"{len(butir)} butir"},
            {"Keterangan": "Terisi", "Nilai": f"{int(hasil.notna().sum())} dari {len(hasil)}"},
            {
                "Keterangan": "Rata-rata",
                "Nilai": ("-" if hasil.dropna().empty else f"{hasil.mean():.4f}"),
            },
            {
                "Keterangan": "Simpangan baku",
                "Nilai": (
                    "-" if len(hasil.dropna()) < 2 else f"{hasil.std(ddof=1):.4f}"
                ),
            },
            {
                "Keterangan": "Rentang",
                "Nilai": (
                    "-"
                    if hasil.dropna().empty
                    else f"{hasil.min():.4f} sampai {hasil.max():.4f}"
                ),
            },
        ]
    )
