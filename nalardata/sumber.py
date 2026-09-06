"""Lihat sumber angka: menelusuri tiap angka pada narasi kembali ke sel tabelnya.

Laporan yang disusun otomatis menghadapi satu masalah kepercayaan yang tidak dapat
diselesaikan dengan meyakinkan pembaca: dari mana angka ini berasal. Pembimbing yang
menerima paragraf "B = -269,2985; SE = 18,0748" tidak punya cara memeriksanya selain
percaya, dan pengguna yang harus mempertahankannya di sidang berada dalam keadaan
yang sama.

Modul ini menutup celah itu secara mekanis. Setiap sel tabel hasil dicatat beserta
nilainya, lalu setiap angka pada kalimat dicocokkan kembali ke sel asalnya. Bukan
dengan mencocokkan teks — kalimat membulatkan t menjadi -14,90 sementara tabel
menuliskannya -14,899 — melainkan dengan membandingkan nilai pada ketelitian yang
dipakai kalimat itu.

Akibat sampingannya penting: kalimat yang memuat angka yang **tidak** ada di tabel
mana pun akan langsung terlihat. Angka karangan menjadi mustahil disembunyikan,
bukan karena dilarang, melainkan karena tidak punya sumber.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Angka bergaya Indonesia: titik sebagai pemisah ribuan, koma sebagai desimal.
POLA_ANGKA = re.compile(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?|-?\d+(?:,\d+)?")

# Bilangan bulat kecil seperti "2 dari 3 kelompok" akan cocok dengan hampir semua
# tabel dan hanya menghasilkan derau. Yang ditelusuri hanyalah angka berdesimal,
# atau bilangan bulat yang cukup besar untuk bermakna.
MIN_BULAT = 100

# Tahun terbitan pada sitasi - "Hair dkk. (2019)" - bukan hasil analisis dan tidak
# punya sel asal. Menelusurinya hanya menghasilkan laporan "tanpa sumber" yang
# keliru, dan laporan keliru membuat penelusuran ini berhenti dipercaya.
TAHUN_MIN, TAHUN_MAKS = 1900, 2100


@dataclass(frozen=True)
class Sel:
    """Satu sel tabel hasil beserta letak dan nilainya."""

    tabel: str
    judul: str
    baris: str
    kolom: str
    nilai: float
    tampil: str

    def letak(self) -> str:
        """Sebutan letak yang dapat dibaca manusia."""
        baris = f", baris '{self.baris}'" if self.baris else ""
        return f"{self.tabel}{baris}, kolom '{self.kolom}'"


@dataclass
class Petunjuk:
    """Satu angka pada kalimat beserta sel yang mungkin menjadi sumbernya."""

    tampil: str
    nilai: float
    awal: int
    akhir: int
    sel: list[Sel]

    @property
    def bersumber(self) -> bool:
        return bool(self.sel)


def ke_angka(teks: str) -> float | None:
    """Ubah angka bergaya Indonesia menjadi bilangan; kembalikan None bila gagal."""
    bersih = teks.strip().replace(".", "").replace(",", ".")
    try:
        return float(bersih)
    except ValueError:
        return None


def _desimal(teks: str) -> int:
    return len(teks.split(",")[1]) if "," in teks else 0


def indeks(tabel: dict) -> list[Sel]:
    """Kumpulkan seluruh sel bernilai angka dari tabel-tabel laporan.

    Kolom pertama tiap tabel diperlakukan sebagai label baris, mengikuti bentuk
    tabel hasil di aplikasi ini — Prediktor, Variabel, Asumsi, dan seterusnya.
    """
    hasil: list[Sel] = []
    for nomor, isi in (tabel or {}).items():
        try:
            judul, bingkai, _ = isi
        except (TypeError, ValueError):
            continue
        if bingkai is None or bingkai.empty:
            continue

        kolom_label = str(bingkai.columns[0])
        for _, baris in bingkai.iterrows():
            label = str(baris[kolom_label])
            for kolom in bingkai.columns[1:]:
                mentah = baris[kolom]
                nilai = _nilai(mentah)
                if nilai is None:
                    continue
                hasil.append(
                    Sel(
                        tabel=str(nomor),
                        judul=str(judul),
                        baris=label,
                        kolom=str(kolom),
                        nilai=nilai,
                        tampil=str(mentah),
                    )
                )
    return hasil


def _nilai(mentah) -> float | None:
    """Nilai numerik sebuah sel, apa pun bentuk tersimpannya.

    Tabel hasil menyimpan sebagian angka sebagai teks terformat ("-269,2985",
    "< 0,001") agar tampilannya seragam, jadi keduanya perlu ditangani.
    """
    if isinstance(mentah, (int, float)) and not isinstance(mentah, bool):
        return float(mentah)
    teks = str(mentah).strip()
    if not teks or teks in {"—", "-", "nan"}:
        return None
    # Sel seperti "χ²(10) = 718,01" memuat dua angka: derajat bebas dan statistik
    # ujinya. Yang dikutip narasi adalah yang sesudah tanda sama dengan; mengambil
    # angka pertama akan mengindeks derajat bebasnya dan membuat statistik uji
    # tampak tidak bersumber.
    if "=" in teks:
        teks = teks.rsplit("=", 1)[1].strip()
    cocok = POLA_ANGKA.search(teks)
    return ke_angka(cocok.group()) if cocok else None


def cari(nilai: float, desimal: int, daftar: list[Sel]) -> list[Sel]:
    """Sel yang nilainya sama dengan ``nilai`` pada ketelitian ``desimal``.

    Perbandingan dilakukan setelah kedua sisi dibulatkan ke ketelitian yang dipakai
    kalimat, karena kalimat lazimnya menyebut angka lebih ringkas daripada tabel.

    Bila tidak ada yang cocok, dicoba pula nilai yang sama sebagai persentase:
    narasi menyebut "65,6%" untuk R² yang tersimpan sebagai 0,656. Itu penyajian
    ulang, bukan angka baru, dan melaporkannya sebagai tanpa sumber akan keliru.
    """
    sasaran = round(nilai, desimal)
    cocok = [s for s in daftar if round(s.nilai, desimal) == sasaran]
    if cocok:
        return cocok

    for ubah, tambahan in ((nilai / 100, 2), (nilai * 100, 0)):
        ketelitian = desimal + tambahan
        kandidat = [
            s for s in daftar if round(s.nilai, ketelitian) == round(ubah, ketelitian)
        ]
        if kandidat:
            return kandidat
    return []


def telusuri(teks: str, daftar: list[Sel]) -> list[Petunjuk]:
    """Telusuri setiap angka pada sebuah kalimat kembali ke sel tabelnya."""
    hasil: list[Petunjuk] = []
    for cocok in POLA_ANGKA.finditer(teks or ""):
        tampil = cocok.group()
        nilai = ke_angka(tampil)
        if nilai is None:
            continue
        desimal = _desimal(tampil)
        if desimal == 0 and abs(nilai) < MIN_BULAT:
            continue
        if desimal == 0 and TAHUN_MIN <= nilai <= TAHUN_MAKS and _tampak_sitasi(teks, cocok):
            continue
        hasil.append(
            Petunjuk(
                tampil=tampil,
                nilai=nilai,
                awal=cocok.start(),
                akhir=cocok.end(),
                sel=cari(nilai, desimal, daftar),
            )
        )
    return hasil


def _tampak_sitasi(teks: str, cocok: re.Match) -> bool:
    """Angka empat digit yang diapit tanda kurung atau menyusul nama penulis."""
    sebelum = teks[max(0, cocok.start() - 12) : cocok.start()]
    sesudah = teks[cocok.end() : cocok.end() + 2]
    return "(" in sebelum or sesudah.startswith(")")


def tanpa_sumber(teks: str, daftar: list[Sel]) -> list[str]:
    """Angka pada kalimat yang tidak ditemukan di tabel mana pun.

    Dipakai sebagai invarian pada uji: narasi aplikasi tidak boleh memuat angka
    yang tidak dapat ditelusuri, karena angka semacam itu tidak dapat diperiksa
    pembimbing maupun dipertahankan penulisnya.
    """
    return [p.tampil for p in telusuri(teks, daftar) if not p.bersumber]


def ringkas(teks: str, daftar: list[Sel]) -> pd.DataFrame:
    """Tabel penelusuran satu kalimat, untuk ditampilkan di bawah paragrafnya."""
    baris = []
    for petunjuk in telusuri(teks, daftar):
        if not petunjuk.sel:
            baris.append(
                {"Angka": petunjuk.tampil, "Sumber": "tidak ditemukan di tabel mana pun"}
            )
            continue
        for sel in petunjuk.sel[:3]:
            baris.append({"Angka": petunjuk.tampil, "Sumber": sel.letak()})
    return pd.DataFrame(baris, columns=["Angka", "Sumber"])
