"""Ruang Proyek: apa yang diteliti, dengan desain apa, dan atas dasar sampel apa.

Perangkat statistik lazimnya mulai dari data. Akibatnya ia tidak pernah tahu apakah
angka yang dihasilkannya boleh dibaca sebagai sebab-akibat, atau apakah kesimpulannya
boleh diberlakukan pada populasi. Kedua hal itu **tidak ditentukan oleh data maupun
oleh metode**, melainkan oleh rancangan penelitian dan cara sampel diambil.

Modul ini menyimpan rancangan itu, dan dari sanalah dua kunci pengaman bekerja:

* **Kunci kausalitas** — kata "menyebabkan" hanya terbuka bila desainnya mendukung.
  Regresi yang signifikan pada data potong lintang tetap hanya menunjukkan hubungan.
* **Kunci generalisasi** — kesimpulan boleh diberlakukan pada populasi hanya bila
  sampelnya diambil secara acak. Besar sampel tidak menggantikan cara pengambilannya;
  seribu responden yang dipilih seadanya tetap tidak mewakili populasi.

Keduanya menjawab dua kekeliruan yang paling sering lolos dari pemeriksaan pembimbing
justru karena keduanya tidak terlihat pada keluaran statistik mana pun.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# --------------------------------------------------------------------------- #
# Desain penelitian
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Desain:
    """Satu rancangan penelitian beserta batas kesimpulan yang melekat padanya."""

    kode: str
    nama: str
    keterangan: str
    boleh_sebab: bool
    alasan: str
    perlu_penugasan_acak: bool = False


DESAIN: dict[str, Desain] = {
    "potong_lintang": Desain(
        kode="potong_lintang",
        nama="Potong lintang (cross-sectional)",
        keterangan="Seluruh variabel diukur pada satu waktu, lazimnya lewat satu kuesioner.",
        boleh_sebab=False,
        alasan=(
            "Sebab dan akibat diukur bersamaan, sehingga arah pengaruhnya tidak dapat "
            "dipastikan dan perancu yang tidak diukur tetap mungkin berperan."
        ),
    ),
    "eksperimen": Desain(
        kode="eksperimen",
        nama="Eksperimen",
        keterangan="Peneliti memberi perlakuan dan menempatkan subjek ke kelompok.",
        boleh_sebab=True,
        alasan=(
            "Perlakuan ditetapkan peneliti dan subjek ditempatkan secara acak, sehingga "
            "perbedaan hasil antar kelompok dapat dibaca sebagai akibat perlakuan."
        ),
        perlu_penugasan_acak=True,
    ),
    "kuasi_eksperimen": Desain(
        kode="kuasi_eksperimen",
        nama="Kuasi-eksperimen",
        keterangan="Ada perlakuan, tetapi kelompoknya sudah terbentuk sebelumnya.",
        boleh_sebab=False,
        alasan=(
            "Ada perlakuan, namun subjek tidak ditempatkan secara acak, sehingga "
            "perbedaan antar kelompok dapat berasal dari perbedaan yang sudah ada "
            "sejak semula."
        ),
    ),
    "longitudinal": Desain(
        kode="longitudinal",
        nama="Longitudinal",
        keterangan="Responden yang sama diikuti pada beberapa waktu pengukuran.",
        boleh_sebab=False,
        alasan=(
            "Urutan waktunya diketahui, sehingga arah hubungan lebih meyakinkan "
            "daripada potong lintang. Namun tanpa penugasan acak, perancu yang tidak "
            "diukur tetap dapat menjelaskan hasilnya."
        ),
    ),
    "panel": Desain(
        kode="panel",
        nama="Data panel",
        keterangan="Banyak unit diamati pada beberapa periode.",
        boleh_sebab=False,
        alasan=(
            "Model efek tetap dapat menyingkirkan perancu yang tidak berubah menurut "
            "waktu, tetapi tidak yang berubah. Kesimpulan sebab-akibat menuntut asumsi "
            "identifikasi yang harus dinyatakan sendiri."
        ),
    ),
    "deret_waktu": Desain(
        kode="deret_waktu",
        nama="Deret waktu",
        keterangan="Satu unit diamati pada banyak periode berurutan.",
        boleh_sebab=False,
        alasan=(
            "Hubungan antar deret dapat timbul dari tren bersama. Bahkan uji kausalitas "
            "Granger hanya menunjukkan urutan waktu, bukan sebab-akibat."
        ),
    ),
    "studi_kasus": Desain(
        kode="studi_kasus",
        nama="Studi kasus",
        keterangan="Satu atau beberapa kasus ditelaah mendalam.",
        boleh_sebab=False,
        alasan=(
            "Kasus dipilih justru karena kekhususannya, sehingga hasilnya menjelaskan "
            "kasus itu dan bukan populasi."
        ),
    ),
}

# --------------------------------------------------------------------------- #
# Teknik pengambilan sampel
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sampling:
    """Satu teknik pengambilan sampel beserta akibatnya pada generalisasi."""

    kode: str
    nama: str
    probabilitas: bool
    alasan: str


SAMPLING: dict[str, Sampling] = {
    "acak_sederhana": Sampling(
        kode="acak_sederhana",
        nama="Acak sederhana",
        probabilitas=True,
        alasan="Setiap anggota populasi berpeluang sama terpilih.",
    ),
    "sistematis": Sampling(
        kode="sistematis",
        nama="Sistematis",
        probabilitas=True,
        alasan="Pemilihan berjarak tetap dari daftar populasi yang lengkap.",
    ),
    "berlapis": Sampling(
        kode="berlapis",
        nama="Acak berlapis (stratified)",
        probabilitas=True,
        alasan="Populasi dibagi menurut lapisan, lalu diacak di dalam tiap lapisan.",
    ),
    "gugus": Sampling(
        kode="gugus",
        nama="Acak gugus (cluster)",
        probabilitas=True,
        alasan=(
            "Gugus dipilih secara acak. Perlu diingat bahwa pengamatan dalam satu gugus "
            "cenderung mirip, sehingga galat bakunya perlu disesuaikan."
        ),
    ),
    "purposif": Sampling(
        kode="purposif",
        nama="Purposif (pertimbangan tertentu)",
        probabilitas=False,
        alasan="Responden dipilih menurut kriteria peneliti, bukan secara acak.",
    ),
    "insidental": Sampling(
        kode="insidental",
        nama="Insidental / kebetulan (accidental)",
        probabilitas=False,
        alasan="Responden adalah siapa saja yang kebetulan ditemui dan bersedia.",
    ),
    "kuota": Sampling(
        kode="kuota",
        nama="Kuota",
        probabilitas=False,
        alasan="Jumlah per kelompok ditetapkan, namun pemilihan di dalamnya tidak acak.",
    ),
    "bola_salju": Sampling(
        kode="bola_salju",
        nama="Bola salju (snowball)",
        probabilitas=False,
        alasan=(
            "Responden merujuk responden berikutnya, sehingga sampel mengikuti jaringan "
            "pertemanan dan tidak mewakili populasi."
        ),
    ),
    "jenuh": Sampling(
        kode="jenuh",
        nama="Sampel jenuh / sensus",
        probabilitas=True,
        alasan=(
            "Seluruh anggota populasi diambil, sehingga tidak ada galat pengambilan "
            "sampel. Uji signifikansi menjadi kurang bermakna karena tidak ada "
            "penarikan kesimpulan dari sampel ke populasi."
        ),
    ),
}

SUMBER_DATA = {
    "primer": "Primer — dikumpulkan sendiri oleh peneliti",
    "sekunder": "Sekunder — berasal dari pihak lain atau data yang sudah ada",
    "campuran": "Campuran primer dan sekunder",
}

# Kata yang menyiratkan sebab-akibat. Dipakai kunci kausalitas untuk memeriksa
# kalimat yang dihasilkan aplikasi sendiri, bukan tulisan pengguna.
KATA_SEBAB = (
    "menyebabkan",
    "penyebab",
    "disebabkan",
    "mengakibatkan",
    "berakibat",
    "akibat dari",
    "berdampak pada",
    "dampak dari",
    "berpengaruh terhadap",
    "berpengaruh pada",
    "mempengaruhi",
    "memengaruhi",
    "pengaruh terhadap",
    "meningkatkan",
    "menurunkan",
    "menaikkan",
)


# --------------------------------------------------------------------------- #
# Praregistrasi
# --------------------------------------------------------------------------- #


@dataclass
class Praregistrasi:
    """Hipotesis dan rencana uji yang dicatat **sebelum** data dilihat.

    Perbandingan antara rencana ini dan yang benar-benar dijalankan bukan alat
    penghakiman: menyimpang dari rencana adalah hal biasa dalam penelitian. Yang
    tidak biasa adalah menyimpang tanpa menyebutkannya.
    """

    waktu: str = ""
    hipotesis: list[str] = field(default_factory=list)
    uji_direncanakan: list[str] = field(default_factory=list)
    catatan: str = ""

    def __post_init__(self) -> None:
        if not self.waktu:
            self.waktu = datetime.now().strftime("%d-%m-%Y %H:%M")

    @property
    def sidik(self) -> str:
        """Sidik isi praregistrasi, agar perubahan diam-diam dapat terlihat."""
        isi = json.dumps(
            {"h": self.hipotesis, "u": self.uji_direncanakan, "c": self.catatan},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(isi.encode("utf-8")).hexdigest()[:12]

    def kosong(self) -> bool:
        return not (self.hipotesis or self.uji_direncanakan)


# --------------------------------------------------------------------------- #
# Ruang Proyek
# --------------------------------------------------------------------------- #


@dataclass
class ProyekPenelitian:
    """Rancangan penelitian yang menjadi dasar seluruh batas kesimpulan."""

    judul: str = ""
    bidang: str = ""
    pertanyaan: list[str] = field(default_factory=list)
    hipotesis: list[str] = field(default_factory=list)
    populasi: str = ""
    ukuran_populasi: int | None = None
    unit_analisis: str = ""
    desain: str = "potong_lintang"
    penugasan_acak: bool = False
    teknik_sampling: str = "purposif"
    sumber_data: str = "primer"
    target_sampel: int | None = None
    dibuat: str = ""
    praregistrasi: Praregistrasi | None = None

    def __post_init__(self) -> None:
        if self.desain not in DESAIN:
            raise ValueError(f"Desain '{self.desain}' tidak dikenal. Pilih dari {list(DESAIN)}.")
        if self.teknik_sampling not in SAMPLING:
            raise ValueError(
                f"Teknik sampling '{self.teknik_sampling}' tidak dikenal. "
                f"Pilih dari {list(SAMPLING)}."
            )
        if self.sumber_data not in SUMBER_DATA:
            raise ValueError(f"Sumber data '{self.sumber_data}' tidak dikenal.")
        if not self.dibuat:
            self.dibuat = datetime.now().strftime("%d-%m-%Y %H:%M")

    # -- keterangan turunan ------------------------------------------------- #

    @property
    def rancangan(self) -> Desain:
        return DESAIN[self.desain]

    @property
    def cara_sampling(self) -> Sampling:
        return SAMPLING[self.teknik_sampling]

    @property
    def boleh_sebab(self) -> bool:
        """Apakah kesimpulan sebab-akibat dibolehkan oleh rancangannya."""
        rancangan = self.rancangan
        if not rancangan.boleh_sebab:
            return False
        return self.penugasan_acak if rancangan.perlu_penugasan_acak else True

    @property
    def alasan_sebab(self) -> str:
        """Mengapa bahasa sebab-akibat terbuka atau tertutup, dalam satu kalimat."""
        rancangan = self.rancangan
        if rancangan.perlu_penugasan_acak and not self.penugasan_acak:
            return (
                "Rancangan eksperimen dipilih, namun subjek tidak ditempatkan secara "
                "acak. Tanpa penugasan acak, perbedaan antar kelompok dapat berasal "
                "dari perbedaan yang sudah ada sejak semula."
            )
        return rancangan.alasan

    @property
    def boleh_generalisasi(self) -> bool:
        """Apakah kesimpulan boleh diberlakukan pada populasi."""
        return self.cara_sampling.probabilitas

    @property
    def alasan_generalisasi(self) -> str:
        cara = self.cara_sampling
        if cara.probabilitas:
            return cara.alasan
        return (
            f"{cara.alasan} Karena itu kesimpulan berlaku bagi responden yang diteliti, "
            "bukan bagi seluruh populasi — berapa pun banyaknya responden."
        )

    # -- batas kesimpulan --------------------------------------------------- #

    def batas_kesimpulan(self) -> list[str]:
        """Kalimat batas yang wajib ikut pada setiap laporan.

        Ditulis dari rancangan, bukan dari hasil, sehingga sudah dapat dibaca
        sebelum satu uji pun dijalankan.
        """
        batas = []
        rancangan = self.rancangan
        if self.boleh_sebab:
            batas.append(
                f"Rancangan {rancangan.nama.lower()} mendukung kesimpulan sebab-akibat. "
                f"{self.alasan_sebab}"
            )
        else:
            batas.append(
                f"Hasil ini menunjukkan hubungan, bukan sebab-akibat. {self.alasan_sebab}"
            )

        if self.boleh_generalisasi:
            batas.append(
                f"Sampel diambil secara {self.cara_sampling.nama.lower()}, sehingga "
                "kesimpulan dapat diberlakukan pada populasi yang menjadi kerangka "
                "sampelnya."
            )
        else:
            batas.append(self.alasan_generalisasi)

        if self.sumber_data == "sekunder":
            batas.append(
                "Data berasal dari pihak lain, sehingga definisi variabel dan mutu "
                "pengukurannya mengikuti sumber aslinya dan tidak dapat diperbaiki "
                "pada tahap analisis."
            )
        return batas

    # -- kelengkapan -------------------------------------------------------- #

    def kekurangan(self) -> list[str]:
        """Bagian yang belum diisi, disebutkan sebagai daftar yang dapat ditindaklanjuti."""
        kurang = []
        if not self.judul.strip():
            kurang.append("Judul penelitian")
        if not self.pertanyaan:
            kurang.append("Pertanyaan penelitian")
        if not self.populasi.strip():
            kurang.append("Populasi penelitian")
        if not self.unit_analisis.strip():
            kurang.append("Unit analisis")
        return kurang

    def lengkap(self) -> bool:
        return not self.kekurangan()

    def ringkas(self) -> pd.DataFrame:
        """Ringkasan rancangan untuk ditampilkan dan ikut ke laporan."""
        return pd.DataFrame(
            [
                {"Keterangan": "Judul", "Isi": self.judul or "-"},
                {"Keterangan": "Bidang", "Isi": self.bidang or "-"},
                {
                    "Keterangan": "Pertanyaan penelitian",
                    "Isi": f"{len(self.pertanyaan)} butir" if self.pertanyaan else "-",
                },
                {
                    "Keterangan": "Hipotesis",
                    "Isi": f"{len(self.hipotesis)} butir" if self.hipotesis else "-",
                },
                {"Keterangan": "Populasi", "Isi": self.populasi or "-"},
                {"Keterangan": "Unit analisis", "Isi": self.unit_analisis or "-"},
                {"Keterangan": "Desain", "Isi": self.rancangan.nama},
                {"Keterangan": "Teknik sampling", "Isi": self.cara_sampling.nama},
                {"Keterangan": "Sumber data", "Isi": SUMBER_DATA[self.sumber_data]},
                {
                    "Keterangan": "Kesimpulan sebab-akibat",
                    "Isi": "Boleh" if self.boleh_sebab else "Tidak boleh",
                },
                {
                    "Keterangan": "Generalisasi ke populasi",
                    "Isi": "Boleh" if self.boleh_generalisasi else "Tidak boleh",
                },
                {
                    "Keterangan": "Praregistrasi",
                    "Isi": (
                        f"Dicatat {self.praregistrasi.waktu}"
                        if self.praregistrasi and not self.praregistrasi.kosong()
                        else "Belum ada"
                    ),
                },
            ]
        )

    # -- penyimpanan -------------------------------------------------------- #

    def ke_dict(self) -> dict:
        isi = {
            "judul": self.judul,
            "bidang": self.bidang,
            "pertanyaan": list(self.pertanyaan),
            "hipotesis": list(self.hipotesis),
            "populasi": self.populasi,
            "ukuran_populasi": self.ukuran_populasi,
            "unit_analisis": self.unit_analisis,
            "desain": self.desain,
            "penugasan_acak": self.penugasan_acak,
            "teknik_sampling": self.teknik_sampling,
            "sumber_data": self.sumber_data,
            "target_sampel": self.target_sampel,
            "dibuat": self.dibuat,
        }
        if self.praregistrasi is not None:
            isi["praregistrasi"] = {
                "waktu": self.praregistrasi.waktu,
                "hipotesis": list(self.praregistrasi.hipotesis),
                "uji_direncanakan": list(self.praregistrasi.uji_direncanakan),
                "catatan": self.praregistrasi.catatan,
            }
        return isi

    @classmethod
    def dari_dict(cls, isi: dict | None) -> "ProyekPenelitian":
        """Pulihkan dari berkas proyek; nilai yang cacat kembali ke bawaan."""
        if not isinstance(isi, dict):
            return cls()

        def teks(kunci: str) -> str:
            return str(isi.get(kunci) or "")

        def daftar(kunci: str) -> list[str]:
            nilai = isi.get(kunci)
            return [str(x) for x in nilai] if isinstance(nilai, list) else []

        def bilangan(kunci: str) -> int | None:
            try:
                nilai = isi.get(kunci)
                return int(nilai) if nilai is not None else None
            except (TypeError, ValueError):
                return None

        desain = teks("desain") or "potong_lintang"
        sampling = teks("teknik_sampling") or "purposif"
        sumber = teks("sumber_data") or "primer"

        pra = isi.get("praregistrasi")
        praregistrasi = None
        if isinstance(pra, dict):
            praregistrasi = Praregistrasi(
                waktu=str(pra.get("waktu") or ""),
                hipotesis=[str(x) for x in (pra.get("hipotesis") or [])],
                uji_direncanakan=[str(x) for x in (pra.get("uji_direncanakan") or [])],
                catatan=str(pra.get("catatan") or ""),
            )

        return cls(
            judul=teks("judul"),
            bidang=teks("bidang"),
            pertanyaan=daftar("pertanyaan"),
            hipotesis=daftar("hipotesis"),
            populasi=teks("populasi"),
            ukuran_populasi=bilangan("ukuran_populasi"),
            unit_analisis=teks("unit_analisis"),
            desain=desain if desain in DESAIN else "potong_lintang",
            penugasan_acak=bool(isi.get("penugasan_acak", False)),
            teknik_sampling=sampling if sampling in SAMPLING else "purposif",
            sumber_data=sumber if sumber in SUMBER_DATA else "primer",
            target_sampel=bilangan("target_sampel"),
            dibuat=teks("dibuat"),
            praregistrasi=praregistrasi,
        )

    def kosong(self) -> bool:
        """Proyek yang belum diisi sama sekali; dipakai untuk memutuskan penyimpanan."""
        return not any(
            [
                self.judul.strip(),
                self.bidang.strip(),
                self.pertanyaan,
                self.hipotesis,
                self.populasi.strip(),
                self.unit_analisis.strip(),
                self.praregistrasi is not None,
            ]
        )
