"""Jejak keputusan dan jejak uji: apa yang dijalankan, kapan, dan atas dasar apa.

Dua hal yang tidak terlihat pada keluaran statistik mana pun, padahal keduanya
menentukan apakah hasilnya dapat dipercaya:

* **Berapa uji yang sudah dijalankan.** Menjalankan lima belas uji lalu melaporkan
  satu yang signifikan bukan kecurangan yang disengaja — ia lazimnya terjadi karena
  pengguna lupa menghitung. Peluang menemukan hasil signifikan yang sebenarnya palsu
  naik cepat: pada lima belas uji, peluangnya sudah lebih dari lima puluh persen.
* **Apakah analisisnya sesuai rencana.** Menyimpang dari praregistrasi adalah hal
  biasa dalam penelitian; menyimpang tanpa menyebutkannya tidak.

Jejak ini bukan pengawas, melainkan bahan jawaban ketika penguji bertanya "mengapa
Anda memilih ini". Ia mencatat, dan pengguna yang menjelaskan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

KEPUTUSAN = "keputusan"
UJI = "uji"
PERUBAHAN = "perubahan data"
JENIS = (KEPUTUSAN, UJI, PERUBAHAN)

METODE_KOREKSI = {
    "holm": "Holm — menjaga peluang salah tolak keseluruhan, lebih peka daripada Bonferroni",
    "bonferroni": "Bonferroni — paling ketat dan paling mudah dijelaskan",
    "fdr_bh": "Benjamini-Hochberg — mengendalikan proporsi temuan palsu, cocok untuk penjajakan",
}


@dataclass
class Langkah:
    """Satu langkah yang tercatat."""

    jenis: str
    ringkas: str
    halaman: str = ""
    rincian: str = ""
    p: float | None = None
    waktu: str = ""

    def __post_init__(self) -> None:
        if self.jenis not in JENIS:
            raise ValueError(f"Jenis langkah '{self.jenis}' tidak dikenal. Pilih dari {JENIS}.")
        if not self.waktu:
            self.waktu = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    @property
    def sidik(self) -> str:
        isi = f"{self.jenis}|{self.halaman}|{self.ringkas}"
        return hashlib.sha1(isi.encode("utf-8")).hexdigest()[:12]


@dataclass
class BandingRencana:
    """Perbandingan antara uji yang direncanakan dan yang benar-benar dijalankan."""

    sesuai_rencana: list[str] = field(default_factory=list)
    di_luar_rencana: list[str] = field(default_factory=list)
    direncanakan_belum_jalan: list[str] = field(default_factory=list)
    ada_praregistrasi: bool = False

    def kalimat(self) -> str:
        """Satu kalimat yang siap masuk bagian keterbatasan laporan."""
        if not self.ada_praregistrasi:
            return (
                "Tidak ada praregistrasi, sehingga seluruh analisis dilaporkan sebagai "
                "eksploratori."
            )
        bagian = []
        if self.sesuai_rencana:
            bagian.append(f"{len(self.sesuai_rencana)} uji dijalankan sesuai rencana")
        if self.di_luar_rencana:
            bagian.append(
                f"{len(self.di_luar_rencana)} uji dijalankan di luar rencana "
                f"({', '.join(self.di_luar_rencana)})"
            )
        if self.direncanakan_belum_jalan:
            bagian.append(
                f"{len(self.direncanakan_belum_jalan)} uji yang direncanakan belum dijalankan"
            )
        return ("; ".join(bagian) + ".") if bagian else "Belum ada uji yang dijalankan."


@dataclass
class Jejak:
    """Seluruh langkah yang tercatat pada satu sesi analisis."""

    langkah: list[Langkah] = field(default_factory=list)

    # -- pencatatan --------------------------------------------------------- #

    def catat(
        self,
        jenis: str,
        ringkas: str,
        halaman: str = "",
        rincian: str = "",
        p: float | None = None,
    ) -> Langkah:
        """Catat satu langkah. Langkah yang sama persis tidak dicatat dua kali.

        Menjalankan ulang halaman yang sama karena Streamlit merender ulang bukan
        keputusan baru, dan menghitungnya sebagai uji tambahan akan membuat
        peringatan uji berganda menyala tanpa sebab.
        """
        baru = Langkah(jenis=jenis, ringkas=ringkas, halaman=halaman, rincian=rincian, p=p)
        for lama in self.langkah:
            if lama.sidik == baru.sidik and lama.p == baru.p:
                return lama
        self.langkah.append(baru)
        return baru

    def catat_uji(
        self, nama: str, halaman: str = "", p: float | None = None, rincian: str = ""
    ) -> Langkah:
        return self.catat(UJI, nama, halaman=halaman, rincian=rincian, p=p)

    def catat_keputusan(self, ringkas: str, halaman: str = "", rincian: str = "") -> Langkah:
        return self.catat(KEPUTUSAN, ringkas, halaman=halaman, rincian=rincian)

    def catat_perubahan(self, ringkas: str, halaman: str = "", rincian: str = "") -> Langkah:
        return self.catat(PERUBAHAN, ringkas, halaman=halaman, rincian=rincian)

    # -- pembacaan ---------------------------------------------------------- #

    def kosong(self) -> bool:
        return not self.langkah

    def uji(self) -> list[Langkah]:
        return [l for l in self.langkah if l.jenis == UJI]

    def nilai_p(self) -> list[float]:
        return [l.p for l in self.uji() if l.p is not None]

    def signifikan(self, alfa: float = 0.05) -> list[Langkah]:
        return [l for l in self.uji() if l.p is not None and l.p < alfa]

    def kosongkan(self) -> None:
        self.langkah = []

    # -- uji berganda ------------------------------------------------------- #

    def peringatan_uji_berganda(self, alfa: float = 0.05, ambang: int = 5) -> str:
        """Peringatan bila banyak uji dijalankan dan sebagian dilaporkan signifikan.

        Kosong bila belum melewati ambang, sehingga pemanggilnya dapat menampilkan
        tanpa memeriksa lebih dulu.
        """
        semua = self.nilai_p()
        if len(semua) < ambang:
            return ""
        n = len(semua)
        peluang = 1 - (1 - alfa) ** n
        return (
            f"{n} uji sudah dijalankan pada sesi ini. Bila seluruhnya tidak berhubungan "
            f"sama sekali, peluang memperoleh sekurang-kurangnya satu hasil signifikan "
            f"secara kebetulan adalah {peluang:.0%}. Pertimbangkan koreksi uji berganda "
            "sebelum melaporkan hasil yang signifikan."
        )

    def koreksi_berganda(self, metode: str = "holm", alfa: float = 0.05) -> pd.DataFrame:
        """Terapkan koreksi uji berganda pada seluruh nilai p yang tercatat."""
        from statsmodels.stats.multitest import multipletests

        if metode not in METODE_KOREKSI:
            raise ValueError(
                f"Metode koreksi '{metode}' tidak dikenal. Pilih dari {list(METODE_KOREKSI)}."
            )
        beruji = [l for l in self.uji() if l.p is not None]
        if not beruji:
            raise ValueError("Belum ada uji dengan nilai p yang dapat dikoreksi.")

        p = [float(l.p) for l in beruji]
        tolak, p_koreksi, _, _ = multipletests(p, alpha=alfa, method=metode)
        return pd.DataFrame(
            {
                "Uji": [l.ringkas for l in beruji],
                "Halaman": [l.halaman for l in beruji],
                "p asli": p,
                "p terkoreksi": [float(x) for x in p_koreksi],
                "Signifikan sebelum": [x < alfa for x in p],
                "Signifikan sesudah": [bool(x) for x in tolak],
            }
        )

    def berubah_setelah_koreksi(self, metode: str = "holm", alfa: float = 0.05) -> int:
        """Berapa temuan yang kehilangan signifikansinya setelah dikoreksi."""
        try:
            tabel = self.koreksi_berganda(metode, alfa)
        except ValueError:
            return 0
        return int((tabel["Signifikan sebelum"] & ~tabel["Signifikan sesudah"]).sum())

    # -- praregistrasi ------------------------------------------------------ #

    def banding_rencana(self, praregistrasi) -> BandingRencana:
        """Bandingkan uji yang dijalankan dengan yang dipraregistrasikan."""
        dijalankan = [l.ringkas for l in self.uji()]
        if praregistrasi is None or not praregistrasi.uji_direncanakan:
            return BandingRencana(di_luar_rencana=dijalankan, ada_praregistrasi=False)

        rencana = [r.strip() for r in praregistrasi.uji_direncanakan if r.strip()]
        sesuai, luar = [], []
        for nama in dijalankan:
            cocok = any(_serupa(nama, r) for r in rencana)
            (sesuai if cocok else luar).append(nama)
        belum = [r for r in rencana if not any(_serupa(nama, r) for nama in dijalankan)]
        return BandingRencana(
            sesuai_rencana=sesuai,
            di_luar_rencana=luar,
            direncanakan_belum_jalan=belum,
            ada_praregistrasi=True,
        )

    # -- tampilan ----------------------------------------------------------- #

    def ringkas(self) -> pd.DataFrame:
        """Seluruh langkah dalam satu tabel, untuk lapis Reviewer dan lampiran laporan."""
        return pd.DataFrame(
            [
                {
                    "Waktu": l.waktu,
                    "Jenis": l.jenis.capitalize(),
                    "Halaman": l.halaman,
                    "Langkah": l.ringkas,
                    "p": np.nan if l.p is None else float(l.p),
                    "Keterangan": l.rincian,
                }
                for l in self.langkah
            ]
        )

    def hitungan(self) -> dict[str, int]:
        return {
            "Langkah tercatat": len(self.langkah),
            "Uji dijalankan": len(self.uji()),
            "Hasil signifikan": len(self.signifikan()),
            "Perubahan data": sum(1 for l in self.langkah if l.jenis == PERUBAHAN),
        }

    # -- penyimpanan -------------------------------------------------------- #

    def ke_dict(self) -> dict:
        return {
            "langkah": [
                {
                    "jenis": l.jenis,
                    "ringkas": l.ringkas,
                    "halaman": l.halaman,
                    "rincian": l.rincian,
                    "p": l.p,
                    "waktu": l.waktu,
                }
                for l in self.langkah
            ]
        }

    @classmethod
    def dari_dict(cls, isi: dict | None) -> "Jejak":
        """Pulihkan dari berkas proyek; langkah yang cacat dilewati."""
        if not isinstance(isi, dict):
            return cls()
        hasil = []
        for butir in isi.get("langkah") or []:
            if not isinstance(butir, dict):
                continue
            try:
                nilai = butir.get("p")
                hasil.append(
                    Langkah(
                        jenis=str(butir.get("jenis", KEPUTUSAN)),
                        ringkas=str(butir.get("ringkas", "")),
                        halaman=str(butir.get("halaman", "")),
                        rincian=str(butir.get("rincian", "")),
                        p=None if nilai is None else float(nilai),
                        waktu=str(butir.get("waktu", "")),
                    )
                )
            except (ValueError, TypeError):
                continue
        return cls(hasil)


def _serupa(satu: str, dua: str) -> bool:
    """Dua nama uji dianggap sama bila salah satunya memuat yang lain.

    Pengguna menuliskan rencananya dengan kata-katanya sendiri ("regresi berganda"),
    sedangkan aplikasi mencatat namanya sendiri ("Regresi linear berganda"). Menuntut
    kecocokan persis akan menandai hampir semua uji sebagai di luar rencana.
    """
    a, b = satu.strip().lower(), dua.strip().lower()
    return bool(a) and bool(b) and (a in b or b in a)
