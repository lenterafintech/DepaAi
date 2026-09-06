"""Basis data pengguna, autentikasi, dan masa uji coba.

Penyimpanan memakai SQLite karena aplikasi ini dijalankan sebagai satu proses di
satu server: tidak perlu layanan basis data terpisah, berkasnya mudah dicadangkan,
dan pemindahan ke PostgreSQL kelak hanya menyentuh modul ini.

Kata sandi tidak pernah disimpan apa adanya. Yang disimpan adalah hasil PBKDF2-
HMAC-SHA256 dengan garam acak per pengguna, sehingga isi basis data yang bocor
tetap tidak membuka kata sandi siapa pun.

Batas yang perlu diketahui: modul ini belum memuat verifikasi alamat surel,
pemulihan kata sandi, maupun pembatasan percobaan masuk. Ketiganya diperlukan
sebelum aplikasi dibuka untuk umum.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Panjang masa uji coba: seluruh fitur terbuka selama tiga hari sejak pendaftaran.
HARI_UJI_COBA = 3
PAKET_UJI_COBA = "profesional"

ITERASI_PBKDF2 = 200_000
PANJANG_GARAM = 16
POLA_SUREL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")
PANJANG_SANDI_MINIMAL = 8

BERKAS_BAWAAN = Path(__file__).resolve().parents[1] / "data" / "pengguna.db"

SKEMA = """
CREATE TABLE IF NOT EXISTS pengguna (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    surel TEXT NOT NULL UNIQUE,
    nama TEXT NOT NULL,
    sandi_hash TEXT NOT NULL,
    garam TEXT NOT NULL,
    paket TEXT NOT NULL DEFAULT 'gratis',
    institusi TEXT,
    dibuat TEXT NOT NULL,
    uji_coba_selesai TEXT,
    langganan_selesai TEXT,
    terakhir_masuk TEXT,
    aktif INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_pengguna_surel ON pengguna(surel);
"""


@dataclass
class Pengguna:
    """Satu akun beserta status langganannya."""

    id: int
    surel: str
    nama: str
    paket: str
    institusi: str | None
    dibuat: datetime
    uji_coba_selesai: datetime | None
    langganan_selesai: datetime | None
    terakhir_masuk: datetime | None
    aktif: bool

    def dalam_uji_coba(self, sekarang: datetime | None = None) -> bool:
        if self.uji_coba_selesai is None:
            return False
        return (sekarang or _sekarang()) < self.uji_coba_selesai

    def sisa_uji_coba(self, sekarang: datetime | None = None) -> timedelta:
        if not self.dalam_uji_coba(sekarang):
            return timedelta(0)
        return self.uji_coba_selesai - (sekarang or _sekarang())

    def langganan_berlaku(self, sekarang: datetime | None = None) -> bool:
        """Paket berbayar dianggap berlaku bila belum melewati tanggal berakhirnya."""
        if self.paket == "gratis":
            return True
        if self.langganan_selesai is None:
            # Selama masa perkenalan, paket berbayar diaktifkan tanpa tanggal akhir.
            return True
        return (sekarang or _sekarang()) < self.langganan_selesai

    def paket_efektif(self, sekarang: datetime | None = None) -> str:
        """Paket yang benar-benar berlaku saat ini, memperhitungkan uji coba."""
        if self.dalam_uji_coba(sekarang):
            return PAKET_UJI_COBA
        if not self.langganan_berlaku(sekarang):
            return "gratis"
        return self.paket

    def alasan_paket(self, sekarang: datetime | None = None) -> str:
        if self.dalam_uji_coba(sekarang):
            sisa = self.sisa_uji_coba(sekarang)
            hari = sisa.days
            jam = sisa.seconds // 3600
            return f"Masa uji coba, sisa {hari} hari {jam} jam"
        if self.paket != "gratis" and not self.langganan_berlaku(sekarang):
            return "Langganan berakhir; kembali ke paket Gratis"
        return "Paket aktif"


class GalatPengguna(Exception):
    """Kesalahan yang pesannya aman ditampilkan kepada pengguna."""


def _sekarang() -> datetime:
    return datetime.now(timezone.utc)


def _teks_waktu(waktu: datetime | None) -> str | None:
    return waktu.isoformat() if waktu else None


def _baca_waktu(teks: str | None) -> datetime | None:
    if not teks:
        return None
    waktu = datetime.fromisoformat(teks)
    return waktu if waktu.tzinfo else waktu.replace(tzinfo=timezone.utc)


def jalur_basis_data() -> Path:
    """Lokasi berkas basis data; dapat diarahkan lewat LENTERA_DB."""
    return Path(os.environ.get("LENTERA_DB", str(BERKAS_BAWAAN)))


@contextmanager
def koneksi(berkas: Path | str | None = None):
    jalur = Path(berkas) if berkas else jalur_basis_data()
    jalur.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(jalur)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def siapkan(berkas: Path | str | None = None) -> None:
    """Buat tabel bila belum ada. Aman dipanggil berkali-kali."""
    with koneksi(berkas) as conn:
        conn.executescript(SKEMA)


# --------------------------------------------------------------------------- #
# Kata sandi
# --------------------------------------------------------------------------- #


def _hash_sandi(sandi: str, garam: str) -> str:
    turunan = hashlib.pbkdf2_hmac(
        "sha256", sandi.encode("utf-8"), bytes.fromhex(garam), ITERASI_PBKDF2
    )
    return turunan.hex()


def periksa_kekuatan_sandi(sandi: str) -> None:
    if len(sandi) < PANJANG_SANDI_MINIMAL:
        raise GalatPengguna(
            f"Kata sandi minimal {PANJANG_SANDI_MINIMAL} karakter."
        )
    if sandi.isdigit() or sandi.isalpha():
        raise GalatPengguna("Gabungkan huruf dan angka agar kata sandi lebih kuat.")


def bakukan_surel(surel: str) -> str:
    bersih = str(surel).strip().lower()
    if not POLA_SUREL.match(bersih):
        raise GalatPengguna("Alamat surel tidak valid.")
    return bersih


# --------------------------------------------------------------------------- #
# Operasi akun
# --------------------------------------------------------------------------- #


def _dari_baris(baris: sqlite3.Row) -> Pengguna:
    return Pengguna(
        id=int(baris["id"]),
        surel=baris["surel"],
        nama=baris["nama"],
        paket=baris["paket"],
        institusi=baris["institusi"],
        dibuat=_baca_waktu(baris["dibuat"]),
        uji_coba_selesai=_baca_waktu(baris["uji_coba_selesai"]),
        langganan_selesai=_baca_waktu(baris["langganan_selesai"]),
        terakhir_masuk=_baca_waktu(baris["terakhir_masuk"]),
        aktif=bool(baris["aktif"]),
    )


def daftar(
    surel: str,
    nama: str,
    sandi: str,
    paket: str = "gratis",
    institusi: str | None = None,
    berkas: Path | str | None = None,
) -> Pengguna:
    """Buat akun baru sekaligus memulai masa uji coba tiga hari."""
    surel = bakukan_surel(surel)
    nama = str(nama).strip()
    if not nama:
        raise GalatPengguna("Nama tidak boleh kosong.")
    periksa_kekuatan_sandi(sandi)

    garam = secrets.token_hex(PANJANG_GARAM)
    sekarang = _sekarang()
    siapkan(berkas)
    with koneksi(berkas) as conn:
        ada = conn.execute("SELECT 1 FROM pengguna WHERE surel = ?", (surel,)).fetchone()
        if ada:
            raise GalatPengguna("Surel ini sudah terdaftar. Silakan masuk.")
        conn.execute(
            """
            INSERT INTO pengguna
                (surel, nama, sandi_hash, garam, paket, institusi, dibuat, uji_coba_selesai)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                surel,
                nama,
                _hash_sandi(sandi, garam),
                garam,
                paket,
                institusi,
                _teks_waktu(sekarang),
                _teks_waktu(sekarang + timedelta(days=HARI_UJI_COBA)),
            ),
        )
    return ambil(surel, berkas)


def masuk(surel: str, sandi: str, berkas: Path | str | None = None) -> Pengguna:
    """Verifikasi kredensial dan catat waktu masuk."""
    surel = bakukan_surel(surel)
    siapkan(berkas)
    with koneksi(berkas) as conn:
        baris = conn.execute(
            "SELECT * FROM pengguna WHERE surel = ?", (surel,)
        ).fetchone()
        # Pesan gagal sengaja sama untuk surel tak dikenal maupun sandi salah,
        # supaya tidak menjadi cara menebak surel mana yang terdaftar.
        if baris is None:
            raise GalatPengguna("Surel atau kata sandi salah.")
        if not hmac.compare_digest(
            baris["sandi_hash"], _hash_sandi(sandi, baris["garam"])
        ):
            raise GalatPengguna("Surel atau kata sandi salah.")
        if not baris["aktif"]:
            raise GalatPengguna("Akun ini dinonaktifkan. Hubungi pengelola.")
        conn.execute(
            "UPDATE pengguna SET terakhir_masuk = ? WHERE id = ?",
            (_teks_waktu(_sekarang()), baris["id"]),
        )
    return ambil(surel, berkas)


def ambil(surel: str, berkas: Path | str | None = None) -> Pengguna:
    siapkan(berkas)
    with koneksi(berkas) as conn:
        baris = conn.execute(
            "SELECT * FROM pengguna WHERE surel = ?", (bakukan_surel(surel),)
        ).fetchone()
    if baris is None:
        raise GalatPengguna("Pengguna tidak ditemukan.")
    return _dari_baris(baris)


def ambil_dengan_id(id_pengguna: int, berkas: Path | str | None = None) -> Pengguna | None:
    siapkan(berkas)
    with koneksi(berkas) as conn:
        baris = conn.execute(
            "SELECT * FROM pengguna WHERE id = ?", (int(id_pengguna),)
        ).fetchone()
    return _dari_baris(baris) if baris else None


def set_paket(
    surel: str,
    paket: str,
    berlaku_sampai: datetime | None = None,
    institusi: str | None = None,
    berkas: Path | str | None = None,
) -> Pengguna:
    """Ubah paket pengguna; tanggal berakhir kosong berarti tanpa batas waktu."""
    siapkan(berkas)
    with koneksi(berkas) as conn:
        conn.execute(
            "UPDATE pengguna SET paket = ?, langganan_selesai = ?, institusi = COALESCE(?, institusi) WHERE surel = ?",
            (paket, _teks_waktu(berlaku_sampai), institusi, bakukan_surel(surel)),
        )
    return ambil(surel, berkas)


def ganti_sandi(
    surel: str, sandi_lama: str, sandi_baru: str, berkas: Path | str | None = None
) -> None:
    pengguna = masuk(surel, sandi_lama, berkas)  # memverifikasi sandi lama
    periksa_kekuatan_sandi(sandi_baru)
    garam = secrets.token_hex(PANJANG_GARAM)
    with koneksi(berkas) as conn:
        conn.execute(
            "UPDATE pengguna SET sandi_hash = ?, garam = ? WHERE id = ?",
            (_hash_sandi(sandi_baru, garam), garam, pengguna.id),
        )


def akhiri_uji_coba(surel: str, berkas: Path | str | None = None) -> Pengguna:
    """Hentikan masa uji coba lebih awal (dipakai pengujian dan pengelolaan)."""
    with koneksi(berkas) as conn:
        conn.execute(
            "UPDATE pengguna SET uji_coba_selesai = ? WHERE surel = ?",
            (_teks_waktu(_sekarang() - timedelta(seconds=1)), bakukan_surel(surel)),
        )
    return ambil(surel, berkas)


def jumlah_pengguna(berkas: Path | str | None = None) -> int:
    siapkan(berkas)
    with koneksi(berkas) as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM pengguna").fetchone()["n"])


def daftar_pengguna(berkas: Path | str | None = None) -> list[Pengguna]:
    """Seluruh akun, terbaru lebih dulu. Dipakai halaman pengelolaan."""
    siapkan(berkas)
    with koneksi(berkas) as conn:
        baris = conn.execute("SELECT * FROM pengguna ORDER BY id DESC").fetchall()
    return [_dari_baris(b) for b in baris]
