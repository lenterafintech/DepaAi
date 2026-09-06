"""Simpan dan buka proyek: data, keranjang hasil, dan konfigurasi dalam satu berkas.

Sampai kini NalarData tidak menyimpan apa pun. Data yang diunggah, hasil yang
dikumpulkan di keranjang, dan pengaturan cakupan analisis semuanya hilang begitu
sesi berakhir — dan pada layanan hosting yang menidurkan aplikasi, itu terjadi
berkali-kali sehari. Modul ini menutup celah tersebut.

Berkas proyek adalah arsip zip biasa berakhiran ``.nalardata``:

    proyek.json          manifest: format, versi, waktu, daftar isi
    data.csv             data aktif, selalu ada sebagai cadangan yang bisa dibaca
    data.parquet         salinan yang mempertahankan tipe data (bila engine tersedia)
    kamus.json           skala, peran, dan definisi tiap variabel
    konfigurasi.json     pilihan variabel pada halaman laporan
    keranjang.json       daftar hasil yang disimpan pengguna
    keranjang/NN.csv     tabel tiap hasil

CSV selalu ditulis meski Parquet berhasil, supaya proyek tetap terbuka di komputer
yang tidak memiliki engine Parquet — dan supaya isinya dapat diperiksa manusia.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from nalardata import kamus as km
from nalardata import keranjang as kr

FORMAT = "nalardata-proyek"
EKSTENSI = "nalardata"
# Berkas yang disimpan sebelum aplikasi berganti nama tetap dapat dibuka; menolak
# berkas milik pengguna sendiri hanya karena namanya berubah tidak dapat dibenarkan.
FORMAT_LAMA = {"lentera-mva-proyek", "mv-statlab-proyek"}
VERSI = 1
VERSI_DIDUKUNG = {1}

# Batas keamanan saat membuka arsip. Berkas proyek berasal dari luar aplikasi, jadi
# ukurannya diperiksa sebelum apa pun dibaca: arsip kecil yang mengembang menjadi
# raksasa saat dibuka ("zip bomb") dapat menghabiskan memori server.
MAKS_BERKAS = 250 * 1024 * 1024  # 250 MB arsip terkompresi
MAKS_TERBUKA = 500 * 1024 * 1024  # 500 MB setelah dibuka
MAKS_ANGGOTA = 500  # jumlah berkas di dalam arsip


@dataclass
class Proyek:
    """Isi satu berkas proyek yang sudah dibuka."""

    data: pd.DataFrame
    nama_data: str = "data"
    keranjang: kr.Keranjang = field(default_factory=kr.Keranjang)
    kamus: km.Kamus = field(default_factory=km.Kamus)
    konfigurasi: dict = field(default_factory=dict)
    dibuat: str = ""
    versi: int = VERSI

    def ringkas(self) -> pd.DataFrame:
        """Ringkasan isi proyek untuk ditampilkan sebelum dimuat."""
        return pd.DataFrame(
            [
                {"Keterangan": "Sumber data", "Isi": self.nama_data},
                {
                    "Keterangan": "Ukuran data",
                    "Isi": f"{len(self.data)} baris × {self.data.shape[1]} kolom",
                },
                {
                    "Keterangan": "Hasil tersimpan",
                    "Isi": f"{len(self.keranjang.item)} objek",
                },
                {
                    "Keterangan": "Kamus variabel",
                    "Isi": (
                        f"{len(self.kamus)} kolom, "
                        f"{len(self.kamus.perlu_diperiksa())} perlu diperiksa"
                        if len(self.kamus)
                        else "belum ada"
                    ),
                },
                {"Keterangan": "Dibuat", "Isi": self.dibuat or "-"},
                {"Keterangan": "Versi proyek", "Isi": str(self.versi)},
            ]
        )


def _nama_aman(nomor: int, teks: str) -> str:
    bersih = re.sub(r"[^0-9A-Za-z_-]+", "_", str(teks)).strip("_")[:40] or "tabel"
    return f"keranjang/{nomor:02d}_{bersih}.csv"


def _tulis_parquet(arsip: zipfile.ZipFile, nama: str, tabel: pd.DataFrame) -> bool:
    """Tulis Parquet bila engine-nya ada; kegagalan bukan galat karena CSV sudah ada."""
    try:
        penampung = io.BytesIO()
        tabel.to_parquet(penampung, index=False)
        arsip.writestr(nama, penampung.getvalue())
        return True
    except Exception:  # noqa: BLE001 - engine tidak ada atau tipe tak didukung
        return False


# --------------------------------------------------------------------------- #
# Menyimpan
# --------------------------------------------------------------------------- #


def simpan_proyek(
    df: pd.DataFrame,
    nama_data: str = "data",
    keranjang: kr.Keranjang | None = None,
    konfigurasi: dict | None = None,
    kamus: km.Kamus | None = None,
) -> bytes:
    """Susun berkas proyek dari keadaan sesi saat ini."""
    if df is None or df.empty:
        raise ValueError("Tidak ada data aktif yang dapat disimpan.")

    isi = keranjang or kr.Keranjang()
    penampung = io.BytesIO()
    manifest: dict = {
        "format": FORMAT,
        "versi": VERSI,
        "dibuat": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "nama_data": str(nama_data),
        "berkas_data": ["data.csv"],
        "punya_keranjang": not isi.kosong(),
        "punya_kamus": bool(kamus is not None and len(kamus)),
    }

    with zipfile.ZipFile(penampung, "w", zipfile.ZIP_DEFLATED) as arsip:
        arsip.writestr("data.csv", df.to_csv(index=False))
        if _tulis_parquet(arsip, "data.parquet", df):
            # Parquet didahulukan saat dibuka karena tipe datanya utuh.
            manifest["berkas_data"].insert(0, "data.parquet")

        arsip.writestr(
            "konfigurasi.json",
            json.dumps(konfigurasi or {}, ensure_ascii=False, indent=2, default=str),
        )
        arsip.writestr(
            "kamus.json",
            json.dumps(
                kamus.ke_dict() if kamus is not None else {},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )

        if not isi.kosong():
            daftar = []
            for nomor, item in enumerate(isi.item, start=1):
                catatan = {
                    "bagian": item.bagian,
                    "judul": item.judul,
                    "jenis": item.jenis,
                    "teks": item.teks,
                    "catatan": item.catatan,
                    "tanda_data": item.tanda_data,
                    "waktu": item.waktu,
                }
                if item.tabel is not None:
                    berkas = _nama_aman(nomor, f"{item.bagian}_{item.judul}")
                    arsip.writestr(berkas, item.tabel.to_csv(index=False))
                    catatan["tabel"] = berkas
                daftar.append(catatan)
            arsip.writestr(
                "keranjang.json",
                json.dumps(
                    {
                        "judul": isi.judul,
                        "peneliti": isi.peneliti,
                        "item": daftar,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        arsip.writestr("proyek.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return penampung.getvalue()


def nama_berkas_proyek(nama_data: str = "data") -> str:
    dasar = str(nama_data).rsplit(".", 1)[0]
    bersih = re.sub(r"[^0-9A-Za-z_-]+", "_", dasar).strip("_")[:40]
    return f"{bersih or 'nalardata'}.{EKSTENSI}"


# --------------------------------------------------------------------------- #
# Membuka
# --------------------------------------------------------------------------- #


def _periksa_arsip(arsip: zipfile.ZipFile) -> set[str]:
    anggota = arsip.infolist()
    if len(anggota) > MAKS_ANGGOTA:
        raise ValueError(
            f"Proyek memuat {len(anggota)} berkas internal, melebihi batas aman "
            f"{MAKS_ANGGOTA}."
        )
    total = sum(a.file_size for a in anggota)
    if total > MAKS_TERBUKA:
        raise ValueError(
            f"Isi proyek membengkak menjadi {total / 1024 / 1024:.0f} MB setelah dibuka, "
            f"melebihi batas aman {MAKS_TERBUKA // 1024 // 1024} MB."
        )
    nama = {a.filename for a in anggota}
    if "proyek.json" not in nama:
        raise ValueError(
            "Berkas ini bukan proyek NalarData: proyek.json tidak ditemukan di dalamnya."
        )
    return nama


def _baca_data(arsip: zipfile.ZipFile, nama: set[str], manifest: dict) -> pd.DataFrame:
    urutan = list(manifest.get("berkas_data") or [])
    urutan += [n for n in ("data.parquet", "data.csv") if n not in urutan]
    galat_parquet: Exception | None = None

    for berkas in urutan:
        if berkas not in nama:
            continue
        if berkas.endswith(".parquet"):
            try:
                return pd.read_parquet(io.BytesIO(arsip.read(berkas)))
            except Exception as exc:  # noqa: BLE001 - jatuh ke CSV
                galat_parquet = exc
                continue
        if berkas.endswith(".csv"):
            return pd.read_csv(io.BytesIO(arsip.read(berkas)))

    if galat_parquet is not None:
        raise ValueError(
            "Data proyek tersimpan sebagai Parquet, tetapi tidak ada engine yang "
            "dapat membacanya dan salinan CSV-nya tidak ditemukan."
        ) from galat_parquet
    raise ValueError("Proyek tidak memuat data utama.")


def _baca_keranjang(arsip: zipfile.ZipFile, nama: set[str]) -> kr.Keranjang:
    if "keranjang.json" not in nama:
        return kr.Keranjang()

    isi_mentah = json.loads(arsip.read("keranjang.json"))
    hasil = kr.Keranjang(
        judul=str(isi_mentah.get("judul") or "Laporan Hasil Analisis Statistik"),
        peneliti=str(isi_mentah.get("peneliti") or ""),
    )
    for catatan in isi_mentah.get("item", []):
        tabel = None
        berkas = catatan.get("tabel")
        if berkas:
            if berkas not in nama:
                # Satu tabel hilang tidak boleh menggagalkan seluruh proyek.
                continue
            tabel = pd.read_csv(io.BytesIO(arsip.read(berkas)))
        try:
            hasil.tambah(
                kr.Item(
                    bagian=str(catatan.get("bagian", "Hasil")),
                    judul=str(catatan.get("judul", "Tabel")),
                    jenis=str(catatan.get("jenis", "tabel")),
                    tabel=tabel,
                    teks=str(catatan.get("teks", "")),
                    catatan=str(catatan.get("catatan", "")),
                    tanda_data=str(catatan.get("tanda_data", "")),
                    waktu=str(catatan.get("waktu", "")),
                )
            )
        except ValueError:
            # Item yang tidak lagi sah dilewati, sisanya tetap dipulihkan.
            continue
    return hasil


def _baca_json(arsip: zipfile.ZipFile, nama: set[str], berkas: str) -> dict:
    """Baca satu berkas JSON pendamping; isi yang rusak menjadi kosong, bukan galat."""
    if berkas not in nama:
        return {}
    try:
        isi = json.loads(arsip.read(berkas))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return isi if isinstance(isi, dict) else {}


def buka_proyek(isi: bytes) -> Proyek:
    """Baca berkas proyek dan kembalikan isinya.

    Seluruh kegagalan dinyatakan sebagai ``ValueError`` berbahasa Indonesia agar
    halaman dapat menampilkannya apa adanya kepada pengguna.
    """
    if not isi:
        raise ValueError("Berkas proyek kosong.")
    if len(isi) > MAKS_BERKAS:
        raise ValueError(
            f"Berkas proyek berukuran {len(isi) / 1024 / 1024:.0f} MB, melebihi batas "
            f"aman {MAKS_BERKAS // 1024 // 1024} MB."
        )

    try:
        with zipfile.ZipFile(io.BytesIO(isi)) as arsip:
            nama = _periksa_arsip(arsip)
            manifest = json.loads(arsip.read("proyek.json"))

            format_berkas = manifest.get("format")
            if format_berkas not in {FORMAT, *FORMAT_LAMA}:
                raise ValueError(
                    "Format berkas tidak dikenali sebagai proyek NalarData."
                )
            try:
                versi = int(manifest.get("versi"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Nomor versi proyek tidak sah.") from exc
            if versi not in VERSI_DIDUKUNG:
                raise ValueError(
                    f"Proyek versi {versi} belum didukung. Versi yang dikenali: "
                    f"{', '.join(str(v) for v in sorted(VERSI_DIDUKUNG))}."
                )

            data = _baca_data(arsip, nama, manifest)
            if data.shape[1] == 0:
                raise ValueError("Proyek tidak memuat satu kolom pun.")

            konfigurasi = _baca_json(arsip, nama, "konfigurasi.json")
            # Kamus yang rusak dipulihkan sebagian; menolak seluruh proyek karena
            # satu keterangan variabel cacat akan menghukum pengguna terlalu keras.
            kamus = km.Kamus.dari_dict(_baca_json(arsip, nama, "kamus.json"))
            if len(kamus):
                kamus = kamus.selaraskan(data)

            return Proyek(
                data=data,
                nama_data=str(manifest.get("nama_data") or "data"),
                keranjang=_baca_keranjang(arsip, nama),
                kamus=kamus,
                konfigurasi=konfigurasi,
                dibuat=str(manifest.get("dibuat") or ""),
                versi=versi,
            )
    except zipfile.BadZipFile as exc:
        raise ValueError(
            "Berkas rusak atau bukan arsip proyek yang sah."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
        raise ValueError("Struktur proyek tidak lengkap atau rusak.") from exc
