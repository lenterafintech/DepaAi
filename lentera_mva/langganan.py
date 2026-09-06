"""Paket langganan dan pembatasan fitur.

Modul ini murni logika: ia mendefinisikan paket, batas pemakaian, dan aturan
siapa boleh memakai fitur apa. Penyimpanan status langganan serta antarmukanya
berada di tempat lain, sehingga aturan di sini dapat diuji tanpa Streamlit dan
kelak dapat dipasangkan ke basis data mana pun.

Perlu ditegaskan: belum ada pembayaran sungguhan di sini. Kerangka ini menyiapkan
tempat bagi penagihan yang akan dipasang kemudian.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Kode fitur yang dapat dibatasi. Dipakai sebagai kunci di seluruh aplikasi.
FITUR = {
    "dasar": "Eksplorasi, korelasi, dan uji asumsi",
    "reduksi": "PCA dan analisis faktor eksploratori",
    "klaster": "Analisis klaster",
    "regresi": "Regresi linear dan logistik",
    "uji_beda": "MANOVA, diskriminan, korelasi kanonik",
    "instrumen": "Reliabilitas dan validitas instrumen",
    "sem": "CFA, analisis jalur, dan SEM",
    "ringkasan_eksekutif": "Ringkasan eksekutif",
    "ringkasan_akademik": "Ringkasan akademik",
    "ringkasan_profesional": "Ringkasan profesional",
    "entri_data": "Membuat dan menyunting data di aplikasi",
    "unduh_laporan": "Mengunduh laporan HTML dan Markdown",
}


@dataclass(frozen=True)
class Paket:
    """Satu tingkatan langganan beserta batas pemakaiannya."""

    kode: str
    nama: str
    harga_bulanan: int  # rupiah; 0 berarti gratis
    ringkas: str
    maks_baris: int
    maks_variabel: int
    fitur: frozenset[str] = field(default_factory=frozenset)

    def punya(self, kode_fitur: str) -> bool:
        return kode_fitur in self.fitur

    def fitur_terkunci(self) -> list[str]:
        return [k for k in FITUR if k not in self.fitur]


_DASAR = {"dasar", "reduksi", "klaster", "regresi", "entri_data", "ringkasan_eksekutif"}
_LENGKAP = set(FITUR)

PAKET: dict[str, Paket] = {
    "gratis": Paket(
        kode="gratis",
        nama="Gratis",
        harga_bulanan=0,
        ringkas="Untuk mencoba dan mengerjakan analisis berskala kecil.",
        maks_baris=300,
        maks_variabel=10,
        fitur=frozenset(_DASAR),
    ),
    "pro": Paket(
        kode="pro",
        nama="Pro",
        harga_bulanan=99_000,
        ringkas="Seluruh metode, ketiga ringkasan, dan unduhan laporan.",
        maks_baris=50_000,
        maks_variabel=100,
        fitur=frozenset(_LENGKAP),
    ),
    "institusi": Paket(
        kode="institusi",
        nama="Institusi",
        harga_bulanan=750_000,
        ringkas="Untuk kampus dan perusahaan; data besar dan banyak pengguna.",
        maks_baris=500_000,
        maks_variabel=500,
        fitur=frozenset(_LENGKAP),
    ),
}
PAKET_BAWAAN = "gratis"


def ambil_paket(kode: str | None) -> Paket:
    """Paket berdasarkan kodenya; kode tak dikenal jatuh ke paket gratis."""
    return PAKET.get(str(kode or "").lower(), PAKET[PAKET_BAWAAN])


@dataclass
class Pelanggaran:
    """Alasan sebuah tindakan tidak diizinkan pada paket yang sedang aktif."""

    jenis: str  # fitur | baris | variabel
    pesan: str
    saran_paket: str | None = None


def periksa_fitur(paket: Paket, kode_fitur: str) -> Pelanggaran | None:
    if paket.punya(kode_fitur):
        return None
    nama_fitur = FITUR.get(kode_fitur, kode_fitur)
    return Pelanggaran(
        jenis="fitur",
        pesan=f"{nama_fitur} tidak termasuk dalam paket {paket.nama}.",
        saran_paket=paket_terkecil_dengan(kode_fitur),
    )


def _ribuan(angka: int) -> str:
    """Angka dengan titik sebagai pemisah ribuan, tanpa menyentuh tanda baca lain."""
    return f"{int(angka):,}".replace(",", ".")


def periksa_ukuran(paket: Paket, n_baris: int, n_kolom: int) -> Pelanggaran | None:
    """Periksa apakah ukuran data masih di dalam batas paket."""
    if n_baris > paket.maks_baris:
        return Pelanggaran(
            jenis="baris",
            pesan=(
                f"Data berisi {_ribuan(n_baris)} baris, sedangkan paket {paket.nama} "
                f"membatasi {_ribuan(paket.maks_baris)} baris."
            ),
            saran_paket=paket_terkecil_untuk_ukuran(n_baris, n_kolom),
        )
    if n_kolom > paket.maks_variabel:
        return Pelanggaran(
            jenis="variabel",
            pesan=(
                f"Data berisi {n_kolom} kolom, sedangkan paket {paket.nama} membatasi "
                f"{paket.maks_variabel} kolom."
            ),
            saran_paket=paket_terkecil_untuk_ukuran(n_baris, n_kolom),
        )
    return None


def _urut_harga() -> list[Paket]:
    return sorted(PAKET.values(), key=lambda p: p.harga_bulanan)


def paket_terkecil_dengan(kode_fitur: str) -> str | None:
    """Paket termurah yang memuat sebuah fitur."""
    for paket in _urut_harga():
        if paket.punya(kode_fitur):
            return paket.kode
    return None


def paket_terkecil_untuk_ukuran(n_baris: int, n_kolom: int) -> str | None:
    for paket in _urut_harga():
        if n_baris <= paket.maks_baris and n_kolom <= paket.maks_variabel:
            return paket.kode
    return None


def ringkas_paket(paket: Paket) -> list[tuple[str, str]]:
    """Daftar (label, nilai) untuk ditampilkan pada kartu paket."""
    return [
        (
            "Harga",
            "Gratis"
            if not paket.harga_bulanan
            else f"Rp {_ribuan(paket.harga_bulanan)}/bulan",
        ),
        ("Batas data", f"{_ribuan(paket.maks_baris)} baris · {paket.maks_variabel} kolom"),
        ("Metode", f"{len(paket.fitur)} dari {len(FITUR)} fitur"),
    ]
