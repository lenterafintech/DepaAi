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

# Batas ukuran data ditetapkan agar paket Gratis muat memuat satu penelitian
# kuesioner yang sungguh-sungguh: skripsi Indonesia lazimnya 100-400 responden
# dengan 20-40 butir ditambah data demografi. Pembeda antar paket adalah
# metodenya - SEM, panel, ekspor laporan, simulasi sidang - bukan banyaknya
# baris. Membatasi ukuran justru menutup pintu bagi pengguna yang paling
# membutuhkan pemandu uji, yakni yang belum mampu berlangganan.

# Kode fitur yang dapat dibatasi. Dipakai sebagai kunci di seluruh aplikasi.
FITUR = {
    "dasar": "Eksplorasi, korelasi, dan uji asumsi",
    "kamus": "Kamus variabel: skala, peran, dan definisi operasional",
    "pemandu": "Pemandu uji: saran metode berdasarkan data Anda",
    "reduksi": "PCA dan analisis faktor eksploratori",
    "klaster": "Analisis klaster",
    "regresi": "Regresi linear dan logistik",
    "uji_beda": "MANOVA, diskriminan, korelasi kanonik",
    "nonparametrik": "Uji non-parametrik dan uji lanjutan",
    "instrumen": "Reliabilitas dan validitas instrumen",
    "sem": "CFA, analisis jalur, dan SEM",
    "ringkasan_eksekutif": "Ringkasan eksekutif",
    "ringkasan_akademik": "Ringkasan akademik",
    "ringkasan_profesional": "Ringkasan profesional",
    "entri_data": "Membuat dan menyunting data di aplikasi",
    "keranjang": "Mengumpulkan hasil analisis menjadi satu laporan",
    "unduh_laporan": "Ekspor laporan (Word, PDF, Excel, PPT, HTML, sintaks)",
    "sidang": "Simulasi sidang: latihan menjawab pertanyaan penguji",
}


@dataclass(frozen=True)
class Paket:
    """Satu tingkatan langganan beserta batas pemakaiannya."""

    kode: str
    nama: str
    harga_bulanan: int  # rupiah; 0 berarti gratis atau ditetapkan per kesepakatan
    ringkas: str
    maks_baris: int
    maks_variabel: int
    # Urutan tingkatan ditetapkan sendiri, bukan diturunkan dari harga: paket
    # institusi berharga kesepakatan (0) sehingga pengurutan berdasarkan harga
    # akan keliru menganggapnya paket paling murah.
    urutan: int = 0
    fitur: frozenset[str] = field(default_factory=frozenset)

    def punya(self, kode_fitur: str) -> bool:
        return kode_fitur in self.fitur

    def fitur_terkunci(self) -> list[str]:
        return [k for k in FITUR if k not in self.fitur]


# Kamus variabel masuk paket gratis dengan sengaja: pengguna yang paling
# membutuhkannya justru yang belum menguasai skala pengukuran.
_DASAR = {
    "dasar",
    "kamus",
    "pemandu",
    "reduksi",
    "klaster",
    "regresi",
    "nonparametrik",
    "entri_data",
    "keranjang",
    "ringkasan_eksekutif",
}
_AKADEMIK = _DASAR | {
    "uji_beda",
    "instrumen",
    "sem",
    "ringkasan_akademik",
    "unduh_laporan",
    "sidang",
}
_LENGKAP = set(FITUR)

# Harga dicantumkan sebagai rencana. Selama masa perkenalan seluruh paket dapat
# diaktifkan tanpa pembayaran; halaman akun menyatakan hal itu secara terbuka.
PAKET: dict[str, Paket] = {
    "gratis": Paket(
        kode="gratis",
        urutan=0,
        nama="Gratis",
        harga_bulanan=0,
        ringkas=(
            "Cukup untuk satu penelitian kuesioner utuh: pemandu uji, deskriptif, "
            "uji beda, korelasi, dan regresi."
        ),
        maks_baris=1_000,
        maks_variabel=50,
        fitur=frozenset(_DASAR),
    ),
    "mahasiswa": Paket(
        kode="mahasiswa",
        urutan=1,
        nama="Mahasiswa & Pengajar",
        harga_bulanan=49_000,
        ringkas=(
            "Seluruh metode yang dibutuhkan skripsi, tesis, dan penelitian kelas, "
            "termasuk CFA dan SEM."
        ),
        maks_baris=25_000,
        maks_variabel=200,
        fitur=frozenset(_AKADEMIK),
    ),
    "profesional": Paket(
        kode="profesional",
        urutan=2,
        nama="Profesional",
        harga_bulanan=149_000,
        ringkas="Seluruh fitur, ketiga ringkasan, dan data berukuran kerja.",
        maks_baris=250_000,
        maks_variabel=500,
        fitur=frozenset(_LENGKAP),
    ),
    "institusi": Paket(
        kode="institusi",
        urutan=3,
        nama="Institusi (Khusus)",
        harga_bulanan=0,  # ditetapkan per kesepakatan
        ringkas=(
            "Untuk kampus dan perusahaan: banyak pengguna, data besar, dan "
            "kebutuhan khusus. Harga dan ketentuan disepakati tersendiri."
        ),
        maks_baris=2_000_000,
        maks_variabel=1_000,
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


def urut_tingkatan() -> list[Paket]:
    """Paket dari yang paling terbatas ke yang paling lengkap."""
    return sorted(PAKET.values(), key=lambda p: p.urutan)


def paket_terkecil_dengan(kode_fitur: str) -> str | None:
    """Paket termurah yang memuat sebuah fitur."""
    for paket in urut_tingkatan():
        if paket.punya(kode_fitur):
            return paket.kode
    return None


def paket_terkecil_untuk_ukuran(n_baris: int, n_kolom: int) -> str | None:
    for paket in urut_tingkatan():
        if n_baris <= paket.maks_baris and n_kolom <= paket.maks_variabel:
            return paket.kode
    return None


def harga_tampil(paket: Paket) -> str:
    """Harga yang ditampilkan; paket institusi ditawarkan lewat kesepakatan."""
    if paket.kode == "institusi":
        return "Sesuai kesepakatan"
    if not paket.harga_bulanan:
        return "Gratis"
    return f"Rp {_ribuan(paket.harga_bulanan)}/bulan"


def ringkas_paket(paket: Paket) -> list[tuple[str, str]]:
    """Daftar (label, nilai) untuk ditampilkan pada kartu paket."""
    return [
        ("Harga", harga_tampil(paket)),
        ("Batas data", f"{_ribuan(paket.maks_baris)} baris · {paket.maks_variabel} kolom"),
        ("Metode", f"{len(paket.fitur)} dari {len(FITUR)} fitur"),
    ]
