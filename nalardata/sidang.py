"""Simulasi Sidang: pertanyaan penguji yang lahir dari analisis Anda sendiri.

Janji jasa olah data di Indonesia bukan "kami kirim output", melainkan "didampingi
sampai paham, siap sidang". Itu penanda bahwa yang sesungguhnya dibeli bukan angka,
melainkan kemampuan mempertanggungjawabkan angka itu di depan penguji.

Modul ini menyusun pertanyaan penguji dari keadaan analisis yang benar-benar
dijalankan — rancangan penelitiannya, asumsi yang dilanggar, banyaknya uji yang
dicoba, temuan yang tidak signifikan — beserta jawaban model yang diturunkan dari
angka pengguna sendiri, bukan dari contoh buku.

Yang membuatnya berbeda dari daftar "pertanyaan sidang yang sering muncul" di
internet: pertanyaan di sini **tidak dapat disusun tanpa melihat hasil Anda**. Ia
menanyakan R² Anda, kelompok Anda yang gagal uji normalitas, dan uji Anda yang di
luar praregistrasi.

Batasnya dinyatakan terbuka pada antarmukanya: penguji sungguhan dapat menanyakan
apa saja, termasuk hal yang tidak terbaca dari data. Lulus simulasi ini bukan
jaminan lulus sidang.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# Kategori pertanyaan, dipakai mengelompokkan dan memastikan cakupannya merata.
RANCANGAN = "rancangan"
METODE = "metode"
ASUMSI = "asumsi"
HASIL = "hasil"
INTEGRITAS = "integritas"
DATA = "data"
KATEGORI = (RANCANGAN, METODE, ASUMSI, HASIL, INTEGRITAS, DATA)

LABEL_KATEGORI = {
    RANCANGAN: "Rancangan penelitian",
    METODE: "Pilihan metode",
    ASUMSI: "Asumsi statistik",
    HASIL: "Tafsir hasil",
    INTEGRITAS: "Integritas analisis",
    DATA: "Mutu data",
}

WAJIB = "wajib"
MUNGKIN = "mungkin"

# Jumlah pertanyaan yang disodorkan sekaligus. Lebih dari ini berubah menjadi
# daftar yang dilewati, bukan latihan yang dikerjakan.
MAKS_PERTANYAAN = 12


@dataclass
class Pertanyaan:
    """Satu pertanyaan penguji beserta jawaban model dan kata kuncinya."""

    pertanyaan: str
    jawaban: str
    kategori: str
    bobot: str = WAJIB
    kunci: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kategori not in KATEGORI:
            raise ValueError(f"Kategori '{self.kategori}' tidak dikenal.")
        if self.bobot not in {WAJIB, MUNGKIN}:
            raise ValueError(f"Bobot '{self.bobot}' tidak dikenal.")

    def nilai(self, jawaban_pengguna: str) -> tuple[bool, list[str]]:
        """Periksa apakah jawaban pengguna menyentuh butir-butir kuncinya.

        Pemeriksaannya kasar dan disebut kasar di antarmuka: ia hanya melihat
        apakah gagasan kuncinya disebut, bukan apakah kalimatnya benar. Menilai
        pemahaman bukan wewenang aplikasi — yang dapat dilakukannya adalah
        menunjukkan butir yang belum disinggung sama sekali.
        """
        teks = _bakukan(jawaban_pengguna)
        terlewat = [k for k in self.kunci if _bakukan(k) not in teks]
        cukup = len(terlewat) <= len(self.kunci) // 3
        return (bool(self.kunci) and cukup and bool(teks.strip())), terlewat


def _bakukan(teks: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(teks).lower())


@dataclass
class Simulasi:
    """Seluruh pertanyaan untuk satu analisis."""

    pertanyaan: list[Pertanyaan] = field(default_factory=list)

    @property
    def wajib(self) -> list[Pertanyaan]:
        return [p for p in self.pertanyaan if p.bobot == WAJIB]

    def kosong(self) -> bool:
        return not self.pertanyaan

    def per_kategori(self) -> dict[str, list[Pertanyaan]]:
        hasil: dict[str, list[Pertanyaan]] = {}
        for butir in self.pertanyaan:
            hasil.setdefault(butir.kategori, []).append(butir)
        return hasil

    def ringkas(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Kategori": LABEL_KATEGORI[p.kategori],
                    "Pertanyaan": p.pertanyaan,
                    "Bobot": p.bobot,
                }
                for p in self.pertanyaan
            ]
        )


# --------------------------------------------------------------------------- #
# Penyusun pertanyaan
# --------------------------------------------------------------------------- #


def susun(laporan, penelitian=None, jejak=None, audit=None) -> Simulasi:
    """Susun pertanyaan sidang dari keadaan analisis yang benar-benar dijalankan.

    Seluruh masukan boleh kosong: pengguna yang melewati Ruang Proyek tetap
    memperoleh pertanyaan, dan justru pertanyaan tentang apa yang belum ia isi.
    """
    hasil = Simulasi()
    for bangun in (
        _tanya_rancangan,
        _tanya_sampel,
        _tanya_metode,
        _tanya_asumsi,
        _tanya_hasil,
        _tanya_integritas,
        _tanya_data,
    ):
        hasil.pertanyaan.extend(bangun(laporan, penelitian, jejak, audit))

    # Yang wajib didahulukan; sisanya mengisi sampai batas.
    wajib = [p for p in hasil.pertanyaan if p.bobot == WAJIB]
    mungkin = [p for p in hasil.pertanyaan if p.bobot == MUNGKIN]
    hasil.pertanyaan = (wajib + mungkin)[:MAKS_PERTANYAAN]
    return hasil


def _tanya_rancangan(laporan, penelitian, jejak, audit) -> list[Pertanyaan]:
    if penelitian is None or penelitian.kosong():
        return [
            Pertanyaan(
                pertanyaan=(
                    "Apa rancangan penelitian Anda, dan apa yang boleh Anda simpulkan "
                    "darinya?"
                ),
                jawaban=(
                    "Rancangan penelitian belum diisi pada Ruang Proyek, sehingga "
                    "aplikasi memakai anggapan paling berhati-hati: hasil dibaca "
                    "sebagai hubungan, bukan sebab-akibat, dan kesimpulan berlaku "
                    "bagi responden yang diteliti saja. Isi Ruang Proyek agar batas "
                    "ini dapat dinyatakan dengan tepat pada naskah."
                ),
                kategori=RANCANGAN,
                kunci=["hubungan", "bukan sebab"],
            )
        ]

    rancangan = penelitian.rancangan
    tanya = [
        Pertanyaan(
            pertanyaan=(
                f"Penelitian Anda berancangan {rancangan.nama.lower()}. "
                + (
                    "Mengapa Anda berani menyimpulkan sebab-akibat?"
                    if penelitian.boleh_sebab
                    else "Mengapa hasil ini tidak boleh disebut sebab-akibat?"
                )
            ),
            jawaban=penelitian.alasan_sebab,
            kategori=RANCANGAN,
            kunci=(
                ["acak", "perlakuan"]
                if penelitian.boleh_sebab
                else ["hubungan", "arah", "perancu"]
            ),
        )
    ]

    if not penelitian.boleh_generalisasi:
        tanya.append(
            Pertanyaan(
                pertanyaan=(
                    f"Sampel Anda diambil secara {penelitian.cara_sampling.nama.lower()}. "
                    "Kepada siapa kesimpulan ini berlaku?"
                ),
                jawaban=penelitian.alasan_generalisasi,
                kategori=RANCANGAN,
                kunci=["responden yang diteliti", "bukan seluruh populasi"],
            )
        )
    return tanya


def _tanya_sampel(laporan, penelitian, jejak, audit) -> list[Pertanyaan]:
    n = int(getattr(laporan, "n_baris", 0) or 0)
    if n <= 0:
        return []

    jawaban = (
        f"Analisis dijalankan pada {_num(n)} pengamatan. Kecukupan sampel tidak "
        "ditentukan angka mutlak, melainkan oleh daya uji: besar pengaruh yang "
        "hendak dideteksi, taraf signifikansi, dan banyaknya prediktor. Hitungan "
        "daya uji tersedia pada halaman Ruang Proyek."
    )
    if penelitian is not None and penelitian.target_sampel:
        selisih = n - int(penelitian.target_sampel)
        jawaban += (
            f" Target sampel yang Anda tetapkan {_num(penelitian.target_sampel)}, "
            + (
                f"dan terkumpul {_num(selisih)} lebih banyak."
                if selisih >= 0
                else f"sedangkan yang terkumpul kurang {_num(abs(selisih))}. "
                "Kekurangan ini wajib disebutkan pada keterbatasan."
            )
        )
    return [
        Pertanyaan(
            pertanyaan=f"Mengapa {_num(n)} responden Anda anggap cukup?",
            jawaban=jawaban,
            kategori=RANCANGAN,
            bobot=MUNGKIN,
            kunci=["daya uji", "besar pengaruh"],
        )
    ]


# Metode yang menjadi jawaban pertanyaan penelitian, bukan pemeriksaan pendahuluan.
# Penguji menanyakan mengapa Anda memakai regresi, bukan mengapa Anda menghitung
# jarak Mahalanobis - yang terakhir adalah langkah pemeriksaan, bukan temuan.
METODE_UTAMA = (
    "regresi",
    "logistik",
    "manova",
    "ancova",
    "diskriminan",
    "klaster",
    "faktor",
    "komponen utama",
    "kanonik",
    "sem",
    "jalur",
    "moderasi",
    "kruskal",
    "mann-whitney",
    "anova",
)


def _utama(temuan) -> bool:
    gabungan = f"{temuan.metode} {temuan.judul}".lower()
    return any(nama in gabungan for nama in METODE_UTAMA)


def _tanya_metode(laporan, penelitian, jejak, audit) -> list[Pertanyaan]:
    tanya = []
    semua = list(getattr(laporan, "temuan", []))
    # Metode utama didahulukan; pemeriksaan pendahuluan hanya mengisi bila kosong.
    berurut = [t for t in semua if _utama(t)] or semua
    for temuan in berurut[:3]:
        tanya.append(
            Pertanyaan(
                pertanyaan=f"Mengapa Anda memakai {temuan.metode} untuk menjawab '{temuan.judul}'?",
                jawaban=(
                    f"{temuan.metode} dipilih karena sesuai dengan skala variabel dan "
                    "tujuan analisisnya. Halaman Pemandu Uji mencatat alasan pemilihan "
                    "beserta metode alternatif yang tidak dipilih dan sebabnya — "
                    "bagian itulah yang perlu Anda sampaikan, bukan sekadar nama "
                    "metodenya."
                ),
                kategori=METODE,
                bobot=MUNGKIN,
                kunci=["skala", "alternatif"],
            )
        )
    return tanya[:2]


def _tanya_asumsi(laporan, penelitian, jejak, audit) -> list[Pertanyaan]:
    lampu = [l for l in getattr(laporan, "lampu", []) if l.status in {"kritis", "perhatian"}]
    tanya = []
    for butir in lampu[:2]:
        tanya.append(
            Pertanyaan(
                pertanyaan=(
                    f"Pemeriksaan '{butir.label}' menunjukkan {butir.nilai}. "
                    "Bagaimana Anda mempertanggungjawabkannya?"
                ),
                jawaban=(
                    f"{butir.catatan} Pelanggaran asumsi tidak membatalkan hasil, "
                    "tetapi wajib disebutkan pada naskah beserta langkah yang diambil "
                    "— memakai uji yang lebih longgar asumsinya, memakai galat baku "
                    "robust, atau menyatakannya sebagai keterbatasan."
                ),
                kategori=ASUMSI,
                kunci=["disebutkan", "keterbatasan"],
            )
        )
    return tanya


def _tanya_hasil(laporan, penelitian, jejak, audit) -> list[Pertanyaan]:
    tanya = []
    pendorong = getattr(laporan, "pendorong", [])

    tabel = getattr(laporan, "tabel", {}) or {}
    for judul, isi, _ in tabel.values():
        if "ringkasan model" not in str(judul).lower() or isi is None or isi.empty:
            continue
        baris = isi.set_index(isi.columns[0])[isi.columns[1]]
        if "R²" in baris.index:
            nilai = str(baris["R²"])
            tanya.append(
                Pertanyaan(
                    pertanyaan=(
                        f"R² model Anda {nilai}. Apa artinya, dan apakah itu cukup?"
                    ),
                    jawaban=(
                        f"R² {nilai} berarti model menjelaskan sebagian ragam variabel "
                        "terikat, sisanya berasal dari hal yang tidak tercatat dalam "
                        "data ini. Tidak ada ambang mutlak: R² yang wajar pada penelitian "
                        "perilaku jauh lebih rendah daripada pada data teknis. Yang perlu "
                        "Anda bandingkan adalah R² penelitian sejenis di bidang yang sama, "
                        "bukan angka umum."
                    ),
                    kategori=HASIL,
                    kunci=["menjelaskan", "tidak ada ambang", "bidang"],
                )
            )
            break

    tidak_signifikan = [p for p in pendorong if not p.signifikan]
    if tidak_signifikan:
        nama = ", ".join(p.nama for p in tidak_signifikan[:3])
        tanya.append(
            Pertanyaan(
                pertanyaan=(
                    f"Variabel {nama} tidak signifikan. Apakah berarti tidak ada "
                    "hubungan sama sekali?"
                ),
                jawaban=(
                    "Tidak. Tidak signifikan berarti data ini belum cukup kuat "
                    "menunjukkan hubungan, bukan bahwa hubungannya terbukti tidak ada. "
                    "Sampel yang kecil, ragam yang besar, atau pengukuran yang kurang "
                    "tepat sama-sama dapat menghasilkan hasil tidak signifikan meski "
                    "hubungannya sebenarnya ada. Laporkan apa adanya beserta selang "
                    "kepercayaannya."
                ),
                kategori=HASIL,
                kunci=["belum cukup", "bukan berarti tidak ada"],
            )
        )
    return tanya


def _tanya_integritas(laporan, penelitian, jejak, audit) -> list[Pertanyaan]:
    tanya = []
    if jejak is not None and len(jejak.nilai_p()) >= 3:
        jumlah = len(jejak.nilai_p())
        berubah = jejak.berubah_setelah_koreksi()
        tanya.append(
            Pertanyaan(
                pertanyaan=(
                    f"Anda menjalankan {_num(jumlah)} uji. Bagaimana Anda mengendalikan "
                    "kesalahan tipe I?"
                ),
                jawaban=(
                    f"Menjalankan {_num(jumlah)} uji menaikkan peluang memperoleh hasil "
                    "signifikan secara kebetulan. Pada data ini, koreksi Holm mencabut "
                    f"signifikansi {_num(berubah)} temuan. Sebutkan berapa uji yang "
                    "dijalankan dan koreksi yang dipakai, atau nyatakan analisisnya "
                    "eksploratori."
                ),
                kategori=INTEGRITAS,
                kunci=["koreksi", "eksploratori"],
            )
        )

    pra = getattr(penelitian, "praregistrasi", None) if penelitian else None
    if jejak is not None and pra is not None and not pra.kosong():
        banding = jejak.banding_rencana(pra)
        if banding.di_luar_rencana:
            tanya.append(
                Pertanyaan(
                    pertanyaan=(
                        f"Uji {', '.join(banding.di_luar_rencana[:2])} tidak ada pada "
                        "rencana awal Anda. Mengapa dijalankan?"
                    ),
                    jawaban=(
                        "Menyimpang dari rencana adalah hal biasa dalam penelitian; "
                        "yang tidak biasa adalah menyimpang tanpa menyebutkannya. "
                        "Nyatakan uji mana yang direncanakan sejak awal dan mana yang "
                        "muncul kemudian, lalu laporkan yang belakangan sebagai "
                        "eksploratori."
                    ),
                    kategori=INTEGRITAS,
                    kunci=["eksploratori", "disebutkan"],
                )
            )
    return tanya


def _tanya_data(laporan, penelitian, jejak, audit) -> list[Pertanyaan]:
    if audit is None:
        return []
    tanya = []
    for temuan in (audit.kritis + audit.peringatan)[:2]:
        tanya.append(
            Pertanyaan(
                pertanyaan=(
                    f"Rapor data menemukan '{temuan.perkara}' pada {temuan.kolom}. "
                    "Apa yang Anda lakukan terhadapnya?"
                ),
                jawaban=(
                    f"{temuan.rincian} {temuan.dampak} Yang perlu Anda sampaikan adalah "
                    "tindakan yang diambil beserta alasannya — termasuk bila Anda "
                    f"memilih membiarkannya. {temuan.saran}"
                ),
                kategori=DATA,
                bobot=MUNGKIN,
                kunci=["tindakan", "alasan"],
            )
        )
    return tanya


def _num(nilai) -> str:
    """Angka bergaya Indonesia tanpa memuat modul format demi satu pemakaian."""
    return f"{int(nilai):,}".replace(",", ".")


# --------------------------------------------------------------------------- #
# Kesiapan
# --------------------------------------------------------------------------- #


def kesiapan(simulasi: Simulasi, jawaban: dict[int, str]) -> tuple[int, int, bool]:
    """Berapa pertanyaan wajib yang sudah dijawab memadai, dan apakah sudah siap.

    Lencana kesiapan hanya diberikan setelah pengguna menjawab sendiri. Aplikasi
    boleh menyusun kalimat; ia tidak boleh menggantikan pemahaman.
    """
    wajib = simulasi.wajib
    if not wajib:
        return 0, 0, False

    lulus = 0
    for nomor, butir in enumerate(simulasi.pertanyaan):
        if butir.bobot != WAJIB:
            continue
        cukup, _ = butir.nilai(jawaban.get(nomor, ""))
        lulus += int(cukup)
    return lulus, len(wajib), lulus == len(wajib)
