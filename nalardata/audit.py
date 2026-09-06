"""Audit kualitas data: satu keputusan, bukan tumpukan pemeriksaan terpisah.

Aplikasi ini sudah memeriksa banyak hal — nilai hilang, pencilan, normalitas — namun
hasilnya terserak di beberapa halaman dan pengguna harus menyimpulkan sendiri apakah
datanya layak dianalisis. Modul ini menjalankan seluruh pemeriksaan sekaligus lalu
menjawab satu pertanyaan: **apakah data ini siap dianalisis, dan bila belum, apa yang
harus diperbaiki lebih dulu.**

Yang penting: audit ini **tidak mengubah data sama sekali**. Ia hanya melaporkan.
Setiap perbaikan tetap keputusan pengguna, karena membuang baris atau mengisi nilai
hilang secara diam-diam dapat mengubah kesimpulan tanpa disadari.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Tingkat keparahan, dari yang menghentikan analisis sampai yang sekadar dicatat.
KRITIS = "kritis"
PERINGATAN = "peringatan"
CATATAN = "catatan"

URUTAN = {KRITIS: 0, PERINGATAN: 1, CATATAN: 2}
LABEL = {
    KRITIS: "Kritis",
    PERINGATAN: "Perlu dicermati",
    CATATAN: "Catatan",
}

# Ambang yang dipakai. Dikumpulkan di satu tempat agar dapat ditinjau dan dikutip.
AMBANG_HILANG_KRITIS = 0.20  # proporsi nilai hilang per kolom
AMBANG_HILANG_PERINGATAN = 0.05
AMBANG_PENCILAN = 0.05  # proporsi pencilan IQR per kolom
AMBANG_KEMENCENGAN = 2.0
AMBANG_KATEGORI_BANYAK = 30  # tingkat kategori pada kolom teks


@dataclass
class Penanganan:
    """Satu pilihan tindakan atas sebuah temuan, beserta akibatnya bila dipilih.

    Akibatnya disebutkan lebih dulu, bukan sesudah tindakan dijalankan. Pengguna
    yang baru melihat akibatnya setelah datanya berubah tidak sedang memutuskan;
    ia sedang menerima keputusan aplikasi.
    """

    kode: str
    label: str
    akibat: str


@dataclass
class Temuan:
    """Satu masalah yang ditemukan pada data, dalam enam bagian.

    Masalahnya (``perkara``), apa yang terdampak (``kolom`` dan ``baris``),
    akibatnya terhadap analisis (``dampak``), apa yang sebaiknya dilakukan
    (``saran``), dan pilihan tindakan yang tersedia (``penanganan``). Bagian
    keenam - membatalkan tindakan - dipegang antarmuka, karena hanya di sanalah
    riwayat data tersimpan.
    """

    tingkat: str
    perkara: str
    kolom: str
    rincian: str
    dampak: str
    saran: str
    baris: list = field(default_factory=list)
    penanganan: list = field(default_factory=list)

    @property
    def n_baris(self) -> int:
        return len(self.baris)

    def ringkas_baris(self) -> str:
        """Sebutan baris terdampak yang dapat dibaca, bukan daftar indeks panjang."""
        if not self.baris:
            return "seluruh kolom"
        if len(self.baris) <= 5:
            return "baris " + ", ".join(str(b + 1) for b in self.baris)
        awal = ", ".join(str(b + 1) for b in self.baris[:3])
        return f"{len(self.baris)} baris (antara lain {awal})"


@dataclass
class HasilAudit:
    """Seluruh temuan beserta kesimpulan keseluruhannya."""

    temuan: list[Temuan] = field(default_factory=list)
    n_baris: int = 0
    n_kolom: int = 0

    @property
    def kritis(self) -> list[Temuan]:
        return [t for t in self.temuan if t.tingkat == KRITIS]

    @property
    def peringatan(self) -> list[Temuan]:
        return [t for t in self.temuan if t.tingkat == PERINGATAN]

    def status(self) -> str:
        if self.kritis:
            return KRITIS
        if self.peringatan:
            return PERINGATAN
        return "baik"

    def kesimpulan(self) -> str:
        """Satu kalimat yang menjawab 'apakah data ini siap dianalisis?'."""
        if self.kritis:
            return (
                f"Data belum siap dianalisis: {len(self.kritis)} masalah kritis harus "
                "diperbaiki lebih dulu karena dapat membuat hasil analisis menyesatkan."
            )
        if self.peringatan:
            return (
                f"Data dapat dianalisis, namun {len(self.peringatan)} hal perlu "
                "dicermati dan sebaiknya disebutkan pada bagian keterbatasan."
            )
        return (
            "Tidak ditemukan masalah kualitas yang berarti. Data siap dianalisis."
        )

    def tabel(self) -> pd.DataFrame:
        """Seluruh temuan, diurutkan dari yang paling mendesak."""
        if not self.temuan:
            return pd.DataFrame(
                columns=["Tingkat", "Perkara", "Kolom", "Rincian", "Dampak", "Saran"]
            )
        baris = [
            {
                "Tingkat": LABEL[t.tingkat],
                "Perkara": t.perkara,
                "Kolom": t.kolom,
                "Rincian": t.rincian,
                "Dampak": t.dampak,
                "Saran": t.saran,
            }
            for t in sorted(self.temuan, key=lambda t: (URUTAN[t.tingkat], t.perkara))
        ]
        return pd.DataFrame(baris)

    def ringkas(self) -> dict[str, int]:
        return {
            "Baris": self.n_baris,
            "Kolom": self.n_kolom,
            "Kritis": len(self.kritis),
            "Perlu dicermati": len(self.peringatan),
        }


# --------------------------------------------------------------------------- #
# Pemeriksaan
# --------------------------------------------------------------------------- #


def _persen(bagian: float, total: float) -> str:
    return f"{bagian / total * 100:.1f}%".replace(".", ",") if total else "-"


def _periksa_hilang(df: pd.DataFrame, hasil: HasilAudit) -> None:
    n = len(df)
    for kolom in df.columns:
        hilang = int(df[kolom].isna().sum())
        if not hilang:
            continue
        rasio = hilang / n
        if rasio >= AMBANG_HILANG_KRITIS:
            tingkat, dampak = KRITIS, (
                "Sebagian besar responden tidak punya nilai pada kolom ini, sehingga "
                "analisis yang memakainya hanya mewakili sebagian kecil data."
            )
        elif rasio >= AMBANG_HILANG_PERINGATAN:
            tingkat, dampak = PERINGATAN, (
                "Penghapusan baris tak lengkap akan mengurangi ukuran sampel."
            )
        else:
            tingkat, dampak = CATATAN, "Pengaruhnya pada ukuran sampel kecil."
        hasil.temuan.append(
            Temuan(
                tingkat=tingkat,
                perkara="Nilai hilang",
                kolom=str(kolom),
                rincian=f"{hilang} dari {n} baris kosong ({_persen(hilang, n)})",
                dampak=dampak,
                saran=(
                    "Pertimbangkan membuang kolom ini atau mencari sumber datanya."
                    if tingkat == KRITIS
                    else "Periksa apakah kekosongannya acak atau berpola tertentu."
                ),
            )
        )


def _periksa_duplikasi(df: pd.DataFrame, hasil: HasilAudit) -> None:
    ganda = int(df.duplicated().sum())
    if not ganda:
        return
    hasil.temuan.append(
        Temuan(
            tingkat=PERINGATAN,
            perkara="Baris kembar",
            kolom="(seluruh kolom)",
            rincian=f"{ganda} baris identik dengan baris lain ({_persen(ganda, len(df))})",
            dampak=(
                "Baris yang terhitung dua kali membuat hubungan tampak lebih kuat "
                "daripada sebenarnya dan mengecilkan nilai p secara keliru."
            ),
            saran="Pastikan pengulangan ini memang nyata, bukan kesalahan penggabungan data.",
        )
    )


def _periksa_konstan(df: pd.DataFrame, hasil: HasilAudit) -> None:
    for kolom in df.columns:
        isi = df[kolom].dropna()
        if isi.empty:
            hasil.temuan.append(
                Temuan(
                    tingkat=KRITIS,
                    perkara="Kolom kosong",
                    kolom=str(kolom),
                    rincian="Seluruh nilainya hilang",
                    dampak="Kolom ini tidak dapat dipakai analisis apa pun.",
                    saran="Buang kolom ini dari data.",
                )
            )
        elif isi.nunique() == 1:
            hasil.temuan.append(
                Temuan(
                    tingkat=KRITIS,
                    perkara="Kolom tidak beragam",
                    kolom=str(kolom),
                    rincian=f"Seluruh baris bernilai sama: {isi.iloc[0]!r}",
                    dampak=(
                        "Variabel tanpa keragaman tidak dapat berkorelasi dengan apa pun; "
                        "memasukkannya ke model akan menggagalkan perhitungan."
                    ),
                    saran="Keluarkan kolom ini dari pilihan variabel.",
                )
            )


def _periksa_campuran(df: pd.DataFrame, hasil: HasilAudit) -> None:
    """Kolom teks yang sebenarnya berisi angka — penyebab umum analisis gagal."""
    for kolom in df.columns:
        if pd.api.types.is_numeric_dtype(df[kolom]):
            continue
        isi = df[kolom].dropna()
        if isi.empty:
            continue
        angka = pd.to_numeric(isi, errors="coerce")
        rasio = float(angka.notna().mean())
        if rasio >= 0.8 and rasio < 1.0:
            bukan_angka = isi[angka.isna()].unique()[:3]
            hasil.temuan.append(
                Temuan(
                    tingkat=PERINGATAN,
                    perkara="Angka bercampur teks",
                    kolom=str(kolom),
                    rincian=(
                        f"{_persen(float(angka.notna().sum()), len(isi))} isinya angka; "
                        f"sisanya teks seperti {', '.join(map(repr, bukan_angka))}"
                    ),
                    dampak=(
                        "Kolom ini dibaca sebagai teks, sehingga tidak muncul pada "
                        "pilihan variabel numerik."
                    ),
                    saran="Seragamkan isinya, misalnya ganti 'n/a' menjadi sel kosong.",
                )
            )
        elif rasio == 1.0:
            hasil.temuan.append(
                Temuan(
                    tingkat=PERINGATAN,
                    perkara="Angka tersimpan sebagai teks",
                    kolom=str(kolom),
                    rincian="Seluruh isinya angka, tetapi tipenya teks",
                    dampak="Kolom ini tidak muncul pada pilihan variabel numerik.",
                    saran="Periksa pemisah desimal; koma perlu diubah menjadi titik.",
                )
            )


def _periksa_takhingga(df: pd.DataFrame, hasil: HasilAudit) -> None:
    for kolom in df.select_dtypes(include="number").columns:
        jumlah = int(np.isinf(df[kolom].to_numpy(dtype="float64", na_value=np.nan)).sum())
        if jumlah:
            hasil.temuan.append(
                Temuan(
                    tingkat=KRITIS,
                    perkara="Nilai tak hingga",
                    kolom=str(kolom),
                    rincian=f"{jumlah} nilai tak hingga",
                    dampak=(
                        "Nilai tak hingga menjalar ke seluruh perhitungan dan membuat "
                        "rata-rata, korelasi, serta model menjadi tidak terdefinisi."
                    ),
                    saran="Telusuri asalnya — biasanya pembagian dengan nol.",
                )
            )


def _periksa_pencilan(df: pd.DataFrame, hasil: HasilAudit) -> None:
    for kolom in df.select_dtypes(include="number").columns:
        isi = df[kolom].dropna()
        if len(isi) < 8:
            continue
        q1, q3 = isi.quantile(0.25), isi.quantile(0.75)
        jarak = q3 - q1
        if jarak <= 0:
            continue
        luar = int(((isi < q1 - 1.5 * jarak) | (isi > q3 + 1.5 * jarak)).sum())
        if luar and luar / len(isi) >= AMBANG_PENCILAN:
            hasil.temuan.append(
                Temuan(
                    tingkat=PERINGATAN,
                    perkara="Pencilan",
                    kolom=str(kolom),
                    rincian=f"{luar} nilai di luar 1,5 × IQR ({_persen(luar, len(isi))})",
                    dampak=(
                        "Pencilan menarik rata-rata dan garis regresi ke arahnya, "
                        "sehingga koefisien dapat berubah banyak bila dibuang."
                    ),
                    saran=(
                        "Periksa apakah nilainya salah catat atau memang kasus nyata; "
                        "uji non-parametrik lebih tahan terhadap pencilan."
                    ),
                )
            )


def _periksa_kemencengan(df: pd.DataFrame, hasil: HasilAudit) -> None:
    for kolom in df.select_dtypes(include="number").columns:
        isi = df[kolom].dropna()
        if len(isi) < 20 or isi.nunique() < 3:
            continue
        menceng = float(isi.skew())
        if abs(menceng) > AMBANG_KEMENCENGAN:
            arah = "ke kanan" if menceng > 0 else "ke kiri"
            hasil.temuan.append(
                Temuan(
                    tingkat=CATATAN,
                    perkara="Sebaran menceng",
                    kolom=str(kolom),
                    rincian=f"Kemencengan {menceng:.2f} ({arah})".replace(".", ","),
                    dampak="Uji yang menuntut normalitas menjadi kurang dapat diandalkan.",
                    saran=(
                        "Pertimbangkan transformasi logaritma, atau pakai uji "
                        "non-parametrik dan galat baku robust."
                    ),
                )
            )


def _periksa_kategori(df: pd.DataFrame, hasil: HasilAudit) -> None:
    for kolom in df.columns:
        if pd.api.types.is_numeric_dtype(df[kolom]):
            continue
        isi = df[kolom].dropna().astype(str)
        if isi.empty:
            continue

        # Kategori yang sama tetapi berbeda huruf besar-kecil atau berspasi lebih.
        rapi = isi.str.strip().str.lower()
        if rapi.nunique() < isi.nunique():
            contoh = (
                isi[rapi.duplicated(keep=False)].unique()[:4]
                if isi.nunique() > rapi.nunique()
                else []
            )
            hasil.temuan.append(
                Temuan(
                    tingkat=PERINGATAN,
                    perkara="Kategori tidak seragam",
                    kolom=str(kolom),
                    rincian=(
                        f"{isi.nunique()} kategori menyusut menjadi {rapi.nunique()} "
                        f"setelah spasi dan huruf besar-kecil disamakan"
                        + (f"; misalnya {', '.join(map(repr, contoh))}" if len(contoh) else "")
                    ),
                    dampak=(
                        "Satu kelompok terpecah menjadi beberapa, sehingga uji beda "
                        "membandingkan kelompok yang sebenarnya sama."
                    ),
                    saran="Seragamkan penulisannya sebelum dianalisis.",
                )
            )

        if isi.nunique() > AMBANG_KATEGORI_BANYAK:
            hasil.temuan.append(
                Temuan(
                    tingkat=CATATAN,
                    perkara="Kategori sangat banyak",
                    kolom=str(kolom),
                    rincian=f"{isi.nunique()} kategori berbeda",
                    dampak=(
                        "Kolom seperti ini biasanya pengenal, bukan variabel kelompok, "
                        "dan tidak cocok untuk uji beda."
                    ),
                    saran="Pastikan ini memang variabel kelompok, bukan nomor identitas.",
                )
            )


def _periksa_ukuran(df: pd.DataFrame, hasil: HasilAudit) -> None:
    if len(df) < 30:
        hasil.temuan.append(
            Temuan(
                tingkat=PERINGATAN,
                perkara="Sampel kecil",
                kolom="(seluruh data)",
                rincian=f"Hanya {len(df)} baris",
                dampak=(
                    "Pada sampel sekecil ini, uji yang menuntut normalitas bertumpu "
                    "penuh pada asumsi yang sulit diperiksa."
                ),
                saran="Utamakan uji non-parametrik dan laporkan ukuran efek.",
            )
        )
    lengkap = int(df.dropna().shape[0])
    if lengkap and lengkap < len(df) * 0.5:
        hasil.temuan.append(
            Temuan(
                tingkat=KRITIS,
                perkara="Baris lengkap sedikit",
                kolom="(seluruh data)",
                rincian=f"Hanya {lengkap} dari {len(df)} baris terisi penuh",
                dampak=(
                    "Analisis yang membuang baris tak lengkap akan berjalan pada "
                    "kurang dari separuh data."
                ),
                saran="Kurangi variabel yang dipilih, atau lengkapi kolom yang paling kosong.",
            )
        )


def jalankan_audit(df: pd.DataFrame, kamus=None) -> HasilAudit:
    """Jalankan seluruh pemeriksaan kualitas tanpa mengubah data.

    ``kamus`` membuka pemeriksaan yang menuntut pengetahuan tentang maksud kolom:
    jawaban seragam dan bias metode tunggal perlu tahu butir mana yang ordinal,
    dan struktur panel perlu tahu kolom mana penanda unit dan waktu. Tanpa kamus,
    pemeriksaan itu dilewati diam-diam alih-alih menebak.
    """
    if df is None or df.empty:
        raise ValueError("Tidak ada data yang dapat diaudit.")

    hasil = HasilAudit(n_baris=len(df), n_kolom=df.shape[1])
    for periksa in (
        _periksa_ukuran,
        _periksa_hilang,
        _periksa_pola_hilang,
        _periksa_duplikasi,
        _periksa_konstan,
        _periksa_campuran,
        _periksa_takhingga,
        _periksa_pencilan,
        _periksa_kemencengan,
        _periksa_nol_berlebih,
        _periksa_kategori,
        _periksa_data_pribadi,
    ):
        periksa(df, hasil)

    for periksa_kamus in (
        _periksa_jawaban_seragam,
        _periksa_metode_tunggal,
        _periksa_struktur_panel,
    ):
        periksa_kamus(df, hasil, kamus)
    return hasil

# --------------------------------------------------------------------------- #
# Pemeriksaan lanjutan: pola, struktur, dan mutu jawaban kuesioner
# --------------------------------------------------------------------------- #

AMBANG_NOL = 0.50  # proporsi nilai nol pada kolom cacahan
AMBANG_LURUS = 0.10  # proporsi responden yang menjawab seragam
MIN_BUTIR_LURUS = 5  # butir minimum agar pola jawaban seragam bermakna
AMBANG_HARMAN = 0.50  # ragam yang terserap satu faktor tunggal

# Kata pada nama kolom yang lazim menandai data pribadi. Dipakai untuk
# mengingatkan, tidak pernah untuk menghapus sendiri.
KATA_PRIBADI = {
    "nama", "name", "nik", "ktp", "npwp", "alamat", "address", "telepon",
    "telp", "hp", "handphone", "wa", "whatsapp", "email", "surel", "rekening",
    "norek", "kartu", "paspor", "passport", "lahir",
}


def _periksa_pola_hilang(df: pd.DataFrame, hasil: HasilAudit) -> None:
    """Apakah nilai hilang menumpuk pada kelompok responden tertentu.

    Nilai hilang yang tersebar acak dapat diabaikan atau diisi tanpa banyak
    akibat. Nilai hilang yang berkaitan dengan jawaban lain — misalnya responden
    berpendapatan tinggi yang enggan mengisi kolom pendapatan — akan membuat
    kesimpulan bias ke arah yang tidak terlihat pada data yang tersisa.
    """
    from scipy import stats as _stats

    angka = df.select_dtypes(include="number")
    if angka.shape[1] < 2:
        return

    for kolom in df.columns:
        hilang = df[kolom].isna()
        n_hilang = int(hilang.sum())
        if n_hilang < 10 or n_hilang == len(df):
            continue

        berkaitan = []
        for lain in angka.columns:
            if lain == kolom:
                continue
            ada = pd.to_numeric(angka.loc[~hilang, lain], errors="coerce").dropna()
            tiada = pd.to_numeric(angka.loc[hilang, lain], errors="coerce").dropna()
            if len(ada) < 5 or len(tiada) < 5:
                continue
            if float(_stats.mannwhitneyu(ada, tiada).pvalue) < 0.01:
                berkaitan.append(lain)

        if berkaitan:
            hasil.temuan.append(
                Temuan(
                    tingkat=PERINGATAN,
                    perkara="Nilai hilang tidak acak",
                    kolom=kolom,
                    rincian=(
                        f"{n_hilang} nilai hilang pada '{kolom}' berkaitan dengan "
                        f"{_sebut(berkaitan)}: responden yang tidak mengisi berbeda "
                        "nyata dari yang mengisi."
                    ),
                    dampak=(
                        "Menghapus atau mengisi nilai hilang di sini akan menggeser "
                        "kesimpulan ke arah kelompok yang mengisi, dan pergeseran itu "
                        "tidak terlihat pada data yang tersisa."
                    ),
                    saran=(
                        "Jangan hapus barisnya begitu saja. Sebutkan pola ini pada "
                        "bagian keterbatasan, dan pertimbangkan imputasi berganda."
                    ),
                    baris=list(df.index[hilang][:200]),
                    penanganan=[
                        Penanganan(
                            "tandai",
                            "Cukup catat pada keterbatasan",
                            "Data tidak diubah; catatan ini ikut ke laporan.",
                        ),
                    ],
                )
            )


def _periksa_nol_berlebih(df: pd.DataFrame, hasil: HasilAudit) -> None:
    """Kolom cacahan yang didominasi nol menuntut model tersendiri."""
    for kolom in df.select_dtypes(include="number").columns:
        nilai = pd.to_numeric(df[kolom], errors="coerce").dropna()
        if len(nilai) < 30 or nilai.min() < 0:
            continue
        if not bool(np.all(np.equal(np.mod(nilai, 1), 0))):
            continue
        bagian = float((nilai == 0).mean())
        if bagian < AMBANG_NOL or nilai.nunique() < 3:
            continue

        hasil.temuan.append(
            Temuan(
                tingkat=CATATAN,
                perkara="Nol berlebih",
                kolom=kolom,
                rincian=f"{_persen(float((nilai == 0).sum()), len(nilai))} nilainya nol.",
                dampak=(
                    "Rata-rata dan regresi linear biasa akan menyesatkan pada kolom "
                    "seperti ini, karena sebarannya menumpuk di satu titik."
                ),
                saran=(
                    "Pertimbangkan memisahkan pertanyaan 'apakah terjadi' dari "
                    "'seberapa sering', atau memakai model cacahan berinflasi nol."
                ),
            )
        )


def _periksa_jawaban_seragam(df: pd.DataFrame, hasil: HasilAudit, kamus=None) -> None:
    """Responden yang menjawab sama persis pada seluruh butir (straight-lining).

    Pada kuesioner panjang, sebagian responden mengisi satu kolom jawaban dari
    atas ke bawah tanpa membaca. Jawaban itu tetap terhitung dan tetap menaikkan
    alpha Cronbach, sehingga instrumen tampak makin reliabel justru karena
    jawaban yang tidak dipikirkan.
    """
    if kamus is None:
        return
    butir = [k for k in kamus.dengan_skala("ordinal") if k in df.columns]
    if len(butir) < MIN_BUTIR_LURUS:
        return

    bagian = df[butir]
    seragam = bagian.nunique(axis=1) == 1
    seragam = seragam & bagian.notna().all(axis=1)
    n = int(seragam.sum())
    if n == 0:
        return

    proporsi = n / len(df)
    hasil.temuan.append(
        Temuan(
            tingkat=PERINGATAN if proporsi >= AMBANG_LURUS else CATATAN,
            perkara="Jawaban seragam",
            kolom=f"{len(butir)} butir ordinal",
            rincian=(
                f"{n} responden ({_persen(float(n), len(df))}) memberi jawaban yang "
                f"sama persis pada seluruh {len(butir)} butir."
            ),
            dampak=(
                "Jawaban yang tidak dipikirkan tetap menaikkan alpha Cronbach, "
                "sehingga instrumen tampak makin reliabel justru karena jawaban itu."
            ),
            saran=(
                "Periksa lama pengisian bila tercatat. Bila jawaban ini dikeluarkan, "
                "sebutkan jumlah dan alasannya pada bagian metode."
            ),
            baris=list(df.index[seragam][:200]),
            penanganan=[
                Penanganan(
                    "hapus_baris",
                    f"Keluarkan {n} responden ini",
                    f"Data menjadi {len(df) - n} baris. Wajib disebutkan pada Bab III.",
                ),
                Penanganan(
                    "tandai",
                    "Biarkan, cukup dicatat",
                    "Data tidak diubah; catatan ini ikut ke laporan.",
                ),
            ],
        )
    )


def _periksa_metode_tunggal(df: pd.DataFrame, hasil: HasilAudit, kamus=None) -> None:
    """Uji faktor tunggal Harman: indikasi common-method bias pada kuesioner.

    Bila seluruh butir diukur dengan satu cara pada satu waktu oleh satu orang,
    sebagian korelasi antar konstruk dapat berasal dari cara pengukurannya, bukan
    dari hubungan yang diteliti.
    """
    if kamus is None:
        return
    butir = [k for k in kamus.dengan_skala("ordinal") if k in df.columns]
    if len(butir) < 6:
        return

    bagian = df[butir].apply(pd.to_numeric, errors="coerce").dropna()
    if len(bagian) < 30 or bagian.shape[1] < 6:
        return
    ragam = bagian.var(numeric_only=True)
    if float(ragam.min()) <= 0:
        return

    try:
        terpusat = bagian - bagian.mean()
        nilai_singular = np.linalg.svd(terpusat.to_numpy(), compute_uv=False)
        ragam_komponen = nilai_singular**2
        bagian_pertama = float(ragam_komponen[0] / ragam_komponen.sum())
    except np.linalg.LinAlgError:
        return

    if bagian_pertama < AMBANG_HARMAN:
        return

    hasil.temuan.append(
        Temuan(
            tingkat=PERINGATAN,
            perkara="Indikasi bias metode tunggal",
            kolom=f"{len(butir)} butir ordinal",
            rincian=(
                f"Satu faktor tunggal menyerap {bagian_pertama:.0%} ragam seluruh "
                f"butir (uji faktor tunggal Harman, ambang {AMBANG_HARMAN:.0%})."
            ),
            dampak=(
                "Sebagian korelasi antar konstruk dapat berasal dari cara "
                "pengukurannya — satu kuesioner, satu waktu, satu responden — "
                "bukan dari hubungan yang Anda teliti."
            ),
            saran=(
                "Uji Harman lemah dan hanya memberi isyarat. Sebutkan pada "
                "keterbatasan, dan pada penelitian berikutnya pisahkan waktu "
                "pengukuran variabel bebas dan terikat."
            ),
        )
    )


def _periksa_struktur_panel(df: pd.DataFrame, hasil: HasilAudit, kamus=None) -> None:
    """Pasangan unit-waktu yang berulang atau panel yang tidak seimbang."""
    if kamus is None:
        return
    unit = kamus.dengan_peran("id")
    waktu = kamus.dengan_peran("waktu")
    if not unit or not waktu:
        return

    kolom_unit, kolom_waktu = unit[0], waktu[0]
    if kolom_unit not in df.columns or kolom_waktu not in df.columns:
        return

    pasangan = df[[kolom_unit, kolom_waktu]].dropna()
    kembar = pasangan.duplicated(keep=False)
    if kembar.any():
        hasil.temuan.append(
            Temuan(
                tingkat=KRITIS,
                perkara="Unit dan waktu berulang",
                kolom=f"{kolom_unit} × {kolom_waktu}",
                rincian=(
                    f"{int(kembar.sum())} baris memuat pasangan unit-waktu yang sama, "
                    "sehingga satu unit tercatat lebih dari sekali pada periode yang sama."
                ),
                dampak=(
                    "Model panel menganggap tiap baris satu pengamatan. Pengulangan "
                    "membuat galat baku terlalu kecil dan hasil tampak lebih "
                    "signifikan daripada seharusnya."
                ),
                saran="Periksa apakah ini penggandaan saat penggabungan data.",
                baris=list(pasangan.index[kembar][:200]),
                penanganan=[
                    Penanganan(
                        "hapus_baris",
                        f"Hapus {int(kembar.sum()) - pasangan.duplicated().sum()} baris kembar",
                        "Menyisakan satu baris untuk tiap pasangan unit-waktu.",
                    )
                ],
            )
        )
        return

    per_unit = pasangan.groupby(kolom_unit)[kolom_waktu].nunique()
    if len(per_unit) > 1 and per_unit.nunique() > 1:
        hasil.temuan.append(
            Temuan(
                tingkat=CATATAN,
                perkara="Panel tidak seimbang",
                kolom=f"{kolom_unit} × {kolom_waktu}",
                rincian=(
                    f"Banyaknya periode per unit berkisar {int(per_unit.min())} sampai "
                    f"{int(per_unit.max())}."
                ),
                dampak=(
                    "Panel tidak seimbang masih dapat dianalisis, namun unit yang "
                    "keluar lebih awal dapat berbeda sistematis dari yang bertahan."
                ),
                saran="Sebutkan ketidakseimbangan ini beserta sebabnya pada bagian metode.",
            )
        )


def _periksa_data_pribadi(df: pd.DataFrame, hasil: HasilAudit) -> None:
    """Kolom yang tampaknya memuat identitas orang.

    Diperiksa karena data responden lazimnya diunggah ke aplikasi ini apa adanya
    dari lembar Excel, lengkap dengan nama dan nomor telepon yang tidak
    diperlukan analisis mana pun.
    """
    tertuduh = []
    for kolom in df.columns:
        kata = set(str(kolom).lower().replace("-", "_").split("_"))
        if kata & KATA_PRIBADI:
            tertuduh.append(str(kolom))

    if not tertuduh:
        return

    hasil.temuan.append(
        Temuan(
            tingkat=PERINGATAN,
            perkara="Kemungkinan data pribadi",
            kolom=_sebut(tertuduh),
            rincian=(
                f"{len(tertuduh)} kolom bernama seperti data pribadi: {_sebut(tertuduh)}."
            ),
            dampak=(
                "Identitas responden tidak diperlukan analisis mana pun, tetapi ikut "
                "tersimpan pada berkas proyek dan dapat ikut terekspor ke laporan."
            ),
            saran=(
                "Hapus kolomnya sebelum menganalisis, atau ganti dengan nomor urut. "
                "Kerahasiaan responden lazimnya sudah Anda janjikan pada lembar "
                "persetujuan."
            ),
            penanganan=[
                Penanganan(
                    "hapus_kolom",
                    f"Hapus {len(tertuduh)} kolom identitas",
                    "Kolom dikeluarkan dari data aktif; berkas asli Anda tidak tersentuh.",
                ),
                Penanganan(
                    "tandai",
                    "Biarkan, saya memerlukannya",
                    "Data tidak diubah.",
                ),
            ],
        )
    )


def _sebut(nama: list[str], maksimal: int = 3) -> str:
    """Sebutan daftar kolom yang ringkas, tidak menumpahkan seluruh isinya."""
    if len(nama) <= maksimal:
        return ", ".join(f"'{n}'" for n in nama)
    awal = ", ".join(f"'{n}'" for n in nama[:maksimal])
    return f"{awal}, dan {len(nama) - maksimal} lainnya"


# --------------------------------------------------------------------------- #
# Penerapan penanganan
# --------------------------------------------------------------------------- #


def terapkan(df: pd.DataFrame, temuan: Temuan, kode: str) -> tuple[pd.DataFrame, str]:
    """Terapkan satu penanganan dan kembalikan data baru beserta catatannya.

    Data lama tidak disentuh: yang dikembalikan salinan. Antarmuka menyimpan
    versi sebelumnya agar tindakan ini dapat dibatalkan, dan catatannya masuk ke
    jejak keputusan supaya laporan dapat menyebutkan apa yang diubah dan mengapa.
    """
    tersedia = {p.kode for p in temuan.penanganan}
    if kode not in tersedia:
        raise ValueError(
            f"Penanganan '{kode}' tidak tersedia untuk temuan '{temuan.perkara}'. "
            f"Pilih dari {sorted(tersedia)}."
        )

    if kode == "tandai":
        return df.copy(), f"{temuan.perkara} dicatat tanpa mengubah data."

    if kode == "hapus_baris":
        buang = [b for b in temuan.baris if b in df.index]
        hasil = df.drop(index=buang)
        return hasil, (
            f"{len(buang)} baris dihapus karena {temuan.perkara.lower()}. "
            f"Data menjadi {len(hasil)} baris."
        )

    if kode == "hapus_kolom":
        nama = [k.strip().strip("'") for k in temuan.kolom.split(",")]
        buang = [k for k in nama if k in df.columns]
        hasil = df.drop(columns=buang)
        return hasil, (
            f"{len(buang)} kolom dihapus karena {temuan.perkara.lower()}: "
            + ", ".join(buang)
            + "."
        )

    raise ValueError(f"Penanganan '{kode}' belum diterapkan di modul ini.")

