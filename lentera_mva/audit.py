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
class Temuan:
    """Satu masalah yang ditemukan pada data."""

    tingkat: str
    perkara: str
    kolom: str
    rincian: str
    dampak: str
    saran: str


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


def jalankan_audit(df: pd.DataFrame) -> HasilAudit:
    """Jalankan seluruh pemeriksaan kualitas tanpa mengubah data."""
    if df is None or df.empty:
        raise ValueError("Tidak ada data yang dapat diaudit.")

    hasil = HasilAudit(n_baris=len(df), n_kolom=df.shape[1])
    for periksa in (
        _periksa_ukuran,
        _periksa_hilang,
        _periksa_duplikasi,
        _periksa_konstan,
        _periksa_campuran,
        _periksa_takhingga,
        _periksa_pencilan,
        _periksa_kemencengan,
        _periksa_kategori,
    ):
        periksa(df, hasil)
    return hasil
