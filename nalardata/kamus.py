"""Kamus variabel: tempat aplikasi mengetahui **maksud** kolom, bukan sekadar bentuknya.

Sampai kini aplikasi menebak tipe kolom sesaat sebelum dipakai — ``numeric_columns``
memanggil dtype pandas, dan selesai. Penebakan itu cukup untuk menghitung, tetapi tidak
cukup untuk menasihati. Kolom berisi angka 1 sampai 5 bisa berarti skor Likert, jumlah
anak, atau kode wilayah; ketiganya menuntut uji yang berbeda, dan **tidak satu pun dapat
dibedakan dari dtype**.

Modul ini menyimpan yang tidak dapat dibaca dari data: skala pengukuran, peran variabel
dalam penelitian, definisi operasional, satuan, label nilai, dan kode nilai hilang.

Pembagian kerjanya tegas dan disengaja:

* **Aplikasi menebak dari bentuk data** — dan selalu menyertakan seberapa yakin ia serta
  alasannya.
* **Pengguna mengonfirmasi dari maksudnya** — karena hanya ia yang tahu apa yang diukur.

Tebakan yang belum dikonfirmasi tetap dipakai agar aplikasi bisa berjalan, tetapi
ditandai, dan halaman yang bergantung padanya wajib menampilkan tanda itu. Menebak
diam-diam lalu menyajikan hasilnya sebagai kepastian adalah cara paling halus untuk
menyesatkan pengguna yang belum bisa menilai sendiri.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Kosakata
# --------------------------------------------------------------------------- #

NOMINAL = "nominal"
ORDINAL = "ordinal"
INTERVAL = "interval"
RASIO = "rasio"
SKALA = (NOMINAL, ORDINAL, INTERVAL, RASIO)

LABEL_SKALA = {
    NOMINAL: "Nominal — kategori tanpa urutan",
    ORDINAL: "Ordinal — kategori berurutan, jarak antar tingkat tidak sama",
    INTERVAL: "Interval — angka berjarak sama, nol bukan berarti tidak ada",
    RASIO: "Rasio — angka berjarak sama, nol berarti tidak ada",
}

# Peran variabel dalam penelitian. "belum ditentukan" adalah keadaan awal yang jujur:
# aplikasi tidak berpura-pura tahu variabel mana yang menjadi outcome.
BELUM = "belum ditentukan"
PERAN = (
    BELUM,
    "outcome",
    "prediktor",
    "kelompok",
    "kovariat",
    "mediator",
    "moderator",
    "indikator",
    "id",
    "waktu",
    "tidak dipakai",
)

LABEL_PERAN = {
    BELUM: "Belum ditentukan",
    "outcome": "Variabel terikat (outcome)",
    "prediktor": "Variabel bebas (prediktor)",
    "kelompok": "Penanda kelompok",
    "kovariat": "Kovariat / variabel kontrol",
    "mediator": "Mediator",
    "moderator": "Moderator",
    "indikator": "Indikator konstruk laten",
    "id": "Penanda unit (ID)",
    "waktu": "Penanda waktu",
    "tidak dipakai": "Tidak dipakai dalam analisis",
}

PASTI = "pasti"
MUNGKIN = "mungkin"
TEBAKAN = "tebakan"
KEYAKINAN = (PASTI, MUNGKIN, TEBAKAN)

LABEL_KEYAKINAN = {
    PASTI: "Pasti",
    MUNGKIN: "Mungkin",
    TEBAKAN: "Tebakan — mohon diperiksa",
}

# Kata yang lazim menyusun skala ordinal berbentuk teks pada kuesioner Indonesia.
# Dipakai hanya untuk menaikkan dugaan menjadi ordinal, tidak pernah untuk memutuskan.
KATA_ORDINAL = {
    "sangat tidak setuju", "tidak setuju", "netral", "setuju", "sangat setuju",
    "sangat rendah", "rendah", "sedang", "tinggi", "sangat tinggi",
    "sangat buruk", "buruk", "cukup", "baik", "sangat baik",
    "tidak pernah", "jarang", "kadang-kadang", "sering", "selalu",
    "sangat tidak puas", "tidak puas", "puas", "sangat puas",
    "sd", "smp", "sma", "smk", "diploma", "sarjana", "magister", "doktor",
    "s1", "s2", "s3", "d3",
}

# Kode nilai hilang yang lazim dipakai pada SPSS dan kuesioner.
KODE_HILANG_LAZIM = (-999.0, -99.0, -9.0, -1.0, 9.0, 99.0, 999.0, 9999.0)

# Batas atas skala Likert yang masih masuk akal. Di atas ini, angka bulat berjarak
# rapat lebih mungkin merupakan cacahan daripada skor.
MAKS_TINGKAT_LIKERT = 10
MIN_TINGKAT_LIKERT = 3


# --------------------------------------------------------------------------- #
# Satu variabel
# --------------------------------------------------------------------------- #


@dataclass
class Variabel:
    """Keterangan satu kolom: apa isinya, bagaimana diukur, dan untuk apa dipakai."""

    nama: str
    nama_lengkap: str = ""
    definisi: str = ""
    satuan: str = ""
    skala: str = NOMINAL
    peran: str = BELUM
    label_nilai: dict = field(default_factory=dict)
    kode_hilang: list = field(default_factory=list)
    keyakinan: str = TEBAKAN
    alasan: str = ""
    dikonfirmasi: bool = False

    def __post_init__(self) -> None:
        if self.skala not in SKALA:
            raise ValueError(f"Skala '{self.skala}' tidak dikenal. Pilih dari {SKALA}.")
        if self.peran not in PERAN:
            raise ValueError(f"Peran '{self.peran}' tidak dikenal. Pilih dari {PERAN}.")
        if self.keyakinan not in KEYAKINAN:
            raise ValueError(f"Keyakinan '{self.keyakinan}' tidak dikenal.")

    @property
    def numerik(self) -> bool:
        """Boleh dihitung rata-ratanya."""
        return self.skala in {INTERVAL, RASIO}

    @property
    def kategorik(self) -> bool:
        return self.skala in {NOMINAL, ORDINAL}

    @property
    def judul(self) -> str:
        """Nama untuk ditampilkan; nama lengkap bila ada, nama kolom bila tidak."""
        return self.nama_lengkap.strip() or self.nama

    @property
    def perlu_diperiksa(self) -> bool:
        """Tebakan yang belum dikonfirmasi dan tidak meyakinkan."""
        return not self.dikonfirmasi and self.keyakinan != PASTI


# --------------------------------------------------------------------------- #
# Menebak skala dari data
# --------------------------------------------------------------------------- #


def _tanpa_hilang(s: pd.Series) -> pd.Series:
    return s.dropna()


def tebak_skala(s: pd.Series) -> tuple[str, str, str]:
    """Menebak skala satu kolom.

    Mengembalikan ``(skala, keyakinan, alasan)``. Alasannya ikut dikembalikan karena
    tebakan tanpa alasan tidak dapat dinilai pengguna — dan pengguna yang menilai.
    """
    isi = _tanpa_hilang(s)
    if isi.empty:
        return NOMINAL, TEBAKAN, "Kolom kosong sehingga tidak ada dasar untuk menebak."

    unik = isi.unique()
    n_unik = len(unik)

    if pd.api.types.is_bool_dtype(s):
        return NOMINAL, PASTI, "Isinya benar/salah, jadi dua kategori tanpa urutan."

    if pd.api.types.is_datetime64_any_dtype(s):
        return INTERVAL, PASTI, "Berisi tanggal atau waktu, yang berjarak sama tanpa nol mutlak."

    if not pd.api.types.is_numeric_dtype(s):
        teks = {str(v).strip().lower() for v in unik}
        if len(teks) >= MIN_TINGKAT_LIKERT and teks <= KATA_ORDINAL:
            return (
                ORDINAL,
                MUNGKIN,
                "Seluruh isinya kata bertingkat yang lazim dipakai pada skala ordinal.",
            )
        if n_unik == 2:
            return NOMINAL, PASTI, "Berisi teks dengan dua kategori."
        return NOMINAL, PASTI, f"Berisi teks dengan {n_unik} kategori tanpa urutan yang terbaca."

    nilai = pd.to_numeric(isi, errors="coerce").dropna()
    if nilai.empty:
        return NOMINAL, TEBAKAN, "Kolom bertipe angka namun tidak ada nilai yang terbaca."

    bulat = bool(np.all(np.equal(np.mod(nilai, 1), 0)))
    minimum, maksimum = float(nilai.min()), float(nilai.max())

    if n_unik == 1:
        return NOMINAL, TEBAKAN, "Seluruh baris bernilai sama, sehingga skalanya tidak terbaca."

    if n_unik == 2:
        return (
            NOMINAL,
            MUNGKIN,
            "Hanya dua nilai berbeda, jadi lebih mungkin penanda kategori daripada ukuran.",
        )

    if (
        bulat
        and MIN_TINGKAT_LIKERT <= n_unik <= MAKS_TINGKAT_LIKERT
        and minimum >= 0
        and maksimum <= MAKS_TINGKAT_LIKERT
    ):
        # Skala Likert hampir selalu mulai dari 1; cacahan lazim mulai dari 0.
        # Perbedaan satu angka ini memisahkan "sangat tidak setuju sampai sangat
        # setuju" dari "jumlah tanggungan", dan keduanya menuntut uji berbeda.
        if minimum >= 1:
            return (
                ORDINAL,
                MUNGKIN,
                f"Angka bulat {minimum:.0f}\u2013{maksimum:.0f} dengan {n_unik} tingkat "
                "dan tanpa nol, pola yang khas untuk skala Likert. Bila ini sebenarnya "
                "cacahan, ubah menjadi rasio.",
            )
        return (
            RASIO,
            MUNGKIN,
            f"Angka bulat mulai dari nol sampai {maksimum:.0f}, pola yang khas untuk "
            "cacahan. Bila ini sebenarnya skala bertingkat, ubah menjadi ordinal.",
        )

    if minimum < 0:
        return (
            INTERVAL,
            MUNGKIN,
            "Ada nilai negatif, sehingga nol bukan titik 'tidak ada'.",
        )

    return (
        RASIO,
        TEBAKAN,
        "Angka menyebar tanpa nilai negatif. Interval dan rasio tidak dapat "
        "dibedakan dari data — hanya Anda yang tahu apakah nol berarti 'tidak ada'.",
    )


def tebak_peran(nama: str, s: pd.Series, skala: str) -> str:
    """Menebak peran hanya untuk hal yang benar-benar terbaca: penanda unit dan waktu.

    Peran lain — outcome, prediktor, mediator — adalah keputusan penelitian, bukan
    sifat data. Menebaknya berarti menyarankan hipotesis, dan itu bukan wewenang
    aplikasi.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return "waktu"

    kata = nama.strip().lower().replace("-", "_").split("_")
    isi = _tanpa_hilang(s)
    unik_semua = len(isi) > 0 and isi.is_unique
    if unik_semua and ({"id", "no", "nomor", "kode", "nik", "urut"} & set(kata)):
        return "id"

    # Nama saja tidak cukup: "lama_usaha_tahun" memuat kata "tahun" tetapi berisi
    # lama waktu, bukan penanda periode. Penanda waktu harus disebut di depan
    # **dan** nilainya harus tampak seperti periode.
    if kata and kata[0] in {"tahun", "bulan", "periode", "waktu", "tanggal", "kuartal"}:
        if _tampak_periode(s):
            return "waktu"

    return BELUM


def _tampak_periode(s: pd.Series) -> bool:
    """Nilainya berupa tahun kalender atau nomor periode yang berulang."""
    if not pd.api.types.is_numeric_dtype(s):
        return True  # teks di bawah nama berawalan periode, misalnya "2021-Q1"
    nilai = pd.to_numeric(_tanpa_hilang(s), errors="coerce").dropna()
    if nilai.empty or not bool(np.all(np.equal(np.mod(nilai, 1), 0))):
        return False
    tahun = bool(nilai.min() >= 1900 and nilai.max() <= 2100)
    periode_pendek = bool(nilai.min() >= 1 and nilai.max() <= 12 and nilai.nunique() <= 12)
    return tahun or periode_pendek


def tebak_kode_hilang(s: pd.Series) -> list[float]:
    """Mencari kode nilai hilang yang lazim (99, 999, -1) sebagai **usulan**.

    Hanya diusulkan bila nilainya terpencil jauh dari sisa data. Angka 99 pada kolom
    usia adalah kode; angka 99 pada kolom skor ujian mungkin nilai sungguhan. Aplikasi
    tidak dapat membedakannya, jadi ia bertanya alih-alih memutuskan.
    """
    if not pd.api.types.is_numeric_dtype(s):
        return []
    nilai = pd.to_numeric(_tanpa_hilang(s), errors="coerce").dropna()
    if len(nilai) < 5:
        return []

    usulan: list[float] = []
    for kode in KODE_HILANG_LAZIM:
        if kode not in set(nilai.unique()):
            continue
        sisa = nilai[nilai != kode]
        if sisa.empty:
            continue
        rentang = float(sisa.max() - sisa.min())
        # Terpencil bila jaraknya dari data lain melebihi rentang data itu sendiri.
        jarak = kode - float(sisa.max()) if kode > sisa.max() else float(sisa.min()) - kode
        if jarak > max(rentang, 1.0):
            usulan.append(float(kode))
    return usulan


# --------------------------------------------------------------------------- #
# Kamus
# --------------------------------------------------------------------------- #


@dataclass
class Kamus:
    """Kumpulan keterangan variabel untuk satu himpunan data."""

    variabel: dict[str, Variabel] = field(default_factory=dict)

    # -- akses dasar -------------------------------------------------------- #

    def __contains__(self, nama: object) -> bool:
        return nama in self.variabel

    def __getitem__(self, nama: str) -> Variabel:
        try:
            return self.variabel[nama]
        except KeyError:
            raise KeyError(f"Kolom '{nama}' tidak ada dalam kamus.") from None

    def __iter__(self):
        return iter(self.variabel.values())

    def __len__(self) -> int:
        return len(self.variabel)

    @property
    def kolom(self) -> list[str]:
        return list(self.variabel)

    def judul(self, nama: str) -> str:
        """Nama tampilan sebuah kolom; kolom asing dikembalikan apa adanya."""
        butir = self.variabel.get(nama)
        return butir.judul if butir else nama

    # -- penyaringan -------------------------------------------------------- #

    def dengan_skala(self, *skala: str) -> list[str]:
        return [v.nama for v in self if v.skala in skala]

    def dengan_peran(self, *peran: str) -> list[str]:
        return [v.nama for v in self if v.peran in peran]

    def numerik(self) -> list[str]:
        """Kolom yang boleh dihitung rata-ratanya menurut kamus, bukan menurut dtype."""
        return [v.nama for v in self if v.numerik and v.peran != "id"]

    def kategorik(self) -> list[str]:
        return [v.nama for v in self if v.kategorik and v.peran != "id"]

    def perlu_diperiksa(self) -> list[str]:
        return [v.nama for v in self if v.perlu_diperiksa]

    # -- penyuntingan ------------------------------------------------------- #

    def tetapkan(self, nama: str, **ubah) -> Variabel:
        """Menyunting satu variabel. Setiap suntingan menandainya sudah dikonfirmasi."""
        lama = self[nama]
        tidak_dikenal = set(ubah) - {f for f in vars(lama)}
        if tidak_dikenal:
            raise ValueError(f"Ruas tidak dikenal: {sorted(tidak_dikenal)}")
        ubah.setdefault("dikonfirmasi", True)
        baru = replace(lama, **ubah)
        self.variabel[nama] = baru
        return baru

    def selaraskan(self, df: pd.DataFrame) -> "Kamus":
        """Menyesuaikan kamus dengan data yang berubah.

        Kolom baru ditebak; kolom yang hilang dibuang; keterangan kolom yang bertahan
        **dipertahankan** — pengguna tidak boleh kehilangan definisi operasional yang
        sudah ia tulis hanya karena satu kolom ditambahkan.
        """
        hasil: dict[str, Variabel] = {}
        for nama in df.columns:
            teks = str(nama)
            hasil[teks] = self.variabel.get(teks) or _tebak_variabel(teks, df[nama])
        return Kamus(hasil)

    # -- penerapan ke data -------------------------------------------------- #

    def terapkan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menerapkan kode nilai hilang dan label nilai pada salinan data.

        Ini satu-satunya tempat kamus mengubah data, dan hanya memakai kode yang sudah
        ditetapkan pengguna. Salinan dikembalikan agar data asli tetap utuh dan dapat
        dibandingkan.
        """
        hasil = df.copy()
        for butir in self:
            if butir.nama not in hasil.columns:
                continue
            kolom = hasil[butir.nama]
            if butir.kode_hilang:
                kolom = kolom.replace(list(butir.kode_hilang), np.nan)
            if butir.label_nilai:
                peta = {k: v for k, v in butir.label_nilai.items()}
                kolom = kolom.map(lambda x: peta.get(x, x))
            hasil[butir.nama] = kolom
        return hasil

    # -- tampilan ----------------------------------------------------------- #

    def ringkas(self) -> pd.DataFrame:
        """Tabel kamus untuk ditampilkan dan diperiksa."""
        baris = []
        for v in self:
            baris.append(
                {
                    "Kolom": v.nama,
                    "Nama lengkap": v.judul,
                    "Skala": v.skala,
                    "Peran": LABEL_PERAN[v.peran],
                    "Satuan": v.satuan,
                    "Status": "Dikonfirmasi" if v.dikonfirmasi else LABEL_KEYAKINAN[v.keyakinan],
                    "Dasar dugaan": v.alasan,
                }
            )
        return pd.DataFrame(baris)

    # -- penyimpanan -------------------------------------------------------- #

    def ke_dict(self) -> dict:
        return {
            nama: {
                "nama": v.nama,
                "nama_lengkap": v.nama_lengkap,
                "definisi": v.definisi,
                "satuan": v.satuan,
                "skala": v.skala,
                "peran": v.peran,
                "label_nilai": {str(k): s for k, s in v.label_nilai.items()},
                "kode_hilang": list(v.kode_hilang),
                "keyakinan": v.keyakinan,
                "alasan": v.alasan,
                "dikonfirmasi": v.dikonfirmasi,
            }
            for nama, v in self.variabel.items()
        }

    @classmethod
    def dari_dict(cls, isi: dict | None) -> "Kamus":
        """Memulihkan kamus dari berkas proyek; isi cacat dilewati, bukan menggagalkan."""
        if not isinstance(isi, dict):
            return cls()
        hasil: dict[str, Variabel] = {}
        for nama, ruas in isi.items():
            if not isinstance(ruas, dict):
                continue
            try:
                hasil[str(nama)] = Variabel(
                    nama=str(ruas.get("nama", nama)),
                    nama_lengkap=str(ruas.get("nama_lengkap", "")),
                    definisi=str(ruas.get("definisi", "")),
                    satuan=str(ruas.get("satuan", "")),
                    skala=str(ruas.get("skala", NOMINAL)),
                    peran=str(ruas.get("peran", BELUM)),
                    label_nilai=dict(ruas.get("label_nilai") or {}),
                    kode_hilang=list(ruas.get("kode_hilang") or []),
                    keyakinan=str(ruas.get("keyakinan", TEBAKAN)),
                    alasan=str(ruas.get("alasan", "")),
                    dikonfirmasi=bool(ruas.get("dikonfirmasi", False)),
                )
            except ValueError:
                continue
        return cls(hasil)

    @classmethod
    def dari_data(cls, df: pd.DataFrame, label_spss: dict | None = None) -> "Kamus":
        """Menyusun kamus awal dengan menebak dari data.

        ``label_spss`` berisi label kolom dari berkas ``.sav`` bila ada — keterangan
        yang sudah ditulis peneliti sendiri, jadi lebih dipercaya daripada tebakan.
        """
        label_spss = label_spss or {}
        hasil: dict[str, Variabel] = {}
        for nama in df.columns:
            teks = str(nama)
            butir = _tebak_variabel(teks, df[nama])
            if label_spss.get(teks):
                butir = replace(butir, nama_lengkap=str(label_spss[teks]))
            hasil[teks] = butir
        return cls(hasil)


def _tebak_variabel(nama: str, s: pd.Series) -> Variabel:
    skala, keyakinan, alasan = tebak_skala(s)
    return Variabel(
        nama=nama,
        skala=skala,
        peran=tebak_peran(nama, s, skala),
        keyakinan=keyakinan,
        alasan=alasan,
        kode_hilang=[],
    )


def usulan_kode_hilang(df: pd.DataFrame, kamus: Kamus) -> dict[str, list[float]]:
    """Usulan kode nilai hilang per kolom, untuk ditawarkan bukan diterapkan."""
    hasil: dict[str, list[float]] = {}
    for butir in kamus:
        if butir.nama not in df.columns or butir.kode_hilang:
            continue
        usul = tebak_kode_hilang(df[butir.nama])
        if usul:
            hasil[butir.nama] = usul
    return hasil
