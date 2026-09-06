"""Naskah skripsi: Bab III, IV, dan V yang disusun dari analisis yang dijalankan.

Laporan analisis dan naskah skripsi bukan hal yang sama. Laporan menjawab "apa yang
ditemukan"; naskah harus menjawab "bagaimana Anda memperolehnya, apa hasilnya, dan
apa artinya" dalam urutan yang sudah ditetapkan pedoman penulisan. Menyodorkan
laporan analisis lalu meminta pengguna menyusunnya ulang menjadi bab adalah persis
pekerjaan yang membuatnya kewalahan.

Modul ini menyusun kerangka bab dari bahan yang sudah ada: rancangan penelitian dari
Ruang Proyek, definisi operasional dari Kamus Variabel, hasil dan tabel dari laporan,
serta batas kesimpulan dari kunci kausalitas dan generalisasi.

Yang dihasilkan **kerangka berisi**, bukan naskah jadi. Setiap bab dibuka dengan
peringatan bahwa pembahasan teoretis, kaitan dengan penelitian terdahulu, dan
argumentasi tetap harus ditulis sendiri — bagian itulah yang dinilai penguji, dan
bagian itu pula yang tidak dapat diturunkan dari data mana pun.
"""

from __future__ import annotations

import pandas as pd

from nalardata.ekspor import Blok, Dokumen

BAB3 = "bab3"
BAB4 = "bab4"
BAB5 = "bab5"
IMRAD = "imrad"

GAYA = {
    BAB3: "Bab III — Metode Penelitian",
    BAB4: "Bab IV — Hasil dan Pembahasan",
    BAB5: "Bab V — Kesimpulan dan Saran",
    IMRAD: "Artikel jurnal (IMRAD)",
}

# Ditempatkan di awal tiap bab, bukan di akhir. Pembaca yang menemukannya setelah
# menyalin seluruh isi sudah terlanjur menganggapnya naskah jadi.
PERINGATAN = (
    "Kerangka ini disusun otomatis dari analisis Anda. Pembahasan teoretis, kaitan "
    "dengan penelitian terdahulu, dan argumentasi harus Anda tulis sendiri — bagian "
    "itulah yang dinilai penguji, dan bagian itu tidak dapat diturunkan dari data. "
    "Periksa pula kesesuaiannya dengan pedoman penulisan kampus Anda."
)


def susun(laporan, gaya: str, penelitian=None, kamus=None) -> Dokumen:
    """Susun satu bab naskah sebagai dokumen yang siap diekspor ke format apa pun."""
    if gaya not in GAYA:
        raise ValueError(f"Gaya naskah '{gaya}' tidak dikenal. Pilih dari {list(GAYA)}.")

    penyusun = {BAB3: _bab3, BAB4: _bab4, BAB5: _bab5, IMRAD: _imrad}
    blok = penyusun[gaya](laporan, penelitian, kamus)

    judul = GAYA[gaya]
    if penelitian is not None and penelitian.judul.strip():
        judul = f"{GAYA[gaya]} — {penelitian.judul.strip()}"

    meta = (
        f"Sumber data: {laporan.dataset} · {laporan.n_baris} baris × "
        f"{laporan.n_kolom} kolom · {laporan.tanggal}"
    )
    return Dokumen(
        judul=judul,
        meta=meta,
        blok=[Blok("judul", judul), Blok("meta", meta), Blok("catatan", PERINGATAN)] + blok,
        nama_dasar=f"naskah_{gaya}",
    )


# --------------------------------------------------------------------------- #
# Bab III — Metode Penelitian
# --------------------------------------------------------------------------- #


def _bab3(laporan, penelitian, kamus) -> list[Blok]:
    blok = [Blok("subjudul", "3.1 Jenis dan Pendekatan Penelitian")]

    if penelitian is None or penelitian.kosong():
        blok.append(
            Blok(
                "paragraf",
                "Rancangan penelitian belum diisi pada halaman Ruang Proyek, sehingga "
                "bagian ini belum dapat disusun. Isi rancangan, populasi, teknik "
                "sampling, dan unit analisis lebih dulu agar bab ini terisi dengan "
                "keterangan yang benar.",
            )
        )
        return blok

    rancangan = penelitian.rancangan
    blok.append(
        Blok(
            "paragraf",
            f"Penelitian ini menggunakan pendekatan kuantitatif dengan rancangan "
            f"{rancangan.nama.lower()}. {rancangan.keterangan} "
            f"{penelitian.alasan_sebab}",
        )
    )

    blok.append(Blok("subjudul", "3.2 Populasi dan Sampel"))
    populasi = penelitian.populasi.strip() or "[belum diisi]"
    kalimat = f"Populasi penelitian ini adalah {populasi}."
    if penelitian.ukuran_populasi:
        kalimat += f" Jumlah populasi tercatat {_num(penelitian.ukuran_populasi)}."
    kalimat += (
        f" Unit analisisnya adalah {penelitian.unit_analisis.strip() or '[belum diisi]'}. "
        f"Pengambilan sampel dilakukan secara {penelitian.cara_sampling.nama.lower()}. "
        f"{penelitian.alasan_generalisasi}"
    )
    if penelitian.target_sampel:
        kalimat += (
            f" Target sampel ditetapkan {_num(penelitian.target_sampel)} responden, "
            f"dan data yang berhasil dianalisis berjumlah {_num(laporan.n_baris)}."
        )
    else:
        kalimat += f" Data yang berhasil dianalisis berjumlah {_num(laporan.n_baris)}."
    blok.append(Blok("paragraf", kalimat))

    blok.append(Blok("subjudul", "3.3 Definisi Operasional Variabel"))
    tabel = _tabel_definisi(kamus)
    if tabel is None:
        blok.append(
            Blok(
                "paragraf",
                "Definisi operasional belum diisi pada Kamus Variabel. Bagian ini "
                "hampir selalu ditanyakan penguji, karena dari sanalah terbaca apa "
                "yang sebenarnya Anda ukur.",
            )
        )
    else:
        blok.append(
            Blok(
                "tabel",
                tabel=tabel,
                catatan="Diambil dari Kamus Variabel; lengkapi yang masih kosong.",
            )
        )

    blok.append(Blok("subjudul", "3.4 Teknik Analisis Data"))
    metode = laporan.metode_terpakai or []
    blok.append(
        Blok(
            "paragraf",
            "Analisis data dilakukan dengan bantuan aplikasi NalarData. Tahapan yang "
            "dijalankan meliputi "
            + (", ".join(m.lower() for m in metode) if metode else "[belum ada analisis]")
            + ". Seluruh pengujian memakai taraf signifikansi 5 persen, dan hasil "
            "pemeriksaan asumsi dilaporkan berdampingan dengan hasil pengujiannya.",
        )
    )
    if penelitian.praregistrasi is not None and penelitian.praregistrasi.uji_direncanakan:
        blok.append(
            Blok(
                "paragraf",
                "Rencana analisis dicatat sebelum data diperiksa, meliputi "
                + ", ".join(penelitian.praregistrasi.uji_direncanakan)
                + f" (dicatat {penelitian.praregistrasi.waktu}).",
            )
        )
    return blok


def _tabel_definisi(kamus) -> pd.DataFrame | None:
    """Tabel definisi operasional; None bila kamus belum diisi sama sekali."""
    if kamus is None or not len(kamus):
        return None
    dipakai = [v for v in kamus if v.peran not in {"id", "tidak dipakai"}]
    if not dipakai:
        return None
    return pd.DataFrame(
        {
            "Variabel": [v.judul for v in dipakai],
            "Definisi operasional": [v.definisi or "[belum diisi]" for v in dipakai],
            "Skala": [v.skala for v in dipakai],
            "Satuan": [v.satuan or "—" for v in dipakai],
        }
    )


# --------------------------------------------------------------------------- #
# Bab IV — Hasil
# --------------------------------------------------------------------------- #


def _bab4(laporan, penelitian, kamus) -> list[Blok]:
    blok = [
        Blok("subjudul", "4.1 Gambaran Umum Data"),
        Blok(
            "paragraf",
            f"Data yang dianalisis berjumlah {_num(laporan.n_baris)} pengamatan dengan "
            f"{_num(laporan.n_kolom)} variabel. {laporan.headline} {laporan.subheadline}",
        ),
    ]

    if laporan.lampu:
        blok.append(Blok("subjudul", "4.2 Hasil Pemeriksaan Asumsi"))
        blok.append(
            Blok(
                "tabel",
                tabel=pd.DataFrame(
                    {
                        "Pemeriksaan": [l.label for l in laporan.lampu],
                        "Nilai": [l.nilai for l in laporan.lampu],
                        "Status": [l.status_label for l in laporan.lampu],
                    }
                ),
                catatan=(
                    "Pelanggaran asumsi tetap dilaporkan apa adanya; langkah yang "
                    "diambil atasnya diuraikan pada pembahasan."
                ),
            )
        )

    blok.append(Blok("subjudul", "4.3 Hasil Pengujian"))
    for temuan in laporan.temuan:
        blok.append(Blok("paragraf", f"{temuan.judul} ({temuan.metode})"))
        blok.append(Blok("paragraf", temuan.akademik))

    if laporan.tabel:
        blok.append(Blok("subjudul", "4.4 Tabel Hasil"))
        for nomor, (judul, tabel, catatan) in laporan.tabel.items():
            blok.append(Blok("paragraf", f"{nomor}. {judul}"))
            blok.append(Blok("tabel", tabel=tabel, catatan=catatan))

    if laporan.paragraf:
        blok.append(Blok("subjudul", "4.5 Pelaporan Statistik Siap Salin"))
        for paragraf in laporan.paragraf:
            blok.append(Blok("paragraf", paragraf.bagian))
            blok.append(Blok("paragraf", paragraf.teks))

    blok.append(
        Blok(
            "catatan",
            "Pembahasan — mengaitkan temuan ini dengan teori dan penelitian terdahulu "
            "— belum termuat di sini dan harus Anda tulis sendiri.",
        )
    )
    return blok


# --------------------------------------------------------------------------- #
# Bab V — Kesimpulan dan Saran
# --------------------------------------------------------------------------- #


def _bab5(laporan, penelitian, kamus) -> list[Blok]:
    blok = [Blok("subjudul", "5.1 Kesimpulan")]

    pertanyaan = list(getattr(penelitian, "pertanyaan", []) or [])
    if pertanyaan:
        blok.append(
            Blok(
                "paragraf",
                "Kesimpulan disusun menjawab pertanyaan penelitian satu per satu.",
            )
        )
        blok.append(
            Blok("poin", poin=[f"{n}. {t}" for n, t in enumerate(pertanyaan, start=1)])
        )
    if laporan.temuan:
        blok.append(
            Blok("poin", poin=[f"{t.judul}: {t.ringkas}" for t in laporan.temuan])
        )

    if laporan.rekomendasi:
        blok.append(Blok("subjudul", "5.2 Implikasi dan Saran"))
        blok.append(
            Blok("poin", poin=[f"{r.judul} — {r.alasan}" for r in laporan.rekomendasi])
        )

    blok.append(Blok("subjudul", "5.3 Keterbatasan Penelitian"))
    blok.append(Blok("poin", poin=list(laporan.keterbatasan)))

    blok.append(
        Blok(
            "catatan",
            "Batas kesimpulan di atas berasal dari rancangan penelitian, bukan dari "
            "hasil statistiknya. Menghapusnya dari naskah tidak membuatnya tidak "
            "berlaku — ia hanya membuat penguji yang menemukannya lebih dulu.",
        )
    )
    return blok


# --------------------------------------------------------------------------- #
# Artikel jurnal
# --------------------------------------------------------------------------- #


def _imrad(laporan, penelitian, kamus) -> list[Blok]:
    blok = [
        Blok("subjudul", "Abstrak"),
        Blok(
            "paragraf",
            " ".join([laporan.headline, laporan.subheadline])
            + " "
            + " ".join(t.ringkas for t in laporan.temuan[:3]),
        ),
        Blok("subjudul", "Metode"),
    ]
    blok += [b for b in _bab3(laporan, penelitian, kamus) if b.jenis != "subjudul"]
    blok.append(Blok("subjudul", "Hasil"))
    for temuan in laporan.temuan:
        blok.append(Blok("paragraf", temuan.akademik))
    if laporan.tabel:
        for nomor, (judul, tabel, catatan) in laporan.tabel.items():
            blok.append(Blok("paragraf", f"{nomor}. {judul}"))
            blok.append(Blok("tabel", tabel=tabel, catatan=catatan))
    blok.append(Blok("subjudul", "Diskusi"))
    blok.append(
        Blok(
            "paragraf",
            "Bagian diskusi harus Anda tulis sendiri: bagaimana temuan ini sejalan "
            "atau berbeda dengan penelitian terdahulu, dan apa yang menjelaskannya.",
        )
    )
    blok.append(Blok("subjudul", "Keterbatasan"))
    blok.append(Blok("poin", poin=list(laporan.keterbatasan)))
    return blok


def _num(nilai) -> str:
    return f"{int(nilai):,}".replace(",", ".")
