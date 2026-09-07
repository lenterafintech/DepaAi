"""Analisis teks dasar: statistik teks, tokenisasi, dan frekuensi kata.

Cakupan sengaja dibatasi pada yang **deskriptif** — statistik teks, tokenisasi,
dan frekuensi kata bersih dari imbuhan — bukan pengkodean tematik atau topic
modeling penuh, mengikuti kesepakatan cakupan saat modul ini dirancang.

Tiga pustaka dipakai langsung, bukan sekadar terpasang untuk kelengkapan:

* **spaCy** (pipeline kosong ``spacy.blank("id")``) untuk tokenisasi kata yang
  lebih andal daripada pemisahan spasi biasa, dan untuk daftar stopword bahasa
  Indonesia bawaannya (tidak perlu mengunduh model terlatih apa pun — pipeline
  kosong dan daftar stopword-nya sudah ikut terpasang bersama paketnya).
* **Sastrawi** untuk stemming bahasa Indonesia, sehingga "menganalisis",
  "dianalisis", dan "analisis" dihitung sebagai kata yang sama pada frekuensi
  kata — tanpa ini, imbuhan bahasa Indonesia yang kaya akan memecah satu kata
  bermakna sama menjadi entri terpisah dan menyesatkan hitungannya.
* **NLTK** (``FreqDist``) untuk menghitung frekuensi kata — dipilih karena
  fungsinya sudah teruji luas, bukan karena tanpanya penghitungan tidak
  mungkin dilakukan.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import spacy
from nltk.probability import FreqDist
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from spacy.lang.id.stop_words import STOP_WORDS as STOPWORD_ID

# Singleton malas: pipeline spaCy dan stemmer Sastrawi mahal dibuat, dan modul
# ini dapat dipanggil berkali-kali dalam satu sesi Streamlit yang sama.
_NLP = None
_STEMMER = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.blank("id")
    return _NLP


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        _STEMMER = StemmerFactory().create_stemmer()
    return _STEMMER


def _tokenisasi(dokumen: str) -> list[str]:
    """Token kata saja (tanda baca dan angka dibuang) lewat tokenizer spaCy."""
    return [t.text for t in _nlp()(str(dokumen)) if t.is_alpha]


@dataclass
class StatistikTeks:
    """Ringkasan ukuran korpus — jumlah dokumen, kata, karakter, dan kosakata."""

    n_dokumen: int
    jumlah_karakter: int
    jumlah_kata: int
    rata_kata_per_dokumen: float
    rata_karakter_per_kata: float
    kosakata_unik: int


def statistik_dasar(teks: pd.Series) -> StatistikTeks:
    """Statistik deskriptif dasar pada satu kolom teks (satu baris = satu dokumen)."""
    dokumen = teks.dropna().astype(str)
    dokumen = dokumen[dokumen.str.strip() != ""]
    if dokumen.empty:
        raise ValueError("Tidak ada teks yang dapat dianalisis setelah baris kosong dibuang.")

    token_per_dokumen = [_tokenisasi(d) for d in dokumen]
    semua_kata = [k for token in token_per_dokumen for k in token]
    jumlah_karakter = int(dokumen.str.len().sum())
    jumlah_kata = len(semua_kata)

    return StatistikTeks(
        n_dokumen=len(dokumen),
        jumlah_karakter=jumlah_karakter,
        jumlah_kata=jumlah_kata,
        rata_kata_per_dokumen=jumlah_kata / len(dokumen) if len(dokumen) else float("nan"),
        rata_karakter_per_kata=jumlah_karakter / jumlah_kata if jumlah_kata else float("nan"),
        kosakata_unik=len({w.lower() for w in semua_kata}),
    )


def bersihkan_token(
    teks: pd.Series,
    hapus_stopword: bool = True,
    stem: bool = True,
    stopword_tambahan: list[str] | None = None,
) -> list[str]:
    """Token bersih dari seluruh dokumen, digabung jadi satu daftar.

    Urutan pembersihan: huruf kecil -> stopword dibuang (opsional) -> stemming
    Sastrawi (opsional) -> stopword diperiksa ulang (bentuk dasar hasil stem
    kadang juga stopword, mis. "nya" dari "-nya"). Token kosong hasil stem
    dibuang.
    """
    stopword = {s.lower() for s in STOPWORD_ID}
    if stopword_tambahan:
        stopword |= {s.strip().lower() for s in stopword_tambahan if s.strip()}

    dokumen = teks.dropna().astype(str)
    mesin_stem = _stemmer() if stem else None
    hasil: list[str] = []
    for d in dokumen:
        for token in _tokenisasi(d):
            kata = token.lower()
            if hapus_stopword and kata in stopword:
                continue
            if mesin_stem is not None:
                kata = mesin_stem.stem(kata)
                if not kata or (hapus_stopword and kata in stopword):
                    continue
            hasil.append(kata)
    return hasil


def frekuensi_kata(token: list[str], n_teratas: int = 30) -> pd.DataFrame:
    """Tabel kata tersering — pengganti word cloud yang dapat diurutkan dan diunduh."""
    if not token:
        return pd.DataFrame(columns=["Kata", "Frekuensi", "Persentase"])
    total = len(token)
    fd = FreqDist(token)
    return pd.DataFrame(
        [
            {"Kata": kata, "Frekuensi": jumlah, "Persentase": jumlah / total * 100}
            for kata, jumlah in fd.most_common(n_teratas)
        ]
    )
