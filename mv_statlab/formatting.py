"""Format angka dan penanda statistik bergaya penulisan Indonesia.

Dipakai bersama oleh penyusun narasi maupun tabel antarmuka, supaya angka yang
sama tampil identik di mana pun ia muncul: koma sebagai pemisah desimal, titik
sebagai pemisah ribuan.
"""

from __future__ import annotations

import numpy as np

# Kolom yang berisi nilai p dikenali dari namanya agar dapat diformat khusus.
KOLOM_P = ("p-value", "p value", "p_value", "sig.", "signifikansi", "p")
# Kolom keputusan diberi penanda warna pada tabel.
KOLOM_KEPUTUSAN = (
    "signifikan",
    "kesimpulan",
    "keputusan",
    "pembeda",
    "status",
    "interpretasi",
)
NILAI_BAIK = {"ya", "terpenuhi", "signifikan", "layak", "baik", "memadai", "aman", "didukung"}
NILAI_BURUK = {
    "tidak",
    "tidak terpenuhi",
    "tidak signifikan",
    "tidak layak",
    "bermasalah",
    "tidak normal",
}


def num(value: float | int | None, digits: int = 2) -> str:
    """Angka bergaya Indonesia: 1234.5678 → "1.234,57"."""
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}".replace(",", ".")
    if not np.isfinite(float(value)):
        return "-"
    text = f"{float(value):,.{digits}f}"
    # Tukar pemisah ribuan dan desimal sekaligus ke gaya penulisan Indonesia.
    return text.translate(str.maketrans({",": ".", ".": ","}))


def num_auto(value: float | None, digits: int = 4) -> str:
    """Seperti :func:`num`, tetapi beralih ke notasi ilmiah untuk nilai sangat kecil."""
    if value is None:
        return "-"
    angka = float(value)
    if not np.isfinite(angka):
        return "-"
    if angka != 0 and abs(angka) < 10**-digits:
        return f"{angka:.2e}".replace(".", ",")
    return num(angka, digits)


def pval(p: float | None) -> str:
    """Nilai p dalam kalimat: "p < 0,001" atau "p = 0,023"."""
    if p is None or not np.isfinite(p):
        return "p tidak tersedia"
    if p < 0.001:
        return "p < 0,001"
    return f"p = {num(p, 3)}"


def pval_ringkas(p: float | None) -> str:
    """Nilai p untuk sel tabel, tanpa awalan "p"."""
    if p is None or not np.isfinite(p):
        return "-"
    if p < 0.001:
        return "< 0,001"
    return num(p, 3)


def pct(value: float, digits: int = 1) -> str:
    return f"{num(value, digits)}%"


def bintang(p: float) -> str:
    """Penanda signifikansi bergaya APA."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def kolom_p(nama: object) -> bool:
    teks = str(nama).strip().lower()
    return teks in KOLOM_P or teks.startswith("p-value")


def kolom_keputusan(nama: object) -> bool:
    return str(nama).strip().lower() in KOLOM_KEPUTUSAN


def nada_keputusan(nilai: object) -> str:
    """Kembalikan "baik", "buruk", atau "" untuk isi sel kolom keputusan."""
    teks = str(nilai).strip().lower()
    if teks in NILAI_BAIK:
        return "baik"
    if teks in NILAI_BURUK or teks.startswith("tidak"):
        return "buruk"
    return ""
