"""Buat berkas contoh data nasabah sintetis untuk mencoba aplikasi.

Jalankan: python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "contoh_data_nasabah.csv"
N = 400
SEED = 7


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    # Tiga faktor laten: kapasitas ekonomi, kedisiplinan bayar, dan intensitas transaksi.
    kapasitas = rng.normal(0, 1, N)
    disiplin = rng.normal(0, 1, N)
    intensitas = 0.35 * kapasitas + rng.normal(0, 0.9, N)

    pendapatan = np.exp(15.2 + 0.45 * kapasitas + rng.normal(0, 0.25, N))
    saldo_tabungan = np.exp(14.0 + 0.55 * kapasitas + 0.2 * disiplin + rng.normal(0, 0.4, N))
    plafon = np.exp(15.8 + 0.5 * kapasitas + rng.normal(0, 0.3, N))
    lama_usaha = np.clip(4 + 2.2 * kapasitas + rng.normal(0, 1.6, N), 0.5, 30)
    usia = np.clip(34 + 5 * kapasitas + rng.normal(0, 7, N), 20, 68).round(0)
    tanggungan = np.clip(rng.poisson(2.2, N) - 0.4 * kapasitas, 0, 8).round(0)

    skor_kredit = np.clip(620 + 55 * disiplin + 20 * kapasitas + rng.normal(0, 25, N), 300, 850)
    rasio_utang = np.clip(0.42 - 0.09 * disiplin - 0.05 * kapasitas + rng.normal(0, 0.07, N), 0.02, 0.95)
    keterlambatan = np.clip(np.round(2.4 - 1.3 * disiplin + rng.normal(0, 0.9, N)), 0, 12)
    riwayat_lunas = np.clip(np.round(6 + 2.5 * disiplin + rng.normal(0, 1.8, N)), 0, 20)

    frekuensi_transaksi = np.clip(np.round(18 + 7 * intensitas + rng.normal(0, 4, N)), 0, 90)
    nilai_transaksi = np.exp(14.6 + 0.4 * intensitas + 0.35 * kapasitas + rng.normal(0, 0.35, N))
    tenor = np.clip(np.round(12 + 6 * kapasitas + rng.normal(0, 5, N) / 1.0), 3, 60)

    segmen = pd.cut(
        kapasitas,
        bins=[-np.inf, -0.45, 0.55, np.inf],
        labels=["Mikro", "Kecil", "Menengah"],
    ).astype(str)
    wilayah = rng.choice(
        ["Jawa", "Sumatera", "Kalimantan", "Sulawesi", "Indonesia Timur"],
        size=N,
        p=[0.42, 0.24, 0.14, 0.12, 0.08],
    )
    jenis_usaha = rng.choice(
        ["Dagang", "Jasa", "Produksi", "Pertanian"], size=N, p=[0.4, 0.28, 0.19, 0.13]
    )

    logit = -3.3 - 1.5 * disiplin - 0.7 * kapasitas + 2.4 * rasio_utang + 0.12 * keterlambatan
    gagal_bayar = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "id_nasabah": [f"NSB{i + 1:04d}" for i in range(N)],
            "usia": usia.astype(int),
            "jumlah_tanggungan": tanggungan.astype(int),
            "lama_usaha_tahun": lama_usaha.round(1),
            "pendapatan_bulanan": pendapatan.round(0),
            "saldo_tabungan": saldo_tabungan.round(0),
            "plafon_pinjaman": plafon.round(0),
            "tenor_bulan": tenor.astype(int),
            "rasio_utang_pendapatan": rasio_utang.round(3),
            "skor_kredit": skor_kredit.round(0),
            "jumlah_keterlambatan": keterlambatan.astype(int),
            "riwayat_pinjaman_lunas": riwayat_lunas.astype(int),
            "frekuensi_transaksi_bulanan": frekuensi_transaksi.astype(int),
            "nilai_transaksi_bulanan": nilai_transaksi.round(0),
            "segmen_usaha": segmen,
            "wilayah": wilayah,
            "jenis_usaha": jenis_usaha,
            "gagal_bayar": gagal_bayar,
        }
    )
    return df


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(OUTPUT, index=False)
    print(f"Tersimpan: {OUTPUT}")
