"""Pembantu antarmuka Streamlit yang dipakai bersama oleh seluruh halaman."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from nalardata import formatting, jejak as jj, kamus as km, keranjang as kr, langganan
from nalardata import pengguna as pg, preprocessing
from nalardata import proyek_penelitian as pp

DATA_KEY = "dataset"
NAME_KEY = "dataset_name"
KERANJANG_KEY = "keranjang_hasil"
KAMUS_KEY = "kamus_variabel"
CONTOH_KEY = "data_adalah_contoh"
PENELITIAN_KEY = "proyek_penelitian"
JEJAK_KEY = "jejak_langkah"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "contoh_data_nasabah.csv"

# Palet terang. Warna status sengaja terpisah dari aksen agar "berhasil" dan
# "aksen merek" tidak pernah tertukar maknanya.
WARNA = {
    "tinta": "#131a2b",
    "tinta2": "#3d4860",
    "redup": "#6f7a91",
    "garis": "#dde3ee",
    "garis2": "#eef1f6",
    "kertas": "#ffffff",
    "kertas2": "#f6f8fc",
    "aksen": "#26356b",
    "aksen2": "#3b4ea0",
    "aksenSamar": "#eef1f8",
    "baik": "#1b6f4a",
    "baikSamar": "#e4f0ea",
    "perhatian": "#96690b",
    "perhatianSamar": "#f8f1e0",
    "kritis": "#9c3327",
    "kritisSamar": "#f7e7e4",
}

# Palet gelap: bukan pembalikan otomatis, melainkan langkah yang dipilih sendiri.
# Tinta dan kertas bertukar peran, sementara aksen dinaikkan terangnya agar tetap
# terbaca di atas dasar gelap.
WARNA_GELAP = {
    "tinta": "#e9edf6",
    "tinta2": "#b3bccf",
    "redup": "#8b96ac",
    "garis": "#2a344c",
    "garis2": "#212a3e",
    "kertas": "#141b2c",
    "kertas2": "#1a2235",
    "aksen": "#93a6ea",
    "aksen2": "#8098e0",
    "aksenSamar": "#1d2740",
    "baik": "#5cbb8c",
    "baikSamar": "#162b22",
    "perhatian": "#d8a53f",
    "perhatianSamar": "#2e2513",
    "kritis": "#e08074",
    "kritisSamar": "#331b18",
}


def gelap() -> bool:
    """Apakah Streamlit sedang memakai tema gelap?

    Tema Streamlit yang menentukan, bukan ``prefers-color-scheme`` peramban:
    pengguna dapat memilih tema di dalam aplikasi, dan pilihan itu tidak selalu
    sama dengan pengaturan sistem operasinya.
    """
    return getattr(getattr(st, "context", None), "theme", None) is not None and (
        getattr(st.context.theme, "type", "light") == "dark"
    )


def palet() -> dict[str, str]:
    return WARNA_GELAP if gelap() else WARNA


def _token(warna: dict[str, str]) -> str:
    """Palet menjadi custom property CSS, sehingga warna hidup di satu tempat."""
    return "\n".join(f"  --{nama}: {nilai};" for nama, nilai in warna.items())


def _gaya() -> str:
    """Lembar gaya halaman, disusun dari token palet yang sedang berlaku."""
    p = palet()
    return f"""
<style>
:root {{
{_token(p)}
}}

/* Lebar halaman dibatasi agar baris teks tidak membentang terlalu panjang. */
/* Bilah header Streamlit melayang di atas isi halaman; padding ini menjaga
   kicker tidak tersembunyi di baliknya. */
.block-container {{max-width: 1240px; padding-top: 4rem; padding-bottom: 4rem}}
.block-container [data-testid="stMarkdownContainer"] p,
.block-container [data-testid="stMarkdownContainer"] li {{max-width: 76ch}}

/* ---- Kepala halaman ---- */
.mva-head {{margin: 0 0 1.3rem}}
.mva-head .kicker {{display: inline-flex; align-items: center; gap: .5rem;
  font-size: .68rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--aksen2); font-weight: 700; margin-bottom: .5rem}}
.mva-head .tanda {{display: grid; place-items: center; width: 20px; height: 20px;
  border-radius: 6px; background: var(--aksen); color: #fff; font-size: .62rem;
  font-weight: 800; letter-spacing: 0}}
.mva-head h1 {{font-size: 1.85rem; line-height: 1.2; font-weight: 700; margin: 0;
  letter-spacing: -.015em; color: var(--tinta)}}
.mva-head .desc {{font-size: .95rem; line-height: 1.6; color: var(--tinta2);
  margin: .5rem 0 0; max-width: 76ch}}
.mva-head hr {{border: 0; border-top: 1px solid var(--garis); margin: 1.1rem 0 0}}

/* ---- Bilah status data aktif, di atas halaman bukan tersembunyi di sidebar ---- */
.mva-strip {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  border: 1px solid var(--garis); border-radius: 11px; background: var(--kertas2);
  margin: 0 0 1.25rem; overflow: hidden}}
.mva-strip .sel {{padding: .6rem .9rem; min-width: 0}}
.mva-strip .sel + .sel {{border-left: 1px solid var(--garis)}}
.mva-strip .lb {{display: block; font-size: .63rem; font-weight: 750;
  letter-spacing: .09em; text-transform: uppercase; color: var(--redup);
  margin-bottom: .15rem}}
.mva-strip .nl {{display: block; font-size: .85rem; font-weight: 700;
  color: var(--tinta); white-space: nowrap; overflow: hidden; text-overflow: ellipsis}}
.mva-strip .titik {{display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  margin-right: .4rem; background: var(--baik)}}
.mva-strip .titik.sepi {{background: var(--redup)}}

/* ---- Judul bagian dengan batang aksen ---- */
.mva-bagian {{display: flex; align-items: flex-start; gap: .7rem; margin: 1.7rem 0 .7rem}}
.mva-bagian .batang {{width: 3px; align-self: stretch; min-height: 34px;
  border-radius: 3px; background: var(--aksen2)}}
.mva-bagian .kicker {{font-size: .62rem; font-weight: 800; letter-spacing: .12em;
  text-transform: uppercase; color: var(--aksen2); margin-bottom: .15rem}}
.mva-bagian .jd {{font-size: 1.06rem; font-weight: 700; color: var(--tinta);
  letter-spacing: -.012em; line-height: 1.3}}
.mva-bagian .ket {{font-size: .81rem; color: var(--redup); line-height: 1.5;
  margin-top: .2rem; max-width: 76ch}}

/* ---- Keadaan kosong: mengarahkan, bukan sekadar memberi tahu ---- */
.mva-kosong {{text-align: center; border: 1px dashed var(--garis);
  border-radius: 14px; padding: 2rem 1.4rem; margin: 1rem 0;
  background: var(--kertas2); color: var(--tinta2)}}
.mva-kosong .ikon {{display: grid; place-items: center; width: 44px; height: 44px;
  margin: 0 auto .7rem; border-radius: 13px; background: var(--aksenSamar);
  color: var(--aksen2); font-size: 1.2rem}}
.mva-kosong b {{display: block; color: var(--tinta); font-size: 1rem;
  margin-bottom: .3rem}}
.mva-kosong .ket {{font-size: .86rem; line-height: 1.6; max-width: 56ch;
  margin: 0 auto}}

/* ---- Kotak "cara membaca" ---- */
.mva-baca {{border-left: 3px solid var(--aksen2); background: var(--aksenSamar);
  padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0 1rem;
  font-size: .88rem; line-height: 1.6; color: var(--tinta2); max-width: 82ch}}
.mva-baca b {{color: var(--tinta)}}

/* ---- Panel sidebar ---- */
.mva-data, .mva-akun {{border: 1px solid var(--garis); border-radius: 9px;
  padding: .6rem .75rem}}
.mva-data {{background: var(--aksenSamar)}}
.mva-akun {{margin-bottom: .5rem}}
.mva-data .nama, .mva-akun .nm {{font-size: .83rem; font-weight: 650;
  color: var(--tinta); overflow-wrap: anywhere}}
.mva-data .rinci {{font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .72rem; color: var(--redup); margin-top: .2rem}}
.mva-akun .pk {{font-size: .74rem; font-weight: 700; letter-spacing: .04em;
  text-transform: uppercase; color: var(--aksen2); margin-top: .2rem}}
.mva-akun .al {{font-size: .72rem; color: var(--redup); margin-top: .25rem;
  line-height: 1.4}}

/* ---- Komponen bawaan Streamlit ---- */
[data-testid="stMetric"] {{border: 1px solid var(--garis); border-radius: 11px;
  padding: .8rem .9rem; background: var(--kertas2)}}
[data-testid="stExpander"] {{border: 1px solid var(--garis); border-radius: 11px}}
[data-testid="stDataFrame"] {{border: 1px solid var(--garis); border-radius: 10px;
  overflow: hidden}}
.stButton > button, .stDownloadButton > button {{border-radius: 8px; font-weight: 650}}

/* Gerak dihentikan bagi yang memintanya lewat pengaturan sistem. */
@media (prefers-reduced-motion: reduce) {{
  * {{transition: none !important; animation: none !important}}
}}
@media (max-width: 800px) {{
  .block-container {{padding-left: .8rem; padding-right: .8rem}}
  .mva-head h1 {{font-size: 1.5rem}}
  .mva-strip .sel + .sel {{border-left: 0; border-top: 1px solid var(--garis)}}
}}
</style>
"""


def strip_status() -> None:
    """Bilah status data aktif di atas halaman.

    Sebelumnya keterangan ini hanya ada di sidebar, yang mudah terlewat dan tertutup
    pada layar sempit. Menaruhnya di atas membuat pengguna selalu tahu data mana yang
    sedang dianalisis — kekeliruan yang paling mahal adalah menganalisis data yang
    salah tanpa menyadarinya.
    """
    df = get_dataset()
    paket = paket_aktif()

    if df is None:
        sel = [
            ("Data aktif", '<span class="titik sepi"></span>Belum ada'),
            ("Paket", escape(paket.nama)),
        ]
    else:
        nama = str(st.session_state.get(NAME_KEY, "data"))
        numerik = len(preprocessing.numeric_columns(df))
        sel = [
            ("Data aktif", f'<span class="titik"></span>{escape(nama)}'),
            ("Ukuran", f"{formatting.num(len(df))} baris × {df.shape[1]} kolom"),
            ("Numerik", f"{numerik} kolom"),
            ("Paket", escape(paket.nama)),
        ]

    isi = "".join(
        f'<div class="sel"><span class="lb">{label}</span>'
        f'<span class="nl">{nilai}</span></div>'
        for label, nilai in sel
    )
    st.html(f'<div class="mva-strip">{isi}</div>')


def judul_bagian(judul: str, keterangan: str = "", kicker: str = "") -> None:
    """Judul bagian dengan batang aksen, seragam di seluruh halaman."""
    bagian = [
        f'<div class="kicker">{escape(kicker)}</div>' if kicker else "",
        f'<div class="jd">{escape(judul)}</div>',
        f'<div class="ket">{escape(keterangan)}</div>' if keterangan else "",
    ]
    st.html(
        '<div class="mva-bagian"><div class="batang"></div><div>'
        + "".join(bagian)
        + "</div></div>"
    )


def keadaan_kosong(judul: str, keterangan: str, ikon: str = "○") -> None:
    """Keadaan kosong yang mengarahkan langkah berikutnya, bukan sekadar memberi tahu."""
    st.html(
        f'<div class="mva-kosong"><div class="ikon">{escape(ikon)}</div>'
        f"<b>{escape(judul)}</b>"
        f'<div class="ket">{escape(keterangan)}</div></div>'
    )


def page_setup(title: str, kicker: str = "NalarData", description: str = "") -> None:
    """Kepala halaman standar: kicker, judul, deskripsi, garis pemisah.

    Ikon sengaja tidak dipakai di sini — penanda visual tiap halaman sudah ada pada
    menu sisi kiri, sehingga judul cukup berupa teks dan tidak menyaingi isi halaman.
    """
    if not st.session_state.get("_page_configured"):
        st.set_page_config(page_title="NalarData", page_icon="📊", layout="wide")
        st.session_state["_page_configured"] = True
    st.html(_gaya())
    deskripsi = f'<p class="desc">{description}</p>' if description else ""
    st.html(
        f'<div class="mva-head"><div class="kicker">'
        f'<span class="tanda">ND</span>{escape(kicker)}</div>'
        f"<h1>{escape(title)}</h1>{deskripsi}<hr></div>"
    )
    strip_status()


PAKET_KEY = "paket_langganan"
PENGGUNA_KEY = "pengguna_id"


def pengguna_aktif() -> pg.Pengguna | None:
    """Akun yang sedang masuk pada sesi ini, atau None bila belum masuk."""
    id_pengguna = st.session_state.get(PENGGUNA_KEY)
    if id_pengguna is None:
        return None
    return pg.ambil_dengan_id(int(id_pengguna))


def set_pengguna(akun: pg.Pengguna) -> None:
    st.session_state[PENGGUNA_KEY] = akun.id
    # Paket mengikuti akun, sehingga penanda paket pada sesi tidak lagi dipakai.
    st.session_state.pop(PAKET_KEY, None)


def keluar() -> None:
    """Akhiri sesi dan bersihkan data yang sedang dianalisis."""
    for kunci in (PENGGUNA_KEY, PAKET_KEY, DATA_KEY, NAME_KEY, KERANJANG_KEY):
        st.session_state.pop(kunci, None)


def paket_aktif() -> langganan.Paket:
    """Paket yang berlaku: dari akun bila sudah masuk, dengan uji coba diperhitungkan.

    Penanda paket pada sesi tetap dihormati agar pengujian otomatis dan peragaan
    dapat memilih paket tanpa membuat akun.
    """
    penanda_sesi = st.session_state.get(PAKET_KEY)
    if penanda_sesi:
        return langganan.ambil_paket(penanda_sesi)
    akun = pengguna_aktif()
    if akun is not None:
        return langganan.ambil_paket(akun.paket_efektif())
    return langganan.ambil_paket(langganan.PAKET_BAWAAN)


def set_paket(kode: str) -> None:
    """Ubah paket: tersimpan ke akun bila sudah masuk, selain itu hanya di sesi."""
    aman = langganan.ambil_paket(kode).kode
    akun = pengguna_aktif()
    if akun is not None:
        pg.set_paket(akun.surel, aman)
        st.session_state.pop(PAKET_KEY, None)
    else:
        st.session_state[PAKET_KEY] = aman


def _ajakan_naik(pelanggaran: langganan.Pelanggaran) -> None:
    saran = langganan.PAKET.get(pelanggaran.saran_paket or "")
    st.warning(pelanggaran.pesan)
    if saran:
        st.caption(
            f"Tersedia pada paket **{saran.nama}**. Buka halaman *Akun & Langganan* "
            "untuk mengubah paket."
        )


def butuh_fitur(kode_fitur: str) -> None:
    """Hentikan halaman bila fitur tidak termasuk paket yang sedang aktif."""
    pelanggaran = langganan.periksa_fitur(paket_aktif(), kode_fitur)
    if pelanggaran is not None:
        page_setup(
            langganan.FITUR.get(kode_fitur, "Fitur terkunci"),
            "Terkunci",
            "Halaman ini belum termasuk dalam paket Anda.",
        )
        _ajakan_naik(pelanggaran)
        st.stop()


def set_dataset(
    df: pd.DataFrame,
    name: str,
    label_spss: dict | None = None,
    contoh: bool = False,
) -> None:
    """Pasang data aktif dan selaraskan kamus variabelnya.

    Kamus diselaraskan, bukan disusun ulang: pengguna yang mengunggah data versi
    perbaikan tidak boleh kehilangan definisi operasional yang sudah ia tulis.
    """
    st.session_state[DATA_KEY] = df
    st.session_state[NAME_KEY] = name
    st.session_state[CONTOH_KEY] = bool(contoh)
    lama = st.session_state.get(KAMUS_KEY)
    if lama is None:
        st.session_state[KAMUS_KEY] = km.Kamus.dari_data(df, label_spss)
    else:
        st.session_state[KAMUS_KEY] = lama.selaraskan(df)


def kamus() -> km.Kamus:
    """Kamus variabel untuk data aktif, dibuat saat pertama kali dipakai."""
    df = get_dataset()
    simpan = st.session_state.get(KAMUS_KEY)
    if simpan is None:
        simpan = km.Kamus() if df is None else km.Kamus.dari_data(df)
        st.session_state[KAMUS_KEY] = simpan
    elif df is not None and simpan.kolom != [str(k) for k in df.columns]:
        # Data berganti tanpa lewat set_dataset (misalnya dipulihkan dari proyek).
        simpan = simpan.selaraskan(df)
        st.session_state[KAMUS_KEY] = simpan
    return simpan


def set_kamus(baru: km.Kamus) -> None:
    st.session_state[KAMUS_KEY] = baru


def penelitian() -> pp.ProyekPenelitian:
    """Rancangan penelitian sesi ini, dibuat dengan nilai bawaan bila belum diisi.

    Selalu mengembalikan objek, tidak pernah None: batas kesimpulan harus tetap
    dapat dicetak sekalipun pengguna melewati Ruang Proyek — dan bawaannya adalah
    yang paling berhati-hati (potong lintang, sampling purposif).
    """
    if PENELITIAN_KEY not in st.session_state:
        st.session_state[PENELITIAN_KEY] = pp.ProyekPenelitian()
    return st.session_state[PENELITIAN_KEY]


def set_penelitian(baru: pp.ProyekPenelitian) -> None:
    st.session_state[PENELITIAN_KEY] = baru


def jejak() -> jj.Jejak:
    """Jejak langkah sesi ini, dibuat saat pertama kali dipakai."""
    if JEJAK_KEY not in st.session_state:
        st.session_state[JEJAK_KEY] = jj.Jejak()
    return st.session_state[JEJAK_KEY]


def set_jejak(baru: jj.Jejak) -> None:
    st.session_state[JEJAK_KEY] = baru


def catat_uji(nama: str, halaman: str = "", p: float | None = None, rincian: str = "") -> None:
    """Catat satu uji yang benar-benar dijalankan.

    Dipanggil halaman metode sesudah hasilnya keluar. Nilai p yang dicatat adalah
    nilai p utama uji itu — yang akan dilaporkan — bukan seluruh nilai p pada tabel.
    """
    jejak().catat_uji(nama, halaman=halaman, p=p, rincian=rincian)


def get_dataset() -> pd.DataFrame | None:
    return st.session_state.get(DATA_KEY)


def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def require_dataset() -> pd.DataFrame:
    """Ambil data aktif; hentikan halaman dengan pesan bila belum ada."""
    df = get_dataset()
    if df is None:
        st.warning(
            "Belum ada data. Buka halaman **Beranda & Data** untuk mengunggah berkas "
            "CSV/Excel atau memuat contoh data."
        )
        if st.button("Muat contoh data nasabah", type="primary"):
            set_dataset(load_sample(), "contoh_data_nasabah.csv", contoh=True)
            st.rerun()
        st.stop()
    # Contoh data bawaan dikecualikan dari batas ukuran paket. Ia lebih besar
    # daripada batas paket Gratis, sehingga tanpa pengecualian ini tombol
    # "Muat contoh data" milik aplikasi sendiri justru mengantar pengguna baru
    # ke dinding berbayar sebelum ia sempat melihat apa pun.
    if not st.session_state.get(CONTOH_KEY):
        pelanggaran = langganan.periksa_ukuran(paket_aktif(), len(df), df.shape[1])
        if pelanggaran is not None:
            _ajakan_naik(pelanggaran)
            st.stop()
    return df


def sidebar_info() -> None:
    """Ringkasan data aktif pada sidebar, dibuat ringkas agar tidak memakan ruang."""
    df = get_dataset()
    akun = pengguna_aktif()
    with st.sidebar:
        st.divider()
        if akun is not None:
            paket = paket_aktif()
            st.html(
                f'<div class="mva-akun"><div class="nm">{akun.nama}</div>'
                f'<div class="pk">{paket.nama}</div>'
                f'<div class="al">{akun.alasan_paket()}</div></div>'
            )
            if st.button("Keluar", key="tombol_keluar", width="stretch"):
                keluar()
                st.rerun()
        if df is None:
            st.caption("Belum ada data dimuat.")
            return
        paket = paket_aktif()
        st.caption(f"Paket {paket.nama}")
        nama = st.session_state.get(NAME_KEY, "data")
        numerik = len(preprocessing.numeric_columns(df))
        st.html(
            f'<div class="mva-data"><div class="nama">{nama}</div>'
            f'<div class="rinci">{formatting.num(len(df))} baris · {df.shape[1]} kolom · '
            f"{numerik} numerik</div></div>"
        )


def numeric_selector(
    df: pd.DataFrame,
    label: str = "Variabel numerik yang dianalisis",
    default_count: int = 6,
    min_selection: int = 2,
    key: str | None = None,
) -> list[str]:
    options = preprocessing.numeric_columns(df)
    if len(options) < min_selection:
        st.error(
            f"Data hanya punya {len(options)} kolom numerik, minimal {min_selection} dibutuhkan."
        )
        st.stop()
    default = options[: min(default_count, len(options))]
    selected = st.multiselect(label, options, default=default, key=key)
    if len(selected) < min_selection:
        st.info(f"Pilih minimal {min_selection} variabel untuk melanjutkan.")
        st.stop()
    return selected


def group_selector(
    df: pd.DataFrame,
    label: str = "Variabel kelompok",
    max_levels: int = 20,
    key: str | None = None,
) -> str:
    candidates = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= max_levels]
    # Kolom kategorik didahulukan karena lebih lazim berperan sebagai penanda kelompok.
    options = [c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])] + [
        c for c in candidates if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not options:
        st.error(
            f"Tidak ada kolom yang cocok sebagai variabel kelompok "
            f"(butuh 2 sampai {max_levels} kategori)."
        )
        st.stop()
    return st.selectbox(label, options, key=key)


def format_number(value: object) -> str:
    """Angka desimal bergaya Indonesia, dengan notasi ilmiah untuk nilai ekstrem."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    number = float(value)
    besaran = abs(number)
    if besaran != 0 and (besaran < 1e-4 or besaran >= 1e9):
        return f"{number:.3e}".replace(".", ",")
    if number.is_integer():
        return formatting.num(int(number))
    return formatting.num(number, 4)


def _warna_keputusan(nilai: object) -> str:
    nada = formatting.nada_keputusan(nilai)
    if nada == "baik":
        return f"color: {WARNA['baik']}; font-weight: 600"
    if nada == "buruk":
        return f"color: {WARNA['kritis']}; font-weight: 600"
    return ""


def keranjang() -> kr.Keranjang:
    """Keranjang hasil milik sesi ini, dibuat saat pertama kali dipakai."""
    if KERANJANG_KEY not in st.session_state:
        st.session_state[KERANJANG_KEY] = kr.Keranjang()
    return st.session_state[KERANJANG_KEY]


def tanda_data() -> str:
    """Identitas data aktif, dipakai menandai hasil yang datanya sudah berganti."""
    df = get_dataset()
    if df is None:
        return ""
    nama = st.session_state.get(NAME_KEY, "data")
    return f"{nama}#{len(df)}x{df.shape[1]}"


def _judul_dari_berkas(filename: str) -> str:
    """Judul cadangan dari nama berkas unduhan, misalnya regresi_koefisien.csv."""
    dasar = str(filename).rsplit(".", 1)[0]
    return dasar.replace("_", " ").strip().capitalize() or "Tabel hasil"


def styled(df: pd.DataFrame):
    """Format angka, nilai p, dan beri warna pada kolom keputusan.

    Penataan dilewati pada tabel panjang (misalnya pratinjau data mentah) karena
    biayanya tidak sebanding dengan manfaatnya di sana.
    """
    if len(df) > 250:
        return df
    format_kolom: dict = {}
    for kolom in df.columns:
        if not pd.api.types.is_numeric_dtype(df[kolom]):
            continue
        if formatting.kolom_p(kolom):
            format_kolom[kolom] = formatting.pval_ringkas
        elif pd.api.types.is_float_dtype(df[kolom]):
            format_kolom[kolom] = format_number
        else:
            format_kolom[kolom] = formatting.num
    kolom_keputusan = [c for c in df.columns if formatting.kolom_keputusan(c)]
    if not format_kolom and not kolom_keputusan:
        return df
    gaya = df.style.format(format_kolom) if format_kolom else df.style
    if kolom_keputusan:
        gaya = gaya.map(_warna_keputusan, subset=kolom_keputusan)
    return gaya


def show_table(
    df: pd.DataFrame,
    filename: str,
    height: int | None = None,
    bagian: str | None = None,
    judul: str | None = None,
    catatan: str = "",
) -> None:
    """Tampilkan tabel beserta tombol unduh CSV.

    Bila ``bagian`` diisi, tabel dapat disimpan ke keranjang hasil sehingga ikut
    terbawa ke berkas ekspor. Argumen itu sengaja opsional agar puluhan pemanggilan
    yang sudah ada tetap berjalan tanpa diubah.
    """
    tampilan = styled(df)
    if height is None:
        st.dataframe(tampilan, width="stretch", hide_index=True)
    else:
        st.dataframe(tampilan, width="stretch", height=height, hide_index=True)
    st.download_button(
        "Unduh tabel (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}_{abs(hash(tuple(df.columns))) % 10**6}",
    )
    if bagian:
        simpan_ke_keranjang(bagian, judul or _judul_dari_berkas(filename), df, catatan)


# Penanda markdown sederhana yang dipakai pada teks tafsiran. Urutannya penting:
# ** harus diproses sebelum * agar tebal tidak terbaca sebagai dua miring.
_MARKDOWN_RINGKAS = (
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"<i>\1</i>"),
)


def _markdown_ringkas(teks: str) -> str:
    """Ubah tebal, miring, dan kode menjadi HTML setelah isinya diamankan.

    ``st.html`` tidak memproses markdown, sehingga ``**penting**`` sebelumnya tampil
    sebagai bintang harfiah di layar. Teks diamankan lebih dulu agar tanda kurung
    sudut pada nama variabel tidak pernah menjadi tag.
    """
    hasil = escape(teks)
    for pola, ganti in _MARKDOWN_RINGKAS:
        hasil = pola.sub(ganti, hasil)
    return hasil


def interpretation(text: str, bagian: str | None = None) -> None:
    st.html(f'<div class="mva-baca"><b>Cara membaca:</b> {_markdown_ringkas(text)}</div>')
    if bagian:
        simpan_ke_keranjang(bagian, "Cara membaca", teks=text, jenis="tafsiran")


def simpan_ke_keranjang(
    bagian: str,
    judul: str,
    tabel: pd.DataFrame | None = None,
    catatan: str = "",
    teks: str = "",
    jenis: str = "tabel",
) -> None:
    """Tombol simpan satu hasil ke keranjang, beserta penanda bila sudah tersimpan.

    Penyimpanan sengaja atas permintaan pengguna, bukan otomatis: menangkap setiap
    tabel yang pernah terlihat akan memenuhi laporan dengan keluaran percobaan yang
    tidak jadi dipakai.
    """
    isi = keranjang()
    calon = kr.Item(
        bagian=bagian,
        judul=judul,
        jenis=jenis,
        tabel=tabel,
        teks=teks,
        catatan=catatan,
        tanda_data=tanda_data(),
    )
    sudah = any(i.sidik == calon.sidik for i in isi.item)
    kunci = f"simpan_{calon.sidik}"
    if sudah:
        st.caption(":material/check: Tersimpan di **Laporan Hasil**.")
        return
    if st.button(
        "Simpan ke laporan",
        key=kunci,
        help="Hasil ini akan muncul di halaman Laporan Hasil dan ikut saat diekspor.",
    ):
        isi.tambah(calon)
        st.rerun()


def method_note(title: str, body: str) -> None:
    with st.expander(f"Tentang {title}"):
        st.markdown(body)


def preprocessing_controls(key_prefix: str = "") -> tuple[str, str]:
    """Kontrol standar penanganan missing dan penskalaan."""
    col1, col2 = st.columns(2)
    with col1:
        missing = st.selectbox(
            "Penanganan nilai hilang",
            preprocessing.MISSING_STRATEGIES,
            key=f"{key_prefix}_missing",
        )
    with col2:
        scaling = st.selectbox(
            "Penskalaan variabel",
            preprocessing.SCALING_METHODS,
            index=1,
            key=f"{key_prefix}_scaling",
        )
    return missing, scaling


def prepare_numeric(
    df: pd.DataFrame, columns: list[str], missing: str, scaling: str
) -> pd.DataFrame:
    subset = preprocessing.handle_missing(df[columns], missing)
    return preprocessing.scale(subset, scaling)
