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
PEMANDU_KEY = "pemandu_konfigurasi"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "contoh_data_nasabah.csv"

# Palet terang: gaya "modern SaaS analytics" (acuan Vercel/Stripe/Linear).
# Warna status sengaja terpisah dari aksen agar "berhasil" dan "aksen merek"
# tidak pernah tertukar maknanya.
WARNA = {
    "tinta": "#0f172a",
    "tinta2": "#334155",
    "redup": "#64748b",
    "garis": "#e2e8f0",
    "garis2": "#f1f5f9",
    "kertas": "#ffffff",
    "kertas2": "#ffffff",
    "dasar": "#f8fafc",
    "aksen": "#4f46e5",
    "aksen2": "#7c6cf0",
    "aksenSamar": "#eef0ff",
    "baik": "#15803d",
    "baikSamar": "#dcfce7",
    "perhatian": "#b45309",
    "perhatianSamar": "#fef3c7",
    "kritis": "#b91c1c",
    "kritisSamar": "#fee2e2",
}

# Palet gelap: bukan pembalikan otomatis, melainkan langkah yang dipilih sendiri.
# Tinta dan kertas bertukar peran, sementara aksen dinaikkan terangnya agar tetap
# terbaca di atas dasar gelap.
WARNA_GELAP = {
    "tinta": "#f1f5f9",
    "tinta2": "#cbd5e1",
    "redup": "#94a3b8",
    "garis": "#293548",
    "garis2": "#1e293b",
    "kertas": "#1e293b",
    "kertas2": "#1e293b",
    "dasar": "#0f172a",
    "aksen": "#818cf8",
    "aksen2": "#6366f1",
    "aksenSamar": "#1e2244",
    "baik": "#4ade80",
    "baikSamar": "#14251b",
    "perhatian": "#fbbf24",
    "perhatianSamar": "#2c2210",
    "kritis": "#f87171",
    "kritisSamar": "#331a1a",
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
    """Lembar gaya halaman, disusun dari token palet yang sedang berlaku.

    Bahasa visualnya "modern SaaS analytics" (acuan Vercel/Stripe/Linear):
    latar netral lembut, kartu putih dengan bayangan halus, aksen biru listrik,
    dan tab berbentuk pil bersegmen menggantikan garis bawah bawaan Streamlit.
    """
    p = palet()
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');

:root {{
{_token(p)}
  --bayang: 0 1px 2px rgba(79, 70, 229, 0.05), 0 6px 20px -10px rgba(79, 70, 229, 0.18);
  --bayang-hover: 0 10px 26px -10px rgba(79, 70, 229, 0.32);
}}

html, body, [class*="css"], .stApp {{
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}
.stApp {{background: var(--dasar)}}

/* ---- Judul & angka besar: serif Fraunces, sisanya tetap Plus Jakarta Sans ---- */
.mva-head h1, .mva-bagian .jd, .mva-langkah .jd,
[data-testid="stMetricValue"] {{font-family: 'Fraunces', serif}}

/* ---- Label kapital kecil (kicker, status strip, metric label): gaya data monospace ---- */
.mva-head .kicker, .mva-bagian .kicker, .mva-strip .lb,
[data-testid="stMetricLabel"] {{
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
}}

/* Lebar halaman dibatasi agar baris teks tidak membentang terlalu panjang. */
/* Bilah header Streamlit melayang di atas isi halaman; padding ini menjaga
   kicker tidak tersembunyi di baliknya. */
.block-container {{max-width: 1240px; padding-top: 4rem; padding-bottom: 4rem}}
.block-container [data-testid="stMarkdownContainer"] p,
.block-container [data-testid="stMarkdownContainer"] li {{max-width: 76ch}}

/* ---- Kepala halaman ---- */
.mva-head {{margin: 0 0 1.3rem; position: relative}}
.mva-head::after {{content: ""; position: absolute; top: -2.5rem; right: -4rem;
  width: 22rem; height: 16rem; pointer-events: none; z-index: -1;
  background: radial-gradient(circle at 60% 30%, var(--aksenSamar), transparent 70%);
  opacity: .7}}
.mva-head .kicker {{display: inline-flex; align-items: center; gap: .5rem;
  font-size: .68rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--aksen2); font-weight: 700; margin-bottom: .5rem}}
.mva-head .tanda {{display: grid; place-items: center; width: 20px; height: 20px;
  border-radius: 6px; background: var(--aksen); color: #fff; font-size: .62rem;
  font-weight: 800; letter-spacing: 0}}
.mva-head h1 {{font-size: 1.85rem; line-height: 1.2; font-weight: 800; margin: 0;
  letter-spacing: -.02em; color: var(--tinta)}}
.mva-head .desc {{font-size: .95rem; line-height: 1.6; color: var(--tinta2);
  margin: .5rem 0 0; max-width: 76ch}}
.mva-head hr {{border: 0; border-top: 1px solid var(--garis); margin: 1.1rem 0 0}}

/* ---- Bilah status data aktif: floating header card, mengambang di atas isi ---- */
.mva-strip {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  border: 1px solid var(--garis); border-radius: 12px; background: var(--kertas);
  box-shadow: var(--bayang); margin: 0 0 1.25rem; overflow: hidden}}
.mva-strip .sel {{padding: .7rem 1rem; min-width: 0}}
.mva-strip .sel + .sel {{border-left: 1px solid var(--garis)}}
.mva-strip .lb {{display: block; font-size: .63rem; font-weight: 700;
  letter-spacing: .09em; text-transform: uppercase; color: var(--redup);
  margin-bottom: .2rem}}
.mva-strip .nl {{display: inline-flex; align-items: center; font-size: .78rem;
  font-weight: 700; color: var(--tinta); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; background: var(--aksenSamar); padding: .15rem .55rem;
  border-radius: 999px; max-width: 100%}}
.mva-strip .titik {{display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  margin-right: .4rem; background: var(--baik); flex: 0 0 auto}}
.mva-strip .titik.sepi {{background: var(--redup)}}

/* ---- Badge pil status, dipakai di mana pun perlu menonjolkan satu nilai ---- */
.mva-pil {{display: inline-flex; align-items: center; gap: .35rem; font-size: .72rem;
  font-weight: 700; padding: .2rem .65rem; border-radius: 999px; line-height: 1.5}}
.mva-pil.baik {{background: var(--baikSamar); color: var(--baik)}}
.mva-pil.perhatian {{background: var(--perhatianSamar); color: var(--perhatian)}}
.mva-pil.kritis {{background: var(--kritisSamar); color: var(--kritis)}}
.mva-pil.info {{background: var(--aksenSamar); color: var(--aksen2)}}
.mva-pil.netral {{background: var(--garis2); color: var(--tinta2)}}
.mva-pil + .mva-pil {{margin-left: .4rem}}

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

/* ---- Kartu umum: dasar bagi grid langkah, KPI, dan kartu kustom lainnya ---- */
.mva-kartu2 {{background: var(--kertas); border: 1px solid var(--garis);
  border-radius: 12px; padding: 1.1rem 1.2rem; box-shadow: var(--bayang);
  transition: transform .15s ease, box-shadow .15s ease; position: relative; overflow: hidden}}
.mva-kartu2:hover {{transform: translateY(-2px); box-shadow: var(--bayang-hover)}}
.mva-kartu2::before {{content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--aksen), var(--aksen2))}}

/* ---- Grid langkah bernomor (alur kerja, wizard) ---- */
.mva-langkah {{display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: .8rem; margin: .6rem 0 1rem}}
.mva-langkah .k {{background: var(--kertas); border: 1px solid var(--garis);
  border-radius: 12px; padding: 1rem 1.1rem; box-shadow: var(--bayang);
  transition: transform .15s ease, box-shadow .15s ease; position: relative; overflow: hidden}}
.mva-langkah .k:hover {{transform: translateY(-2px); box-shadow: var(--bayang-hover)}}
.mva-langkah .k::before {{content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--aksen), var(--aksen2))}}
.mva-langkah .no {{display: inline-grid; place-items: center; width: 26px; height: 26px;
  border-radius: 999px; background: var(--aksenSamar); color: var(--aksen);
  font-size: .74rem; font-weight: 800; margin-bottom: .55rem}}
.mva-langkah .jd {{font-size: .92rem; font-weight: 700; color: var(--tinta);
  margin-bottom: .3rem}}
.mva-langkah .ket {{font-size: .81rem; line-height: 1.5; color: var(--redup)}}

/* ---- Kartu highlight beraksen kiri (ringkasan eksekutif, dsb.) ---- */
.mva-sorot {{background: var(--kertas); border: 1px solid var(--garis);
  border-left: 4px solid var(--aksen); border-radius: 0 12px 12px 0;
  padding: 1.1rem 1.3rem; box-shadow: var(--bayang); margin: .6rem 0 1.1rem}}

/* ---- Keadaan kosong: mengarahkan, bukan sekadar memberi tahu ---- */
.mva-kosong {{text-align: center; border: 1px dashed var(--garis);
  border-radius: 14px; padding: 2rem 1.4rem; margin: 1rem 0;
  background: var(--kertas); color: var(--tinta2)}}
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

/* ---- Tab bersegmen (pill tabs), menggantikan garis bawah bawaan Streamlit ---- */
.stTabs [data-baseweb="tab-list"] {{gap: 4px; background: var(--garis2);
  padding: 5px; border-radius: 12px; border: 1px solid var(--garis)}}
.stTabs [data-baseweb="tab"] {{height: 40px; border-radius: 8px; padding: 0 16px;
  font-weight: 600; font-size: .88rem; color: var(--redup); border: none !important;
  background: transparent; transition: color .15s ease}}
.stTabs [data-baseweb="tab"]:hover {{color: var(--tinta2)}}
.stTabs [aria-selected="true"] {{background: var(--kertas) !important;
  color: var(--aksen) !important;
  box-shadow: var(--bayang), 0 0 0 1px var(--aksenSamar)}}
.stTabs [data-baseweb="tab-highlight"] {{background: transparent}}
.stTabs [data-baseweb="tab-border"] {{display: none}}

/* ---- Komponen bawaan Streamlit ---- */
[data-testid="stMetric"] {{border: 1px solid var(--garis); border-radius: 12px;
  padding: 1rem 1.1rem; background: var(--kertas); box-shadow: var(--bayang);
  transition: transform .15s ease, box-shadow .15s ease; position: relative; overflow: hidden}}
[data-testid="stMetric"]::before {{content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--aksen), var(--aksen2))}}
[data-testid="stMetric"]:hover {{transform: translateY(-2px); box-shadow: var(--bayang-hover)}}
[data-testid="stMetricValue"] {{font-weight: 800; letter-spacing: -.01em}}
[data-testid="stMetricValue"] [data-testid="stMarkdownContainer"],
[data-testid="stMetricValue"] p {{
  white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; line-height: 1.15}}
[data-testid="stExpander"] {{border: 1px solid var(--garis); border-radius: 12px;
  background: var(--kertas); box-shadow: var(--bayang); overflow: hidden}}
[data-testid="stExpander"] summary {{font-weight: 600}}
[data-testid="stDataFrame"] {{border: 1px solid var(--garis); border-radius: 12px;
  overflow: hidden; box-shadow: var(--bayang)}}

.stButton > button, .stDownloadButton > button {{border-radius: 8px; font-weight: 600;
  transition: background .15s ease, box-shadow .15s ease}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background: linear-gradient(135deg, var(--aksen), var(--aksen2)); border-color: var(--aksen)}}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{
  background: linear-gradient(135deg, var(--aksen2), var(--aksen)); border-color: var(--aksen2)}}

div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input {{border-radius: 8px; border-color: var(--garis)}}
div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stNumberInput"] input:focus {{border-color: var(--aksen) !important;
  box-shadow: 0 0 0 3px var(--aksenSamar) !important}}

/* ---- Tag pilihan multiselect (variabel terpilih) sebagai pil bulat netral ---- */
[data-testid="stMultiSelectTagsContainer"] span[data-baseweb="tag"] {{
  background: var(--garis2) !important; border-radius: 999px !important}}
[data-testid="stMultiSelectTagsContainer"] span[data-baseweb="tag"] span {{color: var(--tinta2) !important}}

/* ---- Banner notifikasi (st.info/success/warning/error) ---- */
div[data-testid="stAlertContainer"] {{border-radius: 10px !important; border: 1px solid transparent}}
div[data-testid="stAlertContainer"] p {{font-size: .92rem}}
div[data-testid="stAlertContainer"]:has(> div[data-testid="stAlertContentInfo"]) {{
  background: var(--aksenSamar) !important; color: var(--aksen) !important; border-color: #d8d4fa !important}}
div[data-testid="stAlertContainer"]:has(> div[data-testid="stAlertContentSuccess"]) {{
  background: var(--baikSamar) !important; color: var(--baik) !important; border-color: #bbf7d0 !important}}
div[data-testid="stAlertContainer"]:has(> div[data-testid="stAlertContentWarning"]) {{
  background: var(--perhatianSamar) !important; color: var(--perhatian) !important; border-color: #fde68a !important}}
div[data-testid="stAlertContainer"]:has(> div[data-testid="stAlertContentError"]) {{
  background: var(--kritisSamar) !important; color: var(--kritis) !important; border-color: #fecaca !important}}

/* Menu bawaan Streamlit Cloud (Deploy/Fork) tidak relevan bagi pengguna produk publik. */
[data-testid="stToolbar"], [data-testid="stDecoration"] {{display: none !important}}

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


def langkah_grid(daftar: list[tuple[str, str]]) -> None:
    """Grid kartu bernomor untuk alur kerja atau wizard, satu kartu per langkah.

    ``daftar`` berisi pasangan (judul, keterangan); nomornya diambil dari urutan
    tampil, bukan diketik manual di judulnya, supaya tidak pernah salah hitung.
    """
    kartu = "".join(
        f'<div class="k"><div class="no">{i}</div>'
        f'<div class="jd">{escape(judul)}</div>'
        f'<div class="ket">{escape(ket)}</div></div>'
        for i, (judul, ket) in enumerate(daftar, start=1)
    )
    st.html(f'<div class="mva-langkah">{kartu}</div>')


def pil(teks: str, jenis: str = "netral") -> str:
    """Satu badge pil status siap-tempel di dalam blok HTML lain (mis. tabel/kartu).

    ``jenis``: baik, perhatian, kritis, info, atau netral (bawaan).
    """
    return f'<span class="mva-pil {escape(jenis)}">{escape(teks)}</span>'


def siapkan_aplikasi() -> None:
    """Konfigurasi global, dipanggil sekali di puncak ``app.py`` sebelum tab mana pun dirender.

    Aplikasi satu halaman tidak lagi memanggil ``page_setup`` per halaman — ukuran
    halaman, gaya, dan giliran penyimpanan hasil (``awali_giliran``) hanya berlaku
    sekali per render, bukan sekali per halaman seperti pada arsitektur lama.
    """
    if not st.session_state.get("_page_configured"):
        st.set_page_config(page_title="NalarData", page_icon="📊", layout="wide")
        st.session_state["_page_configured"] = True
    st.html(_gaya())
    awali_giliran()


def kepala_aplikasi() -> None:
    """Bilah merek tipis di puncak aplikasi, tampil pada tab apa pun yang sedang dibuka."""
    st.html(
        '<div class="mva-head" style="margin-bottom:.7rem"><div class="kicker">'
        '<span class="tanda">ND</span>NalarData — analisis multivariat, tanpa harus '
        "jadi ahli statistik</div></div>"
    )


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
            f"Tersedia pada paket **{saran.nama}**. Buka tab 👤 **Akun** "
            "untuk mengubah paket."
        )


def butuh_fitur(kode_fitur: str, judul: str = "") -> bool:
    """Tampilkan pesan kunci bila fitur tidak termasuk paket aktif; kembalikan izinnya.

    Sebelum aplikasi satu halaman, fungsi ini menghentikan seluruh render lewat
    ``st.stop()`` — sah ketika satu berkas memang satu-satunya isi layar. Pada
    tata letak bertab, menghentikan seluruh render dari dalam satu tab akan ikut
    menutup tab lain yang tidak terkait. Sekarang ia mengembalikan ``False`` dan
    pemanggilnya sendiri yang menghentikan render **bagian itu saja**
    (``if not ui.butuh_fitur(...): return``), bukan seluruh aplikasi.

    ``judul`` dipakai sebagai judul kartu terkunci. Tanpa itu, keterangan fitur
    yang panjang dipakai apa adanya sebagai tajuk — "Simulasi sidang: latihan
    menjawab pertanyaan penguji" terbaca sebagai kalimat, bukan nama bagian.
    """
    pelanggaran = langganan.periksa_fitur(paket_aktif(), kode_fitur)
    if pelanggaran is None:
        return True
    keterangan = langganan.FITUR.get(kode_fitur, "")
    judul_bagian(
        judul or keterangan.split(":")[0].strip() or "Fitur terkunci",
        f"{keterangan} — belum termasuk dalam paket Anda." if keterangan
        else "Bagian ini belum termasuk dalam paket Anda.",
        kicker="Terkunci",
    )
    _ajakan_naik(pelanggaran)
    return False


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


def set_konfigurasi_pemandu(konfig: dict) -> None:
    """Simpan penetapan variabel dari Pemandu Uji agar halaman metode terisi sendiri."""
    st.session_state[PEMANDU_KEY] = dict(konfig)


def konfigurasi_pemandu(metode: str | None = None) -> dict:
    """Penetapan variabel dari Pemandu, bila memang untuk metode yang diminta.

    Dikembalikan kosong bila pengguna membuka halaman ini sendiri, bukan lewat
    pemandu — mengisi pilihan orang yang tidak memintanya justru membingungkan.
    """
    konfig = st.session_state.get(PEMANDU_KEY) or {}
    if metode and konfig.get("metode") != metode:
        return {}
    return dict(konfig)


def indeks_pilihan(daftar: list, nilai, bawaan: int = 0) -> int:
    """Letak ``nilai`` pada daftar pilihan; bawaan bila tidak ada.

    Dipakai memasang pilihan awal selectbox tanpa menggagalkan halaman ketika
    kolom yang disarankan pemandu sudah tidak ada pada data yang sekarang.
    """
    try:
        return daftar.index(nilai)
    except (ValueError, AttributeError):
        return bawaan


def banner_dipandu(dipandu: dict) -> None:
    """Banner "disiapkan dari Pemandu Uji", dipanggil dari tiap halaman metode
    yang menerima konfigurasi dari :func:`konfigurasi_pemandu`.

    Tidak menampilkan apa pun bila ``dipandu`` kosong (pengguna membuka
    halaman ini sendiri, bukan lewat Pemandu) — mengisi pilihan orang yang
    tidak memintanya justru membingungkan.
    """
    if not dipandu.get("metode"):
        return
    st.success(
        f"Disiapkan dari Pemandu Uji: **{dipandu['metode']}**. Pilihan di bawah sudah "
        "terisi sesuai variabel yang Anda tentukan di sana, dan tetap dapat diubah.",
        icon=":material/explore:",
    )


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


def dataset_valid(df: pd.DataFrame | None) -> bool:
    """Apakah data aktif ada dan berada dalam batas paket.

    Menggantikan ``require_dataset`` lama yang menghentikan seluruh skrip lewat
    ``st.stop()`` begitu data kosong atau melampaui batas — pada aplikasi satu
    halaman itu akan ikut menghentikan tab lain yang tidak terkait. ``app.py``
    memanggil ini sekali per render, lalu meneruskan hasilnya ke setiap panel
    yang membutuhkan data (lihat ``pesan_data_diperlukan`` untuk pesannya).
    """
    if df is None:
        return False
    # Contoh data bawaan dikecualikan dari batas ukuran paket. Ia lebih besar
    # daripada batas paket Gratis, sehingga tanpa pengecualian ini tombol
    # "Muat contoh data" milik aplikasi sendiri justru mengantar pengguna baru
    # ke dinding berbayar sebelum ia sempat melihat apa pun.
    if st.session_state.get(CONTOH_KEY):
        return True
    return langganan.periksa_ukuran(paket_aktif(), len(df), df.shape[1]) is None


def pesan_data_diperlukan(df: pd.DataFrame | None) -> None:
    """Pesan yang tepat untuk data yang belum ada atau melampaui batas paket."""
    if df is None:
        keadaan_kosong(
            "Perlu data terlebih dahulu",
            "Muat data pada tab Data — unggah berkas, coba data contoh, atau buka "
            "proyek yang tersimpan.",
            ikon="📁",
        )
        return
    pelanggaran = langganan.periksa_ukuran(paket_aktif(), len(df), df.shape[1])
    if pelanggaran is not None:
        _ajakan_naik(pelanggaran)


def numeric_selector(
    df: pd.DataFrame,
    label: str = "Variabel numerik yang dianalisis",
    default_count: int = 6,
    min_selection: int = 2,
    key: str | None = None,
    default: list[str] | None = None,
) -> list[str]:
    """``default`` (mis. dari :func:`konfigurasi_pemandu`) menggantikan
    ``default_count`` kolom pertama bila diberikan dan valid — dipakai
    halaman yang menerima variabel dari Pemandu Uji."""
    options = preprocessing.numeric_columns(df)
    if len(options) < min_selection:
        st.error(
            f"Data hanya punya {len(options)} kolom numerik, minimal {min_selection} dibutuhkan."
        )
        st.stop()
    dari_pemandu = [c for c in (default or []) if c in options]
    awal = dari_pemandu if len(dari_pemandu) >= min_selection else options[: min(default_count, len(options))]
    selected = st.multiselect(label, options, default=awal, key=key)
    if len(selected) < min_selection:
        st.info(f"Pilih minimal {min_selection} variabel untuk melanjutkan.")
        st.stop()
    return selected


def group_selector(
    df: pd.DataFrame,
    label: str = "Variabel kelompok",
    max_levels: int = 20,
    key: str | None = None,
    default: str | None = None,
) -> str:
    """``default`` (mis. dari :func:`konfigurasi_pemandu`) dipakai sebagai
    pilihan awal bila ada di antara kandidat — dipakai halaman yang
    menerima variabel dari Pemandu Uji."""
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
    return st.selectbox(label, options, index=indeks_pilihan(options, default), key=key)


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
        catat_hasil(bagian, judul or _judul_dari_berkas(filename), tabel=df, catatan=catatan)


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
        catat_hasil(bagian, "Cara membaca", teks=text, jenis="tafsiran")


# Kategori (bagian) yang sudah disegarkan pada render skrip yang sedang berjalan.
# Direset sekali di awal tiap giliran lewat ``awali_giliran``, supaya beberapa
# pemanggilan ``catat_hasil`` dengan bagian yang sama dalam satu render (misalnya
# tabel koefisien dan tabel asumsi pada panel Regresi) saling menambah, bukan
# saling menimpa — sementara hasil dari render sebelumnya tetap terganti bersih.
_DISEGARKAN_KEY = "kategori_disegarkan_giliran_ini"


def awali_giliran() -> None:
    """Reset penanda kategori yang sudah disegarkan. Dipanggil sekali di puncak app.py.

    Tanpa ini, kategori yang pernah disegarkan pada giliran pertama tidak akan
    pernah disegarkan lagi pada giliran berikutnya — hasil lama dari metode yang
    sama akan menumpuk alih-alih tergantikan.
    """
    st.session_state[_DISEGARKAN_KEY] = set()


def catat_hasil(
    bagian: str,
    judul: str,
    tabel: pd.DataFrame | None = None,
    teks: str = "",
    catatan: str = "",
    jenis: str = "tabel",
) -> None:
    """Simpan satu keluaran ke Laporan, otomatis — tidak ada lagi tombol manual.

    Setiap kategori metode menyimpan **satu hasil terakhir**: menjalankan ulang
    metode yang sama mengganti hasil lama sekategori, bukan menumpuknya. Ini
    menyatukan mesin laporan otomatis dan keranjang manual yang sebelumnya
    terpisah — pengguna tidak lagi memilih apakah harus menekan "simpan".
    """
    isi = keranjang()
    disegarkan = st.session_state.setdefault(_DISEGARKAN_KEY, set())
    if bagian not in disegarkan:
        isi.hapus_bagian(bagian)
        disegarkan.add(bagian)
    isi.tambah(
        kr.Item(
            bagian=bagian,
            judul=judul,
            jenis=jenis,
            tabel=tabel,
            teks=teks,
            catatan=catatan,
            tanda_data=tanda_data(),
        )
    )


def sumber_angka(teks: str, indeks, kunci: str, label: str = "Lihat sumber angka") -> None:
    """Tampilkan penelusuran tiap angka pada sebuah paragraf kembali ke sel tabelnya.

    Ditaruh berdampingan dengan paragrafnya, bukan di lampiran: pembaca yang harus
    mencari sendiri ke halaman lain tidak akan memeriksanya, dan paragraf yang tidak
    diperiksa sama saja dengan paragraf yang harus dipercaya begitu saja.
    """
    from nalardata import sumber as sm

    if not indeks or not str(teks).strip():
        return

    tabel = sm.ringkas(teks, indeks)
    if tabel.empty:
        return

    hilang = int((tabel["Sumber"] == "tidak ditemukan di tabel mana pun").sum())
    judul = label if not hilang else f"{label} — {hilang} angka belum bertabel"
    with st.expander(judul, expanded=False):
        st.dataframe(tabel, width="stretch", hide_index=True, key=f"sumber_{kunci}")
        if hilang:
            st.caption(
                "Angka yang belum bertabel tetap benar — ia dihitung mesin statistik "
                "yang sama — namun belum punya sel yang dapat ditunjuk pembimbing. "
                "Sebutkan sendiri asalnya bila dikutip pada naskah."
            )


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
