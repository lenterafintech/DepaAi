"""Pembantu antarmuka Streamlit yang dipakai bersama oleh seluruh halaman."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lentera_mva import formatting, keranjang as kr, langganan, pengguna as pg, preprocessing

DATA_KEY = "dataset"
NAME_KEY = "dataset_name"
KERANJANG_KEY = "keranjang_hasil"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "contoh_data_nasabah.csv"

# Palet laporan: netral berbias biru, aksen navy, warna status terpisah dari aksen.
WARNA = {
    "tinta": "#131a2b",
    "tinta2": "#3d4860",
    "redup": "#6f7a91",
    "garis": "#dde3ee",
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

_GAYA = f"""
<style>
/* Lebar halaman dibatasi agar baris teks tidak membentang terlalu panjang. */
.block-container {{max-width: 1200px; padding-top: 2.2rem; padding-bottom: 4rem}}
.block-container [data-testid="stMarkdownContainer"] p,
.block-container [data-testid="stMarkdownContainer"] li {{max-width: 76ch}}

/* Kepala halaman: kicker kecil, judul, deskripsi, lalu garis pemisah. */
.mva-head {{margin: 0 0 1.4rem}}
.mva-head .kicker {{font-size: .68rem; letter-spacing: .14em; text-transform: uppercase;
  color: {WARNA['aksen2']}; font-weight: 700; margin-bottom: .45rem}}
.mva-head h1 {{font-size: 1.85rem; line-height: 1.2; font-weight: 700; margin: 0;
  letter-spacing: -.015em; color: {WARNA['tinta']}}}
.mva-head .desc {{font-size: .95rem; line-height: 1.6; color: {WARNA['tinta2']};
  margin: .5rem 0 0; max-width: 76ch}}
.mva-head hr {{border: 0; border-top: 1px solid {WARNA['garis']}; margin: 1.1rem 0 0}}

/* Kotak "cara membaca" dibuat tenang, bukan biru menyala bawaan. */
.mva-baca {{border-left: 3px solid {WARNA['aksen2']}; background: {WARNA['aksenSamar']};
  padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .6rem 0 1rem;
  font-size: .88rem; line-height: 1.6; color: {WARNA['tinta2']}; max-width: 82ch}}
.mva-baca b {{color: {WARNA['tinta']}}}

/* Panel data aktif pada sidebar. */
.mva-data {{border: 1px solid {WARNA['garis']}; border-radius: 9px; padding: .6rem .75rem;
  background: {WARNA['aksenSamar']}}}
.mva-data .nama {{font-size: .82rem; font-weight: 650; color: {WARNA['tinta']};
  overflow-wrap: anywhere}}
.mva-data .rinci {{font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .72rem; color: {WARNA['redup']}; margin-top: .2rem}}
.mva-akun {{border: 1px solid {WARNA['garis']}; border-radius: 9px; padding: .6rem .75rem;
  margin-bottom: .5rem}}
.mva-akun .nm {{font-size: .85rem; font-weight: 650; color: {WARNA['tinta']};
  overflow-wrap: anywhere}}
.mva-akun .pk {{font-size: .74rem; font-weight: 700; letter-spacing: .04em;
  text-transform: uppercase; color: {WARNA['aksen2']}; margin-top: .2rem}}
.mva-akun .al {{font-size: .72rem; color: {WARNA['redup']}; margin-top: .25rem;
  line-height: 1.4}}
</style>
"""


def page_setup(title: str, kicker: str = "Lentera MVA", description: str = "") -> None:
    """Kepala halaman standar: kicker, judul, deskripsi, garis pemisah.

    Ikon sengaja tidak dipakai di sini — penanda visual tiap halaman sudah ada pada
    menu sisi kiri, sehingga judul cukup berupa teks dan tidak menyaingi isi halaman.
    """
    if not st.session_state.get("_page_configured"):
        st.set_page_config(page_title="Lentera MVA", page_icon="📊", layout="wide")
        st.session_state["_page_configured"] = True
    st.html(_GAYA)
    deskripsi = f'<p class="desc">{description}</p>' if description else ""
    st.html(
        f'<div class="mva-head"><div class="kicker">{kicker}</div>'
        f"<h1>{title}</h1>{deskripsi}<hr></div>"
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


def set_dataset(df: pd.DataFrame, name: str) -> None:
    st.session_state[DATA_KEY] = df
    st.session_state[NAME_KEY] = name


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
            set_dataset(load_sample(), "contoh_data_nasabah.csv")
            st.rerun()
        st.stop()
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


def interpretation(text: str, bagian: str | None = None) -> None:
    st.html(f'<div class="mva-baca"><b>Cara membaca:</b> {text}</div>')
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
