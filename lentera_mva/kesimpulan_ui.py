"""Komponen antarmuka bersama untuk ketiga halaman ringkasan.

Ringkasan Eksekutif, Akademik, dan Profesional berdiri sebagai halaman terpisah,
namun berbagi satu pengaturan cakupan analisis dan satu hasil perhitungan. Modul
ini menyimpan bagian yang dipakai bersama: pengaturan variabel, penjalanan
analisis yang di-cache, gaya tampilan, serta kartu dan grafik yang berulang.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lentera_mva import ekspor, narrative as nr
from lentera_mva import preprocessing, ui

TANPA = "(tidak ada)"

# --------------------------------------------------------------------------- #
# Palet dan gaya, mengikuti tema terang/gelap Streamlit
# --------------------------------------------------------------------------- #

_PALET_GELAP = {
    "sheet": "rgba(255,255,255,0.04)",
    "sheet2": "rgba(255,255,255,0.07)",
    "rule": "rgba(255,255,255,0.16)",
    "ink": "inherit",
    "ink2": "rgba(233,237,246,0.78)",
    "muted": "rgba(233,237,246,0.55)",
    "accent": "#93a6ea",
    "accent2": "#6b81d6",
    "accentWash": "rgba(107,129,214,0.18)",
    "good": "#5cbb8c",
    "warn": "#d8a53f",
    "crit": "#e08074",
}
_PALET_TERANG = {
    "sheet": "rgba(19,26,43,0.025)",
    "sheet2": "rgba(19,26,43,0.05)",
    "rule": "rgba(19,26,43,0.14)",
    "ink": "inherit",
    "ink2": "rgba(19,26,43,0.78)",
    "muted": "rgba(19,26,43,0.55)",
    "accent": "#26356b",
    "accent2": "#3b4ea0",
    "accentWash": "rgba(59,78,160,0.12)",
    "good": "#1b6f4a",
    "warn": "#96690b",
    "crit": "#9c3327",
}


def gelap() -> bool:
    return getattr(getattr(st.context, "theme", None), "type", "light") == "dark"


def palet() -> dict[str, str]:
    return _PALET_GELAP if gelap() else _PALET_TERANG


def warna_status() -> dict[str, str]:
    p = palet()
    return {"baik": p["good"], "perhatian": p["warn"], "kritis": p["crit"]}


def pasang_gaya() -> None:
    """Sisipkan gaya kartu, batang, dan daftar. Dipanggil sekali per halaman."""
    p = palet()
    st.html(
        f"""
<style>
.mva-headline{{border-left:3px solid {p['accent2']};background:{p['sheet2']};
  padding:20px 24px;border-radius:0 10px 10px 0;margin-bottom:6px}}
.mva-headline .stmt{{font-size:1.45rem;line-height:1.32;font-weight:650;margin:0 0 8px;
  max-width:40ch;color:{p['ink']}}}
.mva-headline .note{{font-size:1rem;line-height:1.6;margin:0;color:{p['ink2']};max-width:70ch}}
.mva-lamps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:10px;margin:4px 0 6px}}
.mva-lamp{{border:1px solid {p['rule']};border-radius:10px;padding:13px 15px;background:{p['sheet']}}}
.mva-lamp .lb{{font-size:.86rem;font-weight:650;display:flex;gap:8px;align-items:center;color:{p['ink']}}}
.mva-dot{{width:9px;height:9px;border-radius:50%;flex:0 0 auto}}
.mva-lamp .lv{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.75rem;
  color:{p['muted']};margin:6px 0 5px;font-variant-numeric:tabular-nums}}
.mva-lamp .lc{{font-size:.79rem;line-height:1.45;color:{p['ink2']}}}
.mva-bar{{display:grid;grid-template-columns:22px 1fr;gap:12px;padding:12px 0;
  border-top:1px solid {p['rule']}}}
.mva-bar:first-child{{border-top:0;padding-top:2px}}
.mva-rank{{font-family:ui-monospace,Menlo,monospace;font-size:.78rem;color:{p['muted']};text-align:right}}
.mva-bname{{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px}}
.mva-bname .n{{font-weight:650;font-size:.94rem;color:{p['ink']}}}
.mva-bname .v{{font-family:ui-monospace,Menlo,monospace;font-size:.86rem;font-weight:650;
  font-variant-numeric:tabular-nums;color:{p['ink2']}}}
.mva-track{{height:15px;background:{p['sheet2']};border:1px solid {p['rule']};
  border-radius:4px;overflow:hidden}}
.mva-track i{{display:block;height:100%;background:{p['accent2']}}}
.mva-track i.turun{{background:{p['crit']}}}
.mva-track i.tak{{background:{p['muted']};opacity:.5}}
.mva-bnote{{font-size:.79rem;color:{p['muted']};margin-top:6px;line-height:1.45}}
.mva-list{{margin:0;padding:0;list-style:none;counter-reset:mva}}
.mva-list li{{counter-increment:mva;position:relative;padding:13px 0 13px 36px;
  border-top:1px solid {p['rule']};max-width:78ch}}
.mva-list li:first-child{{border-top:0}}
.mva-list li::before{{content:counter(mva);position:absolute;left:0;top:14px;
  font-family:ui-monospace,Menlo,monospace;font-size:.72rem;font-weight:650;color:{p['accent']};
  background:{p['accentWash']};width:23px;height:23px;border-radius:6px;display:grid;place-items:center}}
.mva-rt{{font-weight:650;font-size:.94rem;margin-bottom:3px;color:{p['ink']}}}
.mva-rd{{font-size:.88rem;line-height:1.6;color:{p['ink2']}}}
.mva-chip{{display:inline-block;font-size:.66rem;font-weight:700;padding:2px 8px;border-radius:999px;
  margin-left:8px;vertical-align:2px;letter-spacing:.03em;text-transform:uppercase}}
.mva-quote{{border:1px solid {p['rule']};border-left:3px solid {p['accent2']};
  background:{p['sheet']};padding:14px 16px;border-radius:0 8px 8px 0;margin-bottom:10px}}
.mva-quote .qh{{font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;
  color:{p['muted']};font-weight:700;margin-bottom:7px}}
.mva-quote .qb{{font-size:.94rem;line-height:1.7;color:{p['ink2']}}}
.mva-meta{{font-family:ui-monospace,Menlo,monospace;font-size:.74rem;color:{p['muted']};margin:0 0 8px}}
</style>
"""
    )


# --------------------------------------------------------------------------- #
# Kartu dan grafik
# --------------------------------------------------------------------------- #


def kartu_headline(laporan: nr.Laporan) -> None:
    st.html(
        f'<div class="mva-headline"><p class="stmt">{escape(laporan.headline)}</p>'
        f'<p class="note">{escape(laporan.subheadline)}</p></div>'
    )


def kartu_lampu(laporan: nr.Laporan) -> None:
    warna = warna_status()
    isi = "".join(
        f'<div class="mva-lamp"><div class="lb">'
        f'<span class="mva-dot" style="background:{warna[l.status]}"></span>{escape(l.label)}</div>'
        f'<div class="lv">{escape(l.nilai)}</div>'
        f'<div class="lc"><span style="color:{warna[l.status]};font-weight:650">'
        f"{escape(l.status_label)}</span> — {escape(l.catatan)}</div></div>"
        for l in laporan.lampu
    )
    st.html(f'<div class="mva-lamps">{isi}</div>')


def batang_pendorong(laporan: nr.Laporan) -> None:
    baris = []
    for i, p in enumerate(laporan.pendorong, start=1):
        kelas = "tak" if not p.signifikan else ("turun" if p.arah == "turun" else "")
        nilai = f"{p.satuan} {nr.num(p.nilai, 3)}"
        baris.append(
            f'<div class="mva-bar"><div class="mva-rank">{i}</div><div>'
            f'<div class="mva-bname"><span class="n">{escape(p.nama)}</span>'
            f'<span class="v">{escape(nilai)}</span></div>'
            f'<div class="mva-track"><i class="{kelas}" '
            f'style="width:{max(p.kekuatan, 0.02) * 100:.1f}%"></i></div>'
            f'<div class="mva-bnote">{escape(p.catatan)}</div></div></div>'
        )
    st.html(
        f'<p class="mva-meta">Sumber: {escape(laporan.pendorong_sumber)} · '
        "batang abu-abu = pengaruh belum terbukti signifikan · "
        "batang merah = arah menurunkan</p>" + "".join(baris)
    )


def daftar_bernomor(items: list[tuple[str, str, str | None]]) -> None:
    p = palet()
    warna_chip = {"tinggi": p["crit"], "sedang": p["warn"], "rendah": p["accent"]}
    butir = []
    for judul, isi, chip in items:
        label = (
            f'<span class="mva-chip" style="background:{p["sheet2"]};'
            f'color:{warna_chip.get(chip, p["muted"])}">prioritas {escape(chip)}</span>'
            if chip
            else ""
        )
        kepala = f'<div class="mva-rt">{escape(judul)}{label}</div>' if judul else ""
        butir.append(f'<li>{kepala}<div class="mva-rd">{escape(isi)}</div></li>')
    st.html(f'<ul class="mva-list">{"".join(butir)}</ul>')


def matriks_prioritas(laporan: nr.Laporan) -> go.Figure:
    """Sebar pendorong: kepentingan (kekuatan pengaruh) terhadap kinerja saat ini."""
    p = palet()
    titik = [d for d in laporan.pendorong if d.kinerja is not None]
    # Label langsung hanya untuk faktor yang kekuatannya minimal separuh dari yang
    # terkuat; faktor lemah menumpuk di kuadran bawah, jadi namanya cukup lewat
    # tooltip agar tidak saling menimpa.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[d.kinerja for d in titik],
            y=[d.kekuatan * 100 for d in titik],
            mode="markers+text",
            text=[d.nama if (d.signifikan and d.kekuatan >= 0.5) else "" for d in titik],
            textposition="top center",
            textfont=dict(size=11),
            customdata=[d.nama for d in titik],
            marker=dict(
                size=14,
                color=[p["accent2"] if d.signifikan else p["muted"] for d in titik],
                line=dict(width=1.5, color="rgba(255,255,255,.7)"),
            ),
            hovertemplate=(
                "<b>%{customdata}</b><br>Kepentingan: %{y:.0f}/100<br>"
                "Kinerja saat ini: persentil %{x:.0f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=50, line_dash="dot", line_color=p["muted"])
    fig.add_vline(x=50, line_dash="dot", line_color=p["muted"])
    for x, y, teks, jangkar, warna in [
        (2, 98, "PRIORITAS PERBAIKAN", "left", p["warn"]),
        (98, 98, "PERTAHANKAN", "right", p["good"]),
        (2, 2, "PRIORITAS RENDAH", "left", p["muted"]),
        (98, 2, "KEMUNGKINAN BERLEBIH", "right", p["muted"]),
    ]:
        fig.add_annotation(
            x=x, y=y, text=teks, showarrow=False, xanchor=jangkar,
            font=dict(size=10, color=warna),
        )
    fig.update_layout(
        template="plotly_dark" if gelap() else "plotly_white",
        title="Matriks prioritas: kepentingan terhadap kinerja",
        xaxis=dict(title="Kinerja saat ini (persentil)", range=[0, 100]),
        yaxis=dict(title="Kepentingan (0–100)", range=[0, 108]),
        margin=dict(l=70, r=20, t=60, b=55),
        height=430,
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# Pengaturan cakupan dan penjalanan analisis
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def _korelasi_absolut(data: pd.DataFrame, kolom: list[str]) -> pd.DataFrame:
    return data[kolom].corr().abs()


def _prediktor_awal(
    df: pd.DataFrame, numerik: list[str], korelasi: pd.DataFrame, target: str, jumlah: int = 4
) -> list[str]:
    """Prediktor yang paling berkaitan dengan target, agar model awal informatif."""
    kandidat = [c for c in numerik if c != target]
    if target in korelasi.columns:
        urut = korelasi[target].drop(labels=[target], errors="ignore")
        return [str(c) for c in urut.sort_values(ascending=False).head(jumlah).index]
    if target in df.columns:
        # Target kategorik dua nilai: pakai korelasi terhadap kodenya (point-biserial).
        kode = pd.Series(pd.Categorical(df[target]).codes.astype(float), index=df.index)
        kaitan = {c: abs(df[c].corr(kode)) for c in kandidat}
        urut = pd.Series(kaitan).dropna().sort_values(ascending=False)
        if not urut.empty:
            return [str(c) for c in urut.head(jumlah).index]
    return kandidat[:jumlah]


def siapkan_laporan(df: pd.DataFrame) -> tuple[nr.Analisis, nr.Laporan]:
    """Tampilkan pengaturan cakupan lalu kembalikan hasil analisis dan laporannya.

    Kunci widget sengaja dipakai bersama oleh ketiga halaman ringkasan, sehingga
    pengaturan yang dibuat di satu halaman langsung berlaku di halaman lainnya dan
    analisis tidak dihitung ulang saat pengguna berpindah halaman.
    """
    nama_data = st.session_state.get(ui.NAME_KEY, "data")
    numerik = preprocessing.numeric_columns(df)
    if len(numerik) < 2:
        st.error("Halaman ini memerlukan minimal 2 kolom numerik.")
        st.stop()

    biner = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    kandidat = [
        c
        for c in df.columns
        if 2 <= df[c].nunique(dropna=True) <= 10 and not pd.api.types.is_float_dtype(df[c])
    ]
    # Kolom kategorik didahulukan karena lebih lazim berperan sebagai penanda kelompok.
    kelompok_kandidat = [c for c in kandidat if not pd.api.types.is_numeric_dtype(df[c])] + [
        c for c in kandidat if pd.api.types.is_numeric_dtype(df[c])
    ]

    # Pilihan awal diarahkan ke variabel yang paling banyak berbagi informasi dengan
    # variabel lain, supaya ringkasan pertama yang dilihat pengguna sudah bermakna.
    korelasi = _korelasi_absolut(df, numerik)
    target_awal = str((korelasi.sum() - 1).sort_values(ascending=False).index[0])

    with st.expander("Atur cakupan analisis", expanded=False):
        st.caption(
            "Pengaturan ini berlaku untuk ketiga ringkasan. Kesimpulan disusun dari "
            "analisis yang benar-benar dijalankan pada data Anda — makin lengkap "
            "pilihan di bawah, makin banyak sudut pandang yang dilaporkan."
        )
        variabel = st.multiselect(
            "Variabel numerik yang dianalisis",
            numerik,
            default=numerik[: min(8, len(numerik))],
            key="kesimpulan_var",
        )
        kol1, kol2 = st.columns(2)
        with kol1:
            target_numerik = st.selectbox(
                "Variabel yang ingin dijelaskan (regresi linear)",
                [TANPA] + numerik,
                index=numerik.index(target_awal) + 1,
                key="kesimpulan_y",
            )
            target_biner = st.selectbox(
                "Variabel hasil dua kategori (regresi logistik)",
                [TANPA] + biner,
                index=1 if biner else 0,
                key="kesimpulan_biner",
            )
        with kol2:
            prediktor_kandidat = [c for c in numerik if c != target_numerik]
            prediktor = st.multiselect(
                "Faktor penjelas (prediktor)",
                prediktor_kandidat,
                default=[
                    c
                    for c in _prediktor_awal(df, numerik, korelasi, target_numerik)
                    if c in prediktor_kandidat
                ],
                key="kesimpulan_x",
            )
            kelompok = st.selectbox(
                "Variabel kelompok (uji beda & diskriminan)",
                [TANPA] + kelompok_kandidat,
                index=1 if kelompok_kandidat else 0,
                key="kesimpulan_kelompok",
            )

        if target_biner != TANPA:
            kandidat_biner = [c for c in numerik if c != target_biner]
            prediktor_biner = st.multiselect(
                f"Faktor penjelas untuk model dua kategori ({target_biner})",
                kandidat_biner,
                default=[
                    c
                    for c in _prediktor_awal(df, numerik, korelasi, target_biner)
                    if c in kandidat_biner
                ],
                key="kesimpulan_x_biner",
            )
        else:
            prediktor_biner = []

    if len(variabel) < 2:
        st.info("Pilih minimal 2 variabel numerik pada 'Atur cakupan analisis'.")
        st.stop()

    konfig = nr.Konfigurasi(
        variabel=variabel,
        nama_data=nama_data,
        target_numerik=None if target_numerik == TANPA else target_numerik,
        prediktor=prediktor,
        target_biner=None if target_biner == TANPA else target_biner,
        prediktor_biner=prediktor_biner,
        kelompok=None if kelompok == TANPA else kelompok,
    )

    # Analisis lengkap cukup mahal, jadi hasilnya dipakai ulang lintas halaman
    # sampai konfigurasi atau datanya berubah.
    tanda = (
        nama_data,
        df.shape,
        tuple(konfig.variabel),
        konfig.target_numerik,
        tuple(konfig.prediktor),
        konfig.target_biner,
        tuple(konfig.prediktor_biner),
        konfig.kelompok,
    )
    if st.session_state.get("kesimpulan_tanda") != tanda:
        with st.spinner("Menjalankan seluruh analisis dan menyusun kesimpulan…"):
            analisis = nr.jalankan_analisis(df, konfig)
            st.session_state["kesimpulan_analisis"] = analisis
            st.session_state["kesimpulan_laporan"] = nr.susun_laporan(analisis)
            st.session_state["kesimpulan_tanda"] = tanda

    laporan = st.session_state["kesimpulan_laporan"]
    st.caption(
        f"{len(laporan.metode_terpakai)} metode dijalankan pada {nr.num(laporan.n_baris)} "
        f"baris data · {laporan.tanggal}"
    )
    return st.session_state["kesimpulan_analisis"], laporan


def buka_ringkasan(
    judul: str, pengantar: str, fitur: str = "ringkasan_eksekutif"
) -> tuple[nr.Analisis, nr.Laporan]:
    """Rangkaian pembuka yang sama untuk ketiga halaman ringkasan."""
    ui.butuh_fitur(fitur)
    ui.page_setup(judul, "Ringkasan Kesimpulan", pengantar)
    df = ui.require_dataset()
    ui.sidebar_info()
    pasang_gaya()
    analisis, laporan = siapkan_laporan(df)
    kartu_headline(laporan)
    st.subheader("Status pemeriksaan")
    kartu_lampu(laporan)
    return analisis, laporan


def analisis_yang_dilewati(laporan: nr.Laporan) -> None:
    if laporan.dilewati:
        with st.expander(f"{len(laporan.dilewati)} analisis tidak dapat dijalankan"):
            for catatan in laporan.dilewati:
                st.markdown(f"- {catatan}")


def unduhan(laporan: nr.Laporan, pembaca: str) -> None:
    """Panel ekspor: pilih ragam laporan, pilih format, lalu unduh.

    Dua ragam yang ditawarkan sesuai isi yang sudah disusun: **Ringkasan** untuk
    register halaman ini saja, dan **Laporan Lengkap** yang memuat ketiga register
    beserta seluruh tabel, kalimat siap salin, dan rujukan ambang.
    """
    st.subheader("Ekspor hasil analisis")
    if not ui.paket_aktif().punya("unduh_laporan"):
        st.info(
            "Ekspor laporan tersedia mulai paket Mahasiswa & Pengajar. Isi ringkasan di "
            "atas tetap dapat dibaca dan disalin.",
            icon=":material/lock:",
        )
        return

    st.caption(
        "Isi seluruh format disusun dari sumber yang sama, sehingga Word, PDF, Excel, "
        "PowerPoint, HTML, dan Markdown memuat angka serta kesimpulan yang identik."
    )

    kiri, kanan = st.columns([1, 1])
    with kiri:
        ragam = st.radio(
            "Ragam laporan",
            ["ringkas", "lengkap"],
            format_func=lambda r: (
                f"Ringkasan {nr.AUDIENCE_LABELS[pembaca]}"
                if r == "ringkas"
                else "Laporan Lengkap (ketiga pembaca)"
            ),
            key=f"ekspor_ragam_{pembaca}",
            horizontal=False,
        )
        lengkap = ragam == "lengkap"
        st.caption(
            "Seluruh temuan dalam register halaman ini, tabel hasil, rekomendasi, dan "
            "batas kesimpulan."
            if not lengkap
            else "Ketiga register pembaca dalam satu berkas, ditambah seluruh tabel, "
            "kalimat siap salin, rujukan ambang, dan catatan analisis yang dilewati."
        )
    with kanan:
        kode = st.selectbox(
            "Format berkas",
            list(ekspor.FORMAT),
            format_func=lambda k: ekspor.FORMAT[k].nama,
            key=f"ekspor_format_{pembaca}",
        )
        st.caption(ekspor.FORMAT[kode].keterangan)

    nama = ekspor.nama_berkas(laporan, kode, pembaca, lengkap)
    try:
        with st.spinner(f"Menyusun berkas {ekspor.FORMAT[kode].nama}…"):
            isi = ekspor.bangun(laporan, kode, pembaca, lengkap)
    except Exception as galat:  # noqa: BLE001 - kegagalan ekspor tidak boleh menghentikan halaman
        st.error(
            f"Berkas {ekspor.FORMAT[kode].nama} gagal disusun: {galat}. "
            "Coba format lain, atau kurangi cakupan analisis.",
            icon=":material/error:",
        )
        return

    st.download_button(
        f"Unduh {ekspor.FORMAT[kode].nama} · {_ukuran(len(isi))}",
        isi,
        file_name=nama,
        mime=ekspor.FORMAT[kode].mime,
        type="primary",
        width="stretch",
        key=f"unduh_{pembaca}_{kode}_{int(lengkap)}",
    )
    st.caption(f"Nama berkas: `{nama}`")

    with st.expander("Format apa yang sebaiknya dipakai?"):
        for f in ekspor.FORMAT.values():
            st.markdown(f"- **{f.nama}** — {f.keterangan}")
        st.caption(
            "Paket lengkap (ZIP) berisi seluruh format sekaligus ditambah setiap tabel "
            "hasil dalam bentuk CSV, sehingga angkanya dapat diolah ulang."
        )


def _ukuran(jumlah_byte: int) -> str:
    """Ukuran berkas dalam satuan yang mudah dibaca."""
    if jumlah_byte < 1024:
        return f"{jumlah_byte} B"
    if jumlah_byte < 1024 * 1024:
        return f"{jumlah_byte / 1024:.0f} KB".replace(".", ",")
    return f"{jumlah_byte / 1024 / 1024:.1f} MB".replace(".", ",")
