"""Halaman kesimpulan otomatis dalam tiga register pembaca.

Seluruh analisis dijalankan sekali, lalu hasilnya diterjemahkan menjadi narasi
untuk eksekutif/pengguna awam, kalangan akademik, dan praktisi profesional.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lentera_mva import narrative as nr
from lentera_mva import preprocessing, ui
from lentera_mva.report_html import laporan_html, laporan_html_semua

ui.page_setup("Kesimpulan Analisis", "🧾")
df = ui.require_dataset()
ui.sidebar_info()
nama_data = st.session_state.get(ui.NAME_KEY, "data")

TANPA = "(tidak ada)"


# --------------------------------------------------------------------------- #
# Gaya tampilan yang mengikuti tema terang/gelap Streamlit
# --------------------------------------------------------------------------- #

_gelap = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
_PALET = (
    {
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
    if _gelap
    else {
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
)

STATUS_WARNA = {"baik": _PALET["good"], "perhatian": _PALET["warn"], "kritis": _PALET["crit"]}

st.html(
    f"""
<style>
.mva-headline{{border-left:3px solid {_PALET['accent2']};background:{_PALET['sheet2']};
  padding:20px 24px;border-radius:0 10px 10px 0;margin-bottom:6px}}
.mva-headline .stmt{{font-size:1.45rem;line-height:1.32;font-weight:650;margin:0 0 8px;
  max-width:40ch;color:{_PALET['ink']}}}
.mva-headline .note{{font-size:1rem;line-height:1.6;margin:0;color:{_PALET['ink2']};max-width:70ch}}
.mva-lamps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:10px;margin:4px 0 6px}}
.mva-lamp{{border:1px solid {_PALET['rule']};border-radius:10px;padding:13px 15px;background:{_PALET['sheet']}}}
.mva-lamp .lb{{font-size:.86rem;font-weight:650;display:flex;gap:8px;align-items:center;color:{_PALET['ink']}}}
.mva-dot{{width:9px;height:9px;border-radius:50%;flex:0 0 auto}}
.mva-lamp .lv{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.75rem;
  color:{_PALET['muted']};margin:6px 0 5px;font-variant-numeric:tabular-nums}}
.mva-lamp .lc{{font-size:.79rem;line-height:1.45;color:{_PALET['ink2']}}}
.mva-bar{{display:grid;grid-template-columns:22px 1fr;gap:12px;padding:12px 0;
  border-top:1px solid {_PALET['rule']}}}
.mva-bar:first-child{{border-top:0;padding-top:2px}}
.mva-rank{{font-family:ui-monospace,Menlo,monospace;font-size:.78rem;color:{_PALET['muted']};text-align:right}}
.mva-bname{{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px}}
.mva-bname .n{{font-weight:650;font-size:.94rem;color:{_PALET['ink']}}}
.mva-bname .v{{font-family:ui-monospace,Menlo,monospace;font-size:.86rem;font-weight:650;
  font-variant-numeric:tabular-nums;color:{_PALET['ink2']}}}
.mva-track{{height:15px;background:{_PALET['sheet2']};border:1px solid {_PALET['rule']};
  border-radius:4px;overflow:hidden}}
.mva-track i{{display:block;height:100%;background:{_PALET['accent2']}}}
.mva-track i.turun{{background:{_PALET['crit']}}}
.mva-track i.tak{{background:{_PALET['muted']};opacity:.5}}
.mva-bnote{{font-size:.79rem;color:{_PALET['muted']};margin-top:6px;line-height:1.45}}
.mva-list{{margin:0;padding:0;list-style:none;counter-reset:mva}}
.mva-list li{{counter-increment:mva;position:relative;padding:13px 0 13px 36px;
  border-top:1px solid {_PALET['rule']};max-width:78ch}}
.mva-list li:first-child{{border-top:0}}
.mva-list li::before{{content:counter(mva);position:absolute;left:0;top:14px;
  font-family:ui-monospace,Menlo,monospace;font-size:.72rem;font-weight:650;color:{_PALET['accent']};
  background:{_PALET['accentWash']};width:23px;height:23px;border-radius:6px;display:grid;place-items:center}}
.mva-rt{{font-weight:650;font-size:.94rem;margin-bottom:3px;color:{_PALET['ink']}}}
.mva-rd{{font-size:.88rem;line-height:1.6;color:{_PALET['ink2']}}}
.mva-chip{{display:inline-block;font-size:.66rem;font-weight:700;padding:2px 8px;border-radius:999px;
  margin-left:8px;vertical-align:2px;letter-spacing:.03em;text-transform:uppercase}}
.mva-quote{{border:1px solid {_PALET['rule']};border-left:3px solid {_PALET['accent2']};
  background:{_PALET['sheet']};padding:14px 16px;border-radius:0 8px 8px 0;margin-bottom:10px}}
.mva-quote .qh{{font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;
  color:{_PALET['muted']};font-weight:700;margin-bottom:7px}}
.mva-quote .qb{{font-size:.94rem;line-height:1.7;color:{_PALET['ink2']}}}
.mva-meta{{font-family:ui-monospace,Menlo,monospace;font-size:.74rem;color:{_PALET['muted']};margin:0 0 8px}}
</style>
"""
)


def kartu_headline(laporan: nr.Laporan) -> None:
    st.html(
        f'<div class="mva-headline"><p class="stmt">{escape(laporan.headline)}</p>'
        f'<p class="note">{escape(laporan.subheadline)}</p></div>'
    )


def kartu_lampu(laporan: nr.Laporan) -> None:
    isi = "".join(
        f'<div class="mva-lamp"><div class="lb">'
        f'<span class="mva-dot" style="background:{STATUS_WARNA[l.status]}"></span>{escape(l.label)}</div>'
        f'<div class="lv">{escape(l.nilai)}</div>'
        f'<div class="lc"><span style="color:{STATUS_WARNA[l.status]};font-weight:650">'
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
        f"batang abu-abu = pengaruh belum terbukti signifikan · "
        f"batang merah = arah menurunkan</p>" + "".join(baris)
    )


def daftar_bernomor(items: list[tuple[str, str, str | None]]) -> None:
    warna_chip = {
        "tinggi": _PALET["crit"],
        "sedang": _PALET["warn"],
        "rendah": _PALET["accent"],
    }
    butir = []
    for judul, isi, chip in items:
        label = (
            f'<span class="mva-chip" style="background:{_PALET["sheet2"]};'
            f'color:{warna_chip.get(chip, _PALET["muted"])}">prioritas {escape(chip)}</span>'
            if chip
            else ""
        )
        kepala = f'<div class="mva-rt">{escape(judul)}{label}</div>' if judul else ""
        butir.append(f'<li>{kepala}<div class="mva-rd">{escape(isi)}</div></li>')
    st.html(f'<ul class="mva-list">{"".join(butir)}</ul>')


def matriks_prioritas(laporan: nr.Laporan) -> go.Figure:
    """Sebar pendorong: kepentingan (kekuatan pengaruh) vs kinerja saat ini."""
    titik = [p for p in laporan.pendorong if p.kinerja is not None]
    fig = go.Figure()
    warna = [_PALET["accent2"] if p.signifikan else _PALET["muted"] for p in titik]
    fig.add_trace(
        go.Scatter(
            x=[p.kinerja for p in titik],
            y=[p.kekuatan * 100 for p in titik],
            mode="markers+text",
            text=[p.nama for p in titik],
            textposition="top center",
            marker=dict(size=14, color=warna, line=dict(width=1.5, color="rgba(255,255,255,.7)")),
            hovertemplate=(
                "<b>%{text}</b><br>Kepentingan: %{y:.0f}/100<br>"
                "Kinerja saat ini: persentil %{x:.0f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=50, line_dash="dot", line_color=_PALET["muted"])
    fig.add_vline(x=50, line_dash="dot", line_color=_PALET["muted"])
    fig.add_annotation(
        x=2, y=98, text="PRIORITAS PERBAIKAN", showarrow=False, xanchor="left",
        font=dict(size=10, color=_PALET["warn"]),
    )
    fig.add_annotation(
        x=98, y=98, text="PERTAHANKAN", showarrow=False, xanchor="right",
        font=dict(size=10, color=_PALET["good"]),
    )
    fig.add_annotation(
        x=2, y=2, text="PRIORITAS RENDAH", showarrow=False, xanchor="left",
        font=dict(size=10, color=_PALET["muted"]),
    )
    fig.add_annotation(
        x=98, y=2, text="KEMUNGKINAN BERLEBIH", showarrow=False, xanchor="right",
        font=dict(size=10, color=_PALET["muted"]),
    )
    fig.update_layout(
        template="plotly_dark" if _gelap else "plotly_white",
        title="Matriks prioritas: kepentingan terhadap kinerja",
        xaxis=dict(title="Kinerja saat ini (persentil)", range=[0, 100]),
        yaxis=dict(title="Kepentingan (0–100)", range=[0, 108]),
        margin=dict(l=70, r=20, t=60, b=55),
        height=430,
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# Konfigurasi analisis
# --------------------------------------------------------------------------- #

numerik = preprocessing.numeric_columns(df)
if len(numerik) < 2:
    st.error("Halaman ini memerlukan minimal 2 kolom numerik.")
    st.stop()

biner = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
_kandidat = [
    c
    for c in df.columns
    if 2 <= df[c].nunique(dropna=True) <= 10 and not pd.api.types.is_float_dtype(df[c])
]
# Kolom kategorik didahulukan karena lebih lazim berperan sebagai penanda kelompok.
kelompok_kandidat = [c for c in _kandidat if not pd.api.types.is_numeric_dtype(df[c])] + [
    c for c in _kandidat if pd.api.types.is_numeric_dtype(df[c])
]


@st.cache_data(show_spinner=False)
def peta_korelasi(data: pd.DataFrame, kolom: list[str]) -> pd.DataFrame:
    return data[kolom].corr().abs()


# Pilihan awal diarahkan ke variabel yang paling banyak berbagi informasi dengan
# variabel lain, supaya kesimpulan pertama yang dilihat pengguna sudah bermakna.
korelasi_abs = peta_korelasi(df, numerik)
peringkat = (korelasi_abs.sum() - 1).sort_values(ascending=False)
target_awal = str(peringkat.index[0])


def prediktor_awal(target: str, jumlah: int = 4) -> list[str]:
    """Prediktor yang paling berkaitan dengan target, agar model awal informatif."""
    kandidat = [c for c in numerik if c != target]
    if target in korelasi_abs.columns:
        urut = korelasi_abs[target].drop(labels=[target], errors="ignore")
        return [str(c) for c in urut.sort_values(ascending=False).head(jumlah).index]
    if target in df.columns:
        # Target kategorik dua nilai: pakai korelasi terhadap kodenya (point-biserial).
        kode = pd.Categorical(df[target]).codes.astype(float)
        kaitan = {c: abs(df[c].corr(pd.Series(kode, index=df.index))) for c in kandidat}
        urut = pd.Series(kaitan).dropna().sort_values(ascending=False)
        if not urut.empty:
            return [str(c) for c in urut.head(jumlah).index]
    return kandidat[:jumlah]

with st.expander("Atur cakupan analisis", expanded=False):
    st.caption(
        "Kesimpulan disusun dari analisis yang benar-benar dijalankan pada data Anda. "
        "Makin lengkap pilihan di bawah, makin banyak sudut pandang yang dilaporkan."
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
            default=[c for c in prediktor_awal(target_numerik) if c in prediktor_kandidat],
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
            default=[c for c in prediktor_awal(target_biner) if c in kandidat_biner],
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

# Analisis lengkap cukup mahal, jadi hasilnya disimpan sampai konfigurasinya berubah.
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

analisis: nr.Analisis = st.session_state["kesimpulan_analisis"]
laporan: nr.Laporan = st.session_state["kesimpulan_laporan"]

st.caption(
    f"{len(laporan.metode_terpakai)} metode dijalankan pada {nr.num(laporan.n_baris)} baris "
    f"data · {laporan.tanggal}"
)

pembaca_label = st.segmented_control(
    "Kesimpulan ini ditujukan untuk",
    options=list(nr.AUDIENCES),
    format_func=lambda a: nr.AUDIENCE_LABELS[a],
    default="eksekutif",
    key="kesimpulan_pembaca",
)
pembaca = pembaca_label or "eksekutif"

kartu_headline(laporan)
st.subheader("Status pemeriksaan")
kartu_lampu(laporan)

# --------------------------------------------------------------------------- #
# Bagian khusus tiap pembaca
# --------------------------------------------------------------------------- #

if pembaca == "eksekutif":
    if laporan.pendorong:
        st.subheader("Peringkat pendorong")
        kiri, kanan = st.columns([1.15, 1])
        with kiri:
            batang_pendorong(laporan)
        with kanan:
            if any(p.kinerja is not None for p in laporan.pendorong):
                st.plotly_chart(matriks_prioritas(laporan), width="stretch")
        ui.interpretation(
            "Panjang batang menunjukkan seberapa kuat pengaruh sebuah faktor dibanding "
            "faktor terkuat. Pada matriks di sampingnya, faktor di kuadran kiri atas "
            "adalah yang penting tetapi kinerjanya masih di bawah rata-rata — di sanalah "
            "perbaikan paling terasa hasilnya."
        )

    st.subheader("Apa yang ditemukan")
    for temuan in laporan.temuan:
        with st.expander(f"**{temuan.judul}** — {temuan.ringkas}"):
            st.write(temuan.eksekutif)

    st.subheader("Rekomendasi tindakan")
    daftar_bernomor([(r.judul, r.alasan, r.prioritas) for r in laporan.rekomendasi])

    st.subheader("Batas kesimpulan")
    daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

elif pembaca == "akademik":
    st.subheader("Ikhtisar hasil")
    st.html(
        '<div class="mva-quote"><div class="qh">Abstrak temuan</div><div class="qb">'
        + escape(" ".join(t.ringkas for t in laporan.temuan))
        + "</div></div>"
    )

    st.subheader("Temuan dan pelaporan statistik")
    for temuan in laporan.temuan:
        st.markdown(f"**{temuan.judul}**")
        st.caption(f"Metode: {temuan.metode}")
        st.write(temuan.akademik)

    if laporan.tabel:
        st.subheader("Tabel hasil")
        for nomor, (judul, tabel, catatan) in laporan.tabel.items():
            st.markdown(f"**{nomor}.** {judul}")
            st.dataframe(tabel, width="stretch", hide_index=True)
            if catatan:
                st.caption(f"*Catatan.* {catatan}")

    st.subheader("Kalimat siap salin")
    st.caption(
        "Paragraf berikut mengikuti konvensi pelaporan statistik dan dapat langsung "
        "disalin ke naskah. Sesuaikan nama variabel dengan istilah pada penelitian Anda."
    )
    for paragraf in laporan.paragraf:
        st.markdown(f"**{paragraf.bagian}**")
        st.code(paragraf.teks, language=None, wrap_lines=True)

    st.subheader("Keterbatasan dan saran penelitian lanjutan")
    daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

    st.subheader("Rujukan ambang yang dipakai")
    st.caption(
        "Daftar ini memuat rujukan ambang statistik yang dipakai aplikasi, bukan rujukan "
        "teoretis penelitian Anda. Sesuaikan gaya sitasi dengan pedoman institusi."
    )
    for rujukan in laporan.rujukan:
        st.markdown(f"- {rujukan}")

else:  # profesional
    st.subheader("Metrik kunci")
    metrik = st.columns(4)
    slot = 0
    if analisis.regresi is not None:
        metrik[slot].metric("R² model", nr.num(analisis.regresi.model.rsquared, 3))
        slot += 1
    if analisis.logistik is not None:
        metrik[slot].metric("AUC", nr.num(analisis.logistik.auc, 3))
        slot += 1
    if analisis.klaster is not None:
        metrik[slot].metric("Segmen", analisis.klaster.n_clusters)
        slot += 1
    if analisis.diskriminan is not None and slot < 4:
        diskriminan = analisis.diskriminan
        punya_cv = bool(pd.notna(diskriminan.cv_accuracy))
        metrik[slot].metric(
            "Akurasi CV" if punya_cv else "Akurasi latih",
            nr.pct((diskriminan.cv_accuracy if punya_cv else diskriminan.accuracy) * 100),
            help=None if punya_cv else "Validasi silang gagal: ada kelompok beranggota < 2.",
        )
        slot += 1
    if analisis.vif is not None and not analisis.vif.empty and slot < 4:
        metrik[slot].metric("VIF maks", nr.num(float(analisis.vif["VIF"].max())))

    if laporan.pendorong:
        st.subheader("Kontribusi fitur")
        batang_pendorong(laporan)

    st.subheader("Temuan teknis")
    for temuan in laporan.temuan:
        with st.expander(f"**{temuan.judul}** — {temuan.metode}", expanded=False):
            st.write(temuan.profesional)

    asumsi = nr.tabel_asumsi(analisis)
    if not asumsi.empty:
        st.subheader("Ringkasan pemeriksaan asumsi")
        st.dataframe(asumsi, width="stretch", hide_index=True)

    st.subheader("Tindak lanjut yang disarankan")
    daftar_bernomor([(r.judul, r.alasan, r.prioritas) for r in laporan.rekomendasi])

    st.subheader("Risiko dan batas pemakaian")
    daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

if laporan.dilewati:
    with st.expander(f"{len(laporan.dilewati)} analisis tidak dapat dijalankan"):
        for catatan in laporan.dilewati:
            st.markdown(f"- {catatan}")

# --------------------------------------------------------------------------- #
# Unduhan
# --------------------------------------------------------------------------- #

st.subheader("Unduh laporan")
st.caption(
    "Laporan HTML dapat dibuka di peramban mana pun dan dicetak menjadi PDF; berkas "
    "Markdown cocok untuk disunting lebih lanjut."
)
unduh1, unduh2, unduh3 = st.columns(3)
unduh1.download_button(
    f"Laporan HTML — {nr.AUDIENCE_LABELS[pembaca]}",
    laporan_html(laporan, pembaca).encode("utf-8"),
    file_name=f"kesimpulan_{pembaca}.html",
    mime="text/html",
    width="stretch",
)
unduh2.download_button(
    f"Ringkasan Markdown — {nr.AUDIENCE_LABELS[pembaca]}",
    laporan.markdown(pembaca).encode("utf-8"),
    file_name=f"kesimpulan_{pembaca}.md",
    mime="text/markdown",
    width="stretch",
)
unduh3.download_button(
    "Laporan HTML — ketiga pembaca sekaligus",
    laporan_html_semua(laporan).encode("utf-8"),
    file_name="kesimpulan_lengkap.html",
    mime="text/html",
    width="stretch",
    type="primary",
)
