"""Pembuat laporan HTML mandiri dari hasil :class:`~lentera_mva.narrative.Laporan`.

Berkas yang dihasilkan berdiri sendiri (tanpa aset eksternal), mengikuti tema
terang/gelap peramban pembaca, dan siap dicetak menjadi PDF.
"""

from __future__ import annotations

from html import escape

import pandas as pd

from lentera_mva.narrative import AUDIENCE_LABELS, AUDIENCES, Laporan

_CSS = """
:root{
  color-scheme: light;
  --paper:#eef1f6; --sheet:#ffffff; --sheet-2:#f6f8fc;
  --ink:#131a2b; --ink-2:#3d4860; --muted:#6f7a91;
  --rule:#d5dce7; --accent:#26356b; --accent-2:#3b4ea0; --accent-wash:#eaedf6;
  --good:#1b6f4a; --good-wash:#e4f0ea;
  --warn:#96690b; --warn-wash:#f8f1e0;
  --crit:#9c3327; --crit-wash:#f7e7e4;
  --shadow:0 1px 3px rgba(19,26,43,.06), 0 18px 44px -30px rgba(19,26,43,.4);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --paper:#090e18; --sheet:#141b2c; --sheet-2:#1a2235;
    --ink:#e9edf6; --ink-2:#b6c0d3; --muted:#8b96ac;
    --rule:#2b3550; --accent:#93a6ea; --accent-2:#6b81d6; --accent-wash:#1d2740;
    --good:#5cbb8c; --good-wash:#173026;
    --warn:#d8a53f; --warn-wash:#2e2513;
    --crit:#e08074; --crit-wash:#341c19;
    --shadow:0 1px 3px rgba(0,0,0,.5), 0 18px 44px -30px rgba(0,0,0,.9);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --paper:#090e18; --sheet:#141b2c; --sheet-2:#1a2235;
  --ink:#e9edf6; --ink-2:#b6c0d3; --muted:#8b96ac;
  --rule:#2b3550; --accent:#93a6ea; --accent-2:#6b81d6; --accent-wash:#1d2740;
  --good:#5cbb8c; --good-wash:#173026;
  --warn:#d8a53f; --warn-wash:#2e2513;
  --crit:#e08074; --crit-wash:#341c19;
  --shadow:0 1px 3px rgba(0,0,0,.5), 0 18px 44px -30px rgba(0,0,0,.9);
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:Georgia,"Times New Roman",serif;font-size:16px;line-height:1.7}
.sheet{max-width:860px;margin:0 auto;background:var(--sheet);box-shadow:var(--shadow);
  padding:0 clamp(20px,5vw,60px) 60px}
.masthead{padding:32px 0 20px;border-bottom:2.5px solid var(--ink)}
.kicker{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;
  letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin-bottom:16px}
h1{font-size:clamp(24px,4vw,33px);line-height:1.22;margin:0 0 10px;font-weight:600;
  letter-spacing:-.015em;max-width:24ch}
.sub{font-size:17px;color:var(--ink-2);margin:0 0 20px;font-style:italic;max-width:62ch}
.specs{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px 24px;
  font-family:system-ui,-apple-system,sans-serif}
.specs div{border-top:1px solid var(--rule);padding-top:8px}
.specs dt{font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin-bottom:3px}
.specs dd{margin:0;font-size:12.5px;font-family:ui-monospace,Menlo,monospace;color:var(--ink);
  overflow-wrap:anywhere}
.headline{background:var(--sheet-2);border-left:3px solid var(--accent);padding:22px 26px;margin:28px 0}
.headline p.stmt{margin:0 0 10px;font-size:23px;line-height:1.35;font-weight:600;max-width:34ch}
.headline p.note{margin:0;font-size:16px;color:var(--ink-2);max-width:66ch}
.lamps{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:10px;margin:22px 0}
.lamp{border:1px solid var(--rule);border-radius:9px;padding:13px 15px;background:var(--sheet-2);
  font-family:system-ui,-apple-system,sans-serif}
.lamp .lb{font-size:13px;font-weight:600;display:flex;gap:8px;align-items:center}
.dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto}
.dot.baik{background:var(--good)} .dot.perhatian{background:var(--warn)} .dot.kritis{background:var(--crit)}
.lamp .lv{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--muted);margin:5px 0 4px}
.lamp .lc{font-size:12.5px;color:var(--ink-2);line-height:1.45}
section{margin-top:38px}
section>h2{font-family:system-ui,-apple-system,sans-serif;font-size:18px;font-weight:600;
  margin:0 0 10px;padding-bottom:9px;border-bottom:1px solid var(--rule)}
h3{font-family:system-ui,-apple-system,sans-serif;font-size:15px;font-weight:600;margin:22px 0 4px}
.meta{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--muted);margin:0 0 8px}
p{max-width:68ch}
.bars{font-family:system-ui,-apple-system,sans-serif;margin-top:6px}
.bar{display:grid;grid-template-columns:22px 1fr;gap:12px;padding:12px 0;border-top:1px solid var(--rule)}
.bar:first-child{border-top:0}
.rank{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--muted);text-align:right}
.bname{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:6px}
.bname .n{font-weight:600;font-size:14.5px}
.bname .v{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;font-weight:600}
.track{height:15px;background:var(--sheet-2);border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.track i{display:block;height:100%;background:var(--accent-2)}
.track i.turun{background:var(--crit)}
.track i.tak{background:var(--muted);opacity:.45}
.bnote{font-size:12.5px;color:var(--muted);margin-top:6px;line-height:1.45}
table{border-collapse:collapse;width:100%;font-family:system-ui,-apple-system,sans-serif;
  font-size:13px;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink);margin-top:6px}
thead th{font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);
  font-weight:600;padding:9px 10px;text-align:left;border-bottom:1px solid var(--ink);white-space:nowrap}
tbody td{padding:8px 10px;border-bottom:1px solid var(--rule);color:var(--ink-2);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.scroll{overflow-x:auto}
.note{font-size:13px;font-style:italic;color:var(--muted);margin:8px 0 0;max-width:72ch}
ol.list{margin:0;padding:0;list-style:none;counter-reset:l}
ol.list li{counter-increment:l;position:relative;padding:13px 0 13px 36px;border-top:1px solid var(--rule);max-width:72ch}
ol.list li:first-child{border-top:0}
ol.list li::before{content:counter(l);position:absolute;left:0;top:15px;font-family:ui-monospace,Menlo,monospace;
  font-size:11.5px;font-weight:600;color:var(--accent);background:var(--accent-wash);
  width:23px;height:23px;border-radius:6px;display:grid;place-items:center}
.rt{font-family:system-ui,-apple-system,sans-serif;font-weight:600;font-size:14.5px;margin-bottom:3px}
.rd{font-size:14px;color:var(--ink-2)}
.chip{display:inline-block;font-family:system-ui,-apple-system,sans-serif;font-size:10.5px;font-weight:600;
  padding:2px 8px;border-radius:999px;margin-left:8px;vertical-align:1px}
.chip.tinggi{background:var(--crit-wash);color:var(--crit)}
.chip.sedang{background:var(--warn-wash);color:var(--warn)}
.chip.rendah{background:var(--accent-wash);color:var(--accent)}
.quote{border:1px solid var(--rule);border-left:3px solid var(--accent-2);background:var(--sheet-2);
  padding:15px 18px;margin-top:14px}
.quote .qh{font-family:system-ui,-apple-system,sans-serif;font-size:10.5px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:8px}
.quote p{margin:0;font-size:15.5px}
footer{max-width:860px;margin:0 auto;padding:18px clamp(20px,5vw,60px) 40px;
  font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);line-height:1.6}
@media print{body{background:#fff}.sheet{box-shadow:none;max-width:none}}
"""


def _tabel_html(df: pd.DataFrame) -> str:
    kepala = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    badan = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in baris) + "</tr>"
        for baris in df.itertuples(index=False)
    )
    return f'<div class="scroll"><table><thead><tr>{kepala}</tr></thead><tbody>{badan}</tbody></table></div>'


def _bagian(laporan: Laporan, pembaca: str) -> str:
    """Susun seluruh bagian isi laporan untuk satu register pembaca."""
    bagian: list[str] = []

    lampu = "".join(
        f'<div class="lamp"><div class="lb"><span class="dot {l.status}"></span>'
        f"{escape(l.label)}</div><div class=\"lv\">{escape(l.nilai)}</div>"
        f'<div class="lc">{escape(l.status_label)} — {escape(l.catatan)}</div></div>'
        for l in laporan.lampu
    )
    bagian.append(f'<section><h2>Status pemeriksaan</h2><div class="lamps">{lampu}</div></section>')

    if laporan.pendorong and pembaca != "akademik":
        baris = []
        for i, p in enumerate(laporan.pendorong, start=1):
            kelas = "tak" if not p.signifikan else ("turun" if p.arah == "turun" else "")
            nilai = f"{p.satuan} {p.nilai:.3f}".replace(".", ",")
            baris.append(
                f'<div class="bar"><div class="rank">{i}</div><div>'
                f'<div class="bname"><span class="n">{escape(p.nama)}</span>'
                f'<span class="v">{escape(nilai)}</span></div>'
                f'<div class="track"><i class="{kelas}" style="width:{max(p.kekuatan, 0.02) * 100:.1f}%"></i></div>'
                f'<div class="bnote">{escape(p.catatan)}</div></div></div>'
            )
        bagian.append(
            f"<section><h2>Peringkat pendorong</h2>"
            f'<p class="meta">Sumber: {escape(laporan.pendorong_sumber)} · '
            "batang abu-abu menandai pengaruh yang belum terbukti signifikan</p>"
            f'<div class="bars">{"".join(baris)}</div></section>'
        )

    temuan = []
    for t in laporan.temuan:
        temuan.append(
            f"<h3>{escape(t.judul)}</h3>"
            f'<p class="meta">Metode: {escape(t.metode)}</p>'
            f"<p>{escape(t.teks(pembaca))}</p>"
        )
    bagian.append(f"<section><h2>Temuan</h2>{''.join(temuan)}</section>")

    if pembaca == "akademik" and laporan.tabel:
        isi = []
        for nomor, (judul, tabel, catatan) in laporan.tabel.items():
            isi.append(
                f"<h3>{escape(nomor)}. {escape(judul)}</h3>"
                + _tabel_html(tabel)
                + (f'<p class="note">Catatan. {escape(catatan)}</p>' if catatan else "")
            )
        bagian.append(f"<section><h2>Tabel hasil</h2>{''.join(isi)}</section>")

    if pembaca == "akademik" and laporan.paragraf:
        kutipan = "".join(
            f'<div class="quote"><div class="qh">{escape(p.bagian)}</div>'
            f"<p>{escape(p.teks)}</p></div>"
            for p in laporan.paragraf
        )
        bagian.append(
            "<section><h2>Kalimat siap salin</h2>"
            "<p>Paragraf berikut disusun mengikuti konvensi pelaporan statistik dan dapat "
            "langsung disalin ke naskah, dengan penyesuaian nama variabel.</p>"
            f"{kutipan}</section>"
        )

    if laporan.rekomendasi:
        butir = "".join(
            f'<li><div class="rt">{escape(r.judul)}'
            f'<span class="chip {r.prioritas}">prioritas {escape(r.prioritas)}</span></div>'
            f'<div class="rd">{escape(r.alasan)}</div></li>'
            for r in laporan.rekomendasi
        )
        bagian.append(f'<section><h2>Rekomendasi tindakan</h2><ol class="list">{butir}</ol></section>')

    if laporan.keterbatasan:
        butir = "".join(f'<li><div class="rd">{escape(k)}</div></li>' for k in laporan.keterbatasan)
        bagian.append(f'<section><h2>Batas kesimpulan</h2><ol class="list">{butir}</ol></section>')

    if pembaca == "akademik" and laporan.rujukan:
        butir = "".join(f'<li><div class="rd">{escape(r)}</div></li>' for r in laporan.rujukan)
        bagian.append(f'<section><h2>Rujukan ambang yang dipakai</h2><ol class="list">{butir}</ol></section>')

    if laporan.dilewati:
        butir = "".join(f'<li><div class="rd">{escape(d)}</div></li>' for d in laporan.dilewati)
        bagian.append(
            f'<section><h2>Analisis yang tidak dapat dijalankan</h2><ol class="list">{butir}</ol></section>'
        )

    return "".join(bagian)


def _kepala(laporan: Laporan, judul: str, keterangan: str) -> str:
    return f"""<header class="masthead">
  <div class="kicker">Lentera MVA · Kesimpulan Analisis Multivariat</div>
  <h1>{escape(judul)}</h1>
  <p class="sub">{escape(keterangan)}</p>
  <dl class="specs">
    <div><dt>Sumber data</dt><dd>{escape(laporan.dataset)}</dd></div>
    <div><dt>Ukuran data</dt><dd>{laporan.n_baris} baris · {laporan.n_kolom} kolom</dd></div>
    <div><dt>Metode dijalankan</dt><dd>{len(laporan.metode_terpakai)} metode</dd></div>
    <div><dt>Tanggal analisis</dt><dd>{escape(laporan.tanggal)}</dd></div>
  </dl>
</header>
<div class="headline">
  <p class="stmt">{escape(laporan.headline)}</p>
  <p class="note">{escape(laporan.subheadline)}</p>
</div>"""


_FOOTER = """<footer>
  Disusun otomatis oleh Lentera MVA dari data yang diunggah pengguna. Kesimpulan statistik
  menunjukkan pola dalam data, bukan bukti sebab-akibat; keputusan akhir tetap memerlukan
  pertimbangan konteks dan keahlian bidang.
</footer>"""


def laporan_html(laporan: Laporan, pembaca: str) -> str:
    """Bangun laporan HTML mandiri untuk satu register pembaca."""
    label = AUDIENCE_LABELS[pembaca]
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kesimpulan Analisis Multivariat — {escape(label)}</title>
<style>{_CSS}</style></head>
<body><article class="sheet">
{_kepala(laporan, label, "Laporan ini disusun otomatis dari data yang dianalisis; seluruh angka dihitung ulang setiap kali analisis dijalankan.")}
{_bagian(laporan, pembaca)}
</article>
{_FOOTER}</body></html>"""


def laporan_html_semua(laporan: Laporan) -> str:
    """Satu berkas berisi ketiga register pembaca dengan tombol pengalih."""
    tombol = "".join(
        f'<button type="button" data-pembaca="{p}" aria-pressed="{str(i == 0).lower()}">'
        f"{escape(AUDIENCE_LABELS[p])}</button>"
        for i, p in enumerate(AUDIENCES)
    )
    panel = "".join(
        f'<div class="panel" data-panel="{p}"{"" if i == 0 else " hidden"}>{_bagian(laporan, p)}</div>'
        for i, p in enumerate(AUDIENCES)
    )
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kesimpulan Analisis Multivariat</title>
<style>{_CSS}
.switch{{display:flex;flex-wrap:wrap;gap:3px;background:var(--sheet-2);border:1px solid var(--rule);
  border-radius:9px;padding:3px;margin:24px 0 4px;font-family:system-ui,-apple-system,sans-serif}}
.switch button{{flex:1 1 auto;font-size:13px;font-weight:600;color:var(--ink-2);background:transparent;
  border:0;border-radius:7px;padding:9px 14px;cursor:pointer}}
.switch button[aria-pressed="true"]{{background:var(--accent);color:var(--sheet)}}
.switch button:focus-visible{{outline:2px solid var(--accent-2);outline-offset:2px}}
[hidden]{{display:none!important}}
@media print{{.switch{{display:none}}.panel[hidden]{{display:block!important}}}}
</style></head>
<body><article class="sheet">
{_kepala(laporan, "Kesimpulan Analisis Multivariat", "Satu hasil analisis, tiga cara membacanya. Pilih sudut pandang pembaca di bawah; seluruh angka dihitung dari data yang dianalisis.")}
<div class="switch" role="group" aria-label="Pilih register pembaca">{tombol}</div>
{panel}
</article>
{_FOOTER}
<script>
(function () {{
  var tombol = document.querySelectorAll(".switch button");
  var panel = document.querySelectorAll(".panel");
  tombol.forEach(function (b) {{
    b.addEventListener("click", function () {{
      tombol.forEach(function (x) {{ x.setAttribute("aria-pressed", String(x === b)); }});
      panel.forEach(function (p) {{ p.hidden = p.dataset.panel !== b.dataset.pembaca; }});
      window.scrollTo({{ top: 0, behavior: "smooth" }});
    }});
  }});
}})();
</script>
</body></html>"""
