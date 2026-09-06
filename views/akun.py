"""Halaman akun: paket langganan yang berlaku dan batasnya."""

from __future__ import annotations

import streamlit as st

from lentera_mva import langganan, ui

ui.page_setup(
    "Akun & Langganan",
    "Akun",
    "Paket menentukan metode yang terbuka dan ukuran data yang dapat dianalisis.",
)
ui.sidebar_info()

paket = ui.paket_aktif()

st.info(
    "**Mode uji coba.** Pembayaran belum terpasang, jadi paket dapat diganti bebas di "
    "halaman ini untuk mencoba batasannya. Ketika penagihan sudah aktif, perubahan "
    "paket hanya terjadi setelah pembayaran berhasil.",
    icon=":material/science:",
)

st.html(
    f"""
<style>
.mva-paket {{display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px; margin: .4rem 0 1.2rem}}
.mva-paket .p {{border: 1px solid {ui.WARNA['garis']}; border-radius: 10px; padding: 16px 18px;
  background: #fff}}
.mva-paket .p.aktif {{border-color: {ui.WARNA['aksen']}; border-width: 2px;
  background: {ui.WARNA['aksenSamar']}}}
.mva-paket .nm {{font-weight: 700; font-size: 1.05rem; color: {ui.WARNA['tinta']}}}
.mva-paket .hg {{font-family: ui-monospace, Menlo, monospace; font-size: .95rem;
  color: {ui.WARNA['aksen2']}; margin: .3rem 0 .5rem; font-weight: 600}}
.mva-paket .rk {{font-size: .84rem; line-height: 1.55; color: {ui.WARNA['tinta2']};
  margin-bottom: .6rem}}
.mva-paket .bt {{font-size: .78rem; color: {ui.WARNA['redup']};
  font-family: ui-monospace, Menlo, monospace}}
.mva-paket .tag {{display: inline-block; font-size: .64rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; color: {ui.WARNA['aksen']};
  background: #fff; border: 1px solid {ui.WARNA['aksen']}; border-radius: 999px;
  padding: 1px 8px; margin-left: 8px; vertical-align: 2px}}
</style>
"""
)

kartu = []
for p in sorted(langganan.PAKET.values(), key=lambda x: x.harga_bulanan):
    aktif = " aktif" if p.kode == paket.kode else ""
    tag = '<span class="tag">Paket Anda</span>' if p.kode == paket.kode else ""
    harga = "Gratis" if not p.harga_bulanan else f"Rp {p.harga_bulanan:,}/bulan".replace(",", ".")
    batas = f"{p.maks_baris:,} baris · {p.maks_variabel} kolom".replace(",", ".")
    kartu.append(
        f'<div class="p{aktif}"><div class="nm">{p.nama}{tag}</div>'
        f'<div class="hg">{harga}</div><div class="rk">{p.ringkas}</div>'
        f'<div class="bt">{batas} · {len(p.fitur)}/{len(langganan.FITUR)} fitur</div></div>'
    )
st.html(f'<div class="mva-paket">{"".join(kartu)}</div>')

st.subheader("Ganti paket")
pilihan = st.radio(
    "Paket yang dicoba",
    list(langganan.PAKET),
    format_func=lambda k: langganan.PAKET[k].nama,
    index=list(langganan.PAKET).index(paket.kode),
    horizontal=True,
    key="pilih_paket",
)
if pilihan != paket.kode and st.button("Terapkan paket ini", type="primary"):
    ui.set_paket(pilihan)
    st.rerun()

st.subheader("Rincian fitur")
terbuka = [f"**{nama}**" for kode, nama in langganan.FITUR.items() if paket.punya(kode)]
terkunci = [
    (nama, langganan.paket_terkecil_dengan(kode))
    for kode, nama in langganan.FITUR.items()
    if not paket.punya(kode)
]

kiri, kanan = st.columns(2)
with kiri:
    st.markdown("**Terbuka pada paket ini**")
    for nama in terbuka:
        st.markdown(f"- {nama}")
with kanan:
    st.markdown("**Belum terbuka**")
    if not terkunci:
        st.markdown("- Seluruh fitur sudah terbuka.")
    for nama, saran in terkunci:
        paket_saran = langganan.PAKET.get(saran or "")
        keterangan = f" — tersedia pada {paket_saran.nama}" if paket_saran else ""
        st.markdown(f"- {nama}{keterangan}")

st.caption(
    "Batas ukuran data diperiksa saat data dimuat; melampauinya menghentikan analisis "
    "dengan pesan yang menyebutkan paket mana yang mencukupi."
)
