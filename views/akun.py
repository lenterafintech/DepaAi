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
akun = ui.pengguna_aktif()

if akun is None:
    st.warning(
        "Anda belum masuk, sehingga paket hanya berlaku untuk sesi ini dan hilang saat "
        "halaman ditutup. Buka halaman *Masuk / Daftar* untuk membuat akun.",
        icon=":material/person_off:",
    )
else:
    st.caption(f"Akun: **{akun.nama}** ({akun.surel}) · {akun.alasan_paket()}")
    if akun.dalam_uji_coba():
        sisa = akun.sisa_uji_coba()
        st.success(
            f"Masa uji coba berjalan: seluruh fitur terbuka selama "
            f"{sisa.days} hari {sisa.seconds // 3600} jam lagi. Setelah itu akun "
            f"berlanjut pada paket **{langganan.ambil_paket(akun.paket).nama}**.",
            icon=":material/schedule:",
        )

st.info(
    "**Masa perkenalan.** Pembayaran belum terpasang, sehingga seluruh paket dapat "
    "diaktifkan tanpa biaya. Ketika penagihan sudah aktif, perubahan paket hanya "
    "terjadi setelah pembayaran berhasil.",
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
for p in langganan.urut_tingkatan():
    aktif = " aktif" if p.kode == paket.kode else ""
    tag = '<span class="tag">Paket Anda</span>' if p.kode == paket.kode else ""
    harga = langganan.harga_tampil(p)
    batas = f"{p.maks_baris:,} baris · {p.maks_variabel} kolom".replace(",", ".")
    kartu.append(
        f'<div class="p{aktif}"><div class="nm">{p.nama}{tag}</div>'
        f'<div class="hg">{harga}</div><div class="rk">{p.ringkas}</div>'
        f'<div class="bt">{batas} · {len(p.fitur)}/{len(langganan.FITUR)} fitur</div></div>'
    )
st.html(f'<div class="mva-paket">{"".join(kartu)}</div>')

st.subheader("Ganti paket")
kode_urut = [p.kode for p in langganan.urut_tingkatan()]
pilihan = st.radio(
    "Paket yang dipilih",
    kode_urut,
    format_func=lambda k: f"{langganan.PAKET[k].nama} — {langganan.harga_tampil(langganan.PAKET[k])}",
    index=kode_urut.index(paket.kode) if paket.kode in kode_urut else 0,
    key="pilih_paket",
)
if pilihan == "institusi":
    st.caption(
        "Paket institusi memerlukan kesepakatan tersendiri. Selama masa perkenalan "
        "Anda dapat mengaktifkannya untuk mencoba, namun pengelola akan menghubungi "
        "Anda untuk verifikasi."
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
