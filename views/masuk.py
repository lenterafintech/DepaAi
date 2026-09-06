"""Halaman masuk dan pendaftaran akun."""

from __future__ import annotations

import streamlit as st

from lentera_mva import langganan, pengguna as pg, ui

ui.page_setup(
    "Masuk ke Lentera MVA",
    "Akun",
    "Masuk untuk menyimpan paket dan melanjutkan analisis Anda. Akun baru langsung "
    f"mendapat {pg.HARI_UJI_COBA} hari uji coba dengan seluruh fitur terbuka.",
)

akun = ui.pengguna_aktif()
if akun is not None:
    st.success(f"Anda sudah masuk sebagai **{akun.nama}** ({akun.surel}).")
    paket = ui.paket_aktif()
    st.caption(f"Paket berlaku: {paket.nama} — {akun.alasan_paket()}")
    kiri, kanan = st.columns(2)
    if kiri.button("Keluar dari akun", width="stretch"):
        ui.keluar()
        st.rerun()
    kanan.page_link("views/akun.py", label="Kelola paket", width="stretch")
    st.stop()

tab_masuk, tab_daftar = st.tabs(["Masuk", "Daftar akun baru"])

with tab_masuk:
    with st.form("form_masuk"):
        surel = st.text_input("Surel", placeholder="nama@contoh.com")
        sandi = st.text_input("Kata sandi", type="password")
        kirim = st.form_submit_button("Masuk", type="primary", width="stretch")
    if kirim:
        try:
            terdaftar = pg.masuk(surel, sandi)
        except pg.GalatPengguna as exc:
            st.error(str(exc))
        else:
            ui.set_pengguna(terdaftar)
            st.rerun()

with tab_daftar:
    st.caption(
        f"Seluruh fitur terbuka selama {pg.HARI_UJI_COBA} hari sejak pendaftaran. "
        "Setelah masa itu berakhir, akun berlanjut pada paket yang Anda pilih."
    )
    with st.form("form_daftar"):
        nama = st.text_input("Nama lengkap")
        surel_baru = st.text_input("Surel", key="daftar_surel")
        kol1, kol2 = st.columns(2)
        sandi_baru = kol1.text_input("Kata sandi", type="password", key="daftar_sandi")
        ulang = kol2.text_input("Ulangi kata sandi", type="password")
        pilihan_paket = st.selectbox(
            "Paket setelah masa uji coba",
            [p.kode for p in langganan.urut_tingkatan()],
            format_func=lambda k: f"{langganan.PAKET[k].nama} — {langganan.harga_tampil(langganan.PAKET[k])}",
        )
        institusi = ""
        if pilihan_paket == "institusi":
            institusi = st.text_input(
                "Nama institusi",
                help="Paket institusi diverifikasi lebih dulu oleh pengelola.",
            )
        setuju = st.checkbox(
            "Saya memahami bahwa data yang saya unggah diproses untuk keperluan analisis "
            "saya sendiri."
        )
        daftar = st.form_submit_button("Buat akun", type="primary", width="stretch")

    if daftar:
        if sandi_baru != ulang:
            st.error("Kedua kata sandi tidak sama.")
        elif not setuju:
            st.error("Centang pernyataan persetujuan untuk melanjutkan.")
        elif pilihan_paket == "institusi" and not institusi.strip():
            st.error("Sebutkan nama institusi Anda.")
        else:
            try:
                baru = pg.daftar(
                    surel_baru,
                    nama,
                    sandi_baru,
                    paket=pilihan_paket,
                    institusi=institusi.strip() or None,
                )
            except pg.GalatPengguna as exc:
                st.error(str(exc))
            else:
                ui.set_pengguna(baru)
                st.rerun()

st.divider()
st.subheader("Paket yang tersedia")
st.html(
    f"""
<style>
.mva-tingkat {{display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
  gap: 12px; margin-top: .4rem}}
.mva-tingkat .t {{border: 1px solid {ui.WARNA['garis']}; border-radius: 10px;
  padding: 14px 16px; background: #fff}}
.mva-tingkat .nm {{font-weight: 650; font-size: .95rem; color: {ui.WARNA['tinta']}}}
.mva-tingkat .hg {{font-family: ui-monospace, Menlo, monospace; font-size: .82rem;
  color: {ui.WARNA['aksen2']}; font-weight: 600; margin: .25rem 0 .45rem}}
.mva-tingkat .rk {{font-size: .82rem; line-height: 1.55; color: {ui.WARNA['tinta2']}}}
</style>
"""
)
st.html(
    '<div class="mva-tingkat">'
    + "".join(
        f'<div class="t"><div class="nm">{p.nama}</div>'
        f'<div class="hg">{langganan.harga_tampil(p)}</div>'
        f'<div class="rk">{p.ringkas}</div></div>'
        for p in langganan.urut_tingkatan()
    )
    + "</div>"
)
st.caption(
    "Selama masa perkenalan seluruh paket dapat diaktifkan tanpa pembayaran. "
    "Kata sandi disimpan dalam bentuk terenkripsi satu arah (PBKDF2-SHA256), namun "
    "pemulihan kata sandi dan verifikasi surel belum tersedia — simpan kata sandi Anda "
    "baik-baik."
)
