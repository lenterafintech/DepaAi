"""Akun: masuk/daftar dan paket langganan digabung jadi satu tab.

Sebelumnya dua halaman terpisah (Masuk/Daftar, Akun & Langganan) yang saling
menautkan lewat ``page_link``. Digabung karena keduanya sama-sama tentang satu
hal — identitas dan paket pengguna — dan navigasi bertautan antar-halaman tidak
lagi berlaku begitu aplikasi menjadi satu halaman bertab.
"""

from __future__ import annotations

import streamlit as st

from nalardata import langganan, pengguna as pg, ui


def render() -> None:
    akun = ui.pengguna_aktif()
    if akun is None:
        _render_masuk()
    else:
        _render_ringkasan_akun(akun)

    st.divider()
    _render_paket(akun)


def _render_ringkasan_akun(akun: pg.Pengguna) -> None:
    paket = ui.paket_aktif()
    st.success(f"Anda masuk sebagai **{akun.nama}** ({akun.surel}).")
    st.caption(f"Paket berlaku: {paket.nama} — {akun.alasan_paket()}")
    if akun.dalam_uji_coba():
        sisa = akun.sisa_uji_coba()
        st.info(
            f"Masa uji coba berjalan: seluruh fitur terbuka selama "
            f"{sisa.days} hari {sisa.seconds // 3600} jam lagi. Setelah itu akun "
            f"berlanjut pada paket **{langganan.ambil_paket(akun.paket).nama}**.",
            icon=":material/schedule:",
        )
    if st.button("Keluar dari akun"):
        ui.keluar()
        st.rerun()


def _render_masuk() -> None:
    st.warning(
        "Anda belum masuk, sehingga paket hanya berlaku untuk sesi ini dan hilang saat "
        "aplikasi ditutup.",
        icon=":material/person_off:",
    )
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
                "Saya memahami bahwa data yang saya unggah diproses untuk keperluan "
                "analisis saya sendiri."
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


def _render_paket(akun: pg.Pengguna | None) -> None:
    paket = ui.paket_aktif()
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
.mva-paket .p {{position: relative; border: 1px solid {ui.WARNA['garis']}; border-radius: 12px;
  padding: 18px; background: {ui.WARNA['kertas']}; transition: transform .15s ease}}
.mva-paket .p:hover {{transform: translateY(-2px)}}
.mva-paket .p.aktif {{border-color: {ui.WARNA['aksen']}; border-width: 2px;
  background: {ui.WARNA['aksenSamar']}}}
.mva-paket .p.populer {{border-color: {ui.WARNA['aksen']}; border-width: 2px}}
.mva-paket .pop {{position: absolute; top: -11px; right: 16px; font-size: .62rem;
  font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #fff;
  background: {ui.WARNA['aksen']}; border-radius: 999px; padding: 3px 10px}}
.mva-paket .nm {{font-weight: 700; font-size: 1.05rem; color: {ui.WARNA['tinta']}}}
.mva-paket .hg {{font-family: ui-monospace, Menlo, monospace; font-size: .95rem;
  color: {ui.WARNA['aksen2']}; margin: .3rem 0 .5rem; font-weight: 600}}
.mva-paket .rk {{font-size: .84rem; line-height: 1.55; color: {ui.WARNA['tinta2']};
  margin-bottom: .6rem}}
.mva-paket .bt {{font-size: .78rem; color: {ui.WARNA['redup']};
  font-family: ui-monospace, Menlo, monospace}}
.mva-paket .tag {{display: inline-block; font-size: .64rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; color: {ui.WARNA['aksen']};
  background: {ui.WARNA['kertas']}; border: 1px solid {ui.WARNA['aksen']}; border-radius: 999px;
  padding: 1px 8px; margin-left: 8px; vertical-align: 2px}}
</style>
"""
    )

    kartu = []
    for p in langganan.urut_tingkatan():
        aktif = p.kode == paket.kode
        populer = p.kode == "profesional"
        kelas = (" aktif" if aktif else "") + (" populer" if populer and not aktif else "")
        lencana = '<span class="pop">Populer</span>' if populer else ""
        tag = '<span class="tag">Paket Anda</span>' if aktif else ""
        harga = langganan.harga_tampil(p)
        batas = f"{p.maks_baris:,} baris · {p.maks_variabel} kolom".replace(",", ".")
        kartu.append(
            f'<div class="p{kelas}">{lencana}<div class="nm">{p.nama}{tag}</div>'
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
        "dengan pesan yang menyebutkan paket mana yang mencukupi. Kata sandi disimpan "
        "dalam bentuk terenkripsi satu arah (PBKDF2-SHA256), namun pemulihan kata sandi "
        "dan verifikasi surel belum tersedia."
    )
