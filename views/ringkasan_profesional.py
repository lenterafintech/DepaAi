"""Laporan profesional: metrik model, kontribusi fitur, dan tindak lanjut teknis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lentera_mva import kesimpulan_ui as kui
from lentera_mva import narrative as nr
from lentera_mva import ui

analisis, laporan, lengkap = kui.buka_ringkasan(
    "Laporan Profesional",
    "Ditulis untuk analis dan praktisi: kinerja model, keterbatasannya, serta langkah "
    "teknis berikutnya sebelum hasil dipakai untuk keputusan operasional.",
    "ringkasan_profesional",
    "profesional",
)

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
    kui.batang_pendorong(laporan)

st.subheader("Temuan teknis")
if lengkap:
    for temuan in laporan.temuan:
        st.markdown(f"**{temuan.judul}**")
        st.caption(f"Metode: {temuan.metode}")
        st.write(temuan.profesional)
else:
    for temuan in laporan.temuan:
        with st.expander(f"**{temuan.judul}** — {temuan.metode}"):
            st.write(temuan.profesional)

asumsi = nr.tabel_asumsi(analisis)
if not asumsi.empty:
    st.subheader("Ringkasan pemeriksaan asumsi")
    ui.show_table(asumsi, "pemeriksaan_asumsi.csv")

st.subheader("Tindak lanjut yang disarankan")
kui.daftar_bernomor([(r.judul, r.alasan, r.prioritas) for r in laporan.rekomendasi])

st.subheader("Risiko dan batas pemakaian")
kui.daftar_bernomor([("", k, None) for k in laporan.keterbatasan])

if lengkap and laporan.tabel:
    st.subheader("Tabel hasil")
    for nomor, (judul, tabel, catatan) in laporan.tabel.items():
        st.markdown(f"**{nomor}.** {judul}")
        ui.show_table(tabel, f"{nomor.lower().replace(' ', '_')}_profesional.csv")
        if catatan:
            st.caption(f"*Catatan.* {catatan}")

if lengkap:
    kui.analisis_yang_dilewati(laporan)

kui.unduhan(laporan, "profesional", lengkap)
