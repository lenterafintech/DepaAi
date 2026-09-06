#!/usr/bin/env bash
# Jalankan Lentera MVA di port 8503.
#
# Port tidak dikunci di .streamlit/config.toml karena layanan hosting memeriksa
# port bawaannya sendiri; port ditetapkan di sini agar berlaku lokal saja.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "Lingkungan virtual .venv belum dibuat." >&2
    echo "Jalankan lebih dulu: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

echo "Membuka http://localhost:8503"
exec streamlit run app.py --server.port "${PORT:-8503}"
