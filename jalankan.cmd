@echo off
REM Jalankan Lentera MVA di port 8503.
REM
REM Port tidak dikunci di .streamlit/config.toml karena layanan hosting memeriksa
REM port bawaannya sendiri; port ditetapkan di sini agar berlaku lokal saja.
REM Klik dua kali berkas ini, atau jalankan lewat PowerShell: .\jalankan.cmd

setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo Lingkungan virtual .venv belum dibuat.
    echo Jalankan lebih dulu:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

set STREAMLIT_SERVER_PORT=8503
echo Membuka http://localhost:8503
streamlit run app.py --server.port 8503
endlocal
