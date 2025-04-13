@echo off
echo ----------------------------------------
echo Mengecek dan membuat virtual environment...
echo ----------------------------------------

:: Cek apakah folder .env ada
if not exist .env (
    echo Membuat virtual environment...
    python -m venv .env
)

:: Aktifkan virtual environment
call ".env\Scripts\activate.bat"

:: Pastikan pip tersedia
where pip >nul 2>nul
if errorlevel 1 (
    echo pip tidak ditemukan. Pastikan Python terinstall.
    exit /b
)

:: Install requirements.txt
echo ----------------------------------------
echo Menginstall requirements dari requirements.txt...
echo ----------------------------------------
pip install -r requirements.txt

echo.
echo Virtual environment dan dependencies telah siap.
pause
