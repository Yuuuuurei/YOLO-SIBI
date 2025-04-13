@echo off
echo ----------------------------------------
echo Mengaktifkan environment dan menjalankan pipeline...
echo ----------------------------------------

:: Aktifkan virtual environment
call ".env\Scripts\activate.bat"

echo ----------------------------------------
echo Menjalankan main.py...
echo ----------------------------------------
python main.py

echo.
echo Selesai menjalankan script!
pause
