@echo off
echo ----------------------------------------
echo Mengaktifkan environment dan menjalankan pipeline...
echo ----------------------------------------

:: Aktifkan virtual environment
call ".env\Scripts\activate.bat"

echo ----------------------------------------
echo Menjalankan training model...
echo ----------------------------------------
python classifier/train.py

echo.
echo Training selesai!
pause
