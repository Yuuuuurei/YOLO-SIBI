@echo off
echo ----------------------------------------
echo Mengaktifkan environment dan menjalankan pipeline...
echo ----------------------------------------

:: Aktifkan virtual environment
call ".env\Scripts\activate.bat"

:: Cek apakah folder dataset sudah lengkap (train dan test)
if exist dataset\train (
    if exist dataset\test (
        echo ----------------------------------------
        echo Dataset sudah ada. Siap untuk training!
        echo ----------------------------------------
        goto :end
    )
)

echo ----------------------------------------
echo Dataset belum ditemukan. Memulai proses download dan split...
echo ----------------------------------------

:: Jalankan download dataset
echo Menjalankan download_dataset.py...
python run\download_dataset.py

:: Cek apakah download berhasil (folder sibi_dataset)
if exist run\sibi_dataset (
    echo ----------------------------------------
    echo Menjalankan split_dataset.py...
    echo ----------------------------------------
    python run\split_dataset.py
) else (
    echo Gagal menemukan folder 'sibi_dataset'. Pastikan download berhasil.
    goto :end
)

:: Cek ulang apakah folder dataset sudah dibuat
if exist dataset\train (
    if exist dataset\test (
        echo ----------------------------------------
        echo Dataset berhasil disiapkan! Siap untuk training.
        echo ----------------------------------------
    ) else (
        echo Split dataset gagal. Folder 'test' tidak ditemukan.
    )
) else (
    echo Split dataset gagal. Folder 'train' tidak ditemukan.
)

:end
echo.
echo Semua selesai!
pause
