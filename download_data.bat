@echo off
chcp 65001 >nul 2>&1
title ancserFX - Unzip Test Data

echo.
echo  =============================================
echo   ancserFX - Unzip Test Data
echo  =============================================
echo.

:: Check test-data.zip exists
if not exist "test-data.zip" (
    echo  [ERROR] test-data.zip not found.
    echo          Make sure you are in the ancserFX root directory.
    pause
    exit /b 1
)

:: Check if data/parquet already has content
if exist "data\parquet\es\5min\data.parquet" (
    echo  [SKIP] data\parquet\ already has data.
    echo         Delete data\parquet\ first if you want to re-extract.
    pause
    exit /b 0
)

:: Unzip using Python
echo  Extracting test-data.zip to data\parquet\ ...
python -c "import zipfile, pathlib; pathlib.Path('data/parquet').mkdir(parents=True, exist_ok=True); zipfile.ZipFile('test-data.zip').extractall('data/parquet'); print('  [OK] Done.')"

if errorlevel 1 (
    echo.
    echo  [ERROR] Extract failed.
    pause
    exit /b 1
)

echo.
echo  =============================================
echo   Data ready! Start the dashboard:
echo     start_dashboard.bat
echo   Or: streamlit run dashboard.py
echo  =============================================
echo.
pause
