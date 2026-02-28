@echo off
cd /d "%~dp0"
echo Starting ancserFX Dashboard...
echo.
echo Open browser at: http://localhost:8501
echo.
streamlit run dashboard.py
pause
