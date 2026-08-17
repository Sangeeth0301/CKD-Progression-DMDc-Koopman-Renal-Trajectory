@echo off
title Launching Deep DMDc Clinical Decision Dashboard
echo ======================================================================
echo Launching Deep DMDc CKD Progression Decision Support System...
echo ======================================================================
cd /d "%~dp0"
python -m streamlit run dashboard/app.py
pause
