@echo off
cd /d "%~dp0\..\.."
python scripts\dce_real_validation.py ponte --data-root data --quick || exit /b 1
python scripts\dce_real_validation.py grideye --data-root data --quick || exit /b 1
