@echo off
cd /d "%~dp0\..\.."
python scripts\download_all.py --profile core --root data --max-gb 5
