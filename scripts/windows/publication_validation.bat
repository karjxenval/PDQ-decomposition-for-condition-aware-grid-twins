@echo off
cd /d "%~dp0\..\.."
python scripts\dce_real_validation.py ponte --data-root data --publication || exit /b 1
python scripts\dce_real_validation.py grideye --data-root data --publication || exit /b 1
python scripts\build_publication_evidence.py --results-root results --data-root data --out publication_evidence || exit /b 1
