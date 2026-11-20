@echo off
setlocal
cd /d "%~dp0"
python comparison_runners\composite_priority\retune_revision.py tune --workers 4
if errorlevel 1 exit /b 1
python comparison_runners\composite_priority\retune_revision.py evaluate-generated
if errorlevel 1 exit /b 1
python comparison_runners\composite_priority\retune_revision.py evaluate-industrial
if errorlevel 1 exit /b 1
python comparison_runners\composite_priority\retune_revision.py evaluate-components --device auto
if errorlevel 1 exit /b 1
endlocal
