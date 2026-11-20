@echo off
setlocal
cd /d "%~dp0"
python comparison_runners\revision_followup.py train-shaping --device auto
if errorlevel 1 exit /b 1
python comparison_runners\revision_followup.py summarize-shaping
if errorlevel 1 exit /b 1
python comparison_runners\revision_followup.py evaluate-shaping --device auto
if errorlevel 1 exit /b 1
python comparison_runners\revision_followup.py evaluate-composite-generated
if errorlevel 1 exit /b 1
python comparison_runners\revision_followup.py evaluate-composite-industrial
if errorlevel 1 exit /b 1
endlocal
