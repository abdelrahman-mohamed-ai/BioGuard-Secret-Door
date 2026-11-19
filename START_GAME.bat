@echo off
setlocal
cd /d "%~dp0"
echo.
echo BioGuard Secret Door
echo --------------------
echo Using the existing .venv environment.
echo On first launch, create the local Admin PIN in the browser.
echo.
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe was not found.
  echo Run setup_venv.ps1 or open PowerShell in the project root.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "flask_game\app.py"
pause
