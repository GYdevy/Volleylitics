@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-windows\Scripts\pythonw.exe" (
    echo First-time setup is required.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
    if errorlevel 1 (
        echo.
        echo Setup failed. Review the message above, then try again.
        pause
        exit /b 1
    )
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows_gui.ps1"
if errorlevel 1 (
    echo.
    echo Volleylitics could not open. Startup details:
    if exist "%~dp0output\gui-startup.log" type "%~dp0output\gui-startup.log"
    echo.
    pause
    exit /b 1
)
endlocal
