@echo off
setlocal
title NC Voice Dataset Factory - Repair setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install_runtime.ps1"
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
echo.
if "%SETUP_EXIT_CODE%"=="0" (
    echo Setup finished. Open NC Voice Dataset Factory from the Start menu.
) else (
    echo Setup failed with error %SETUP_EXIT_CODE%. See the setup-logs folder for details.
)
pause
exit /b %SETUP_EXIT_CODE%
