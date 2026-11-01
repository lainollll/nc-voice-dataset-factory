@echo off
setlocal
title NC Voice Dataset Factory
pushd "%~dp0"
if errorlevel 1 (
    echo Unable to open the application folder.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo NC Voice Dataset Factory needs setup before its first launch.
    echo.
    echo Double-click setup.cmd to install prerequisites and app dependencies.
    echo.
    echo After setup finishes, double-click this launcher again.
    pause
    popd
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo Application startup or shutdown returned error %APP_EXIT_CODE%.
    echo Review the message above. If dependencies are missing, double-click setup.cmd.
    pause
)
popd
exit /b %APP_EXIT_CODE%
