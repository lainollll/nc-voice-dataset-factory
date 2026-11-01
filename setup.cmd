@echo off
setlocal
title NC Voice Dataset Factory - Setup
pushd "%~dp0"
if errorlevel 1 (
    echo Unable to open the application folder.
    pause
    exit /b 1
)
echo NC Voice Dataset Factory - dependency setup
echo Missing Python 3.11, Git and FFmpeg will be installed using WinGet.
echo App packages will be installed in this folder's .venv.
echo Internet access and several GB of free space are required.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1" %*
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
echo.
if "%SETUP_EXIT_CODE%"=="0" (
    echo Setup finished. Double-click Start_NC_Voice_Dataset_Factory.cmd to open the app.
) else (
    echo Setup failed with error %SETUP_EXIT_CODE%. Review the message above, then run setup.cmd again.
)
pause
popd
exit /b %SETUP_EXIT_CODE%
