@echo off
REM Create the Inno Setup installer (ISCC discovery: PATH first, then known locations)
setlocal
cd /d "%~dp0.."

if not exist dist\VoiceStudio (
  echo [FAIL] dist\VoiceStudio is missing.
  echo Run scripts\build_windows.bat first.
  exit /b 1
)

REM 1. Prefer ISCC from PATH.
where iscc >nul 2>nul
if %errorlevel%==0 (
  set "ISCC=iscc"
  goto :run_iscc
)

REM 2. Check common Inno Setup 6 installation locations.
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
  set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
  goto :run_iscc
)

if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" (
  set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
  goto :run_iscc
)

if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
  set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
  goto :run_iscc
)

echo [FAIL] Inno Setup 6 compiler ISCC.exe was not found.
echo Install Inno Setup 6 and try again.
exit /b 1

:run_iscc
echo Using Inno Setup compiler:
echo %ISCC%

"%ISCC%" installer\voice-studio.iss
if errorlevel 1 (
  echo [FAIL] Inno Setup compilation failed.
  exit /b 1
)

echo.
echo INSTALLER_OK installer\Output
exit /b 0
