@echo off
REM Create the Inno Setup installer (iscc required)
setlocal
cd /d "%~dp0.."
if not exist dist\VoiceStudio (
  echo dist\VoiceStudio is missing. Run build_windows.bat first.
  exit /b 1
)
iscc installer\voice-studio.iss || exit /b 1
echo INSTALLER_OK installer\Output
