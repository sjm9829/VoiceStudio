@echo off
REM Inno Setup 설치 프로그램 생성 (iscc 필요)
setlocal
cd /d "%~dp0.."
if not exist dist\VoiceStudio (
  echo dist\VoiceStudio 가 없습니다. build_windows.bat 를 먼저 실행하세요.
  exit /b 1
)
iscc installer\voice-studio.iss || exit /b 1
echo INSTALLER_OK installer\Output
