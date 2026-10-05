@echo off
REM 보이스 스튜디오 Windows 빌드 (PyInstaller onedir)
setlocal
cd /d "%~dp0.."
python -m venv .venv || goto :err
call .venv\Scripts\activate
pip install -e ".[ui,audio,transcribe,dev]" || goto :err
pip install pyinstaller || goto :err
pyinstaller --noconfirm --clean --onedir --name VoiceStudio ^
  --windowed ^
  --add-data "third_party;third_party" ^
  src\voice_studio\main.py || goto :err
echo BUILD_OK dist\VoiceStudio
goto :eof
:err
echo BUILD_FAILED
exit /b 1
