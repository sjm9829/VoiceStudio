@echo off
REM frozen 빌드 셀프 스모크(P12.2-21): 설치 폴더에서 실행해 설치 무결성을 1차 확인한다.
REM 사용법: scripts\smoke_frozen.bat [설치 폴더 경로]  (기본: repo\dist\VoiceStudio)
setlocal
cd /d "%~dp0.."
if "%~1"=="" (
  set APP_DIR=%CD%\dist\VoiceStudio
) else (
  set APP_DIR=%~1
)
if not exist "%APP_DIR%\VoiceStudio.exe" (
  echo SMOKE_FAILED: %APP_DIR%\VoiceStudio.exe not found
  exit /b 1
)
"%APP_DIR%\VoiceStudio.exe" --smoke-test
if errorlevel 1 (
  echo SMOKE_FAILED
  exit /b 1
)
echo SMOKE_OK
