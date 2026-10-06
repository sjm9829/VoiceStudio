@echo off
REM Frozen build self smoke (P12.2-21): run from the install folder for a first
REM integrity check.
REM Usage: scripts\smoke_frozen.bat [install folder path]  (default: repo\dist\VoiceStudio)
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
