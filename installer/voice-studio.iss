; 보이스 스튜디오 설치 프로그램 (Inno Setup)
#define MyAppName "보이스 스튜디오"
#define MyAppVersion "0.1.1"

[Setup]
AppId={{B6A6C0E2-VOICE-STUDIO-0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\VoiceStudio
DefaultGroupName={#MyAppName}
OutputBaseFilename=VoiceStudio-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes

[Files]
Source: "..\dist\VoiceStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\VoiceStudio.exe"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\VoiceStudio.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "바탕 화면에 바로가기 만들기"; GroupDescription: "추가 작업:"
