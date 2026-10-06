r"""P13 Windows installer portability regression for scripts/make_installer.bat.

Real Windows build succeeded through PyInstaller (BUILD_OK dist\VoiceStudio)
but make_installer.bat failed with "'iscc' is not recognized" because Inno
Setup 6 was installed without ISCC.exe on PATH. The script must discover
ISCC.exe from PATH first and fall back to common Inno Setup 6 locations,
instead of failing when PATH alone does not contain it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / "scripts" / "make_installer.bat"


def _source() -> str:
    return BAT.read_text(encoding="ascii")


def test_a_dist_prerequisite_check_present():
    src = _source()
    assert "if not exist dist\\VoiceStudio" in src
    assert "[FAIL] dist\\VoiceStudio is missing." in src
    assert "Run scripts\\build_windows.bat first." in src


def test_b_path_is_searched_first():
    src = _source()
    path_check = src.index("where iscc")
    fallback_check = src.index("%LOCALAPPDATA%\\Programs\\Inno Setup 6\\ISCC.exe")
    assert path_check < fallback_check
    assert "if %errorlevel%==0" in src
    assert 'set "ISCC=iscc"' in src


def test_c_fallback_locations_covered():
    src = _source()
    for loc in (
        "%LOCALAPPDATA%\\Programs\\Inno Setup 6\\ISCC.exe",
        "%ProgramFiles%\\Inno Setup 6\\ISCC.exe",
        "%ProgramFiles(x86)%\\Inno Setup 6\\ISCC.exe",
    ):
        assert f'if exist "{loc}"' in src, loc


def test_d_clear_fail_without_compiler():
    src = _source()
    assert "ISCC.exe was not found" in src
    assert "exit /b 1" in src
    assert src.count("exit /b 1") >= 3  # dist missing, not found, iscc failure


def test_e_iscc_invoked_with_quoted_path():
    src = _source()
    assert '"%ISCC%" installer\\voice-studio.iss' in src


def test_f_compiler_failure_returns_nonzero():
    src = _source()
    assert 'if errorlevel 1' in src
    assert "exit /b 1" in src


def test_g_success_marker_printed():
    src = _source()
    assert "echo INSTALLER_OK installer\\Output" in src
    assert "Using Inno Setup compiler" in src


def test_no_user_specific_hardcoded_paths():
    src = _source()
    assert "sjm98" not in src
    assert "C:\\Users" not in src


def test_bat_stays_ascii_only():
    data = BAT.read_bytes()
    bad = [b for b in data if b >= 128]
    assert bad == []
