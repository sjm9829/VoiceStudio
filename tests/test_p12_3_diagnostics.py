"""P12.3-07: frozen diagnostics dispatch regression."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_dev_diagnostics_command_uses_main_entrypoint():
    from voice_studio.workers.launcher import diagnostics_command
    program, args = diagnostics_command()
    assert program == sys.executable
    assert args[:4] == ["-m", "voice_studio.main", "--diagnostics", "--json"]
    assert args[-1] == "--log"
    assert "-m voice_studio.diagnostics" not in " ".join(args)


def test_frozen_diagnostics_command_uses_exe_flag(monkeypatch):
    import sys as _sys
    from voice_studio.workers import launcher
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    program, args = launcher.diagnostics_command()
    assert program == _sys.executable
    assert args == ["--diagnostics", "--json", "--log"]
    assert "-m" not in args  # frozen에서 -m 방식 금지


def test_main_has_diagnostics_mode_before_qt():
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "main.py").read_text(encoding="utf-8")
    assert 'if "--diagnostics" in args:' in src
    assert src.index('if "--diagnostics" in args:') < src.index("QApplication(args)")
    assert "diagnostics_main(args)" in src


def test_diagnostics_main_accepts_argv():
    import voice_studio.diagnostics as dg
    import inspect
    sig = inspect.signature(dg.main)
    assert "argv" in sig.parameters
    assert sig.parameters["argv"].default is None


def test_settings_dialog_uses_launcher_helper():
    src = (Path(__file__).resolve().parents[1] / "src" / "voice_studio" / "ui" /
           "settings_dialog.py").read_text(encoding="utf-8")
    assert "diagnostics_command" in src
    assert '"-m", "voice_studio.diagnostics"' not in src
