import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_studio.infra.settings_repository import SettingsRepository

def test_defaults_and_roundtrip(tmp_path):
    repo = SettingsRepository(tmp_path / "s.json")
    data = repo.load()
    assert data["mp3_bitrate_kbps"] == 192
    assert data["release_gpu_after_use"] is True
    data["mp3_output_dir"] = str(tmp_path / "내 폴더")
    repo.save(data)
    assert repo.load()["mp3_output_dir"] == str(tmp_path / "내 폴더")

def test_corrupted_settings_recover_to_defaults(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{broken", encoding="utf-8")
    assert SettingsRepository(p).load()["mp3_bitrate_kbps"] == 192

def test_main_process_never_imports_qwen_or_torch():
    """메인 프로세스 구조 검증: app_context/ui는 qwen_tts/torch/faster_whisper를 import하지 않는다."""
    src = Path(__file__).resolve().parents[1] / "src" / "voice_studio"
    forbidden = ("qwen_tts", "torch", "faster_whisper")
    offenders = []
    for sub in ("app_context.py", "main.py", "ui"):
        target = src / sub
        files = target.rglob("*.py") if target.is_dir() else [target]
        for py in files:
            text = py.read_text(encoding="utf-8")
            for fw in forbidden:
                if fw in text:
                    offenders.append(f"{py.name}:{fw}")
    assert offenders == [], offenders
