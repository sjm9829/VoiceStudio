import sys, json, re
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
        for py in (src / sub).rglob("*.py") if (src / sub).is_dir() else [src / sub]:
            text = py.read_text(encoding="utf-8")
            for f in forbidden:
                if re.search(rf"^\s*(import|from)\s+{f}", text, re.M):
                    offenders.append(str(py.relative_to(src)))
    assert offenders == [], offenders

def test_no_family_specific_terms_in_ui_and_docs():
    """특정 사용자/가족 표현(아빠/아버지/father) 0건 검사. docs는 용어 정책 언급 제외."""
    root = Path(__file__).resolve().parents[1]
    banned = ("아빠", "아버지", "father")
    hits = []
    for base in (root / "src", root / "README.md"):
        files = base.rglob("*.py") if base.is_dir() else [base]
        for p in files:
            text = p.read_text(encoding="utf-8")
            for w in banned:
                if w in text.lower() if w == "father" else w in text:
                    hits.append(f"{p.relative_to(root)}:{w}")
    assert hits == [], hits
