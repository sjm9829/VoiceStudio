"""P12.3 packaging regression: collect_all 계약, spec 경로, FFmpeg prerequisite."""

import ast, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


class _FakeAnalysis:
    def __init__(self):
        self.datas = []
        self.binaries = []
        self.hiddenimports = ["preexisting"]


def test_collect_all_return_contract_mapping():
    """P12.3-01/22: datas→a.datas, binaries→a.binaries, hiddenimports→a.hiddenimports."""
    import spec_helpers

    def fake_collector(name):
        assert name == "ctranslate2"
        return ["d1", "d2"], ["b1"], ["h1", "preexisting"]

    a = _FakeAnalysis()
    spec_helpers.apply_collect(a, "ctranslate2", collector=fake_collector)
    assert a.datas == ["d1", "d2"]
    assert a.binaries == ["b1"]
    assert a.hiddenimports == ["preexisting", "h1"]


def test_spec_no_longer_unpacks_collect_all_wrongly():
    """기존 버그(libs, bins, datas = collect_all) 패턴이 spec에 남아 있지 않은지."""
    spec = (ROOT / "packaging" / "VoiceStudio.spec").read_text(encoding="utf-8")
    assert "libs, bins, datas" not in spec
    assert "apply_collect" in spec


def test_spec_helpers_contract_is_ast_verified():
    """collect_package의 unpack 순서가 정확히 datas, binaries, hiddenimports인지."""
    import spec_helpers
    tree = ast.parse((ROOT / "packaging" / "spec_helpers.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and isinstance(node.ctx, ast.Store):
            names = [e.id for e in node.elts if isinstance(e, ast.Name)]
    assert names[:3] == ["datas", "binaries", "hiddenimports"]


def test_spec_repo_root_paths():
    """P12.3-02: SPECPATH 계약으로 계산한 root에 필수 파일이 존재한다."""
    import spec_helpers
    root = spec_helpers.compute_repo_root(str(ROOT / "packaging"))
    assert root == ROOT
    assert (root / "src" / "voice_studio" / "main.py").is_file()
    assert (root / "third_party").is_dir()
    assert (root / "third_party" / "bin").is_dir()
    # 실제 spec의 ROOT 계산식과 헬퍼가 동일한 값을 내는지
    spec = (ROOT / "packaging" / "VoiceStudio.spec").read_text(encoding="utf-8")
    assert "ROOT = Path(SPECPATH).resolve().parent" in spec


def test_build_requires_ffmpeg_binaries():
    """P12.3-03: binary가 없으면 빌드가 중단된다(BUILD_OK 전 prerequisite 검사)."""
    bat = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="utf-8")
    assert "if not exist third_party\\bin\\ffmpeg.exe" in bat
    assert "if not exist third_party\\bin\\ffprobe.exe" in bat
    assert bat.index("ffmpeg.exe missing") < bat.index("python -m venv")


def test_check_ffmpeg_script_contract():
    src = (ROOT / "scripts" / "check_ffmpeg.py").read_text(encoding="utf-8")
    assert "libmp3lame" in src
    assert "--enable-gpl" in src          # GPL 여부 판별
    assert "-version" in src              # buildconf 기록 근거
    assert "ffprobe" in src


def test_check_dist_script_matches_frozen_search_path():
    """P12.3-05: 검증 경로가 RealFfmpegAdapter._resolve_binary 탐색 위치(_internal/bin)와 일치."""
    src = (ROOT / "scripts" / "check_dist.py").read_text(encoding="utf-8")
    assert 'dist / "_internal" / "bin"' in src
    assert "ffmpeg.exe" in src and "ffprobe.exe" in src
    from voice_studio.infra import ffmpeg_adapter as fa
    import inspect
    assert '"_internal" / "bin"' in inspect.getsource(fa.bundled_bin_dirs)


def test_build_script_torch_torchaudio_same_cuda_index():
    """P12.3-11/12: torch+torchaudio 동일 index 설치, 설치 전후 버전 기록."""
    bat = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="utf-8")
    assert "pip install torch torchaudio --index-url" in bat
    assert "TORCHAUDIO_BEFORE_QWEN_TTS" in bat and "TORCHAUDIO_AFTER_QWEN_TTS" in bat
    assert "assert torch.version.cuda is not None" in bat
