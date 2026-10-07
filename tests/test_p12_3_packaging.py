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


def test_merge_collect_contract():
    """P12.3-01/22 + Final Hotfix: merge_collect는 Analysis 이전 list에 병합한다."""
    import spec_helpers

    def fake_collector(name):
        assert name == "ctranslate2"
        return ["d1", "d2"], ["b1"], ["h1", "preexisting"]

    datas, binaries, hiddenimports = [], [], ["preexisting"]
    spec_helpers.merge_collect("ctranslate2", datas, binaries, hiddenimports,
                               collector=fake_collector)
    assert datas == ["d1", "d2"]
    assert binaries == ["b1"]
    assert hiddenimports == ["preexisting", "h1"]


def test_spec_no_longer_unpacks_collect_all_wrongly():
    """기존 버그(libs, bins, datas = collect_all) 패턴이 spec에 남아 있지 않은지."""
    spec = (ROOT / "packaging" / "VoiceStudio.spec").read_text(encoding="utf-8")
    assert "libs, bins, datas" not in spec
    assert "merge_collect" in spec
    assert "apply_collect" not in spec


def test_spec_helpers_contract_is_ast_verified():
    """collect_package는 collect_all을 그대로 반환하고, merge_collect의 unpack 순서가
    정확히 datas, binaries, hiddenimports인지 검증한다(P12.3-01, Final Hotfix)."""
    import spec_helpers
    tree = ast.parse((ROOT / "packaging" / "spec_helpers.py").read_text(encoding="utf-8"))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    # collect_package는 collect_all(name) 호출을 그대로 반환해야 한다.
    rets = [n for n in ast.walk(funcs["collect_package"]) if isinstance(n, ast.Return)]
    assert any(isinstance(r.value, ast.Call) and r.value.func.id == "collect_all" for r in rets)
    # merge_collect의 unpack은 datas, binaries, hiddenimports 순서 계약을 따른다.
    orders = []
    for node in ast.walk(funcs["merge_collect"]):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            orders.append([e.id for e in node.targets[0].elts if isinstance(e, ast.Name)])
    assert ["pkg_datas", "pkg_binaries", "pkg_hiddenimports"] in orders


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
    assert "ROOT = SPEC_DIR.parent" in spec


def test_build_requires_ffmpeg_binaries():
    """P13: 빌드 전 prepare_ffmpeg.py로 pair를 구성하고 check_ffmpeg.py로 검증한다."""
    bat = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="utf-8")
    assert "prepare_ffmpeg.py" in bat and "check_ffmpeg.py" in bat
    assert bat.index("prepare_ffmpeg.py") < bat.index("python -m venv")


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
    assert "=== BEFORE QWEN-TTS ===" in bat and "=== AFTER QWEN-TTS ===" in bat
    assert "assert torch.version.cuda is not None" in bat


def test_spec_adds_spec_dir_to_sys_path_before_helper_import():
    """Final Hotfix 2: spec은 packaging/ 를 sys.path 에 명시 추가한 뒤 spec_helpers를
    import한다. sys.path.insert 가 import spec_helpers 보다 먼저 실행되고, SPEC_DIR이
    실제 packaging/ 디렉터리이며 ROOT가 repository root임을 검증한다."""
    spec_path = ROOT / "packaging" / "VoiceStudio.spec"
    spec = spec_path.read_text(encoding="utf-8")
    spec_dir = spec_path.resolve().parent

    assert "SPEC_DIR = Path(SPECPATH).resolve()" in spec
    assert "sys.path.insert" in spec
    assert "import spec_helpers" in spec

    tree = ast.parse(spec)
    insert_pos = None
    import_pos = None
    for i, node in enumerate(tree.body):
        src = ast.get_source_segment(spec, node) or ""
        if insert_pos is None and isinstance(node, ast.If) and "sys.path.insert" in src:
            insert_pos = i
        if import_pos is None and isinstance(node, ast.Import) \
                and any(a.name == "spec_helpers" for a in node.names):
            import_pos = i
    assert insert_pos is not None and import_pos is not None
    assert insert_pos < import_pos  # sys.path.insert 가 import spec_helpers 보다 먼저

    # SPEC_DIR / ROOT 계산식을 실제 packaging/ 경로로 평가해 기대값을 확인한다.
    ns = {"SPECPATH": str(spec_dir)}
    header_nodes = [
        n for n in tree.body[:import_pos]
        if not (isinstance(n, ast.ImportFrom) and n.module == "PyInstaller.utils.hooks")
    ]
    header = ast.unparse(ast.Module(body=header_nodes, type_ignores=[]))
    exec(compile(header, str(spec_path), "exec"), ns)
    assert ns["SPEC_DIR"] == spec_dir
    assert (spec_dir / "VoiceStudio.spec").is_file()
    assert (spec_dir / "spec_helpers.py").is_file()
    assert ns["ROOT"] == ROOT
    assert (ROOT / "src" / "voice_studio" / "main.py").is_file()
    assert (ROOT / "third_party").is_dir()


def test_spec_evaluation_imports_spec_helpers_smoke():
    """Final Hotfix 2: PyInstaller 없이 spec 전체를 평가해 spec_helpers import가
    sys.path 계약으로 성공하는지 lightweight smoke로 확인한다.
    Analysis/PYZ/EXE/COLLECT와 PyInstaller hooks는 테스트용 가짜로 대체한다."""
    import types
    spec_path = ROOT / "packaging" / "VoiceStudio.spec"

    fake_pyinstaller = types.ModuleType("PyInstaller")
    fake_utils = types.ModuleType("PyInstaller.utils")
    fake_hooks = types.ModuleType("PyInstaller.utils.hooks")
    fake_hooks.collect_submodules = lambda pkg: [pkg]
    fake_hooks.collect_data_files = lambda pkg, include_py_files=False: []
    fake_hooks.collect_all = lambda pkg: ([], [], [])
    fake_utils.hooks = fake_hooks
    fake_pyinstaller.utils = fake_utils
    saved = {k: sys.modules.get(k) for k in ("PyInstaller", "PyInstaller.utils", "PyInstaller.utils.hooks")}
    sys.modules["PyInstaller"] = fake_pyinstaller
    sys.modules["PyInstaller.utils"] = fake_utils
    sys.modules["PyInstaller.utils.hooks"] = fake_hooks
    try:
        class _FakeAnalysis:
            def __init__(self, *a, **kw):
                self.pure = []
                self.zipped_data = []
                self.scripts = []
                self.binaries = []
                self.zipfiles = []
                self.datas = []
        ns = {
            "SPECPATH": str(spec_path.resolve().parent),
            "Analysis": _FakeAnalysis,
            "PYZ": lambda *a, **kw: object(),
            "EXE": lambda *a, **kw: object(),
            "COLLECT": lambda *a, **kw: object(),
        }
        exec(compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert ns["ROOT"] == ROOT
    assert "spec_helpers" in sys.modules
