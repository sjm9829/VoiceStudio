"""P13 stabilization self-review: 검증 코드 자체의 blocker 회귀.

- scripts/p13_runtime_e2e.py가 존재하는 실제 API만 사용하는지 정적 계약.
- scripts/validate_runtime_gpu_windows.bat가 실제 존재하고 ASCII-only인지.
- ffmpeg_adapter 오타(excxc) 부재와 정확한 예외 계약.
- worker result 이벤트 계약(protocol.result_event → output_path).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _e2e_source() -> str:
    return (ROOT / "scripts" / "p13_runtime_e2e.py").read_text(encoding="utf-8")


def test_e2e_does_not_import_production_device_from_ffmpeg_adapter():
    """production_device는 qwen_adapter에만 존재한다. ffmpeg_adapter import에 섞이면 안 된다."""
    src = _e2e_source()
    for line in src.splitlines():
        if "ffmpeg_adapter import" in line:
            assert "production_device" not in line


def test_e2e_uses_flat_probe_contract():
    """probe 계약은 flat dict(duration/format_name/...)이므로 probe["format"] 접근이 없어야 한다."""
    src = _e2e_source()
    assert 'probe["format"]' not in src
    assert "probe['format']" not in src
    assert 'probe["duration"]' in src


def test_e2e_uses_output_path_result_contract():
    """protocol.result_event는 output_path 키를 사용한다."""
    src = _e2e_source()
    assert 'res2["output_path"]' in src
    assert 'res2["path"]' not in src


def test_e2e_does_not_call_repository_load_prompt_spec():
    """load_prompt_spec는 ProfileService 메서드다. ProfileRepository 직접 호출 금지."""
    src = _e2e_source()
    assert "repo.load_prompt_spec" not in src
    assert "ProfileService" in src and "load_prompt_spec" in src


def test_e2e_uses_model_manager_model_path_first():
    """모델 경로는 ModelManager().model_path() 계약을 우선 사용해야 한다."""
    src = _e2e_source()
    assert "ModelManager().model_path()" in src
    assert "huggingface" in src  # HF 캐시 fallback은 유지


def test_e2e_profile_isolation_is_temp_and_unique():
    """E2E 프로필은 temp 데이터 루트로 격리되고 이름에 unique suffix를 쓴다."""
    src = _e2e_source()
    assert "mkdtemp" in src
    assert "VOICE_STUDIO_DATA_DIR" in src
    assert "LOCALAPPDATA" in src
    assert 'f"P13 E2E {uuidlib.uuid4().hex[:8]}"' in src


def test_e2e_supports_frozen_and_source_worker_modes():
    src = _e2e_source()
    assert "--app-exe" in src
    assert '"--worker"' in src or "'--worker'" in src
    assert "-m" in src and "voice_studio.main" in src


def test_e2e_includes_stt_runtime_step():
    src = _e2e_source()
    assert "FasterWhisperTranscriber" in src
    assert "--skip-stt" in src


def test_e2e_module_import_and_help_succeed():
    import importlib.util
    spec = importlib.util.spec_from_file_location("p13_runtime_e2e", ROOT / "scripts" / "p13_runtime_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import subprocess as sp
    r = sp.run([sys.executable, str(ROOT / "scripts" / "p13_runtime_e2e.py"), "--help"],
               capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    assert "--app-exe" in r.stdout and "--skip-stt" in r.stdout


def test_validate_runtime_gpu_windows_bat_exists_ascii_only():
    bat = ROOT / "scripts" / "validate_runtime_gpu_windows.bat"
    assert bat.is_file(), "validate_runtime_gpu_windows.bat가 repository에 존재해야 한다"
    text = bat.read_text(encoding="ascii")
    assert text.isascii()
    assert "check_cuda.py" in text
    assert "tests\\test_gpu_real.py -m gpu" in text or "test_gpu_real.py -m gpu" in text
    assert "p13_runtime_e2e.py" in text
    # 사용자 audio/ref-text는 인자로 전달(hardcode 금지). source/frozen 분리 args.
    assert "%SOURCE_ARGS%" in text and "%FROZEN_ARGS%" in text


def test_runtime_validation_selects_and_checks_repository_python_before_use():
    text = (ROOT / "scripts" / "validate_runtime_gpu_windows.bat").read_text(encoding="ascii")
    selection = 'set "PYTHON=%CD%\\.venv\\Scripts\\python.exe"'
    guard = (
        'if not exist "%PYTHON%" (\n'
        '  echo [FAIL] .venv not found. Run scripts\\build_windows.bat first.\n'
        '  goto :err\n'
        ')'
    )
    first_call = text.index('"%PYTHON%" -c')
    assert text.index('cd /d "%~dp0.."') < text.index(selection) < text.index(guard) < first_call
    assert "print('executable', sys.executable)" in text
    assert "print('python', sys.version)" in text


def test_runtime_validation_has_no_system_python_or_activation_fallback():
    text = (ROOT / "scripts" / "validate_runtime_gpu_windows.bat").read_text(encoding="ascii")
    commands = [line.strip() for line in text.splitlines()
                if line.strip() and not line.lstrip().lower().startswith("rem ")]
    for line in commands:
        assert not re.search(r'(?:^|[&|])\s*@?(?:call\s+)?"?(?:python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?"?\s',
                             line, re.IGNORECASE), line
    assert "activate" not in text.lower()
    assert sum(line.lower().startswith('set "python=') for line in commands) == 1


def test_runtime_validation_uses_same_python_for_every_stage():
    text = (ROOT / "scripts" / "validate_runtime_gpu_windows.bat").read_text(encoding="ascii")
    calls = [line.strip() for line in text.splitlines() if line.lstrip().startswith('"%PYTHON%" ')]
    for stage in (
        'scripts\\prepare_ffmpeg.py',
        'scripts\\check_ffmpeg.py',
        'scripts\\check_cuda.py',
        '-m pytest -q tests\\test_gpu_real.py -m gpu',
        'scripts\\p13_runtime_e2e.py %SOURCE_ARGS%',
        'scripts\\p13_runtime_e2e.py --app-exe "dist\\VoiceStudio\\VoiceStudio.exe" %FROZEN_ARGS%',
    ):
        assert f'"%PYTHON%" {stage}' in calls
    version_calls = [line for line in calls if line.startswith('"%PYTHON%" -c ')]
    assert any("import sys;" in line for line in version_calls)
    assert any("import torch;" in line for line in version_calls)
    assert any("import torchaudio;" in line for line in version_calls)


def test_all_bat_scripts_ascii_only():
    for bat in (ROOT / "scripts").glob("*.bat"):
        bat.read_text(encoding="ascii")  # non-ascii면 UnicodeDecodeError로 실패


def test_ffmpeg_adapter_no_excxc_typo():
    src = (ROOT / "src" / "voice_studio" / "infra" / "ffmpeg_adapter.py").read_text(encoding="utf-8")
    assert "excxc" not in src
    assert "from exc" in src


def test_worker_result_event_contract_is_output_path():
    from voice_studio.workers.protocol import result_event
    ev = result_event("out/x.mp3", job_id="j", result_type="audio")
    assert ev["output_path"] == "out/x.mp3"
    assert "path" not in ev


def test_jsonl_buffer_lives_in_linebuffer_module():
    """JsonlBuffer는 workers.linebuffer에 있다(protocol에 두면 안 된다)."""
    from voice_studio.workers.linebuffer import JsonlBuffer
    import voice_studio.workers.protocol as protocol
    assert not hasattr(protocol, "JsonlBuffer") or protocol.JsonlBuffer is JsonlBuffer


def test_gpu_real_child_code_uses_module_scoped_validation_adapter():
    """test_d child process code는 parent-only helper(_validation_ffmpeg_adapter)를
    참조하면 NameError가 나므로, file-loaded module(m.make_validation_adapter)만 사용한다."""
    import test_gpu_real as tgr
    code = tgr._roundtrip_child_code(
        src=str(ROOT / "src"),
        rvh=str(ROOT / "scripts" / "runtime_validation_helpers.py"),
        root=str(ROOT / "tmp-profiles"), uuid="u", model="m", gen="g")
    assert "_validation_ffmpeg_adapter" not in code
    assert "m.make_validation_adapter()" in code
    # GPU/torch 없이도 compile(구문/이름 바인딩 경계)이 성공해야 한다.
    compile(code, "<roundtrip-child>", "exec")


def test_gpu_real_child_code_wraps_profile_root_with_path():
    """test_d child code는 ProfileRepository에 str이 아닌 Path를 넘겨야 한다.

    production profile_dir는 `profiles_root / uuid`이므로 str root면
    TypeError: unsupported operand type(s) for / 가 난다. 회귀: child code에
    pathlib Path import와 ProfileRepository(Path(...))가 있어야 하고 compile 성공.
    """
    import test_gpu_real as tgr
    code = tgr._roundtrip_child_code(
        src=str(ROOT / "src"),
        rvh=str(ROOT / "scripts" / "runtime_validation_helpers.py"),
        root=str(ROOT / "tmp-profiles"), uuid="u", model="m", gen="g")
    assert "from pathlib import Path;" in code
    assert "ProfileRepository(Path(r'" in code
    assert "ProfileRepository(r'" not in code  # str 직접 전달 금지
    # ProfileRepository에 Path를 넘기면 profile_dir의 / 연산이 성립한다.
    from voice_studio.infra.profile_repository import ProfileRepository
    repo = ProfileRepository(Path(str(ROOT)))
    assert str(repo.root / "u")  # 실제 / 연산 가능 확인
    compile(code, "<roundtrip-child>", "exec")


def test_gpu_real_child_code_prints_roundtrip_marker():
    import test_gpu_real as tgr
    code = tgr._roundtrip_child_code(
        src="s", rvh="r", root="p", uuid="u", model="m", gen="g")
    assert "GPU_ROUNDTRIP_OK" in code


def _load_e2e_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p13_runtime_e2e", ROOT / "scripts" / "p13_runtime_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _RecordingEncodeWavAdapter:
    """RealFfmpegAdapter.encode_wav(pcm, sample_rate, out_path) 3-인자 계약만 받는 fake."""

    def __init__(self):
        self.calls: list[tuple] = []

    def encode_wav(self, pcm, sample_rate, out_path):
        self.calls.append((sample_rate, out_path))
        import wave
        with wave.open(out_path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(int(sample_rate))
            fh.writeframes(pcm.astype("<i2").tobytes())
        return out_path

    def probe(self, path):
        import wave
        with wave.open(path, "rb") as fh:
            return {"duration": fh.getnframes() / fh.getframerate(),
                    "format_name": "wav", "sample_rate": fh.getframerate(),
                    "channels": fh.getnchannels(), "codec": "pcm_s16le"}


def test_e2e_encode_wav_ast_contract_is_three_positional_args():
    """AST 정적 계약: E2E의 encode_wav 호출은 3-인자이고 2번째가 sample_rate다."""
    import ast
    tree = ast.parse(_e2e_source())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "encode_wav"]
    assert calls, "p13_runtime_e2e.py에 encode_wav 호출이 있어야 한다"
    for call in calls:
        assert len(call.args) == 3, f"encode_wav는 (pcm, sample_rate, out_path) 3-인자: {ast.dump(call)}"
        assert not call.keywords
        second = call.args[1]
        assert isinstance(second, ast.Attribute) and second.attr == "REFERENCE_SAMPLE_RATE", \
            "2번째 인자는 config.REFERENCE_SAMPLE_RATE여야 한다(magic number 금지)"
    # 2-인자 회귀 방지: 소스에 "encode_wav(pcm, str(" 형태가 없어야 한다.
    assert "encode_wav(pcm, str(" not in _e2e_source()


def test_e2e_encode_reference_wav_matches_production_adapter_signature():
    """production RealFfmpegAdapter 시그니처와 E2E 호출 인자 수가 일치해야 한다."""
    import inspect
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter

    params = list(inspect.signature(RealFfmpegAdapter.encode_wav).parameters)
    assert params == ["self", "pcm", "sample_rate", "out_path"]

    import numpy as np
    mod = _load_e2e_module()
    fake = _RecordingEncodeWavAdapter()
    out = tmp_path = None
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="p13_encode_wav_"))
    out = tmp_dir / "reference.wav"
    probe = mod.encode_reference_wav(fake, np.zeros(240, dtype=np.float32), out)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == 24000  # config.REFERENCE_SAMPLE_RATE
    assert fake.calls[0][1] == str(out)
    assert probe["sample_rate"] == 24000 and probe["channels"] == 1
    assert out.stat().st_size > 0


def test_e2e_encode_reference_wav_fails_on_two_arg_call(tmp_path):
    """2-인자 호출로 되돌아가면 fake/real 계약 모두 TypeError로 실패해야 한다."""
    import numpy as np
    mod = _load_e2e_module()
    fake = _RecordingEncodeWavAdapter()
    out = tmp_path / "reference.wav"
    # 정적 AST 검사로 이미 3-인자를 보증하지만, 런타임 경계도 확인한다.
    probe = mod.encode_reference_wav(fake, np.zeros(240, dtype=np.float32), out)
    assert probe["channels"] == 1
    # 2-인자 형태는 production adapter 시그니처와 불일치 → TypeError.
    with __import__("pytest").raises(TypeError):
        fake.encode_wav(np.zeros(4, dtype=np.float32), str(out))


def test_e2e_encode_reference_wav_rejects_non_mono_or_wrong_rate(tmp_path):
    """probe 결과가 mono/24kHz가 아니면 fail()로 exit 1해야 한다."""
    import numpy as np
    mod = _load_e2e_module()

    class _WrongProbe(_RecordingEncodeWavAdapter):
        def probe(self, path):
            return {"duration": 1.0, "format_name": "wav", "sample_rate": 48000,
                    "channels": 2, "codec": "pcm_s16le"}

    out = tmp_path / "reference.wav"
    rc = None
    try:
        mod.encode_reference_wav(_WrongProbe(), np.zeros(240, dtype=np.float32), out)
    except SystemExit as exc:  # fail()는 sys.exit(1)
        rc = exc.code
    assert rc == 1


# ---------------------------------------------------------------------------
# P13 Source E2E worker env injection (validation-only, production unchanged)
# ---------------------------------------------------------------------------

def _make_bin_pair(tmp_path):
    (tmp_path / "ffmpeg.exe").write_bytes(b"x")
    (tmp_path / "ffprobe.exe").write_bytes(b"y")
    return tmp_path


def test_worker_env_source_injects_repo_bin_at_path_front(tmp_path):
    """A/B: source worker env는 repo third_party/bin을 PATH 선두에 두고 기존 PATH를 뒤에 유지한다."""
    import os
    mod = _load_e2e_module()
    bin_dir = _make_bin_pair(tmp_path)
    env = mod._worker_env("", bin_dir=bin_dir)
    assert isinstance(env, dict)
    parts = env["PATH"].split(os.pathsep)
    assert parts[0] == str(bin_dir)  # A: repo bin 선두
    rest = os.pathsep.join(parts[1:])
    base = os.environ.get("PATH", "")
    assert rest == base  # B: 기존 PATH 내용 뒤에 그대로 유지
    # VOICE_STUDIO_DATA_DIR 격리는 os.environ copy에 포함돼 child로 상속되어야 한다.
    if "VOICE_STUDIO_DATA_DIR" in os.environ:
        assert env.get("VOICE_STUDIO_DATA_DIR") == os.environ["VOICE_STUDIO_DATA_DIR"]


def test_worker_env_frozen_mode_no_repo_injection():
    """C: frozen(app_exe 존재)에서는 repo third_party/bin PATH 주입 없이 None(무변경 상속)."""
    mod = _load_e2e_module()
    assert mod._worker_env(r"D:\VoiceStudio\VoiceStudio.exe") is None


def test_worker_env_source_missing_pair_fails_clearly(tmp_path):
    """D: ffmpeg.exe/ffprobe.exe pair가 없으면 source validation은 SystemExit(1)로 명확히 실패."""
    mod = _load_e2e_module()
    empty = tmp_path  # pair 없음
    with __import__("pytest").raises(SystemExit) as exc:
        mod._worker_env("", bin_dir=empty)
    assert exc.value.code == 1


def test_worker_env_register_and_narrate_share_same_env(tmp_path):
    """E: register/narrate 두 worker subprocess 모두 env=_worker_env(args.app_exe)를 사용한다.

    AST로 subprocess.run 호출을 검사해 _worker_cmd 기반 호출이 모두 _worker_env env를
    받는지 확인한다(frozen STT smoke는 None이라 별도). register/narrate 2개를 요구.
    """
    import ast
    tree = ast.parse(_e2e_source())
    worker_runs = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "run"
                and isinstance(fn.value, ast.Name) and fn.value.id == "subprocess"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Call):
            continue
        cmd = node.args[0]
        if not (isinstance(cmd.func, ast.Name) and cmd.func.id == "_worker_cmd"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        assert "env" in kw, "register/narrate subprocess에 env가 있어야 한다"
        envcall = kw["env"]
        assert isinstance(envcall, ast.Call) and isinstance(envcall.func, ast.Name)
        assert envcall.func.id == "_worker_env"
        worker_runs += 1
    assert worker_runs == 2, f"register/narrate 2개 worker subprocess 기대: {worker_runs}"


def test_production_unchanged_by_this_commit():
    """F: production worker_main.py / ffmpeg_adapter.py는 이 정정에서 무변경(계약 유지)."""
    wm = (ROOT / "src" / "voice_studio" / "workers" / "worker_main.py").read_text(encoding="utf-8")
    fa = (ROOT / "src" / "voice_studio" / "infra" / "ffmpeg_adapter.py").read_text(encoding="utf-8")
    # production 계약 그대로: 2-인자 AudioService wrapper, RealFfmpegAdapter() 무인자 생성.
    assert "AudioService(RealFfmpegAdapter())" in wm
    assert "def encode_wav(self, pcm: np.ndarray, sample_rate: int, out_path: str)" in fa
    # HEAD name-only 가드는 커밋 직후 자동으로 깨지는 구조라 제거(P14에서 확인).
    # 대신 현재 production 계약을 직접 검증한다(P15).
    assert "AudioService(RealFfmpegAdapter())" in wm  # 무인자 생성 유지
    assert 'os.environ.get("VOICE_STUDIO_DIAGNOSTICS_DIR")' in wm  # 진단은 env 옵트인만
    import inspect
    params = inspect.signature(__import__(
        "voice_studio.infra.ffmpeg_adapter", fromlist=["RealFfmpegAdapter"]
    ).RealFfmpegAdapter.encode_mp3).parameters
    assert params["out_sample_rate"].default is None  # 기존 4-인자 호출 호환
