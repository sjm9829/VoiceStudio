"""P13 validation correction: frozen STT smoke CLI + E2E source/frozen 분리 회귀.

- main.py --stt-smoke dispatch/exit code/profile side-effect 없음.
- p13_runtime_e2e.py source STT(transcriber) vs frozen STT(exe --stt-smoke) 분기.
- worker source/frozen 명령 분리와 model path가 LOCALAPPDATA 격리보다 먼저 resolve됨.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_e2e():
    spec = importlib.util.spec_from_file_location("p13_runtime_e2e", ROOT / "scripts" / "p13_runtime_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(argv, monkeypatch, *, decode=None, transcribe=None, raise_in=None, logs_dir=None):
    """main.run_stt_smoke를 fake adapter/transcriber로 실행한다."""
    import voice_studio.main as main_mod
    import voice_studio.infra.ffmpeg_adapter as fa
    import voice_studio.services.transcription_service as ts

    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def decode_segment(self, path, start_s, end_s, sample_rate):
            if raise_in == "decode":
                from voice_studio.core.errors import UnsupportedAudioError
                raise UnsupportedAudioError("decode boom")
            self.calls.append((path, start_s, end_s, sample_rate))
            return decode if decode is not None else np.zeros(24000, dtype=np.float32)

    class FakeTranscriber:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, pcm, sample_rate):
            if raise_in == "transcribe":
                raise RuntimeError("stt boom")
            transcribe_calls.append((pcm.dtype, pcm.shape, sample_rate))
            return "가짜 받아쓰기 결과"

    transcribe_calls = []
    monkeypatch.setattr(fa, "RealFfmpegAdapter", FakeAdapter)
    monkeypatch.setattr(ts, "FasterWhisperTranscriber", FakeTranscriber)
    import voice_studio.core.paths as paths
    if logs_dir is None:
        logs_dir = tmp_path / "logs"
    monkeypatch.setattr(paths, "logs_dir", lambda: logs_dir)
    code = main_mod.run_stt_smoke(argv)
    return code, transcribe_calls, logs_dir


def test_stt_smoke_source_dispatch_succeeds(monkeypatch, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    pcm = np.ones(4800, dtype=np.float32)
    code, calls, _ = _run_main(["--stt-smoke", "--audio", str(audio),
                                "--start", "0", "--end", "6"],
                               monkeypatch, decode=pcm, logs_dir=tmp_path / "l1")
    assert code == 0
    assert calls and calls[0] == (pcm.dtype, pcm.shape, 24000)


def test_stt_smoke_decode_failure_nonzero(monkeypatch, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    code, _, _ = _run_main(["--stt-smoke", "--audio", str(audio)], monkeypatch, raise_in="decode", logs_dir=tmp_path / "l2")
    assert code == 1


def test_stt_smoke_transcriber_failure_nonzero(monkeypatch, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    code, _, _ = _run_main(["--stt-smoke", "--audio", str(audio)], monkeypatch, raise_in="transcribe", logs_dir=tmp_path / "l3")
    assert code == 1


def test_stt_smoke_missing_audio_nonzero(monkeypatch, tmp_path):
    code, _, _ = _run_main(["--stt-smoke", "--audio", str(tmp_path / "nope.wav")], monkeypatch, logs_dir=tmp_path / "l4")
    assert code == 1


def test_stt_smoke_success_marker_in_log(monkeypatch, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    code, _, logs_dir = _run_main(["--stt-smoke", "--audio", str(audio)], monkeypatch, logs_dir=tmp_path / "l5")
    assert code == 0
    log = (logs_dir / "stt-smoke.log").read_text(encoding="utf-8")
    assert "STT_SMOKE_OK" in log
    assert "STT_SMOKE_FAILED" not in log


def test_stt_smoke_creates_no_profile(monkeypatch, tmp_path):
    """STT smoke는 profile을 생성하지 않는다(P13 §9)."""
    import voice_studio.core.paths as paths
    profiles_before = set()
    if paths.profiles_dir().exists():
        profiles_before = {p.name for p in paths.profiles_dir().iterdir()}
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake")
    code, _, _ = _run_main(["--stt-smoke", "--audio", str(audio)], monkeypatch, logs_dir=tmp_path / "l5")
    assert code == 0
    profiles_after = set()
    if paths.profiles_dir().exists():
        profiles_after = {p.name for p in paths.profiles_dir().iterdir()}
    assert profiles_after == profiles_before


def test_main_routes_stt_smoke(monkeypatch):
    """main()이 --stt-smoke를 run_stt_smoke로 dispatch한다."""
    import voice_studio.main as main_mod
    called = []
    monkeypatch.setattr(main_mod, "run_stt_smoke", lambda argv: called.append(list(argv)) or 7)
    assert main_mod.main(["--stt-smoke", "--audio", "x"]) == 7
    assert called == [["--stt-smoke", "--audio", "x"]]


# ---------------- E2E 구조 회귀 ----------------

def test_e2e_worker_cmd_source_vs_frozen():
    """source는 python -m voice_studio.main, frozen은 exe --worker. 혼합 금지(P13 §13)."""
    e2e = _load_e2e()
    job = Path("job.json")
    src_cmd = e2e._worker_cmd("", job)
    frozen_cmd = e2e._worker_cmd("D:/x/VoiceStudio.exe", job)
    assert src_cmd == [sys.executable, "-m", "voice_studio.main", "--worker", str(job)]
    assert frozen_cmd == ["D:/x/VoiceStudio.exe", "--worker", str(job)]


def test_e2e_stt_smoke_cmd_is_frozen_app_call():
    """frozen STT는 exe --stt-smoke로 exe 프로세스 안에서 실행된다(P13 §10, §12)."""
    e2e = _load_e2e()
    cmd = e2e._stt_smoke_cmd("dist/VoiceStudio/VoiceStudio.exe", "a.wav", 0.0, 6.0)
    assert cmd[0] == "dist/VoiceStudio/VoiceStudio.exe"
    assert cmd[1:3] == ["--stt-smoke", "--audio"]
    assert cmd[3] == "a.wav"
    assert cmd[4:6] == ["--start", "0.000"]
    assert cmd[6:8] == ["--end", "6.000"]


def test_e2e_stt_branches_source_vs_frozen():
    """frozen mode에서는 source transcriber를 실행하지 않는다(P13 §12)."""
    src = (ROOT / "scripts" / "p13_runtime_e2e.py").read_text(encoding="utf-8")
    i_skip = src.index('if args.skip_stt:')
    i_frozen = src.index('elif args.app_exe:', i_skip)
    i_source = src.index('else:', i_frozen)
    frozen_block = src[i_frozen:i_source]
    source_block = src[i_source:src.index('\n', src.index('stt_text=', i_source))]
    assert '--stt-smoke' in frozen_block and '_stt_smoke_cmd' in frozen_block
    assert 'FasterWhisperTranscriber' in source_block
    assert 'FasterWhisperTranscriber' not in frozen_block


def test_e2e_model_resolved_before_data_root_isolation():
    """LOCALAPPDATA 격리보다 모델 경로 resolve가 먼저여야 한다(P13 §16)."""
    tree = ast.parse((ROOT / "scripts" / "p13_runtime_e2e.py").read_text(encoding="utf-8"))
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    order = []
    for node in ast.walk(main_fn):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            if name in ("_resolve_model_dir", "_isolate_data_root"):
                order.append((node.lineno, name))
    order.sort()
    names = [n for _, n in order]
    assert names.index("_resolve_model_dir") < names.index("_isolate_data_root")


def test_e2e_step_output_has_no_fixed_total():
    """고정 total 제거: [P13] 형식이고 [n/total] 형식이 없다(P13 §17B)."""
    src = (ROOT / "scripts" / "p13_runtime_e2e.py").read_text(encoding="utf-8")
    assert 'print(f"[{STEP}/{TOTAL}]' not in src
    assert 'f"[P13] {msg}"' in src
    assert "TOTAL =" not in src


def test_e2e_verifies_profile_artifacts():
    """register 후 metadata/prompt.safetensors/reference.flac 검증이 있다(P13 §19)."""
    src = (ROOT / "scripts" / "p13_runtime_e2e.py").read_text(encoding="utf-8")
    for key in ("schema_version", "uuid", "name", "ref_text", "reference_duration_ms",
                "model_id", "x_vector_only_mode", "icl_mode", "ref_code_kind"):
        assert f'"{key}"' in src, key
    assert 'prompt.safetensors' in src and 'reference.flac' in src


def test_e2e_worker_failure_reports_diagnostics():
    """실패 시 worker events와 log 위치를 안내한다(P13 §25)."""
    src = (ROOT / "scripts" / "p13_runtime_e2e.py").read_text(encoding="utf-8")
    assert "worker-stderr.log" in src
    assert "last events" in src


def test_validate_bat_runs_both_e2e_modes_and_gates():
    """validation bat는 prepare→check→cuda→gpu pytest→source E2E→frozen smoke→frozen E2E 순서다."""
    src = (ROOT / "scripts" / "validate_runtime_gpu_windows.bat").read_text(encoding="ascii")
    markers = ["prepare_ffmpeg.py", "check_ffmpeg.py", "check_cuda.py",
               "test_gpu_real.py -m gpu", "p13_runtime_e2e.py %SOURCE_ARGS%",
               "SOURCE_E2E_OK", "--smoke-test --require-gpu",
               '--app-exe "dist\\VoiceStudio\\VoiceStudio.exe" %FROZEN_ARGS%',
               "FROZEN_E2E_OK", "GPU_VALIDATION_OK"]
    pos = -1
    for m in markers:
        i = src.index(m)
        assert i > pos, f"marker out of order: {m}"
        pos = i
    assert "SOURCE_ARGS" in src and "FROZEN_ARGS" in src
    assert '--app-exe "dist\\VoiceStudio\\VoiceStudio.exe" %SOURCE_ARGS%' not in src
