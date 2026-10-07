#!/usr/bin/env python3
"""P13 런타임 E2E 스크립트. 실기 Windows + RTX GPU에서 실행한다.

사용법(사용자 파일은 절대 저장소로 복사하지 않는다):
    python scripts/p13_runtime_e2e.py --audio "<user audio file>" \
        --ref-text "참조 대사" --generate-text "생성할 문장" \
        [--model-dir ...] [--app-exe "D:/.../VoiceStudio.exe"] [--skip-stt]

흐름(15 단계): 모델 스냅샷 확인 → ffprobe(flat contract) → waveform → 구간 디코딩 →
WAV 인코딩 → temp profile root 격리 → register worker(실제 프로세스) → 결과/프로필
확인(ProfileService.load_prompt_spec) → 자동 받아쓰기(faster-whisper small int8) →
narrate worker → MP3 생성 → ffprobe 검증 → 결과 보고.

- 프로필/작업/출력은 temp 데이터 루트에서만 생성하고 실제 사용자 profiles를
  절대 변경하지 않는다(종료 후 temp를 삭제한다).
- 모델 캐시는 실제 VoiceStudio가 받은 스냅샷(ModelManager().model_path())을 재사용한다.
- --app-exe가 주어지면 register/narrate 모두 frozen exe로 worker를 실행한다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid as uuidlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def step(msg: str) -> None:
    """고정 total 없이 단계 이름만 출력한다(P13 §17 option B)."""
    print(f"[P13] {msg}", flush=True)


def fail(msg: str) -> "NoReturn":
    print(f"E2E_FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def read_events(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]


def _worker_cmd(app_exe: str, job_path: Path) -> list[str]:
    """source(python -m voice_studio.main) / frozen(exe --worker) 두 모드."""
    if app_exe:
        return [app_exe, "--worker", str(job_path)]
    return [sys.executable, "-m", "voice_studio.main", "--worker", str(job_path)]


def _worker_env(app_exe: str, bin_dir: "Path | None" = None) -> "dict[str, str] | None":
    """source worker child에만 repository third_party/bin을 PATH 선두로 주입한다.

    - source mode(app_exe가 비어 있음): 검증된 repository third_party/bin을 PATH
      앞에 붙인 child env copy를 반환한다. 시스템 PATH에 의존하는 것이 아니라
      validation harness가 검증한 repo 경로를 child에 명시적으로 노출한다.
    - frozen mode(app_exe 존재): None를 반환해 env를 무변경 상속한다. repo PATH를
      넣으면 packaged bundled FFmpeg 누락이 PATH fallback으로 숨어 거짓 양성이 된다.
    - global os.environ는 변경하지 않고 child env copy만 만든다. 이 스크립트는
      frozen E2E에도 쓰이므로 전역 PATH 오염을 피한다. VOICE_STUDIO_DATA_DIR는
      copy에 이미 포함되므로 temp 데이터 격리는 그대로 child에 상속된다.
    """
    if app_exe:
        return None
    import os
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "runtime_validation_helpers", ROOT / "scripts" / "runtime_validation_helpers.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    target = bin_dir if bin_dir is not None else _mod.REPO_BIN_DIR
    try:
        _mod.validation_ffmpeg_paths(target)
    except _mod.ValidationFfmpegMissing as exc:
        fail(str(exc))
    env = os.environ.copy()
    env["PATH"] = str(target) + os.pathsep + env.get("PATH", "")
    return env


def _isolate_data_root() -> Path:
    """E2E 전용 데이터 루트로 격리한다. 실제 사용자 profiles는 건드리지 않는다.

    - Windows/비-Windows 모두 VOICE_STUDIO_DATA_DIR override를 사용한다
      (paths.app_data_dir는 override를 OS 무관하게 우선한다).
    - LOCALAPPDATA 전체를 바꾸지 않으므로 Windows의 다른 library/cache 동작에
      영향이 없고, child source/frozen worker가 env를 상속해 동일 temp
      profile/cache를 사용한다.
    모델 캐시 경로는 격리 전에 먼저 확정하므로 기존 다운로드 모델이 재사용된다.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="p13_e2e_data_"))
    import os
    os.environ["VOICE_STUDIO_DATA_DIR"] = str(tmp_root)
    return tmp_root


def encode_reference_wav(audio, pcm, ref_wav: Path) -> dict:
    """reference.wav를 adapter 계약대로 인코딩하고 validation artifact를 확인한다.

    RealFfmpegAdapter.encode_wav(pcm, sample_rate, out_path)의 3-인자 계약을
    사용한다(AudioService의 2-인자 wrapper와 혼동하지 말 것). 이 WAV는 E2E
    validation artifact일 뿐이며 production register worker 입력 구조와 무관하다.
    """
    from voice_studio.core import config

    audio.encode_wav(pcm, config.REFERENCE_SAMPLE_RATE, str(ref_wav))
    wav_path = Path(ref_wav)
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        fail(f"reference.wav missing/empty: {wav_path}")
    probe = audio.probe(str(wav_path))
    if int(probe.get("sample_rate", 0)) != config.REFERENCE_SAMPLE_RATE:
        fail(f"reference.wav sample_rate mismatch: {probe}")
    if int(probe.get("channels", 0)) != 1:
        fail(f"reference.wav is not mono: {probe}")
    return probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="사용자 참조 오디오 파일(한글 경로 가능)")
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--generate-text", required=True)
    ap.add_argument("--model-dir", default="", help="로컬 Qwen 스냅샷 디렉터리(비우면 ModelManager 기본 경로 우선)")
    ap.add_argument("--app-exe", default="", help="frozen worker 실행용 VoiceStudio.exe 경로(비우면 source 모드)")
    ap.add_argument("--skip-stt", action="store_true", help="자동 받아쓰기 단계 생략(기본은 실행)")
    args = ap.parse_args()

    src_audio = Path(args.audio)
    if not src_audio.is_file():
        fail(f"audio not found: {src_audio}")

    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "runtime_validation_helpers", ROOT / "scripts" / "runtime_validation_helpers.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    audio = _mod.make_validation_adapter()  # third_party/bin 고정(PATH 비의존)
    from voice_studio.core import config
    from voice_studio.infra.profile_repository import ProfileRepository
    from voice_studio.services.audio_service import AudioService
    from voice_studio.services.model_manager import ModelManager
    from voice_studio.services.profile_service import ProfileService

    model_dir = args.model_dir or _resolve_model_dir()
    if not model_dir:
        fail("모델 스냅샷을 찾지 못했습니다. 앱에서 모델을 받았는지 확인하거나 --model-dir을 지정하세요.")

    tmp_data = _isolate_data_root()
    tmp_work = Path(tempfile.mkdtemp(prefix="p13_e2e_work_"))
    try:
        # 1 모델 스냅샷(실제 앱이 받은 캐시 재사용)
        step(f"model snapshot: {model_dir}")
        if not Path(model_dir).is_dir():
            fail("model snapshot missing")

        # 2 ffprobe(flat contract: duration/format_name/...)
        step(f"ffprobe: {src_audio.name}")
        probe = audio.probe(str(src_audio))
        duration = float(probe["duration"])
        print(f"      duration={duration:.2f}s format={probe['format_name']} codec={probe['codec']}")

        # 3 waveform
        step("waveform buckets")
        wf = audio.waveform(str(src_audio), 64)
        if len(wf) != 64:
            fail("waveform bucket count mismatch")

        # 4 구간 선택/디코딩
        start_s, end_s = 0.0, min(6.0, duration)
        step(f"decode segment {start_s:.1f}-{end_s:.1f}s")
        pcm = audio.decode_segment(str(src_audio), start_s, end_s, config.REFERENCE_SAMPLE_RATE)
        if pcm.dtype != __import__("numpy").float32 or pcm.ndim != 1 or pcm.size == 0:
            fail("decoded PCM sanity failed")

        # 5 WAV 인코딩 + validation artifact 확인
        step("encode reference wav")
        ref_wav = tmp_work / "reference.wav"
        ref_probe = encode_reference_wav(audio, pcm, ref_wav)
        print(f"      reference.wav sr={int(ref_probe['sample_rate'])} "
              f"channels={int(ref_probe['channels'])} "
              f"size={ref_wav.stat().st_size}B")

        # 6 job 준비(register). profiles root는 temp 격리 + unique 이름.
        step("prepare register job (temp profile root, unique name)")
        profiles_root = _profiles_dir()
        profile_name = f"P13 E2E {uuidlib.uuid4().hex[:8]}"
        job = {
            "schema_version": 1, "job_id": f"e2e-{uuidlib.uuid4().hex[:8]}",
            "mode": "register", "profile_uuid": "",
            "profile_dir": str(profiles_root), "model_path": str(model_dir),
            "name": profile_name, "source_path": str(src_audio),
            "start_s": start_s, "end_s": end_s, "ref_text": args.ref_text,
        }
        job_path = tmp_work / "register_job.json"
        job_path.write_text(json.dumps(job, ensure_ascii=True), encoding="utf-8")

        # 7 register worker 실행(실제 프로세스 경계)
        step("run register worker process")
        r = subprocess.run(_worker_cmd(args.app_exe, job_path),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=1200,
                           env=_worker_env(args.app_exe))
        events = read_events(r.stdout)
        kinds = [e.get("kind") for e in events]
        print(f"      exit={r.returncode} events={kinds}")
        if r.returncode != 0:
            last_events = events[-3:] if events else []
            fail(f"register worker failed: {r.stderr[-500:]} | last events: {last_events} | "
                 f"diagnostics: {_logs_dir() / 'worker-stderr.log'}")

        # 8 result 이벤트 확인
        step("verify register result event")
        result = next((e for e in events if e.get("kind") == "result" and e.get("result_type") == "profile"), None)
        if result is None:
            fail("no profile result event")
        profile_uuid = result["profile_uuid"]

        # 9 프로필 저장 확인(ProfileService.load_prompt_spec 공식 경로)
        step("reload profile via ProfileService.load_prompt_spec")
        repo = ProfileRepository(profiles_root)
        service = ProfileService(repo, AudioService(audio))
        spec = service.load_prompt_spec(profile_uuid)
        if spec.ref_code is None or spec.ref_spk_embedding is None:
            fail("persisted profile missing ref_code/ref_spk_embedding")
        step(f"profile persisted: {profile_uuid}")

        # 9b 프로필 artifact 존재/무결성(P13 §19)
        step("verify profile artifacts on disk")
        pdir = Path(profiles_root) / profile_uuid
        metadata = json.loads((pdir / "metadata.json").read_text(encoding="utf-8"))
        for key in ("schema_version", "uuid", "name", "ref_text", "reference_duration_ms",
                    "model_id", "x_vector_only_mode", "icl_mode", "ref_code_kind"):
            if key not in metadata:
                fail(f"metadata missing key: {key}")
        prompt_path = pdir / "prompt.safetensors"
        ref_flac = pdir / "reference.flac"
        if not prompt_path.is_file() or prompt_path.stat().st_size == 0:
            fail("prompt.safetensors missing/empty")
        if not ref_flac.is_file() or ref_flac.stat().st_size == 0:
            fail("reference.flac missing/empty")
        audio.probe(str(ref_flac))  # decode 가능해야 한다
        step("profile artifacts OK")

        # 10 자동 받아쓰기. source면 source transcriber, frozen이면 exe --stt-smoke.
        if args.skip_stt:
            step("STT skipped (--skip-stt)")
        elif args.app_exe:
            step("STT smoke via frozen app (VoiceStudio.exe --stt-smoke)")
            r_stt = subprocess.run(_stt_smoke_cmd(args.app_exe, str(src_audio), start_s, end_s),
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=1200,
                                   env=_worker_env(args.app_exe))
            print(f"      exit={r_stt.returncode}")
            if r_stt.returncode != 0:
                fail(f"frozen STT smoke failed: {(r_stt.stderr or '')[-500:]}")
        else:
            step("STT transcribe (source faster-whisper small/int8)")
            try:
                from voice_studio.services.transcription_service import FasterWhisperTranscriber
                text = FasterWhisperTranscriber().transcribe(pcm, config.REFERENCE_SAMPLE_RATE)
            except Exception as exc:
                fail(f"STT load/transcribe failed: {type(exc).__name__}: {exc}")
            print(f"      stt_text={text!r}")

        # 11 narrate job 준비
        step("prepare narrate job")
        out_mp3 = tmp_work / "e2e_output.mp3"
        njob = {
            "schema_version": 1, "job_id": f"e2e-{uuidlib.uuid4().hex[:8]}",
            "mode": "narrate", "profile_uuid": profile_uuid,
            "profile_dir": str(profiles_root), "model_path": str(model_dir),
            "segments": [args.generate_text], "gap_flags": [False],
            "bitrate_kbps": 192, "output_path": str(out_mp3),
        }
        njob_path = tmp_work / "narrate_job.json"
        njob_path.write_text(json.dumps(njob, ensure_ascii=True), encoding="utf-8")

        # 12 narrate worker 실행
        step("run narrate worker process")
        r2 = subprocess.run(_worker_cmd(args.app_exe, njob_path),
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=1200,
                            env=_worker_env(args.app_exe))
        events2 = read_events(r2.stdout)
        kinds2 = [e.get("kind") for e in events2]
        print(f"      exit={r2.returncode} events={kinds2}")
        if r2.returncode != 0:
            last_events = events2[-3:] if events2 else []
            fail(f"narrate worker failed: {r2.stderr[-500:]} | last events: {last_events} | "
                 f"diagnostics: {_logs_dir() / 'worker-stderr.log'}")
        step("progress events observed" if any(k == "progress" for k in kinds2) else "(progress 없이 완료)")

        # 13 MP3 결과 확인(protocol.result_event 계약: output_path)
        step("verify audio result event (output_path)")
        res2 = next((e for e in events2 if e.get("kind") == "result" and e.get("result_type") == "audio"), None)
        if res2 is None:
            fail("no audio result event")
        mp3 = Path(res2["output_path"])
        if not mp3.is_file() or mp3.stat().st_size == 0:
            fail("mp3 missing/empty")

        # 14 ffprobe로 MP3 무결성 확인(flat contract)
        step("ffprobe output mp3")
        probe2 = audio.probe(str(mp3))
        if "mp3" not in str(probe2.get("format_name", "")):
            fail(f"output not mp3: {probe2}")
        step(f"mp3 duration={float(probe2['duration']):.2f}s")

        # 15 종료(temp 데이터 정리 후 성공)
        print(f"      profile_name={profile_name} (temp data root: {tmp_data})")
        step("E2E complete")
        print("P13_E2E_OK")
        return 0
    finally:
        # temp 격리 데이터/작업물 정리(실제 사용자 프로필에는 영향 없음)
        shutil.rmtree(tmp_data, ignore_errors=True)
        shutil.rmtree(tmp_work, ignore_errors=True)


def _stt_smoke_cmd(app_exe: str, audio_path: str, start_s: float, end_s: float) -> list[str]:
    """frozen STT smoke 명령(exe 내부의 bundled FFmpeg/faster-whisper 사용)."""
    return [app_exe, "--stt-smoke", "--audio", audio_path,
            "--start", f"{start_s:.3f}", "--end", f"{end_s:.3f}"]


def _logs_dir() -> Path:
    from voice_studio.core.paths import logs_dir
    return logs_dir()


def _profiles_dir() -> Path:
    """격리된 profiles 디렉터리(격리 env 반영 직후 조회)."""
    from voice_studio.core.paths import profiles_dir
    d = profiles_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_model_dir() -> str:
    """ModelManager().model_path() 계약을 우선 사용하고, 없으면 HF 캐시에서 탐색한다."""
    from voice_studio.core.errors import ModelNotDownloadedError
    from voice_studio.services.model_manager import ModelManager
    try:
        return ModelManager().model_path()
    except ModelNotDownloadedError:
        pass
    import os
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    base = Path(hf_home) / "hub"
    for cand in sorted(base.glob("models--Qwen--Qwen3-TTS-12Hz-0.6B-Base" + os.sep + "snapshots" + os.sep + "*"), reverse=True):
        if cand.is_dir():
            return str(cand)
    return ""


if __name__ == "__main__":
    sys.exit(main())
