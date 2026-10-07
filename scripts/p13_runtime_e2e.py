#!/usr/bin/env python3
"""P13 런타임 E2E 스크립트(§16-17). 실기 Windows + RTX GPU에서 실행한다.

사용법(사용자 파일은 절대 저장소로 복사하지 않는다):
    python scripts/p13_runtime_e2e.py --audio "<user audio file>" \
        --ref-text "참조 대사" --generate-text "생성할 문장" [--model-dir ...]

흐름(23 단계): 모델 스냅샷 확인 → ffprobe → waveform → 구간 디코딩 → WAV 인코딩 →
register worker(실제 프로세스) → JSONL 이벤트 파싱 → 프로필 저장 확인 → worker 종료 →
프로필 재로드 → narrate worker(실제 프로세스) → 진행 이벤트 → MP3 생성 →
ffprobe 검증 → 결과 보고. 각 단계는 [n/23] 라벨과 함께 stdout으로 진행을 출력한다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import uuid as uuidlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STEP = 0


def step(msg: str) -> None:
    global STEP
    STEP += 1
    print(f"[{STEP}/23] {msg}", flush=True)


def fail(msg: str) -> "NoReturn":
    print(f"E2E_FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def read_events(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="사용자 참조 오디오 파일(한글 경로 가능)")
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--generate-text", required=True)
    ap.add_argument("--model-dir", default="", help="로컬 Qwen 스냅샷 디렉터리(비우면 기본 HF 캐시 탐색)")
    ap.add_argument("--data-dir", default="", help="VOICE_STUDIO_DATA_DIR(기본: 실제 앱 데이터 디렉터리)")
    args = ap.parse_args()

    src_audio = Path(args.audio)
    if not src_audio.is_file():
        fail(f"audio not found: {src_audio}")

    import os
    if args.data_dir:
        os.environ["VOICE_STUDIO_DATA_DIR"] = args.data_dir

    from voice_studio.core import config
    from voice_studio.core.paths import profiles_dir, preview_cache_dir
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter, production_device  # noqa: F401
    from voice_studio.infra.qwen_adapter import RealQwenAdapter, production_device
    from voice_studio.infra.profile_repository import ProfileRepository

    audio = RealFfmpegAdapter()
    model_dir = args.model_dir or _resolve_model_dir()

    # 1 모델 스냅샷
    step(f"model snapshot: {model_dir}")
    if not Path(model_dir).is_dir():
        fail("model snapshot missing")

    # 2 ffprobe
    step(f"ffprobe: {src_audio.name}")
    probe = audio.probe(str(src_audio))
    duration = float(probe["format"]["duration"])
    print(f"      duration={duration:.2f}s format={probe['format'].get('format_name')}")

    # 3 waveform
    step("waveform buckets")
    wf = audio.waveform(str(src_audio), 64)
    if len(wf) != 64:
        fail("waveform bucket count mismatch")

    # 4 구간 선택/디코딩
    start_s, end_s = 0.0, min(6.0, duration)
    step(f"decode segment {start_s:.1f}-{end_s:.1f}s")
    pcm = audio.decode_segment(str(src_audio), start_s, end_s, 24000)
    if pcm.dtype != __import__("numpy").float32 or pcm.ndim != 1 or pcm.size == 0:
        fail("decoded PCM sanity failed")

    # 5 WAV 인코딩(preview cache 재사용)
    step("encode reference wav")
    tmp = Path(tempfile.mkdtemp(prefix="p13_e2e_"))
    ref_wav = tmp / "reference.wav"
    audio.encode_wav(pcm, str(ref_wav))

    # 6 job 준비(register)
    step("prepare register job")
    profiles_root = profiles_dir()
    job = {
        "schema_version": 1, "job_id": f"e2e-{uuidlib.uuid4().hex[:8]}",
        "mode": "register", "profile_uuid": "",
        "profile_dir": str(profiles_root), "model_path": str(model_dir),
        "name": "P13 E2E", "source_path": str(src_audio),
        "start_s": start_s, "end_s": end_s, "ref_text": args.ref_text,
    }
    job_path = tmp / "register_job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=True), encoding="utf-8")

    # 7-9 register worker 실행(실제 프로세스 경계) + 이벤트 파싱
    step("run register worker process")
    r = subprocess.run([sys.executable, "-m", "voice_studio.main", "--worker", str(job_path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200)
    events = read_events(r.stdout)
    kinds = [e.get("kind") for e in events]
    print(f"      exit={r.returncode} events={kinds}")
    if r.returncode != 0:
        fail(f"register worker failed: {r.stderr[-500:]}")

    # 10 result 이벤트/프로필 저장 확인
    step("verify register result event")
    result = next((e for e in events if e.get("kind") == "result" and e.get("result_type") == "profile"), None)
    if result is None:
        fail("no profile result event")
    profile_uuid = result["profile_uuid"]
    repo = ProfileRepository(profiles_root)
    spec = repo.load_prompt_spec(profile_uuid)
    if spec.ref_code is None or spec.ref_spk_embedding is None:
        fail("persisted profile missing ref_code/ref_spk_embedding")
    step(f"profile persisted: {profile_uuid}")

    # 11 worker 프로세스 종료 확인(GPU 반환 후 완전히 끝난 상태)
    step("register worker exited")

    # 12 narrate job 준비
    step("prepare narrate job")
    out_mp3 = tmp / "e2e_output.mp3"
    njob = {
        "schema_version": 1, "job_id": f"e2e-{uuidlib.uuid4().hex[:8]}",
        "mode": "narrate", "profile_uuid": profile_uuid,
        "profile_dir": str(profiles_root), "model_path": str(model_dir),
        "segments": [args.generate_text], "gap_flags": [False],
        "bitrate_kbps": 192, "output_path": str(out_mp3),
    }
    njob_path = tmp / "narrate_job.json"
    njob_path.write_text(json.dumps(njob, ensure_ascii=True), encoding="utf-8")

    # 13-15 narrate worker 실행
    step("run narrate worker process")
    r2 = subprocess.run([sys.executable, "-m", "voice_studio.main", "--worker", str(njob_path)],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200)
    events2 = read_events(r2.stdout)
    kinds2 = [e.get("kind") for e in events2]
    print(f"      exit={r2.returncode} events={kinds2}")
    if r2.returncode != 0:
        fail(f"narrate worker failed: {r2.stderr[-500:]}")
    step("progress events observed" if any(k == "progress" for k in kinds2) else "(progress 없이 완료)")

    # 16-18 MP3 결과 확인
    step("verify audio result event")
    res2 = next((e for e in events2 if e.get("kind") == "result" and e.get("result_type") == "audio"), None)
    if res2 is None:
        fail("no audio result event")
    mp3 = Path(res2["path"])
    step(f"mp3 produced: {mp3.name}")
    if not mp3.is_file() or mp3.stat().st_size == 0:
        fail("mp3 missing/empty")

    # 19-21 ffprobe로 MP3 무결성 확인
    step("ffprobe output mp3")
    probe2 = audio.probe(str(mp3))
    if "mp3" not in str(probe2.get("format", {}).get("format_name", "")):
        fail(f"output not mp3: {probe2}")
    step(f"mp3 duration={float(probe2['format']['duration']):.2f}s")

    # 22-23 종료
    step("profile reload in new worker process verified")
    step("E2E complete")
    print("P13_E2E_OK")
    return 0


def _resolve_model_dir() -> str:
    """HF 캐시에서 Qwen3-TTS-12Hz-0.6B-Base 로컬 스냅샷을 찾는다."""
    import os
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    base = Path(hf_home) / "hub"
    for cand in sorted(base.glob("models--Qwen--Qwen3-TTS-12Hz-0.6B-Base" + os.sep + "snapshots" + os.sep + "*"), reverse=True):
        if cand.is_dir():
            return str(cand)
    return ""


if __name__ == "__main__":
    sys.exit(main())
