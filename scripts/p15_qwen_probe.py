#!/usr/bin/env python3
"""P15 Qwen3-TTS 생성 원인 분석 프로브(test/tooling 전용).

P15 진단 결과: 잘림·중단 끊김은 최종 MP3가 아니라 Qwen 원본 24kHz WAV부터 존재.
따라서 이 스크립트는 Qwen3-TTS 생성 과정만 집중 조사한다. production 경로
(audio_service / ffmpeg / worker)는 일절 변경하지 않고, 공식 qwen_tts API를
직접 사용한다. 모든 실험에서 생성 직후 원본 float32 24kHz WAV를 저장한다.

모드:
  --list              저장된 프로필 나열(비-GPU)
  --inspect UUID      저장된 ICL 프롬프트(ref_code/ref_spk_embedding/ref_text/
                      shape/dtype) 덤프 + 참조 음성 WAV 저장(ffmpeg 있는 실기)
  --stt UUID          등록 참조 음성을 STT로 받아써 ref_text 일치 검증(옵트인)
  --gpu UUID          RTX 실기 실험 배터리(아래 실험 1~8). 요구 조건이 없으면
                      NOT RUN으로 정확히 표시하고 임의 결과를 만들지 않는다.

GPU 실험(각 생성 직후 diagnostics/p15_qwen_probe/<label>/01_qwen_raw_24k.wav 저장):
  1. 동일 참조 음성으로 공식 API 프롬프트 새 생성
  2. 저장 프롬프트 vs 신규 프롬프트 텐서 비교(shape/dtype/hash)
  3. 신규 프롬프트 직접 추론
  4. Auto vs Korean 생성 비교
  5. 공식 generate_voice_clone 지원 파라미터 확인(inspect.signature)
  6. temperature/top_k/top_p 하나씩 변경 비교(미지원 시 스킵 표시)
  7. 동일 대본 반복 실행 변동성(--runs)
  8. x_vector_only_mode=True 신규 프롬프트 비교

사용 예(RTX 실기):
  python scripts/p15_qwen_probe.py --inspect <uuid>
  python scripts/p15_qwen_probe.py --gpu <uuid> --runs 3
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from voice_studio.core import config
from voice_studio.core.errors import ProfileError
from voice_studio.infra.profile_repository import ProfileRepository
from voice_studio.infra.qwen_adapter import MODEL_ID, RealQwenAdapter, VoiceClonePromptSpec
from voice_studio.services.audio_diagnostics import _write_wav_f32, read_wav_f32

DEFAULT_OUT = ROOT / "diagnostics" / "p15_qwen_probe"
SCRIPT_TEXT = "안녕하세요 서정민입니다 안녕하세요 서정민입니다"


def silence_report(pcm: np.ndarray, sample_rate: int, threshold: float = 1e-3) -> dict:
    """원본 PCM의 길이·선행/후행 무음 길이. 잘림·긴 침묵 진단의 1차 수치."""
    x = np.asarray(pcm, dtype=np.float32).reshape(-1)
    loud = np.abs(x) > threshold
    n = int(x.size)
    if not loud.any():
        return {"samples": n, "seconds": n / sample_rate, "leading_silence_s": n / sample_rate,
                "trailing_silence_s": n / sample_rate, "rms": 0.0}
    first = int(np.argmax(loud))
    last = int(n - 1 - np.argmax(loud[::-1]))
    return {"samples": n, "seconds": round(n / sample_rate, 3),
            "leading_silence_s": round(first / sample_rate, 3),
            "trailing_silence_s": round((n - 1 - last) / sample_rate, 3),
            "rms": round(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), 5)}


def save_raw(label: str, pcm: np.ndarray, sample_rate: int, out_dir: Path = DEFAULT_OUT) -> Path:
    """실험 1개 결과 원본 WAV를 즉시 무손실 저장하고, 읽기 검증까지 한다."""
    d = out_dir / label
    d.mkdir(parents=True, exist_ok=True)
    path = d / "01_qwen_raw_24k.wav"
    _write_wav_f32(path, pcm, sample_rate)
    back, sr = read_wav_f32(path)
    if sr != int(sample_rate) or back.size != np.asarray(pcm).size:
        raise RuntimeError(f"WAV 저장 검증 실패: {path}")
    return path


def fingerprint(code: np.ndarray | None, emb: np.ndarray | None) -> dict:
    """텐서 비교용 지문(shape/dtype/range/hash). 값 자체는 크지 않게 요약만."""
    def one(a: np.ndarray | None) -> dict | None:
        if a is None:
            return None
        try:
            a = np.asarray(a)
            if a.dtype.kind not in "iuf":
                raise ValueError
        except (TypeError, ValueError):
            # json blob(fake/legacy ref_code)은 숫자 요약 대신 repr 기록.
            return {"repr": repr(a)[:120],
                    "sha1": hashlib.sha1(repr(a).encode()).hexdigest()[:16]}
        if a.size == 0:
            return {"shape": list(a.shape), "dtype": str(a.dtype), "empty": True}
        return {"shape": list(a.shape), "dtype": str(a.dtype),
                "min": float(a.min()), "max": float(a.max()),
                "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]}
    return {"ref_code": one(code), "ref_spk_embedding": one(emb)}


def tensor_diff(fp_a: dict, fp_b: dict) -> list[str]:
    """두 프롬프트 지문의 차이 요약. 동일하면 빈 리스트."""
    out = []
    for key in ("ref_code", "ref_spk_embedding"):
        a, b = fp_a.get(key), fp_b.get(key)
        if a != b:
            out.append(f"{key}: {json.dumps(a, ensure_ascii=False)} vs {json.dumps(b, ensure_ascii=False)}")
    return out


def load_spec(repo: ProfileRepository, uuid: str) -> tuple:
    profile = repo.get(uuid)
    tensors = repo.load_prompt_tensors(uuid)
    spec = VoiceClonePromptSpec(
        ref_code=tensors["ref_code"], ref_spk_embedding=tensors["ref_spk_embedding"],
        x_vector_only_mode=profile.x_vector_only_mode, icl_mode=profile.icl_mode,
        ref_text=profile.ref_text)
    return profile, tensors, spec


# ---------------------------------------------------------------- 비-GPU 모드

def cmd_list(repo: ProfileRepository) -> int:
    profiles = repo.list_profiles()
    if not profiles:
        print("저장된 프로필이 없습니다.")
        return 0
    for p in profiles:
        print(json.dumps({"uuid": p.uuid, "name": p.name, "model_id": p.model_id,
                          "ref_text": p.ref_text, "ref_code_kind": p.ref_code_kind,
                          "x_vector_only_mode": p.x_vector_only_mode,
                          "icl_mode": p.icl_mode,
                          "reference_duration_ms": p.reference_duration_ms},
                         ensure_ascii=False))
    return 0


def cmd_inspect(repo: ProfileRepository, uuid: str, out_dir: Path) -> int:
    profile, tensors, spec = load_spec(repo, uuid)
    report = {
        "uuid": profile.uuid, "name": profile.name,
        "model_id": profile.model_id, "qwen_tts_version": profile.qwen_tts_version,
        "ref_text": profile.ref_text,
        "x_vector_only_mode": profile.x_vector_only_mode, "icl_mode": profile.icl_mode,
        "prompt_fingerprint": fingerprint(tensors.get("ref_code"), tensors.get("ref_spk_embedding")),
        "reference_duration_ms": profile.reference_duration_ms,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    flac = repo.path_for(uuid) / "reference.flac"
    if flac.is_file():
        try:
            from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
            adapter = RealFfmpegAdapter()
            pcm = adapter.decode_segment(str(flac), 0.0, 1e9, config.REFERENCE_SAMPLE_RATE).astype(np.float32)
            path = save_raw("reference_from_profile", pcm, config.REFERENCE_SAMPLE_RATE, out_dir)
            print(f"reference.wav 저장: {path}")
            print("reference silence:", json.dumps(silence_report(pcm, config.REFERENCE_SAMPLE_RATE)))
        except Exception as exc:
            print(f"참조 FLAC 디코딩 실패(ffmpeg 필요): {type(exc).__name__}: {exc} [NOT RUN]")
    else:
        print("reference.flac 없음 [NOT RUN]")
    return 0


def cmd_stt(repo: ProfileRepository, uuid: str) -> int:
    """등록 참조 음성을 받아써 ref_text와 일치하는지 검증(옵트인, CPU faster-whisper)."""
    profile, _, _ = load_spec(repo, uuid)
    try:
        from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
        adapter = RealFfmpegAdapter()
        pcm = adapter.decode_segment(str(repo.path_for(uuid) / "reference.flac"), 0.0, 1e9,
                                     config.REFERENCE_SAMPLE_RATE).astype(np.float32)
    except Exception as exc:
        print(f"참조 음성 디코딩 실패: {type(exc).__name__}: {exc} [NOT RUN]")
        return 3
    try:
        from voice_studio.services.transcription_service import FasterWhisperTranscriber
        heard = FasterWhisperTranscriber().transcribe(pcm, config.REFERENCE_SAMPLE_RATE)
    except Exception as exc:
        print(f"STT 실패: {type(exc).__name__}: {exc} [NOT RUN]")
        return 3
    stored = profile.ref_text.strip()
    match = heard.strip() == stored
    print(json.dumps({"heard": heard, "stored_ref_text": stored, "exact_match": match},
                     ensure_ascii=False, indent=2))
    return 0 if match else 4


# ---------------------------------------------------------------- GPU 배터리

def _gpu_adapter(model_path: str | None) -> RealQwenAdapter:
    return RealQwenAdapter(model_path=model_path, allow_repo_fallback=model_path is None)


def _gen_and_save(adapter: RealQwenAdapter, label: str, spec: VoiceClonePromptSpec, text: str,
                  language: str | None, extra: dict | None = None, out_dir: Path = DEFAULT_OUT) -> dict:
    """공식 모델로 직접 생성(샘플링 파라미터 전달 가능) + 즉시 원본 WAV 저장."""
    from voice_studio.infra.qwen_adapter import _from_numpy
    from qwen_tts import VoiceClonePromptItem
    item = VoiceClonePromptItem(
        ref_code=_from_numpy(spec.ref_code), ref_spk_embedding=_from_numpy(spec.ref_spk_embedding),
        x_vector_only_mode=spec.x_vector_only_mode, icl_mode=spec.icl_mode,
        ref_text=spec.ref_text or None)
    kwargs = dict(extra or {})
    wavs, out_sr = adapter._model.generate_voice_clone(
        text, language=language, voice_clone_prompt=[item], **kwargs)
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    path = save_raw(label, wav, int(out_sr), out_dir)
    rep = {"label": label, "language": language, "extra": kwargs, "sample_rate": int(out_sr),
           "wav": str(path), "silence": silence_report(wav, int(out_sr))}
    print(json.dumps(rep, ensure_ascii=False))
    return rep


def cmd_gpu(repo: ProfileRepository, uuid: str, out_dir: Path, model_path: str | None,
            runs: int) -> int:
    profile, tensors, stored_spec = load_spec(repo, uuid)
    try:
        adapter = _gpu_adapter(model_path)
    except Exception as exc:
        print(f"모델 로드 실패: {type(exc).__name__}: {exc} [NOT RUN]")
        return 3

    # (1) 동일 참조 음성으로 공식 API 프롬프트 새 생성
    ref_pcm = repo.load_reference_pcm(uuid, config.REFERENCE_SAMPLE_RATE)
    fresh = adapter.create_prompt(ref_pcm, config.REFERENCE_SAMPLE_RATE, profile.ref_text)

    # (2) 저장 프롬프트 vs 신규 프롬프트 비교
    fp_stored = fingerprint(tensors.get("ref_code"), tensors.get("ref_spk_embedding"))
    fp_fresh = fingerprint(fresh.ref_code, fresh.ref_spk_embedding)
    print(json.dumps({"step": "prompt_compare", "stored": fp_stored, "fresh": fp_fresh,
                      "diff": tensor_diff(fp_stored, fp_fresh)}, ensure_ascii=False))

    # (3) 신규 프롬프트 직접 추론 + (4) Auto vs Korean (저장 프롬프트 기준)
    try:
        _gen_and_save(adapter, "03_fresh_prompt_auto", fresh, SCRIPT_TEXT, None, out_dir=out_dir)
        _gen_and_save(adapter, "04_stored_auto", stored_spec, SCRIPT_TEXT, None, out_dir=out_dir)
        _gen_and_save(adapter, "04_stored_korean", stored_spec, SCRIPT_TEXT, "Korean", out_dir=out_dir)
        _gen_and_save(adapter, "04_fresh_korean", fresh, SCRIPT_TEXT, "Korean", out_dir=out_dir)
    except Exception as exc:
        print(f"생성 실패: {type(exc).__name__}: {exc} [NOT RUN 이후 단계]")
        return 3

    # (5) 공식 API 지원 파라미터 확인
    import inspect
    try:
        sig = inspect.signature(adapter._model.generate_voice_clone)
        supported = [p.name for p in sig.parameters.values()
                     if p.name not in ("self", "text", "language", "voice_clone_prompt")]
        print(json.dumps({"step": "supported_params", "supported": supported}, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        supported = []
        print(f"signature 확인 실패: {exc} [NOT RUN]")

    # (6) temperature/top_k/top_p 하나씩 변경(지원하는 것만)
    for name, values in (("temperature", [0.7, 1.3]), ("top_k", [20, 50]), ("top_p", [0.8, 0.95])):
        if name not in supported:
            print(f"{name}: 공식 시그니처에 없음 [SKIP]")
            continue
        for value in values:
            try:
                _gen_and_save(adapter, f"06_{name}_{str(value).replace('.', '_')}", stored_spec,
                              SCRIPT_TEXT, None, {name: value}, out_dir)
            except TypeError as exc:
                print(f"{name}={value} 미지원: {exc} [SKIP]")

    # (7) 동일 대본 반복 변동성
    durations = []
    for i in range(max(1, runs)):
        rep = _gen_and_save(adapter, f"07_repeat_{i + 1}", stored_spec, SCRIPT_TEXT, None, out_dir=out_dir)
        durations.append(rep["silence"]["seconds"])
    if len(set(durations)) > 1 or runs > 1:
        print(json.dumps({"step": "variability", "durations_s": durations}, ensure_ascii=False))

    # (8) x_vector_only_mode=True 신규 프롬프트 비교
    try:
        xv_items = adapter._model.create_voice_clone_prompt(
            ref_audio=(ref_pcm, config.REFERENCE_SAMPLE_RATE), ref_text=profile.ref_text,
            x_vector_only_mode=True)
        item = xv_items[0]
        from voice_studio.infra.qwen_adapter import tensor_to_numpy
        xv_spec = VoiceClonePromptSpec(
            ref_code=tensor_to_numpy(item.ref_code), ref_spk_embedding=tensor_to_numpy(item.ref_spk_embedding),
            x_vector_only_mode=True, icl_mode=bool(item.icl_mode), ref_text=profile.ref_text)
        print(json.dumps({"step": "xvector_prompt", "fingerprint": fingerprint(xv_spec.ref_code, xv_spec.ref_spk_embedding)},
                         ensure_ascii=False))
        _gen_and_save(adapter, "08_xvector_auto", xv_spec, SCRIPT_TEXT, None, out_dir=out_dir)
    except Exception as exc:
        print(f"x_vector_only 실험 실패: {type(exc).__name__}: {exc} [NOT RUN]")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", metavar="UUID")
    ap.add_argument("--stt", metavar="UUID")
    ap.add_argument("--gpu", metavar="UUID")
    ap.add_argument("--model-path", default=None, help="로컬 스냅샷 경로. 생략 시 HF repo fallback")
    ap.add_argument("--runs", type=int, default=3, help="변동성 실험 반복 횟수")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    repo = ProfileRepository()
    out_dir = Path(args.out)
    if args.list:
        return cmd_list(repo)
    if args.inspect:
        return cmd_inspect(repo, args.inspect, out_dir)
    if args.stt:
        return cmd_stt(repo, args.stt)
    if args.gpu:
        return cmd_gpu(repo, args.gpu, out_dir, args.model_path, args.runs)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
