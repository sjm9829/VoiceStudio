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
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from voice_studio.core import config
from voice_studio.core.errors import ProfileError, UnsupportedAudioError
from voice_studio.infra.profile_repository import ProfileRepository
from voice_studio.infra.qwen_adapter import production_device
from voice_studio.services.model_manager import ModelManager
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
    if path.exists():
        # 이전 진단 WAV를 조용히 덮어쓰지 않는다: 같은 라벨 재실행 시 번호를 붙여 보존한다.
        n = 2
        while (d / f"01_qwen_raw_24k_{n}.wav").exists():
            n += 1
        path = d / f"01_qwen_raw_24k_{n}.wav"
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

def _probe_ffmpeg_adapter():
    """프로브 디코딩 전용 FFmpeg. 시스템 PATH에 의존하지 않고 repo third_party/bin을 명시 사용."""
    from voice_studio.infra.ffmpeg_adapter import RealFfmpegAdapter
    repo_bin = ROOT / "third_party" / "bin"
    return RealFfmpegAdapter(ffmpeg=str(repo_bin / "ffmpeg.exe"),
                             ffprobe=str(repo_bin / "ffprobe.exe"))


def load_reference_via_probe(repo: ProfileRepository, uuid: str) -> tuple[np.ndarray, dict]:
    """참조 음성 로딩을 --inspect/--stt/--gpu 공용 경로로 통일한다.

    repo third_party/bin FFmpeg만 사용하고, ffprobe로 실측 duration을 구한 뒤
    그 구간을 REFERENCE_SAMPLE_RATE float32로 디코딩한다. 실측 길이와 디코딩
    샘플 수가 맞지 않으면 즉시 오류로 실패시켜 부정확한 참조 PCM이 실험에
    유입되지 않게 한다.
    """
    flac = repo.path_for(uuid) / "reference.flac"
    if not flac.is_file():
        raise ProfileError(f"참조 음성 파일이 없습니다: {flac}")
    adapter = _probe_ffmpeg_adapter()
    info = adapter.probe(str(flac))
    duration = float(info["duration"])
    if duration <= 0:
        raise UnsupportedAudioError(f"ffprobe duration이 비정상입니다: {duration}")
    pcm = adapter.decode_segment(str(flac), 0.0, duration,
                                 config.REFERENCE_SAMPLE_RATE).astype(np.float32)
    expected = int(round(duration * config.REFERENCE_SAMPLE_RATE))
    tol = int(round(config.REFERENCE_SAMPLE_RATE * 0.05))
    if abs(int(pcm.size) - expected) > tol:
        raise UnsupportedAudioError(
            f"디코딩 길이 불일치: ffprobe {duration:.3f}s(기대 {expected} samples) vs 실제 {pcm.size} samples")
    meta = {"duration_s": round(duration, 3), "probe_sample_rate": int(info["sample_rate"]),
            "probe_codec": info.get("codec", ""), "decoded_samples": int(pcm.size),
            "decoded_sample_rate": int(config.REFERENCE_SAMPLE_RATE)}
    return pcm, meta


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
    try:
        pcm, ref_meta = load_reference_via_probe(repo, uuid)
        print("reference probe:", json.dumps(ref_meta, ensure_ascii=False))
        path = save_raw("reference_from_profile", pcm, config.REFERENCE_SAMPLE_RATE, out_dir)
        print(f"reference.wav 저장: {path}")
        print("reference silence:", json.dumps(silence_report(pcm, config.REFERENCE_SAMPLE_RATE)))
    except Exception as exc:
        print(f"참조 음성 검증·디코딩 실패(ffmpeg/파일 필요): {type(exc).__name__}: {exc} [NOT RUN]")
    return 0


def cmd_stt(repo: ProfileRepository, uuid: str) -> int:
    """등록 참조 음성을 받아써 ref_text와 일치하는지 검증(옵트인, CPU faster-whisper)."""
    profile, _, _ = load_spec(repo, uuid)
    try:
        pcm, ref_meta = load_reference_via_probe(repo, uuid)
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
    """GPU 프로브 전용 어댑터. production 정책을 그대로 따른다.

    - device는 production_device()(CUDA 없으면 GpuUnavailableError, CPU fallback 없음).
      dtype은 RealQwenAdapter 내부 preferred_dtype(FP16/BF16) 정책을 그대로 사용.
    - model_path 미지정 시 HF repo로 자동 다운로드하지 않고, 앱이 이미 받아둔
      ModelManager 로컬 스냅샷을 사용한다(없으면 ModelNotDownloadedError → NOT RUN).
    """
    if model_path is None:
        model_path = ModelManager().model_path()
    return RealQwenAdapter(model_path=model_path, device=production_device())


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
    t0 = time.monotonic()
    wavs, out_sr = adapter._model.generate_voice_clone(
        text, language=language, voice_clone_prompt=[item], **kwargs)
    elapsed = round(time.monotonic() - t0, 3)
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    path = save_raw(label, wav, int(out_sr), out_dir)
    rep = {"label": label, "language": language, "extra": kwargs, "sample_rate": int(out_sr),
           "wav": str(path), "generation_s": elapsed, "silence": silence_report(wav, int(out_sr))}
    print(json.dumps(rep, ensure_ascii=False))
    return rep


def _mark(results: dict, step: str, status: str, **extra) -> None:
    """실험별 상태 기록. 동일 스텝 재기록 시 목록으로 병합한다(반복 실험 대응)."""
    entry = {"status": status, **extra}
    if step in results:
        prev = results[step]
        if isinstance(prev, list):
            prev.append(entry)
        else:
            results[step] = [prev, entry]
        return
    results[step] = entry


def supported_params_probe(model) -> dict:
    """공식 generate_voice_clone 지원 파라미터 확인.

    **kwargs만으로 전달되는 샘플링 파라미터는 시그니처에 이름이 없어도 공식 API가
    받을 수 있으므로 SKIP 판정 근거로 쓰지 않는다(실행 TypeError만 SKIP 근거).
    inspect.signature 예외 시에도 accepts_kwargs를 항상 False로 초기화해
    미정의 변수 오류를 내지 않는다.
    """
    import inspect
    try:
        sig = inspect.signature(model.generate_voice_clone)
        supported = [p.name for p in sig.parameters.values()
                     if p.name not in ("self", "text", "language", "voice_clone_prompt")]
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values())
        return {"supported": supported, "accepts_kwargs": accepts_kwargs, "error": None}
    except (TypeError, ValueError) as exc:
        return {"supported": [], "accepts_kwargs": False, "error": str(exc)}


def _gpu_adapter(model_path: str | None) -> RealQwenAdapter:
    """GPU 프로브 전용 어댑터. production 정책을 그대로 따른다.

    - device는 production_device()(CUDA 없으면 GpuUnavailableError, CPU fallback 없음).
      dtype은 RealQwenAdapter 내부 preferred_dtype(FP16/BF16) 정책을 그대로 사용.
    - model_path 미지정 시 HF repo로 자동 다운로드하지 않고, 앱이 이미 받아둔
      ModelManager 로컬 스냅샷을 사용한다(없으면 ModelNotDownloadedError → NOT RUN).
    """
    if model_path is None:
        model_path = ModelManager().model_path()
    return RealQwenAdapter(model_path=model_path, device=production_device())


REQUIRED_EXPERIMENTS = ("prompt_compare", "03_fresh_prompt_auto", "04_stored_auto",
                        "04_stored_korean", "04_fresh_korean", "repeat_variability")


def cmd_gpu(repo: ProfileRepository, uuid: str, out_dir: Path, model_path: str | None,
            runs: int) -> int:
    results: dict = {}
    profile, tensors, stored_spec = load_spec(repo, uuid)

    # FFmpeg·참조 음성 유효성을 GPU 모델 로드 전에 확인한다. 실패 시 NOT RUN과
    # 구체적 원인을 출력하고 Qwen 모델을 불필요하게 적재하지 않는다.
    try:
        ref_pcm, ref_meta = load_reference_via_probe(repo, uuid)
    except Exception as exc:
        print(f"참조 음성 검증 실패: {type(exc).__name__}: {exc} [NOT RUN]")
        return 3
    print("reference probe:", json.dumps(ref_meta, ensure_ascii=False))

    try:
        device = production_device()
        resolved_model = str(model_path) if model_path else str(ModelManager().model_path())
        adapter = _gpu_adapter(model_path)
    except Exception as exc:
        print(f"모델 로드 실패: {type(exc).__name__}: {exc} [NOT RUN]")
        return 3
    try:
        dtype = str(__import__("voice_studio.infra.qwen_adapter", fromlist=["preferred_dtype"]).preferred_dtype(device))
    except Exception:
        dtype = "n/a"

    # (1) 동일 참조 음성으로 공식 API 프롬프트 새 생성 + (2) 저장 프롬프트와 비교
    try:
        fresh = adapter.create_prompt(ref_pcm, config.REFERENCE_SAMPLE_RATE, profile.ref_text)
        fp_stored = fingerprint(tensors.get("ref_code"), tensors.get("ref_spk_embedding"))
        fp_fresh = fingerprint(fresh.ref_code, fresh.ref_spk_embedding)
        diff = tensor_diff(fp_stored, fp_fresh)
        print(json.dumps({"step": "prompt_compare", "stored": fp_stored, "fresh": fp_fresh,
                          "diff": diff}, ensure_ascii=False))
        _mark(results, "prompt_compare", "FAIL" if diff else "OK", diff=diff)
    except Exception as exc:
        print(f"프롬프트 재생성·비교 실패: {type(exc).__name__}: {exc}")
        _mark(results, "prompt_compare", "FAIL", error=f"{type(exc).__name__}: {exc}")

    # (3) 신규 프롬프트 직접 추론 + (4) Auto vs Korean (각각 독립 상태 기록)
    gen_plan = (("03_fresh_prompt_auto", fresh, None),
                ("04_stored_auto", stored_spec, None),
                ("04_stored_korean", stored_spec, "Korean"),
                ("04_fresh_korean", fresh, "Korean"))
    for label, spec, lang in gen_plan:
        try:
            rep = _gen_and_save(adapter, label, spec, SCRIPT_TEXT, lang, out_dir=out_dir)
            _mark(results, label, "OK", wav=rep["wav"], generation_s=rep["generation_s"],
                  params=rep["extra"], language=rep["language"])
        except Exception as exc:
            print(f"{label} 생성 실패: {type(exc).__name__}: {exc}")
            _mark(results, label, "FAIL", error=f"{type(exc).__name__}: {exc}")

    # (5) 공식 API 지원 파라미터 확인
    probe_res = supported_params_probe(adapter._model)
    if probe_res["error"] is None:
        print(json.dumps({"step": "supported_params",
                          "supported": probe_res["supported"],
                          "accepts_kwargs": probe_res["accepts_kwargs"]}, ensure_ascii=False))
    else:
        print(f"signature 확인 실패: {probe_res['error']} [NOT RUN]")
    supported, accepts_kwargs = probe_res["supported"], probe_res["accepts_kwargs"]

    # (6) temperature/top_k/top_p 하나씩 변경(지원하는 것만, SKIP/FAIL 구분)
    for name, values in (("temperature", [0.7, 1.3]), ("top_k", [20, 50]), ("top_p", [0.8, 0.95])):
        if name not in supported and not accepts_kwargs:
            _mark(results, f"06_{name}", "SKIP", reason="공식 시그니처에 없음(확장 없음)")
            print(f"{name}: 공식 시그니처에 없음(확장 없음) [SKIP]")
            continue
        for value in values:
            key = f"06_{name}"
            try:
                rep = _gen_and_save(adapter, f"06_{name}_{str(value).replace('.', '_')}", stored_spec,
                                    SCRIPT_TEXT, None, {name: value}, out_dir)
                _mark(results, key, "OK", wav=rep["wav"], generation_s=rep["generation_s"],
                      params={name: value})
            except TypeError as exc:
                _mark(results, key, "SKIP", reason=f"미지원: {type(exc).__name__}: {exc}")
                print(f"{name}={value} 미지원: {exc} [SKIP]")
            except Exception as exc:
                _mark(results, key, "FAIL", error=f"{type(exc).__name__}: {exc}")
                print(f"{name}={value} 생성 실패: {type(exc).__name__}: {exc}")

    # (7) 동일 대본 반복 변동성
    durations, wavs = [], []
    fail = None
    for i in range(max(1, runs)):
        try:
            rep = _gen_and_save(adapter, f"07_repeat_{i + 1}", stored_spec, SCRIPT_TEXT, None, out_dir=out_dir)
            durations.append(rep["silence"]["seconds"])
            wavs.append(rep["wav"])
        except Exception as exc:
            fail = f"{type(exc).__name__}: {exc}"
            break
    if durations:
        _mark(results, "repeat_variability", "OK", durations_s=durations, wavs=wavs)
        print(json.dumps({"step": "variability", "durations_s": durations}, ensure_ascii=False))
    else:
        _mark(results, "repeat_variability", "FAIL", error=fail or "no run")

    # (8) x_vector_only_mode=True 신규 프롬프트 비교(참고 실험)
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
        rep = _gen_and_save(adapter, "08_xvector_auto", xv_spec, SCRIPT_TEXT, None, out_dir=out_dir)
        _mark(results, "08_xvector_auto", "OK", wav=rep["wav"], generation_s=rep["generation_s"])
    except Exception as exc:
        print(f"x_vector_only 실험 실패: {type(exc).__name__}: {exc}")
        _mark(results, "08_xvector_auto", "FAIL", error=f"{type(exc).__name__}: {exc}")

    # 요약: 필수 실험 실패는 전체 성공으로 표시하지 않는다.
    failed_required = [s for s in REQUIRED_EXPERIMENTS
                       if (lambda e: e.get("status") == "FAIL" if isinstance(e, dict) else False)(results.get(s, {}))]
    overall = "SUCCESS" if not failed_required else "FAILED"
    print(json.dumps({"step": "summary", "overall": overall, "failed_required": failed_required,
                      "experiments": results, "model_path": resolved_model,
                      "device": str(device), "dtype": dtype}, ensure_ascii=False, indent=2))
    return 4 if failed_required else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", metavar="UUID")
    ap.add_argument("--stt", metavar="UUID")
    ap.add_argument("--gpu", metavar="UUID")
    ap.add_argument("--model-path", default=None, help="로컬 모델 스냅샷 경로. 생략 시 앱이 받아둔 ModelManager 로컬 스냅샷 사용(HF 자동 다운로드 없음)")
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
