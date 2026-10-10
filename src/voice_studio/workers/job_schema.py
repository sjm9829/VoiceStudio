"""worker job 스키마 파서/검증기.

메인 프로세스가 완성한 job dict를 worker가 실행 전에 검증한다.
누락 필드는 KeyError traceback이 아니라 사용자용 오류 코드로 변환된다.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

JOB_SCHEMA_VERSION = 1

class JobSchemaError(Exception):
    """job JSON이 계약을 만족하지 않을 때. code는 worker 이벤트로 전달된다."""

    def __init__(self, code: str, user_message: str, detail: str = ""):
        super().__init__(f"{code}: {detail or user_message}")
        self.code = code
        self.user_message = user_message
        self.detail = detail

COMMON_FIELDS = ("job_id", "mode", "profile_uuid", "profile_dir", "model_path")
# P17-C: tts_backend은 optional("official" 기본). gguf면 GGUF 백엔드로 worker가 분기한다.
BACKEND_OFFICIAL = "official"
BACKEND_GGUF = "gguf"
NARRATE_FIELDS = ("segments", "gap_flags", "bitrate_kbps", "output_path")
REGISTER_FIELDS = ("name", "source_path", "start_s", "end_s", "ref_text")

@dataclass
class BaseJob:
    job_id: str
    mode: str
    profile_uuid: str
    profile_dir: str
    model_path: str
    tts_backend: str = BACKEND_OFFICIAL
    schema_version: int = JOB_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class NarrateJob(BaseJob):
    segments: list[str] = field(default_factory=list)
    gap_flags: list[bool] = field(default_factory=list)
    bitrate_kbps: int = 192
    output_path: str = ""

@dataclass
class RegisterJob(BaseJob):
    name: str = ""
    source_path: str = ""
    start_s: float = 0.0
    end_s: float = 0.0
    ref_text: str = ""

def _require(data: dict, key: str, types: tuple[type, ...], job: str) -> Any:
    if key not in data or data[key] is None:
        raise JobSchemaError("E_JOB_MISSING_FIELD", f"작업 정보가 불완전합니다({key}).", f"missing field: {key}")
    if not isinstance(data[key], types):
        raise JobSchemaError("E_JOB_BAD_VALUE", f"작업 정보 값이 올바르지 않습니다({key}).",
                             f"field {key}: expected {types}, got {type(data[key]).__name__}")
    return data[key]

def parse_job(data: dict) -> NarrateJob | RegisterJob:
    """job dict를 검증해 타입 job 객체로 변환한다."""
    if not isinstance(data, dict):
        raise JobSchemaError("E_JOB_INVALID", "작업 파일을 읽을 수 없습니다.", "job root is not an object")
    version = int(data.get("schema_version", 0))
    if version != JOB_SCHEMA_VERSION:
        raise JobSchemaError("E_JOB_VERSION", "프로그램 버전이 맞지 않습니다. 다시 시도해 주세요.",
                             f"schema_version={version}")
    mode = _require(data, "mode", (str,), "job")
    if mode not in ("register", "narrate"):
        raise JobSchemaError("E_JOB_MODE", f"알 수 없는 작업 모드: {mode}", f"mode={mode}")
    base_kwargs = {}
    for key in COMMON_FIELDS:
        if key == "mode":
            continue
        if key == "profile_uuid" and mode == "register" and not data.get("profile_uuid"):
            base_kwargs[key] = ""  # 등록 job: uuid는 worker가 새로 만든다
        else:
            base_kwargs[key] = _require(data, key, (str,), mode)
    backend = data.get("tts_backend", BACKEND_OFFICIAL)
    if backend not in (BACKEND_OFFICIAL, BACKEND_GGUF):
        raise JobSchemaError("E_JOB_BAD_VALUE", "음성 백엔드 값이 올바르지 않습니다.",
                             f"tts_backend={backend}")
    base_kwargs["tts_backend"] = backend
    base_kwargs["mode"] = mode
    if mode == "narrate":
        segments = _require(data, "segments", (list,), mode)
        if not all(isinstance(s, str) and s.strip() for s in segments):
            raise JobSchemaError("E_JOB_BAD_VALUE", "대본 구간이 올바르지 않습니다.", "segments contain non-string/empty")
        flags = _require(data, "gap_flags", (list,), mode)
        if len(flags) != len(segments) or not all(isinstance(f, bool) for f in flags):
            raise JobSchemaError("E_JOB_BAD_VALUE", "구간 간격 정보가 대본과 맞지 않습니다.",
                                 f"gap_flags len={len(flags)} segments len={len(segments)}")
        bitrate = _require(data, "bitrate_kbps", (int,), mode)
        if bitrate not in (128, 192, 256):
            raise JobSchemaError("E_JOB_BAD_VALUE", "지원하지 않는 음질 값입니다.", f"bitrate_kbps={bitrate}")
        return NarrateJob(segments=list(segments), gap_flags=list(flags), bitrate_kbps=bitrate,
                          output_path=_require(data, "output_path", (str,), mode), **base_kwargs)
    start = _require(data, "start_s", (int, float), mode)
    end = _require(data, "end_s", (int, float), mode)
    if not (0 <= start < end):
        raise JobSchemaError("E_JOB_BAD_VALUE", "선택 구간이 올바르지 않습니다.", f"range {start}-{end}")
    return RegisterJob(start_s=float(start), end_s=float(end),
                       name=_require(data, "name", (str,), mode),
                       source_path=_require(data, "source_path", (str,), mode),
                       ref_text=_require(data, "ref_text", (str,), mode), **base_kwargs)

def build_narrate_payload(*, job_id: str, profile_uuid: str, profile_dir: str, model_path: str,
                          segments: list[str], gap_flags: list[bool], bitrate_kbps: int,
                          output_path: str, tts_backend: str = BACKEND_OFFICIAL) -> dict[str, Any]:
    """메인 프로세스가 worker에 넘기는 narrate job을 완성한다."""
    return {"schema_version": JOB_SCHEMA_VERSION, "job_id": job_id, "mode": "narrate",
            "profile_uuid": profile_uuid, "profile_dir": profile_dir, "model_path": model_path,
            "tts_backend": tts_backend,
            "segments": list(segments), "gap_flags": list(gap_flags),
            "bitrate_kbps": int(bitrate_kbps), "output_path": output_path}

def build_register_payload(*, job_id: str, profile_dir: str, model_path: str,
                           name: str, source_path: str, start_s: float, end_s: float,
                           ref_text: str, profile_uuid: str = "",
                           tts_backend: str = BACKEND_OFFICIAL) -> dict[str, Any]:
    """메인 프로세스가 worker에 넘기는 register job을 완성한다.

    profile_uuid는 등록 전에는 없으므로 생략한다(worker가 새 uuid를 만들어 저장).
    """
    return {"schema_version": JOB_SCHEMA_VERSION, "job_id": job_id, "mode": "register",
            "profile_uuid": profile_uuid, "profile_dir": profile_dir, "model_path": model_path,
            "tts_backend": tts_backend,
            "name": name, "source_path": source_path, "start_s": float(start_s),
            "end_s": float(end_s), "ref_text": ref_text}
