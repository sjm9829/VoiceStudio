"""목소리 프로필 도메인 객체."""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class VoiceProfile:
    name: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    ref_text: str = ""
    reference_duration_ms: int = 0
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    qwen_tts_version: str = "unknown"
    x_vector_only_mode: bool = False
    icl_mode: bool = True
    ref_code_kind: str = "json"  # "tensor": 실제 정수/실수 텐서, "json": JSON blob(fake/legacy)
    schema_version: int = 1

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "VoiceProfile":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)

def profile_dir(profiles_root: Path, profile_uuid: str) -> Path:
    return profiles_root / profile_uuid
