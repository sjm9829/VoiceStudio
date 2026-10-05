"""설정 영속화(JSON). 사용자가 볼 수 없는 내부 설정만 담는다."""

from __future__ import annotations
import json
from pathlib import Path
from ..core import config
from ..core.paths import settings_file, default_mp3_dir

DEFAULTS = {
    "mp3_output_dir": str(default_mp3_dir()),
    "mp3_bitrate_kbps": config.MP3_BITRATE_KBPS_DEFAULT,
    "gpu_enabled": True,
    "release_gpu_after_use": True,   # v1 고정 ON
    "model_id": config.DEFAULT_MODEL_ID,
    "stt_model_id": config.STT_MODEL_ID,
}

class SettingsRepository:
    def __init__(self, path: Path | None = None):
        self.path = path or settings_file()

    def load(self) -> dict:
        data = dict(DEFAULTS)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data.update({k: raw[k] for k in DEFAULTS if k in raw})
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError):
            pass  # 손상 시 기본값으로 복구
        return data

    def save(self, data: dict) -> None:
        clean = {k: data.get(k, DEFAULTS[k]) for k in DEFAULTS}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
