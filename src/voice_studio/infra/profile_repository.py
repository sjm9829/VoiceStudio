"""목소리 프로필 영속 저장소.

파일 구성: metadata.json / prompt.safetensors / reference.flac
- safetensors에 ref_code와 ref_spk_embedding을 저장(pickle 임의 객체 저장 금지).
- 로드 시 dtype/shape를 검증한다.
- schema_version 기반 migration 훅을 제공한다.
"""

from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Any
import numpy as np
from safetensors.numpy import save_file, load_file
from ..core.errors import ProfileError
from ..core.paths import profiles_dir, PROFILE_SCHEMA_VERSION
from ..domain.voice_profile import VoiceProfile, profile_dir

REF_CODE_KEY = "ref_code"
EMB_KEY = "ref_spk_embedding"
EXPECTED_EMB_DTYPE = np.float32

class ProfileRepository:
    def __init__(self, root: Path | None = None):
        self.root = root or profiles_dir()

    def _dir(self, profile_uuid: str) -> Path:
        d = profile_dir(self.root, profile_uuid)
        if not d.is_dir():
            raise ProfileError(f"목소리를 찾을 수 없습니다: {profile_uuid}")
        return d

    # ---- CRUD ----
    def list_profiles(self) -> list[VoiceProfile]:
        out: list[VoiceProfile] = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir()):
            meta = d / "metadata.json"
            if d.is_dir() and meta.is_file():
                try:
                    out.append(VoiceProfile.from_metadata(self._read_meta(meta)))
                except ProfileError:
                    continue
        return out

    def get(self, profile_uuid: str) -> VoiceProfile:
        return VoiceProfile.from_metadata(self._read_meta(self._dir(profile_uuid) / "metadata.json"))

    def save(self, profile: VoiceProfile, prompt_tensors: dict[str, np.ndarray], reference_flac: bytes) -> Path:
        d = profile_dir(self.root, profile.uuid)
        d.mkdir(parents=True, exist_ok=True)
        self._validate_tensors(prompt_tensors)
        save_file({k: np.ascontiguousarray(v) for k, v in prompt_tensors.items()}, d / "prompt.safetensors")
        (d / "reference.flac").write_bytes(reference_flac)
        (d / "metadata.json").write_text(
            json.dumps(self._migrate(profile.to_metadata()), ensure_ascii=False, indent=2), encoding="utf-8")
        return d

    def update_metadata(self, profile: VoiceProfile) -> None:
        d = self._dir(profile.uuid)
        (d / "metadata.json").write_text(
            json.dumps(self._migrate(profile.to_metadata()), ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, profile_uuid: str) -> None:
        d = self._dir(profile_uuid)
        shutil.rmtree(d)

    # ---- 텐서 접근 ----
    def load_prompt_tensors(self, profile_uuid: str) -> dict[str, np.ndarray]:
        path = self._dir(profile_uuid) / "prompt.safetensors"
        try:
            tensors = load_file(str(path))
        except Exception as exc:
            raise ProfileError(f"프로필 텐서를 읽을 수 없습니다: {exc}") from exc
        self._validate_tensors(tensors)
        return tensors

    def load_reference_pcm(self, profile_uuid: str, sample_rate: int) -> np.ndarray:
        """저장된 참조 FLAC을 numpy로 재생성(마이그레이션 시 사용). fake 저장소는 .npy 보조 파일을 사용."""
        from ..core import config
        d = self._dir(profile_uuid)
        npy = d / "reference.flac.npy"
        if npy.is_file():  # fake adapter가 남긴 보조 데이터
            return np.load(npy).astype(np.float32)
        adapter_pcm = self._decode_flac(d / "reference.flac", sample_rate)
        if adapter_pcm is None:
            raise ProfileError("참조 음성을 디코딩할 수 없습니다.")
        return adapter_pcm

    def _decode_flac(self, flac: Path, sample_rate: int) -> np.ndarray | None:
        from .ffmpeg_adapter import RealFfmpegAdapter
        try:
            adapter = RealFfmpegAdapter()
        except ProfileError:
            return None
        except Exception:
            return None
        pcm = adapter.decode_segment(str(flac), 0.0, 1e9, sample_rate)  # 전체 구간
        return pcm.astype(np.float32)

    # ---- 내부 ----
    def _validate_tensors(self, tensors: dict[str, np.ndarray]) -> None:
        if REF_CODE_KEY not in tensors or EMB_KEY not in tensors:
            raise ProfileError("프로필 텐서에 ref_code/ref_spk_embedding이 없습니다.")
        emb = tensors[EMB_KEY]
        if emb.dtype != EXPECTED_EMB_DTYPE:
            raise ProfileError(f"ref_spk_embedding dtype이 {EXPECTED_EMB_DTYPE}이 아닙니다: {emb.dtype}")
        if emb.ndim != 1 or emb.size == 0:
            raise ProfileError(f"ref_spk_embedding shape가 올바르지 않습니다: {emb.shape}")
        code = tensors[REF_CODE_KEY]
        if not isinstance(code, np.ndarray) or code.ndim < 1 or code.size == 0:
            raise ProfileError(f"ref_code shape가 올바르지 않습니다: {getattr(code, 'shape', code)}")
        if code.dtype.kind not in "iuf":
            raise ProfileError(f"ref_code dtype이 숫자가 아닙니다: {code.dtype}")

    def _read_meta(self, meta: Path) -> dict[str, Any]:
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProfileError("metadata.json이 없습니다.") from exc
        except json.JSONDecodeError as exc:
            raise ProfileError("metadata.json이 손상되었습니다.") from exc
        return self._migrate(data)

    def _migrate(self, data: dict[str, Any]) -> dict[str, Any]:
        v = int(data.get("schema_version", 0))
        if v > PROFILE_SCHEMA_VERSION:
            raise ProfileError(f"더 새 버전의 프로필입니다(v{v}). 프로그램을 업데이트해 주세요.")
        if v < PROFILE_SCHEMA_VERSION:
            data = self._migrate_from(data, v)
        data["schema_version"] = PROFILE_SCHEMA_VERSION
        return data

    def _migrate_from(self, data: dict[str, Any], version: int) -> dict[str, Any]:
        # v0(초기 무버전) → v1: 기본값 채우기
        if version < 1:
            data.setdefault("x_vector_only_mode", False)
            data.setdefault("icl_mode", True)
            data.setdefault("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        data.setdefault("ref_code_kind", "json")
        return data
