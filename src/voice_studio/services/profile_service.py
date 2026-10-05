"""목소리 프로필 응용 서비스. 등록/편집/삭제/중복 확인/동의 검증을 담당한다."""

from __future__ import annotations
import numpy as np
from ..core.errors import (DuplicateNameError, ConsentRequiredError, TranscriptRequiredError,
                           ProfileError)
from ..domain.voice_profile import VoiceProfile
from ..infra.profile_repository import ProfileRepository
from ..infra.qwen_adapter import VoiceClonePromptSpec, QwenAdapter
from .audio_service import AudioService

class ProfileService:
    def __init__(self, repository: ProfileRepository, audio: AudioService, qwen: QwenAdapter):
        self.repository = repository
        self.audio = audio
        self.qwen = qwen

    def list_profiles(self) -> list[VoiceProfile]:
        return self.repository.list_profiles()

    def get(self, uuid: str) -> VoiceProfile:
        return self.repository.get(uuid)

    def name_exists(self, name: str, *, exclude_uuid: str | None = None) -> bool:
        for p in self.repository.list_profiles():
            if p.name == name and p.uuid != exclude_uuid:
                return True
        return False

    def register(self, *, name: str, source_path: str, start_s: float, end_s: float,
                 ref_text: str, consent: bool, prompt: VoiceClonePromptSpec | None = None,
                 waveform: np.ndarray | None = None, sample_rate: int | None = None) -> VoiceProfile:
        """등록. prompt가 없으면 참조 구간을 어댑터로 분석해 만든다(별도 worker 프로세스에서 호출됨)."""
        if not consent:
            raise ConsentRequiredError()
        if not name.strip():
            raise ProfileError("목소리 이름을 입력해 주세요.")
        if self.name_exists(name.strip()):
            raise DuplicateNameError(name)
        if not ref_text.strip():
            raise TranscriptRequiredError()
        pcm = waveform if waveform is not None else self.audio.decode_reference_segment(
            source_path, start_s, end_s)
        sr = sample_rate or 24000
        spec = prompt or self.qwen.create_prompt(pcm, sr, ref_text.strip())
        profile = VoiceProfile(
            name=name.strip(), ref_text=spec.ref_text,
            reference_duration_ms=int(len(pcm) / sr * 1000),
            model_id=self.qwen.model_id, qwen_tts_version=self.qwen.model_version,
            x_vector_only_mode=spec.x_vector_only_mode, icl_mode=spec.icl_mode)
        tensors = {"ref_code": _encode_ref_code(spec.ref_code),
                   "ref_spk_embedding": spec.ref_spk_embedding}
        self.repository.save(profile, tensors, self._reference_bytes(source_path, start_s, end_s))
        return profile

    def _reference_bytes(self, path: str, start_s: float, end_s: float) -> bytes:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            out = tmp.name
        try:
            self.audio.save_reference_flac(path, start_s, end_s, out)
            return open(out, "rb").read()
        finally:
            try: os.unlink(out)
            except OSError: pass

    def rename(self, uuid: str, new_name: str) -> VoiceProfile:
        profile = self.repository.get(uuid)
        if not new_name.strip():
            raise ProfileError("목소리 이름을 입력해 주세요.")
        if self.name_exists(new_name.strip(), exclude_uuid=uuid):
            raise DuplicateNameError(new_name)
        profile.name = new_name.strip()
        from ..domain.voice_profile import _now_iso
        profile.updated_at = _now_iso()
        self.repository.update_metadata(profile)
        return profile

    def delete(self, uuid: str) -> None:
        self.repository.delete(uuid)

    def load_prompt_spec(self, uuid: str) -> VoiceClonePromptSpec:
        """저장된 프로필에서 공식 VoiceClonePromptItem 재구성용 spec을 복원한다."""
        profile = self.repository.get(uuid)
        tensors = self.repository.load_prompt_tensors(uuid)
        code = _decode_ref_code(tensors["ref_code"])
        return VoiceClonePromptSpec(
            ref_code=code, ref_spk_embedding=tensors["ref_spk_embedding"],
            x_vector_only_mode=profile.x_vector_only_mode, icl_mode=profile.icl_mode,
            ref_text=profile.ref_text)

def _encode_ref_code(code) -> np.ndarray:
    """직렬화 가능한 ref_code를 safetensors 저장용 float32 배열로 변환한다."""
    import json
    blob = json.dumps({"code": code}, ensure_ascii=False).encode("utf-8")
    arr = np.frombuffer(blob, dtype=np.uint8).astype(np.float32)
    return np.concatenate([np.array([arr.size], dtype=np.float32), arr])

def _decode_ref_code(arr: np.ndarray):
    import json
    arr = np.asarray(arr, dtype=np.float32)
    n = int(arr[0])
    blob = arr[1:1 + n].astype(np.uint8).tobytes()
    return json.loads(blob.decode("utf-8"))["code"]
