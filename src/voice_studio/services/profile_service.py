"""목소리 프로필 응용 서비스. 등록/편집/삭제/중복 확인/동의 검증을 담당한다."""

from __future__ import annotations
import numpy as np
from ..core import config
from ..core.errors import (DuplicateNameError, ConsentRequiredError, TranscriptRequiredError,
                           ProfileError)
from ..domain.voice_profile import VoiceProfile
from ..infra.profile_repository import ProfileRepository
from ..infra.qwen_adapter import VoiceClonePromptSpec, QwenAdapter
from .audio_service import AudioService

class ProfileService:
    def __init__(self, repository: ProfileRepository, audio: AudioService,
                 qwen: QwenAdapter | None = None):
        """qwen은 dev 컨텍스트/worker 내부 등록 경로에서만 필요하다. production UI 프로세스는 None."""
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
        if prompt is None and self.qwen is None:
            raise ProfileError("등록은 별도 worker 프로세스에서 실행해야 합니다.")
        spec = prompt or self.qwen.create_prompt(pcm, sr, ref_text.strip())  # type: ignore[union-attr]
        from ..infra.gguf_adapter import GgufPromptSpec
        backend = "gguf" if isinstance(spec, GgufPromptSpec) else "official"
        profile = VoiceProfile(
            name=name.strip(), ref_text=spec.ref_text,
            reference_duration_ms=int(len(pcm) / sr * 1000),
            model_id=getattr(self.qwen, "model_id", config.DEFAULT_MODEL_ID) if self.qwen else config.DEFAULT_MODEL_ID,
            qwen_tts_version=getattr(self.qwen, "model_version", "official-base") if self.qwen else "official-base",
            tts_backend=backend,
            x_vector_only_mode=spec.x_vector_only_mode, icl_mode=spec.icl_mode,
            ref_code_kind=ref_code_kind_of(spec.ref_code))
        tensors = {"ref_code": encode_ref_code(spec.ref_code),
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
        code = decode_ref_code(tensors["ref_code"], profile.ref_code_kind)
        return VoiceClonePromptSpec(
            ref_code=code, ref_spk_embedding=tensors["ref_spk_embedding"],
            x_vector_only_mode=profile.x_vector_only_mode, icl_mode=profile.icl_mode,
            ref_text=profile.ref_text)

def ref_code_kind_of(code) -> str:
    """ref_code의 실제 종류를 판별한다. 실제 텐서/숫자 배열이면 "tensor"를 유지한다."""
    try:
        import torch
    except ImportError:
        torch = None  # UI 프로세스/테스트 환경에는 torch가 없을 수 있다
    if torch is not None and isinstance(code, torch.Tensor):
        return "tensor"
    if isinstance(code, np.ndarray) and np.issubdtype(code.dtype, np.number) and code.ndim >= 1:
        return "tensor"
    return "json"


def encode_ref_code(code) -> np.ndarray:
    """ref_code를 실제 의미에 맞는 dtype/shape로 safetensors 저장용 배열로 변환한다.

    - 공식 Qwen ref_code(torch.Tensor, (T,) 또는 (T,Q), 정수 code) → 원래 dtype을 유지한 numpy 배열.
    - torch.Tensor/ndarray 이외(JSON 직렬화 가능한 fake 값) → 기존 float32 blob 인코딩 유지.
    """
    if isinstance(code, np.ndarray):
        if code.dtype.kind in "iuf" and code.ndim >= 1:
            return code  # 실제 텐서: 정수 code면 정수 dtype 유지
        raise ProfileError(f"ref_code 배열 dtype/shape가 올바르지 않습니다: {code.dtype}, {code.shape}")
    try:
        import torch
    except ImportError:
        torch = None  # UI 프로세스/테스트 환경에는 torch가 없을 수 있다
    if torch is not None and isinstance(code, torch.Tensor):
        arr = code.detach().cpu().numpy()
        if arr.dtype.kind not in "iuf" or arr.ndim < 1:
            raise ProfileError(f"ref_code 텐서 dtype/shape가 올바르지 않습니다: {arr.dtype}, {arr.shape}")
        return arr
    import json
    blob = json.dumps({"code": code}, ensure_ascii=False).encode("utf-8")
    arr = np.frombuffer(blob, dtype=np.uint8).astype(np.float32)
    return np.concatenate([np.array([arr.size], dtype=np.float32), arr])

def decode_ref_code(arr: np.ndarray, kind: str):
    """kind("tensor"|"json")에 따라 저장된 ref_code를 복원한다."""
    arr = np.asarray(arr)
    if kind == "tensor":
        return arr  # 원래 dtype/shape 유지
    return _decode_ref_code(arr)

_encode_ref_code = encode_ref_code  # 하위 호환 별칭

def _decode_ref_code(arr: np.ndarray):
    import json
    arr = np.asarray(arr, dtype=np.float32)
    n = int(arr[0])
    blob = arr[1:1 + n].astype(np.uint8).tobytes()
    return json.loads(blob.decode("utf-8"))["code"]
