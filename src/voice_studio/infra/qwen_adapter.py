"""Qwen3-TTS 어댑터.

공식 API 함정 대응:
- 공식 `_prompt_items_to_voice_clone_prompt()` dict에는 ref_text가 포함되지 않으므로,
  생성 시 공식 `VoiceClonePromptItem`을 ref_text와 함께 재구성해 **list로 전달**하는 경로를 사용한다.
- private API 종속은 이 어댑터 안에만 존재한다. UI/서비스 계층은 이 어댑터의 결과(JSON)만 본다.
- 실제 모델 import는 worker 프로세스에서만 발생한다(지연 import).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Protocol
import numpy as np

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

@dataclass
class VoiceClonePromptSpec:
    """프로필에 영속화되는 복제 프롬프트 데이터. 어댑터가 이것을 공식 객체로 재구성한다."""
    ref_code: Any | None = None            # 모델이 산출한 ref_code (직렬화 가능 형태)
    ref_spk_embedding: np.ndarray | None = None
    x_vector_only_mode: bool = False
    icl_mode: bool = True
    ref_text: str = ""

    def to_serializable(self) -> dict[str, Any]:
        d = asdict(self)
        if self.ref_spk_embedding is not None:
            d["ref_spk_embedding"] = self.ref_spk_embedding.astype(np.float32).tolist()
        return d

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> "VoiceClonePromptSpec":
        emb = data.get("ref_spk_embedding")
        return cls(
            ref_code=data.get("ref_code"),
            ref_spk_embedding=np.asarray(emb, dtype=np.float32) if emb is not None else None,
            x_vector_only_mode=bool(data.get("x_vector_only_mode", False)),
            icl_mode=bool(data.get("icl_mode", True)),
            ref_text=str(data.get("ref_text", "")),
        )

class QwenAdapter(Protocol):
    """worker가 사용하는 음성 모델 인터페이스."""
    model_id: str
    model_version: str

    def create_prompt(self, waveform: np.ndarray, sample_rate: int, ref_text: str) -> VoiceClonePromptSpec: ...
    def generate(self, prompt: VoiceClonePromptSpec, text: str, sample_rate: int) -> np.ndarray: ...

class RealQwenAdapter:
    """공식 Qwen3-TTS 모델을 사용하는 어댑터. worker 프로세스에서만 생성한다.

    NOTE(P00, 미검증): 공식 qwen_tts 패키지의 정확한 클래스/시그니처는
    https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
    기준이며, 실제 환경 접근이 불가해 아래 경로를 런타임에 재확인해야 한다.
    """

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as exc:
            raise ImportError("qwen_tts 패키지가 설치되어 있지 않습니다.") from exc
        self.model_id = model_id
        self.model_version = "official-base"
        self._model = Qwen3TTSModel.from_pretrained(model_id, device_map=device or "auto")

    def create_prompt(self, waveform: np.ndarray, sample_rate: int, ref_text: str) -> VoiceClonePromptSpec:
        # 공식: create_voice_clone_prompt(ref_audio=(waveform, sample_rate), ref_text=ref_text,
        #                                  x_vector_only_mode=False)
        item = self._model.create_voice_clone_prompt(
            ref_audio=(waveform, sample_rate), ref_text=ref_text, x_vector_only_mode=False)
        ref_code = getattr(item, "ref_code", None)
        emb = getattr(item, "ref_spk_embedding", None)
        if emb is not None and not isinstance(emb, np.ndarray):
            emb = np.asarray(emb, dtype=np.float32)
        return VoiceClonePromptSpec(
            ref_code=ref_code, ref_spk_embedding=emb,
            x_vector_only_mode=False, icl_mode=True, ref_text=ref_text)

    def generate(self, prompt: VoiceClonePromptSpec, text: str, sample_rate: int) -> np.ndarray:
        # 공식: generate_voice_clone([VoiceClonePromptItem(...)], text)
        # ref_text가 dict 경로에서 누락되므로 VoiceClonePromptItem을 직접 재구성해 list로 전달한다.
        from qwen_tts.core.models import VoiceClonePromptItem  # type: ignore  # private: 어댑터 내부에 한정
        item = VoiceClonePromptItem(
            ref_code=prompt.ref_code,
            ref_spk_embedding=prompt.ref_spk_embedding,
            ref_text=prompt.ref_text,
            x_vector_only_mode=prompt.x_vector_only_mode,
            icl_mode=prompt.icl_mode,
        )
        wav = self._model.generate_voice_clone([item], text)
        wav = np.asarray(wav[0] if isinstance(wav, (list, tuple)) else wav).squeeze()
        return wav.astype(np.float32)

class FakeQwenAdapter:
    """테스트/오프라인 개발용 가짜 어댑터. 사인파 나레이션을 합성한다."""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.model_version = "fake"
        self.calls: list[str] = []

    def create_prompt(self, waveform: np.ndarray, sample_rate: int, ref_text: str) -> VoiceClonePromptSpec:
        # 실제 어댑터와 동일한 계약: ref_code는 직렬화 가능해야 하고 embedding은 float32 1-D여야 한다.
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        return VoiceClonePromptSpec(
            ref_code={"fake": True, "peak": peak, "ref_text": ref_text},
            ref_spk_embedding=np.full(256, peak, dtype=np.float32),
            x_vector_only_mode=False, icl_mode=True, ref_text=ref_text)

    def generate(self, prompt: VoiceClonePromptSpec, text: str, sample_rate: int) -> np.ndarray:
        self.calls.append(text)
        seconds = max(0.4, len(text) * 0.08)
        t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        base = 0.2 * np.sin(2 * np.pi * (200 + 30 * (prompt.ref_spk_embedding[0] if prompt.ref_spk_embedding is not None else 0)) * t)
        gain = float(np.abs(prompt.ref_spk_embedding[0]) + 0.5) if prompt.ref_spk_embedding is not None else 1.0
        return (base * np.minimum(1.0, gain)).astype(np.float32)
