"""Qwen3-TTS 어댑터.

공식 API(qwen_tts) 실제 시그니처 기준(2026-02, QwenLM/Qwen3-TTS main):
- `Qwen3TTSModel.from_pretrained(repo_id_or_local_path, **kwargs)` — kwargs는 transformers AutoModel로 전달.
- `create_voice_clone_prompt(ref_audio, ref_text=None, x_vector_only_mode=False) -> List[VoiceClonePromptItem]`
  (반환은 항상 리스트)
- `generate_voice_clone(text, language=None, ..., voice_clone_prompt=[item], ...) -> (List[np.ndarray], sample_rate)`
  (반환은 (wav 리스트, 샘플레이트) 튜플)
- `VoiceClonePromptItem`는 `from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem` 공식 export.
- ref_code는 torch.Tensor((T, Q) 또는 (T,)), ref_spk_embedding은 torch.Tensor((D,)).

공식 dict 변환 경로(`_prompt_items_to_voice_clone_prompt`)에는 ref_text가 포함되지 않으므로,
프로필 저장 시 ref_code/ref_spk_embedding/ref_text/x_vector_only_mode/icl_mode를 모두 영속화하고
생성 시 공식 VoiceClonePromptItem을 재구성해 **list로 전달**한다.
private API 종속은 이 어댑터 안에만 존재하고, 실제 모델 import는 worker 프로세스에서만 발생한다.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Protocol
import numpy as np

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

def _to_numpy(value: Any) -> np.ndarray | None:
    """torch.Tensor/ndarray/list를 numpy로 변환한다. torch.Tensor는 detach().cpu() 후 변환."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch  # 어댑터 내부(worker)에서만 import
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def _from_numpy(value: np.ndarray | None) -> Any:
    """저장된 numpy 배열을 공식 API가 기대하는 torch.Tensor로 되돌린다."""
    if value is None:
        return None
    import torch  # 어댑터 내부에서만 import
    return torch.from_numpy(np.ascontiguousarray(value))

@dataclass
class VoiceClonePromptSpec:
    """프로필에 영속화되는 복제 프롬프트 데이터. 어댑터가 이것을 공식 객체로 재구성한다."""
    ref_code: Any | None = None            # 공식: torch.Tensor → 어댑터에서 numpy로 보관/저장
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

def default_device() -> str:
    """CUDA가 가능하면 cuda:0, 아니면 cpu. worker에서만 호출한다."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"

class RealQwenAdapter:
    """공식 Qwen3-TTS 모델을 사용하는 어댑터. worker 프로세스에서만 생성한다.

    model_path가 주어지면 로컬 스냅샷 경로를 from_pretrained에 그대로 사용한다(오프라인 동작).
    """

    def __init__(self, model_path: str | None = None, model_id: str = MODEL_ID,
                 device: str | None = None):
        from qwen_tts import Qwen3TTSModel  # 공식 export
        self.model_id = model_id
        self.model_version = "official-base"
        source = model_path or model_id
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device_map"] = device
        self._model = Qwen3TTSModel.from_pretrained(source, **kwargs)

    def create_prompt(self, waveform: np.ndarray, sample_rate: int, ref_text: str) -> VoiceClonePromptSpec:
        # 공식 반환은 List[VoiceClonePromptItem]. 단일 참조 오디오 → 첫 항목 사용.
        items = self._model.create_voice_clone_prompt(
            ref_audio=(np.asarray(waveform, dtype=np.float32), int(sample_rate)),
            ref_text=ref_text, x_vector_only_mode=False)
        item = items[0]
        emb = _to_numpy(item.ref_spk_embedding)
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return VoiceClonePromptSpec(
            ref_code=_to_numpy(item.ref_code), ref_spk_embedding=emb,
            x_vector_only_mode=bool(item.x_vector_only_mode), icl_mode=bool(item.icl_mode),
            ref_text=ref_text)

    def generate(self, prompt: VoiceClonePromptSpec, text: str, sample_rate: int) -> np.ndarray:
        """공식 VoiceClonePromptItem을 재구성해 list로 전달한다(ref_text 포함)."""
        from qwen_tts import VoiceClonePromptItem  # 공식 export
        item = VoiceClonePromptItem(
            ref_code=_from_numpy(prompt.ref_code),
            ref_spk_embedding=_from_numpy(prompt.ref_spk_embedding),
            x_vector_only_mode=prompt.x_vector_only_mode,
            icl_mode=prompt.icl_mode,
            ref_text=prompt.ref_text or None,
        )
        wavs, out_sr = self._model.generate_voice_clone(text, voice_clone_prompt=[item])
        wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if int(out_sr) != int(sample_rate):
            # 인터페이스 계약 유지: 요청 샘플레이트로 선형 리샘플(호출자는 24kHz를 기대).
            duration = wav.size / float(out_sr)
            n_out = int(duration * sample_rate)
            src_t = np.arange(wav.size, dtype=np.float64) / out_sr
            dst_t = np.arange(n_out, dtype=np.float64) / sample_rate
            wav = np.interp(dst_t, src_t, wav.astype(np.float64)).astype(np.float32)
        return wav

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
