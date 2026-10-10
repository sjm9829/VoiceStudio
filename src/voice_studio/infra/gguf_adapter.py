"""llama.cpp llama-tts 기반 GGUF 백엔드 어댑터(P17-C).

- QwenAdapter 프로토콜(create_prompt/generate)을 그대로 만족한다.
- 실제 실행은 worker 프로세스의 subprocess(llama-tts)이며 Qt 메인 프로세스는 관여하지 않는다.
- --tts-speaker-file 음성복제: 참조 음성 파일을 그대로 사용한다(공식 ICL 프롬프트와 다른 경로).
  ref_text는 llama-tts가 사용하지 않지만 프로필 호환 검증을 위해 보존한다.
- CUDA 없는 환경에서 조용한 CPU fallback 금지: GPU 사용 불가면 명시적 오류를 낸다.
"""

from __future__ import annotations
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..core import gguf as gguf_cfg
from ..core.errors import WorkerError

DEFAULT_NGPU = 12          # RTX 2070 SUPER 8GB 기준 mmproj+backbone 여유치(측정 전 보수값)
DEFAULT_N_CTX = 4096       # H5: KV 캐시 VRAM 점유 제한용. b11540 기본 대비 보수적.
                           # 실측 전 추정값이며, 문장 단위 TTS 입력(수백 토큰)이라 품질 영향은 작음.
DEFAULT_TEMP = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95


@dataclass
class GgufPromptSpec:
    """GGUF 백엔드용 복제 프롬프트. 참조 음성 파일 경로와 설정을 담는다."""

    # 공용 프로필 계약(VoiceClonePromptSpec 호환): GGUF는 ref_code 텐서를 만들지 않는다.
    ref_code: object | None = None
    ref_spk_embedding: object | None = None
    x_vector_only_mode: bool = False
    icl_mode: bool = False
    speaker_wav: str = ""
    ref_text: str = ""
    language: str = "ko"
    ngpu_layers: int = DEFAULT_NGPU
    n_ctx: int = DEFAULT_N_CTX
    temp: float = DEFAULT_TEMP
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    model_dir: str = ""
    engine_path: str = ""


def _wav_to_pcm16(path: str | Path) -> tuple[np.ndarray, int]:
    """wave 모듈로 PCM16 wav을 읽어 (mono float32, sr)로 반환한다."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        width = w.getsampwidth()
        if width != 2:
            raise WorkerError(f"지원하지 않는 WAV 형식입니다(width={width}).")
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data.astype(np.float32), int(sr)


def _trim_edge_silence(pcm: np.ndarray, sample_rate: int, *,
                       keep_s: float = 0.12, rel: float = 0.05, floor: float = 0.004) -> np.ndarray:
    """P18-3: 생성 결과 앞뒤 무음을 정리한다(원인: 모델이 리드/테일 무음을 임의로 포함).

    무음을 '추가'하는 것이 아니라 제거하는 동작이다. 임계값은 피크 기준 상대값.
    """
    if pcm.size == 0:
        return pcm
    peak = float(np.abs(pcm).max())
    th = max(floor, peak * rel)
    idx = np.where(np.abs(pcm) > th)[0]
    if idx.size == 0:
        return pcm
    keep = int(keep_s * sample_rate)
    a = max(0, int(idx[0]) - keep)
    b = min(pcm.size, int(idx[-1]) + 1 + keep)
    return pcm[a:b]


def _resample(wav: np.ndarray, src: int, dst: int) -> np.ndarray:
    if int(src) == int(dst):
        return wav
    duration = wav.size / float(src)
    n_out = max(1, int(duration * dst))
    t_src = np.arange(wav.size, dtype=np.float64) / src
    t_dst = np.arange(n_out, dtype=np.float64) / dst
    return np.interp(t_dst, t_src, wav.astype(np.float64)).astype(np.float32)


def write_pcm16_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> Path:
    """float32 mono pcm을 16bit wav로 저장한다(참조 음성 캐시용)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    ints = (data * 32767.0).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(ints.tobytes())
    return p


class GgufQwenAdapter:
    """llama-tts 서브프로세스로 1.7B Q8_0 GGUF를 구동하는 백엔드.

    engine_path는 검증된 llama-tts 실행 파일, model_dir는 검증된 GGUF 캐시 디렉터리.
    """

    model_version = "gguf-qwen17-q8"

    def __init__(self, model_dir: str | Path, engine_path: str | Path, *,
                 language: str = "ko", ngpu_layers: int = DEFAULT_NGPU,
                 n_ctx: int = DEFAULT_N_CTX, temp: float = DEFAULT_TEMP,
                 top_k: int = DEFAULT_TOP_K,
                 top_p: float = DEFAULT_TOP_P, runner=None):
        self.model_dir = str(model_dir)
        self.engine_path = str(engine_path)
        self.language = language
        self.ngpu_layers = int(ngpu_layers)
        self.n_ctx = int(n_ctx)
        self.temp = float(temp)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self._runner = runner or self._run
        self.last_command: list[str] | None = None

    # -- QwenAdapter 계약 --
    def create_prompt(self, waveform: np.ndarray, sample_rate: int, ref_text: str) -> GgufPromptSpec:
        """참조 pcm을 24kHz mono wav로 캐시해 --tts-speaker-file 경로를 준비한다."""
        # 저장 계약(safetensors ref_code/ref_spk_embedding)을 채우기 위한 최소 placeholder.
        # 0.6B 텐서가 아니며 gguf는 이를 생성 단계에서 사용하지 않는다(재사용 아님).
        return GgufPromptSpec(
            ref_code=np.zeros(1, dtype=np.float32),
            ref_spk_embedding=np.zeros(1, dtype=np.float32),
            speaker_wav="",  # generate에서 job temp로 채워짐(직렬화 안 함)
            ref_text=ref_text,
            language=self.language,
            ngpu_layers=self.ngpu_layers,
            n_ctx=self.n_ctx,
            temp=self.temp,
            top_k=self.top_k,
            top_p=self.top_p,
            model_dir=self.model_dir,
            engine_path=self.engine_path,
        )

    def prepare_speaker(self, prompt, speaker_wav: str | Path):
        """narrate 워커가 저장된 참조 음성 경로를 prompt에 연결한다(새 spec 반환).

        official VoiceClonePromptSpec이 들어오면 gguf spec으로 변환한다(텐서는 미사용).
        """
        import dataclasses
        from .qwen_adapter import VoiceClonePromptSpec
        if isinstance(prompt, VoiceClonePromptSpec):
            prompt = GgufPromptSpec(
                ref_text=getattr(prompt, "ref_text", ""),
                language=self.language,
                ngpu_layers=self.ngpu_layers,
                n_ctx=self.n_ctx,
                temp=self.temp,
                top_k=self.top_k,
                top_p=self.top_p,
                model_dir=self.model_dir,
                engine_path=self.engine_path,
            )
        return dataclasses.replace(prompt, speaker_wav=str(speaker_wav))

    def generate(self, prompt: GgufPromptSpec, text: str, sample_rate: int,
                 language: str | None = None, workdir: str | Path | None = None) -> np.ndarray:
        """llama-tts를 실행해 wav를 읽어 요청 sr의 float32 pcm으로 반환한다."""
        if not prompt.speaker_wav or not Path(prompt.speaker_wav).is_file():
            raise WorkerError("참조 음성 파일이 없어 목소리 복제를 시작할 수 없습니다.")
        if not Path(self.engine_path).is_file():
            raise WorkerError("GGUF 음성 엔진(llama-tts)을 찾을 수 없습니다. 프로그램을 다시 설치해 주세요.")
        out_wav = Path(workdir or Path(self.model_dir).parent) / "gguf_out.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        if out_wav.exists():
            out_wav.unlink()
        cmd = self.build_command(text=text, out_wav=out_wav, speaker=str(prompt.speaker_wav),
                                 language=language or prompt.language)
        self.last_command = cmd
        completed = self._runner(cmd)
        if not out_wav.is_file() or out_wav.stat().st_size == 0:
            raise WorkerError("음성 생성 결과가 비어 있습니다.")
        pcm, src_sr = _wav_to_pcm16(out_wav)
        if pcm.size == 0:
            raise WorkerError("음성 생성 결과가 비어 있습니다.")
        pcm = _trim_edge_silence(pcm, src_sr)
        try:
            out_wav.unlink()
        except OSError:
            pass
        return _resample(pcm, src_sr, sample_rate)

    # -- 내부 --
    def build_command(self, *, text: str, out_wav: Path, speaker: str,
                      language: str = "ko") -> list[str]:
        """llama-tts 인자 구성. 실제 바이너리 --help 기준 확인된 long 옵션만 사용한다."""
        return [
            self.engine_path,
            "-m", str(Path(self.model_dir) / gguf_cfg.GGUF_MAIN_FILE),
            "--mmproj", str(Path(self.model_dir) / gguf_cfg.GGUF_MMPROJ_FILE),
            "-p", text,
            "--tts-lang", language,
            "--tts-speaker-file", speaker,
            "--output", str(out_wav),
            "-ngl", str(self.ngpu_layers),
            "-c", str(self.n_ctx),
            "--temp", str(self.temp),
            "--top-k", str(self.top_k),
            "--top-p", str(self.top_p),
        ]

    @staticmethod
    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        """worker 프로세스 내 llama-tts 실행. 한국어 경로/공백 안전(list 인자)."""
        from .subprocess_runner import run
        return run(cmd, capture_output=True, text=True, timeout=1800)

    def engine_info(self) -> dict:
        return {"engine": "llama.cpp", "binary": self.engine_path,
                "version": _engine_version(self.engine_path)}


def _engine_version(engine_path: str | Path) -> str:
    """llama-tts --version 출력의 첫 줄(빌드 태그)만 안전하게 추출한다."""
    try:
        from .subprocess_runner import run as _run_no_window
        r = _run_no_window([str(engine_path), "--version"], capture_output=True, text=True, timeout=30)
        line = (r.stdout or r.stderr or "").strip().splitlines()
        return line[0] if line else "unknown"
    except Exception:
        return "unknown"
