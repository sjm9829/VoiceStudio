"""GGUF 모델/엔진 다운로드 관리자(P17-C).

정책:
- 고정 revision(gguf.py 상수)에서 정확한 파일명으로만 내려받는다. latest 의존 금지.
- SHA-256 불일치/부분 다운로드는 정상 캐시를 훼손하지 않는다(.tmp 하에서 작업).
- 완료 판정은 .complete 마커 + 두 파일 존재 + 해시 재검증이다.
"""

from __future__ import annotations
import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Callable
from ..core import gguf as gguf_cfg
from ..core.errors import OfflineError
from ..core.paths import gguf_models_dir, gguf_engine_dir

_logger = logging.getLogger(__name__)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _verify_pair(directory: Path) -> list[str]:
    """디렉터리의 모델 파일 존재/해시를 검증하고 문제 목록을 반환한다."""
    problems: list[str] = []
    for name, digest in (
        (gguf_cfg.GGUF_MAIN_FILE, gguf_cfg.GGUF_MAIN_SHA256),
        (gguf_cfg.GGUF_MMPROJ_FILE, gguf_cfg.GGUF_MMPROJ_SHA256),
    ):
        f = directory / name
        if not f.is_file() or f.stat().st_size == 0:
            problems.append(f"missing:{name}")
            continue
        actual = _sha256(f)
        if actual != digest:
            problems.append(f"hash-mismatch:{name}:{actual}")
    return problems


class GgufModelManager:
    """고정 revision GGUF 모델 다운로드/검증. UI 스레드에서 import해도 안전(지연 import)."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir if cache_dir is not None else gguf_models_dir() / gguf_cfg.GGUF_MODEL_REPO.replace("/", "__")

    def is_downloaded(self) -> bool:
        return not self.missing_or_broken()

    def missing_or_broken(self) -> list[str]:
        """완료 마커/파일/해시 문제를 반환. 빈 리스트면 사용 가능."""
        if not (self.cache_dir / ".complete").is_file():
            return ["marker:.complete"]
        return self.verify_files()

    def verify_files(self) -> list[str]:
        if not self.cache_dir.is_dir():
            return ["missing:cache-dir"]
        return _verify_pair(self.cache_dir)

    def status_text(self) -> str:
        if self.is_downloaded():
            return f"받아짐 ({gguf_cfg.GGUF_MAIN_FILE})"
        if self.cache_dir.is_dir() and any(self.cache_dir.iterdir()):
            return "받는 중 문제가 있었습니다. 다시 받아 주세요."
        return "아직 받지 않음 (Qwen3-TTS 1.7B Q8_0)"

    def model_path(self) -> str:
        """UI/job payload용 GGUF 모델 디렉터리 문자열. 없으면 ModelNotDownloadedError."""
        return str(self.model_dir())

    def model_dir(self) -> Path:
        """검증된 모델 디렉터리. 없으면 ModelNotDownloadedError."""
        from ..core.errors import ModelNotDownloadedError
        problems = self.missing_or_broken()
        if problems:
            raise ModelNotDownloadedError(
                "GGUF 음성 모델이 아직 받아지지 않았습니다. 설정에서 받은 뒤 다시 시도해 주세요.")
        return self.cache_dir

    def download(self, progress_cb: Callable[[dict], None] | None = None,
                 force: bool = False) -> Path:
        """고정 revision에서 두 파일을 내려받는다. 실패 시 정상 캐시를 훼손하지 않는다.

        성공 검증 전까지는 .tmp에서만 작업하고, 해시 재검증 통과 시에만
        정식 경로를 교체한다. 기존 정상 캐시는 force=False에서 유지된다.
        """
        problems = self.missing_or_broken()
        if not problems and not force:
            return self.cache_dir
        try:
            from huggingface_hub import hf_hub_download  # 지연 import: UI는 의존하지 않음
        except ImportError as exc:
            from ..core.errors import ModelNotDownloadedError
            raise ModelNotDownloadedError("모델 다운로드 라이브러리를 찾을 수 없습니다.") from exc
        self._disable_hub_console_progress()
        tmp = self.cache_dir.with_name(self.cache_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            for name in (gguf_cfg.GGUF_MAIN_FILE, gguf_cfg.GGUF_MMPROJ_FILE):
                if progress_cb is not None:
                    progress_cb({"stage": "download", "file": name})
                hf_hub_download(
                    repo_id=gguf_cfg.GGUF_MODEL_REPO,
                    filename=name,
                    revision=gguf_cfg.GGUF_MODEL_REVISION,
                    local_dir=tmp,
                )
            broken = _verify_pair(tmp)
            if broken:
                raise OfflineError(
                    f"GGUF 모델 파일 검증 실패: {', '.join(broken[:4])}",
                    user_message="모델 파일이 손상되어 받기에 실패했습니다. 다시 시도해 주세요.")
        except OfflineError:
            raise
        except Exception as exc:
            _logger.warning("GGUF 모델 다운로드 실패: %s", exc, exc_info=True)
            shutil.rmtree(tmp, ignore_errors=True)
            if self.is_downloaded():
                return self.cache_dir  # 기존 정상 캐시 유지
            from ..core.errors import ModelNotDownloadedError
            raise ModelNotDownloadedError(
                "음성 모델을 받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.") from exc
        # 성공 시에만 정식 경로로 교체(atomic swap)
        old = self.cache_dir.with_name(self.cache_dir.name + ".old")
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if self.cache_dir.exists():
            self.cache_dir.rename(old)
        tmp.rename(self.cache_dir)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        return self.cache_dir

    # ---- llama.cpp 엔진 다운로드(P17-C) ----
    def engine_dir(self) -> Path:
        return gguf_engine_dir()

    def engine_binary(self) -> Path:
        return self.engine_dir() / ("llama-tts.exe" if sys.platform == "win32" else "llama-tts")

    def is_engine_ready(self) -> bool:
        return self.engine_binary().is_file() and (self.engine_dir() / ".complete").is_file()

    def download_engine(self, progress_cb=None, force: bool = False) -> Path:
        """llama.cpp 고정 release b11540의 Windows CUDA zip 2종을 받아 풀어 검증한다.

        성공 조건: engine_dir/llama-tts(.exe) 존재. 실패 시 기존 engine dir을 훼손하지 않는다.
        """
        import zipfile
        import urllib.request
        engine = self.engine_dir()
        binary = self.engine_binary()
        marker = engine / ".complete"
        if marker.is_file() and binary.is_file() and not force:
            return engine
        base = f"https://github.com/ggml-org/llama.cpp/releases/download/{gguf_cfg.LLAMA_RELEASE_TAG}"
        tmp = engine.with_name(engine.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            for name in (gguf_cfg.LLAMA_WIN_CUDA_ASSET, gguf_cfg.LLAMA_CUDART_ASSET):
                if progress_cb is not None:
                    progress_cb({"stage": "engine", "file": name})
                dest = tmp / name
                self._download_file(f"{base}/{name}", dest)
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(tmp / "unpacked")
                dest.unlink()
            binaries = sorted((tmp / "unpacked").glob("llama-tts*"))
            exe = next((p for p in binaries if p.is_file()), None)
            if exe is None:
                raise OfflineError("llama-tts 실행 파일을 zip에서 찾지 못했습니다.",
                                   user_message="음성 엔진 설치에 실패했습니다. 다시 시도해 주세요.")
            # unpacked 내용을 engine dir로 이동
            for item in (tmp / "unpacked").iterdir():
                shutil.move(str(item), str(tmp / item.name))
            (tmp / "unpacked").rmdir()
            (tmp / ".complete").write_text(gguf_cfg.LLAMA_RELEASE_TAG)
            # atomic swap
            old = engine.with_name(engine.name + ".old")
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            if engine.exists():
                engine.rename(old)
            tmp.rename(engine)
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            if not self.engine_binary().is_file():
                raise OfflineError("엔진 실행 파일이 예상 경로에 없습니다.")
            return engine
        except OfflineError:
            raise
        except Exception as exc:
            _logger.warning("GGUF 엔진 다운로드 실패: %s", exc, exc_info=True)
            shutil.rmtree(tmp, ignore_errors=True)
            if marker.is_file() and self.engine_binary().is_file():
                return engine
            from ..core.errors import ModelNotDownloadedError
            raise ModelNotDownloadedError(
                "음성 엔진을 받지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요.") from exc

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        import urllib.request
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
        tmp.rename(dest)

    @staticmethod
    def _disable_hub_console_progress() -> None:
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except ImportError:
            import os
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
