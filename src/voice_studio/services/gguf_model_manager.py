"""GGUF 모델/엔진 다운로드 관리자(P17-C).

정책:
- 고정 revision(gguf.py 상수)에서 정확한 파일명으로만 내려받는다. latest 의존 금지.
- SHA-256 불일치/부분 다운로드는 정상 캐시를 훼손하지 않는다(.tmp 하에서 작업).
- 완료 판정은 .complete 마커 + 두 파일 존재 + 해시 재검증이다.
"""

from __future__ import annotations
import hashlib
import logging
import os
import shutil
import sys
import threading
import time
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


class _SingleFlight:
    """프로세스 단일 다운로드 조정자(P18-F2).

    서로 다른 창/워커가 동시에 download()를 호출해도 실제 네트워크 다운로드는
    한 번만 수행되고, 나머지 호출자는 같은 완료를 기다렸다가 결과를 재사용한다.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._owner: threading.Thread | None = None
        self._result: Path | None = None
        self._error: BaseException | None = None
        self._active = False

    def run(self, key: str, fn: Callable[[], Path]) -> Path:
        with self._cv:
            if self._active:
                me = threading.current_thread()
                if self._owner is not me:
                    # 진행 중 다운로드에 합류: 완료를 기다렸다가 같은 결과 사용.
                    while self._active:
                        self._cv.wait(timeout=1.0)
                    if self._error is not None and self._owner_key == key:
                        raise self._error
                    if self._result is not None and self._owner_key == key:
                        return self._result
                    # key 불일치/오류: 아래에서 자체 실행 폴백
                else:
                    raise RuntimeError("reentrant download")  # 방어용
            self._active = True
            self._owner = threading.current_thread()
            self._owner_key = key
            self._result = None
            self._error = None
        try:
            result = fn()
        except BaseException as exc:
            with self._cv:
                self._error = exc
                self._active = False
                self._owner = None
                self._cv.notify_all()
            raise
        with self._cv:
            self._result = result
            self._active = False
            self._owner = None
            self._cv.notify_all()
        return result

_SINGLE_FLIGHT = _SingleFlight()


def _pid_alive(pid: int) -> bool:
    """PID 생존 확인. Windows에서 os.kill(pid, 0)를 쓰지 않는다(P18-G: signal 0은
    존재하지 않는 권한/경합에서 종료 유발 가능)."""
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        SYNCHRONIZE = 0x00100000
        h = k32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not h:
            return False
        k32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)  # POSIX: signal 0은 확인 전용
        return True
    except OSError:
        return False


class _FileLock:
    """프로세스 간 배타적 잠금(P18-G). OS-native advisory lock만 사용한다.

    - Windows: msvcrt.locking(LK_NBLCK) / POSIX: fcntl.flock(LOCK_EX|LOCK_NB).
    - 잠금 파일은 대상 디렉터의 형제 경로(교체 rename의 대상이 아닌 안정 위치)에 둔다.
    - 커널/OS가 홀더 사망 시 잠금을 해제하므로 stale 정리·PID 재사용 판단이 불필요하다.
    - 획득 실패와 timeout은 정상 사용자 오류로 보고한다. 잠금 없이 진행하는 폴백 금지.
    """

    TIMEOUT_S = 600.0

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._locked = False

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.TIMEOUT_S
        last_err: Exception | None = None
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o666)
                self._lock_fd()
                self._locked = True
                try:  # 진단용 홀더 기록(갱신 실패는 무해)
                    os.ftruncate(self._fd, 0)
                    os.write(self._fd, str(os.getpid()).encode())
                except OSError:
                    pass
                return self
            except OSError as exc:
                last_err = exc
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except OSError:
                        pass
                    self._fd = None
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.25)
        raise TimeoutError(
            f"다른 VoiceStudio 프로세스가 파일 잠금을 보유 중입니다: {self.path} ({last_err})")

    def _lock_fd(self) -> None:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, *exc) -> None:
        if self._locked and self._fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        os.lseek(self._fd, 0, os.SEEK_SET)
                        msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
                    except OSError:
                        pass
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
        self._locked = False
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None


class GgufModelManager:
    """고정 revision GGUF 모델 다운로드/검증. UI 스레드에서 import해도 안전(지연 import)."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir if cache_dir is not None else gguf_models_dir() / gguf_cfg.GGUF_MODEL_REPO.replace("/", "__")

    def is_downloaded(self) -> bool:
        return not self.missing_or_broken()

    def quickly_ready(self) -> bool:
        """UI용 저비용 준비 확인: 마커+파일 크기만 보고 SHA-256 해시는 생략한다(P18-F2).

        전체 해시 검증은 백그라운드 download()/워커의 model_dir()에서 수행된다.
        """
        if not (self.cache_dir / ".complete").is_file():
            return False
        for name in (gguf_cfg.GGUF_MAIN_FILE, gguf_cfg.GGUF_MMPROJ_FILE):
            f = self.cache_dir / name
            if not f.is_file() or f.stat().st_size == 0:
                return False
        return True

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
        """UI/job payload용 GGUF 모델 디렉터리 문자열. 없으면 ModelNotDownloadedError.

        P18-G: UI 스레드 SHA-256 전량 계산 방지 - 마지막 백그라운드 검증 서명
        (파일 개수+mtime+size)이 유지되는 한 재검증 없이 경로만 반환한다.
        서명 불일치/부재 시 전체 검증을 수행한다(백그라운드 워커 경로와 동일).
        """
        return str(self.verified_model_dir())

    def _content_signature(self) -> tuple | None:
        try:
            sig = []
            for name in (gguf_cfg.GGUF_MAIN_FILE, gguf_cfg.GGUF_MMPROJ_FILE):
                st = (self.cache_dir / name).stat()
                sig.append((name, st.st_mtime_ns, st.st_size))
            return tuple(sig)
        except OSError:
            return None

    def note_verified(self) -> None:
        """백그라운드 검증 성공 후 서명을 기록한다(ModelPrepareWorker 경로)."""
        self._verified_sig = self._content_signature()

    def verified_model_dir(self) -> Path:
        """검증된 모델 디렉터. UI 신속 경로: 유효 서명이 있으면 해시 생략."""
        from ..core.errors import ModelNotDownloadedError
        if getattr(self, "_verified_sig", None) is not None:
            if (self.cache_dir / ".complete").is_file() \
                    and self._content_signature() == self._verified_sig:
                return self.cache_dir
            self._verified_sig = None  # 파일 변경/손상: 무효화 후 재검증
        problems = self.missing_or_broken()
        if problems:
            raise ModelNotDownloadedError(
                "GGUF 음성 모델이 아직 받아지지 않았습니다. 설정에서 받은 뒤 다시 시도해 주세요.")
        self._verified_sig = self._content_signature()
        return self.cache_dir

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
        # P18-F2: 프로세스 전체 단일 다운로드. 두 번째 호출자는 진행 중 다운로드에 합류한다.
        return _SINGLE_FLIGHT.run(
            f"model:{self.cache_dir}",
            lambda: self._download_locked(progress_cb, force))

    def _download_locked(self, progress_cb=None, force: bool = False) -> Path:
        try:
            from huggingface_hub import hf_hub_download  # 지연 import: UI는 의존하지 않음
        except ImportError as exc:
            from ..core.errors import ModelNotDownloadedError
            raise ModelNotDownloadedError("모델 다운로드 라이브러리를 찾을 수 없습니다.") from exc
        tmp = self.cache_dir.with_name(self.cache_dir.name + ".tmp")
        # P18-F4: 진입 시 tmp를 지우지 않는다. .incomplete/.gguf 조각을 보존해 이어받기.
        tmp.mkdir(parents=True, exist_ok=True)
        # P18-G: 잠금 파일은 교체(rename)되지 않는 형제 경로에 둔다.
        with _FileLock(self.cache_dir.with_name(self.cache_dir.name + ".download.lock")):
            # P18-G: 선행 프로세스가 방금 완료했으면 재검증 후 그 결과를 재사용한다.
            if not force and not self.missing_or_broken():
                return self.cache_dir
            return self._download_inner(tmp, progress_cb)

    def _download_inner(self, tmp: Path, progress_cb=None) -> Path:
        try:
            sizes = self._remote_file_sizes()
            total_bytes = sum(sizes.get(n, 0) for n in (gguf_cfg.GGUF_MAIN_FILE, gguf_cfg.GGUF_MMPROJ_FILE))
            done_bytes = 0
            # P18-F4: 이어받기 기준점 - 기존 조각 크기를 진행률에 반영한다.
            for _n in (gguf_cfg.GGUF_MAIN_FILE, gguf_cfg.GGUF_MMPROJ_FILE):
                inc = tmp / (_n + ".incomplete")
                fin = tmp / _n
                if inc.is_file():
                    done_bytes += inc.stat().st_size
                elif fin.is_file() and sizes.get(_n, 0) and fin.stat().st_size == sizes[_n]:
                    done_bytes += fin.stat().st_size
            last_emit = 0.0

            def _emit(force: bool = False) -> None:
                # P18-F3: 실수신 바이트 기반 진행률(0.5초 스로틀).
                nonlocal last_emit
                if progress_cb is None:
                    return
                now = time.monotonic()
                if not force and now - last_emit < 0.5:
                    return
                last_emit = now
                if total_bytes > 0:
                    progress_cb({"stage": "download", "done": min(done_bytes, total_bytes),
                                 "total": total_bytes})
                else:
                    progress_cb({"stage": "download", "done": done_bytes, "total": 0})

            def _fetch(name: str) -> None:
                """단일 파일 스트리밍 다운로드(.incomplete 재개 지원). 성공 시 tmp/name 완성."""
                nonlocal done_bytes
                fin0 = tmp / name
                expected0 = sizes.get(name, 0)
                if fin0.is_file() and expected0 and fin0.stat().st_size == expected0:
                    done_bytes += 0  # 완전한 크기의 조각 존재: 검증은 _verify_pair가 담당
                    return
                from huggingface_hub import hf_hub_url
                import requests
                url = hf_hub_url(gguf_cfg.GGUF_MODEL_REPO, filename=name, revision=gguf_cfg.GGUF_MODEL_REVISION)
                incomplete = tmp / (name + ".incomplete")
                headers: dict = {}
                resume = incomplete.stat().st_size if incomplete.exists() else 0
                if resume > 0:
                    headers["Range"] = f"bytes={resume}-"
                expected = sizes.get(name, 0)
                with requests.get(url, stream=True, timeout=(10, 60), headers=headers,
                                  allow_redirects=True) as r:
                    r.raise_for_status()
                    length = r.headers.get("Content-Length")
                    # P18-F3: Content-Length/HEAD 부재 시 임의 퍼센트 대신 총량 미지 상태 유지.
                    if resume > 0 and r.status_code == 200:  # Range 무시 응답: 처음부터
                        resume = 0
                    partial = resume > 0 and r.status_code == 206
                    if not partial:
                        resume = 0
                    else:
                        done_bytes += resume
                    _emit(force=True)
                    with open(incomplete, "ab" if partial else "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            done_bytes += len(chunk)
                            _emit()
                actual = incomplete.stat().st_size
                if expected and actual != expected:
                    raise OfflineError(
                        f"크기 불일치: {name} {actual}/{expected}",
                        user_message="모델 파일 받기가 불완전합니다. 다시 시도해 주세요.")
                os.replace(incomplete, tmp / name)

            _emit(force=True)
            for name in (gguf_cfg.GGUF_MAIN_FILE, gguf_cfg.GGUF_MMPROJ_FILE):
                _fetch(name)
            _emit(force=True)
            if progress_cb is not None:
                progress_cb({"stage": "verify", "done": 0, "total": 0})
            broken = _verify_pair(tmp)
            if broken:
                # P18-F4: 손상된 완성 조각은 제거해 다음 시도에서 재수신. incomplete은 보존.
                for item in broken:
                    kind, _, rest = item.partition(":")
                    if kind in ("hash-mismatch", "missing") and rest:
                        bad = rest.split(":", 1)[0] if kind == "hash-mismatch" else rest
                        try:
                            (tmp / bad).unlink(missing_ok=True)
                        except OSError:
                            pass
                raise OfflineError(
                    f"GGUF 모델 파일 검증 실패: {', '.join(broken[:4])}",
                    user_message="모델 파일이 손상되어 받기에 실패했습니다. 다시 시도해 주세요.")
            (tmp / ".complete").write_text(gguf_cfg.GGUF_MODEL_REVISION)  # 완료 마커도 swap 대상
        except OfflineError:
            raise
        except Exception as exc:
            _logger.warning("GGUF 모델 다운로드 실패: %s", exc, exc_info=True)
            # P18-F4: 부분 다운로드(.incomplete)를 보존해 다음 시도에서 이어받기.
            try:
                (tmp / ".complete").unlink(missing_ok=True)
            except OSError:
                pass
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
        # P18-G: 엔진 staging도 프로세스 간 배타 잠금으로 보호한다.
        lock_ctx = _FileLock(engine.with_name(engine.name + ".download.lock"))
        lock_ctx.__enter__()
        try:
            if marker.is_file() and self.engine_binary().is_file():
                return engine  # 선행 프로세스가 완료한 엔진 재사용(P18-G)
            for name in (gguf_cfg.LLAMA_WIN_CUDA_ASSET, gguf_cfg.LLAMA_CUDART_ASSET):
                if progress_cb is not None:
                    progress_cb({"stage": "engine", "file": name})
                dest = tmp / name
                self._download_file(f"{base}/{name}", dest,
                                    progress=progress_cb, label=name)
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
        finally:
            lock_ctx.__exit__()

    @staticmethod
    def _download_file(url: str, dest: Path, progress=None, label: str = "") -> None:
        """P18-6: 실시간 바이트 진행. content-length 기반 done/total 콜백."""
        import urllib.request
        tmp = dest.with_suffix(dest.suffix + ".part")
        done0 = 0
        req = urllib.request.Request(url, method="HEAD") if False else None
        with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last = 0.0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if progress and (now - last > 0.2 or done >= total):
                    last = now
                    progress({"stage": "engine", "file": label, "done": done, "total": total})
        tmp.rename(dest)

    def _remote_file_sizes(self) -> dict[str, int]:
        """P18-6: 고정 revision 파일들의 실제 크기(바이트). 실패 시 {}."""
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(gguf_cfg.GGUF_MODEL_REPO, revision=gguf_cfg.GGUF_MODEL_REVISION,
                                      files_metadata=True)
            return {s.rfilename: int(s.size or 0) for s in info.siblings or []}
        except Exception:
            return {}

    @staticmethod
    def _disable_hub_console_progress() -> None:
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except ImportError:
            import os
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
