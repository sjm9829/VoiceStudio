"""애플리케이션 데이터 경로 결정. Windows와 개발(비Windows) 환경 모두 지원."""

from __future__ import annotations
import os, sys
from pathlib import Path

APP_DIR_NAME = "VoiceStudio"
PROFILE_SCHEMA_VERSION = 1

def app_data_dir() -> Path:
    r"""데이터 루트 결정.

    우선순위: VOICE_STUDIO_DATA_DIR override(OS 무관) > Windows %LOCALAPPDATA%
    > 비-Windows ~/.local/state. 최종 경로는 base / VoiceStudio.
    override는 dev/test/E2E isolation 용도이며 일반 설치 앱에는 없다.
    """
    override = os.environ.get("VOICE_STUDIO_DATA_DIR")
    if override:
        return Path(override) / APP_DIR_NAME
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = str(Path.home() / ".local" / "state")
    return Path(base) / APP_DIR_NAME

def profiles_dir() -> Path:
    return app_data_dir() / "profiles"

def jobs_cache_dir() -> Path:
    return app_data_dir() / "cache" / "jobs"

def logs_dir() -> Path:
    return app_data_dir() / "logs"

def worker_logs_dir() -> Path:
    """worker 실패 진단 로그 위치(P12.2-23). UI traceback 노출 대신 파일로 남긴다."""
    return logs_dir()

def settings_file() -> Path:
    return app_data_dir() / "settings.json"

def default_mp3_dir() -> Path:
    """기본 MP3 저장 폴더: 내 문서 아래 Music/보이스 스튜디오."""
    home = Path.home()
    if sys.platform == "win32":
        music = Path(os.environ.get("USERPROFILE", str(home))) / "Music"
    else:
        music = Path(os.environ.get("VOICE_STUDIO_MUSIC_DIR", str(home)))
    return music / "보이스 스튜디오"

def ensure_app_dirs() -> dict[str, Path]:
    paths = {"app": app_data_dir(), "profiles": profiles_dir(), "jobs": jobs_cache_dir(),
             "logs": logs_dir(), "preview": preview_cache_dir()}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def preview_cache_dir() -> Path:
    r"""미리 듣기 임시 WAV 관리 경로(%LOCALAPPDATA%\VoiceStudio\cache\preview). temp에 무작위 파일을 남기지 않는다."""
    return app_data_dir() / "cache" / "preview"

def cleanup_preview_cache(max_age_hours: float = 24.0) -> int:
    """관리 경로의 오래된 미리 듣기 임시 파일 삭제. 개별 파일 삭제 실패는 무시한다."""
    import time
    d = preview_cache_dir()
    if not d.is_dir():
        return 0
    now = time.time()
    removed = 0
    for f in d.iterdir():
        if not f.is_file():
            continue
        try:
            if now - f.stat().st_mtime > max_age_hours * 3600:
                f.unlink()
                removed += 1
        except OSError:
            continue  # 재생 프로그램이 잡고 있으면 다음 시작에 다시 시도
    return removed

def cleanup_stale_jobs(max_age_hours: float = 24.0) -> int:
    """startup에서 오래된 작업 캐시 폴더(jobs/<uuid>)를 정리한다(P12.3-19).

    앱 비정상 종료 시 남은 cache/jobs/<uuid>를 제거한다. 시작 시점에는 활성
    worker가 없으므로 max_age보다 오래된 폴더 전부를 삭제해도 안전하다.
    프로필/모델/설정/로그 폴더는 절대 건드리지 않는다.
    """
    import shutil, time
    d = jobs_cache_dir()
    if not d.is_dir():
        return 0
    now = time.time()
    removed = 0
    for entry in d.iterdir():
        if not entry.is_dir():
            continue
        try:
            if now - entry.stat().st_mtime > max_age_hours * 3600:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def safe_job_cache_dir(job_id: str) -> Path:
    """작업별 임시 폴더. job_id는 uuid이므로 경로 탈출 위험이 없다."""
    d = jobs_cache_dir() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d
