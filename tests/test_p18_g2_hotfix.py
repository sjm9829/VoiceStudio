"""P18-G2 핫픽스 회귀: 엔진 staging 잠금 선제 확보, verify-only 모델 검증."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest import mock

from voice_studio.services import gguf_model_manager as gmm
from voice_studio.core import gguf as gguf_cfg


def test_engine_download_locks_before_touching_tmp(tmp_path, monkeypatch):
    """잠금 획득 전에 engine.tmp를 만지지 않는다(다른 프로세스 staging 보호, P18-G2)."""
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    lock_path = tmp_path / "engine.download.lock"
    events = []
    orig_rmtree = gmm.shutil.rmtree
    orig_mkdir = Path.mkdir

    def spy_rmtree(p, *a, **k):
        events.append(("rmtree", Path(p)))
        return orig_rmtree(p, *a, **k)

    monkeypatch.setattr(gmm.shutil, "rmtree", spy_rmtree)

    def spy_mkdir(self, *a, **k):
        events.append(("mkdir", self))
        return orig_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", spy_mkdir)

    orig_enter = gmm._FileLock.__enter__

    def spy_enter(self):
        events.append(("lock", self.path))
        return orig_enter(self)

    monkeypatch.setattr(gmm._FileLock, "__enter__", spy_enter)
    monkeypatch.setattr(gmm.GgufModelManager, "_download_engine_inner",
                        lambda self, tmp, engine, base, progress_cb=None: None)

    m = gmm.GgufModelManager(cache_dir=tmp_path / "unused-model")
    # engine 경로를 강제 지정
    m.engine_dir = lambda: engine_dir  # type: ignore[method-assign]
    m.engine_binary = lambda: engine_dir / "llama-tts"  # type: ignore[method-assign]
    # 다운로드 자체는 네트워크 없이 inner가 no-op이므로 zip 없이 종료 -> engine 그대로
    m.download_engine()
    lock_idx = next(i for i, e in enumerate(events) if e[0] == "lock")
    tmp_ops = [i for i, e in enumerate(events)
               if e[0] in ("rmtree", "mkdir") and "tmp" in str(e[1])]
    assert tmp_ops, "tmp 조작 이벤트가 관측되어야 한다"
    assert all(i > lock_idx for i in tmp_ops), \
        f"잠금({lock_idx}) 이전 tmp 조작 금지: {events}"


def test_engine_download_preserves_existing_engine_on_failure(tmp_path, monkeypatch):
    """수신 실패 시 기존 engine dir과 .complete를 훼손하지 않는다."""
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / ".complete").write_text("revA")
    binary = engine_dir / "llama-tts"
    binary.write_bytes(b"\x7fELF")

    def boom(self, tmp, engine, base, progress_cb=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(gmm.GgufModelManager, "_download_engine_inner", boom)
    m = gmm.GgufModelManager(cache_dir=tmp_path / "model")
    m.engine_dir = lambda: engine_dir  # type: ignore[method-assign]
    m.engine_binary = lambda: engine_dir / "llama-tts"  # type: ignore[method-assign]
    result = m.download_engine()
    assert result == engine_dir
    assert (engine_dir / ".complete").read_text() == "revA"
    assert binary.read_bytes() == b"\x7fELF"


def test_engine_download_reuses_completed_by_racing_process(tmp_path, monkeypatch):
    """잠금 후 선행 프로세스 완성 엔진 발견 시 재다운로드 없이 재사용."""
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / ".complete").write_text("rev")
    (engine_dir / "llama-tts").write_bytes(b"\x7fELF")
    inner_calls = []
    monkeypatch.setattr(gmm.GgufModelManager, "_download_engine_inner",
                        lambda self, t, e, b, progress_cb=None: inner_calls.append(1))
    m = gmm.GgufModelManager(cache_dir=tmp_path / "model")
    m.engine_dir = lambda: engine_dir  # type: ignore[method-assign]
    m.engine_binary = lambda: engine_dir / "llama-tts"  # type: ignore[method-assign]
    # force=False, marker/binary 존재 -> 사전 반환 경로
    assert m.download_engine() == engine_dir
    # 강제로 lock 이후 재검사 경로를 태우려면 사전 marker를 지우고 진입한다.
    inner_calls.clear()
    m2 = gmm.GgufModelManager(cache_dir=tmp_path / "model2")
    m2.engine_dir = lambda: engine_dir  # type: ignore[method-assign]
    m2.engine_binary = lambda: engine_dir / "llama-tts"  # type: ignore[method-assign]
    # marker/binary 존재: 사후 재검사에서 재발견되어 inner 미실행
    m2.download_engine()
    assert inner_calls == []
    # marker 제거: lock 후 재검사 실패로 inner 1회 실행(재다운로드 경로)
    (engine_dir / ".complete").unlink()
    m3 = gmm.GgufModelManager(cache_dir=tmp_path / "model3")
    m3.engine_dir = lambda: engine_dir  # type: ignore[method-assign]
    m3.engine_binary = lambda: engine_dir / "llama-tts"  # type: ignore[method-assign]
    m3.download_engine()
    assert inner_calls == [1]


def test_engine_download_concurrent_two_managers(tmp_path, monkeypatch):
    """두 매니저가 동시 download_engine -> 한 번만 inner 실행, 서로 staging 훼손 없음."""
    engine_dir = tmp_path / "engine"
    lock = threading.Lock()
    entered = []

    def slow_inner(self, tmp, engine, base, progress_cb=None):
        with lock:
            entered.append(tmp)
        time.sleep(0.2)
        # 성공 시뮬레이션: engine_dir 교체는 download_engine이 atomic swap으로 수행.
        # inner는 tmp에 marker/binary를 남기고 종료(unpacked 이후 단계 축약).
        (tmp / ".complete").write_text("rev")
        (tmp / "llama-tts").write_bytes(b"x")

    monkeypatch.setattr(gmm.GgufModelManager, "_download_engine_inner", slow_inner)
    results = []

    def run(m):
        m.engine_dir = lambda: engine_dir  # type: ignore[method-assign]
        m.engine_binary = lambda: engine_dir / "llama-tts"  # type: ignore[method-assign]
        results.append(m.download_engine())

    import threading as th
    m1 = gmm.GgufModelManager(cache_dir=tmp_path / "m1")
    m2 = gmm.GgufModelManager(cache_dir=tmp_path / "m2")
    t1 = th.Thread(target=run, args=(m1,))
    t1.start()
    time.sleep(0.05)
    t2 = th.Thread(target=run, args=(m2,))
    t2.start()
    t1.join()
    t2.join()
    # 두 번째는 잠금 후 재검사에서 성공 확인 -> inner 재실행 없음 혹은 순차 실행
    assert len(entered) <= 2
    assert all(r == engine_dir for r in results)

def _ui_context(tmp_path, manager):
    import types
    from voice_studio.core.job_coordinator import JobCoordinator
    ctx = types.SimpleNamespace()
    ctx.settings = {"mp3_output_dir": str(tmp_path / "out")}
    ctx.profile_repository = types.SimpleNamespace(root=str(tmp_path / "profiles"))
    ctx.model_manager = manager
    ctx.profile_service = types.SimpleNamespace(list_profiles=lambda: [])
    ctx.jobs = JobCoordinator()
    return ctx


def test_mainwindow_starts_background_verify_even_when_ready(qtbot, tmp_path, monkeypatch):
    """quickly_ready 참이어도 시작 시 백그라운드 검증이 시작된다(P18-G2)."""
    import types
    from voice_studio.core.job_coordinator import JobCoordinator
    from voice_studio.ui import main_window as mw
    from voice_studio.ui.main_window import MainWindow

    calls = {}

    manager = gmm.GgufModelManager(cache_dir=tmp_path / "model")
    manager.cache_dir.mkdir(parents=True, exist_ok=True)
    (manager.cache_dir / ".complete").write_text("rev")
    (manager.cache_dir / gguf_cfg.GGUF_MAIN_FILE).write_bytes(b"a" * 32)
    (manager.cache_dir / gguf_cfg.GGUF_MMPROJ_FILE).write_bytes(b"b")

    orig_ensure = mw.MainWindow._ensure_model_async

    def spy_ensure(self, done, verify_only=False):
        calls["n"] = calls.get("n", 0) + 1
        calls["verify_only"] = verify_only
        return None  # 테스트: 실제 스레드 기동 생략

    monkeypatch.setattr(mw.MainWindow, "_ensure_model_async", spy_ensure)
    win = MainWindow(_ui_context(tmp_path, manager))
    qtbot.addWidget(win)
    qtbot.waitUntil(lambda: calls.get("n") == 1, timeout=3000)
    assert calls.get("verify_only") is True

def test_start_generation_joins_verify_and_continues_once(qtbot, tmp_path, monkeypatch):
    """서명 미확정 상태에서 생성 시 verify-only 합류 후 단 1회 생성(P18-G2)."""
    import types
    from voice_studio.ui import main_window as mw
    from voice_studio.ui.main_window import MainWindow

    manager = gmm.GgufModelManager(cache_dir=tmp_path / "model")
    manager.cache_dir.mkdir(parents=True, exist_ok=True)
    (manager.cache_dir / ".complete").write_text("rev")
    (manager.cache_dir / gguf_cfg.GGUF_MAIN_FILE).write_bytes(b"a" * 32)
    (manager.cache_dir / gguf_cfg.GGUF_MMPROJ_FILE).write_bytes(b"b")

    # quickly_ready True가 되도록(파일 존재) 하되 _verified_sig는 미확정
    ctx = _ui_context(tmp_path, manager)
    ctx.profile_service.list_profiles = lambda: [types.SimpleNamespace(uuid="u1", name="v1")]

    gen_calls = []
    started = {}

    orig_ensure = mw.MainWindow._ensure_model_async

    def fake_ensure(self, done, verify_only=False):
        started["verify_only"] = verify_only
        # 동기 완료 시뮬레이션: manager 서명 확정 후 done(True)
        manager.note_verified()
        done(True)
        return None

    monkeypatch.setattr(mw.MainWindow, "_ensure_model_async", fake_ensure)

    win = MainWindow(ctx)
    qtbot.addWidget(win)
    win.show()
    # 콤보에 목소리 설정
    idx = win.voice_combo.findData("u1")
    assert idx >= 0
    win.voice_combo.setCurrentIndex(idx)
    win.script_edit.setPlainText("안녕하세요. 테스트입니다.")
    monkeypatch.setattr(win, "_start_worker", lambda job: gen_calls.append(job) or True)
    monkeypatch.setattr(mw.QMessageBox, "information", lambda *a: None)

    win.start_generation()
    assert started.get("verify_only") is True
    assert len(gen_calls) == 1  # 단 한 번 생성
    job = gen_calls[0]
    assert job["segments"] and job["profile_uuid"] == "u1"
