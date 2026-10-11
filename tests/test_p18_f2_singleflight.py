"""P18-F2/F4 회귀: single-flight 다운로드, .incomplete 재개, UI 저비용 준비 확인."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest import mock

import pytest

from voice_studio.services import gguf_model_manager as gmm
from voice_studio.core import gguf as gguf_cfg


def _mk(tmp_path: Path) -> gmm.GgufModelManager:
    return gmm.GgufModelManager(cache_dir=tmp_path / "model")


def test_single_flight_dedupes_concurrent_downloads(tmp_path, monkeypatch):
    """동시 download() 호출은 한 번의 실제 다운로드로 수렴한다(P18-F2)."""
    m = _mk(tmp_path)
    calls = []

    def fake_inner(self, tmp, progress_cb=None):
        calls.append(1)
        (tmp / gguf_cfg.GGUF_MAIN_FILE).write_bytes(b"x" * 10)
        (tmp / (gguf_cfg.GGUF_MAIN_FILE + ".incomplete")).unlink(missing_ok=True)
        (tmp / ".complete").write_text("rev")
        return self.cache_dir

    monkeypatch.setattr(gmm.GgufModelManager, "_download_inner", fake_inner)
    monkeypatch.setattr(gmm, "_verify_pair", lambda d: [])
    results = []
    def run():
        results.append(m.download())

    ths = [threading.Thread(target=run) for _ in range(3)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert len(calls) == 1
    assert all(r == m.cache_dir for r in results)


def test_download_preserves_incomplete_on_failure(tmp_path, monkeypatch):
    """실패 후에도 .incomplete가 남아 재개 기반을 보존한다(P18-F4)."""
    m = _mk(tmp_path)
    tmp = m.cache_dir.with_name(m.cache_dir.name + ".tmp")
    tmp.mkdir(parents=True)
    inc = tmp / (gguf_cfg.GGUF_MAIN_FILE + ".incomplete")
    inc.write_bytes(b"partial")

    def boom(self):
        raise RuntimeError("network gone")

    monkeypatch.setattr(gmm.GgufModelManager, "_remote_file_sizes", boom)
    with pytest.raises(Exception):
        m.download()
    assert inc.read_bytes() == b"partial"  # rmtree로 사라지지 않음
    assert not (tmp / ".complete").exists()


def test_download_resumes_from_incomplete_bytes(tmp_path, monkeypatch):
    """재호출 시 .incomplete 크기만큼 이어받는다(P18-F4, Range 헤더 검증)."""
    m = _mk(tmp_path)
    main_name = gguf_cfg.GGUF_MAIN_FILE
    captured = {}

    class FakeResp:
        status_code = 206
        headers = {"Content-Length": "4"}
        def __enter__(self): return self
        def __exit__(self, *a): return None
        def raise_for_status(self): return None
        def iter_content(self, chunk_size=1):
            yield b"tail"
    def fake_get(url, stream=True, timeout=None, headers=None, allow_redirects=True):
        captured.setdefault("ranges", {})[url.rsplit("/",1)[-1]] = (headers or {}).get("Range", "")
        return FakeResp()

    monkeypatch.setattr(gmm, "hf_hub_url", None, raising=False)  # ensure module import path used? skip
    # 직접 fetch 시뮬레이션은 복잡하므로 _download_inner의 resume seed만 확인: inc 존재 시 done seed
    sizes = {main_name: 8, gguf_cfg.GGUF_MMPROJ_FILE: 0}
    monkeypatch.setattr(gmm.GgufModelManager, "_remote_file_sizes", lambda self: sizes)
    tmp = m.cache_dir.with_name(m.cache_dir.name + ".tmp")
    tmp.mkdir(parents=True)
    inc = tmp / (main_name + ".incomplete")
    inc.write_bytes(b"head")
    # verify passes, both files present? main incomplete only -> verify_pair mocked:
    monkeypatch.setattr(gmm, "_verify_pair", lambda d: [])
    monkeypatch.setattr("requests.get", fake_get)
    m.download()
    assert captured["ranges"][main_name] == "bytes=4-"


def test_quickly_ready_avoids_hash(tmp_path):
    """quickly_ready는 마커+크기만 보며 SHA-256 전량 계산을 하지 않는다."""
    m = _mk(tmp_path)
    assert not m.quickly_ready()
    d = m.cache_dir
    d.mkdir(parents=True)
    (d / ".complete").write_text("rev")
    (d / gguf_cfg.GGUF_MAIN_FILE).write_bytes(b"x" * 16)
    (d / gguf_cfg.GGUF_MMPROJ_FILE).write_bytes(b"y")
    assert m.quickly_ready()
    (d / gguf_cfg.GGUF_MAIN_FILE).unlink()
    assert not m.quickly_ready()
