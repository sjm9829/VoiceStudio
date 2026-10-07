"""P13 validation correction: prepare_ffmpeg / source PATH-independent FFmpeg 회귀.

테스트는 실제 인터넷에 의존하지 않고 downloader/extractor/verifier를 주입한다(P13 §8).
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_prepare():
    spec = importlib.util.spec_from_file_location("prepare_ffmpeg", ROOT / "scripts" / "prepare_ffmpeg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_helpers():
    spec = importlib.util.spec_from_file_location("runtime_validation_helpers", ROOT / "scripts" / "runtime_validation_helpers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeArchive:
    """download → verify → extract 흐름을 흉내내는 mock."""

    def __init__(self, payload: bytes = b"FFMPEG-STATIC-PAYLOAD", fail_extract: bool = False):
        self.payload = payload
        self.fail_extract = fail_extract
        self.download_calls = 0
        self.sha = hashlib.sha256(payload).hexdigest()

    def downloader(self, url: str, dest: Path) -> None:
        self.download_calls += 1
        assert url == _load_prepare().FFMPEG_URL, "고정 URL만 사용해야 한다"
        dest.write_bytes(self.payload)

    def verifier(self, path: Path, expected: str) -> None:
        import hashlib
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, "SHA-256 검증 실패"
        assert expected == self.sha

    def extractor(self, archive: Path, out_dir: Path) -> None:
        if self.fail_extract:
            raise RuntimeError("extract failed")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ffmpeg.exe").write_bytes(self.payload)
        (out_dir / "ffprobe.exe").write_bytes(self.payload)


def _write_pair(bin_dir: Path, marker: bytes = b"GOOD-EXISTING") -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "ffmpeg.exe").write_bytes(marker)
    (bin_dir / "ffprobe.exe").write_bytes(marker)


def test_a_existing_pair_skips_download(tmp_path):
    """A. 정상 pair가 이미 있으면 download 호출 0회."""
    mod = _load_prepare()
    bin_dir = tmp_path / "bin"
    _write_pair(bin_dir)
    fake = FakeArchive()
    mod.prepare(bin_dir=bin_dir, downloader=fake.downloader, extractor=fake.extractor,
                verifier=fake.verifier, expected_sha256=fake.sha)
    assert fake.download_calls == 0


def test_b_missing_pair_downloads_verifies_places(tmp_path):
    """B. 둘 다 없으면 download → SHA-256 verify → extract → 두 exe 배치."""
    mod = _load_prepare()
    bin_dir = tmp_path / "bin"
    fake = FakeArchive()
    mod.prepare(bin_dir=bin_dir, downloader=fake.downloader, extractor=fake.extractor,
                verifier=fake.verifier, expected_sha256=fake.sha)
    assert fake.download_calls == 1
    assert (bin_dir / "ffmpeg.exe").read_bytes() == fake.payload
    assert (bin_dir / "ffprobe.exe").read_bytes() == fake.payload


def test_c_one_missing_counts_as_incomplete(tmp_path):
    """C. 한 개만 있으면 incomplete로 판단하고 정상 pair를 구성한다."""
    mod = _load_prepare()
    bin_dir = tmp_path / "bin"
    _write_pair(bin_dir)
    (bin_dir / "ffprobe.exe").unlink()
    fake = FakeArchive(payload=b"NEW-PAIR")
    mod.prepare(bin_dir=bin_dir, downloader=fake.downloader, extractor=fake.extractor,
                verifier=fake.verifier, expected_sha256=fake.sha)
    assert fake.download_calls == 1
    assert (bin_dir / "ffmpeg.exe").read_bytes() == b"NEW-PAIR"
    assert (bin_dir / "ffprobe.exe").read_bytes() == b"NEW-PAIR"


def test_d_bad_sha_fails_and_preserves_existing(tmp_path):
    """D. SHA-256 mismatch면 실패하고 기존 정상 binary를 훼손하지 않는다."""
    mod = _load_prepare()
    bin_dir = tmp_path / "bin"
    _write_pair(bin_dir, marker=b"GOOD")
    (bin_dir / "ffprobe.exe").unlink()  # incomplete → prepare 시도

    def bad_verifier(path, expected):
        raise mod.PrepareFfmpegError("SHA-256 mismatch")

    try:
        mod.prepare(bin_dir=bin_dir, downloader=FakeArchive().downloader,
                    extractor=FakeArchive().extractor, verifier=bad_verifier)
        raise AssertionError("bad sha must fail")
    except mod.PrepareFfmpegError:
        pass
    assert (bin_dir / "ffmpeg.exe").read_bytes() == b"GOOD"
    leftovers = [p.name for p in bin_dir.iterdir()]
    assert leftovers == ["ffmpeg.exe"], leftovers  # staging/partial 없음


def test_e_partial_download_leaves_no_pollution(tmp_path):
    """E. partial download → third_party/bin 오염 없음."""
    mod = _load_prepare()
    bin_dir = tmp_path / "bin"

    def partial_downloader(url, dest):
        dest.with_suffix(dest.suffix + ".part").write_bytes(b"PART")
        raise mod.PrepareFfmpegError("connection reset")

    try:
        mod.prepare(bin_dir=bin_dir, downloader=partial_downloader,
                    extractor=FakeArchive().extractor, verifier=FakeArchive().verifier)
        raise AssertionError("partial download must fail")
    except mod.PrepareFfmpegError:
        pass
    assert list(bin_dir.iterdir()) == [] or not bin_dir.exists()


def test_f_extract_failure_preserves_existing(tmp_path):
    """F. archive extract 실패 → 기존 정상 파일 보존."""
    mod = _load_prepare()
    bin_dir = tmp_path / "bin"
    _write_pair(bin_dir, marker=b"GOOD")
    (bin_dir / "ffmpeg.exe").unlink()  # incomplete
    fake = FakeArchive(fail_extract=True)
    try:
        mod.prepare(bin_dir=bin_dir, downloader=fake.downloader, extractor=fake.extractor,
                    verifier=fake.verifier, expected_sha256=fake.sha)
        raise AssertionError("extract failure must fail")
    except Exception:
        pass
    assert (bin_dir / "ffprobe.exe").read_bytes() == b"GOOD"
    assert not (bin_dir / "ffmpeg.exe").exists()


def test_g_fixed_url_and_sha256_constants():
    """고정 version/artifact/SHA-256이며 latest URL이 아니다(P13 §1C)."""
    mod = _load_prepare()
    assert "latest" not in mod.FFMPEG_URL
    assert "win64-lgpl" in mod.FFMPEG_URL
    assert mod.FFMPEG_URL.startswith("https://github.com/BtbN/FFmpeg-Builds/releases/download/")
    assert len(mod.FFMPEG_SHA256) == 64
    int(mod.FFMPEG_SHA256, 16)  # hex
    assert mod.FFMPEG_SIZE_BYTES > 0


def test_h_build_windows_runs_prepare_before_check_ffmpeg():
    """build_windows.bat는 prepare_ffmpeg.py → check_ffmpeg.py 순서다(P13 §3, §8H)."""
    src = (ROOT / "scripts" / "build_windows.bat").read_text(encoding="ascii")
    i_prep = src.index("prepare_ffmpeg.py")
    i_check = src.index("check_ffmpeg.py")
    assert i_prep < i_check
    assert "if not exist third_party\\bin\\ffmpeg.exe" not in src
    i_gate = src.index('-m "not gpu and not stt"')
    assert i_check < i_gate


def test_i_source_validation_is_path_independent(monkeypatch, tmp_path):
    """PATH에서 ffmpeg를 제거해도 validation helper는 third_party/bin을 쓴다(P13 §8I)."""
    import voice_studio.infra.ffmpeg_adapter as fa
    monkeypatch.setattr(shutil, "which", lambda name: None)
    helpers = _load_helpers()
    bin_dir = tmp_path / "bin"
    _write_pair(bin_dir)
    ffmpeg, ffprobe = helpers.validation_ffmpeg_paths(bin_dir)
    assert ffmpeg == str(bin_dir / "ffmpeg.exe")
    assert ffprobe == str(bin_dir / "ffprobe.exe")
    adapter = helpers.make_validation_adapter(bin_dir)
    assert adapter.ffmpeg == str(bin_dir / "ffmpeg.exe")
    assert adapter.ffprobe == str(bin_dir / "ffprobe.exe")


def test_j_missing_binaries_give_prepare_instruction(tmp_path):
    """binary가 없으면 prepare_ffmpeg.py 실행 안내와 함께 실패한다."""
    helpers = _load_helpers()
    try:
        helpers.validation_ffmpeg_paths(tmp_path / "empty")
        raise AssertionError("missing binaries must raise")
    except helpers.ValidationFfmpegMissing as exc:
        assert "prepare_ffmpeg.py" in str(exc)


def test_k_repo_third_party_bin_is_the_default():
    """기본 bin_dir는 repository의 third_party/bin이다."""
    helpers = _load_helpers()
    assert helpers.REPO_BIN_DIR == ROOT / "third_party" / "bin"
