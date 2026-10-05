import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from voice_studio.infra.profile_repository import ProfileRepository
from voice_studio.core.errors import ProfileError
from voice_studio.domain.voice_profile import VoiceProfile

def tensors():
    return {"ref_code": np.array([5.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            "ref_spk_embedding": np.random.default_rng(0).normal(size=256).astype(np.float32)}

def test_save_and_load_roundtrip(tmp_path):
    repo = ProfileRepository(tmp_path)
    p = VoiceProfile(name="테스트", ref_text="대사입니다.")
    d = repo.save(p, tensors(), b"FAKEFLAC")
    assert (d / "metadata.json").is_file()
    assert (d / "prompt.safetensors").is_file()
    assert (d / "reference.flac").is_file()
    loaded = repo.get(p.uuid)
    assert loaded.name == "테스트" and loaded.ref_text == "대사입니다."
    t = repo.load_prompt_tensors(p.uuid)
    assert t["ref_spk_embedding"].dtype == np.float32 and t["ref_spk_embedding"].ndim == 1

def test_metadata_schema_version_required(tmp_path):
    repo = ProfileRepository(tmp_path)
    p = VoiceProfile(name="스키마")
    d = repo.save(p, tensors(), b"f")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    assert "schema_version" in meta and meta["schema_version"] >= 1

def test_future_schema_rejected(tmp_path):
    repo = ProfileRepository(tmp_path)
    p = VoiceProfile(name="미래")
    d = repo.save(p, tensors(), b"f")
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    meta["schema_version"] = 99
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    try:
        repo.get(p.uuid)
        assert False, "future schema must fail"
    except ProfileError:
        pass

def test_legacy_v0_metadata_migrates(tmp_path):
    repo = ProfileRepository(tmp_path)
    d = tmp_path / "legacy"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps({"name": "옛 프로필", "uuid": "legacy"}), encoding="utf-8")
    p = repo.get("legacy")
    assert p.icl_mode is True and p.x_vector_only_mode is False

def test_delete_removes_dir(tmp_path):
    repo = ProfileRepository(tmp_path)
    p = VoiceProfile(name="삭제대상")
    repo.save(p, tensors(), b"f")
    repo.delete(p.uuid)
    assert not (tmp_path / p.uuid).exists()

def test_missing_profile_raises(tmp_path):
    repo = ProfileRepository(tmp_path)
    try:
        repo.get("없는uuid")
        assert False
    except ProfileError:
        pass

def test_bad_embedding_dtype_rejected(tmp_path):
    repo = ProfileRepository(tmp_path)
    p = VoiceProfile(name="잘못된dtype")
    bad = tensors(); bad["ref_spk_embedding"] = bad["ref_spk_embedding"].astype(np.float64)
    try:
        repo.save(p, bad, b"f")
        assert False
    except ProfileError:
        pass

def test_list_profiles(tmp_path):
    repo = ProfileRepository(tmp_path)
    for i in range(3):
        repo.save(VoiceProfile(name=f"목소리{i}"), tensors(), b"f")
    assert len(repo.list_profiles()) == 3
