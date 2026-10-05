import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from voice_studio.core.errors import (DuplicateNameError, ConsentRequiredError,
                                      TranscriptRequiredError)
from voice_studio.infra.qwen_adapter import VoiceClonePromptSpec

BASE = dict(name="내 목소리", source_path="x.wav", start_s=1.0, end_s=16.0,
            ref_text="안녕하세요. 테스트입니다.", consent=True)

def test_register_success(registered):
    service, repo, audio, qwen, p = registered
    assert p.name == "테스트 목소리"
    assert p.icl_mode is True and p.x_vector_only_mode is False
    assert p.reference_duration_ms > 0
    assert repo.get(p.uuid).ref_text == "안녕하세요. 테스트 대사입니다."

def test_duplicate_name_rejected(services):
    service, *_ = services
    service.register(**BASE)
    with pytest.raises(DuplicateNameError):
        service.register(**dict(BASE, source_path="y.wav"))

def test_consent_required(services):
    service, *_ = services
    with pytest.raises(ConsentRequiredError):
        service.register(**dict(BASE, consent=False))

def test_transcript_required(services):
    service, *_ = services
    with pytest.raises(TranscriptRequiredError):
        service.register(**dict(BASE, ref_text="  "))

def test_rename_and_conflict(registered):
    service, repo, audio, qwen, _ = registered
    u = _uuid(service)
    service.rename(u, "새 이름")
    assert service.get(u).name == "새 이름"
    # 다른 프로필과 이름 충돌은 막는다
    other = service.register(name="두번째", source_path="y.wav", start_s=1.0, end_s=16.0,
                             ref_text="대사", consent=True)
    with pytest.raises(DuplicateNameError):
        service.rename(u, "두번째")
    # 자기 자신의 원래 이름으로 되돌리는 것은 허용
    service.rename(u, "새 이름")

def _uuid(service):
    return service.list_profiles()[0].uuid

def test_ref_code_roundtrip(registered):
    from voice_studio.services.profile_service import _encode_ref_code, _decode_ref_code
    code = {"fake": True, "ref_text": "한글 대사"}
    assert _decode_ref_code(_encode_ref_code(code)) == code

def _tensors():
    import numpy as np
    return {"ref_code": np.array([5.0, 1.0, 2.0], dtype=np.float32),
            "ref_spk_embedding": np.zeros(256, dtype=np.float32)}

def test_load_prompt_spec_restores_ref_text(registered):
    service, *_ = registered
    u = _uuid(service)
    spec = service.load_prompt_spec(u)
    assert spec.ref_text == "안녕하세요. 테스트 대사입니다."
    assert spec.icl_mode is True
    assert spec.ref_spk_embedding is not None and spec.ref_spk_embedding.dtype.name == "float32"
    assert spec.ref_code is not None

def test_delete_then_list(registered):
    service, *_ = registered
    u = _uuid(service)
    service.delete(u)
    assert service.list_profiles() == []
