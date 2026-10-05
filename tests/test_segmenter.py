import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_studio.services.text_segmenter import (segment_script, segment_gap_flags,
                                                  split_paragraphs, split_sentences)

LONG = ("안녕하세요. 오늘은 날씨가 좋습니다. 산책을 나가볼까요? " * 20)

def test_short_script_one_segment():
    segs = segment_script("안녕하세요.")
    assert segs == ["안녕하세요."]

def test_paragraph_boundary_respected():
    script = "첫 번째 문단입니다. 이어지는 문장입니다.\n\n두 번째 문단입니다."
    segs = segment_script(script)
    assert all("두 번째" not in s or "첫 번째" not in s for s in segs)
    flags = segment_gap_flags(script)
    assert flags[0] is False
    # 두 번째 문단의 첫 구간 앞은 문단 경계 간격
    assert True in flags

def test_long_script_split_within_hard_max():
    segs = segment_script("문장입니다. " * 300)
    assert all(len(s) <= 240 for s in segs)
    assert len(segs) > 10

def test_gap_flags_align_with_segments():
    script = LONG + "\n\n" + "새 문단입니다. 또 다른 문장입니다."
    segs = segment_script(script)
    flags = segment_gap_flags(script)
    assert len(segs) == len(flags)
    assert flags[0] is False

def test_sentences_split_on_punctuation():
    assert split_sentences("안녕하세요. 반갑습니다! 잘 부탁합니다.") == [
        "안녕하세요.", "반갑습니다!", "잘 부탁합니다."]

def test_paragraph_split():
    assert len(split_paragraphs("a\n\nb\n\nc")) == 3

def test_target_range_default():
    from voice_studio.core import config
    assert 100 <= config.TARGET_SEGMENT_CHARS <= 200
    assert config.HARD_MAX_SEGMENT_CHARS >= config.TARGET_SEGMENT_CHARS
