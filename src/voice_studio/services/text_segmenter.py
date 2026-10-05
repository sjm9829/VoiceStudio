"""Korean-first 대본 분할기.

- 문단(빈 줄)과 문장 부호를 우선 존중한다.
- 너무 짧은 문장은 이웃 문장과 합친다.
- 목표/상한 길이는 core.config 상수로 분리되어 실제 샘플 테스트 후 조정한다.
"""

from __future__ import annotations
import re
from ..core import config

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。])\s+")

def split_paragraphs(script: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", script.strip()) if p.strip()]

def split_sentences(paragraph: str) -> list[str]:
    flat = paragraph.replace("\n", " ").strip()
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(flat) if s.strip()]
    return parts or ([flat] if flat else [])

def _force_split(sentence: str, hard_max: int) -> list[str]:
    out: list[str] = []
    while len(sentence) > hard_max:
        cut = sentence.rfind(" ", 0, hard_max)
        cut = cut if cut > 0 else hard_max
        out.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        out.append(sentence)
    return out

def segment_with_flags(script: str, *, target: int = config.TARGET_SEGMENT_CHARS,
                       hard_max: int = config.HARD_MAX_SEGMENT_CHARS,
                       min_chars: int = config.MIN_SEGMENT_CHARS) -> tuple[list[str], list[bool]]:
    """(구간 목록, 구간 앞 문단 경계 여부) 반환. flags[0]은 항상 False(앞에 간격 없음)."""
    segments: list[str] = []
    flags: list[bool] = []
    paragraphs = split_paragraphs(script)
    for p_idx, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        chunks: list[str] = []
        buf = ""
        for sentence in sentences:
            for piece in _force_split(sentence, hard_max):
                candidate = (buf + " " + piece).strip() if buf else piece
                if len(candidate) > target:
                    if buf:
                        chunks.append(buf)
                    buf = piece
                else:
                    buf = candidate
        if buf:
            chunks.append(buf)
        # 문단 내 첫 chunk가 min_chars보다 짧으면 다음 문단과 합치지 않는다(문단 경계 존중).
        for c_idx, chunk in enumerate(chunks):
            segments.append(chunk)
            flags.append(p_idx > 0 and c_idx == 0)
    return segments, flags

def segment_script(script: str, **kw) -> list[str]:
    return segment_with_flags(script, **kw)[0]

def segment_gap_flags(script: str, **kw) -> list[bool]:
    """chunk 사이 무음 간격 플래그: True면 문단 경계(PARAGRAPH_GAP_MS) 간격."""
    return segment_with_flags(script, **kw)[1]
