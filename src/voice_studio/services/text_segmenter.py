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
        # 너무 짧은 chunk를 같은 문단 내 이웃과 합친다(docs "너무 짧은 문장은
        # 이웃 문장과 합친다" 계약, P13 §23). hard_max를 넘기지 않고,
        # 문단 경계를 넘는 merge는 하지 않는다.
        chunks = _merge_short_chunks(chunks, hard_max=hard_max, min_chars=min_chars)
        for c_idx, chunk in enumerate(chunks):
            segments.append(chunk)
            flags.append(p_idx > 0 and c_idx == 0)
    return segments, flags

def _merge_short_chunks(chunks: list[str], *, hard_max: int,
                        min_chars: int) -> list[str]:
    """문단 내 인접 chunk를 앞쪽으로 합쳐 min_chars 이상으로 만든다.

    - 문단 경계 merge는 하지 않는다(segment_with_flags에서 문단별로 호출).
    - 합친 결과가 hard_max를 초과하면 merge하지 않는다.
    - gap_flags semantics 유지: 문단 내부 merge는 gap(False) 구간끼리만
      결합되므로 flags 재계산 결과는 동일한 문단 경계 패턴을 유지한다.
    """
    if min_chars <= 0 or not chunks:
        return chunks
    merged: list[str] = []
    buf = ""
    for chunk in chunks:
        candidate = f"{buf} {chunk}" if buf else chunk
        if buf and len(buf) < min_chars:
            if len(candidate) <= hard_max:
                buf = candidate
                continue
            merged.append(buf)
            buf = chunk
        else:
            if buf:
                merged.append(buf)
            buf = chunk
    if buf:
        merged.append(buf)
    return merged

def segment_script(script: str, **kw) -> list[str]:
    return segment_with_flags(script, **kw)[0]

def segment_gap_flags(script: str, **kw) -> list[bool]:
    """chunk 사이 무음 간격 플래그: True면 문단 경계(PARAGRAPH_GAP_MS) 간격."""
    return segment_with_flags(script, **kw)[1]
