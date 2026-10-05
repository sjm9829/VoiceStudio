"""worker stdout JSONL 수신 버퍼.

QProcess readyRead는 chunk 단위로 들어오므로 완결된 줄만 파싱하고
반쪽 JSON은 다음 chunk와 이어 붙인다.
"""

from __future__ import annotations
import json

class JsonlBuffer:
    def __init__(self):
        self._buf = ""

    def feed(self, data: bytes) -> list[dict]:
        text = self._buf + data.decode("utf-8", "replace")
        lines = text.split("\n")
        self._buf = lines.pop()  # 마지막 조각은 아직 완결되지 않았을 수 있다
        events: list[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue  # 손상된 줄은 버리고 계속(로그에는 stderr로 남는다)
            if isinstance(ev, dict):
                events.append(ev)
        return events
