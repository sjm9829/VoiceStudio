"""애플리케이션 수준 worker 동시성 제한(P12.3-25).

목표 하드웨어(RTX 2070 SUPER 8GB)에서 Qwen GPU worker는 최대 1개만 허용한다.
register worker(음성 등록)와 narrate worker(대본 생성)가 동시에 뜨는 것을
AppContext에 공유된 coordinator 하나로 막는다. IPC lock은 불필요하다.
"""


class JobCoordinator:
    """단순 카운터 기반 단일 worker 게이트. 재진입(reentrant)이 아니다."""

    def __init__(self):
        self._active = False

    def try_acquire(self) -> bool:
        """worker 슬롯을 얻는다. 이미 실행 중이면 False."""
        if self._active:
            return False
        self._active = True
        return True

    def release(self) -> None:
        self._active = False

    @property
    def busy(self) -> bool:
        return self._active
