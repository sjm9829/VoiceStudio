"""PyInstaller spec용 순수 헬퍼(P12.3-01/02/22).

spec 실행 규칙(SPECPATH = packaging/ 디렉터리)과 collect_all() 반환 계약
(datas, binaries, hiddenimports 순서)을 테스트 가능한 형태로 분리한다.
PyInstaller는 함수 내부에서 지연 import하므로 개발 환경 테스트에서도 이 모듈을
import할 수 있다.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable


def compute_repo_root(specpath: str) -> Path:
    """SPECPATH(packaging 디렉터리) 기준 repository root를 계산한다.

    실제 PyInstaller는 SPECPATH를 spec 파일이 있는 디렉터리로 정의하므로
    repository root는 그 부모 디렉터리다(P12.3-02).
    """
    return Path(specpath).resolve().parent


def collect_package(name: str) -> tuple[list, list, list]:
    """collect_all() 반환 계약을 올바른 순서로 반환한다(P12.3-01).

    PyInstaller 계약: datas, binaries, hiddenimports = collect_all(package)
    """
    from PyInstaller.utils.hooks import collect_all
    return collect_all(name)


def apply_collect(a, name: str, *, collector: Callable[[str], tuple] | None = None) -> None:
    """collect_all 결과를 Analysis 객체에 올바르게 연결한다(P12.3-22).

    테스트에서는 collector를 주입해 실제 PyInstaller 없이 매핑 계약을 검증한다.
    """
    datas, binaries, hiddenimports = (collector or collect_package)(name)
    a.datas += list(datas)
    a.binaries += list(binaries)
    existing = set(a.hiddenimports)
    a.hiddenimports += [h for h in hiddenimports if h not in existing]
