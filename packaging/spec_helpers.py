"""PyInstaller spec용 순수 헬퍼(P12.3-01/02/22, P12.3 Final Hotfix).

spec 실행 규칙(SPECPATH = packaging/ 디렉터리)과 collect_all() 반환 계약
(datas, binaries, hiddenimports 순서)을 테스트 가능한 형태로 분리한다.
PyInstaller는 함수 내부에서 지연 import하므로 개발 환경 테스트에서도 이 모듈을
import할 수 있다.

Hotfix 계약: helper는 Analysis 객체를 수정하지 않는다. datas / binaries /
hiddenimports list를 Analysis 생성 **전에** 완성하는 데만 사용한다
(PyInstaller의 import graph 분석은 Analysis(...) 생성 중에 수행된다).
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable


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


def merge_collect(
    package: str,
    datas: list,
    binaries: list,
    hiddenimports: list,
    *,
    collector: Callable[[str], tuple] | None = None,
) -> None:
    """collect_all 결과를 Analysis 생성 전의 list에 병합한다(Final Hotfix).

    테스트에서는 collector를 주입해 실제 PyInstaller 없이 매핑 계약을 검증한다.
    반환 계약: datas → datas, binaries → binaries, hiddenimports → hiddenimports.
    """
    pkg_datas, pkg_binaries, pkg_hiddenimports = (collector or collect_package)(package)
    datas.extend(list(pkg_datas))
    binaries.extend(list(pkg_binaries))
    existing = set(hiddenimports)
    for item in pkg_hiddenimports:
        if item not in existing:
            hiddenimports.append(item)
            existing.add(item)
