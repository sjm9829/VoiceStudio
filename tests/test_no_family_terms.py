import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathlib import Path as P

def test_no_family_specific_terms_in_code_and_ui():
    """특정 사용자/가족 표현(아빠/아버지/father)이 src와 README에 0건인지 검사."""
    root = P(__file__).resolve().parents[1]
    banned = ("아빠", "아버지", "father")
    hits = []
    for base in (root / "src", root / "README.md"):
        files = base.rglob("*.py") if base.is_dir() else [base]
        for p in files:
            text = p.read_text(encoding="utf-8").lower()
            for w in banned:
                if w in text:
                    hits.append(f"{p.relative_to(root)}:{w}")
    assert hits == [], hits
