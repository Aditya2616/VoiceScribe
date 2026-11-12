"""Copy the web UI (with its recorded demo run) into docs/ so it can be hosted as a static site."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB, DOCS = ROOT / "web", ROOT / "docs"


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for src in WEB.rglob("*"):
        if src.is_file():
            dst = DOCS / src.relative_to(WEB)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    (DOCS / ".nojekyll").write_text("", encoding="utf8")
    print(f"copied {WEB} -> {DOCS}")


if __name__ == "__main__":
    main()
