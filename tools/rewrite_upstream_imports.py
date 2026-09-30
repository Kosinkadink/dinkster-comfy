from __future__ import annotations

import argparse
import re
from pathlib import Path


REWRITES = (
    (re.compile(r"\bcomfy\."), "dinkster_inference."),
    (re.compile(r"\bfrom comfy\b"), "from dinkster_inference"),
    (re.compile(r"\bimport comfy\b"), "import dinkster_inference"),
)


def rewrite(path: Path) -> bool:
    original = path.read_text()
    updated = original
    for pattern, replacement in REWRITES:
        updated = pattern.sub(replacement, updated)
    if updated == original:
        return False
    path.write_text(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite upstream ComfyUI Python imports for dinkster_inference."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    changed = 0
    for root in args.paths:
        files = (root.rglob("*.py") if root.is_dir() else (root,))
        for path in files:
            changed += rewrite(path)
    print(f"rewrote {changed} files")  # noqa: T201


if __name__ == "__main__":
    main()
