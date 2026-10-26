"""
Rename BrainVision rest files from 'hx_XX_rest_ZZ' to the default 'hx_XX_ZZ_task_rest'.

Renames the .vhdr/.eeg/.vmrk trio together and fixes the DataFile=/MarkerFile=
lines inside the .vhdr/.vmrk text so they still point at each other.

Usage:
    python rename_rest_files.py /path/to/main_folder          # preview only
    python rename_rest_files.py /path/to/main_folder --apply  # actually rename
"""
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"^(px_\d+)_rest_(pre|post)$", re.IGNORECASE)


def new_name(old_stem: str):
    m = PATTERN.match(old_stem)
    if not m:
        return None
    subj, zz = m.groups()
    return f"{subj}_{zz.lower()}_task_rest"


def rename_trio(vhdr_path: Path, apply: bool):
    old_stem = vhdr_path.stem
    stem = new_name(old_stem)
    if stem is None:
        return

    print(f"{vhdr_path.parent / (old_stem + '.*')}  ->  {stem}.*")
    if not apply:
        return

    for ext in (".vhdr", ".eeg", ".vmrk"):
        src = vhdr_path.with_name(old_stem + ext)
        if src.exists():
            src.rename(src.with_name(stem + ext))

    # fix internal DataFile=/MarkerFile= references (they use bare filenames)
    for ext in (".vhdr", ".vmrk"):
        f = vhdr_path.with_name(stem + ext)
        if f.exists():
            text = f.read_text(encoding="utf-8", errors="replace")
            text = text.replace(old_stem + ".eeg", stem + ".eeg")
            text = text.replace(old_stem + ".vmrk", stem + ".vmrk")
            f.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    apply = "--apply" in sys.argv

    for vhdr in root.glob("px_*/rest/*.vhdr"):
        rename_trio(vhdr, apply)

    if not apply:
        print("\nDry run only — nothing renamed. Re-run with --apply to do it.")
