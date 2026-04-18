"""
Déplace toutes les lignes des CSV `reportLive/safeChase/*.csv` vers
`Datas/csv/<même nom>`, puis réécrit le CSV live avec uniquement son header.

Usage: python rotate_safechase_csv.py [--dry-run]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LIVE_DIR = BASE / "reportLive" / "safeChase"
ARCHIVE_DIR = BASE / "Datas" / "csv"
TS_COL = "timestamp"


def process_file(path: Path, dry_run: bool = False) -> int:
    arch = ARCHIVE_DIR / path.name
    tmp_live = path.with_suffix(path.suffix + ".tmp")

    n_archive = 0

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        header_line = f.readline()
        if not header_line:
            return 0
        header = header_line.rstrip("\n\r")
        fields = header.split(",")
        if not fields or fields[0].strip() != TS_COL:
            raise SystemExit(f"{path}: première colonne attendue {TS_COL!r}, obtenu {fields[0]!r}")

        if dry_run:
            for line in f:
                raw = line.rstrip("\n\r")
                if not raw:
                    continue
                n_archive += 1
            return n_archive

        arch_exists = arch.exists() and arch.stat().st_size > 0
        mode_arch = "a" if arch_exists else "w"
        dst_live = open(tmp_live, "w", encoding="utf-8", newline="\n")
        dst_arch = open(arch, mode_arch, encoding="utf-8", newline="\n")
        try:
            dst_live.write(header + "\n")
            if mode_arch == "w":
                dst_arch.write(header + "\n")

            for line in f:
                raw = line.rstrip("\n\r")
                if not raw:
                    continue
                dst_arch.write(raw + "\n")
                n_archive += 1
        finally:
            dst_live.close()
            dst_arch.close()

    # Fermer le fichier source avant replace (requis sous Windows).
    os.replace(tmp_live, path)
    return n_archive


def main() -> None:
    dry = "--dry-run" in sys.argv
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(LIVE_DIR.glob("*.csv"))
    if not csvs:
        print(f"Aucun CSV dans {LIVE_DIR}")
        return

    for p in csvs:
        a = process_file(p, dry_run=dry)
        if dry:
            print(f"[dry-run] {p.name}: archiver ~{a} lignes, live ensuite vide (header seulement)")
        else:
            print(f"{p.name}: archivées +{a}, live reset header-only -> {ARCHIVE_DIR / p.name}")


if __name__ == "__main__":
    main()
