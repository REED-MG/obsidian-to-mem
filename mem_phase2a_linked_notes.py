#!/usr/bin/env python3
"""
mem_phase2a_linked_notes.py

Utility script to gather every note mentioned in note_refs.csv
(both source_file and target_file columns) and move those files
into a dedicated "Linked Notes" folder so they can be reviewed
or processed separately. Notes are flattened into the destination
folder, and internal note links are updated to remove sub-folder
paths.

USAGE:

    python mem_phase2a_linked_notes.py --root /path/to/vault

By default the script expects note_refs.csv inside the root folder
and moves the notes into "<root>/Linked Notes/". Use --refs-csv or
--linked-folder-name to override those defaults. Pass --dry-run to
preview which files would move without altering anything.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple


WIKI_LINK_RE = re.compile(r"(!)?\[\[([^\]|]+?)(?:\|([^\]]*))?\]\]")
MD_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move every note referenced in note_refs.csv into 'Linked Notes'."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to the vault root that contains the Markdown files.",
    )
    parser.add_argument(
        "--refs-csv",
        help="Path to note_refs.csv (defaults to <root>/note_refs.csv).",
    )
    parser.add_argument(
        "--linked-folder-name",
        default="Linked Notes",
        help="Name of the destination folder created under the root. Default: 'Linked Notes'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the moves that would happen without changing any files.",
    )
    parser.add_argument(
        "--log-path",
        help=(
            "Path to write a detailed log file. Defaults to "
            "<root>/linked_notes_move_log.txt."
        ),
    )
    return parser.parse_args()


def build_flatten_map(
    dest_root: Path,
    relative_paths: Iterable[Path],
) -> Tuple[Dict[Path, str], int]:
    name_map: Dict[Path, str] = {}
    reserved: Set[str] = set()
    collisions = 0

    for rel in sorted({Path(p) for p in relative_paths}):
        desired = rel.name
        stem = Path(desired).stem
        suffix = Path(desired).suffix
        candidate = desired
        counter = 2
        while candidate in reserved or (dest_root / candidate).exists():
            candidate = f"{stem} ({counter}){suffix}"
            counter += 1
        if candidate != desired:
            collisions += 1
        name_map[rel] = candidate
        reserved.add(candidate)

    return name_map, collisions


def gather_note_paths(csv_path: Path) -> Tuple[Set[Path], Set[Path], Set[Path], int]:
    """
    Return (all_paths, source_paths, target_paths, row_count) from note_refs.csv.
    """
    needed_columns = {"source_file", "target_file"}
    note_paths: Set[Path] = set()
    source_paths: Set[Path] = set()
    target_paths: Set[Path] = set()
    row_count = 0

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = needed_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            row_count += 1
            source_rel = (row.get("source_file") or "").strip()
            target_rel = (row.get("target_file") or "").strip()
            if source_rel:
                source_paths.add(Path(source_rel))
                note_paths.add(Path(source_rel))
            if target_rel:
                target_paths.add(Path(target_rel))
                note_paths.add(Path(target_rel))

    return note_paths, source_paths, target_paths, row_count


def move_notes(
    root: Path,
    dest_root: Path,
    relative_paths: Iterable[Path],
    name_map: Dict[Path, str],
    dry_run: bool = False,
    log_handle=None,
    mtime_map: Optional[Dict[Path, Tuple[float, float]]] = None,
) -> None:
    moves: List[str] = []
    missing: List[str] = []

    for rel in sorted({Path(p) for p in relative_paths}):
        src = root / rel
        if not src.exists():
            missing.append(str(rel))
            if log_handle:
                log_handle.write(f"[MISSING] {rel}\n")
            continue

        dest_name = name_map[rel]
        dest = dest_root / dest_name
        if mtime_map is not None:
            stat = src.stat()
            mtime_map[rel] = (stat.st_atime, stat.st_mtime)
        moves.append(f"{src} -> {dest}")
        if log_handle:
            log_handle.write(f"[MOVE] {src} -> {dest}\n")
        if dry_run:
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        if mtime_map is not None:
            os.utime(dest, mtime_map[rel])

    if dry_run:
        print("[DRY-RUN] Planned moves:")
        for line in moves:
            print(f"  {line}")
    else:
        print(f"Moved {len(moves)} files into '{dest_root}'.")

    if missing:
        print(f"Warning: {len(missing)} files listed in CSV were not found under {root}:")
        for rel in missing:
            print(f"  {rel}")


def split_fragment(target: str) -> Tuple[str, str]:
    if "#" in target:
        base, frag = target.split("#", 1)
        return base, f"#{frag}"
    return target, ""


def build_link_maps(name_map: Dict[Path, str]) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    wiki_map: Dict[str, str] = {}
    md_map: Dict[str, Tuple[str, str]] = {}

    for rel, dest_name in name_map.items():
        rel_posix = rel.as_posix()
        rel_no_ext = rel.with_suffix("").as_posix()
        dest_stem = Path(dest_name).stem
        md_map[rel_posix] = (dest_name, dest_stem)
        md_map[rel_no_ext] = (dest_name, dest_stem)
        wiki_map[rel_posix] = dest_stem
        wiki_map[rel_no_ext] = dest_stem

    return wiki_map, md_map


def rewrite_links_in_note(
    path: Path,
    wiki_map: Dict[str, str],
    md_map: Dict[str, Tuple[str, str]],
    dry_run: bool,
    log_handle=None,
    mtime: Optional[Tuple[float, float]] = None,
) -> Tuple[int, bool]:
    content = path.read_text(encoding="utf-8")
    updated = 0

    def replace_wiki(match: re.Match) -> str:
        nonlocal updated
        prefix = "!" if match.group(1) else ""
        target = match.group(2)
        alias = match.group(3)
        base, frag = split_fragment(target)
        new_base = wiki_map.get(base)
        if not new_base:
            return match.group(0)
        updated += 1
        new_target = f"{new_base}{frag}"
        if alias:
            return f"{prefix}[[{new_target}|{alias}]]"
        return f"{prefix}[[{new_target}]]"

    def replace_md(match: re.Match) -> str:
        nonlocal updated
        prefix, label, url = match.group(1), match.group(2), match.group(3).strip()
        if "://" in url or url.startswith("mailto:"):
            return match.group(0)
        base, frag = split_fragment(url)
        decoded = urllib.parse.unquote(base)
        mapped = md_map.get(decoded)
        if not mapped:
            return match.group(0)
        new_name, new_stem = mapped
        use_full = Path(decoded).suffix.lower() == ".md"
        new_base = new_name if use_full else new_stem
        updated += 1
        new_url = f"{new_base}{frag}"
        return f"{prefix}[{label}]({new_url})"

    new_content = MD_LINK_RE.sub(replace_md, content)
    new_content = WIKI_LINK_RE.sub(replace_wiki, new_content)

    if updated and not dry_run:
        path.write_text(new_content, encoding="utf-8")
        if mtime is not None:
            os.utime(path, mtime)
    if updated and log_handle:
        log_handle.write(f"[REWRITE] {path} ({updated} links)\n")
    return updated, updated > 0


def rewrite_links(
    root: Path,
    dest_root: Path,
    name_map: Dict[Path, str],
    mtime_map: Dict[Path, Tuple[float, float]],
    dry_run: bool,
    log_handle=None,
) -> Tuple[int, int]:
    wiki_map, md_map = build_link_maps(name_map)
    total_links = 0
    updated_files = 0

    for rel, dest_name in name_map.items():
        dest_path = dest_root / dest_name
        if dest_path.exists():
            path = dest_path
        elif dry_run:
            path = root / rel
        else:
            continue
        if not path.exists():
            continue
        updated, changed = rewrite_links_in_note(
            path,
            wiki_map,
            md_map,
            dry_run=dry_run,
            log_handle=log_handle,
            mtime=mtime_map.get(rel),
        )
        total_links += updated
        if changed:
            updated_files += 1

    return total_links, updated_files


def main() -> None:
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Root directory '{root}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.refs_csv).expanduser().resolve() if args.refs_csv else root / "note_refs.csv"
    if not csv_path.is_file():
        print(f"[ERROR] note_refs.csv not found at '{csv_path}'.", file=sys.stderr)
        sys.exit(1)

    dest_root = (root / args.linked_folder_name).resolve()
    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)

    note_paths, source_paths, target_paths, row_count = gather_note_paths(csv_path)
    if not note_paths:
        print(f"No linked notes found in {csv_path}. Nothing to do.")
        return

    log_path = (
        Path(args.log_path).expanduser().resolve()
        if args.log_path
        else root / "linked_notes_move_log.txt"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    name_map, collisions = build_flatten_map(dest_root, note_paths)

    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("Linked Notes Move Log\n")
        log_handle.write(f"Timestamp: {timestamp}\n")
        log_handle.write(f"Root: {root}\n")
        log_handle.write(f"CSV: {csv_path}\n")
        log_handle.write(f"Destination: {dest_root}\n")
        log_handle.write(f"Dry run: {args.dry_run}\n")
        log_handle.write("\n")
        log_handle.write(f"Rows in CSV: {row_count}\n")
        log_handle.write(f"Unique source notes: {len(source_paths)}\n")
        log_handle.write(f"Unique target notes: {len(target_paths)}\n")
        log_handle.write(f"Total unique notes to move: {len(note_paths)}\n")
        log_handle.write(f"Filename collisions resolved: {collisions}\n")
        log_handle.write("\n")
        log_handle.write("Moves:\n")

        mtime_map: Dict[Path, Tuple[float, float]] = {}
        print(f"Identified {len(note_paths)} unique files referenced in {csv_path}.")
        move_notes(
            root,
            dest_root,
            note_paths,
            name_map,
            dry_run=args.dry_run,
            log_handle=log_handle,
            mtime_map=mtime_map,
        )

        log_handle.write("\nLink rewrites:\n")
        rewritten_links, rewritten_files = rewrite_links(
            root,
            dest_root,
            name_map,
            mtime_map,
            dry_run=args.dry_run,
            log_handle=log_handle,
        )

        log_handle.write("\nSummary:\n")
        log_handle.write(f"Rows in CSV: {row_count}\n")
        log_handle.write(f"Unique source notes: {len(source_paths)}\n")
        log_handle.write(f"Unique target notes: {len(target_paths)}\n")
        log_handle.write(f"Total unique notes to move: {len(note_paths)}\n")
        log_handle.write(f"Filename collisions resolved: {collisions}\n")
        log_handle.write(f"Notes with link rewrites: {rewritten_files}\n")
        log_handle.write(f"Links rewritten: {rewritten_links}\n")
        log_handle.write(f"Dry run: {args.dry_run}\n")

    print(f"Wrote log to {log_path}")


if __name__ == "__main__":
    main()
