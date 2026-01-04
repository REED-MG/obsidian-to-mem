#!/usr/bin/env python3
"""
Match Mem duplicate rows to Mem note IDs by title + created/modified dates (stdlib-only).

Defaults:
  - Expects input CSVs in the same directory as this script.
  - Writes output CSV to the same directory.

You can override the root directory at runtime:
  --root-dir /path/to/folder

File names (overrideable):
  --notes-file mem_notes.csv
  --dups-file mem_import_duplicates.csv
  --output-file mem_import_duplicates_with_ids.csv

Inputs required columns:
  mem_notes.csv:
    - mem_id
    - title
    - created_at
    - updated_at

  mem_import_duplicates.csv:
    - sample_title
    - sample_created   (UK day-first, e.g. 27/07/2021 07:21)
    - sample_modified

Output:
  Adds columns including matched_mem_id and match diagnostics.
  If multiple candidates match, emits multiple rows (one per candidate).

Usage:
  python match_duplicate_ids.py
  python match_duplicate_ids.py --root-dir /Users/markreed/Downloads/mem
  python match_duplicate_ids.py --root-dir . --tol-seconds 120
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


WS_RE = re.compile(r"\s+")
SLUG_RE = re.compile(r"[^a-z0-9]+")


def norm_title(s: str) -> str:
    s = (s or "").strip()
    s = WS_RE.sub(" ", s)
    return s.casefold()


def slug_title(s: str) -> str:
    s = (s or "").casefold()
    return SLUG_RE.sub("", s)


def parse_uk_datetime(s: str) -> Optional[datetime]:
    """Parse UK-style dd/mm/yyyy hh:mm[:ss] into naive datetime."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def parse_iso_datetime_to_utc(s: str) -> Optional[datetime]:
    """Parse ISO-like Mem timestamps into aware UTC datetime (best-effort)."""
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class Note:
    mem_id: str
    title: str
    title_norm: str
    title_slug: str
    created_utc: Optional[datetime]
    updated_utc: Optional[datetime]
    created_at_raw: str
    updated_at_raw: str


@dataclass
class Candidate:
    note: Note
    dc_s: float
    du_s: float
    score_s: float


def best_diff_seconds(note_utc: Optional[datetime], dup_naive: Optional[datetime], london_tz) -> float:
    """
    Compute min abs diff seconds using two assumptions:
      - dup_naive is Europe/London local time
      - dup_naive is UTC
    """
    if note_utc is None or dup_naive is None:
        return float("inf")

    dup_london = dup_naive.replace(tzinfo=london_tz).astimezone(timezone.utc)
    dup_utc = dup_naive.replace(tzinfo=timezone.utc)

    d1 = abs((note_utc - dup_london).total_seconds())
    d2 = abs((note_utc - dup_utc).total_seconds())
    return min(d1, d2)


def load_notes(notes_csv: Path) -> Tuple[List[Note], Dict[str, List[int]], Dict[str, List[int]]]:
    notes: List[Note] = []
    idx_norm: Dict[str, List[int]] = {}
    idx_slug: Dict[str, List[int]] = {}

    with notes_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"mem_id", "title", "created_at", "updated_at"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(list(required - set(reader.fieldnames or [])))
            raise SystemExit(f"mem_notes CSV missing columns: {missing}")

        for i, row in enumerate(reader):
            title = row.get("title", "") or ""
            created_raw = row.get("created_at", "") or ""
            updated_raw = row.get("updated_at", "") or ""

            n = Note(
                mem_id=row.get("mem_id", "") or "",
                title=title,
                title_norm=norm_title(title),
                title_slug=slug_title(title),
                created_utc=parse_iso_datetime_to_utc(created_raw),
                updated_utc=parse_iso_datetime_to_utc(updated_raw),
                created_at_raw=created_raw,
                updated_at_raw=updated_raw,
            )
            notes.append(n)
            idx_norm.setdefault(n.title_norm, []).append(i)
            idx_slug.setdefault(n.title_slug, []).append(i)

    return notes, idx_norm, idx_slug


def load_dups(dups_csv: Path) -> Tuple[List[dict], List[str]]:
    with dups_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_title", "sample_created", "sample_modified"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = sorted(list(required - set(reader.fieldnames or [])))
            raise SystemExit(f"mem_import_duplicates CSV missing columns: {missing}")
        rows = list(reader)
        headers = list(reader.fieldnames or [])
    return rows, headers


def get_candidate_indices(
    dup_title: str,
    notes: List[Note],
    idx_norm: Dict[str, List[int]],
    idx_slug: Dict[str, List[int]],
    prefix_len: int,
) -> Tuple[List[int], str]:
    tn = norm_title(dup_title)
    ts = slug_title(dup_title)

    if tn in idx_norm:
        return idx_norm[tn], "title_norm_exact"
    if ts in idx_slug:
        return idx_slug[ts], "title_slug_exact"

    if ts:
        pref = ts[: min(prefix_len, len(ts))]
        if pref:
            out = [i for i, n in enumerate(notes) if (n.title_slug or "").startswith(pref)]
            if out:
                return out, f"title_slug_prefix_{len(pref)}"

    return [], "no_title_match"


def main() -> int:
    if ZoneInfo is None:
        print("ERROR: This script requires Python 3.9+ (zoneinfo).", file=sys.stderr)
        return 2

    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description="Match duplicate rows to Mem IDs (defaults to script directory).")
    ap.add_argument(
        "--root-dir",
        default=str(script_dir),
        help="Root directory for inputs/outputs (default: directory containing this script)",
    )
    ap.add_argument("--notes-file", default="mem_notes.csv", help="Notes CSV filename (default: mem_notes.csv)")
    ap.add_argument("--dups-file", default="mem_import_duplicates.csv", help="Duplicates CSV filename (default: mem_import_duplicates.csv)")
    ap.add_argument("--output-file", default="mem_import_duplicates_with_ids.csv", help="Output CSV filename")
    ap.add_argument("--tol-seconds", type=int, default=300, help="Date tolerance seconds (default 300 = 5 min)")
    ap.add_argument("--prefix-len", type=int, default=40, help="Slug prefix length for truncated title matching (default 40)")
    ap.add_argument("--max-fallback-candidates", type=int, default=5, help="If no strict match, emit top N closest (default 5)")
    args = ap.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    notes_csv = root / args.notes_file
    dups_csv = root / args.dups_file
    out_csv = root / args.output_file

    if not notes_csv.exists():
        print(f"ERROR: notes CSV not found: {notes_csv}", file=sys.stderr)
        return 2
    if not dups_csv.exists():
        print(f"ERROR: duplicates CSV not found: {dups_csv}", file=sys.stderr)
        return 2

    london = ZoneInfo("Europe/London")

    print(f"Using root dir: {root}", file=sys.stderr)
    print(f"Reading notes: {notes_csv}", file=sys.stderr)
    print(f"Reading duplicates: {dups_csv}", file=sys.stderr)

    notes, idx_norm, idx_slug = load_notes(notes_csv)
    dup_rows, dup_headers = load_dups(dups_csv)

    out_headers = dup_headers + [
        "matched_mem_id",
        "matched_note_title",
        "matched_created_at",
        "matched_updated_at",
        "title_match_method",
        "created_diff_seconds",
        "updated_diff_seconds",
        "match_score_seconds",
        "match_rank",
        "ambiguous_match",
    ]

    out_rows: List[dict] = []

    total = len(dup_rows)
    no_title_match = 0
    ambiguous_groups = 0

    for i, d in enumerate(dup_rows, start=1):
        if i == 1 or i % 25 == 0 or i == total:
            print(f"Progress: {i}/{total} duplicates processed...", file=sys.stderr)

        dup_title = d.get("sample_title", "") or ""
        dup_created = parse_uk_datetime(d.get("sample_created", "") or "")
        dup_modified = parse_uk_datetime(d.get("sample_modified", "") or "")

        cand_idxs, title_method = get_candidate_indices(dup_title, notes, idx_norm, idx_slug, args.prefix_len)

        if not cand_idxs:
            no_title_match += 1
            row = dict(d)
            row.update(
                {
                    "matched_mem_id": "",
                    "matched_note_title": "",
                    "matched_created_at": "",
                    "matched_updated_at": "",
                    "title_match_method": title_method,
                    "created_diff_seconds": "",
                    "updated_diff_seconds": "",
                    "match_score_seconds": "",
                    "match_rank": "",
                    "ambiguous_match": "False",
                }
            )
            out_rows.append(row)
            continue

        candidates: List[Candidate] = []
        for idx in cand_idxs:
            n = notes[idx]
            dc = best_diff_seconds(n.created_utc, dup_created, london)
            du = best_diff_seconds(n.updated_utc, dup_modified, london)
            candidates.append(Candidate(note=n, dc_s=dc, du_s=du, score_s=dc + du))

        strict = [c for c in candidates if c.dc_s <= args.tol_seconds and c.du_s <= args.tol_seconds]
        use = sorted(strict, key=lambda c: c.score_s) if strict else sorted(candidates, key=lambda c: c.score_s)[: args.max_fallback_candidates]
        ambiguous = len(strict) > 1
        if ambiguous:
            ambiguous_groups += 1

        for rank, c in enumerate(use, start=1):
            row = dict(d)
            row.update(
                {
                    "matched_mem_id": c.note.mem_id,
                    "matched_note_title": c.note.title,
                    "matched_created_at": c.note.created_at_raw,
                    "matched_updated_at": c.note.updated_at_raw,
                    "title_match_method": title_method,
                    "created_diff_seconds": f"{c.dc_s:.3f}" if c.dc_s != float("inf") else "",
                    "updated_diff_seconds": f"{c.du_s:.3f}" if c.du_s != float("inf") else "",
                    "match_score_seconds": f"{c.score_s:.3f}" if c.score_s != float("inf") else "",
                    "match_rank": str(rank),
                    "ambiguous_match": "True" if ambiguous else "False",
                }
            )
            out_rows.append(row)

    # Write output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Done. Wrote: {out_csv}", file=sys.stderr)
    print(f"Summary: duplicates={total}, no_title_match={no_title_match}, ambiguous_groups={ambiguous_groups}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())