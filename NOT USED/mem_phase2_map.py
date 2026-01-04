#!/usr/bin/env python3
"""
mem_phase2_map.py

Phase 2 (mapping only):

- Reads note_refs.csv produced by the Phase 1 Obsidian→Mem script.
- Uses the Mem API to list notes (without content) and obtain each note's
  Mem ID and title.
- For each row in note_refs.csv, it adds:
      source_mem_id, target_mem_id
  by matching on titles.

- Writes note_refs_with_mem_ids.csv with all original columns plus the
  two new IDs.

This script does NOT modify any notes in Mem. It's just building the
UUID ↔ memID mapping you asked for.

USAGE:

    export MEM_API_KEY="your_api_key_here"

    python mem_phase2_map.py \
        --refs-csv /path/to/note_refs.csv \
        --out-csv /path/to/note_refs_with_mem_ids.csv
"""

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

MEM_API_BASE = "https://api.mem.ai/v2"
MEM_API_KEY = os.environ.get("MEM_API_KEY")

if not MEM_API_KEY:
    print("[ERROR] MEM_API_KEY environment variable MEM_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, params: Optional[dict] = None) -> dict:
    url = f"{MEM_API_BASE}{path}"
    headers = {"Authorization": f"Bearer {MEM_API_KEY}"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {path} failed: {resp.status_code} {resp.text}")
    return resp.json()


def fetch_all_notes_basic() -> List[dict]:
    """
    Fetch all notes using GET /v2/notes without note content.

    We page using 'page' / 'next_page' and 'limit'.
    """
    notes: List[dict] = []
    page_cursor: Optional[str] = None
    limit = 50  # per API docs

    while True:
        params = {
            "limit": limit,
            "order_by": "updated_at",
            # IMPORTANT: we do NOT set include_note_content here
        }
        if page_cursor:
            params["page"] = page_cursor

        data = api_get("/notes", params=params)
        results = data.get("results", [])
        notes.extend(results)

        page_cursor = data.get("next_page")
        if not page_cursor:
            break

    return notes


# ---------------------------------------------------------------------------
# Local CSV models
# ---------------------------------------------------------------------------

@dataclass
class RefRow:
    source_file: str
    source_title: str
    source_uuid: str
    final_link: str
    target_file: str
    target_title: str
    target_uuid: str


def read_refs_csv(path: str) -> List[RefRow]:
    rows: List[RefRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [
            "source_file",
            "source_title",
            "source_uuid",
            "final_link",
            "target_file",
            "target_title",
            "target_uuid",
        ]
        for req in required:
            if req not in reader.fieldnames:
                raise ValueError(f"Missing required column '{req}' in {path}")

        for r in reader:
            rows.append(
                RefRow(
                    source_file=r["source_file"],
                    source_title=r["source_title"],
                    source_uuid=r["source_uuid"],
                    final_link=r["final_link"],
                    target_file=r["target_file"],
                    target_title=r["target_title"],
                    target_uuid=r["target_uuid"],
                )
            )
    return rows


def normalise_title(title: str) -> str:
    return title.strip().lower()


def build_title_index(api_notes: List[dict]) -> Dict[str, List[str]]:
    """
    Map lowercased title -> list of Mem IDs that have that title.
    """
    idx: Dict[str, List[str]] = {}
    for n in api_notes:
        title = (n.get("title") or "").strip()
        mem_id = n["id"]
        key = normalise_title(title)
        idx.setdefault(key, []).append(mem_id)
    return idx


def lookup_mem_id(title: str, title_index: Dict[str, List[str]]) -> Optional[str]:
    """
    Resolve a title to a Mem ID, if unique.
    """
    key = normalise_title(title)
    ids = title_index.get(key)
    if not ids:
        return None
    if len(ids) > 1:
        # ambiguous title
        return None
    return ids[0]


# ---------------------------------------------------------------------------
# Main mapping logic
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 mapping: augment note_refs.csv with source_mem_id and "
            "target_mem_id based on Mem's notes list."
        )
    )
    parser.add_argument(
        "--refs-csv",
        required=True,
        help="Path to note_refs.csv produced by the Phase 1 script.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Path to write the enriched CSV with Mem IDs.",
    )
    args = parser.parse_args()

    # 1) Load local refs
    ref_rows = read_refs_csv(args.refs_csv)
    print(f"Loaded {len(ref_rows)} reference rows from {args.refs_csv}")

    # 2) List notes from Mem (no content)
    print("Fetching note list from Mem (no content)...")
    api_notes = fetch_all_notes_basic()
    print(f"Fetched {len(api_notes)} notes from Mem.")

    title_index = build_title_index(api_notes)

    # 3) Build enriched rows
    out_fieldnames = [
        "source_file",
        "source_title",
        "source_uuid",
        "source_mem_id",
        "final_link",
        "target_file",
        "target_title",
        "target_uuid",
        "target_mem_id",
        "match_status",   # OK / SOURCE_AMBIG / SOURCE_MISSING / TARGET_AMBIG / TARGET_MISSING
    ]

    missing_source = 0
    missing_target = 0
    ambiguous_source = 0
    ambiguous_target = 0

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()

        for r in ref_rows:
            src_ids = title_index.get(normalise_title(r.source_title)) or []
            tgt_ids = title_index.get(normalise_title(r.target_title)) or []

            if not src_ids:
                src_mem_id = ""
                src_status = "SOURCE_MISSING"
                missing_source += 1
            elif len(src_ids) > 1:
                src_mem_id = ""
                src_status = "SOURCE_AMBIG"
                ambiguous_source += 1
            else:
                src_mem_id = src_ids[0]
                src_status = "OK"

            if not tgt_ids:
                tgt_mem_id = ""
                tgt_status = "TARGET_MISSING"
                missing_target += 1
            elif len(tgt_ids) > 1:
                tgt_mem_id = ""
                tgt_status = "TARGET_AMBIG"
                ambiguous_target += 1
            else:
                tgt_mem_id = tgt_ids[0]
                tgt_status = "OK"

            # Prioritise worse status for this row
            if src_status != "OK":
                match_status = src_status
            elif tgt_status != "OK":
                match_status = tgt_status
            else:
                match_status = "OK"

            writer.writerow(
                {
                    "source_file": r.source_file,
                    "source_title": r.source_title,
                    "source_uuid": r.source_uuid,
                    "source_mem_id": src_mem_id,
                    "final_link": r.final_link,
                    "target_file": r.target_file,
                    "target_title": r.target_title,
                    "target_uuid": r.target_uuid,
                    "target_mem_id": tgt_mem_id,
                    "match_status": match_status,
                }
            )

    print(f"Enriched CSV written to: {args.out_csv}")
    print("Summary:")
    print(f"  Rows with missing source title match:    {missing_source}")
    print(f"  Rows with ambiguous source title match:  {ambiguous_source}")
    print(f"  Rows with missing target title match:    {missing_target}")
    print(f"  Rows with ambiguous target title match:  {ambiguous_target}")
    print("  (Check match_status column for per-row details.)")


if __name__ == "__main__":
    main()

