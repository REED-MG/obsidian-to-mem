#!/usr/bin/env python3
"""
mem_phase3_rebuild.py

Phase 3: Rebuild Mem notes with stable IDs = your logical UUIDs,
and rewrite note→note links to use Mem-native note links.

It uses the enriched CSV produced by mem_phase2_map.py:

    note_refs_with_mem_ids.csv

Expected columns in that CSV:

    source_file
    source_title
    source_uuid
    source_mem_id
    final_link
    target_file
    target_title
    target_uuid
    target_mem_id
    match_status   # "OK" / SOURCE_MISSING / SOURCE_AMBIG / TARGET_MISSING / TARGET_AMBIG

What this script does:

1. Reads the CSV and:
   - For each row with match_status == "OK":
       - Records that the *source* UUID will need link rewrites.
       - Records which final_link should be replaced with a Mem-style link to target_uuid.

2. Builds a set of all UUIDs involved (both source_uuid and target_uuid).
   For each UUID, it also notes at least one existing mem_id from the import.

3. For each UUID:
   - Fetches the original note using one of its mem_ids:
         GET /v2/notes/{mem_id}
   - If this UUID has outgoing links, rewrites those links in the content:
         [Label](path/to/Target.md#Heading)
       becomes
         [Label](mem://notes/<target_uuid>#Heading)
   - Creates a new note via:
         POST /v2/notes
     with:
         id           = uuid  (so the Mem ID == your logical UUID)
         content      = rewritten content
         collection_ids, created_at, updated_at copied from original.

4. If --apply is set:
   - After successfully creating all new notes, deletes the originals:
         DELETE /v2/notes/{original_mem_id}

By default (no --apply), this is a DRY RUN: it fetches notes and shows what
it *would* do, but does not create or delete anything.

USAGE:

    export MEM_API_KEY="your_mem_api_key_here"

    # Dry run
    python mem_phase3_rebuild.py \
        --in-csv /path/to/note_refs_with_mem_ids.csv

    # Apply changes
    python mem_phase3_rebuild.py \
        --in-csv /path/to/note_refs_with_mem_ids.csv \
        --apply
"""

import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import requests
import re

MEM_API_BASE = "https://api.mem.ai/v2"
MEM_API_KEY = os.environ.get("MEM_API_KEY")

if not MEM_API_KEY:
    print("[ERROR] MEM_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# Mem-native link template for notes
# Mem uses this format in its markdown:
#   [Link Text](mem://notes/<note_id>)
MEM_LINK_TEMPLATE = "mem://notes/{id}"

# Sleep a little between POST/DELETE calls to be gentle with rate limits
APPLY_SLEEP_SECONDS = 0.25

# Simple retry for 429s
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0

# Match a markdown link or image: [label](url) or ![alt](url)
MD_LINK_RE = re.compile(r"(!?)\[(.*?)\]\((.*?)\)")

# Helper for extracting uuid from your metadata footer if needed
META_START = "<!-- MEM-METADATA-START -->"
META_END = "<!-- MEM-METADATA-END -->"
UUID_RE = re.compile(r"uuid:\s*([0-9a-fA-F-]{10,})")


@dataclass
class RefRow:
    """Single reference row from note_refs_with_mem_ids.csv."""
    source_file: str
    source_title: str
    source_uuid: str
    source_mem_id: str
    final_link: str
    target_file: str
    target_title: str
    target_uuid: str
    target_mem_id: str
    match_status: str


@dataclass
class NoteRewrites:
    """
    All rewrites for a single source UUID.

    final_link -> target_uuid
    """
    uuid: str
    mem_ids: Set[str] = field(default_factory=set)          # all original mem_ids seen for this uuid
    link_map: Dict[str, str] = field(default_factory=dict)  # final_link -> target_uuid


@dataclass
class NoteSnapshot:
    """Content + metadata of a note fetched from Mem."""
    mem_id: str
    uuid: str
    title: str
    content: str
    collection_ids: List[str]
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_request(method: str, path: str, *, params: Optional[dict] = None,
                json_body: Optional[dict] = None, allow_404: bool = False) -> dict:
    """
    Generic wrapper with basic error handling + simple retry on 429.
    """
    url = f"{MEM_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {MEM_API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"[ERROR] {method} {path} failed on attempt {attempt}: {e}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        if resp.status_code == 429 and attempt < MAX_RETRIES:
            print(f"[WARN] Rate limit (429) on {method} {path}, retrying...", file=sys.stderr)
            time.sleep(RETRY_SLEEP_SECONDS)
            continue

        if allow_404 and resp.status_code == 404:
            return {}

        if resp.status_code >= 400:
            raise RuntimeError(f"{method} {path} failed: {resp.status_code} {resp.text}")

        if resp.content:
            return resp.json()
        return {}

    # Should not reach here
    raise RuntimeError(f"{method} {path} failed after {MAX_RETRIES} attempts.")


def api_get_note(note_id: str) -> dict:
    return api_request("GET", f"/notes/{note_id}")


def api_create_note(payload: dict) -> dict:
    return api_request("POST", "/notes", json_body=payload)


def api_delete_note(note_id: str) -> None:
    api_request("DELETE", f"/notes/{note_id}")


# ---------------------------------------------------------------------------
# CSV parsing / building structures
# ---------------------------------------------------------------------------

def read_enriched_refs_csv(path: str) -> List[RefRow]:
    rows: List[RefRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = [
            "source_file",
            "source_title",
            "source_uuid",
            "source_mem_id",
            "final_link",
            "target_file",
            "target_title",
            "target_uuid",
            "target_mem_id",
            "match_status",
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
                    source_mem_id=r["source_mem_id"],
                    final_link=r["final_link"],
                    target_file=r["target_file"],
                    target_title=r["target_title"],
                    target_uuid=r["target_uuid"],
                    target_mem_id=r["target_mem_id"],
                    match_status=r["match_status"],
                )
            )
    return rows


def build_rewrites(rows: List[RefRow]) -> Tuple[Dict[str, NoteRewrites], Dict[str, Set[str]]]:
    """
    From the enriched CSV rows, build:

      - rewrites_by_uuid: source_uuid -> NoteRewrites(
            mem_ids = {all source_mem_ids for that uuid},
            link_map = {final_link -> target_uuid}
        )

      - mem_ids_by_uuid: uuid -> set of mem_ids (for both sources and targets)

    We only consider rows with match_status == "OK".
    """
    rewrites_by_uuid: Dict[str, NoteRewrites] = {}
    mem_ids_by_uuid: Dict[str, Set[str]] = defaultdict(set)

    for r in rows:
        if r.match_status != "OK":
            continue

        src_uuid = (r.source_uuid or "").strip()
        tgt_uuid = (r.target_uuid or "").strip()
        src_mem_id = (r.source_mem_id or "").strip()
        tgt_mem_id = (r.target_mem_id or "").strip()
        final_link = r.final_link

        if not src_uuid or not tgt_uuid or not src_mem_id or not tgt_mem_id:
            # Skip rows with incomplete IDs
            continue

        # Track mem_ids for *both* source and target UUIDs
        mem_ids_by_uuid[src_uuid].add(src_mem_id)
        mem_ids_by_uuid[tgt_uuid].add(tgt_mem_id)

        # Build rewrites for the source UUID
        nr = rewrites_by_uuid.get(src_uuid)
        if nr is None:
            nr = NoteRewrites(uuid=src_uuid)
            rewrites_by_uuid[src_uuid] = nr

        nr.mem_ids.add(src_mem_id)

        existing = nr.link_map.get(final_link)
        if existing and existing != tgt_uuid:
            print(
                f"[WARN] Conflicting target_uuid for same final_link under source_uuid={src_uuid}: "
                f"{existing!r} vs {tgt_uuid!r}. Keeping the first.",
                file=sys.stderr,
            )
        else:
            nr.link_map[final_link] = tgt_uuid

    return rewrites_by_uuid, mem_ids_by_uuid


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------

def split_path_query_fragment(raw: str) -> Tuple[str, str, str]:
    """
    Split 'path?query#frag' into (path, query, frag) for local-style URLs.
    We *only* care about the fragment (heading) here, but this keeps behaviour
    consistent with your phase 1 script.
    """
    in_sq = False
    in_dq = False
    q_pos = -1
    f_pos = -1

    for i, ch in enumerate(raw):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "?" and not in_sq and not in_dq and q_pos == -1 and f_pos == -1:
            q_pos = i
        elif ch == "#" and not in_sq and not in_dq and f_pos == -1:
            f_pos = i
            break

    if q_pos == -1 and f_pos == -1:
        return raw, "", ""

    if q_pos != -1 and (f_pos == -1 or q_pos < f_pos):
        path = raw[:q_pos]
        frag = raw[f_pos + 1 :] if f_pos != -1 else ""
        query = raw[q_pos + 1 : (f_pos if f_pos != -1 else None)]
    else:
        path = raw[:f_pos]
        frag = raw[f_pos + 1 :]
        query = ""

    return path, query, frag


def extract_uuid_from_content(content: str) -> Optional[str]:
    """
    Optional sanity check: read 'uuid: ...' from your metadata footer.
    """
    start = content.find(META_START)
    end = content.find(META_END)
    if start != -1 and end != -1 and end > start:
        segment = content[start:end]
    else:
        segment = content

    m = UUID_RE.search(segment)
    if not m:
        return None
    return m.group(1).strip()


def build_new_link(final_link: str, target_uuid: str) -> str:
    """
    Take a path-style link like:
        [Label](Area/Note.md#Heading)
        ![Alt](Assets/img.png)
    and convert it to a Mem note link:
        [Label](mem://notes/<target_uuid>#Heading)

    If parsing fails for some reason, we just return final_link unchanged.
    """
    m = MD_LINK_RE.fullmatch(final_link.strip())
    if not m:
        return final_link

    bang, label, url = m.group(1), m.group(2), m.group(3)
    label = label or ""
    url = url.strip()

    _path, _query, frag = split_path_query_fragment(url)

    new_url = MEM_LINK_TEMPLATE.format(id=target_uuid)
    if frag:
        new_url = f"{new_url}#{frag}"

    return f"{bang}[{label}]({new_url})"


def apply_rewrites(content: str, link_map: Dict[str, str]) -> Tuple[str, int, int]:
    """
    Apply all final_link -> target_uuid rewrites to a note.

    Returns:
        (new_content, total_links, updated_links)
    """
    total = 0
    updated = 0
    new_content = content

    # Group by final_link -> target_uuid, then do plain str.replace
    for final_link, target_uuid in link_map.items():
        total += 1
        new_link = build_new_link(final_link, target_uuid)

        if final_link not in new_content:
            print(
                f"[WARN] final_link not found in content; uuid={target_uuid} "
                f"link={final_link!r}",
                file=sys.stderr,
            )
            continue

        new_content = new_content.replace(final_link, new_link)
        updated += 1

    return new_content, total, updated


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_note_snapshot(uuid: str, mem_ids: Set[str]) -> NoteSnapshot:
    """
    Fetch one of the mem_ids for this uuid from Mem and build a NoteSnapshot.
    If multiple mem_ids exist, pick one (and warn).
    """
    if not mem_ids:
        raise ValueError(f"No mem_ids available for uuid={uuid}")

    mem_id_list = sorted(mem_ids)
    if len(mem_id_list) > 1:
        print(
            f"[WARN] Multiple mem_ids for uuid={uuid}: {mem_id_list}. "
            f"Using the first one.",
            file=sys.stderr,
        )
    mem_id = mem_id_list[0]

    detail = api_get_note(mem_id)
    content = detail.get("content") or ""
    title = (detail.get("title") or "").strip()
    collection_ids = detail.get("collection_ids", []) or []
    created_at = detail.get("created_at", "")
    updated_at = detail.get("updated_at", "")

    # Optional sanity check: confirm embedded uuid matches
    embedded_uuid = extract_uuid_from_content(content)
    if embedded_uuid and embedded_uuid != uuid:
        print(
            f"[WARN] Embedded uuid {embedded_uuid!r} in note {mem_id} "
            f"doesn't match expected {uuid!r}. Continuing anyway.",
            file=sys.stderr,
        )

    return NoteSnapshot(
        mem_id=mem_id,
        uuid=uuid,
        title=title,
        content=content,
        collection_ids=collection_ids,
        created_at=created_at,
        updated_at=updated_at,
    )


def rebuild_notes(
    rewrites_by_uuid: Dict[str, NoteRewrites],
    mem_ids_by_uuid: Dict[str, Set[str]],
    apply_changes: bool,
) -> None:
    """
    For each UUID known to mem_ids_by_uuid:

      - Load original note (one of its mem_ids)
      - If this uuid has rewrites, transform content
      - Create new note with id=uuid and transformed content
      - Optionally delete all original mem_ids for that uuid
    """
    all_uuids = sorted(mem_ids_by_uuid.keys())

    print(f"Total distinct UUIDs involved (sources + targets): {len(all_uuids)}")

    created_count = 0
    updated_notes = 0
    skipped_notes = 0
    errors = 0

    # Track originals to delete (mem_id -> uuid)
    originals_to_delete: Dict[str, str] = {}

    for uuid in all_uuids:
        note_mem_ids = mem_ids_by_uuid[uuid]
        nr = rewrites_by_uuid.get(uuid)

        needs_rewrite = nr is not None and bool(nr.link_map)
        try:
            snapshot = load_note_snapshot(uuid, note_mem_ids)
        except Exception as e:
            print(f"[ERROR] Failed to fetch original note for uuid={uuid}: {e}", file=sys.stderr)
            errors += 1
            continue

        if needs_rewrite:
            new_content, total_links, updated_links = apply_rewrites(
                snapshot.content,
                nr.link_map,
            )
            if updated_links == 0:
                print(
                    f"[INFO] UUID {uuid} ({snapshot.title!r}): "
                    f"no actual link replacements applied (final_links not found).",
                )
                skipped_notes += 1
                continue

            print(
                f"[INFO] UUID {uuid} ({snapshot.title!r}): "
                f"rewrote {updated_links}/{total_links} note-to-note links.",
            )
            updated_notes += 1
        else:
            # This UUID appears only as a *target* (no outgoing links to fix).
            # We still want to create a new canonical note with id=uuid so that
            # mem://notes/<uuid> links from other notes resolve.
            new_content = snapshot.content
            print(
                f"[INFO] UUID {uuid} ({snapshot.title!r}): "
                f"no outbound links; content copied unchanged.",
            )

        if not apply_changes:
            # Dry run: do not call API; just count what we'd do.
            created_count += 1
            # Track originals we'd delete if we *were* applying
            for mid in note_mem_ids:
                originals_to_delete[mid] = uuid
            continue

        # --- Apply changes: create new note with id = uuid ----------------
        payload = {
            "id": uuid,
            "content": new_content,
            "collection_ids": snapshot.collection_ids,
            "created_at": snapshot.created_at,
            "updated_at": snapshot.updated_at,
        }

        try:
            res = api_create_note(payload)
            new_id = res.get("id")
            print(
                f"[INFO] Created new note id={new_id} for uuid={uuid} "
                f"(title={snapshot.title!r})"
            )
            created_count += 1
        except Exception as e:
            print(
                f"[ERROR] Failed to create new note for uuid={uuid}: {e}",
                file=sys.stderr,
            )
            errors += 1
            # Don't schedule deletes if creation fails
            continue

        # Remember all originals for later deletion
        for mid in note_mem_ids:
            originals_to_delete[mid] = uuid

        time.sleep(APPLY_SLEEP_SECONDS)

    # ------------------------------------------------------------------
    # Delete originals if we applied changes
    # ------------------------------------------------------------------
    if apply_changes and originals_to_delete:
        print(
            f"\n[INFO] Deleting {len(originals_to_delete)} original notes "
            f"now that new UUID-based notes are created..."
        )
        for mem_id, uuid in originals_to_delete.items():
            try:
                api_delete_note(mem_id)
                print(f"[INFO] Deleted original note mem_id={mem_id} (uuid={uuid})")
                time.sleep(APPLY_SLEEP_SECONDS)
            except Exception as e:
                print(
                    f"[ERROR] Failed to delete original note mem_id={mem_id}: {e}",
                    file=sys.stderr,
                )
                errors += 1

    print("\n=== SUMMARY ===")
    print(f"  UUIDs processed:          {len(all_uuids)}")
    print(f"  New notes (would be) created: {created_count}")
    print(f"  Notes with link rewrites: {updated_notes}")
    print(f"  Notes copied unchanged:   {len(all_uuids) - updated_notes}")
    print(f"  Notes skipped (no changes applied): {skipped_notes}")
    print(f"  Errors:                   {errors}")
    if not apply_changes:
        print("  (DRY RUN ONLY – no notes were created or deleted.)")
    else:
        print("  (Changes applied – originals deleted where possible.)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3: Use note_refs_with_mem_ids.csv to rebuild notes in Mem with "
            "IDs equal to your logical UUIDs and rewrite note→note links to use "
            "mem://notes/<uuid>."
        )
    )
    parser.add_argument(
        "--in-csv",
        required=True,
        help="Path to note_refs_with_mem_ids.csv from mem_phase2_map.py",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "If set, actually create new notes and delete originals. "
            "Without this flag, runs in DRY RUN mode (no API writes)."
        ),
    )

    args = parser.parse_args()

    rows = read_enriched_refs_csv(args.in_csv)
    print(f"Loaded {len(rows)} rows from {args.in_csv}")

    rewrites_by_uuid, mem_ids_by_uuid = build_rewrites(rows)
    print(
        f"UUIDs with outbound link rewrites: {len(rewrites_by_uuid)} "
        f"(out of {len(mem_ids_by_uuid)} UUIDs involved as source/target)"
    )

    if not mem_ids_by_uuid:
        print("[WARN] No UUIDs with associated mem IDs were found – nothing to do.")
        return

    if not args.apply:
        print(
            "\n[INFO] Running in DRY RUN mode. "
            "No notes will be created or deleted.\n"
        )
    else:
        print(
            "\n[WARNING] --apply is set. This will create new notes with IDs equal "
            "to your UUIDs and delete the originals. Make sure you're running this "
            "on the correct workspace and have a backup or are comfortable with "
            "the changes."
        )
        confirm = input("Type 'YES' to proceed: ").strip()
        if confirm != "YES":
            print("Aborting.")
            return

    rebuild_notes(rewrites_by_uuid, mem_ids_by_uuid, apply_changes=args.apply)


if __name__ == "__main__":
    main()

