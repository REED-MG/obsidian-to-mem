#!/usr/bin/env python3
"""
Verify that Obsidian source notes exist in Mem exports by UUID.

- Reads UUIDs from Obsidian source notes (metadata block or inline uuid:).
- Reads UUIDs from Mem-export notes (prefers explicit uuid: lines).
- Copies missing Obsidian notes into a missing-notes folder, preserving dates.
- Flags missing notes that appear similar to mem-export notes by metadata/content.
- Reports duplicate UUIDs found in Mem-export.
- Writes CSVs for missing notes and duplicates, plus a summary log.
"""

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
UUID_KEY_RE = re.compile(
    r"(?:^|[\s>])\\?-*\s*uuid\s*:\s*([0-9a-fA-F-]{36})\b",
    re.IGNORECASE | re.MULTILINE,
)

META_START = "<!-- MEM-METADATA-START -->"
META_END = "<!-- MEM-METADATA-END -->"
TOP_YAML_RE = re.compile(r"^\s*---\n(.*?)\n---\s*\n?", re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
WORD_RE = re.compile(r"[a-z0-9]{3,}")


@dataclass
class NoteInfo:
    path: Path
    title: str
    created: str
    modified: str
    uuid: str
    content: str


@dataclass
class MemSummary:
    note: NoteInfo
    title_lower: str
    created_date: str
    modified_date: str
    content_tokens: Tuple[str, ...]
    content_len: int


@dataclass
class SimilarMatch:
    note: NoteInfo
    match: Optional[NoteInfo]
    title_ratio: float
    content_ratio: float
    created_match: bool
    modified_match: bool
    score: float
    is_similar: bool


def log_error(errors: List[str], message: str, exc: Optional[Exception] = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if exc:
        errors.append(f"[{timestamp}] {message} ({type(exc).__name__}: {exc})")
    else:
        errors.append(f"[{timestamp}] {message}")


def read_text(path: Path, errors: List[str]) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        log_error(errors, f"Failed to read {path}", exc)
        return ""


def extract_mem_metadata_block(text: str) -> str:
    if META_START in text and META_END in text:
        # Mem metadata blocks are appended at the end; rfind is cheaper than a full parse.
        start = text.rfind(META_START)
        end = text.rfind(META_END)
        if start != -1 and end != -1 and start < end:
            return text[start:end]
    return ""


def strip_mem_metadata(text: str) -> str:
    if META_START in text and META_END in text:
        start = text.rfind(META_START)
        end = text.rfind(META_END)
        if start != -1 and end != -1 and start < end:
            return text[:start] + text[end + len(META_END):]
    return text


def strip_frontmatter(text: str) -> str:
    match = TOP_YAML_RE.match(text or "")
    if not match:
        return text
    return (text or "")[match.end():]


def normalize_text(text: str) -> str:
    text = strip_frontmatter(strip_mem_metadata(text or ""))
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_uuid(text: str) -> Optional[str]:
    if not text:
        return None

    match = UUID_KEY_RE.search(text)
    if match:
        return match.group(1).lower()
    return None


def parse_key_value_lines(lines: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in lines:
        # Mem metadata can include HTML wrappers; strip them early.
        line = HTML_TAG_RE.sub("", raw).strip()
        if not line or line.startswith("#") or line == "---":
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def parse_frontmatter(text: str) -> Dict[str, str]:
    match = TOP_YAML_RE.match(text or "")
    if not match:
        return {}
    # Simple key:value parsing is enough for the Mem UUID metadata we care about.
    return parse_key_value_lines(match.group(1).splitlines())


def parse_mem_metadata(text: str) -> Dict[str, str]:
    block = extract_mem_metadata_block(text)
    if not block:
        return {}
    return parse_key_value_lines(block.splitlines())


def title_from_text(path: Path, text: str, frontmatter: Dict[str, str]) -> str:
    if "title" in frontmatter and frontmatter["title"]:
        return frontmatter["title"]
    # Prefer the first H1 for a human-friendly title fallback.
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line:
            break
    return path.stem


def pick_meta(frontmatter: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        if key in frontmatter and frontmatter[key]:
            return frontmatter[key]
    return "unknown"


def file_timestamps(path: Path, errors: List[str]) -> Tuple[str, str]:
    try:
        stat = path.stat()
    except OSError as exc:
        log_error(errors, f"Failed to stat {path}", exc)
        return "unknown", "unknown"
    created_ts = getattr(stat, "st_birthtime", stat.st_ctime)
    modified_ts = stat.st_mtime
    created = datetime.fromtimestamp(created_ts).strftime("%Y-%m-%d %H:%M:%S")
    modified = datetime.fromtimestamp(modified_ts).strftime("%Y-%m-%d %H:%M:%S")
    return created, modified


def uuid_from_meta(
    frontmatter: Dict[str, str],
    mem_meta: Dict[str, str],
) -> Optional[str]:
    for source in (mem_meta, frontmatter):
        value = source.get("uuid", "")
        if not value:
            continue
        match = UUID_RE.search(value)
        if match:
            return match.group(0).lower()
        if len(value.strip()) == 36:
            return value.strip().lower()
    return None


def uuid_from_filename(path: Path) -> Optional[str]:
    match = UUID_RE.search(path.stem)
    return match.group(0).lower() if match else None


def build_note_info(
    path: Path,
    text: str,
    errors: List[str],
    require_uuid: bool,
    allow_filename_uuid: bool,
) -> Optional[NoteInfo]:
    frontmatter = parse_frontmatter(text)
    mem_meta = parse_mem_metadata(text)
    uuid = uuid_from_meta(frontmatter, mem_meta) or extract_uuid(text)
    if not uuid and allow_filename_uuid:
        uuid = uuid_from_filename(path)
    if require_uuid and not uuid:
        return None
    uuid = uuid or ""
    merged_meta = {**frontmatter, **mem_meta}
    title = title_from_text(path, text, merged_meta)
    created = pick_meta(merged_meta, [
        "created", "created_at", "createdAt", "created_time", "date"
    ])
    modified = pick_meta(merged_meta, [
        "modified", "updated", "updated_at", "updatedAt", "last_modified",
        "modified_time"
    ])
    file_created, file_modified = file_timestamps(path, errors)
    if created == "unknown":
        created = file_created
    if modified == "unknown":
        modified = file_modified
    content = normalize_text(text)
    return NoteInfo(
        path=path,
        title=title,
        created=created,
        modified=modified,
        uuid=uuid,
        content=content,
    )


def normalize_date(value: str) -> str:
    match = DATE_PREFIX_RE.match(value or "")
    return match.group(1) if match else ""


def tokenize_content(content: str, max_tokens: int = 80, sample_chars: int = 2000) -> Tuple[str, ...]:
    tokens: List[str] = []
    seen = set()
    for match in WORD_RE.finditer((content or "")[:sample_chars]):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return tuple(tokens)


def build_mem_summaries(mem_notes: List[NoteInfo]) -> List[MemSummary]:
    summaries: List[MemSummary] = []
    for note in mem_notes:
        summaries.append(MemSummary(
            note=note,
            title_lower=(note.title or "").lower(),
            created_date=normalize_date(note.created),
            modified_date=normalize_date(note.modified),
            content_tokens=tokenize_content(note.content),
            content_len=len(note.content or ""),
        ))
    return summaries


def similarity_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def token_overlap_ratio(left: Tuple[str, ...], right: Tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    inter = left_set.intersection(right_set)
    union = left_set.union(right_set)
    return len(inter) / len(union) if union else 0.0


def find_similar_match(note: NoteInfo, mem_notes: List[MemSummary]) -> SimilarMatch:
    best_match: Optional[NoteInfo] = None
    best_title = 0.0
    best_content = 0.0
    best_score = 0.0
    best_created = False
    best_modified = False
    note_title = (note.title or "").lower()
    note_created = normalize_date(note.created)
    note_modified = normalize_date(note.modified)
    note_tokens = tokenize_content(note.content)
    note_len = len(note.content or "")
    for candidate in mem_notes:
        title_ratio = similarity_ratio(note_title, candidate.title_lower)
        created_match = note_created and note_created == candidate.created_date
        modified_match = note_modified and note_modified == candidate.modified_date
        len_ratio = max(note_len, candidate.content_len) / max(1, min(note_len, candidate.content_len))
        if len_ratio > 4 and title_ratio < 0.7 and not created_match and not modified_match:
            continue
        content_ratio = token_overlap_ratio(note_tokens, candidate.content_tokens)
        score = max(title_ratio, content_ratio)
        if created_match:
            score += 0.05
        if modified_match:
            score += 0.05
        if score > best_score:
            best_score = score
            best_match = candidate.note
            best_title = title_ratio
            best_content = content_ratio
            best_created = created_match
            best_modified = modified_match
    is_similar = best_score >= 0.6
    return SimilarMatch(
        note=note,
        match=best_match,
        title_ratio=best_title,
        content_ratio=best_content,
        created_match=best_created,
        modified_match=best_modified,
        score=best_score,
        is_similar=is_similar,
    )


def build_mem_index(
    mem_root: Path,
    errors: List[str],
    progress_every: int = 200,
) -> Tuple[Dict[str, List[NoteInfo]], List[NoteInfo], List[Path], int]:
    uuid_map: Dict[str, List[NoteInfo]] = {}
    mem_notes: List[NoteInfo] = []
    missing_uuid_files: List[Path] = []
    scanned = 0
    for path in sorted(mem_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        scanned += 1
        if scanned % progress_every == 0:
            print(f"Indexed {scanned} mem-export notes...", flush=True)
        text = read_text(path, errors)
        info = build_note_info(path, text, errors, require_uuid=False, allow_filename_uuid=True)
        if not info:
            continue
        mem_notes.append(info)
        if not info.uuid:
            missing_uuid_files.append(path)
            continue
        uuid_map.setdefault(info.uuid, []).append(info)
    return uuid_map, mem_notes, missing_uuid_files, scanned


def copy_missing_notes(
    obsidian_root: Path,
    missing_root: Path,
    missing_notes: List[Path],
    dry_run: bool,
    errors: List[str],
) -> int:
    copied = 0
    for path in missing_notes:
        rel = path.relative_to(obsidian_root)
        dest = missing_root / rel
        copied += 1
        if dry_run:
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        except Exception as exc:
            log_error(errors, f"Failed to copy {path} to {dest}", exc)
    return copied


def write_missing_notes_report(
    report_path: Path,
    missing: List[SimilarMatch],
    errors: List[str],
) -> None:
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "title",
                "similar_match",
                "similarity_score",
                "path",
                "similar_path",
                "created",
                "modified",
                "uuid",
                "similar_uuid",
                "title_similarity",
                "content_similarity",
                "created_match",
                "modified_match",
            ])
            for item in missing:
                match = item.match
                writer.writerow([
                    item.note.title,
                    item.note.created,
                    item.note.modified,
                    item.note.uuid,
                    str(item.note.path),
                    "yes" if item.is_similar else "no",
                    match.uuid if match else "",
                    str(match.path) if match else "",
                    f"{item.title_ratio:.3f}",
                    f"{item.content_ratio:.3f}",
                    "yes" if item.created_match else "no",
                    "yes" if item.modified_match else "no",
                    f"{item.score:.3f}",
                ])
    except Exception as exc:
        log_error(errors, f"Failed to write missing notes report to {report_path}", exc)


def write_duplicates_report(
    report_path: Path,
    uuid_map: Dict[str, List[NoteInfo]],
    errors: List[str],
) -> None:
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["uuid", "count", "sample_title", "sample_created", "sample_modified", "sample_path"])
            for uuid, infos in sorted(uuid_map.items()):
                if len(infos) <= 1:
                    continue
                sample = infos[0]
                writer.writerow([
                    uuid,
                    len(infos),
                    sample.title,
                    sample.created,
                    sample.modified,
                    str(sample.path),
                ])
    except Exception as exc:
        log_error(errors, f"Failed to write duplicates report to {report_path}", exc)


def write_log(log_path: Path, lines: List[str], errors: List[str]) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        log_error(errors, f"Failed to write log file to {log_path}", exc)


def write_error_log(error_log_path: Path, errors: List[str]) -> None:
    if not errors:
        return
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path.write_text("\n".join(errors) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Obsidian notes were imported into Mem exports using UUIDs."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root folder containing mem-export and obsidian-source.",
    )
    parser.add_argument(
        "--missing-notes",
        default="missing-notes",
        help="Folder to copy missing notes into.",
    )
    parser.add_argument(
        "--missing-notes-report",
        default="mem_import_missing_notes.csv",
        help="CSV report path for Obsidian notes missing in mem-export.",
    )
    parser.add_argument(
        "--duplicates-report",
        default="mem_import_duplicates.csv",
        help="CSV report path for duplicate UUIDs in mem-export.",
    )
    parser.add_argument(
        "--error-log",
        default="mem_import_errors.log",
        help="Log path for errors encountered while processing files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not copy files; only report what would change.",
    )
    parser.add_argument(
        "--copy-all-missing",
        action="store_true",
        help="Copy all missing notes even if a similar mem-export note is found.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    obsidian_root = (root / "obsidian-source").resolve()
    mem_root = (root / "mem-export").resolve()
    missing_root = root / args.missing_notes
    missing_report_path = root / args.missing_notes_report
    duplicates_report_path = root / args.duplicates_report
    error_log_path = root / args.error_log
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = root / f"mem_verify_import_{run_stamp}.log"
    errors: List[str] = []

    if not obsidian_root.exists():
        raise SystemExit(f"Obsidian source folder not found: {obsidian_root}")
    if not mem_root.exists():
        raise SystemExit(f"Mem export folder not found: {mem_root}")

    print("Indexing mem-export notes...", flush=True)
    mem_uuid_map, mem_notes, mem_missing_uuid_files, mem_scanned = build_mem_index(
        mem_root,
        errors,
    )
    print(f"Finished indexing mem-export notes: {mem_scanned} scanned.", flush=True)
    mem_summaries = build_mem_summaries(mem_notes)

    obsidian_notes = sorted(obsidian_root.rglob("*.md"))
    obsidian_missing_uuid: List[Path] = []
    obsidian_missing_in_mem: List[NoteInfo] = []

    print("Scanning Obsidian notes...", flush=True)
    total_obsidian = len(obsidian_notes)
    for idx, path in enumerate(obsidian_notes, start=1):
        if idx % 200 == 0 or idx == total_obsidian:
            print(f"Scanned {idx}/{total_obsidian} Obsidian notes...", flush=True)
        text = read_text(path, errors)
        info = build_note_info(path, text, errors, require_uuid=True, allow_filename_uuid=False)
        if not info:
            obsidian_missing_uuid.append(path)
            continue
        if info.uuid not in mem_uuid_map:
            obsidian_missing_in_mem.append(info)

    print("Checking missing notes for similar mem-export matches...", flush=True)
    missing_matches: List[SimilarMatch] = []
    total_missing = len(obsidian_missing_in_mem)
    for idx, info in enumerate(obsidian_missing_in_mem, start=1):
        if idx % 50 == 0 or idx == total_missing:
            print(f"Similarity matched {idx}/{total_missing} missing notes...", flush=True)
        missing_matches.append(find_similar_match(info, mem_summaries))
    similar_matches = [item for item in missing_matches if item.is_similar]
    if args.copy_all_missing:
        missing_to_copy = [item.note.path for item in missing_matches]
    else:
        missing_to_copy = [item.note.path for item in missing_matches if not item.is_similar]

    copied_count = copy_missing_notes(
        obsidian_root,
        missing_root,
        missing_to_copy,
        dry_run=args.dry_run,
        errors=errors,
    )

    print("Writing reports...", flush=True)
    write_missing_notes_report(missing_report_path, missing_matches, errors)
    write_duplicates_report(duplicates_report_path, mem_uuid_map, errors)

    log_lines = [
        "Verification complete.",
        f"Mem export notes indexed: {mem_scanned}",
        f"Obsidian notes scanned: {len(obsidian_notes)}",
        f"Obsidian notes missing UUID: {len(obsidian_missing_uuid)}",
        f"Obsidian notes missing in mem-export: {len(obsidian_missing_in_mem)}",
        f"Missing notes with similar match: {len(similar_matches)}",
        f"Missing notes copied: {copied_count}",
        f"Mem export notes missing UUID: {len(mem_missing_uuid_files)}",
        f"Duplicate UUID entries: {sum(1 for infos in mem_uuid_map.values() if len(infos) > 1)}",
        f"Missing notes report: {missing_report_path}",
        f"Duplicates report: {duplicates_report_path}",
        f"Error log: {error_log_path}",
        f"Log file: {log_path}",
        f"Errors reported: {len(errors)}",
    ]
    if args.dry_run:
        log_lines.append("NOTE: Dry run mode. No files were copied.")

    write_log(log_path, log_lines, errors)
    write_error_log(error_log_path, errors)
    for line in log_lines:
        print(line)
    if errors:
        print(f"Errors captured: {len(errors)} (see {error_log_path})")


if __name__ == "__main__":
    main()
