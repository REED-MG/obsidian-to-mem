#!/usr/bin/env python3
"""
obsidian_to_mem_phase1.py

Script for migrating Obsidian, Apple Notes and Evernote to Mem.ai.

Overview:

1. Walk the vault (Markdown + attachments) while skipping folders excluded from import.
2. For every note:
   - Strip Daily Note task callouts.
   - Add Scrybble PDF links when metadata is present.
   - Inline local images when enabled, unless the combined payload would exceed ~5 MB, falling back to attachment handling when necessary.
   - Demote any existing inline data:image blobs when a note already exceeds the 5 MB cap so attachments stay importable.
   - Convert wikilinks to Markdown, centralise attachments, and log outbound note references.
   - Append folder-derived collection tags and a stable UUID metadata block.
3. Emit supporting artifacts:
   - obsidian_to_mem_log.csv – per-note stats (images, attachments, UUIDs, etc.).
   - note_refs.csv – one row per note→note reference for downstream mapping.
   - obsidian_to_mem_run.md – Markdown copy of the console log plus summary metrics.
4. Move referenced attachments into /attachments/ (or plan the moves in dry-run mode) and rewrite their links to point at the configured CDN base.
5. Optionally move all non-.md files into /attachments/ (including unreferenced files).
6. Optionally move linked notes into a central folder (flattened) and rewrite note links to drop sub-folder paths.

Important:

- In --dry-run mode:
    - No .md files are modified.
    - No attachments are moved.
    - Linked notes are not moved.
    - CSVs/report still describe what would have happened.

Preparation:

- Create a .env file in the repo (or export env vars in your shell).
- Set ROOT_CDN_BASE to the base URL where attachments will be served.
  Example:
    ROOT_CDN_BASE=https://your-cdn.example/attachments
- If ROOT_CDN_BASE is empty/unset, attachment links stay local (no CDN rewrite).

Ideas for later phases, not implemented:

- PHASE 2: use note_refs.csv + Mem API list to build mapping of UUID ↔ memID.
- PHASE 3: rebuild notes in Mem with IDs == UUIDs and internal mem:// links.

USAGE:

    # 0) BACK UP your vault first.
    cp -r ~/obsidian-vault ~/obsidian-vault-mem-copy

    # 1) DRY RUN (no writes, no moves, CSVs only):
    python obsidian_to_mem_phase1.py --root ~/obsidian-vault-mem-copy --dry-run

    # 2) REAL RUN (no inline images):
    python obsidian_to_mem_phase1.py --root ~/obsidian-vault-mem-copy

    # 3) REAL RUN with inline images enabled:
    python obsidian_to_mem_phase1.py --root ~/obsidian-vault-mem-copy --inline-images
"""

import argparse
import atexit
import base64
import csv
import datetime
import math
import mimetypes
import os
import re
import shutil
import sys
import unicodedata
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote, unquote

from dotenv import load_dotenv

# Load .env early so config values are available below.
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Folders to exclude from processing/import (by name).
EXCLUDE_FOLDERS = {"Flow", "Templates", "Tasks", "RSS Feeds"}

# Image extensions to inline as base64 data: URLs.
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".heic", ".bmp", ".tiff",
}

# Non-image attachment extensions (hint only; REAL rule is: any resolved file
# that is not .md and not an inlined image will be centralised).
ATTACHMENT_EXTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv",
    ".zip", ".mov", ".mp4", ".mp3", ".m4a", ".wav",
    ".json", ".ppt", ".pptx",
}

# Default CDN base where final attachments will be served.
# New URLs are:
#   ROOT_CDN_BASE.rstrip("/") + "/" + <filename>
ROOT_CDN_BASE = os.environ.get("ROOT_CDN_BASE", "")
CDN_PREFIX = ROOT_CDN_BASE.rstrip("/")
if not ROOT_CDN_BASE:
    print(
        "[WARN] ROOT_CDN_BASE is not set; attachment links will remain local.",
        file=sys.stderr,
    )

# Central attachments folder name (relative to vault root).
CENTRAL_ATTACHMENTS_DIRNAME = "attachments"

# Linked notes folder name (relative to vault root).
LINKED_NOTES_DIRNAME = "Linked Notes"

# Inline image limit (5 MB) before we fall back to treating images as attachments.
INLINE_IMAGE_SIZE_LIMIT_BYTES = 5 * 1024 * 1024

# Tagging / Collections:
RUN_HASHTAG_PREFIX = "Import-Obsidian-"
RUN_HASHTAG_DATE = datetime.date.today().isoformat()
RUN_HASHTAG = f"#{RUN_HASHTAG_PREFIX}{RUN_HASHTAG_DATE}"

# Metadata footer markers
META_START = "<!-- MEM-METADATA-START -->"
META_END = "<!-- MEM-METADATA-END -->"

# ---------------------------------------------------------------------------
# REGEXES
# ---------------------------------------------------------------------------

# Wikilinks: [[Note]], [[Note#Heading|Alias]], ![[embed]]
WIKI_LINK_RE = re.compile(r"(!)?\[\[([^\]|]+?)(?:\|([^\]]*))?\]\]")

# Markdown links: [text](url) or ![alt](url)
# NOTE: We must allow `)` inside the URL (e.g. filenames with parentheses)
# and simple nested-parentheses structures (e.g. ([https://...](https://.../))).
# This pattern matches one level of balanced parentheses inside the URL.
MD_LINK_RE = re.compile(
    r"(!?)\[((?:[^\]]*))\]\((?P<url>(?:[^()\\]|\\.|(?:\([^()]*\)))+)\)"
)

# Simple YAML frontmatter at top-of-file: --- ... ---
TOP_YAML_RE = re.compile(r"^\s*---\n(.*?)\n---\s*\n?", re.DOTALL)

# Markdown images
MD_IMG_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\('
    r'\s*(?P<target><[^>]+>|"[^"]+"|\([^)]+\)|[^)\s]+)'
    r'(?:\s+"(?P<title>[^"]*)")?'
    r'\s*\)',
    flags=re.IGNORECASE,
)

# HTML <img src="...">
HTML_IMG_RE = re.compile(
    r'(<img\b[^>]*\bsrc\s*=\s*["\'])(?P<src>[^"\']+)(["\'][^>]*>)',
    flags=re.IGNORECASE,
)

# Obsidian embedded images: ![[file.png]]
OBSIDIAN_IMG_RE = re.compile(
    r'!\[\[(?P<target>[^|\]]+)(?:\|(?P<alt>[^\]]+))?\]\]',
    flags=re.IGNORECASE,
)

# Any "scheme:" at the start of a URL (http:, https:, mailto:, etc.)
URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
WHITESPACE_COLLAPSE_RE = re.compile(r"\s+")

# Obsidian Tasks callouts/code fences to remove from Daily Notes
TASK_CALLOUT_RE = re.compile(
    r"^\s*>+\s*\[!(tasks?|check|thumb|example|list)\b",
    re.IGNORECASE,
)
TASKS_FENCE_START_RE = re.compile(r"^([>\s]*)```tasks\b", re.IGNORECASE)
TASKS_REFERENCE_RE = re.compile(r"\[\[30 days of Tasks\]\]\s*\[\^1\]", re.IGNORECASE)
TASKS_FOOTNOTE_RE = re.compile(r"^\[\^1\]:.*?(?:\n[ \t]+.*?)*", re.MULTILINE)
CALLOUT_ANY_RE = re.compile(r"^\s*>+\s*\[![^\]]+\]", re.IGNORECASE)
TASKS_FENCE_ANY_RE = re.compile(r"^\s*```tasks\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def is_image_ext(ext: str) -> bool:
    return ext.lower() in IMAGE_EXTS


def in_excluded(path: Path, root: Path) -> bool:
    """True if file lives under any excluded folder."""
    rel_parts = path.relative_to(root).parts
    return any(p in EXCLUDE_FOLDERS for p in rel_parts[:-1])


def is_daily_note(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    return "DailyNotes" in rel_parts


def url_encode_path(path_str: str) -> str:
    return "/".join(
        quote(seg, safe="-._~")
        for seg in path_str.replace("\\", "/").split("/")
    )


def path_from_root(root: Path, target: Path) -> str:
    return target.relative_to(root).as_posix()


def elide_middle(s: str, maxlen: int) -> str:
    if len(s) <= maxlen:
        return s
    if maxlen <= 1:
        return s[:maxlen]
    half = max((maxlen - 1) // 2, 1)
    return s[:half] + "…" + s[-(maxlen - half - 1):]


def normalize_title(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u00a0", " ")
    normalized = WHITESPACE_COLLAPSE_RE.sub(" ", normalized).strip()
    return normalized


class _TeeStream:
    """Mirror writes to the original stream while capturing output for reporting."""

    def __init__(self, original, buffer: List[str], prefix: str = ""):
        self._original = original
        self._buffer = buffer
        self._prefix = prefix
        self.encoding = getattr(original, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self._original.write(data)
        self._original.flush()
        if data:
            if self._prefix:
                self._buffer.append(f"{self._prefix}{data}")
            else:
                self._buffer.append(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()


class ConsoleCapture:
    """Capture stdout/stderr while still streaming to the user's console."""

    def __init__(self) -> None:
        self._buffer: List[str] = []
        self._stdout_orig = sys.stdout
        self._stderr_orig = sys.stderr
        self._stdout_proxy: Optional[_TeeStream] = None
        self._stderr_proxy: Optional[_TeeStream] = None
        self._active = False

    def start(self) -> None:
        if self._active:
            return
        self._stdout_proxy = _TeeStream(self._stdout_orig, self._buffer)
        self._stderr_proxy = _TeeStream(self._stderr_orig, self._buffer, prefix="[stderr] ")
        sys.stdout = self._stdout_proxy  # type: ignore
        sys.stderr = self._stderr_proxy  # type: ignore
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        sys.stdout = self._stdout_orig  # type: ignore
        sys.stderr = self._stderr_orig  # type: ignore
        self._active = False

    def get_text(self) -> str:
        return "".join(self._buffer)


def ensure_scrybble_pdf_link(
    body: str,
    meta: Optional[Dict[str, str]],
    note_path: Path,
) -> str:
    """
    Insert a Markdown link to the Scrybble PDF (if present) at the top of the note body.
    """
    if not meta:
        return body
    raw_name = (meta.get("scrybble_filename") or "").strip()
    if not raw_name:
        return body

    label = f"Scrybble PDF: {raw_name}"
    if f"[{label}]" in body:
        return body

    pdf_name = raw_name if raw_name.lower().endswith(".pdf") else f"{raw_name}.pdf"
    note_dir = note_path.parent
    direct_path = note_dir / pdf_name
    pdf_path: Optional[Path] = direct_path if direct_path.exists() else None

    if pdf_path is None:
        target_stem = canonical_stem(Path(raw_name).stem)
        for cand in note_dir.glob("*.pdf"):
            if canonical_stem(cand.stem) == target_stem:
                pdf_path = cand
                break

    if pdf_path is None:
        return body

    link_line = f"[{label}]({pdf_path.name})"
    trimmed = body.lstrip("\n")
    if trimmed:
        return f"{link_line}\n\n{trimmed}"
    return f"{link_line}\n"


def is_pos_in_spans(pos: int, spans: List[Tuple[int, int]]) -> bool:
    """True if pos falls inside any [start, end) span."""
    for start, end in spans:
        if start <= pos < end:
            return True
    return False


def collect_code_spans(text: str) -> List[Tuple[int, int]]:
    """
    Return byte index spans for fenced and inline code blocks.
    Used to avoid rewriting links that appear inside code examples.
    """
    spans: List[Tuple[int, int]] = []
    fence_re = re.compile(r"^[ \t]*([`~]{3,})")

    # Pass 1: fenced code blocks (``` or ~~~).
    pos = 0
    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_start = 0

    for line in text.splitlines(keepends=True):
        line_len = len(line)
        if not in_fence:
            m = fence_re.match(line)
            if m:
                marker = m.group(1)
                fence_char = marker[0]
                fence_len = len(marker)
                in_fence = True
                fence_start = pos
        else:
            closing_pat = rf"^[ \t]*{re.escape(fence_char)}{{{fence_len},}}\s*$"
            if re.match(closing_pat, line):
                spans.append((fence_start, pos + line_len))
                in_fence = False
        pos += line_len

    if in_fence:
        spans.append((fence_start, len(text)))

    # Pass 2: inline code spans using backticks.
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "`" or is_pos_in_spans(i, spans):
            i += 1
            continue

        run_len = 1
        while i + run_len < n and text[i + run_len] == "`":
            run_len += 1
        search_from = i + run_len
        closing = text.find("`" * run_len, search_from)
        found = False

        while closing != -1:
            if not is_pos_in_spans(closing, spans):
                spans.append((i, closing + run_len))
                i = closing + run_len
                found = True
                break
            closing = text.find("`" * run_len, closing + 1)

        if not found:
            i += run_len

    spans.sort()
    return spans


def print_progress(current: int, total: int, filename: Optional[str] = None) -> None:
    """Simple progress bar on one line."""
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 100
    percent = 0 if total == 0 else int(current * 100 / total)
    bar_width = 30
    ratio = 0 if total == 0 else current / total
    filled = int(bar_width * ratio)
    bar = "█" * filled + " " * (bar_width - filled)

    prefix = "Processing "
    counts = f"({current}/{total})"
    base_len = len(prefix) + 2 + bar_width + 1 + 4 + 1 + len(counts) + 1
    fname_space = max(cols - base_len, 0)
    show_name = elide_middle(filename, fname_space) if filename and fname_space > 0 else ""

    line = f"\r{prefix}[{bar}] {percent:3d}% {counts}"
    if show_name:
        line += f" {show_name}"
    sys.stdout.write(line)
    sys.stdout.flush()


def normalize_lookup_text(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def collapse_internal_whitespace(text: str) -> str:
    return WHITESPACE_COLLAPSE_RE.sub(" ", text or "").strip()


def lookup_variants(key: str, collapse_space: bool) -> List[str]:
    """Generate lookup variants to match differently formatted wikilinks."""
    if not key:
        return []
    norm = normalize_lookup_text(key)
    variants = [key, norm, norm.lower()]
    if collapse_space:
        collapsed = collapse_internal_whitespace(norm)
        variants.extend([collapsed, collapsed.lower()])
    ordered: List[str] = []
    seen: Set[str] = set()
    for cand in variants:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        ordered.append(cand)
    return ordered


def canonical_stem(text: str) -> str:
    return collapse_internal_whitespace(normalize_lookup_text(text)).lower()


def common_prefix_len(parts_a: Tuple[str, ...], parts_b: Tuple[str, ...]) -> int:
    count = 0
    for a, b in zip(parts_a, parts_b):
        if a != b:
            break
        count += 1
    return count


def choose_best_note_path(candidates: List[Path], this_file: Optional[Path], root: Path) -> Optional[Path]:
    if not candidates:
        return None
    if this_file is None:
        return candidates[0]

    try:
        this_parts = this_file.relative_to(root).parts
    except ValueError:
        this_parts = this_file.parts

    best = candidates[0]
    best_score = (-1, -len(best.parts), best.as_posix())
    for path in candidates:
        try:
            cand_parts = path.relative_to(root).parts
        except ValueError:
            cand_parts = path.parts
        score = (
            common_prefix_len(this_parts, cand_parts),
            -len(cand_parts),
            path.as_posix(),
        )
        if score > best_score:
            best_score = score
            best = path
    return best


# ---------------------------------------------------------------------------
# YAML-ish metadata helpers
# ---------------------------------------------------------------------------

def load_top_yaml(text: str) -> Tuple[Optional[dict], str]:
    """Parse simple frontmatter YAML block at top of file."""
    m = TOP_YAML_RE.match(text)
    if not m:
        return None, text
    block = m.group(1)
    remainder = text[m.end():]
    meta: Dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, remainder


def strip_existing_bottom_meta(text: str) -> str:
    """Remove existing MEM-METADATA block if present."""
    start = text.find(META_START)
    end = text.find(META_END)
    if start != -1 and end != -1 and end > start:
        pre = text[:start]
        # Remove the divider we add right before the metadata block.
        for marker in ("\n\n---\n", "\n---\n"):
            if pre.endswith(marker):
                pre = pre[: -len(marker)]
                break
        pre = pre.rstrip()
        if pre:
            pre += "\n"
        return pre
    return text


def extract_bottom_meta(text: str) -> Dict[str, str]:
    """
    Parse the MEM metadata footer into a dict (keys/values as strings).
    """
    meta: Dict[str, str] = {}
    start = text.find(META_START)
    end = text.find(META_END)
    if start == -1 or end == -1 or end <= start:
        return meta

    section = text[start:end]
    m = re.search(r"<small>(.*?)</small>", section, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return meta
    inner = m.group(1)
    lines = [line.strip("\n") for line in inner.splitlines()]

    reading = False
    for raw in lines:
        line = raw.strip()
        if line == "---":
            if not reading:
                reading = True
                continue
            break
        if not reading or not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def render_bottom_meta(meta: dict) -> str:
    """Render YAML-ish metadata block wrapped in HTML comments."""
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append("---")

    out: List[str] = []
    out.append(META_START)
    out.append("<small>")
    out.extend(lines)
    out.append("</small>")
    out.append(META_END)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Collection / tags
# ---------------------------------------------------------------------------

def slugify_tag(text: str) -> str:
    tag = re.sub(r"[^\w\-]", "-", text)
    tag = re.sub(r"-{2,}", "-", tag).strip("-")
    return tag


def tags_for_path(root: Path, file: Path) -> List[str]:
    """Create hashtags from folder structure + a per-run import tag."""
    rel = file.relative_to(root)
    parts = list(rel.parts[:-1])  # folders only
    tags: List[str] = []
    for part in parts:
        if part.startswith("."):
            continue
        tag = slugify_tag(part)
        if tag:
            tags.append(f"#{tag}")
    tags.append(RUN_HASHTAG)
    return tags


def strip_trailing_collection_tags_block(text: str, tags_line: str) -> Tuple[str, bool]:
    """
    Remove the previously appended collection tags block (two newlines + tags)
    if it already exists at the end of the text. Returns (new_text, removed?).
    """
    if not tags_line:
        return text, False
    trimmed = text.rstrip()
    block = f"\n\n{tags_line}"
    if trimmed.endswith(block):
        new_text = trimmed[: -len(block)]
        return new_text.rstrip(), True
    return text, False


# ---------------------------------------------------------------------------
# Note index (for resolving wikilinks between notes)
# ---------------------------------------------------------------------------

def build_note_index(root: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], List[Path]]:
    """
    Build lookup for notes, keeping multiple candidates per key so we can
    choose the closest match when duplicates exist.
    """
    files = sorted(
        (p for p in root.rglob("*.md")),
        key=lambda p: p.relative_to(root).as_posix().lower(),
    )
    key_to_paths: Dict[str, List[Path]] = {}
    stem_buckets: Dict[str, List[Path]] = {}

    def register_note_key(raw_key: str, path: Path, collapse_space: bool) -> None:
        for variant in lookup_variants(raw_key, collapse_space=collapse_space):
            bucket = key_to_paths.setdefault(variant, [])
            if path not in bucket:
                bucket.append(path)

    for p in files:
        rel = p.relative_to(root).as_posix()
        register_note_key(rel, p, collapse_space=False)
        register_note_key(p.name, p, collapse_space=True)

        stem_key = canonical_stem(p.stem)
        stem_buckets.setdefault(stem_key, []).append(p)

    for paths in stem_buckets.values():
        paths.sort(key=lambda path: path.relative_to(root).as_posix().lower())

    return key_to_paths, stem_buckets, files


def resolve_note_target(
    target: str,
    root: Path,
    note_index: Dict[str, List[Path]],
    stem_buckets: Dict[str, List[Path]],
    this_dir: Optional[Path] = None,
    this_file: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolve 'Some Note' or 'folder/Some Note.md' to a concrete note path.
    """
    t = target.strip().strip("/")
    if not t:
        return None
    t_md = t if t.lower().endswith(".md") else f"{t}.md"

    def lookup_candidate(candidate: str) -> Optional[Path]:
        collapse = "/" not in candidate and "\\" not in candidate
        for variant in lookup_variants(candidate, collapse_space=collapse):
            matches = note_index.get(variant)
            if not matches:
                continue
            chosen = choose_best_note_path(matches, this_file, root)
            if chosen is not None:
                return chosen
        return None

    # 1) Direct lookup by relative path or filename
    note_path = lookup_candidate(t_md)
    if note_path is not None:
        return note_path
    if t_md != t:
        note_path = lookup_candidate(t)
        if note_path is not None:
            return note_path

    # 2) Same directory
    stem_lower = canonical_stem(Path(t).stem)
    if this_dir is not None and stem_lower:
        same_dir = [
            p for p in this_dir.glob("*.md") if canonical_stem(p.stem) == stem_lower
        ]
        if len(same_dir) == 1:
            return same_dir[0]

    # 3) Vault-wide closest match by stem
    matches = stem_buckets.get(stem_lower)
    if matches:
        return choose_best_note_path(matches, this_file, root)

    return None


# ---------------------------------------------------------------------------
# Attachment index (non-md files)
# ---------------------------------------------------------------------------

def build_attachment_index(root: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """
    Index all non-.md files so we can resolve attachments quickly.

    Returns:
      rel_index:      'path/from/root.ext' -> Path
      basename_index: 'file.ext' -> [Path, Path, ...]
    """
    rel_index: Dict[str, Path] = {}
    basename_index: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".md":
            continue
        rel = p.relative_to(root).as_posix()
        rel_index[rel] = p
        key = p.name.lower()
        basename_index.setdefault(key, []).append(p)
    return rel_index, basename_index


def resolve_attachment_path(
    target: str,
    note_dir: Path,
    root: Path,
    rel_index: Dict[str, Path],
    basename_index: Dict[str, List[Path]],
) -> Optional[Path]:
    """
    Resolve non-md attachment path given how it appears in the markdown.
    """
    t = target.strip()
    if not t:
        return None

    p = Path(t)

    # Absolute path
    if p.is_absolute():
        if p.exists():
            try:
                p.relative_to(root)
            except ValueError:
                return None
            else:
                return p
        return None

    # Relative to note directory
    cand = (note_dir / p).resolve()
    if cand.exists():
        try:
            cand.relative_to(root)
        except ValueError:
            pass
        else:
            return cand

    # Relative to root
    rel = p.as_posix()
    if rel in rel_index:
        return rel_index[rel]

    # Unique basename
    name = p.name.lower()
    matches = basename_index.get(name)
    if matches and len(matches) == 1:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# Image inlining
# ---------------------------------------------------------------------------

def clean_target(raw: str) -> str:
    """Strip <...>, "..." or (...) and URL-decode."""
    s = raw.strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    elif s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return unquote(s)


def guess_mimetype(path: Path) -> str:
    mt, _ = mimetypes.guess_type(path.as_posix())
    return mt or "image/png"


def file_to_data_uri(fp: Path) -> str:
    mime = guess_mimetype(fp)
    data = fp.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def extension_for_mime(mime: str) -> str:
    """Return a reasonable extension (with leading dot) for a MIME type."""
    ext = (mimetypes.guess_extension(mime) or "").lower()
    if ext == ".jpe":
        ext = ".jpg"
    if not ext:
        if mime.lower().startswith("image/"):
            subtype = mime.split("/", 1)[1].split("+", 1)[0]
            ext = f".{subtype}" if subtype else ".img"
        else:
            ext = ".bin"
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def decode_image_data_uri(data_uri: str) -> Optional[Tuple[str, bytes]]:
    """
    Decode a data:image/...;base64,... URI.

    Returns (mime, bytes) or None if the URI is unsupported.
    """
    if not data_uri.lower().startswith("data:"):
        return None

    header, sep, payload = data_uri.partition(",")
    if not sep:
        return None
    meta = header[len("data:"):]
    if not meta:
        return None

    tokens = [tok.strip() for tok in meta.split(";") if tok.strip()]
    if not tokens:
        return None
    mime = tokens[0].lower()
    if not mime.startswith("image/"):
        return None
    if not any(tok.lower() == "base64" for tok in tokens[1:]):
        return None

    try:
        cleaned = WHITESPACE_COLLAPSE_RE.sub("", payload)
        blob = base64.b64decode(cleaned, validate=True)
    except Exception:
        return None
    return mime, blob


def resolve_image_path(
    target: str,
    base_dir: Path,
    vault_root: Path,
    attachments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Resolve image path for inlining.

    - absolute path
    - relative to note
    - relative to vault root
    - inside attachments_dir (if provided)
    - fallback: vault-wide basename search
    """
    p = Path(target)

    # Absolute
    if p.is_absolute():
        return p if p.exists() else None

    # Relative to note
    rel = (base_dir / p).resolve()
    if rel.exists():
        return rel

    # Relative to vault root (handles explicit 'Folder/file.ext' paths)
    root_rel = (vault_root / p).resolve()
    if root_rel.exists():
        try:
            root_rel.relative_to(vault_root)
        except ValueError:
            pass
        else:
            return root_rel

    # Dedicated attachments dir
    if attachments_dir is not None:
        cand = (attachments_dir / p.name).resolve()
        if cand.exists():
            return cand

    # Fallback: search vault
    matches = list(vault_root.rglob(p.name))
    if len(matches) == 1:
        return matches[0]

    return None


def inline_images_in_text(
    text: str,
    note_path: Path,
    vault_root: Path,
    attachments_dir: Optional[Path],
    inline_override: Optional[bool] = None,
) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Inline image references as data: URLs.

    Returns:
        new_text,
        inlined_paths,
        missing_images,
        removed_blob_refs

    inline_override:
        - None: decide automatically based on payload size limits.
        - True: force inlining regardless of payload (used nowhere currently).
        - False: skip inlining entirely (used when the note would exceed limits).
    """
    original_text = text
    base_bytes = len(original_text.encode("utf-8"))
    base_dir = note_path.parent
    inlined: List[str] = []
    missing: List[str] = []
    removed_blob_refs: List[str] = []
    resolution_cache: Dict[str, Optional[Path]] = {}

    def resolve_cached(target: str) -> Optional[Path]:
        key = target.strip()
        if key in resolution_cache:
            return resolution_cache[key]
        resolved = resolve_image_path(key, base_dir, vault_root, attachments_dir)
        resolution_cache[key] = resolved
        return resolved

    def total_inline_candidate_bytes() -> int:
        total = 0

        def consider(target: str) -> None:
            nonlocal total
            if not target:
                return
            lowered = target.lower()
            if lowered.startswith(("data:", "http://", "https://", "mem://")):
                return
            if lowered.startswith(("blob:", "capacitor://")):
                return
            img_path = resolve_cached(target)
            if not img_path or not is_image_ext(img_path.suffix):
                return
            try:
                total += img_path.stat().st_size
            except OSError:
                return

        for m in MD_IMG_RE.finditer(text):
            consider(clean_target(m.group("target") or "").strip())
        for m in HTML_IMG_RE.finditer(text):
            consider(clean_target(m.group("src") or "").strip())
        for m in OBSIDIAN_IMG_RE.finditer(text):
            consider((m.group("target") or "").strip())
        return total

    total_bytes = total_inline_candidate_bytes()
    estimated_inline_size = math.ceil(total_bytes * 4 / 3)
    estimated_total = base_bytes + estimated_inline_size
    base_already_large = base_bytes > INLINE_IMAGE_SIZE_LIMIT_BYTES
    if inline_override is None:
        allow_inline_images = (
            total_bytes > 0
            and not base_already_large
            and estimated_inline_size <= INLINE_IMAGE_SIZE_LIMIT_BYTES
            and estimated_total <= INLINE_IMAGE_SIZE_LIMIT_BYTES
        )
    else:
        allow_inline_images = inline_override
    if not allow_inline_images and total_bytes > 0 and inline_override is None:
        rel_name = ""
        try:
            rel_name = path_from_root(vault_root, note_path)
        except Exception:
            rel_name = note_path.as_posix()
        reasons = []
        if base_already_large:
            reasons.append(f"base text already {base_bytes} bytes")
        if estimated_inline_size > INLINE_IMAGE_SIZE_LIMIT_BYTES:
            reasons.append(
                f"inline payload {estimated_inline_size} bytes exceeds limit"
            )
        if estimated_total > INLINE_IMAGE_SIZE_LIMIT_BYTES and not base_already_large:
            reasons.append(
                f"base+inline total {estimated_total} bytes would exceed limit"
            )
        reason_str = "; ".join(reasons) if reasons else "payload too large"
        print(
            f"[INFO] Skipping inline image conversion for {rel_name}: "
            f"{reason_str} (limit {INLINE_IMAGE_SIZE_LIMIT_BYTES} bytes)",
            file=sys.stderr,
        )
    elif base_already_large and inline_override is None and total_bytes == 0:
        rel_name = ""
        try:
            rel_name = path_from_root(vault_root, note_path)
        except Exception:
            rel_name = note_path.as_posix()
        print(
            f"[WARN] Note base text {base_bytes} bytes already exceeds "
            f"limit {INLINE_IMAGE_SIZE_LIMIT_BYTES}; consider splitting {rel_name}",
            file=sys.stderr,
        )

    def replace_markdown(m: re.Match) -> str:
        alt, raw_target, title = m.group("alt"), m.group("target"), m.group("title")
        target = clean_target(raw_target).strip()

        if target.lower().startswith(("data:", "http://", "https://", "mem://")):
            return m.group(0)

        if target.lower().startswith(("blob:", "capacitor://")):
            removed_blob_refs.append(target)
            return ""

        img_path = resolve_cached(target)
        if not img_path:
            missing.append(target)
            return m.group(0)

        if not is_image_ext(img_path.suffix):
            return m.group(0)

        if not allow_inline_images:
            return m.group(0)

        try:
            data_uri = file_to_data_uri(img_path)
        except Exception as e:
            missing.append(f"{target} (error: {e})")
            return m.group(0)

        inlined.append(img_path.as_posix())
        title_part = f' "{title}"' if title else ""
        return f"![{alt}]({data_uri}{title_part})"

    def replace_html(m: re.Match) -> str:
        prefix, src, suffix = m.group(1), m.group("src"), m.group(3)
        target = clean_target(src).strip()

        if target.lower().startswith(("data:", "http://", "https://", "mem://")):
            return m.group(0)

        if target.lower().startswith(("blob:", "capacitor://")):
            removed_blob_refs.append(target)
            return ""

        img_path = resolve_cached(target)
        if not img_path:
            missing.append(target)
            return m.group(0)

        if not is_image_ext(img_path.suffix):
            return m.group(0)

        if not allow_inline_images:
            return m.group(0)

        try:
            data_uri = file_to_data_uri(img_path)
        except Exception as e:
            missing.append(f"{target} (error: {e})")
            return m.group(0)

        inlined.append(img_path.as_posix())
        return f"{prefix}{data_uri}{suffix}"

    def replace_obsidian(m: re.Match) -> str:
        target = (m.group("target") or "").strip()
        alt = (m.group("alt") or "").strip()

        if target.lower().startswith(("data:", "http://", "https://", "mem://")):
            return m.group(0)

        if target.lower().startswith(("blob:", "capacitor://")):
            removed_blob_refs.append(target)
            return ""

        img_path = resolve_cached(target)
        if not img_path:
            missing.append(target)
            return m.group(0)

        if not is_image_ext(img_path.suffix):
            return m.group(0)

        if not allow_inline_images:
            return m.group(0)

        try:
            data_uri = file_to_data_uri(img_path)
        except Exception as e:
            missing.append(f"{target} (error: {e})")
            return m.group(0)

        inlined.append(img_path.as_posix())
        return f"![{alt}]({data_uri})"

    text = MD_IMG_RE.sub(replace_markdown, text)
    text = HTML_IMG_RE.sub(replace_html, text)
    text = OBSIDIAN_IMG_RE.sub(replace_obsidian, text)

    final_size = len(text.encode("utf-8"))
    if (
        inline_override is None
        and allow_inline_images
        and final_size > INLINE_IMAGE_SIZE_LIMIT_BYTES
        and total_bytes > 0
    ):
        rel_name = ""
        try:
            rel_name = path_from_root(vault_root, note_path)
        except Exception:
            rel_name = note_path.as_posix()
        print(
            f"[INFO] Reverting inline image conversion for {rel_name}: "
            f"final note size {final_size} bytes exceeds "
            f"limit {INLINE_IMAGE_SIZE_LIMIT_BYTES}",
            file=sys.stderr,
        )
        return inline_images_in_text(
            original_text,
            note_path,
            vault_root,
            attachments_dir,
            inline_override=False,
        )

    return text, inlined, missing, removed_blob_refs


def demote_inline_data_uri_images_if_needed(
    text: str,
    note_path: Path,
    vault_root: Path,
    central_dir: Path,
    used_names: Set[str],
    note_attach_seq: Dict[str, int],
    dry_run: bool,
) -> Tuple[str, List[str], int]:
    """
    Convert inline data:image/... blobs back into attachments when a note exceeds
    the INLINE_IMAGE_SIZE_LIMIT_BYTES cap.

    Returns:
        new_text,
        attachment_urls_added,
        demoted_count
    """
    encoded_bytes = len(text.encode("utf-8"))
    if encoded_bytes <= INLINE_IMAGE_SIZE_LIMIT_BYTES:
        return text, [], 0

    lowered = text.lower()
    if "data:image" not in lowered:
        rel_name = ""
        try:
            rel_name = path_from_root(vault_root, note_path)
        except Exception:
            rel_name = note_path.as_posix()
        print(
            f"[WARN] Note {rel_name} exceeds {INLINE_IMAGE_SIZE_LIMIT_BYTES} bytes "
            "but contains no inline data:image URIs to demote.",
            file=sys.stderr,
        )
        return text, [], 0

    rel_name = ""
    try:
        rel_name = path_from_root(vault_root, note_path)
    except Exception:
        rel_name = note_path.as_posix()

    attachment_urls: List[str] = []
    demoted_count = 0

    def persist_data_uri(data_uri: str) -> Optional[str]:
        nonlocal demoted_count
        parsed = decode_image_data_uri(data_uri)
        if not parsed:
            short = elide_middle(data_uri[:64], 48)
            print(
                f"[WARN] Could not decode inline data URI in {rel_name}: {short}",
                file=sys.stderr,
            )
            return None

        mime, payload = parsed
        ext = extension_for_mime(mime)
        rel_new = allocate_central_attachment_name(
            note_stem=note_path.stem,
            ext=ext,
            root=vault_root,
            central_dir=central_dir,
            used_names=used_names,
            note_attach_seq=note_attach_seq,
        )
        dest_path = (vault_root / rel_new).resolve()
        try:
            if not dry_run:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with dest_path.open("wb") as f:
                    f.write(payload)
        except Exception as e:
            print(
                f"[ERROR] Failed to export inline data URI for {rel_name} -> "
                f"{dest_path}: {e}",
                file=sys.stderr,
            )
            return None

        url = build_cdn_url_for_attachment(rel_new)
        attachment_urls.append(url)
        demoted_count += 1
        action = "Would export" if dry_run else "Exported"
        print(
            f"[INFO] {action} inline data URI in {rel_name} -> {rel_new} "
            f"({len(payload)} bytes)",
            file=sys.stderr,
        )
        return url

    def replace_md_img(m: re.Match) -> str:
        alt, raw_target, title = m.group("alt"), m.group("target"), m.group("title")
        target = clean_target(raw_target or "").strip()
        if not target.lower().startswith("data:image"):
            return m.group(0)
        url = persist_data_uri(target)
        if not url:
            return m.group(0)
        title_part = f' "{title}"' if title else ""
        return f"![{alt}]({url}{title_part})"

    def replace_html_img(m: re.Match) -> str:
        prefix, src, suffix = m.group(1), m.group("src"), m.group(3)
        target = clean_target(src or "").strip()
        if not target.lower().startswith("data:image"):
            return m.group(0)
        url = persist_data_uri(target)
        if not url:
            return m.group(0)
        return f"{prefix}{url}{suffix}"

    new_text = MD_IMG_RE.sub(replace_md_img, text)
    new_text = HTML_IMG_RE.sub(replace_html_img, new_text)

    if demoted_count:
        final_size = len(new_text.encode("utf-8"))
        if final_size > INLINE_IMAGE_SIZE_LIMIT_BYTES:
            print(
                f"[WARN] Note {rel_name} remains above limit after demoting "
                f"{demoted_count} inline images ({final_size} bytes).",
                file=sys.stderr,
            )
    else:
        print(
            f"[WARN] Note {rel_name} exceeded size limit but no inline data URIs "
            "were demoted (possibly due to decode errors).",
            file=sys.stderr,
        )

    return new_text, attachment_urls, demoted_count


def remove_obsidian_tasks_blocks(text: str) -> str:
    """Strip Obsidian Tasks callouts, fenced blocks, and specific references."""
    lines = text.splitlines()
    cleaned: List[str] = []
    skip_callout = False
    skip_fence = False

    i = 0
    while i < len(lines):
        line = lines[i]

        if skip_fence:
            if line.strip().startswith("```"):
                skip_fence = False
            i += 1
            continue

        if TASKS_FENCE_START_RE.match(line):
            skip_fence = True
            i += 1
            continue

        if TASK_CALLOUT_RE.match(line):
            skip_callout = True
            i += 1
            continue

        if skip_callout:
            if not line.strip():
                skip_callout = False
            i += 1
            continue

        cleaned.append(line)
        i += 1

    cleaned_text = "\n".join(cleaned)
    cleaned_text = TASKS_REFERENCE_RE.sub("", cleaned_text)
    cleaned_text = TASKS_FOOTNOTE_RE.sub("", cleaned_text)
    return cleaned_text


def count_remaining_callouts_and_tasks(text: str) -> Tuple[int, int]:
    callouts = 0
    tasks_blocks = 0
    for line in text.splitlines():
        if CALLOUT_ANY_RE.match(line):
            callouts += 1
        if TASKS_FENCE_ANY_RE.match(line):
            tasks_blocks += 1
    return callouts, tasks_blocks


# ---------------------------------------------------------------------------
# Link / attachment rewriting
# ---------------------------------------------------------------------------

def split_path_query_fragment(raw: str) -> Tuple[str, str, str]:
    """
    Split 'path?query#frag' into (path, query, frag).
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
        frag = raw[f_pos + 1:] if f_pos != -1 else ""
        query = raw[q_pos + 1:(f_pos if f_pos != -1 else None)]
    else:
        path = raw[:f_pos]
        frag = raw[f_pos + 1:]
        query = ""

    return path, query, frag


def slugify_filename(text: str) -> str:
    """Slug suitable for filenames (letters/digits/_/-)."""
    slug = re.sub(r"[^\w\-]+", "-", text)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "item"


def build_cdn_url_for_attachment(rel_new: str) -> str:
    """
    rel_new is something like 'attachments/Note-Title-1.pdf'.

    We build:
        ROOT_CDN_BASE.rstrip("/") + "/" + filename
    """
    filename = Path(rel_new).name
    if ROOT_CDN_BASE:
        base = ROOT_CDN_BASE.rstrip("/")
        return f"{base}/{url_encode_path(filename)}"
    # Fallback: relative path
    return url_encode_path(rel_new)


def allocate_central_attachment_name(
    note_stem: str,
    ext: str,
    root: Path,
    central_dir: Path,
    used_names: Set[str],
    note_attach_seq: Dict[str, int],
) -> str:
    """Reserve a unique filename inside the central attachments folder."""
    base_note = slugify_filename(note_stem)
    seq = note_attach_seq.get(base_note, 0) + 1
    note_attach_seq[base_note] = seq

    ext = ext or ""
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    if not ext:
        ext = ".bin"

    root_name = f"{base_note}-{seq}"
    unique = f"{root_name}{ext}"
    lower_used = {name.lower() for name in used_names}
    idx = 2
    while unique.lower() in lower_used:
        unique = f"{root_name}_{idx}{ext}"
        idx += 1
    used_names.add(unique)

    try:
        central_rel_dir = central_dir.relative_to(root).as_posix()
    except ValueError:
        central_rel_dir = central_dir.as_posix()

    if central_rel_dir:
        return f"{central_rel_dir}/{unique}"
    return unique


def register_central_attachment(
    src_path: Path,
    note_stem: str,
    root: Path,
    central_dir: Path,
    move_plan: Dict[Path, str],
    used_names: Set[str],
    note_attach_seq: Dict[str, int],
) -> Optional[str]:
    """
    Plan to move an attachment to the central /attachments/ folder.

    - src_path: original absolute path inside vault
    - note_stem: title of current note (for naming)
    - move_plan: maps src_path -> new relative path (root-relative)
    - used_names: set of filenames already planned
    - note_attach_seq: per-note numeric counters for filenames

    Returns:
        new relative path from root (e.g. 'attachments/Note-1.pdf'),
        or None if src_path doesn't exist.
    """
    if not src_path.exists():
        return None

    if src_path in move_plan:
        return move_plan[src_path]

    ext = src_path.suffix
    rel_new = allocate_central_attachment_name(
        note_stem=note_stem,
        ext=ext,
        root=root,
        central_dir=central_dir,
        used_names=used_names,
        note_attach_seq=note_attach_seq,
    )
    move_plan[src_path] = rel_new
    return rel_new


def rewrite_links_and_collect(
    text: str,
    this_file: Path,
    root: Path,
    note_index: Dict[str, List[Path]],
    note_stem_index: Dict[str, List[Path]],
    attach_rel_idx: Dict[str, Path],
    attach_basename_idx: Dict[str, List[Path]],
    central_dir: Path,
    attachment_move_plan: Dict[Path, str],
    used_central_names: Set[str],
    note_uuid_map: Dict[Path, str],
    note_ref_rows: List[Dict[str, str]],
    note_attach_seq: Dict[str, int],
) -> Tuple[str, List[Path], List[str], List[str]]:
    """
    Rewrite links inside a note:

      - Convert wikilinks to notes into standard Markdown links (Option A)
        and log them into note_ref_rows.
      - Convert links to attachments (ANY non-.md resolved file) into CDN URLs,
        planning to move physical files into /attachments/<new-name>.

    Returns:
        new_text,
        outbound note Paths,
        attachment_urls,
        missing_attachments
    """
    outbound_notes: List[Path] = []
    attachment_urls: List[str] = []
    missing_attachments: List[str] = []

    note_dir = this_file.parent

    # Some Evernote/Bing imports use citation-style placeholders inside Markdown links,
    # e.g. [Title](^1^) with a later reference list:
    #   (1) ... https://example.com
    # These MUST NOT be treated as attachments. Where possible, map them to the real URL.
    citation_url_map: Dict[str, str] = {}
    citation_line_re = re.compile(r"^\((?P<n>\d+)\)\s+.*?(?P<url>https?://\S+)", re.IGNORECASE)
    for line in (text or "").splitlines():
        m_cite = citation_line_re.match(line.strip())
        if not m_cite:
            continue
        n = m_cite.group("n")
        url = m_cite.group("url").rstrip(").,;\"")
        if n and url:
            citation_url_map[n] = url

    code_spans = collect_code_spans(text)

    def apply_query_and_fragment(snippet: str, query: str, frag: str) -> str:
        if not snippet or (not query and not frag):
            return snippet
        url_match = re.search(r"\(([^)]+)\)", snippet)
        if not url_match:
            return snippet
        url_base = url_match.group(1)
        if query:
            url_base = f"{url_base}?{query}"
        if frag:
            url_base = f"{url_base}#{frag}"
        return re.sub(r"\([^)]+\)", f"({url_base})", snippet)

    def handle_attachment(
        path_part: str,
        is_image: bool,
        label: str,
        resolved_src: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Turn a local attachment reference into a Markdown link with CDN URL.

        RULES:
          - If the resolved file is .md => NOT an attachment; leave alone.
          - If the resolved file is an image => we still centralise it here
            only if it wasn't already inlined (for missing/failed inlines).
          - Any other resolved file => attachment, move to /attachments/.
        """
        src = resolved_src
        if src is None:
            src = resolve_attachment_path(path_part, note_dir, root, attach_rel_idx, attach_basename_idx)
        if src is None or not src.exists():
            missing_attachments.append(path_part)
            return None

        ext = src.suffix.lower()
        if ext == ".md":
            # We don't treat .md files as attachments here
            return None

        # Treat as image only if extension suggests so. Non-image attachments
        # should never render as inline images, even if the source used '!'.
        is_img = is_image_ext(ext)

        # Plan central move for this attachment
        new_rel = register_central_attachment(
            src_path=src,
            note_stem=this_file.stem,
            root=root,
            central_dir=central_dir,
            move_plan=attachment_move_plan,
            used_names=used_central_names,
            note_attach_seq=note_attach_seq,
        )
        if new_rel is None:
            missing_attachments.append(path_part)
            return None

        url = build_cdn_url_for_attachment(new_rel)
        attachment_urls.append(url)

        alt_text = label or Path(path_part).stem or url
        if is_img:
            return f"![{alt_text}]({url})"
        return f"[{alt_text}]({url})"

    def render_note_link(note_path: Path, fragment: str, preferred_label: str) -> str:
        """Build final Markdown link for a resolved note + log the reference."""
        if in_excluded(note_path, root):
            visible = preferred_label or (fragment or note_path.stem)
            return visible

        outbound_notes.append(note_path)

        rel_path = path_from_root(root, note_path)
        # Encode the path portion so filenames with spaces/parentheses don’t break Markdown parsing.
        url = url_encode_path(rel_path)
        if fragment:
            # Keep fragment as-is to preserve Obsidian-style heading anchors.
            url = f"{url}#{fragment}"

        visible = normalize_title(note_path.stem)
        final_link = f"[{visible}]({url})"

        src_uuid = note_uuid_map.get(this_file, "")
        tgt_uuid = note_uuid_map.get(note_path, "")

        note_ref_rows.append(
            {
                "source_file": path_from_root(root, this_file),
                "source_title": this_file.stem,
                "source_uuid": src_uuid,
                "final_link": final_link,
                "target_file": path_from_root(root, note_path),
                "target_title": note_path.stem,
                "target_uuid": tgt_uuid,
            }
        )

        return final_link

    def resolve_markdown_note_path(path_part: str) -> Optional[Path]:
        """Resolve Markdown link targets that reference other notes."""
        clean = (path_part or "").strip()
        if not clean:
            return None

        note_path = resolve_note_target(
            clean,
            root,
            note_index,
            note_stem_index,
            this_dir=this_file.parent,
            this_file=this_file,
        )
        if note_path is not None and note_path.exists():
            return note_path

        candidates = [clean]
        if not clean.lower().endswith(".md"):
            candidates.append(f"{clean}.md")

        for cand in candidates:
            cand_path = Path(cand)
            if cand_path.suffix.lower() != ".md":
                continue

            possible: List[Path] = []
            if cand_path.is_absolute():
                possible.append(cand_path)
            else:
                possible.append((note_dir / cand_path).resolve())
                possible.append((root / cand_path).resolve())

            for poss in possible:
                if not poss.exists():
                    continue
                if poss.suffix.lower() != ".md":
                    continue
                try:
                    poss.relative_to(root)
                except ValueError:
                    continue
                return poss

        return None

    # ---------- Wikilinks (Option A + CSV logging) ------------------------

    def replace_wiki(m: re.Match) -> str:
        """
        Option A behaviour + CSV logging for note→note references.

        Note links:
          [[Note]]                 -> [Note](path/from/root/Note.md)
          [[Note#Heading]]         -> [Heading](path/from/root/Note.md#Heading)
          [[Note#Heading|Alias]]   -> [Alias](path/from/root/Note.md#Heading)

        For each resolved note→note ref we add a row to note_ref_rows.
        """
        nonlocal outbound_notes

        if is_pos_in_spans(m.start(), code_spans):
            return m.group(0)

        bang, target_raw, label = m.group(1), m.group(2), m.group(3)
        target_raw = (target_raw or "").strip()
        label = (label or "").strip()

        # Evernote → Obsidian imports sometimes produce wikilinks that are actually URLs
        # (e.g. whatsapp://send, x-apple-msg-load://.../../mailto:foo@bar.com) rather than note targets.
        if target_raw:
            # 0) Bare email address in a wikilink: [[someone@example.com|Label]]
            if (
                "@" in target_raw
                and "/" not in target_raw
                and "\\" not in target_raw
                and " " not in target_raw
                and not URL_SCHEME_RE.match(target_raw)
            ):
                email = target_raw
                visible = label or email
                return f"[{visible}](mailto:{email})"

            # 1) Scheme URLs inside a wikilink: [[whatsapp://send|...]]
            if URL_SCHEME_RE.match(target_raw) or target_raw.lower().startswith("www."):
                fixed = target_raw
                if fixed.lower().startswith("www."):
                    fixed = "https://" + fixed

                # Special-case: x-apple-msg-load://.../../mailto:foo@bar.com  -> mailto:foo@bar.com
                m_mailto = re.search(r"(mailto:[^\s]+)", fixed, flags=re.IGNORECASE)
                if m_mailto:
                    fixed = m_mailto.group(1)

                if fixed.lower().startswith("mailto:"):
                    # If no alias provided, display just the email address.
                    if not label:
                        label = fixed.split(":", 1)[1]

                visible = label or fixed
                return f"[{visible}]({fixed})"

        path_part, query, attach_frag = split_path_query_fragment(target_raw)
        resolved_src = resolve_attachment_path(path_part, note_dir, root, attach_rel_idx, attach_basename_idx)
        if resolved_src is not None and resolved_src.suffix.lower() != ".md":
            snippet = handle_attachment(
                path_part,
                is_image=bool(bang),
                label=label,
                resolved_src=resolved_src,
            )
            if snippet is None:
                return m.group(0)
            return apply_query_and_fragment(snippet, query, attach_frag)

        # Split "Note#Heading" into base note + fragment
        base, _, note_frag = target_raw.partition("#")
        base = base.strip()

        # Try to resolve as another note
        note_path = resolve_note_target(
            base,
            root,
            note_index,
            note_stem_index,
            this_dir=this_file.parent,
            this_file=this_file,
        )
        if note_path is not None and note_path.exists():
            return render_note_link(note_path, fragment=note_frag, preferred_label=label)

        # Not a note: treat as potential attachment
        snippet = handle_attachment(
            path_part,
            is_image=bool(bang),
            label=label,
            resolved_src=resolved_src,
        )
        if snippet is None:
            return m.group(0)

        return apply_query_and_fragment(snippet, query, attach_frag)

    # ---------- Markdown links -------------------------------------------

    def replace_md_link(m: re.Match) -> str:
        bang, lbl, url_raw = m.group(1), m.group(2), m.group(3)

        if is_pos_in_spans(m.start(), code_spans):
            return m.group(0)
        label = (lbl or "").strip()
        url_clean = clean_target(url_raw).strip()

        # Citation placeholders like (^1^) are common in imports (Bing/Evernote).
        # If we can map ^N^ to a real URL in the reference list, do so.
        # Otherwise, keep the link unchanged and (critically) do NOT treat it as an attachment.
        m_caret_cite = re.fullmatch(r"\^(?P<n>\d+)\^", url_clean)
        if m_caret_cite:
            n = m_caret_cite.group("n")
            mapped = citation_url_map.get(n)
            if mapped:
                if not label:
                    label = mapped
                return f"{bang}[{label}]({mapped})" if bang else f"[{label}]({mapped})"
            return m.group(0)

        nested_md = re.match(r"^\[[^\]]+\]\(([^)]+)\)$", url_clean)
        if nested_md:
            inner_target = clean_target(nested_md.group(1)).strip()
            if inner_target:
                url_clean = inner_target

        # 1) Anchors (#heading)
        if url_clean.startswith("#"):
            if not label:
                label = url_clean
                return f"[{label}]({url_clean})"
            return m.group(0)

        # 2) External URLs
        if URL_SCHEME_RE.match(url_clean) or url_clean.lower().startswith("www."):
            fixed = url_clean
            if url_clean.lower().startswith("www."):
                fixed = "https://" + url_clean
            if not label:
                label = fixed
            if bang:
                ext = Path(split_path_query_fragment(fixed)[0]).suffix.lower()
                is_img = bool(ext) and is_image_ext(ext)
                bang = "!" if is_img else ""
            return f"{bang}[{label}]({fixed})" if bang else f"[{label}]({fixed})"

        # 3) Local-ish path – possible attachment or note
        path_part, query, frag = split_path_query_fragment(url_clean)
        resolved_src = resolve_attachment_path(path_part, note_dir, root, attach_rel_idx, attach_basename_idx)
        if resolved_src is not None and resolved_src.suffix.lower() != ".md":
            snippet = handle_attachment(
                path_part,
                is_image=bool(bang),
                label=label,
                resolved_src=resolved_src,
            )
            if snippet is None:
                return m.group(0)
            return apply_query_and_fragment(snippet, query, frag)

        if not bang:
            md_note = resolve_markdown_note_path(path_part)
            if md_note is not None:
                return render_note_link(md_note, fragment=frag, preferred_label=label)

        snippet = handle_attachment(path_part, is_image=bool(bang), label=label, resolved_src=resolved_src)
        if snippet is None:
            # Could be a note link we didn't see as wikilink; keep as-is but ensure label
            if not label:
                label = path_part or url_clean
                return f"{bang}[{label}]({url_clean})" if bang else f"[{label}]({url_clean})"
            return m.group(0)

        return apply_query_and_fragment(snippet, query, frag)

    # Apply Markdown links first (existing links), then convert Obsidian wikilinks.
    # This avoids re-processing the Markdown links we generate from wikilinks.
    text = MD_LINK_RE.sub(replace_md_link, text)
    code_spans = collect_code_spans(text)
    text = WIKI_LINK_RE.sub(replace_wiki, text)

    missing_attachments = sorted(set(missing_attachments))
    return text, outbound_notes, attachment_urls, missing_attachments


# ---------------------------------------------------------------------------
# Vault scanning & main
# ---------------------------------------------------------------------------

def count_files(root: Path) -> Tuple[int, int]:
    md, other = 0, 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".md":
            md += 1
        else:
            other += 1
    return md, other


def move_planned_attachments(root: Path, central_dir: Path, move_plan: Dict[Path, str]) -> None:
    """
    After rewriting all notes, physically move attachments into central_dir
    according to move_plan (src_path -> new_rel_path), preserving timestamps.
    """
    if not move_plan:
        print("No attachments to move into central folder.")
        return

    print(f"\nMoving {len(move_plan)} attachments into {central_dir} ...")
    for src_path, rel_new in move_plan.items():
        dest_path = root / rel_new
        if not src_path.exists():
            print(f"[WARN] Planned attachment source missing: {src_path}", file=sys.stderr)
            continue

        # Preserve original atime/mtime
        try:
            st = src_path.stat()
        except OSError as e:
            print(f"[WARN] Could not stat source attachment {src_path}: {e}", file=sys.stderr)
            st = None

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists():
                # Already moved (e.g. prior run); optionally we could still reset times.
                continue
            shutil.move(src_path.as_posix(), dest_path.as_posix())
            if st is not None:
                os.utime(dest_path, (st.st_atime, st.st_mtime))
        except Exception as e:
            print(f"[ERROR] Failed to move attachment {src_path} -> {dest_path}: {e}", file=sys.stderr)


def add_unreferenced_attachments_to_plan(
    root: Path,
    central_dir: Path,
    move_plan: Dict[Path, str],
    used_central_names: Set[str],
    note_attach_seq: Dict[str, int],
) -> Tuple[int, int]:
    """
    Add all non-.md files into the attachment move plan, excluding items already planned
    or already under the central attachments folder.

    Returns:
        added_count,
        skipped_count (files that could not be reserved for move)
    """
    added = 0
    skipped = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".md":
            continue
        try:
            path.relative_to(central_dir)
            continue
        except ValueError:
            pass
        if path in move_plan:
            continue
        new_rel = register_central_attachment(
            src_path=path,
            note_stem=path.stem,
            root=root,
            central_dir=central_dir,
            move_plan=move_plan,
            used_names=used_central_names,
            note_attach_seq=note_attach_seq,
        )
        if new_rel is None:
            skipped += 1
        else:
            added += 1

    return added, skipped


def delete_inlined_images(
    inlined_paths: Set[Path],
    root: Path,
    move_plan: Dict[Path, str],
    dry_run: bool,
) -> Tuple[int, int]:
    """
    Remove original image files that have been fully inlined.

    Returns:
        deleted_count,
        planned_count (for dry-run visibility)
    """
    candidates: List[Path] = []
    for path in inlined_paths:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path in move_plan:
            # Another note still references this file as an attachment.
            continue
        if not path.exists():
            continue
        candidates.append(path)

    planned = len(candidates)
    if dry_run:
        if planned:
            print(f"DRY RUN: would delete {planned} inlined image(s).")
        return 0, planned

    deleted = 0
    for path in candidates:
        try:
            path.unlink()
            deleted += 1
        except Exception as e:
            rel = path_from_root(root, path)
            print(f"[WARN] Failed to delete inlined image {rel}: {e}", file=sys.stderr)
    return deleted, planned


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
        prefix = match.group(1)
        label = match.group(2)
        raw_url = match.group("url").strip()
        wrapper_start = ""
        wrapper_end = ""
        url = raw_url

        if url.startswith("<") and url.endswith(">"):
            wrapper_start, wrapper_end = "<", ">"
            url = url[1:-1]
        elif len(url) >= 2 and url[0] in "\"'" and url[-1] == url[0]:
            wrapper_start = wrapper_end = url[0]
            url = url[1:-1]

        if URL_SCHEME_RE.match(url):
            return match.group(0)

        base, frag = split_fragment(url)
        decoded = unquote(base)
        mapped = md_map.get(decoded)
        if not mapped:
            return match.group(0)
        new_name, new_stem = mapped
        use_full = Path(decoded).suffix.lower() == ".md"
        new_base = new_name if use_full else new_stem
        updated += 1
        new_url = f"{new_base}{frag}"
        return f"{prefix}[{label}]({wrapper_start}{new_url}{wrapper_end})"

    new_content = MD_LINK_RE.sub(replace_md, content)
    new_content = WIKI_LINK_RE.sub(replace_wiki, new_content)

    if updated and not dry_run:
        path.write_text(new_content, encoding="utf-8")
        if mtime is not None:
            os.utime(path, mtime)
    return updated, updated > 0


def rewrite_links(
    root: Path,
    dest_root: Path,
    name_map: Dict[Path, str],
    mtime_map: Dict[Path, Tuple[float, float]],
    dry_run: bool,
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
            mtime=mtime_map.get(rel),
        )
        total_links += updated
        if changed:
            updated_files += 1

    return total_links, updated_files


def move_linked_notes_flat(
    root: Path,
    dest_root: Path,
    note_paths: Iterable[Path],
    dry_run: bool,
) -> Tuple[int, int, int, Dict[Path, str], Dict[Path, Tuple[float, float]]]:
    name_map, collisions = build_flatten_map(dest_root, note_paths)
    moved = 0
    missing = 0
    mtime_map: Dict[Path, Tuple[float, float]] = {}

    for rel in sorted({Path(p) for p in note_paths}):
        src = root / rel
        if not src.exists():
            missing += 1
            continue

        dest_name = name_map[rel]
        dest = dest_root / dest_name
        st = src.stat()
        mtime_map[rel] = (st.st_atime, st.st_mtime)

        moved += 1
        if dry_run:
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        os.utime(dest, mtime_map[rel])

    return moved, missing, collisions, name_map, mtime_map


def write_markdown_report(
    report_path: Path,
    stats: List[Tuple[str, str]],
    console_output: str,
    collection_counts: Dict[str, int],
) -> None:
    """Persist the captured console log + summary stats to a Markdown file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_log = console_output.replace("\r", "\n")
    sanitized_log = cleaned_log.strip("\n")
    if not sanitized_log:
        sanitized_log = "(no console output captured)"
    lines: List[str] = [
        "# Obsidian → Mem Phase 1 Report",
        "",
        "## Summary",
    ]
    for label, value in stats:
        lines.append(f"- **{label}**: {value}")
    lines.extend(["", "## Collection Counts"])
    if collection_counts:
        for tag, count in sorted(collection_counts.items()):
            lines.append(f"- **{tag}**: {count}")
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Console Output",
            "```",
            sanitized_log,
            "```",
            "",
        ]
    )
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_summary_note(summary_path: Path, stats: List[Tuple[str, str]], collections: Set[str]) -> None:
    """Write a concise summary note without the full console log or per-row details."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        f"# Obsidian → Mem Phase 1 Summary ({RUN_HASHTAG_DATE})",
        "",
        "## Summary",
    ]
    for label, value in stats:
        lines.append(f"- **{label}**: {value}")

    lines.extend(
        [
            "",
            "## Collections Added",
        ]
    )
    if collections:
        for tag in sorted(collections):
            lines.append(f"- {tag}")
    else:
        lines.append("- (none)")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PHASE 1: Pre-process Obsidian notes for Mem.ai import "
            "(optional inline images, centralise attachments, add collection tags, "
            "emit UUIDs and note_refs.csv for subsequent phases)."
        )
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Vault root directory (default: current directory).",
    )
    parser.add_argument(
        "--attachments-dir",
        default=None,
        help=(
            "Optional image attachments directory (relative to root) to help "
            "image resolution for inlining (e.g. 'attachments'). This is "
            "ONLY for image inlining; non-image attachments always go to "
            "the central /attachments/ folder."
        ),
    )
    parser.add_argument(
        "--inline-images",
        action="store_true",
        help=(
            "Enable inlining local images as data: URIs "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--log-csv",
        default="obsidian_to_mem_log.csv",
        help="Per-note log CSV filename (default: obsidian_to_mem_log.csv).",
    )
    parser.add_argument(
        "--report-md",
        default="obsidian_to_mem_run.md",
        help="Markdown report file to write console output + summary stats to.",
    )
    parser.add_argument(
        "--move-linked-notes",
        action="store_true",
        default=True,
        help=(
            "Move linked notes into a central folder (flattened) and "
            "rewrite note links to drop sub-folder paths (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-move-linked-notes",
        action="store_false",
        dest="move_linked_notes",
        help="Disable moving linked notes and link rewrites.",
    )
    parser.add_argument(
        "--move-all-non-md",
        action="store_true",
        default=True,
        help="Move all non-.md files into the central attachments folder (default: enabled).",
    )
    parser.add_argument(
        "--no-move-all-non-md",
        action="store_false",
        dest="move_all_non_md",
        help="Disable moving unreferenced non-.md files into the attachments folder.",
    )
    parser.add_argument(
        "--linked-notes-folder",
        default=LINKED_NOTES_DIRNAME,
        help="Destination folder name for linked notes (default: Linked Notes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Process everything and generate CSVs, but DO NOT modify any "
            "files or move attachments."
        ),
    )

    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.is_dir():
        print(f"[ERROR] Root path is not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    report_path = Path(args.report_md)
    if not report_path.is_absolute():
        report_path = root / report_path

    capture = ConsoleCapture()
    capture.start()
    atexit.register(capture.stop)

    run_started = datetime.datetime.now()

    attachments_dir: Optional[Path] = None
    if args.attachments_dir:
        attachments_dir = (root / args.attachments_dir).resolve()
        if not attachments_dir.exists():
            print(
                f"[WARN] attachments-dir '{attachments_dir}' does not exist; "
                "image resolution will fall back to searching the vault.",
                file=sys.stderr,
            )
            attachments_dir = None

    central_dir = (root / CENTRAL_ATTACHMENTS_DIRNAME).resolve()

    total_md, total_other = count_files(root)
    print(f"Discovered Markdown notes: {total_md}")
    print(f"Discovered other files:   {total_other}")

    note_index, note_stem_index, all_notes_all = build_note_index(root)
    attach_rel_idx, attach_basename_idx = build_attachment_index(root)
    total_indexed_attachments = len(attach_rel_idx)

    all_notes = list(all_notes_all)
    proc_notes = [p for p in all_notes if not in_excluded(p, root)]
    excl_notes = [p for p in all_notes if in_excluded(p, root)]

    print(f"Notes to process (after exclusions): {len(proc_notes)}")
    if excl_notes:
        example = excl_notes[0].relative_to(root)
        print(f"Excluded by folder rule: {len(excl_notes)} (e.g. {example})")

    # Stable UUID for each processed note (used in footer and note_refs.csv)
    note_uuid_map: Dict[Path, str] = {}
    for note in proc_notes:
        existing_uuid: Optional[str] = None
        try:
            text = note.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        if text:
            bottom_meta = extract_bottom_meta(text)
            existing_uuid = (
                bottom_meta.get("uuid")
                or bottom_meta.get("UUID")
                or bottom_meta.get("Uuid")
            )
        note_uuid_map[note] = existing_uuid or str(uuid.uuid4())

    # Note-to-note reference rows for Phase 2 + 3
    note_ref_rows: List[Dict[str, str]] = []

    # Attachments move plan: src_path -> rel_new ("attachments/Note-1.pdf")
    attachment_move_plan: Dict[Path, str] = {}
    used_central_names: Set[str] = set()
    note_attach_seq: Dict[str, int] = {}
    all_inlined_image_paths: Set[Path] = set()
    # Aggregate counters for the final summary/report.
    total_images_inlined = 0
    total_attachment_links = 0
    total_missing_images_count = 0
    total_missing_attachments_count = 0
    total_removed_blob_refs = 0
    total_outbound_note_links = 0
    total_data_uri_images_demoted = 0
    notes_changed_count = 0
    collections_added: Set[str] = set()
    collection_note_counts: Dict[str, int] = defaultdict(int)
    remaining_callout_files: List[str] = []
    remaining_tasks_files: List[str] = []
    remaining_callouts_total = 0
    remaining_tasks_total = 0
    read_errors = 0
    failed_notes = 0

    # Per-note log CSV
    log_path = Path(args.log_csv)
    if not log_path.is_absolute():
        log_path = root / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "skipped_excluded",
        "changed",
        "images_inlined",
        "missing_images",
        "removed_blob_refs",
        "data_uri_images_demoted",
        "attachment_links_rewritten",
        "missing_attachments",
        "outbound_note_links",
        "derived_collection_tags",
        "uuid",
    ]
    rows: List[Dict[str, str]] = []

    # Log excluded notes
    for p in excl_notes:
        rel = p.relative_to(root).as_posix()
        rows.append(
            {
                "file": rel,
                "skipped_excluded": "1",
                "changed": "0",
                "images_inlined": "0",
                "missing_images": "",
                "removed_blob_refs": "",
                "data_uri_images_demoted": "0",
                "attachment_links_rewritten": "0",
                "missing_attachments": "",
                "outbound_note_links": "0",
                "derived_collection_tags": "",
                "uuid": "",
            }
        )

    processed = 0
    print_progress(0, len(proc_notes), filename=None)

    for note in proc_notes:
        rel_name = note.relative_to(root).as_posix()
        try:
            original = note.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"\n[ERROR] Failed to read {rel_name}: {e}", file=sys.stderr)
            read_errors += 1
            rows.append(
                {
                    "file": rel_name,
                    "skipped_excluded": "0",
                    "changed": "0",
                "images_inlined": "0",
                "missing_images": f"READ_ERROR: {e}",
                "removed_blob_refs": "",
                "data_uri_images_demoted": "0",
                "attachment_links_rewritten": "0",
                "missing_attachments": "",
                "outbound_note_links": "0",
                "derived_collection_tags": "",
                "uuid": "",
                }
            )
            processed += 1
            print_progress(processed, len(proc_notes), filename=rel_name)
            continue

        try:
            text_wo_bottom = strip_existing_bottom_meta(original)
            top_meta, body = load_top_yaml(text_wo_bottom)

            # 1) Remove Obsidian Tasks callouts/blocks specific to Daily Notes
            if is_daily_note(note, root):
                body_clean = remove_obsidian_tasks_blocks(body)
            else:
                body_clean = body

            remaining_callouts, remaining_tasks = count_remaining_callouts_and_tasks(body_clean)
            if remaining_callouts:
                remaining_callouts_total += remaining_callouts
                remaining_callout_files.append(rel_name)
            if remaining_tasks:
                remaining_tasks_total += remaining_tasks
                remaining_tasks_files.append(rel_name)

            # 2) Insert Scrybble PDF link (if metadata present) before other rewrites
            body_with_scrybble = ensure_scrybble_pdf_link(body_clean, top_meta, note)

            # 3) Inline images (optional)
            body_inlined, inlined_paths, missing_imgs, removed_blobs = inline_images_in_text(
                body_with_scrybble,
                note_path=note,
                vault_root=root,
                attachments_dir=attachments_dir,
                inline_override=None if args.inline_images else False,
            )
            for p_str in inlined_paths:
                try:
                    inlined_path = Path(p_str).resolve()
                except Exception:
                    continue
                all_inlined_image_paths.add(inlined_path)

            # 4) Demote existing inline data URIs if the note is oversized
            (
                body_deduped,
                demoted_attachment_urls,
                demoted_count,
            ) = demote_inline_data_uri_images_if_needed(
                body_inlined,
                note_path=note,
                vault_root=root,
                central_dir=central_dir,
                used_names=used_central_names,
                note_attach_seq=note_attach_seq,
                dry_run=args.dry_run,
            )

            # 5) Rewrite wikilinks (Option A) + attachments
            (
                body_relinked,
                outbound_notes,
                attachment_urls,
                missing_atts,
            ) = rewrite_links_and_collect(
                body_deduped,
                this_file=note,
                root=root,
                note_index=note_index,
                note_stem_index=note_stem_index,
                attach_rel_idx=attach_rel_idx,
                attach_basename_idx=attach_basename_idx,
                central_dir=central_dir,
                attachment_move_plan=attachment_move_plan,
                used_central_names=used_central_names,
                note_uuid_map=note_uuid_map,
                note_ref_rows=note_ref_rows,
                note_attach_seq=note_attach_seq,
            )

            # Combine attachment counts (demoted data URIs + rewritten links)
            attachment_urls = demoted_attachment_urls + attachment_urls

            # 6) Collections from folder structure
            collection_tags = tags_for_path(root, note)
            for tag in set(collection_tags):
                collection_note_counts[tag] += 1
            tags_line = " ".join(collection_tags)
            body_no_dup_tags, _ = strip_trailing_collection_tags_block(body_relinked, tags_line)
            collections_added.update(collection_tags)

            # 7) Metadata footer with UUID from note_uuid_map
            meta = dict(top_meta or {})
            meta["uuid"] = note_uuid_map[note]
            bottom_block = render_bottom_meta(meta)

            final_text = body_no_dup_tags.rstrip()
            if tags_line:
                final_text += "\n\n" + tags_line
            final_text += "\n\n---\n" + bottom_block

            changed = final_text != original

            if changed and not args.dry_run:
                # Preserve timestamps
                try:
                    st = note.stat()
                except OSError as e:
                    print(f"[WARN] Could not stat note {rel_name} before write: {e}", file=sys.stderr)
                    st = None

                tmp = note.with_suffix(note.suffix + ".tmpwrite")
                try:
                    with tmp.open("w", encoding="utf-8", newline="\n") as f:
                        f.write(final_text)
                        if not final_text.endswith("\n"):
                            f.write("\n")
                    os.replace(tmp, note)
                    if st is not None:
                        os.utime(note, (st.st_atime, st.st_mtime))
                except Exception as e:
                    print(f"\n[ERROR] Failed to write {rel_name}: {e}", file=sys.stderr)

            total_images_inlined += len(inlined_paths)
            total_attachment_links += len(attachment_urls)
            total_missing_images_count += len(missing_imgs)
            total_missing_attachments_count += len(missing_atts)
            total_removed_blob_refs += len(removed_blobs)
            total_outbound_note_links += len(outbound_notes)
            total_data_uri_images_demoted += demoted_count
            if changed:
                notes_changed_count += 1

            rows.append(
                {
                    "file": rel_name,
                    "skipped_excluded": "0",
                    # "changed" indicates whether text WOULD change,
                    # even in dry-run mode we keep this info.
                    "changed": "1" if changed else "0",
                    "images_inlined": str(len(inlined_paths)),
                    "missing_images": "; ".join(missing_imgs),
                    "removed_blob_refs": "; ".join(removed_blobs),
                    "data_uri_images_demoted": str(demoted_count),
                    "attachment_links_rewritten": str(len(attachment_urls)),
                    "missing_attachments": "; ".join(missing_atts),
                    "outbound_note_links": str(len(outbound_notes)),
                    "derived_collection_tags": " ".join(collection_tags),
                    "uuid": note_uuid_map[note],
                }
            )

        except Exception as e:
            print(f"\n[ERROR] Unexpected error while processing {rel_name}: {e}", file=sys.stderr)
            failed_notes += 1
            rows.append(
                {
                    "file": rel_name,
                    "skipped_excluded": "0",
                    "changed": "0",
                "images_inlined": "0",
                "missing_images": f"UNEXPECTED_ERROR: {e}",
                "removed_blob_refs": "",
                "data_uri_images_demoted": "0",
                "attachment_links_rewritten": "0",
                "missing_attachments": "",
                "outbound_note_links": "0",
                "derived_collection_tags": "",
                "uuid": "",
                }
            )

        processed += 1
        print_progress(processed, len(proc_notes), filename=rel_name)

    print()  # newline after progress bar

    # Write per-note log CSV
    try:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-note log written to: {log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV log {log_path}: {e}", file=sys.stderr)

    # Write note_refs.csv (for Phase 2 + 3)
    refs_csv_path = root / "note_refs.csv"
    if note_ref_rows:
        refs_fieldnames = [
            "source_file",
            "source_title",
            "source_uuid",
            "final_link",
            "target_file",
            "target_title",
            "target_uuid",
        ]
        try:
            with refs_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=refs_fieldnames)
                writer.writeheader()
                writer.writerows(note_ref_rows)
            print(f"Note-reference CSV written to: {refs_csv_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write note-ref CSV {refs_csv_path}: {e}", file=sys.stderr)
    else:
        print("No note→note references found; note_refs.csv not created.")

    linked_notes_dir = None
    linked_notes_candidates = 0
    linked_notes_moved = 0
    linked_notes_missing = 0
    linked_notes_collisions = 0
    linked_notes_rewritten_links = 0
    linked_notes_rewritten_files = 0
    if args.move_linked_notes:
        if not note_ref_rows:
            print("\nLinked notes move skipped: no note references available.")
        else:
            linked_notes_dir = (root / args.linked_notes_folder).resolve()
            if not args.dry_run:
                linked_notes_dir.mkdir(parents=True, exist_ok=True)

            linked_note_paths: Set[Path] = set()
            for row in note_ref_rows:
                source_rel = (row.get("source_file") or "").strip()
                target_rel = (row.get("target_file") or "").strip()
                if source_rel:
                    linked_note_paths.add(Path(source_rel))
                if target_rel:
                    linked_note_paths.add(Path(target_rel))

            linked_notes_candidates = len(linked_note_paths)
            if linked_notes_candidates:
                print(f"\nMoving linked notes into: {linked_notes_dir}")
                (
                    linked_notes_moved,
                    linked_notes_missing,
                    linked_notes_collisions,
                    linked_name_map,
                    linked_mtime_map,
                ) = move_linked_notes_flat(
                    root,
                    linked_notes_dir,
                    linked_note_paths,
                    dry_run=args.dry_run,
                )
                linked_notes_rewritten_links, linked_notes_rewritten_files = rewrite_links(
                    root,
                    linked_notes_dir,
                    linked_name_map,
                    linked_mtime_map,
                    dry_run=args.dry_run,
                )
                print(f"Linked notes referenced:        {linked_notes_candidates}")
                print(f"Linked notes moved:             {linked_notes_moved}")
                if linked_notes_missing:
                    print(f"Linked notes missing:           {linked_notes_missing}")
                if linked_notes_collisions:
                    print(f"Linked filename collisions:     {linked_notes_collisions}")
                print(f"Linked note links rewritten:    {linked_notes_rewritten_links}")
                if linked_notes_rewritten_files:
                    print(f"Linked notes with rewrites:     {linked_notes_rewritten_files}")
            else:
                print("\nLinked notes move skipped: no linked notes found.")

    added_unreferenced_attachments = 0
    skipped_unreferenced_attachments = 0
    referenced_attachments = len(attachment_move_plan)
    if args.move_all_non_md:
        print("\nAdding unreferenced non-.md files to attachments move plan ...")
        added_unreferenced_attachments, skipped_unreferenced_attachments = (
            add_unreferenced_attachments_to_plan(
                root,
                central_dir,
                attachment_move_plan,
                used_central_names,
                note_attach_seq,
            )
        )
        if added_unreferenced_attachments:
            print(f"Unreferenced attachments added to plan: {added_unreferenced_attachments}")
        if skipped_unreferenced_attachments:
            print(f"Unreferenced attachments skipped:     {skipped_unreferenced_attachments}")

    image_attachments_to_move = sum(
        1 for src_path in attachment_move_plan.keys() if is_image_ext(src_path.suffix)
    )
    images_deleted_count, images_planned_for_deletion = delete_inlined_images(
        all_inlined_image_paths,
        root,
        attachment_move_plan,
        args.dry_run,
    )

    # Move attachments according to plan (unless dry-run)
    total_planned_attachments = len(attachment_move_plan)
    if args.dry_run:
        print("\nDRY RUN: attachments would be moved as follows, but no moves were performed:")
        print(f"  Total indexed attachments in vault:   {total_indexed_attachments}")
        print(f"  Referenced attachments to centralise: {referenced_attachments}")
        if args.move_all_non_md:
            print(f"  Unreferenced attachments to centralise: {added_unreferenced_attachments}")
        print(f"  Total planned attachments to move:    {total_planned_attachments}")
    else:
        move_planned_attachments(root, central_dir, attachment_move_plan)

    # Summary
    notes_missing_imgs = sum(1 for r in rows if r.get("missing_images"))
    notes_missing_atts = sum(1 for r in rows if r.get("missing_attachments"))
    print(f"\nProcessed {processed} notes (of {len(proc_notes)} non-excluded).")
    if notes_missing_imgs:
        print(f"Notes with missing images:       {notes_missing_imgs}")
    if notes_missing_atts:
        print(f"Notes with missing attachments:  {notes_missing_atts}")
    print(f"Notes changed:                   {notes_changed_count}")
    if read_errors:
        print(f"Notes failed to read:            {read_errors}")
    if failed_notes:
        print(f"Notes with processing errors:    {failed_notes}")
    print(f"Images inlined as data URIs:     {total_images_inlined}")
    print(f"Attachment links rewritten:      {total_attachment_links}")
    print(f"Inline data URIs demoted:        {total_data_uri_images_demoted}")
    print(f"Inlined images deleted:          {images_deleted_count}")
    print(f"Image files moved as attachments:{image_attachments_to_move}")
    if args.dry_run and images_planned_for_deletion:
        print(f"(Dry run) Inline images slated for deletion: {images_planned_for_deletion}")
    print(f"Outbound note links logged:      {total_outbound_note_links}")
    if remaining_callouts_total:
        print(f"Remaining callouts:             {remaining_callouts_total}")
    if remaining_tasks_total:
        print(f"Remaining ```tasks blocks:      {remaining_tasks_total}")
    if args.move_linked_notes:
        if linked_notes_dir is not None:
            print(f"Linked notes folder:             {linked_notes_dir}")
        print(f"Linked notes referenced:         {linked_notes_candidates}")
        print(f"Linked notes moved:              {linked_notes_moved}")
        if linked_notes_missing:
            print(f"Linked notes missing:            {linked_notes_missing}")
        if linked_notes_collisions:
            print(f"Linked filename collisions:      {linked_notes_collisions}")
        print(f"Linked note links rewritten:     {linked_notes_rewritten_links}")
        if linked_notes_rewritten_files:
            print(f"Linked notes with rewrites:      {linked_notes_rewritten_files}")
    if total_removed_blob_refs:
        print(f"Removed blob:// image refs:      {total_removed_blob_refs}")
    if total_missing_images_count:
        print(f"Missing image references logged: {total_missing_images_count}")
    if total_missing_attachments_count:
        print(f"Missing attachment refs logged:  {total_missing_attachments_count}")
    print(f"Vault root: {root}")
    print(f"Central attachments folder: {central_dir}")
    print(f"Attachments CDN base: {ROOT_CDN_BASE}")
    print(f"Total indexed attachments in vault:   {total_indexed_attachments}")
    print(f"Referenced attachments to centralise: {referenced_attachments}")
    if args.move_all_non_md:
        print(f"Unreferenced attachments to centralise: {added_unreferenced_attachments}")
        if skipped_unreferenced_attachments:
            print(f"Unreferenced attachments skipped:      {skipped_unreferenced_attachments}")
        print(f"Total planned attachments to move:     {total_planned_attachments}")
    print(f"Run hashtag for this import: {RUN_HASHTAG}")
    if args.dry_run:
        print("NOTE: DRY RUN mode – no files were modified or moved.")

    if remaining_callout_files:
        print("\nFiles with remaining callouts:")
        for name in sorted(set(remaining_callout_files)):
            print(f"  {name}")
    if remaining_tasks_files:
        print("\nFiles with remaining ```tasks blocks:")
        for name in sorted(set(remaining_tasks_files)):
            print(f"  {name}")

    run_finished = datetime.datetime.now()
    duration = run_finished - run_started
    stats_items: List[Tuple[str, str]] = [
        ("Run started", run_started.isoformat()),
        ("Run finished", run_finished.isoformat()),
        ("Duration", str(duration)),
        ("Dry run", "Yes" if args.dry_run else "No"),
        ("Vault root", str(root)),
        ("Central attachments folder", str(central_dir)),
        ("Attachments CDN base", ROOT_CDN_BASE),
        ("Run hashtag", RUN_HASHTAG),
        ("Notes processed", f"{processed} / {len(proc_notes)}"),
        ("Notes excluded", str(len(excl_notes))),
        ("Notes changed", str(notes_changed_count)),
        ("Notes failed to read", str(read_errors)),
        ("Notes with processing errors", str(failed_notes)),
        ("Notes with missing images", str(notes_missing_imgs)),
        ("Missing image references", str(total_missing_images_count)),
        ("Notes with missing attachments", str(notes_missing_atts)),
        ("Missing attachment references", str(total_missing_attachments_count)),
        ("Images inlined", str(total_images_inlined)),
        ("Attachment links rewritten", str(total_attachment_links)),
        ("Inline data URIs demoted", str(total_data_uri_images_demoted)),
        ("Inlined images deleted", str(images_deleted_count)),
        ("Image attachments to move", str(image_attachments_to_move)),
        ("Outbound note links logged", str(total_outbound_note_links)),
        ("Remaining callouts", str(remaining_callouts_total)),
        ("Remaining ```tasks blocks", str(remaining_tasks_total)),
        ("Move all non-md attachments", "Yes" if args.move_all_non_md else "No"),
        ("Referenced attachments to centralise", str(referenced_attachments)),
        ("Unreferenced attachments to centralise", str(added_unreferenced_attachments)),
        ("Unreferenced attachments skipped", str(skipped_unreferenced_attachments)),
        ("Total planned attachments to move", str(total_planned_attachments)),
        ("Linked notes move enabled", "Yes" if args.move_linked_notes else "No"),
        ("Linked notes folder", str(linked_notes_dir) if linked_notes_dir else "n/a"),
        ("Linked notes referenced", str(linked_notes_candidates)),
        ("Linked notes moved", str(linked_notes_moved)),
        ("Linked notes missing", str(linked_notes_missing)),
        ("Linked filename collisions", str(linked_notes_collisions)),
        ("Linked notes with rewrites", str(linked_notes_rewritten_files)),
        ("Linked note links rewritten", str(linked_notes_rewritten_links)),
        ("Blob references removed", str(total_removed_blob_refs)),
        ("Indexed attachments in vault", str(total_indexed_attachments)),
    ]

    console_output = capture.get_text()
    capture.stop()
    try:
        write_markdown_report(report_path, stats_items, console_output, collection_note_counts)
        print(f"Run report written to: {report_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write Markdown report {report_path}: {e}", file=sys.stderr)

    summary_title = f"Obsidian to Mem Summary {RUN_HASHTAG_DATE}.md"
    summary_path = root / summary_title
    try:
        write_summary_note(summary_path, stats_items, collections_added)
        print(f"Summary note written to: {summary_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write summary note {summary_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
