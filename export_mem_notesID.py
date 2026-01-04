#!/usr/bin/env python3
"""
Export Mem notes IDs via List Notes API to CSV with:
- .env support for MEM_API_KEY (python-dotenv)
- Console progress updates
- File + console logging
- Robust error handling + retries
- Rate-limit-aware pacing using Mem headers
- Two-pass export (JSONL temp -> CSV) to include ALL returned fields safely

Setup:
  pip install requests python-dotenv

.env file (same directory as this script):
  MEM_API_KEY=your_actual_mem_api_key_here

Usage:
  python export_mem_notesID.py
  python export_mem_notesID.py --limit 500 --output mem_notes.csv --log-file mem_export.log
  python export_mem_notesID.py --api-key ...   # overrides .env
  python export_mem_notesID.py --debug         # enhanced diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import requests
from dotenv import load_dotenv


# ----------------------------- logging ---------------------------------


def setup_logger(log_file: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("mem_export")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if script is imported/re-run
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ----------------------------- helpers ---------------------------------


def _to_int(val: Optional[str]) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def flatten(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict/list JSON into a single dict with dotted keys.
    Lists are indexed: key.0, key.1, ...
    """
    out: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            out.update(flatten(v, key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}{sep}{i}" if prefix else str(i)
            out.update(flatten(v, key, sep=sep))
    else:
        out[prefix] = obj

    return out


@dataclass
class RateState:
    # Requests
    rate_bucket: Optional[str] = None
    rate_limit: Optional[int] = None
    rate_remaining: Optional[int] = None
    rate_reset_s: Optional[int] = None

    # Complexity tokens
    cx_bucket: Optional[str] = None
    cx_limit: Optional[int] = None
    cx_remaining: Optional[int] = None
    cx_reset_s: Optional[int] = None

    retry_after_s: Optional[int] = None

    def best_wait_seconds_if_needed(self) -> float:
        """
        Decide whether we should pause to avoid 429 by looking at remaining tokens.
        We pause only when we're basically at the edge (<= 1 remaining).
        """
        waits: List[float] = []

        if self.rate_remaining is not None and self.rate_reset_s is not None:
            if self.rate_remaining <= 1:
                waits.append(float(self.rate_reset_s))

        if self.cx_remaining is not None and self.cx_reset_s is not None:
            if self.cx_remaining <= 1:
                waits.append(float(self.cx_reset_s))

        if self.retry_after_s is not None:
            waits.append(float(self.retry_after_s))

        return max(waits) if waits else 0.0


def parse_rate_headers(headers: Dict[str, str]) -> RateState:
    rs = RateState(
        rate_bucket=headers.get("X-RateLimit-Bucket"),
        rate_limit=_to_int(headers.get("X-RateLimit-Limit")),
        rate_remaining=_to_int(headers.get("X-RateLimit-Remaining")),
        rate_reset_s=_to_int(headers.get("X-RateLimit-Reset")),
        cx_bucket=headers.get("X-Complexity-Bucket"),
        cx_limit=_to_int(headers.get("X-Complexity-Limit")),
        cx_remaining=_to_int(headers.get("X-Complexity-Remaining")),
        cx_reset_s=_to_int(headers.get("X-Complexity-Reset")),
    )
    return rs


def log_rate_state(logger: logging.Logger, rs: RateState, level: int = logging.DEBUG) -> None:
    parts = []
    if rs.rate_limit is not None:
        parts.append(
            f"rate[{rs.rate_bucket}] remaining={rs.rate_remaining}/{rs.rate_limit} reset_s={rs.rate_reset_s}"
        )
    if rs.cx_limit is not None:
        parts.append(
            f"complexity[{rs.cx_bucket}] remaining={rs.cx_remaining}/{rs.cx_limit} reset_s={rs.cx_reset_s}"
        )
    if parts:
        logger.log(level, "Rate status: " + " | ".join(parts))


# -------------------------- request layer -------------------------------


class MemApiError(RuntimeError):
    pass


def request_with_rate_aware_retry(
    session: requests.Session,
    logger: logging.Logger,
    method: str,
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Union[str, int]]] = None,
    timeout_s: int = 30,
    max_retries: int = 8,
) -> Dict[str, Any]:
    """
    Make an HTTP request with:
    - Handling for transient errors and 429s
    - Observing Retry-After if present
    - Proactive rate-limit pacing based on response headers
    """
    backoff = 1.0

    for attempt in range(max_retries + 1):
        try:
            resp = session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=timeout_s,
            )
        except requests.RequestException as e:
            if attempt == max_retries:
                raise MemApiError(f"Network error after retries: {e}") from e
            logger.warning("Network error (%s). Retry in %.1fs (attempt %d/%d).", e, backoff, attempt + 1, max_retries)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        # Parse rate headers for logging + pacing (even on errors)
        rs = parse_rate_headers(dict(resp.headers))
        log_rate_state(logger, rs, level=logging.DEBUG)

        # Handle success
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError as e:
                raise MemApiError(f"Invalid JSON response: {resp.text[:500]}") from e

            # Proactively sleep if we're about to hit limits
            wait_s = rs.best_wait_seconds_if_needed()
            if wait_s > 0:
                sleep_s = wait_s + 0.5
                logger.info("Rate limit nearly exhausted; sleeping %.1fs before next request.", sleep_s)
                time.sleep(sleep_s)

            return data

        # Rate limit exceeded
        if resp.status_code == 429:
            retry_after = _to_int(resp.headers.get("Retry-After")) or 1
            rs.retry_after_s = retry_after
            log_rate_state(logger, rs, level=logging.INFO)
            if attempt == max_retries:
                raise MemApiError(f"429 rate limited after retries. Retry-After={retry_after}s. Body={resp.text[:500]}")
            sleep_s = float(retry_after) + 0.5
            logger.warning("429 rate limited. Sleeping %.1fs then retrying (attempt %d/%d).", sleep_s, attempt + 1, max_retries)
            time.sleep(sleep_s)
            continue

        # Transient server errors
        if resp.status_code in (500, 502, 503, 504):
            if attempt == max_retries:
                raise MemApiError(f"Server error {resp.status_code} after retries: {resp.text[:500]}")
            logger.warning(
                "Server error %d. Retry in %.1fs (attempt %d/%d).",
                resp.status_code,
                backoff,
                attempt + 1,
                max_retries,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        # Non-retryable errors
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise MemApiError(f"Request failed: HTTP {resp.status_code} | {detail}")

    raise MemApiError("Unexpected failure: request loop exited.")


# ----------------------------- export ----------------------------------


def iter_notes(
    api_key: str,
    limit: int,
    logger: logging.Logger,
    base_url: str = "https://api.mem.ai/v2",
) -> Iterable[Dict[str, Any]]:
    """
    Yield notes from paginated List Notes endpoint until next_page is null/missing.
    """
    url = f"{base_url.rstrip('/')}/notes"
    headers = {"Authorization": f"Bearer {api_key}"}

    session = requests.Session()

    next_page: Optional[str] = None
    page_num = 0
    total_notes = 0

    while True:
        page_num += 1
        params: Dict[str, Union[str, int]] = {"limit": limit}
        if next_page:
            params["page"] = next_page

        logger.info("Fetching page %d (limit=%d)%s ...", page_num, limit, f", page={next_page}" if next_page else "")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Request URL=%s params=%s", url, params)
        data = request_with_rate_aware_retry(
            session=session,
            logger=logger,
            method="GET",
            url=url,
            headers=headers,
            params=params,
        )

        total = data.get("total")

        notes = data.get("notes")
        if notes is None:
            notes = data.get("results", [])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Response keys=%s next_page=%s", sorted(data.keys()), data.get("next_page"))
            logger.debug("Notes list source=%s count=%s", "notes" if "notes" in data else "results", len(notes) if isinstance(notes, list) else "n/a")
        if not isinstance(notes, list):
            raise MemApiError(
                "Unexpected response shape: expected list at 'notes' or 'results' "
                f"(got {type(notes)})."
            )

        batch_count = len(notes)
        total_notes += batch_count
        logger.info("Fetched page %d: %d notes (total so far: %d).", page_num, batch_count, total_notes)
        if total is not None:
            logger.info("API reported total notes: %s", total)
        print(
            f"Progress: {total_notes} notes processed across {page_num} pages"
            + (f" (API total: {total})" if total is not None else ""),
            flush=True,
        )
        if page_num == 1 and batch_count == 0:
            logger.warning("First page returned zero notes. Check API key, base URL, and account permissions.")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("First page payload (truncated): %s", json.dumps(data, ensure_ascii=False)[:1000])

        for note in notes:
            if isinstance(note, dict):
                yield note
            else:
                yield {"raw_value": note}

        next_page = data.get("next_page")
        if not next_page:
            logger.info("No next_page token returned. Finished paging.")
            break


def export_notes_to_csv(
    api_key: str,
    limit: int,
    output_csv: str,
    logger: logging.Logger,
    base_url: str = "https://api.mem.ai/v2",
) -> None:
    """
    Two-pass export:
      Pass 1: Fetch + flatten notes -> write each flattened row to temp JSONL
              Accumulate union of all keys for the CSV header.
      Pass 2: Read JSONL -> write CSV with stable header order.
    """
    fieldnames_set: Set[str] = set()
    preferred = ["mem_id", "title", "created_at", "updated_at", "raw_json"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name

        logger.info("Pass 1/2: fetching notes and building schema...")
        note_count = 0

        for note in iter_notes(api_key, limit, logger, base_url=base_url):
            note_count += 1

            mem_id = note.get("id") if isinstance(note, dict) else None
            title = note.get("title") if isinstance(note, dict) else None

            flat = flatten(note)
            flat["mem_id"] = mem_id
            flat["title"] = title
            flat["raw_json"] = json.dumps(note, ensure_ascii=False)

            fieldnames_set.update(flat.keys())
            tmp.write(json.dumps(flat, ensure_ascii=False) + "\n")

            if note_count % 250 == 0:
                logger.info("Progress: processed %d notes...", note_count)

        logger.info("Pass 1 complete: processed %d notes.", note_count)
        if note_count == 0:
            logger.warning("No notes were returned. Try --debug to inspect the API response.")

    rest = sorted([f for f in fieldnames_set if f not in preferred])
    fieldnames = [f for f in preferred if f in fieldnames_set] + rest

    logger.info("Pass 2/2: writing CSV to %s ...", output_csv)
    written = 0

    try:
        with open(tmp_path, "r", encoding="utf-8") as src, open(output_csv, "w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSONL line.")
                    continue
                writer.writerow(row)
                written += 1
                if written % 500 == 0:
                    logger.info("CSV progress: wrote %d rows...", written)

        logger.info("Done: wrote %d rows to %s", written, output_csv)
    finally:
        try:
            os.remove(tmp_path)
            logger.debug("Removed temp file: %s", tmp_path)
        except OSError:
            logger.warning("Could not remove temp file: %s", tmp_path)


# ------------------------------ main -----------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Mem notes (List Notes API) to CSV with .env + rate-limit-aware retries")
    parser.add_argument("--api-key", default=None, help="Mem API key (overrides .env/env vars)")
    parser.add_argument("--limit", type=int, default=500, help="List Notes page size (default: 500)")
    parser.add_argument("--output", default="mem_notes.csv", help="Output CSV file (default: mem_notes.csv)")
    parser.add_argument("--log-file", default="mem_export.log", help="Log file path (default: mem_export.log)")
    parser.add_argument("--base-url", default="https://api.mem.ai/v2", help="API base URL (default: https://api.mem.ai/v2)")
    parser.add_argument("--verbose", action="store_true", help="Verbose console logging (debug)")
    parser.add_argument("--debug", action="store_true", help="Enable enhanced debug logging and diagnostics")
    args = parser.parse_args()

    # Load .env from current working directory (and do not overwrite already-set env vars)
    load_dotenv(override=False)

    debug_enabled = args.verbose or args.debug
    logger = setup_logger(args.log_file, debug_enabled)

    api_key = args.api_key or os.environ.get("MEM_API_KEY")
    if not api_key:
        logger.error(
            "Missing API key.\n"
            "Create a .env file in this folder containing:\n"
            "  MEM_API_KEY=your_actual_mem_api_key_here\n"
            "or set MEM_API_KEY as an environment variable, or pass --api-key."
        )
        return 2

    if args.limit <= 0:
        logger.error("--limit must be a positive integer.")
        return 2

    try:
        export_notes_to_csv(
            api_key=api_key,
            limit=args.limit,
            output_csv=args.output,
            logger=logger,
            base_url=args.base_url,
        )
    except MemApiError as e:
        logger.error("Mem API error: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
