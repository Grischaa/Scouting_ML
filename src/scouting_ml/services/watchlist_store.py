"""File-backed watchlist persistence with atomic writes and basic locking."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


logger = logging.getLogger(__name__)
_LOCK_TIMEOUT_SECONDS = 2.0
_LOCK_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class WatchlistReadResult:
    records: list[dict[str, Any]]
    recovered: bool
    skipped_lines: int


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _acquire_lock(path: Path) -> Iterator[None]:
    lock_path = _lock_path(path)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring watchlist lock for {path}") from exc
            time.sleep(_LOCK_POLL_SECONDS)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def read_watchlist_records(path: Path) -> WatchlistReadResult:
    if not path.exists():
        return WatchlistReadResult(records=[], recovered=False, skipped_lines=0)

    records: list[dict[str, Any]] = []
    skipped_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                skipped_lines += 1

    if skipped_lines:
        logger.warning(
            "Recovered watchlist read by skipping malformed lines",
            extra={"path": str(path), "skipped_lines": skipped_lines},
        )

    return WatchlistReadResult(
        records=records,
        recovered=bool(skipped_lines),
        skipped_lines=skipped_lines,
    )


def _write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary watchlist file: %s", temp_path)


def _backup_corrupted_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.copy2(path, backup_path)
    logger.warning("Backed up watchlist before rewriting recovered file: %s", backup_path)
    return backup_path


def _normalized_dedupe_value(key: str, value: Any) -> str:
    text = str(value or "").strip()
    if key in {"player_id", "tag"}:
        return text.casefold()
    if key == "split":
        return text.lower()
    return text


def _dedupe_key(record: dict[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalized_dedupe_value(key, record.get(key)) for key in keys)


def upsert_watchlist_record(
    path: Path,
    record: dict[str, Any],
    *,
    dedupe_keys: Sequence[str] = ("player_id", "split", "season", "tag"),
) -> dict[str, Any]:
    with _acquire_lock(path):
        read_result = read_watchlist_records(path)
        records = list(read_result.records)
        incoming_key = _dedupe_key(record, dedupe_keys)
        updated: dict[str, Any] | None = None

        for idx, existing in enumerate(records):
            if _dedupe_key(existing, dedupe_keys) != incoming_key:
                continue
            merged = dict(existing)
            merged.update(record)
            merged["watch_id"] = existing.get("watch_id", record.get("watch_id"))
            records[idx] = merged
            updated = merged
            break

        if updated is None:
            updated = dict(record)
            records.append(updated)

        if read_result.recovered:
            _backup_corrupted_file(path)
        _write_records(path, records)
        return updated


def delete_watchlist_record(path: Path, watch_id: str) -> bool:
    with _acquire_lock(path):
        read_result = read_watchlist_records(path)
        records = list(read_result.records)
        kept = [row for row in records if str(row.get("watch_id")) != str(watch_id)]
        deleted = len(kept) != len(records)
        if not deleted:
            return False
        if read_result.recovered:
            _backup_corrupted_file(path)
        _write_records(path, kept)
        return True


__all__ = [
    "WatchlistReadResult",
    "delete_watchlist_record",
    "read_watchlist_records",
    "upsert_watchlist_record",
]
