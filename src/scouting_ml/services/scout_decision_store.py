"""File-backed scout decision persistence with atomic writes and append-only semantics."""

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
class ScoutDecisionReadResult:
    """Result wrapper for reading decision records from disk."""

    records: list[dict[str, Any]]
    recovered: bool
    skipped_lines: int


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


@contextmanager
def _acquire_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive file lock for one decision file."""
    lock_path = _lock_path(path)
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring scout-decision lock for {path}") from exc
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


def read_scout_decision_records(path: Path) -> ScoutDecisionReadResult:
    """Read scout-decision records, recovering by skipping malformed lines."""
    if not path.exists():
        return ScoutDecisionReadResult(records=[], recovered=False, skipped_lines=0)

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
            "Recovered scout-decision read by skipping malformed lines",
            extra={"path": str(path), "skipped_lines": skipped_lines},
        )

    return ScoutDecisionReadResult(
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
                logger.warning("Failed to clean up temporary scout-decision file: %s", temp_path)


def _backup_corrupted_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.copy2(path, backup_path)
    logger.warning("Backed up scout-decision file before rewriting recovered file: %s", backup_path)
    return backup_path


def append_scout_decision_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one decision record to the JSONL store using an atomic rewrite."""
    with _acquire_lock(path):
        read_result = read_scout_decision_records(path)
        records = list(read_result.records)
        records.append(dict(record))
        if read_result.recovered:
            _backup_corrupted_file(path)
        _write_records(path, records)
    return dict(record)


__all__ = [
    "ScoutDecisionReadResult",
    "append_scout_decision_record",
    "read_scout_decision_records",
]
