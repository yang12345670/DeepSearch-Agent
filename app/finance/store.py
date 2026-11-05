"""Content-addressed financial snapshots, published atomically without replacement."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from app.config import settings


def _snapshot_path(dataset_id: str) -> Path:
    if not isinstance(dataset_id, str) or re.fullmatch(r"[0-9a-f]{64}", dataset_id) is None:
        raise ValueError("dataset_id must be a 64-character lowercase SHA-256 identifier")
    return Path(settings.financial_cache_dir).resolve() / f"{dataset_id}.json"


def _read_bytes(path: Path) -> bytes:
    try:
        # Never follow a snapshot symlink, even if it points inside the cache.
        if path.is_symlink():
            raise ValueError("Snapshot symlinks are not allowed")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("Snapshot must be a regular file")
            return handle.read()
    except OSError as exc:
        raise ValueError("Snapshot is missing or cannot be read; call get_financials again") from exc


def save_dataset(dataset: dict) -> str:
    """Save canonical JSON once and return its SHA-256 content identifier."""
    if not isinstance(dataset, dict):
        raise ValueError("Financial dataset must be a JSON object")
    try:
        payload = json.dumps(
            dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Financial dataset must contain only finite JSON values") from exc
    dataset_id = hashlib.sha256(payload).hexdigest()
    path = _snapshot_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".snapshot-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Atomic create-if-absent: concurrent writers cannot overwrite a snapshot.
            os.link(temporary, path)
        except FileExistsError:
            if _read_bytes(path) != payload:
                raise ValueError("Existing snapshot failed integrity verification")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dataset_id


def load_dataset(dataset_id: str) -> dict:
    """Load a snapshot only when its bytes still match its identifier."""
    payload = _read_bytes(_snapshot_path(dataset_id))
    if hashlib.sha256(payload).hexdigest() != dataset_id:
        raise ValueError("Snapshot failed integrity verification; call get_financials again")
    try:
        dataset = json.loads(payload)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Snapshot is not valid JSON") from exc
    if not isinstance(dataset, dict):
        raise ValueError("Snapshot must contain a financial dataset object")
    return dataset
