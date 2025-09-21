"""IO utilities."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Mapping


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Mapping) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Mapping:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def atomic_write(path: Path, mode: str = "w", encoding: str = "utf-8") -> Generator:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    ensure_dir(path.parent)
    try:
        with tmp_path.open(mode, encoding=encoding) as handle:
            yield handle
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)