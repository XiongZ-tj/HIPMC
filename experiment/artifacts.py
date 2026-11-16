"""Crash-safe artifact writes used by long-running experiments."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Sequence


def atomic_torch_save(value: object, path: str | Path) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)


def atomic_json_save(value: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_csv_save(
        rows: Sequence[dict],
        path: str | Path,
        *,
        fieldnames: Sequence[str] | None = None) -> None:
    """Atomically persist scalar training records with stable union columns."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    fields = list(fieldnames or dict.fromkeys(
        key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, target)
