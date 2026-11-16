"""Runtime and source provenance captured with every experiment."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import torch


def source_files(workspace: Path):
    roots = [
        workspace / "comparison_runners",
        workspace / "experiment",
        workspace / "methods",
        workspace / "tools",
        workspace / "tests",
    ]
    files = [workspace / "run_experiment.py"]
    for root in roots:
        files.extend(root.rglob("*.py"))
    files.extend(workspace.glob("requirements*.txt"))
    return sorted(
        (path for path in files if "__pycache__" not in path.parts),
        key=lambda path: path.relative_to(workspace).as_posix(),
    )


def source_hashes(workspace: Path) -> dict[str, str]:
    return {
        path.relative_to(workspace).as_posix(): hashlib.sha256(
            path.read_bytes()).hexdigest()
        for path in source_files(workspace)
    }


def source_fingerprint(workspace: Path) -> str:
    digest = hashlib.sha256()
    for name, value in source_hashes(workspace).items():
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def runtime_provenance(workspace: Path) -> dict:
    cuda_devices = (
        [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available() else []
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": len(cuda_devices),
        "cuda_devices": cuda_devices,
        "source_fingerprint_sha256": source_fingerprint(workspace),
    }
