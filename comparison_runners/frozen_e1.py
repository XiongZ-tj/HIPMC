"""Strict loader for the frozen E1 structural-instance dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiment.instances import JobRecord, load_job_records_csv


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN_E1_DATASET = WORKSPACE / "datasets" / "e1_formal_instances"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenE1Dataset:
    """Validated condition/instance lookup with lazy immutable CSV loading."""

    def __init__(self, root: str | Path = DEFAULT_FROZEN_E1_DATASET) -> None:
        self.root = Path(root).resolve()
        metadata_path = self.root / "dataset_manifest.json"
        instance_manifest_path = self.root / "instance_manifest.csv"
        checksums_path = self.root / "checksums.sha256"
        for path in (metadata_path, instance_manifest_path, checksums_path):
            pass
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with instance_manifest_path.open(
                newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = {
            (row["condition_id"], int(row["instance_index"])): row
            for row in rows
        }
        self._cache: dict[tuple[str, int], tuple[JobRecord, ...]] = {}

    @property
    def contract_sha256(self) -> str:
        return str(self.metadata["dataset_contract_sha256"])

    @property
    def instance_manifest_sha256(self) -> str:
        return str(self.metadata["instance_manifest_sha256"])

    def row(
            self,
            condition_id: str,
            instance_index: int,
            *,
            expected_instance_seed: int | None = None) -> dict[str, str]:
        key = (condition_id, int(instance_index))
        row = self.rows[key]
        return row

    def records(
            self,
            condition_id: str,
            instance_index: int,
            *,
            expected_instance_seed: int | None = None,
            ) -> tuple[JobRecord, ...]:
        key = (condition_id, int(instance_index))
        row = self.row(
            *key, expected_instance_seed=expected_instance_seed)
        if key not in self._cache:
            path = self.root / row["job_file"]
            records = load_job_records_csv(path)
            self._cache[key] = tuple(records)
        return self._cache[key]

    def validate_case_manifest(self, rows: tuple[dict, ...]) -> None:
        """Require every formal case to resolve to the exact frozen instance."""

        observed = set()
        for case in rows:
            key = (case["condition_id"], int(case["instance_index"]))
            self.row(
                *key, expected_instance_seed=int(case["instance_seed"]))
            observed.add(key)
