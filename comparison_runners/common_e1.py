"Shared E1 comparison-case manifest and common-random-number contract."

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from experiment.artifacts import atomic_csv_save
from experiment.config import ConditionConfig
from experiment.evaluation import EvaluationCase, EvaluationProtocol


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_E1_CASE_MANIFEST = (
    WORKSPACE / "experiment_manifests" / "e1_formal_comparison_cases.csv"
)
CASE_MANIFEST_SCHEMA_VERSION = 2
CASE_MANIFEST_FIELDS = (
    "schema_version",
    "condition_index",
    "condition_id",
    "nominal_finite_load",
    "due_date_factor",
    "total_jobs",
    "initial_jobs",
    "initial_job_fraction",
    "degradation_multiplier",
    "condition_generation_seed",
    "instance_index",
    "replication_index",
    "instance_seed",
    "simulation_seed",
)


@dataclass(frozen=True)
class E1CaseManifest:
    path: Path
    sha256: str
    rows: tuple[dict, ...]
    cases_by_condition: dict[str, tuple[EvaluationCase, ...]]


def e1_condition_id(condition: ConditionConfig) -> str:
    load = f"{condition.target_load:.2f}".replace(".", "p")
    due = format(condition.due_date_factor, ".12g").replace(".", "p")
    return (
        f"rhoF-{load}_due-{due}_"
        f"jobs-{condition.total_jobs}_initial-{condition.initial_jobs}"
    )


def _expected_rows(
        conditions: Sequence[ConditionConfig],
        protocol: EvaluationProtocol) -> list[dict]:
    rows = []
    cases = tuple(protocol.cases())
    for condition_index, condition in enumerate(conditions):
        identifier = e1_condition_id(condition)
        for case in cases:
            rows.append({
                "schema_version": CASE_MANIFEST_SCHEMA_VERSION,
                "condition_index": condition_index,
                "condition_id": identifier,
                "nominal_finite_load": format(
                    condition.target_load, ".12g"),
                "due_date_factor": condition.due_date_factor,
                "total_jobs": condition.total_jobs,
                "initial_jobs": condition.initial_jobs,
                "initial_job_fraction": format(
                    condition.initial_jobs / condition.total_jobs, ".12g"),
                "degradation_multiplier": format(
                    condition.degradation_multiplier, ".12g"),
                "condition_generation_seed": condition.seed,
                "instance_index": case.instance_index,
                "replication_index": case.replication_index,
                "instance_seed": case.instance_seed,
                "simulation_seed": case.simulation_seed,
            })
    return rows


def _normalized(row: dict) -> dict[str, str]:
    return {field: str(row[field]) for field in CASE_MANIFEST_FIELDS}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_e1_case_manifest(
        path: str | Path,
        conditions: Iterable[ConditionConfig],
        protocol: EvaluationProtocol) -> E1CaseManifest:
    """Create once or strictly validate the shared cross-method case list."""

    target = Path(path).resolve()
    condition_list = tuple(conditions)
    expected = _expected_rows(condition_list, protocol)
    expected_normalized = [_normalized(row) for row in expected]
    if target.is_file():
        with target.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            observed = [dict(row) for row in reader]
        if observed != expected_normalized:
            raise RuntimeError(
                "The existing E1 case manifest does not match the requested "
                "conditions/seeds. Formal comparison methods must not use a "
                f"private case schedule: {target}")
    else:
        atomic_csv_save(expected, target, fieldnames=CASE_MANIFEST_FIELDS)
        observed = expected_normalized

    grouped: dict[str, list[EvaluationCase]] = {}
    for row in observed:
        grouped.setdefault(row["condition_id"], []).append(EvaluationCase(
            instance_index=int(row["instance_index"]),
            replication_index=int(row["replication_index"]),
            instance_seed=int(row["instance_seed"]),
            simulation_seed=int(row["simulation_seed"]),
        ))
    return E1CaseManifest(
        path=target,
        sha256=_file_sha256(target),
        rows=tuple(observed),
        cases_by_condition={key: tuple(value) for key, value in grouped.items()},
    )
