"Materialize and verify the 540 frozen structural instances for formal E1."

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from comparison_runners.common_e1 import (
    DEFAULT_E1_CASE_MANIFEST,
    e1_condition_id,
    ensure_e1_case_manifest,
)
from experiment.artifacts import atomic_csv_save, atomic_json_save
from experiment.config import (
    E1_INSTANCE_COUNT,
    E1_INSTANCE_SEED_START,
    E1_SIMULATION_REPLICATIONS,
    E1_SIMULATION_SEED_START,
    SCENARIOS,
    iter_full_factorial,
    serialize_config,
)
from experiment.evaluation import EvaluationProtocol
from experiment.instances import (
    FROZEN_JOB_RECORD_FIELDS,
    FROZEN_JOB_RECORD_SCHEMA_VERSION,
    INSTANCE_GENERATION_PROTOCOL,
    generate_job_records,
    instance_load,
    load_job_records_csv,
    save_job_records_csv,
    validate_job_records,
)
from experiment.problem import (
    CORRECTIVE_TIME_RANGES,
    DEGRADATION_RATE_RANGES,
    DEGRADATION_STATES,
    FAILURE_TYPE_PROPORTIONS,
    IPM_RECOVERY_RATES,
    IPM_TIMES,
    JOB_SIZE_INTERVAL,
    MACHINE_CAPACITIES,
    MINIMAL_REPAIR_TIMES,
    PM_TIMES,
    PRODUCTION_RATE_RANGE,
    TERMINAL_DEGRADATION_RATES,
    UNEXPECTED_FAILURE_RATE_RANGES,
)
from experiment.provenance import source_fingerprint


DATASET_SCHEMA_VERSION = 1
DATASET_ID = "e1_formal_structural_instances"
DEFAULT_OUTPUT_DIR = WORKSPACE / "datasets" / "e1_formal_instances"
INSTANCE_MANIFEST_FIELDS = (
    "schema_version",
    "dataset_id",
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
    "instance_seed",
    "job_count",
    "arrival_span",
    "dynamic_nominal_workload",
    "initial_nominal_workload",
    "total_nominal_workload",
    "realized_dynamic_load",
    "finite_instance_load",
    "job_file",
    "job_file_sha256",
    "instance_contract_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--case-manifest", type=Path, default=DEFAULT_E1_CASE_MANIFEST)
    parser.add_argument("--scenario", choices=SCENARIOS, default="base")
    return parser.parse_args()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text_save(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def relative_to_workspace(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE).as_posix()
    except ValueError:
        return str(path.resolve())


def generator_source_contract() -> dict[str, str]:
    paths = (
        WORKSPACE / "experiment" / "instances.py",
        WORKSPACE / "experiment" / "config.py",
        WORKSPACE / "experiment" / "problem.py",
        WORKSPACE / "comparison_runners" / "common_e1.py",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(WORKSPACE).as_posix(): file_sha256(path)
        for path in paths
    }


def physical_parameter_contract(scenario_name: str) -> dict:
    scenario = SCENARIOS[scenario_name]
    return {
        "scenario": serialize_config(scenario),
        "machine_capacities": list(MACHINE_CAPACITIES),
        "job_size_interval": list(JOB_SIZE_INTERVAL),
        "degradation_states": DEGRADATION_STATES,
        "degradation_rate_ranges": DEGRADATION_RATE_RANGES,
        "unexpected_failure_rate_ranges": UNEXPECTED_FAILURE_RATE_RANGES,
        "terminal_degradation_rates": TERMINAL_DEGRADATION_RATES,
        "failure_type_proportions": FAILURE_TYPE_PROPORTIONS,
        "ipm_recovery_rates": IPM_RECOVERY_RATES,
        "ipm_times": IPM_TIMES,
        "pm_times": PM_TIMES,
        "corrective_time_ranges": CORRECTIVE_TIME_RANGES,
        "minimal_repair_times": MINIMAL_REPAIR_TIMES,
        "production_rate_range": PRODUCTION_RATE_RANGE,
    }


def export_dataset(
        *, output_dir: Path, case_manifest_path: Path,
        scenario_name: str) -> dict:
    output_dir = output_dir.resolve()
    scenario = SCENARIOS[scenario_name]
    conditions = tuple(iter_full_factorial())
    protocol = EvaluationProtocol(
        instance_count=E1_INSTANCE_COUNT,
        replications=E1_SIMULATION_REPLICATIONS,
        instance_seed_start=E1_INSTANCE_SEED_START,
        simulation_seed_start=E1_SIMULATION_SEED_START,
    )
    case_manifest = ensure_e1_case_manifest(
        case_manifest_path, conditions, protocol)

    rows_by_instance: dict[tuple[str, int], list[dict]] = {}
    for row in case_manifest.rows:
        key = (row["condition_id"], int(row["instance_index"]))
        rows_by_instance.setdefault(key, []).append(row)
    expected_instances = len(conditions) * E1_INSTANCE_COUNT
    if len(rows_by_instance) != expected_instances:
        raise RuntimeError(
            f"Expected {expected_instances} structural instances, got "
            f"{len(rows_by_instance)}.")

    manifest_rows = []
    total_job_records = 0
    for condition_index, condition in enumerate(conditions):
        identifier = e1_condition_id(condition)
        for instance_index in range(E1_INSTANCE_COUNT):
            case_rows = rows_by_instance[(identifier, instance_index)]
            if len(case_rows) != E1_SIMULATION_REPLICATIONS:
                raise RuntimeError(
                    "Every structural instance must have exactly "
                    f"{E1_SIMULATION_REPLICATIONS} formal replications.")
            instance_seeds = {int(row["instance_seed"]) for row in case_rows}
            if len(instance_seeds) != 1:
                raise RuntimeError(
                    "Formal replications disagree on the structural seed.")
            instance_seed = instance_seeds.pop()
            records = generate_job_records(
                target_load=condition.target_load,
                initial_jobs=condition.initial_jobs,
                total_jobs=condition.total_jobs,
                due_date_factor=condition.due_date_factor,
                processing_times=scenario.job_processing_times,
                size_interval=JOB_SIZE_INTERVAL,
                machine_counts=scenario.machine_counts,
                machine_capacities=MACHINE_CAPACITIES,
                seed=instance_seed,
            )
            validate_job_records(
                records,
                initial_jobs=condition.initial_jobs,
                total_jobs=condition.total_jobs,
                due_date_factor=condition.due_date_factor,
                processing_times=scenario.job_processing_times,
                size_interval=JOB_SIZE_INTERVAL,
            )
            relative_job_path = (
                Path("jobs") / identifier / f"instance-{instance_index:03d}.csv")
            job_path = output_dir / relative_job_path
            save_job_records_csv(records, job_path)
            loaded = load_job_records_csv(job_path)
            validate_job_records(
                loaded,
                initial_jobs=condition.initial_jobs,
                total_jobs=condition.total_jobs,
                due_date_factor=condition.due_date_factor,
                processing_times=scenario.job_processing_times,
                size_interval=JOB_SIZE_INTERVAL,
            )
            if loaded != records:
                raise RuntimeError(
                    f"Frozen job round trip changed data: {job_path}")
            generated_load = instance_load(
                records,
                target_load=condition.target_load,
                initial_jobs=condition.initial_jobs,
                processing_times=scenario.job_processing_times,
                machine_counts=scenario.machine_counts,
                machine_capacities=MACHINE_CAPACITIES,
            )
            loaded_load = instance_load(
                loaded,
                target_load=condition.target_load,
                initial_jobs=condition.initial_jobs,
                processing_times=scenario.job_processing_times,
                machine_counts=scenario.machine_counts,
                machine_capacities=MACHINE_CAPACITIES,
            )
            if loaded_load != generated_load:
                raise RuntimeError(
                    f"Frozen job round trip changed load metrics: {job_path}")
            job_hash = file_sha256(job_path)
            instance_contract = {
                "instance_generation_protocol": INSTANCE_GENERATION_PROTOCOL,
                "scenario": serialize_config(scenario),
                "condition": serialize_config(condition),
                "condition_id": identifier,
                "instance_index": instance_index,
                "instance_seed": instance_seed,
                "job_file_sha256": job_hash,
            }
            manifest_rows.append({
                "schema_version": DATASET_SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "condition_index": condition_index,
                "condition_id": identifier,
                "nominal_finite_load": format(condition.target_load, ".17g"),
                "due_date_factor": format(condition.due_date_factor, ".17g"),
                "total_jobs": condition.total_jobs,
                "initial_jobs": condition.initial_jobs,
                "initial_job_fraction": format(
                    condition.initial_jobs / condition.total_jobs, ".17g"),
                "degradation_multiplier": format(
                    condition.degradation_multiplier, ".17g"),
                "condition_generation_seed": condition.seed,
                "instance_index": instance_index,
                "instance_seed": instance_seed,
                "job_count": len(records),
                "arrival_span": format(generated_load.arrival_span, ".17g"),
                "dynamic_nominal_workload": format(
                    generated_load.dynamic_nominal_workload, ".17g"),
                "initial_nominal_workload": format(
                    generated_load.initial_nominal_workload, ".17g"),
                "total_nominal_workload": format(
                    generated_load.total_nominal_workload, ".17g"),
                "realized_dynamic_load": format(
                    generated_load.realized_dynamic_load, ".17g"),
                "finite_instance_load": format(
                    generated_load.finite_instance_load, ".17g"),
                "job_file": relative_job_path.as_posix(),
                "job_file_sha256": job_hash,
                "instance_contract_sha256": canonical_sha256(instance_contract),
            })
            total_job_records += len(records)
        print(
            f"[E1-DATASET] condition {condition_index + 1:02d}/"
            f"{len(conditions)} | {identifier} | instances={E1_INSTANCE_COUNT}")

    instance_manifest_path = output_dir / "instance_manifest.csv"
    atomic_csv_save(
        manifest_rows,
        instance_manifest_path,
        fieldnames=INSTANCE_MANIFEST_FIELDS,
    )
    checksum_entries = [
        (row["job_file_sha256"], row["job_file"])
        for row in manifest_rows
    ]
    checksum_entries.append((
        file_sha256(instance_manifest_path),
        instance_manifest_path.relative_to(output_dir).as_posix(),
    ))
    checksum_text = "".join(
        f"{digest}  {relative}\n"
        for digest, relative in sorted(checksum_entries, key=lambda item: item[1])
    )
    checksums_path = output_dir / "checksums.sha256"
    atomic_text_save(checksum_text, checksums_path)

    generator_sources = generator_source_contract()
    physical_contract = physical_parameter_contract(scenario_name)
    dataset_contract = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "instance_generation_protocol": INSTANCE_GENERATION_PROTOCOL,
        "job_record_schema_version": FROZEN_JOB_RECORD_SCHEMA_VERSION,
        "job_record_fields": FROZEN_JOB_RECORD_FIELDS,
        "scenario": serialize_config(scenario),
        "condition_count": len(conditions),
        "instances_per_condition": E1_INSTANCE_COUNT,
        "simulation_replications_per_instance": E1_SIMULATION_REPLICATIONS,
        "structural_instance_count": len(manifest_rows),
        "job_record_count": total_job_records,
        "formal_case_manifest_sha256": case_manifest.sha256,
        "instance_manifest_sha256": file_sha256(instance_manifest_path),
        "checksums_sha256": file_sha256(checksums_path),
        "generator_source_contract": generator_sources,
        "generator_source_contract_sha256": canonical_sha256(generator_sources),
        "physical_parameter_contract": physical_contract,
        "physical_parameter_contract_sha256": canonical_sha256(
            physical_contract),
        "serialization": {
            "format": "UTF-8 CSV",
            "float_format": ".17g",
            "path_separator": "/",
        },
    }
    dataset_manifest = {
        **dataset_contract,
        "dataset_contract_sha256": canonical_sha256(dataset_contract),
        "created_local": datetime.now().astimezone().isoformat(),
        "workspace_source_fingerprint_sha256": source_fingerprint(WORKSPACE),
        "files": {
            "formal_case_manifest": relative_to_workspace(
                case_manifest.path),
            "instance_manifest": "instance_manifest.csv",
            "checksums": "checksums.sha256",
            "jobs_root": "jobs",
        },
        "verification": {
            "generated_vs_loaded_exact_record_equality": True,
            "generated_vs_loaded_exact_load_equality": True,
            "verified_structural_instances": len(manifest_rows),
            "verified_job_records": total_job_records,
        },
    }
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    atomic_json_save(dataset_manifest, dataset_manifest_path)
    print(json.dumps({
        "output_dir": str(output_dir),
        "dataset_contract_sha256": dataset_manifest[
            "dataset_contract_sha256"],
        "structural_instances": len(manifest_rows),
        "job_records": total_job_records,
        "formal_simulation_cases": len(case_manifest.rows),
        "verification": dataset_manifest["verification"],
    }, ensure_ascii=False, indent=2))
    return dataset_manifest


def main() -> None:
    args = parse_args()
    export_dataset(
        output_dir=args.output_dir,
        case_manifest_path=args.case_manifest,
        scenario_name=args.scenario,
    )


if __name__ == "__main__":
    main()
