"Build and execute the C++ full-information offline memetic planner."

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "comparison_runners" / "offline_memetic_ma" / "cpp"
    / "offline_memetic_planner.cpp"
)
DEFAULT_DATASET = ROOT / "datasets" / "e1_formal_instances"
DEFAULT_OUTPUT = (
    ROOT / "runs"
    / "e1_offline_memetic_joint_neighborhood_plans"
)
DEFAULT_BUILD = ROOT / "build" / "offline_memetic_ma"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_dataset(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "dataset_manifest.json"
    checksum_path = dataset_dir / "checksums.sha256"
    instance_manifest = dataset_dir / "instance_manifest.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata


def resolve_compiler(requested: str | None) -> tuple[str, str]:
    candidates = [requested] if requested else ["g++", "clang++"]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            executable = shutil.which(candidate)
            assert executable is not None
            family = "clang" if "clang" in Path(executable).name.lower() else "gnu"
            return executable, family
    raise RuntimeError(
        "No supported C++ compiler was found in PATH. Install MinGW-w64 g++ "
        "or LLVM clang++, or pass --compiler with its executable path."
    )


def build_executable(
        *, compiler: str | None, build_dir: Path, rebuild: bool) -> Path:
    compiler_path, family = resolve_compiler(compiler)
    build_dir.mkdir(parents=True, exist_ok=True)
    executable = build_dir / (
        "offline_memetic_planner.exe" if os.name == "nt"
        else "offline_memetic_planner"
    )
    if (
            not rebuild
            and executable.is_file()
            and executable.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns):
        print(f"[offline-ma-launcher] using compiled executable: {executable}")
        return executable
    command = [
        compiler_path,
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Wall",
        "-Wextra",
        "-pedantic",
        str(SOURCE),
        "-o",
        str(executable),
    ]
    if family in {"gnu", "clang"}:
        command.append("-pthread")
    print("[offline-ma-launcher] compiling C++ optimizer")
    subprocess.run(command, cwd=ROOT, check=True)
    return executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one zero-time, full-information static memetic plan for "
            "each frozen E1 structural instance."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--compiler")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--search-strategy", choices=("legacy", "joint-neighborhood"),
                        default="joint-neighborhood",
                        help="legacy is retained for paired search diagnostics only.")
    parser.add_argument("--refinement-generations", type=int, default=200,
                        help="Additional refinement generations after the original MA search.")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--local-search-individuals", type=int, default=1)
    parser.add_argument("--local-search-evaluations", type=int, default=500)
    parser.add_argument("--nonimproving-generations", type=int, default=200)
    parser.add_argument("--local-search-interval", type=int, default=20)
    parser.add_argument("--mutation-adaptation-generations", type=int, default=100)
    parser.add_argument("--mutation-probability", type=float, default=0.20)
    parser.add_argument("--formation-probability-ga", type=float, default=0.75)
    parser.add_argument("--formation-probability-ls", type=float, default=0.25)
    parser.add_argument(
        "--local-search-deviation-percent", type=float, default=5.0
    )
    parser.add_argument("--arrival-window-factor", type=float, default=1.0)
    parser.add_argument("--planning-seed-base", type=int, default=40_000_000)
    parser.add_argument(
        "--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=0,
        help="Diagnostic only. Zero selects all 540 formal instances.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def write_contract(
        *, args: argparse.Namespace, dataset_metadata: dict,
        output_dir: Path) -> tuple[dict, str]:
    if args.instance_limit < 0:
        raise ValueError("--instance-limit cannot be negative.")
    if (
            args.population < 4
            or args.population % 2
            or args.generations <= 0
            or args.refinement_generations < 0
            or args.local_search_individuals < 0
            or args.local_search_evaluations < 0
            or args.nonimproving_generations <= 0
            or args.local_search_interval <= 0
            or args.mutation_adaptation_generations <= 0
            or not 0.0 <= args.mutation_probability <= 1.0
            or not 0.0 <= args.formation_probability_ga <= 1.0
            or not 0.0 <= args.formation_probability_ls <= 1.0
            or args.local_search_deviation_percent < 0.0
            or args.arrival_window_factor <= 0.0
            or args.threads <= 0):
        raise ValueError("Invalid offline memetic search parameter.")
    mode = "formal" if args.instance_limit == 0 else "diagnostic"
    contract = {
        "schema_version": 3,
        "method": (
            "full-information joint batch-formation, sequencing, machine-"
            "assignment, and maintenance memetic algorithm"
        ),
        "execution_semantics": (
            "single zero-time offline optimization; immutable static plan; "
            "no rolling re-optimization"
        ),
        "source_reference": (
            "Chiang et al., A memetic algorithm for minimizing total weighted "
            "tardiness on parallel batch machines with incompatible job "
            "families and dynamic job arrival, Computers & Operations "
            "Research 37 (2010) 2257-2269"
        ),
        "adaptations": [
            "arbitrary job sizes and heterogeneous machine capacities",
            "explicit machine-assignment gene",
            "explicit pre-batch no-maintenance/IPM/PM gene",
            "state-dependent expected degradation, production loss, failure "
            "downtime, and maintenance restoration surrogate",
            "diversified due-date, arrival-time, slack, capacity, and machine-"
            "assignment initializations",
            "arrival-window-aware batching and load-balancing machine "
            "neighborhoods",
            "periodic local search with stagnation-adaptive mutation",
            "optional batch-preserving crossover, joint variable-neighborhood "
            "descent, phenotype duplicate control and stagnation immigration",
            "exact memoization of the unchanged processing surrogate",
        ],
        "mode": mode,
        "dataset_contract_sha256": dataset_metadata["dataset_contract_sha256"],
        "dataset_structural_instances": dataset_metadata[
            "structural_instance_count"
        ],
        "instance_limit": args.instance_limit,
        "optimizer_source_sha256": sha256_file(SOURCE),
        "launcher_source_sha256": sha256_file(Path(__file__).resolve()),
        "parameters": {
            "search_strategy": args.search_strategy,
            "refinement_generations": args.refinement_generations,
            "population": args.population,
            "generations": args.generations,
            "local_search_individuals": args.local_search_individuals,
            "local_search_evaluations": args.local_search_evaluations,
            "nonimproving_generations": args.nonimproving_generations,
            "local_search_interval": args.local_search_interval,
            "mutation_adaptation_generations": (
                args.mutation_adaptation_generations
            ),
            "mutation_probability": args.mutation_probability,
            "formation_probability_ga": args.formation_probability_ga,
            "formation_probability_ls": args.formation_probability_ls,
            "local_search_deviation_percent": (
                args.local_search_deviation_percent
            ),
            "arrival_window_factor": args.arrival_window_factor,
            "planning_seed_base": args.planning_seed_base,
        },
        "parallelism": {
            "instance-level_only": True,
            "deterministic_per_instance_independent_of_thread_count": True,
            "thread_count_excluded_from_optimizer_contract": True,
        },
        "plan_schema": {
            "format": "UTF-8 CSV",
            "one_plan_per_structural_instance": True,
            "job_ids_separator": ";",
            "actions": ["PROCESS", "IPM", "PM"],
            "execution_order": (
                "per-machine operations ordered by planned_start, then "
                "operation_index; timestamps are surrogate diagnostics and "
                "do not authorize online re-optimization"
            ),
        },
        "convergence_log_schema": {
            "format": "UTF-8 CSV",
            "one_log_per_structural_instance": True,
            "generation_zero_included": True,
            "population_dispersion_and_unique_chromosomes_recorded": True,
            "local_search_activity_recorded": True,
            "explicit_stop_reason_recorded": True,
        },
    }
    contract_sha = canonical_sha256(contract)
    artifact = {**contract, "optimizer_contract_sha256": contract_sha}
    target = output_dir / "optimizer_contract.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return artifact, contract_sha


def validate_plans(
        *, dataset_dir: Path, output_dir: Path, dataset_metadata: dict,
        optimizer_contract_sha256: str, instance_limit: int) -> dict:
    with (dataset_dir / "instance_manifest.csv").open(
            newline="", encoding="utf-8") as handle:
        instance_rows = list(csv.DictReader(handle))
    if instance_limit:
        instance_rows = instance_rows[:instance_limit]
    expected = {
        (row["condition_id"], int(row["instance_index"])): row
        for row in instance_rows
    }
    with (output_dir / "plan_manifest.csv").open(
            newline="", encoding="utf-8") as handle:
        plan_manifest = list(csv.DictReader(handle))
    if len(plan_manifest) != len(expected):
        raise ValueError(
            "Static plan manifest does not contain the expected number of "
            f"instances: {len(plan_manifest)} != {len(expected)}"
        )
    machine_capacities = [60, 60, 80, 80, 100]
    verified_jobs = 0
    verified_operations = 0
    verified_convergence_rows = 0
    convergence_records = []
    for summary in plan_manifest:
        key = (summary["condition_id"], int(summary["instance_index"]))
        if key not in expected:
            raise ValueError(f"Unexpected plan in manifest: {key}")
        instance = expected[key]
        with (dataset_dir / instance["job_file"]).open(
                newline="", encoding="utf-8") as handle:
            jobs = {
                int(row["job_id"]): row for row in csv.DictReader(handle)
            }
        plan_path = output_dir / Path(summary["plan_file"])
        convergence_path = output_dir / Path(summary["convergence_file"])
        if not convergence_path.is_file():
            raise ValueError(
                f"Static plan convergence log is missing: {convergence_path}"
            )
        with convergence_path.open(newline="", encoding="utf-8") as handle:
            convergence = list(csv.DictReader(handle))
        expected_generations = int(summary["generations"])
        generations = [int(row["generation"]) for row in convergence]
        incumbents = [
            float(row["incumbent_total_tardiness"]) for row in convergence
        ]
        if generations != list(range(expected_generations + 1)):
            raise ValueError(
                f"Convergence generations are not contiguous: {convergence_path}"
            )
        if any(
                later > earlier + 1e-8
                for earlier, later in zip(incumbents, incumbents[1:])):
            raise ValueError(
                f"Convergence incumbent is not monotone: {convergence_path}"
            )
        if not math.isclose(
                incumbents[-1], float(summary["objective_total_tardiness"]),
                rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(
                f"Convergence and plan objectives differ: {convergence_path}"
            )
        with plan_path.open(newline="", encoding="utf-8") as handle:
            operations = list(csv.DictReader(handle))
        if not operations:
            raise ValueError(f"Static plan has no operations: {plan_path}")
        covered: list[int] = []
        previous_end = {machine: 0.0 for machine in range(5)}
        for operation in sorted(
                operations,
                key=lambda row: (
                    int(row["machine_id"]), float(row["planned_start"]),
                    int(row["operation_index"]),
                )):
            machine = int(operation["machine_id"])
            start = float(operation["planned_start"])
            end = float(operation["planned_end"])
            if not 0 <= machine < 5 or start + 1e-8 < previous_end[machine] \
                    or end <= start:
                raise ValueError(f"Invalid or overlapping operation: {plan_path}")
            previous_end[machine] = end
            if operation["action"] == "PROCESS":
                job_ids = [
                    int(value) for value in operation["job_ids"].split(";")
                    if value
                ]
                if not job_ids:
                    raise ValueError(f"Empty production batch: {plan_path}")
                families = {int(jobs[job_id]["family"]) for job_id in job_ids}
                total_size = sum(int(jobs[job_id]["size"]) for job_id in job_ids)
                if len(families) != 1 \
                        or families != {int(operation["family"])}:
                    raise ValueError(f"Incompatible-family batch: {plan_path}")
                if total_size != int(operation["batch_size"]) \
                        or total_size > machine_capacities[machine]:
                    raise ValueError(f"Invalid batch capacity: {plan_path}")
                covered.extend(job_ids)
            elif operation["action"] not in {"IPM", "PM"}:
                raise ValueError(f"Unknown static-plan action: {plan_path}")
        verified_jobs += len(covered)
        verified_operations += len(operations)
        verified_convergence_rows += len(convergence)
        convergence_records.append({
            "schema_version": 1,
            "condition_id": summary["condition_id"],
            "instance_index": int(summary["instance_index"]),
            "convergence_file": Path(summary["convergence_file"]).as_posix(),
            "rows": len(convergence),
            "sha256": sha256_file(convergence_path),
            "final_objective_total_tardiness": incumbents[-1],
            "stop_reason": summary["stop_reason"],
        })
    convergence_manifest = output_dir / "convergence_manifest.csv"
    convergence_temporary = convergence_manifest.with_suffix(".csv.tmp")
    with convergence_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(convergence_records[0]))
        writer.writeheader()
        writer.writerows(convergence_records)
    convergence_temporary.replace(convergence_manifest)
    validation = {
        "schema_version": 1,
        "optimizer_contract_sha256": optimizer_contract_sha256,
        "dataset_contract_sha256": dataset_metadata[
            "dataset_contract_sha256"
        ],
        "verified_structural_instances": len(plan_manifest),
        "verified_job_assignments": verified_jobs,
        "verified_static_operations": verified_operations,
        "verified_convergence_rows": verified_convergence_rows,
        "convergence_manifest_sha256": sha256_file(convergence_manifest),
        "checks": {
            "one_plan_per_selected_instance": True,
            "every_job_processed_exactly_once": True,
            "incompatible_families_never_mixed": True,
            "machine_capacities_respected": True,
            "per_machine_operations_nonoverlapping": True,
            "dataset_and_optimizer_contracts_match": True,
            "convergence_logs_complete_and_monotone": True,
        },
    }
    target = output_dir / "plan_validation.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return validation


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    build_dir = args.build_dir.resolve()
    metadata = validate_dataset(dataset_dir)
    _, contract_sha = write_contract(
        args=args, dataset_metadata=metadata, output_dir=output_dir
    )
    executable = build_executable(
        compiler=args.compiler,
        build_dir=build_dir,
        rebuild=args.rebuild,
    )
    command = [
        str(executable),
        "--dataset-dir", str(dataset_dir),
        "--manifest", str(dataset_dir / "instance_manifest.csv"),
        "--output-dir", str(output_dir),
        "--dataset-contract-sha256", metadata["dataset_contract_sha256"],
        "--optimizer-contract-sha256", contract_sha,
        "--population", str(args.population),
        "--search-strategy", args.search_strategy,
        "--refinement-generations", str(args.refinement_generations),
        "--generations", str(args.generations),
        "--local-search-individuals", str(args.local_search_individuals),
        "--local-search-evaluations", str(args.local_search_evaluations),
        "--nonimproving-generations", str(args.nonimproving_generations),
        "--local-search-interval", str(args.local_search_interval),
        "--mutation-adaptation-generations",
        str(args.mutation_adaptation_generations),
        "--mutation-probability", str(args.mutation_probability),
        "--formation-probability-ga", str(args.formation_probability_ga),
        "--formation-probability-ls", str(args.formation_probability_ls),
        "--local-search-deviation-percent",
        str(args.local_search_deviation_percent),
        "--arrival-window-factor", str(args.arrival_window_factor),
        "--planning-seed-base", str(args.planning_seed_base),
        "--threads", str(args.threads),
        "--instance-limit", str(args.instance_limit),
    ]
    if args.no_resume:
        command.append("--no-resume")
    print(
        "[offline-ma-launcher] dataset contract: "
        f"{metadata['dataset_contract_sha256']}"
    )
    print(f"[offline-ma-launcher] optimizer contract: {contract_sha}")
    print(f"[offline-ma-launcher] output: {output_dir}")
    sys.stdout.flush()
    subprocess.run(command, cwd=ROOT, check=True)
    validation = validate_plans(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        dataset_metadata=metadata,
        optimizer_contract_sha256=contract_sha,
        instance_limit=args.instance_limit,
    )
    print(
        "[offline-ma-launcher] validation passed: "
        f"instances={validation['verified_structural_instances']} "
        f"jobs={validation['verified_job_assignments']} "
        f"operations={validation['verified_static_operations']}"
    )


if __name__ == "__main__":
    main()
