"Generate one full-information offline MA plan per industrial instance."

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from comparison_runners.industrial_case.case_common import (
    DATASET_ROOT,
    MACHINE_CAPACITIES,
    WORKSPACE,
    atomic_csv,
    atomic_json,
    canonical_sha256,
    file_sha256,
    read_csv,
    validate_dataset,
)
from methods.baselines.meta.offline_memetic.static_plan import load_static_plan


CANONICAL_CPP = (
    WORKSPACE / "comparison_runners" / "offline_memetic_ma" / "cpp"
    / "offline_memetic_planner.cpp"
)
DEFAULT_OUTPUT = WORKSPACE / "runs" / "industrial_case_offline_memetic_joint_neighborhood_plans"
DEFAULT_BUILD = WORKSPACE / "build" / "industrial_case_offline_memetic_ma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--compiler")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--search-strategy", choices=("legacy", "joint-neighborhood"),
                        default="joint-neighborhood")
    parser.add_argument("--refinement-generations", type=int, default=200)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--local-search-individuals", type=int, default=1)
    parser.add_argument("--local-search-evaluations", type=int, default=500)
    parser.add_argument("--nonimproving-generations", type=int, default=200)
    parser.add_argument("--local-search-interval", type=int, default=20)
    parser.add_argument("--mutation-adaptation-generations", type=int, default=100)
    parser.add_argument("--mutation-probability", type=float, default=0.20)
    parser.add_argument("--formation-probability-ga", type=float, default=0.75)
    parser.add_argument("--formation-probability-ls", type=float, default=0.25)
    parser.add_argument("--local-search-deviation-percent", type=float, default=5.0)
    parser.add_argument("--arrival-window-factor", type=float, default=1.0)
    parser.add_argument("--planning-seed-base", type=int, default=61_000_000)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--instance-limit", type=int, default=0,
                        help="Diagnostic only; zero selects all 60 instances.")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def adapted_cpp_source() -> str:
    source = CANONICAL_CPP.read_text(encoding="utf-8")
    replacements = {
        "constexpr int kMachineCount = 5;": "constexpr int kMachineCount = 3;",
        "constexpr int kFamilyCount = 5;": "constexpr int kFamilyCount = 6;",
        "constexpr std::array<int, kMachineCount> kMachineClass{0, 0, 1, 1, 2};":
            "constexpr std::array<int, kMachineCount> kMachineClass{0, 1, 2};",
        "constexpr std::array<int, kMachineCount> kMachineCapacity{60, 60, 80, 80, 100};":
            "constexpr std::array<int, kMachineCount> kMachineCapacity{120, 160, 200};",
        "constexpr std::array<double, kFamilyCount> kProcessingTime{190, 158, 202, 128, 104};":
            "constexpr std::array<double, kFamilyCount> kProcessingTime{128, 190, 104, 293, 202, 279};",
        "job.size > 30": "job.size > 60",
        "* static_cast<double>(kMachineCapacity[machine] - size) / 100.0":
            "* static_cast<double>(kMachineCapacity[machine] - size) / 200.0",
        "current.machine = 4;": "current.machine = 2;",
        "for (int capacity : {60, 80, 100})":
            "for (int capacity : {120, 160, 200})",
        "batch_size(merged, instance_.jobs) <= 100":
            "batch_size(merged, instance_.jobs) <= 200",
        "size + next <= 100": "size + next <= 200",
        "Offline memetic planner for frozen E1 batch-scheduling instances":
            "Offline memetic planner for the aerospace heat-treatment industrial case",
    }
    for old, new in replacements.items():
        count = source.count(old)
        if count == 0:
            raise RuntimeError(f"Canonical C++ adaptation anchor disappeared: {old}")
        source = source.replace(old, new)
    return source


def resolve_compiler(requested: str | None) -> str:
    candidates = [requested] if requested else ["g++", "clang++"]
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return str(shutil.which(candidate))
    raise RuntimeError("No C++20 compiler found in PATH (g++ or clang++).")


def build_executable(build_dir: Path, compiler: str | None, rebuild: bool) -> tuple[Path, Path]:
    build_dir.mkdir(parents=True, exist_ok=True)
    generated = build_dir / "industrial_case_offline_memetic_planner.cpp"
    expected = adapted_cpp_source()
    if not generated.is_file() or generated.read_text(encoding="utf-8") != expected:
        generated.write_text(expected, encoding="utf-8")
        rebuild = True
    executable = build_dir / ("industrial_case_offline_memetic_planner.exe" if os.name == "nt" else "industrial_case_offline_memetic_planner")
    if rebuild or not executable.is_file() or executable.stat().st_mtime_ns < generated.stat().st_mtime_ns:
        command = [resolve_compiler(compiler), "-std=c++20", "-O3", "-DNDEBUG",
                   "-Wall", "-Wextra", "-pedantic", str(generated), "-o", str(executable), "-pthread"]
        print("[INDUSTRIAL-MA] compiling adapted C++20 optimizer", flush=True)
        subprocess.run(command, cwd=WORKSPACE, check=True)
    return executable, generated


def write_input_manifest(dataset: dict, output: Path) -> Path:
    rows = []
    load_indices = {0.9: 0, 1.0: 1, 1.1: 2}
    for row in dataset["rows"]:
        load = float(row["target_load"])
        rows.append({
            "condition_index": load_indices[load],
            "condition_id": f"industrial_load_{load:.2f}".replace(".", "p"),
            "instance_index": int(row["instance_number"]),
            "instance_seed": int(row["arrival_seed"]),
            "degradation_multiplier": 1.0,
            "job_file": row["job_file"],
            "job_file_sha256": row["job_file_sha256"],
        })
    target = output / "ma_input_manifest.csv"
    if target.is_file() and read_csv(target) != [
        {key: str(value) for key, value in item.items()} for item in rows
    ]:
        raise RuntimeError("Output directory contains another MA input manifest.")
    if not target.is_file():
        atomic_csv(target, rows)
    return target


def validate_plans(dataset: dict, output: Path, optimizer_sha: str, selected: int) -> dict:
    manifest = read_csv(output / "plan_manifest.csv")
    expected_rows = dataset["rows"][:selected] if selected else dataset["rows"]
    expected = {
        (f"industrial_load_{float(row['target_load']):.2f}".replace(".", "p"),
         int(row["instance_number"])): row for row in expected_rows
    }
    verified_jobs = 0
    verified_operations = 0
    convergence_records = []
    for summary in manifest:
        key = (summary["condition_id"], int(summary["instance_index"]))
        if key not in expected:
            raise ValueError(f"Unexpected static plan: {key}")
        source = expected[key]
        jobs = {int(row["job_id"]): row for row in read_csv(dataset["root"] / source["job_file"])}
        plan_path = output / summary["plan_file"]
        plan = load_static_plan(
            plan_path,
            expected_dataset_contract_sha256=dataset["dataset_contract_sha256"],
            expected_optimizer_contract_sha256=optimizer_sha,
        )
        covered = []
        for operation in plan.operations:
            if not 0 <= operation.machine_id < 3:
                raise ValueError(f"Invalid machine in {plan_path}")
            if operation.machine_capacity != MACHINE_CAPACITIES[operation.machine_id]:
                raise ValueError(f"Machine-capacity mismatch in {plan_path}")
            if operation.action == "PROCESS":
                if operation.batch_size > operation.machine_capacity:
                    raise ValueError(f"Capacity violation in {plan_path}")
                if {int(jobs[j]["family"]) for j in operation.job_ids} != {operation.family}:
                    raise ValueError(f"Family violation in {plan_path}")
                covered.extend(operation.job_ids)
        if sorted(covered) != sorted(jobs):
            raise ValueError(f"Job coverage violation in {plan_path}")
        convergence_path = output / summary["convergence_file"]
        convergence = read_csv(convergence_path)
        if len(convergence) != int(summary["generations"]) + 1:
            raise ValueError(f"Incomplete convergence log: {convergence_path}")
        generations = [int(row["generation"]) for row in convergence]
        incumbents = [float(row["incumbent_total_tardiness"]) for row in convergence]
        if generations != list(range(len(convergence))):
            raise ValueError(f"Non-contiguous convergence log: {convergence_path}")
        if any(later > earlier + 1e-8 for earlier, later in zip(incumbents, incumbents[1:])):
            raise ValueError(f"Non-monotone incumbent in {convergence_path}")
        verified_jobs += len(covered)
        verified_operations += len(plan.operations)
        convergence_records.append({
            "condition_id": key[0], "instance_index": key[1],
            "convergence_file": summary["convergence_file"],
            "rows": len(convergence), "sha256": file_sha256(convergence_path),
            "stop_reason": summary["stop_reason"],
        })
    atomic_csv(output / "convergence_manifest.csv", convergence_records)
    result = {
        "schema_version": 1,
        "dataset_contract_sha256": dataset["dataset_contract_sha256"],
        "optimizer_contract_sha256": optimizer_sha,
        "verified_structural_instances": len(manifest),
        "verified_job_assignments": verified_jobs,
        "verified_static_operations": verified_operations,
        "checks": {
            "one_plan_per_selected_instance": True,
            "every_job_processed_exactly_once": True,
            "incompatible_families_never_mixed": True,
            "machine_capacities_respected": True,
            "convergence_logs_complete": True,
        },
    }
    atomic_json(output / "plan_validation.json", result)
    return result


def main() -> None:
    args = parse_args()
    if args.instance_limit < 0 or args.instance_limit > 60:
        raise ValueError("--instance-limit must be between 0 and 60.")
    dataset = validate_dataset(args.dataset_dir)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_manifest = write_input_manifest(dataset, output)
    executable, generated = build_executable(args.build_dir.resolve(), args.compiler, args.rebuild)
    parameters = {
        "search_strategy": args.search_strategy,
        "refinement_generations": args.refinement_generations,
        "population": args.population, "generations": args.generations,
        "local_search_individuals": args.local_search_individuals,
        "local_search_evaluations": args.local_search_evaluations,
        "nonimproving_generations": args.nonimproving_generations,
        "local_search_interval": args.local_search_interval,
        "mutation_adaptation_generations": args.mutation_adaptation_generations,
        "mutation_probability": args.mutation_probability,
        "formation_probability_ga": args.formation_probability_ga,
        "formation_probability_ls": args.formation_probability_ls,
        "local_search_deviation_percent": args.local_search_deviation_percent,
        "arrival_window_factor": args.arrival_window_factor,
        "planning_seed_base": args.planning_seed_base,
    }
    contract = {
        "schema_version": 1,
        "method": "full-information offline memetic algorithm with immutable static plans",
        "case": "aerospace heat-treatment industrial-order dynamic case",
        "dataset_contract_sha256": dataset["dataset_contract_sha256"],
        "canonical_cpp_sha256": file_sha256(CANONICAL_CPP),
        "adapted_cpp_sha256": file_sha256(generated),
        "machine_counts": [1, 1, 1], "machine_capacities": [120, 160, 200],
        "processing_times": [128, 190, 104, 293, 202, 279],
        "parameters": parameters,
        "instance_limit": args.instance_limit,
        "execution_semantics": "single zero-time optimization without rolling re-optimization",
    }
    optimizer_sha = canonical_sha256(contract)
    artifact = {**contract, "optimizer_contract_sha256": optimizer_sha}
    contract_path = output / "optimizer_contract.json"
    if not contract_path.is_file():
        atomic_json(contract_path, artifact)
    command = [
        str(executable), "--dataset-dir", str(dataset["root"]),
        "--manifest", str(input_manifest), "--output-dir", str(output),
        "--dataset-contract-sha256", dataset["dataset_contract_sha256"],
        "--optimizer-contract-sha256", optimizer_sha,
    ]
    for key, value in parameters.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    command.extend(["--threads", str(args.threads), "--instance-limit", str(args.instance_limit)])
    if args.no_resume:
        command.append("--no-resume")
    print(f"[INDUSTRIAL-MA] instances={60 if not args.instance_limit else args.instance_limit} output={output}", flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)
    validation = validate_plans(dataset, output, optimizer_sha, args.instance_limit)
    print(f"[INDUSTRIAL-MA] validation passed: {validation['verified_structural_instances']} plans", flush=True)


if __name__ == "__main__":
    main()
