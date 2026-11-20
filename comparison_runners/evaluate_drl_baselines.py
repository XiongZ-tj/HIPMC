"""Evaluate learned E1 methods on the shared frozen formal test cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import product
from dataclasses import asdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiment.revision_protocol import LEARNED_METHODS as REVISION_METHODS, TEST_INSTANCES_PER_CONDITION
from experiment.revision_scenarios import REVISION_SCENES, apply_revision_scene
from experiment.revision_selection import selected_conditions
from comparison_runners.training_budgets import budget_model_dir, configure_learning

from comparison_runners.common_e1 import (
    DEFAULT_E1_CASE_MANIFEST,
    e1_condition_id,
    ensure_e1_case_manifest,
)
from comparison_runners.detailed_metrics import (
    consolidate_csv,
    episode_row,
    job_rows,
    machine_rows,
    state_action_rows,
)
from comparison_runners.drl_common import (
    LEARNED_METHODS,
    build_drl_environment,
    file_sha256,
    load_model,
    simulation_core,
)
from comparison_runners.frozen_e1 import (
    DEFAULT_FROZEN_E1_DATASET,
    FrozenE1Dataset,
)
from experiment.artifacts import atomic_csv_save, atomic_json_save
from experiment.config import (
    E1_INSTANCE_COUNT,
    E1_INSTANCE_SEED_START,
    E1_SIMULATION_REPLICATIONS,
    E1_SIMULATION_SEED_START,
    E1_STOCHASTIC_METHOD_SEEDS,
    iter_full_factorial,
)
from experiment.evaluation import EvaluationProtocol
from experiment.provenance import runtime_provenance, source_fingerprint


SCHEMA_VERSION = 2


def canonical_hash(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=LEARNED_METHODS,
        default=list(REVISION_METHODS))
    parser.add_argument(
        "--training-seeds", nargs="+", type=int,
        default=list(E1_STOCHASTIC_METHOD_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument('--scenes',nargs='+',choices=tuple(REVISION_SCENES),default=list(REVISION_SCENES))
    parser.add_argument('--model-budget',type=int,default=500,
                        help='Read the best-validation snapshot for this training budget.')
    parser.add_argument('--model-layout',choices=('standard','grid'),default='standard')
    parser.add_argument('--hyperparameter-file',type=Path)
    parser.add_argument(
        "--case-manifest", type=Path,
        default=DEFAULT_E1_CASE_MANIFEST)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_FROZEN_E1_DATASET,
        help=(
            "Frozen 540-instance E1 dataset. Formal learned-method testing "
            "always loads these CSV records instead of regenerating jobs."),
    )
    parser.add_argument(
        "--model-root", type=Path,
        default=WORKSPACE / "runs" / "e1_drl_training")
    parser.add_argument(
        "--output-dir", type=Path,
        default=WORKSPACE / "runs" / "e1_drl_evaluation")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-job-records", action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--instances", type=int, default=TEST_INSTANCES_PER_CONDITION,
                        help="First N instances per condition from the unchanged frozen manifest.")
    parser.add_argument("--condition-limit", type=int)
    parser.add_argument(
        "--case-limit", type=int,
        help="Diagnostic-only prefix of the 200 frozen cases per condition.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Allow a strict subset of methods, seeds, or conditions.")
    return parser.parse_args()


def resolve(path):
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def block_paths(output, stem, save_jobs):
    result = {
        "run_metrics": output / "partitions" / "run_metrics" / f"{stem}.csv",
        "machine_metrics": output / "partitions" / "machine_metrics" / f"{stem}.csv",
        "state_action_metrics": output / "partitions" / "state_action_metrics" / f"{stem}.csv",
        "manifest": output / "blocks" / f"{stem}.json",
    }
    if save_jobs:
        result["job_records"] = (
            output / "partitions" / "job_records" / f"{stem}.csv")
    return result


def evaluate_block(
        *, output, method, training_seed, condition, cases,
        case_manifest_sha, suite_configuration_sha, source_sha,
        model_dir, device, save_job_records, log_every,
        block_number, total_blocks, frozen_dataset, scene_id='S0', model_budget=500):
    identifier = e1_condition_id(condition)
    method_id = f"DRL-{method.upper()}-seed{training_seed}"
    stem = f"{scene_id}__budget-{model_budget}__{method_id}__{identifier}"
    if len(cases) != E1_INSTANCE_COUNT * E1_SIMULATION_REPLICATIONS:
        subset_key = canonical_hash([asdict(case) for case in cases])[:12]
        stem += f"__subset-{subset_key}"
    paths = block_paths(output, stem, save_job_records)
    if paths["manifest"].is_file():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        for info in manifest["files"].values():
            path = output / info["relative_path"]
        return {**manifest, "skipped": True}
    # A block without its completion manifest is recomputed; its partial CSVs
    # are atomically replaced at completion rather than blocking a restart.

    training_config_path = model_dir / "run_config.json"
    training_config = (
        json.loads(training_config_path.read_text(encoding="utf-8"))
        if training_config_path.is_file() else {})
    reward_contract = training_config.get("reward_normalization", {})
    reward_scale = reward_contract.get("value")
    reward_scale_sha = reward_contract.get("artifact_sha256")
    if reward_contract.get("mode") == "diagnostic_unit_scale":
        reward_scale = None
        reward_scale_sha = None
    environment, resolved_device = build_drl_environment(
        method,
        condition,
        device=device,
        calibrated_reward_scale=reward_scale,
        reward_scale_artifact_sha256=reward_scale_sha,
    )
    artifact_hashes = load_model(environment, method, model_dir)
    hp=training_config.get('hyperparameters',{})
    configure_learning(environment,{'learning_rate':hp.get('learning_rate'),
        'clip_ratio':hp.get('clip_ratio'),'gamma':hp.get('gamma_baa'),
        'reward_formulation':hp.get('reward_formulation','potential_difference')})
    core = simulation_core(environment)
    reference_scene_config=core.config
    model_sha = canonical_hash(artifact_hashes)
    training_config_sha = (
        file_sha256(training_config_path)
        if training_config_path.is_file() else "unavailable")
    rows = []
    machines = []
    state_actions = []
    jobs = []
    failures = 0
    started = time.perf_counter()
    expected = len(cases)
    for run_index, case in enumerate(cases, start=1):
        frozen_row = frozen_dataset.row(
            identifier,
            case.instance_index,
            expected_instance_seed=case.instance_seed,
        )
        identity = {
            "schema_version": SCHEMA_VERSION,
            "method_family": "online_drl",
            "method_id": method_id,
            "method": method,
            "scene_id": scene_id,
            "scene_parameters_json": json.dumps(REVISION_SCENES[scene_id].parameters(reference_scene_config),sort_keys=True),
            "model_budget": model_budget,
            "effective_nominal_load": REVISION_SCENES[scene_id].effective_nominal_load(condition.target_load),
            "training_seed": training_seed,
            "condition_id": identifier,
            "nominal_finite_load": condition.target_load,
            "due_date_factor": condition.due_date_factor,
            "total_jobs_configured": condition.total_jobs,
            "initial_jobs_configured": condition.initial_jobs,
            "initial_job_fraction": (
                condition.initial_jobs / condition.total_jobs),
            "degradation_multiplier": condition.degradation_multiplier,
            "instance_index": case.instance_index,
            "replication_index": case.replication_index,
            "instance_seed": case.instance_seed,
            "simulation_seed": case.simulation_seed,
            "policy_seed": case.simulation_seed,
            "case_manifest_sha256": case_manifest_sha,
            "frozen_dataset_contract_sha256": (
                frozen_dataset.contract_sha256),
            "frozen_instance_manifest_sha256": (
                frozen_dataset.instance_manifest_sha256),
            "frozen_job_file_sha256": frozen_row["job_file_sha256"],
            "model_artifacts_sha256": model_sha,
            "training_run_config_sha256": training_config_sha,
            "suite_configuration_sha256": suite_configuration_sha,
            "source_fingerprint_sha256": source_sha,
            "device": resolved_device,
        }
        metrics = None
        error = None
        run_started = time.perf_counter()
        try:
            apply_revision_scene(core,scene_id,
                instance_seed=case.instance_seed,
                simulation_seed=case.simulation_seed,
                policy_seed=case.simulation_seed,
                job_records=frozen_dataset.records(
                    identifier,
                    case.instance_index,
                    expected_instance_seed=case.instance_seed,
                ),
            )
            metrics = core.run_episode(training=False)
        except Exception as exc:  # retain failures as formal observations
            error = exc
            failures += 1
        wall = time.perf_counter() - run_started
        row = episode_row(
            identity, metrics, core,
            wall_time=wall, error=error)
        rows.append(row)
        if metrics is not None:
            machines.extend(machine_rows(
                identity, core, metrics["makespan"]))
            state_actions.extend(state_action_rows(identity, core))
            if save_job_records:
                jobs.extend(job_rows(identity, core))
        if (
                error is not None or run_index == 1 or run_index == expected
                or (log_every and run_index % log_every == 0)):
            print(
                f"[DRL-EVAL] block {block_number}/{total_blocks} "
                f"run {run_index}/{expected} method={method.upper()} "
                f"seed={training_seed} scene={scene_id} condition={identifier} "
                f"instance_seed={case.instance_seed} "
                f"simulation_seed={case.simulation_seed} "
                f"status={row['status']} tardiness={row['total_tardiness']} "
                f"elapsed={wall:.3f}s"
                + (f" error={row['error_type']}: {row['error_message']}"
                   if error is not None else ""),
                flush=True,
            )
    tables = {
        "run_metrics": rows,
        "machine_metrics": machines,
        "state_action_metrics": state_actions,
    }
    if save_job_records:
        tables["job_records"] = jobs
    for name, table in tables.items():
        if not table:
            raise RuntimeError(
                f"Detailed table {name} is empty for block {stem}.")
        atomic_csv_save(table, paths[name], fieldnames=tuple(table[0]))
    files = {}
    for name, table in tables.items():
        path = paths[name]
        files[name] = {
            "relative_path": path.relative_to(output).as_posix(),
            "rows": len(table),
            "sha256": file_sha256(path),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "block_id": stem,
        "scene_id": scene_id,
        "method": method,
        "training_seed": training_seed,
        "condition_id": identifier,
        "suite_configuration_sha256": suite_configuration_sha,
        "case_manifest_sha256": case_manifest_sha,
        "frozen_dataset_contract_sha256": frozen_dataset.contract_sha256,
        "frozen_instance_manifest_sha256": (
            frozen_dataset.instance_manifest_sha256),
        "model_artifacts": artifact_hashes,
        "model_artifacts_sha256": model_sha,
        "source_fingerprint_sha256": source_sha,
        "device": resolved_device,
        "expected_runs": expected,
        "completed_runs": len(rows),
        "failed_runs": failures,
        "excluded_runs": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "files": files,
    }
    atomic_json_save(manifest, paths["manifest"])
    return {**manifest, "skipped": False}


def main():
    args = parse_args()
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative.")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods contains duplicates.")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("--training-seeds contains duplicates.")
    diagnostic_subset = (
        tuple(args.training_seeds) != E1_STOCHASTIC_METHOD_SEEDS
        or args.condition_limit is not None
        or args.case_limit is not None)
    protocol = EvaluationProtocol(
        instance_count=E1_INSTANCE_COUNT,
        replications=E1_SIMULATION_REPLICATIONS,
        instance_seed_start=E1_INSTANCE_SEED_START,
        simulation_seed_start=E1_SIMULATION_SEED_START)
    all_conditions = list(iter_full_factorial())
    ids_by_scene={s:selected_conditions([e1_condition_id(c) for c in all_conditions],s,
                                      limit=args.condition_limit) for s in args.scenes}
    scene_conditions=[(s,c) for s in args.scenes for c in all_conditions
                      if e1_condition_id(c) in ids_by_scene[s]]
    conditions=[c for c in all_conditions if any(e1_condition_id(c) in ids for ids in ids_by_scene.values())]
    if args.case_limit is not None and args.case_limit <= 0:
        raise ValueError("--case-limit must be positive.")
    case_manifest = ensure_e1_case_manifest(
        args.case_manifest, all_conditions, protocol)
    frozen_dataset = FrozenE1Dataset(resolve(args.dataset_dir))
    frozen_dataset.validate_case_manifest(case_manifest.rows)
    try:
        dataset_reference = frozen_dataset.root.relative_to(WORKSPACE).as_posix()
    except ValueError:
        dataset_reference = str(frozen_dataset.root)
    model_root = resolve(args.model_root)
    output = resolve(args.output_dir)
    source_sha = source_fingerprint(WORKSPACE)
    suite_contract = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E1 learned-method formal evaluation",
        "methods": args.methods,
        "scenes": args.scenes,
        "conditions_by_scene": ids_by_scene,
        "model_budget": args.model_budget,
        "training_seeds": args.training_seeds,
        "conditions": [asdict(condition) for condition in conditions],
        "evaluation_protocol": {**protocol.to_dict(), "instance_count": args.instances},
        "frozen_manifest_protocol": protocol.to_dict(),
        "instance_selection": "original instance_index < instances; original seeds unchanged",
        "diagnostic_case_limit": args.case_limit,
        "case_manifest_sha256": case_manifest.sha256,
        "instance_source": "frozen_csv_dataset",
        "frozen_dataset": {
            # Keep the formal suite contract portable when the project is
            # copied to another computer at a different absolute path.
            "path": dataset_reference,
            "dataset_contract_sha256": frozen_dataset.contract_sha256,
            "instance_manifest_sha256": (
                frozen_dataset.instance_manifest_sha256),
            "structural_instances": len(frozen_dataset.rows),
        },
        "source_fingerprint_sha256": source_sha,
        "requested_device": args.device,
        "save_job_records": args.save_job_records,
        "failed_run_handling": "retain_without_automatic_rerun",
    }
    suite_sha = canonical_hash(suite_contract)
    total_blocks = len(args.methods) * len(args.training_seeds) * len(scene_conditions)
    cases_per_condition = (
        args.case_limit
        if args.case_limit is not None
        else args.instances * E1_SIMULATION_REPLICATIONS)
    total_runs = total_blocks * cases_per_condition
    if args.dry_run:
        print(json.dumps({
            "suite_configuration_sha256": suite_sha,
            "case_manifest_sha256": case_manifest.sha256,
            "methods": len(args.methods),
            "training_seeds": len(args.training_seeds),
            "conditions": len(conditions),
            "blocks": total_blocks,
            "runs": total_runs,
            "instance_source": "frozen_csv_dataset",
            "frozen_dataset_contract_sha256": (
                frozen_dataset.contract_sha256),
        }, ensure_ascii=False, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_save({**suite_contract, "suite_configuration_sha256": suite_sha,
                     "planned_runs": total_runs}, output / "evaluation_selection.json")
    metadata_path = output / "run_metadata.json"
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        atomic_json_save({
            **suite_contract,
            "suite_configuration_sha256": suite_sha,
            "case_manifest": str(case_manifest.path),
            "model_root": str(model_root),
            "planned_blocks": total_blocks,
            "planned_runs": total_runs,
            "runtime": runtime_provenance(WORKSPACE),
        }, metadata_path)
    print(
        f"[DRL-EVAL] plan methods={len(args.methods)} seeds="
        f"{len(args.training_seeds)} conditions={len(conditions)} "
        f"blocks={total_blocks} runs={total_runs} device={args.device}",
        flush=True)
    print(
        f"[DRL-EVAL] cases={case_manifest.path} "
        f"sha256={case_manifest.sha256}", flush=True)
    print(
        f"[DRL-EVAL] frozen_dataset={frozen_dataset.root} "
        f"contract_sha256={frozen_dataset.contract_sha256}",
        flush=True,
    )
    manifests = []
    from experiment.selected_hyperparameters import load_selection,source_directory
    selection=load_selection(resolve(args.hyperparameter_file)) if args.hyperparameter_file else {'methods':{}}
    suite_started = time.perf_counter()
    block_number = 0
    for method in args.methods:
        for training_seed in args.training_seeds:
            model_dir = source_directory(model_root,method,training_seed,args.model_budget,
                selection=selection,grid_layout=args.model_layout=='grid')
            for scene_id,condition in scene_conditions:
                block_number += 1
                identifier = e1_condition_id(condition)
                print(
                    f"[DRL-EVAL] block {block_number}/{total_blocks} START "
                    f"method={method.upper()} seed={training_seed} "
                    f"condition={identifier}", flush=True)
                manifest = evaluate_block(
                    output=output,
                    method=method,
                    training_seed=training_seed,
                    condition=condition,
                    cases=[case for case in case_manifest.cases_by_condition[identifier]
                           if case.instance_index < args.instances][
                        :args.case_limit
                        if args.case_limit is not None else None],
                    case_manifest_sha=case_manifest.sha256,
                    suite_configuration_sha=suite_sha,
                    source_sha=source_sha,
                    model_dir=model_dir,
                    device=args.device,
                    save_job_records=args.save_job_records,
                    log_every=args.log_every,
                    block_number=block_number,
                    total_blocks=total_blocks,
                    frozen_dataset=frozen_dataset,
                    scene_id=scene_id,
                    model_budget=args.model_budget,
                )
                manifests.append(manifest)
                elapsed = time.perf_counter() - suite_started
                eta = elapsed / block_number * (total_blocks - block_number)
                print(
                    f"[DRL-EVAL] block {block_number}/{total_blocks} "
                    f"{'SKIP' if manifest['skipped'] else 'DONE'} "
                    f"failed={manifest['failed_runs']} ETA={eta / 3600:.2f}h",
                    flush=True)
    consolidated = {}
    for table in ("run_metrics", "machine_metrics", "state_action_metrics"):
        print(f"[DRL-EVAL] consolidating {table}.csv ...", flush=True)
        target = output / f"{table}.csv"
        count = consolidate_csv(output / "partitions" / table, target)
        consolidated[table] = {
            "relative_path": target.relative_to(output).as_posix(),
            "rows": count,
            "sha256": file_sha256(target),
        }
    atomic_csv_save([
        {key: value for key, value in manifest.items()
         if key not in {"files", "model_artifacts", "skipped"}}
        for manifest in manifests
    ], output / "block_manifest.csv")
    elapsed = time.perf_counter() - suite_started
    atomic_json_save({
        "schema_version": SCHEMA_VERSION,
        "suite_configuration_sha256": suite_sha,
        "case_manifest_sha256": case_manifest.sha256,
        "blocks": len(manifests),
        "completed_runs": sum(item["completed_runs"] for item in manifests),
        "failed_runs": sum(item["failed_runs"] for item in manifests),
        "excluded_runs": 0,
        "elapsed_seconds": elapsed,
        "consolidated_files": consolidated,
    }, output / "completion_summary.json")
    print(
        f"[DRL-EVAL] finished blocks={len(manifests)} "
        f"runs={sum(item['completed_runs'] for item in manifests)} "
        f"failed={sum(item['failed_runs'] for item in manifests)} "
        f"elapsed={elapsed / 3600:.2f}h output={output}",
        flush=True)


if __name__ == "__main__":
    main()
