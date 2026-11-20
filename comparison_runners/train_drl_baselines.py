"Train HIPMC, DDQNB, PSA-MAPPO and HDRL with common instance and seed schedules."

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from comparison_runners.drl_common import (
    DEFAULT_TRAINING_MANIFEST,
    DEFAULT_VALIDATION_MANIFEST,
    LEARNED_METHODS,
    build_drl_environment,
    checkpoint_name,
    ensure_training_manifest,
    ensure_validation_manifest,
    model_artifact_names,
    training_condition,
)
from experiment.artifacts import atomic_json_save
from experiment.config import E1_STOCHASTIC_METHOD_SEEDS
from experiment.evaluation import seed_simulation
from experiment.provenance import runtime_provenance, source_fingerprint
from experiment.reward_scale import load_reward_scale
from experiment.seeds import TrainingSeeds
from comparison_runners.training_budgets import configure_learning, run_training_budget


from experiment.revision_protocol import (
    STANDARD_EPISODES as FORMAL_EPISODES,
    VALIDATION_INTERVAL as FORMAL_VALIDATION_INTERVAL,
    VALIDATION_INSTANCES as FORMAL_VALIDATION_INSTANCES,
    CHECKPOINT_INTERVAL as FORMAL_CHECKPOINT_INTERVAL,
    LEARNED_METHODS as REVISION_METHODS,
)


def canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=LEARNED_METHODS,
        default=list(REVISION_METHODS))
    parser.add_argument(
        "--training-seeds", nargs="+", type=int,
        default=list(E1_STOCHASTIC_METHOD_SEEDS))
    parser.add_argument("--episodes", type=int, default=FORMAL_EPISODES)
    parser.add_argument(
        "--validation-interval", type=int,
        default=FORMAL_VALIDATION_INTERVAL)
    parser.add_argument(
        "--validation-instances", type=int,
        default=FORMAL_VALIDATION_INSTANCES)
    parser.add_argument(
        "--checkpoint-interval", type=int,
        default=FORMAL_CHECKPOINT_INTERVAL)
    parser.add_argument("--device", default="auto")
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--ppo-clip-ratio',type=float)
    parser.add_argument('--discount-factor',type=float)
    parser.add_argument('--reward-formulation', choices=('potential_difference','discount_consistent_potential'),
                        default='potential_difference')
    parser.add_argument('--hyperparameter-file',type=Path,
                        help='Per-method selections made only on generated training/validation data.')
    parser.add_argument(
        "--reward-scale-file", type=Path,
        help=(
            "Offline-calibrated SDR normalization artifact shared by PPO, "
            "QMIX, MAPPO, and the proposed method."),
    )
    parser.add_argument(
        "--training-manifest", type=Path,
        default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument(
        "--validation-manifest", type=Path,
        default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument(
        "--output-dir", type=Path,
        default=WORKSPACE / "runs" / "e1_learned_method_training")
    parser.add_argument(
        "--diagnostic", action="store_true",
        help="Allow reduced budgets with explicitly separate manifests.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.reward_formulation != 'potential_difference' and args.methods != ['ours_categorical_baa']:
        raise ValueError('The shaping follow-up is restricted to HIPMC.')
    if min(
            args.episodes,
            args.validation_interval,
            args.validation_instances,
            args.checkpoint_interval) <= 0:
        raise ValueError("All training and validation counts must be positive.")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods contains duplicates.")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        raise ValueError("--training-seeds contains duplicates.")
    formal = (
        tuple(args.training_seeds) == E1_STOCHASTIC_METHOD_SEEDS
        and args.episodes == FORMAL_EPISODES
        and args.validation_interval == FORMAL_VALIDATION_INTERVAL
        and args.validation_instances == FORMAL_VALIDATION_INSTANCES
        and args.checkpoint_interval == FORMAL_CHECKPOINT_INTERVAL
    )
    if args.diagnostic and (
            args.training_manifest.resolve()
            == DEFAULT_TRAINING_MANIFEST.resolve()
            or args.validation_manifest.resolve()
            == DEFAULT_VALIDATION_MANIFEST.resolve()):
        raise ValueError(
            "Diagnostic runs must use explicit non-formal --training-manifest "
            "and --validation-manifest paths.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output = args.output_dir
    if not output.is_absolute():
        output = (WORKSPACE / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_manifest = ensure_training_manifest(
        args.training_manifest, args.training_seeds, args.episodes)
    validation_manifest = ensure_validation_manifest(
        args.validation_manifest, args.validation_instances)
    source_sha = source_fingerprint(WORKSPACE)
    if args.dry_run and args.reward_scale_file is None:
        print(json.dumps({
            "methods": args.methods,
            "training_seeds": args.training_seeds,
            "training_tasks": len(args.methods) * len(args.training_seeds),
            "episodes_per_task": args.episodes,
            "total_training_episodes": (
                len(args.methods) * len(args.training_seeds) * args.episodes),
            "training_manifest_sha256": training_manifest.sha256,
            "validation_manifest_sha256": validation_manifest.sha256,
            "reward_scale_file_required_for_execution": True,
        }, ensure_ascii=False, indent=2))
        return
    reward_scale = None
    reward_scale_sha = None
    reward_scale_metadata = None
    if args.reward_scale_file is not None:
        artifact, reward_scale_sha = load_reward_scale(
            args.reward_scale_file,
            workspace=WORKSPACE,
            scenario_name="base",
            condition=training_condition(args.training_seeds[0]),
        )
        reward_scale = float(artifact["scale"]["value"])
        reward_scale_metadata = {
            "artifact": str(args.reward_scale_file.resolve()),
            "artifact_sha256": reward_scale_sha,
            "contract_sha256": artifact["contract_sha256"],
            "value": reward_scale,
        }
    print(
        f"[LEARNED-TRAIN] methods={','.join(args.methods)} "
        f"seeds={','.join(map(str, args.training_seeds))} "
        f"episodes={args.episodes} requested_device={args.device}",
        flush=True,
    )
    print(
        f"[LEARNED-TRAIN] training_manifest={training_manifest.path} "
        f"sha256={training_manifest.sha256} rows={len(training_manifest.rows)}",
        flush=True,
    )
    print(
        f"[LEARNED-TRAIN] validation_manifest={validation_manifest.path} "
        f"sha256={validation_manifest.sha256} rows={len(validation_manifest.rows)}",
        flush=True,
    )
    tasks = [(method, seed) for method in args.methods
             for seed in args.training_seeds]
    suite_started = time.perf_counter()
    for task_index, (method, training_seed) in enumerate(tasks, start=1):
        run_dir = output / method / f"seed-{training_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        condition = training_condition(training_seed)
        contract = {
            "schema_version": 1,
            "experiment": "E1 learned-method training",
            "method": method,
            "training_seed": training_seed,
            "condition": asdict(condition),
            "episodes": args.episodes,
            "validation_interval": args.validation_interval,
            "validation_instances": args.validation_instances,
            "checkpoint_interval": args.checkpoint_interval,
            "training_manifest_sha256": training_manifest.sha256,
            "validation_manifest_sha256": validation_manifest.sha256,
            "source_fingerprint_sha256": source_sha,
            "requested_device": args.device,
            "reward_normalization": (
                reward_scale_metadata or {
                    "mode": "diagnostic_unit_scale",
                    "value": 1.0,
                }),
        }
        configuration_sha = canonical_hash(contract)
        config_path = run_dir / "run_config.json"
        complete_path = run_dir / "run_complete.json"
        if config_path.is_file():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            existing = {}
        if existing and existing.get('hyperparameters',{}).get('reward_formulation','potential_difference') != args.reward_formulation:
            raise ValueError('Output directory contains a different reward formulation; use a separate directory.')
        if complete_path.is_file() and int(json.loads(
                complete_path.read_text(encoding='utf-8')).get('episodes',0)) >= args.episodes:
            print(
                f"[LEARNED-TRAIN] task {task_index}/{len(tasks)} SKIP "
                f"method={method.upper()} seed={training_seed}",
                flush=True,
            )
            continue
        seed_simulation(training_seed)
        environment, resolved_device = build_drl_environment(
            method,
            condition,
            device=args.device,
            calibrated_reward_scale=reward_scale,
            reward_scale_artifact_sha256=reward_scale_sha,
        )
        previous=existing.get('hyperparameters',{})
        selected={
            'learning_rate':args.learning_rate if args.learning_rate is not None else previous.get('learning_rate'),
            'clip_ratio':args.ppo_clip_ratio if args.ppo_clip_ratio is not None else previous.get('clip_ratio'),
            'gamma':args.discount_factor if args.discount_factor is not None else previous.get('gamma_baa'),
        }
        from experiment.selected_hyperparameters import load_selection,selected_parameters
        from_file=selected_parameters(load_selection(args.hyperparameter_file),method)
        selected={**selected,**from_file}
        # Explicit command values allow the controlled sensitivity grid.
        for key,value in [('learning_rate',args.learning_rate),('clip_ratio',args.ppo_clip_ratio),('gamma',args.discount_factor)]:
            if value is not None:selected[key]=value
        selected['reward_formulation']=args.reward_formulation
        effective=configure_learning(environment,selected)
        contract['hyperparameters']=effective
        contract['hyperparameter_selection_file']=str(args.hyperparameter_file) if args.hyperparameter_file else None
        contract['schedule_reference_episodes']=FORMAL_EPISODES
        configuration_sha=canonical_hash(contract)
        atomic_json_save({
            **contract,
            "configuration_sha256": configuration_sha,
            "resolved_device": resolved_device,
            "runtime": runtime_provenance(WORKSPACE),
            "model_artifacts": model_artifact_names(method),
        }, config_path)
        environment.max_episode = args.episodes
        environment.validation_interval = args.validation_interval
        environment.validation_instances = args.validation_instances
        environment.checkpoint_interval = args.checkpoint_interval
        selected_training_rows = [
            row for row in training_manifest.rows
            if int(row["training_seed"]) == training_seed
            and int(row['episode']) <= args.episodes
        ]
        if len(selected_training_rows) != args.episodes:
            raise RuntimeError(
                "Shared training manifest does not contain the exact budget.")
        environment.training_seed_schedule = [TrainingSeeds(
            episode=int(row["episode"]) - 1,
            instance_seed=int(row["instance_seed"]),
            simulation_seed=int(row["simulation_seed"]),
            policy_seed=int(row["policy_seed"]),
        ) for row in selected_training_rows]
        environment.validation_seed_schedule = [{
            "instance_seed": int(row["instance_seed"]),
            "simulation_seed": int(row["simulation_seed"]),
            "policy_seed": int(row["policy_seed"]),
        } for row in validation_manifest.rows]
        checkpoint = run_dir / checkpoint_name(method)
        resume_from = checkpoint if checkpoint.is_file() else None
        print(
            f"[LEARNED-TRAIN] task {task_index}/{len(tasks)} START "
            f"method={method.upper()} seed={training_seed} "
            f"device={resolved_device} resume={bool(resume_from)}",
            flush=True,
        )
        started = time.perf_counter()
        run_training_budget(environment,method,run_dir,args.episodes)
        elapsed = time.perf_counter() - started
        missing = [
            name for name in model_artifact_names(method)
            if not (run_dir / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Training completed without model artifacts: {missing}")
        atomic_json_save({
            "status": "complete",
            "configuration_sha256": configuration_sha,
            "method": method,
            "training_seed": training_seed,
            "episodes": environment.total_episode,
            "best_validation_mean_tardiness": environment.best_validation,
            "elapsed_seconds": elapsed,
            "training_manifest_sha256": training_manifest.sha256,
            "validation_manifest_sha256": validation_manifest.sha256,
            "source_fingerprint_sha256": source_sha,
        }, complete_path)
        print(
            f"[LEARNED-TRAIN] task {task_index}/{len(tasks)} DONE "
            f"method={method.upper()} seed={training_seed} "
            f"best_validation={environment.best_validation:.6f} "
            f"elapsed={elapsed / 3600:.2f}h",
            flush=True,
        )
    print(
        f"[LEARNED-TRAIN] suite finished tasks={len(tasks)} "
        f"elapsed={(time.perf_counter() - suite_started) / 3600:.2f}h "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
