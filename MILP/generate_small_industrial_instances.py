"Construct 10- and 20-job industrial subsets with scaled due allowances."
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

PROCESSING_TIMES = (128, 190, 104, 293, 202, 279)
MACHINE_COUNTS = (1, 1, 1)
MACHINE_CAPACITIES = (120, 160, 200)
INITIAL_JOB_FRACTION = 0.20
MIN_SLACK_RATIO = 0.25
SUPPORTED_JOB_TIERS = (10, 20)
RETIRED_JOB_TIERS = (30,)

WORKSPACE = Path(__file__).resolve().parents[1]
REFERENCE_INSTANCE = (
    WORKSPACE / "datasets/aerospace_heat_treatment_dynamic_case/instances/jobs/"
    "load_1p00/industrial_orders_load-1p00_instance-001.csv")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_job_tiers(values: list[int]) -> tuple[int, ...]:
    """Validate the frozen small-instance size contract."""

    tiers = tuple(values)
    if not tiers:
        raise ValueError("At least one job-count tier is required.")
    if len(set(tiers)) != len(tiers):
        raise ValueError("Job-count tiers cannot contain duplicates.")
    unsupported = sorted(set(tiers) - set(SUPPORTED_JOB_TIERS))
    if unsupported:
        raise ValueError(
            "The formal MILP benchmark only supports 10- and 20-job "
            f"instances; unsupported tier(s): {unsupported}."
        )
    return tiers


def remove_retired_tier_directories(output: Path) -> list[Path]:
    """Remove only explicitly retired generated tiers under this dataset."""

    jobs_root = (output / "jobs").resolve()
    removed = []
    for tier in RETIRED_JOB_TIERS:
        target = (jobs_root / f"jobs-{tier}").resolve()
        if target.parent != jobs_root:
            raise RuntimeError(f"Unsafe retired-tier cleanup target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(target)
    return removed


def load_reference_orders(path: Path) -> list[dict]:
    """从参考实例提取 89 个真实订单 (family, size, due, arr_ref)。"""
    orders = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            orders.append({
                "family": int(r["family"]),
                "size": int(r["size"]),
                "due": float(r["due_time"]),
                "arr_ref": float(r["arrival_time"]),
            })
    if not orders:
        raise ValueError(f"Reference industrial order file is empty: {path}")
    if any(not 0 <= order["family"] < len(PROCESSING_TIMES) for order in orders):
        raise ValueError("Reference orders contain a family outside the scene.")
    return orders


def arrival_span(orders: list[dict], load: float) -> float:
    total_weight = sum(
        o["size"] * PROCESSING_TIMES[o["family"]] for o in orders)
    capacity = sum(
        mc * cap for mc, cap in zip(MACHINE_COUNTS, MACHINE_CAPACITIES))
    return total_weight / (load * capacity)


def gen_arrivals(orders: list[dict], span: float, rng: random.Random) -> list[float]:
    """生成到达时间：~20% 零时刻 + 指数间隔动态到达 + due-feasible 校正。"""
    n = len(orders)
    n_init = max(1, round(n * INITIAL_JOB_FRACTION))
    perm = rng.sample(range(n), n)  # 随机顺序决定谁是初始工件
    dyn_idx = perm[n_init:]
    m = n - n_init
    # 指数间隔 → 归一化到 [0, span]
    gaps = [rng.expovariate(1.0) for _ in range(m)]
    total = sum(gaps)
    cum = 0.0
    dyn_arr = []
    for g in gaps:
        cum += g
        dyn_arr.append(cum / total * span)
    arrivals = [0.0] * n
    for j, raw in zip(dyn_idx, dyn_arr):
        pt = PROCESSING_TIMES[orders[j]["family"]]
        max_arr = max(0.0, orders[j]["due"] - MIN_SLACK_RATIO * pt)
        arrivals[j] = min(raw, max_arr)
    # 避免零时刻以外的动态工件也挤在 0 附近（due-feasible 截断可能造成堆叠）：
    # 对截断到同一个值的工件加微小扰动，保持时间有序且非零
    for j in dyn_idx:
        if arrivals[j] <= 1e-9:
            arrivals[j] = rng.uniform(0.05, 0.2) * span
    return arrivals


def scale_due(orders: list[dict], arrivals: list[float], span: float,
              span_ref: float) -> list[float]:
    """按跨度缩放 slack，保持真实订单的相对紧度。"""
    scale = span / span_ref
    scaled = []
    for o, a in zip(orders, arrivals):
        slack = o["due"] - o["arr_ref"]
        scaled.append(a + slack * scale)
    return scaled


def build_instance(n_jobs: int, load: float, rng: random.Random,
                   orders_all: list[dict], span_ref: float,
                   due_mode: str) -> dict:
    if not 1 <= n_jobs <= len(orders_all):
        raise ValueError(
            f"n_jobs must be between 1 and {len(orders_all)}, got {n_jobs}.")
    if not load > 0:
        raise ValueError("target load must be positive.")
    idx = rng.sample(range(len(orders_all)), n_jobs)
    orders = [dict(orders_all[i]) for i in idx]
    span = arrival_span(orders, load)
    arrivals = gen_arrivals(orders, span, rng)
    if due_mode == "scale":
        dues = scale_due(orders, arrivals, span, span_ref)
    else:
        dues = [o["due"] for o in orders]
    rows = [{
        "job_id": k, "arrival_time": round(arrivals[k], 4),
        "family": orders[k]["family"], "size": orders[k]["size"],
        "due_time": round(dues[k], 4),
    } for k in range(n_jobs)]
    rows.sort(key=lambda r: (r["arrival_time"], r["job_id"]))
    return {
        "rows": rows, "n_jobs": n_jobs, "load": load, "span": span,
        "n_initial": sum(1 for r in rows if r["arrival_time"] == 0),
        "source_order_indices": idx,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, nargs="+", default=[10, 20],
                    help="每档工件数量")
    ap.add_argument("--per", type=int, default=10, help="每档实例数")
    ap.add_argument("--load", type=float, default=1.0,
                    help="目标名义负载（默认 1.0）")
    ap.add_argument("--seed-base", type=int, default=84000000,
                    help="实例随机种子基数")
    ap.add_argument("--due-mode", choices=("scale", "preserve"),
                    default="scale", help="交期处理方式（默认 scale）")
    ap.add_argument("--output", type=Path,
                    default=WORKSPACE / "datasets/industrial_case_small")
    ap.add_argument("--reference", type=Path,
                    default=Path(REFERENCE_INSTANCE))
    args = ap.parse_args()
    args.n_jobs = list(validate_job_tiers(args.n_jobs))

    orders_all = load_reference_orders(args.reference)
    span_ref = max(o["arr_ref"] for o in orders_all)
    print(f"参考订单数: {len(orders_all)} | span_ref: {span_ref:.1f}")

    out = args.output
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "provenance").mkdir(parents=True, exist_ok=True)
    for removed in remove_retired_tier_directories(out):
        print(f"Removed retired 30-job instance tier: {removed}")

    manifest_rows = []
    for n in args.n_jobs:
        d = out / "jobs" / f"jobs-{n}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, args.per + 1):
            # 种子与 (档位, 实例号) 绑定而非全局计数器：任何档位组合/顺序/
            # 数量下重新生成，同一 (jobs-N, instance-i) 的结果始终一致。
            rng = random.Random(args.seed_base + n * 1000 + i)
            inst = build_instance(n, args.load, rng, orders_all, span_ref,
                                  args.due_mode)
            path = d / f"instance-{i:03d}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "job_id", "arrival_time", "family", "size", "due_time"])
                w.writeheader()
                w.writerows(inst["rows"])
            manifest_rows.append({
                "n_jobs": n, "instance_number": i, "target_load": args.load,
                "arrival_span": round(inst["span"], 4),
                "initial_job_count": inst["n_initial"],
                "initial_job_fraction": round(
                    inst["n_initial"] / n, 4),
                "job_file": f"jobs/jobs-{n}/instance-{i:03d}.csv",
                "job_file_sha256": file_sha256(path),
                "source_order_indices": json.dumps(inst["source_order_indices"]),
            })
            print(f"  jobs-{n}/instance-{i:03d}: span={inst['span']:.0f} "
                  f"initial={inst['n_initial']} ({inst['n_initial']/n:.0%})")

    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    config = {
        "schema_version": 2,
        "dataset_id": "derived_small_aerospace_heat_treatment_scheduling_cases",
        "protocol": "stratified_order_subsampling_with_derived_arrivals_and_due_dates",
        "n_jobs_per_tier": args.n_jobs,
        "instances_per_tier": args.per,
        "target_load": args.load,
        "initial_job_fraction": INITIAL_JOB_FRACTION,
        "min_slack_ratio": MIN_SLACK_RATIO,
        "arrival_process": "normalized exponential interarrival spacings",
        "arrival_span_formula": (
            "sum(size*pt) / (load * sum(machine_count*capacity))"),
        "machine_counts": list(MACHINE_COUNTS),
        "machine_capacities": list(MACHINE_CAPACITIES),
        "processing_times": list(PROCESSING_TIMES),
        "due_mode": args.due_mode,
        "reference_instance": str(args.reference.resolve()),
        "reference_instance_sha256": file_sha256(args.reference),
        "reference_span": round(span_ref, 4),
        "order_source": "89 fixed industrial orders (family,size,due)",
        "observed_fields": ["family", "size", "reference_due_time"],
        "derived_fields": ["arrival_time", "due_time_after_horizon_scaling"],
        "seed_base": args.seed_base,
    }
    (out / "provenance" / "generation_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    checksums = {
        "schema_version": 1,
        "files": [
            {"path": row["job_file"], "sha256": row["job_file_sha256"]}
            for row in manifest_rows
        ],
        "manifest_sha256": file_sha256(manifest_path),
    }
    (out / "provenance" / "checksums.json").write_text(
        json.dumps(checksums, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成: {len(manifest_rows)} 个实例 -> {out}")
    print(f"manifest: {manifest_path}")
    print(f"配置: {out / 'provenance' / 'generation_config.json'}")


if __name__ == "__main__":
    main()
