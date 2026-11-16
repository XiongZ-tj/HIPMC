"Solve deterministic parallel-batch production scheduling with Gurobi."
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # 允许无 Gurobi 环境导入并运行输入/结果契约测试
    gp = None
    GRB = None

# ---------------- 默认场景参数（E1 base；可用 scene/CLI 覆盖为工业等） ----------------
PROCESSING_TIMES = (190, 158, 202, 128, 104)   # 5 个工件族加工时长
MACHINE_COUNTS = (2, 2, 1)                     # 3 类机器, 共 5 台
MACHINE_CAPACITIES = (60, 80, 100)             # 每类机器批量容量

MACHINES = []                                  # (machine_id, class_id)
_mid = 0
for cls, count in enumerate(MACHINE_COUNTS):
    for _ in range(count):
        MACHINES.append((_mid, cls))
        _mid += 1
N_MACHINES = len(MACHINES)                     # 5
CLASS_OF = {mid: cls for mid, cls in MACHINES}

# 模块级默认别名（供 build_and_solve 场景重绑定时引用，避免同名局部变量未绑定）
_DEFAULT_PROCESSING_TIMES = PROCESSING_TIMES
_DEFAULT_MACHINE_COUNTS = MACHINE_COUNTS
_DEFAULT_CAPACITIES = MACHINE_CAPACITIES

SCENES = {
    "synthetic_reference": {
        "processing_times": _DEFAULT_PROCESSING_TIMES,
        "machine_counts": _DEFAULT_MACHINE_COUNTS,
        "capacities": _DEFAULT_CAPACITIES,
    },
    "aerospace_heat_treatment_case": {
        "processing_times": (128, 190, 104, 293, 202, 279),
        "machine_counts": (1, 1, 1),
        "capacities": (120, 160, 200),
    },
}

MODEL_SCOPE = "deterministic_offline_production_scheduling_subproblem"
OBJECTIVE_NAME = "total_tardiness"
EXCLUDED_DYNAMICS = (
    "machine_degradation",
    "imperfect_preventive_maintenance",
    "preventive_maintenance",
    "corrective_maintenance",
    "unexpected_failures",
    "stochastic_processing_rates",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_jobs(jobs: list[dict]) -> None:
    if not jobs:
        raise ValueError("The MILP requires at least one job.")
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("job_id values must be unique.")
    for job in jobs:
        for key in ("arrival_time", "due_time", "size"):
            if not math.isfinite(float(job[key])):
                raise ValueError(f"Non-finite {key} for job {job['job_id']}.")
        if job["arrival_time"] < 0 or job["size"] <= 0:
            raise ValueError(
                f"Job {job['job_id']} has a negative arrival or non-positive size.")
        if int(job["family"]) != job["family"] or job["family"] < 0:
            raise ValueError(f"Invalid family for job {job['job_id']}.")


def load_jobs(path: Path) -> list[dict]:
    jobs = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            jobs.append({
                "job_id": int(row["job_id"]),
                "arrival_time": float(row["arrival_time"]),
                "family": int(row["family"]),
                "size": int(row["size"]),
                "due_time": float(row["due_time"]),
            })
    validate_jobs(jobs)
    return jobs


def _validate_scene(
        jobs: list[dict], processing_times: tuple, machine_counts: tuple,
        capacities: tuple) -> None:
    if not processing_times or any(float(value) <= 0 for value in processing_times):
        raise ValueError("All nominal family processing times must be positive.")
    if not machine_counts or any(int(value) != value or value < 0 for value in machine_counts):
        raise ValueError("Machine counts must be non-negative integers.")
    if sum(machine_counts) <= 0 or len(machine_counts) != len(capacities):
        raise ValueError("Machine counts and capacities define an invalid fleet.")
    if any(float(value) <= 0 for value in capacities):
        raise ValueError("All machine capacities must be positive.")
    for job in jobs:
        if job["family"] >= len(processing_times):
            raise ValueError(
                f"Job {job['job_id']} family {job['family']} is outside the scene.")
        if not any(
                count > 0 and job["size"] <= capacities[cls]
                for cls, count in enumerate(machine_counts)):
            raise ValueError(
                f"Job {job['job_id']} fits no machine in the configured fleet.")


def _status_name(status: int) -> str:
    if GRB is None:
        return str(status)
    names = {
        GRB.LOADED: "LOADED", GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE", GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED", GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT", GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED", GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL", GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def audit_schedule(
        jobs: list[dict], schedule: list[dict], *, processing_times: tuple,
        capacities: tuple, tolerance: float = 1e-5) -> dict:
    """Independently verify an extracted deterministic production schedule."""
    by_id = {job["job_id"]: job for job in jobs}
    seen: list[int] = []
    completion: dict[int, float] = {}
    machine_end: dict[int, float] = {}
    for entry in sorted(schedule, key=lambda value: (
            value["machine_id"], value["slot"])):
        mid = int(entry["machine_id"])
        cls = int(entry["machine_class"])
        family = int(entry["family"])
        start = float(entry["start"])
        job_ids = [int(value) for value in entry["job_ids"]]
        if not job_ids:
            raise ValueError("An occupied MILP slot contains no jobs.")
        if any(job_id not in by_id for job_id in job_ids):
            raise ValueError("The extracted schedule contains an unknown job_id.")
        if any(by_id[job_id]["family"] != family for job_id in job_ids):
            raise ValueError("A batch mixes incompatible job families.")
        used = sum(by_id[job_id]["size"] for job_id in job_ids)
        if used > capacities[cls] + tolerance:
            raise ValueError("An extracted batch exceeds machine capacity.")
        if start + tolerance < max(by_id[job_id]["arrival_time"] for job_id in job_ids):
            raise ValueError("A batch starts before one of its jobs arrives.")
        if start + tolerance < machine_end.get(mid, 0.0):
            raise ValueError("Extracted batches overlap on a machine.")
        end = start + processing_times[family]
        machine_end[mid] = end
        for job_id in job_ids:
            seen.append(job_id)
            completion[job_id] = end
    if sorted(seen) != sorted(by_id):
        raise ValueError("Every job must occur exactly once in the extracted schedule.")
    total_tardiness = sum(
        max(0.0, completion[job_id] - by_id[job_id]["due_time"])
        for job_id in by_id)
    return {
        "passed": True,
        "audited_job_count": len(seen),
        "recomputed_total_tardiness": total_tardiness,
    }


def _heuristic_schedule(
        jobs: list[dict],
        *,
        machines: list,
        capacities: tuple,
        processing_times: tuple,
        max_slots: int) -> dict | None:
    """EDD + 容量贪心组批的可行初始解（MIPStart）。

    按最早交期排序逐个分配工件：优先并入当前机器末尾的同族批次（容量允许），
    否则放到"预计完成最早"的可用机器上新建批次。返回
    {(mid, slot): (family, [job_ids], start, end)} 或 None（某机槽数超限/无可行）。

    目标只是给 Gurobi 一个把全部机器利用起来的可行起点，不追求最优。
    """
    order = sorted(
        range(len(jobs)),
        key=lambda j: (jobs[j]["due_time"], jobs[j]["arrival_time"], jobs[j]["job_id"]))
    slots: dict[int, list[dict]] = {mid: [] for mid, _ in machines}
    cur: dict[int, float] = {mid: 0.0 for mid, _ in machines}
    for j in order:
        job = jobs[j]
        fam = job["family"]
        placed = False
        for mid, cls in machines:
            if job["size"] > capacities[cls]:
                continue
            last = slots[mid][-1] if slots[mid] else None
            if last and last["family"] == fam:
                used = sum(jobs[jj]["size"] for jj in last["job_ids"])
                if used + job["size"] <= capacities[cls]:
                    last["job_ids"].append(j)
                    placed = True
                    break
        if placed:
            continue
        best_mid = None
        best_finish = float("inf")
        for mid, cls in machines:
            if job["size"] > capacities[cls]:
                continue
            start = max(cur[mid], job["arrival_time"])
            finish = start + processing_times[fam]
            if finish < best_finish:
                best_finish = finish
                best_mid = mid
        if best_mid is None:
            return None
        if len(slots[best_mid]) >= max_slots:
            return None  # 槽数超限，放弃该初始解
        slots[best_mid].append({"family": fam, "job_ids": [j]})
        cur[best_mid] = max(cur[best_mid], job["arrival_time"]) + processing_times[fam]

    result = {}
    for mid, _ in machines:
        t = 0.0
        for si, sl in enumerate(slots[mid]):
            start = max(t, max(jobs[j]["arrival_time"] for j in sl["job_ids"]))
            dur = processing_times[sl["family"]]
            result[(mid, si)] = (sl["family"], list(sl["job_ids"]), start, start + dur)
            t = start + dur
    return result


def build_and_solve(
        jobs: list[dict],
        *,
        time_limit: float = 600.0,
        k_slots: int | None = None,
        verbose: bool = True,
        threads: int | None = None,
        mip_gap: float = 1e-6,
        solver_seed: int = 0,
        scene: dict | None = None,
        mipstart: bool = True) -> dict:
    """构建并求解纯批调度位置模型 MILP。返回结果字典。

    scene 提供场景参数覆盖：{"processing_times", "machine_counts",
    "capacities"}，未提供则用模块默认（E1 base）。
    """
    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is required to solve the MILP. Run this command with a "
            "Python environment containing Gurobi and a valid license.")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive.")
    if not 0 <= mip_gap < 1:
        raise ValueError("mip_gap must be in [0, 1).")
    validate_jobs(jobs)
    # ---------------- 场景参数（局部绑定；函数内赋值使这些名字成为局部变量，
    # 无论 scene 是否提供都需显式初始化，避免 UnboundLocalError） ----------------
    _mc = (tuple(scene.get("machine_counts", _DEFAULT_MACHINE_COUNTS))
           if scene else _DEFAULT_MACHINE_COUNTS)
    _machines = []
    for _cls, _cnt in enumerate(_mc):
        for _ in range(_cnt):
            _machines.append((len(_machines), _cls))
    MACHINES = _machines
    N_MACHINES = len(MACHINES)
    CLASS_OF = {mid: cls for mid, cls in MACHINES}
    PROCESSING_TIMES = (
        tuple(scene.get("processing_times", _DEFAULT_PROCESSING_TIMES))
        if scene else _DEFAULT_PROCESSING_TIMES)
    MACHINE_CAPACITIES = (
        tuple(scene.get("capacities", _DEFAULT_CAPACITIES))
        if scene else _DEFAULT_CAPACITIES)
    _validate_scene(jobs, PROCESSING_TIMES, _mc, MACHINE_CAPACITIES)
    J = len(jobs)
    F = len(PROCESSING_TIMES)
    if k_slots is None:
        # 每台机器 J 个位置可表示“所有工件均单独在同一台机器加工”的极端情形，
        # 因而不会因位置上限截断原生产调度问题的可行域。
        k_slots = J
    if k_slots <= 0:
        raise ValueError("k_slots must be positive.")
    K = k_slots
    complete_slot_domain = K >= J
    # 可行调度时域：等待全部工件到达后逐件串行加工一定可行。
    total_pt = sum(PROCESSING_TIMES[jobs[j]["family"]] for j in range(J))
    horizon = max(j["arrival_time"] for j in jobs) + total_pt
    bigM = horizon + max(PROCESSING_TIMES)

    m = gp.Model("pure_batch_milp")
    if not verbose:
        m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    solver_threads = threads if threads else max(1, (os.cpu_count() or 1) - 1)
    m.Params.Threads = solver_threads
    m.Params.Seed = solver_seed

    # ---------------- 决策变量 ----------------
    x = {}      # x[j, mid, k]: job j 在机 mid 位置 k 加工
    for j in range(J):
        for mid, cls in MACHINES:
            if jobs[j]["size"] > MACHINE_CAPACITIES[cls]:
                continue
            for k in range(K):
                x[j, mid, k] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{mid}_{k}")

    g = {}      # g[mid, k, f]: 位置 (mid,k) 加工族 f
    for mid, _ in MACHINES:
        for k in range(K):
            for f in range(F):
                g[mid, k, f] = m.addVar(vtype=GRB.BINARY, name=f"g_{mid}_{k}_{f}")

    z = {}      # z[mid, k]: 位置被占用（放了一个批次）
    for mid, _ in MACHINES:
        for k in range(K):
            z[mid, k] = m.addVar(vtype=GRB.BINARY, name=f"z_{mid}_{k}")

    S = {}      # S[mid, k]: 位置开始时间
    for mid, _ in MACHINES:
        for k in range(K):
            S[mid, k] = m.addVar(lb=0.0, name=f"S_{mid}_{k}")

    C = {j: m.addVar(lb=0.0, ub=horizon, name=f"C_{j}") for j in range(J)}
    T = {j: m.addVar(lb=0.0, name=f"T_{j}") for j in range(J)}

    def dur_expr(mid, k):
        return gp.quicksum(
            PROCESSING_TIMES[f] * g[mid, k, f] for f in range(F))

    # ---------------- 约束 ----------------
    # (1) 每个 job 恰好加工一次
    for j in range(J):
        m.addConstr(
            gp.quicksum(x.get((j, mid, k), 0)
                        for mid, _ in MACHINES for k in range(K)) == 1,
            name=f"once_{j}")

    # (2) 位置占用: z = Σg（位置至多放一个族批次）
    for mid, _ in MACHINES:
        for k in range(K):
            m.addConstr(
                z[mid, k] == gp.quicksum(g[mid, k, f] for f in range(F)),
                name=f"zdef_{mid}_{k}")

    # (3) 位置顺序占用 (消除对称性)
    for mid, _ in MACHINES:
        for k in range(K - 1):
            m.addConstr(z[mid, k + 1] <= z[mid, k], name=f"order_{mid}_{k}")

    # (4) 批次族一致 + 容量 + 族激活必须有 job
    for mid, cls in MACHINES:
        cap = MACHINE_CAPACITIES[cls]
        for k in range(K):
            for j in range(J):
                if (j, mid, k) in x:
                    m.addConstr(
                        x[j, mid, k] <= g[mid, k, jobs[j]["family"]],
                        name=f"fam_{j}_{mid}_{k}")
            for f in range(F):
                # 族被激活 -> 该位置至少有 1 个该族 job
                m.addConstr(
                    g[mid, k, f]
                    <= gp.quicksum(
                        x[j, mid, k] for j in range(J)
                        if (j, mid, k) in x and jobs[j]["family"] == f),
                    name=f"gactive_{mid}_{k}_{f}")
            m.addConstr(
                gp.quicksum(
                    jobs[j]["size"] * x[j, mid, k]
                    for j in range(J) if (j, mid, k) in x)
                <= cap * z[mid, k],
                name=f"cap_{mid}_{k}")

    # (4b) 族级聚合容量有效不等式；其由逐位置容量约束蕴含，但可强化 LP 松弛。
    for f in range(F):
        family_size = sum(
            jobs[j]["size"] for j in range(J) if jobs[j]["family"] == f)
        if family_size:
            m.addConstr(
                gp.quicksum(
                    MACHINE_CAPACITIES[cls] * g[mid, k, f]
                    for mid, cls in MACHINES for k in range(K))
                >= family_size,
                name=f"family_capacity_{f}")

    # (5) 位置开始不早于其内任意 job 的到达
    for mid, _ in MACHINES:
        for k in range(K):
            m.addConstr(S[mid, k] <= horizon * z[mid, k],
                        name=f"start_active_{mid}_{k}")
            for j in range(J):
                if (j, mid, k) in x:
                    m.addConstr(
                        S[mid, k]
                        >= jobs[j]["arrival_time"] - bigM * (1 - x[j, mid, k]),
                        name=f"arr_{j}_{mid}_{k}")

    # (6) 位置递推时序：仅当下一位置被占用时启用。若当前位置占用而
    # 下一位置为空，不能再约束空位置的开始时间，否则会与 S<=H*z 冲突并
    # 错误地迫使机器使用全部 K 个位置。
    for mid, _ in MACHINES:
        for k in range(K - 1):
            m.addConstr(
                S[mid, k + 1] >= S[mid, k] + dur_expr(mid, k)
                - bigM * (1 - z[mid, k + 1]),
                name=f"seq_{mid}_{k}")

    # (7) 完成时间线性化：唯一被选择的位置把 C[j] 夹定为 S+p。
    # 不能写成 C=sum((S+p)*x)，后者包含连续变量与二元变量的乘积，
    # 会把本应为 MILP 的模型变成含双线性项的 MIQCP。
    for j in range(J):
        duration = PROCESSING_TIMES[jobs[j]["family"]]
        for mid, _ in MACHINES:
            for k in range(K):
                if (j, mid, k) not in x:
                    continue
                m.addConstr(
                    C[j] >= S[mid, k] + duration
                    - bigM * (1 - x[j, mid, k]),
                    name=f"comp_lb_{j}_{mid}_{k}")
                m.addConstr(
                    C[j] <= S[mid, k] + duration
                    + bigM * (1 - x[j, mid, k]),
                    name=f"comp_ub_{j}_{mid}_{k}")

    # (8) 拖期
    for j in range(J):
        m.addConstr(T[j] >= C[j] - jobs[j]["due_time"], name=f"tardy_{j}")
        m.addConstr(T[j] >= 0, name=f"tardy0_{j}")

    # ---------------- 目标 ----------------
    m.setObjective(gp.quicksum(T[j] for j in range(J)), GRB.MINIMIZE)

    # ---------------- MIPStart：EDD + 容量贪心初始解 ----------------
    if mipstart:
        hs = _heuristic_schedule(
            jobs, machines=MACHINES, capacities=MACHINE_CAPACITIES,
            processing_times=PROCESSING_TIMES, max_slots=K)
        if hs is not None:
            job_slot = {}
            for (mid, si), (fam, jids, start, end) in hs.items():
                g[mid, si, fam].Start = 1.0
                z[mid, si].Start = 1.0
                S[mid, si].Start = start
                for j in jids:
                    x[j, mid, si].Start = 1.0
                    job_slot[j] = (end, jobs[j]["due_time"])
            for j in range(J):
                if j in job_slot:
                    end, due = job_slot[j]
                    C[j].Start = end
                    T[j].Start = max(0.0, end - due)
            m.update()

    # ---------------- 求解 ----------------
    t0 = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter() - t0

    status = m.Status
    has_solution = m.SolCount > 0
    solver_proved_optimality = status == GRB.OPTIMAL
    objective = float(m.ObjVal) if has_solution else None
    try:
        raw_bound = float(m.ObjBound)
        objective_bound = raw_bound if math.isfinite(raw_bound) else None
    except (AttributeError, gp.GurobiError):
        objective_bound = None
    try:
        raw_gap = float(m.MIPGap) if has_solution else None
        relative_gap = (
            raw_gap if raw_gap is not None and math.isfinite(raw_gap) else None)
    except (AttributeError, gp.GurobiError):
        relative_gap = None
    result = {
        "schema_version": 2,
        "model_scope": MODEL_SCOPE,
        "formulation_class": "MILP",
        "information_pattern": "offline_clairvoyant_arrival_times",
        "objective_name": OBJECTIVE_NAME,
        "excluded_dynamics": list(EXCLUDED_DYNAMICS),
        "status": status,
        "status_name": _status_name(status),
        "has_incumbent": bool(has_solution),
        "solver_proved_optimality": bool(solver_proved_optimality),
        "slot_domain_is_complete": complete_slot_domain,
        "is_proven_global_production_optimum": bool(
            has_solution and solver_proved_optimality and complete_slot_domain),
        # 向后兼容字段；其语义现在严格限定为完整生产子问题的全局最优。
        "is_optimal": bool(
            has_solution and solver_proved_optimality and complete_slot_domain),
        "objective": objective,
        "objective_unit": "time_unit",
        "obj_bound": objective_bound,
        "mip_gap": relative_gap,
        "node_count": m.NodeCount,
        "solution_count": m.SolCount,
        "model_variables": m.NumVars,
        "model_linear_constraints": m.NumConstrs,
        "model_quadratic_constraints": m.NumQConstrs,
        "gurobi_version": list(gp.gurobi.version()),
        "solver_seed": solver_seed,
        "solver_threads": solver_threads,
        "mipstart_enabled": bool(mipstart),
        "requested_relative_mip_gap": mip_gap,
        "elapsed_seconds": elapsed,
        "time_limit": time_limit,
        "n_jobs": J,
        "n_machines": N_MACHINES,
        "k_slots": K,
        "scene": {
            "processing_times": list(PROCESSING_TIMES),
            "machine_counts": list(_mc),
            "capacities": list(MACHINE_CAPACITIES),
        },
    }

    if has_solution:
        # 提取调度
        schedule = []
        for mid, cls in MACHINES:
            for k in range(K):
                if z[mid, k].X > 0.5:
                    fams = [f for f in range(F) if g[mid, k, f].X > 0.5]
                    schedule.append({
                        "machine_id": mid, "machine_class": cls, "slot": k,
                        "start": round(S[mid, k].X, 4),
                        "family": fams[0] if fams else None,
                        "job_ids": [jobs[j]["job_id"] for j in range(J)
                                    if (j, mid, k) in x and x[j, mid, k].X > 0.5],
                    })
        result["schedule"] = schedule
        result["job_completion"] = {
            str(jobs[j]["job_id"]): round(C[j].X, 4) for j in range(J)}
        # ---- 方案汇总统计 ----
        completion = {j: C[j].X for j in range(J)}
        by_id = {job["job_id"]: job for job in jobs}
        job_place = {}  # job -> (machine_id, slot)
        for e in schedule:
            for job_id in e["job_ids"]:
                job_place[job_id] = (e["machine_id"], e["slot"])
        tardy = [j for j in range(J) if completion[j] > jobs[j]["due_time"] + 1e-9]
        batch_fills = []
        for e in schedule:
            cap = MACHINE_CAPACITIES[e["machine_class"]]
            used = sum(by_id[job_id]["size"] for job_id in e["job_ids"])
            batch_fills.append(used / cap)
        schedule_audit = audit_schedule(
            jobs, schedule, processing_times=PROCESSING_TIMES,
            capacities=MACHINE_CAPACITIES)
        if not math.isclose(
                schedule_audit["recomputed_total_tardiness"], objective,
                rel_tol=1e-7, abs_tol=1e-4):
            raise RuntimeError(
                "Extracted schedule objective does not match the Gurobi incumbent.")
        result.update({
            "schedule_audit": schedule_audit,
            "n_batches": len(schedule),
            "mean_batch_fill_ratio": round(
                sum(batch_fills) / len(batch_fills), 4) if batch_fills else 0.0,
            "makespan": round(max(completion.values()), 4),
            "n_tardy_jobs": len(tardy),
            "tardy_job_ratio": round(len(tardy) / J, 4),
            "mean_flow_time": round(
                sum(completion[j] - jobs[j]["arrival_time"] for j in range(J)) / J, 4),
            # job 级明细（含拖期与所在机器/槽位）
            "job_details": [{
                "job_id": jobs[j]["job_id"],
                "arrival_time": round(jobs[j]["arrival_time"], 4),
                "family": jobs[j]["family"],
                "size": jobs[j]["size"],
                "due_time": round(jobs[j]["due_time"], 4),
                "completion_time": round(completion[j], 4),
                "tardiness": round(max(0.0, completion[j] - jobs[j]["due_time"]), 4),
                "machine_id": job_place.get(jobs[j]["job_id"], (None, None))[0],
                "slot": job_place.get(jobs[j]["job_id"], (None, None))[1],
            } for j in range(J)],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, required=True,
                        help="冻结 job 文件 (job_id,arrival_time,family,size,due_time)")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/milp_exact_solutions"))
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="单实例求解时限（秒），默认 600")
    parser.add_argument("--k-slots", type=int, default=None)
    parser.add_argument(
        "--scene", choices=tuple(SCENES), default="synthetic_reference",
        help="规范场景名称；其参数可再由下面三个显式参数覆盖")
    parser.add_argument("--processing-times", type=str, default=None,
                        help="逗号分隔的每族加工时长，如 128,190,104,293,202,279（默认 E1 5 族）")
    parser.add_argument("--machine-counts", type=str, default=None,
                        help="逗号分隔的每类机器数，如 1,1,1（默认 E1 2,2,1）")
    parser.add_argument("--capacities", type=str, default=None,
                        help="逗号分隔的每类机器容量，如 120,160,200（默认 E1 60,80,100）")
    parser.add_argument("--no-mipstart", action="store_true",
                        help="不注入 EDD+容量贪心初始解（默认注入）")
    parser.add_argument("--force", action="store_true",
                        help="强制重算：即使该实例已求解过也重新求解")
    parser.add_argument("--quiet", action="store_true",
                        help="关闭 Gurobi 优化日志（默认显示）")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mip-gap", type=float, default=1e-6,
                        help="Gurobi relative MIP gap termination target")
    parser.add_argument("--solver-seed", type=int, default=0,
                        help="Gurobi solver seed (default: 0)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    # 输出文件名含父目录（档位）保证唯一：jobs-10/instance-001.csv ->
    # jobs-10_instance-001_milp.json，避免不同档位同名实例互相覆盖。
    stem = f"{args.job_file.parent.name}_{args.job_file.stem}"
    out_path = args.output / f"{stem}_milp.json"

    scene = dict(SCENES[args.scene])
    if args.processing_times:
        scene["processing_times"] = tuple(
            float(x) for x in args.processing_times.split(","))
    if args.machine_counts:
        scene["machine_counts"] = tuple(
            int(x) for x in args.machine_counts.split(","))
    if args.capacities:
        scene["capacities"] = tuple(
            float(x) for x in args.capacities.split(","))
    requested_scene = {
        "processing_times": list(scene["processing_times"]),
        "machine_counts": list(scene["machine_counts"]),
        "capacities": list(scene["capacities"]),
    }

    # 断点续跑：输出文件已存在、含有效可行解（objective 非空）、且既有求解时限
    # 不低于本次请求（避免低时限次优结果被当作已完成而跳过更高时限的重算）。
    if not args.force and out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if (existing.get("schema_version") == 2
                    and existing.get("job_file_sha256") == file_sha256(args.job_file)
                    and existing.get("scene") == requested_scene
                    and existing.get("k_slots") == (
                        args.k_slots if args.k_slots is not None
                        else existing.get("n_jobs"))
                    and existing.get("solver_seed") == args.solver_seed
                    and existing.get("mipstart_enabled") == (not args.no_mipstart)
                    and float(existing.get(
                        "requested_relative_mip_gap", 1.0)) <= args.mip_gap
                    and existing.get("objective") is not None
                    and float(existing.get("time_limit", 0)) >= args.time_limit):
                print(json.dumps({
                    "skipped": True, "reason": "already solved",
                    "status": existing.get("status_name"),
                    "objective_TT": existing.get("objective"),
                    "existing_time_limit": existing.get("time_limit"),
                    "output": str(out_path),
                }, ensure_ascii=False, indent=2))
                return
        except (json.JSONDecodeError, OSError, TypeError):
            pass  # 输出文件损坏/不完整则重新求解

    jobs = load_jobs(args.job_file)

    result = build_and_solve(
        jobs, time_limit=args.time_limit,
        k_slots=args.k_slots, verbose=not args.quiet, threads=args.threads,
        mip_gap=args.mip_gap, solver_seed=args.solver_seed,
        scene=scene, mipstart=not args.no_mipstart)
    result["job_file"] = str(args.job_file)
    result["job_file_sha256"] = file_sha256(args.job_file)
    result["scene_name"] = args.scene

    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    # job 级明细 CSV（拖期/机器/槽位等，论文表格与仿真回放用）
    if result.get("job_details") is not None:
        jobs_path = args.output / f"{stem}_milp_jobs.csv"
        with jobs_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "job_id", "arrival_time", "family", "size", "due_time",
                "completion_time", "tardiness", "machine_id", "slot"])
            writer.writeheader()
            for d in result["job_details"]:
                writer.writerow(d)
        print(f"jobs_csv: {jobs_path}")

    # 调度 CSV 导出（论文展示 + 仿真执行用）
    if result.get("schedule") is not None:
        csv_path = args.output / f"{stem}_milp_schedule.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "machine_id", "machine_class", "slot", "content",
                "family", "start", "job_ids"])
            writer.writeheader()
            for e in sorted(result["schedule"], key=lambda x: (x["machine_id"], x["slot"])):
                content = f"BATCH-f{e['family']}" if e["family"] is not None else ""
                writer.writerow({
                    "machine_id": e["machine_id"],
                    "machine_class": e["machine_class"],
                    "slot": e["slot"], "content": content,
                    "family": e["family"], "start": e["start"],
                    "job_ids": ";".join(map(str, e["job_ids"])),
                })
        print(f"schedule_csv: {csv_path}")

    print(json.dumps({
        "status": result["status_name"],
        "is_optimal": result.get("is_optimal", False),
        "objective_TT": result["objective"],
        "obj_bound": result["obj_bound"],
        "mip_gap": result["mip_gap"],
        "is_proven_global_production_optimum": result.get(
            "is_proven_global_production_optimum", False),
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
        "n_jobs": result["n_jobs"], "n_machines": result["n_machines"],
        "n_batches": result.get("n_batches"),
        "makespan": result.get("makespan"),
        "n_tardy_jobs": result.get("n_tardy_jobs"),
        "output": str(out_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
