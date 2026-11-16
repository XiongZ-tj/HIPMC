"Shared instance, seed and output utilities for empty-queue maintenance."

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from experiment.config import ConditionConfig
from experiment.provenance import source_fingerprint
from methods.hierarchical.config import HierarchyConfig
from .environment import IdleMaintenanceEnv

WORKSPACE = Path(__file__).resolve().parents[1]
from experiment.revision_protocol import TRAINING_SEEDS
DEFAULT_SCALE = WORKSPACE / "reward_scales/offline_calibrated_sdr_reward_scale_base_finite_load100_jobs150_due2p5_deg100.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def variant_source_fingerprint():
    digest = hashlib.sha256(source_fingerprint(WORKSPACE).encode("ascii"))
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if "__pycache__" not in path.parts:
            digest.update(path.relative_to(WORKSPACE).as_posix().encode("utf-8"))
            digest.update(sha(path).encode("ascii"))
    return digest.hexdigest()


def training_condition(seed=101):
    return ConditionConfig(target_load=1., initial_jobs=30, total_jobs=150,
                           due_date_factor=2.5, degradation_multiplier=1., seed=seed)


class AuditCsvStream:
    """Stream detailed tables without keeping all 5400 episodes in memory."""
    def __init__(self, path, identity_fields, detail_fields):
        self.stream = Path(path).open("w", encoding="utf-8-sig", newline="")
        self.fields = list(dict.fromkeys([*identity_fields, *detail_fields, "extra_fields_json"]))
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fields)
        self.writer.writeheader()

    def write(self, row):
        selected = {k:(json.dumps(v) if isinstance(v,(list,tuple,dict)) else v)
                    for k,v in row.items() if k in self.fields}
        extras = {k:v for k,v in row.items() if k not in self.fields}
        selected["extra_fields_json"] = json.dumps(extras, ensure_ascii=False)
        self.writer.writerow(selected)

    def close(self):
        self.stream.close()


def build_environment(protocol, condition, *, device, scale=None, scale_sha=None,hyperparameters=None):
    config = replace(HierarchyConfig(), gamma_baa=.95, gamma_mma=.95,
                     degradation_multiplier=condition.degradation_multiplier,
                     calibrated_reward_scale=scale, reward_scale_artifact_sha256=scale_sha)
    hp=hyperparameters or {}
    changes={}
    if 'learning_rate' in hp:changes['learning_rate']=hp['learning_rate']
    if 'clip_ratio' in hp:changes['ppo_clip_ratio']=hp['clip_ratio']
    if 'gamma' in hp:changes.update(gamma_baa=hp['gamma'],gamma_mma=hp['gamma'])
    config=replace(config,**changes)
    return IdleMaintenanceEnv(condition.as_job_parameters(), [2,2,1],
        processing_times=(190,158,202,128,104), config=config, device=device,
        maintenance_protocol=protocol)
