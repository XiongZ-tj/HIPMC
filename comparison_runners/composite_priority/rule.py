"""Non-learning priority on the exact HIPMC candidate pool and features."""
import json
from pathlib import Path
import numpy as np
from methods.hierarchical.candidates import build_proposals, BAA_CRITERIA
from methods.hierarchical.environment import HierarchicalSchedulingEnv
DEFAULT_WEIGHTS_FILE=Path(__file__).resolve().parents[2]/'experiment_manifests/tuned_composite_priority_weights.json'


def load_weights(path=DEFAULT_WEIGHTS_FILE):
    return np.asarray(json.loads(Path(path).read_text(encoding='utf-8'))['weights'],dtype=float)


def select_candidate(features, weights, job_ids):
    scores=np.asarray(features)@np.asarray(weights)
    return min(range(len(scores)),key=lambda i:(-float(scores[i]),tuple(sorted(job_ids[i]))))


class Controller:
    def __init__(self, weights):
        self.weights=np.asarray(weights,dtype=float)
        self.proposals=[]

    def begin_episode(self, policy_seed):
        self.proposals=[]

    def build_proposals(self, jobs, capacity, clock, processing_times, **kwargs):
        self.proposals=build_proposals(jobs,capacity,clock,processing_times)
        return self.proposals

    def select_action(self, observation, *, machine_state, retained_batch=False, **kwargs):
        count=len(observation.proposal_features)
        if retained_batch:return 0
        if machine_state>=3:return count+1
        if machine_state>=2:return count
        return select_candidate(observation.proposal_features,self.weights,
                                [p.job_ids for p in self.proposals])

    def observe(self, transition):
        pass


class Environment(HierarchicalSchedulingEnv):
    def _control_proposal_features(self, machine, proposal):
        return self._proposal_features(machine,proposal)


def weight_artifact(weights, **metadata):
    return dict(method='tuned_composite_batch_priority_with_health_state_threshold_maintenance',
        feature_names=list(BAA_CRITERIA),weights=list(map(float,weights)),
        batch_formation='CH',maintenance_policy='health_state_threshold',
        score_direction='maximize',tie_break='sorted_job_id_lexicographic',**metadata)
