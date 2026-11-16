"""Common observed-state encoding for the three literature learning baselines."""
import numpy as np
from methods.hierarchical.candidates import build_proposals
from methods.hierarchical.types import JointObservation

PROPOSAL_DIM = 6
# Physical descriptor, shop context, maintenance descriptor, candidate summary.
MACHINE_DIM = 19 + 16 + 5 + 6


def fixed_mask(observation):
    count = len(observation.proposal_features)
    mask = np.zeros(5, dtype=bool)
    mask[:min(count, 3)] = observation.action_mask[:min(count, 3)]
    mask[3:] = observation.action_mask[count:count + 2]
    return mask


def observation(env, requester, proposals, mask):
    pool = build_proposals(env._available_jobs(), requester.capacity,
                           env.clock, env.job_cluster_pt)
    by_jobs = {p.job_ids: p for p in pool}
    candidates = []
    for proposal in proposals:
        source = by_jobs.get(proposal.job_ids)
        if source is None:
            jobs = [env.jobs[j] for j in proposal.job_ids]
            source = build_proposals(jobs, requester.capacity, env.clock,
                                     env.job_cluster_pt)[0]
            # Keep the same efficiency reference as the complete current pool.
            if pool:
                reference = max(sum(env.jobs[j].size for j in p.job_ids)
                                / env.job_cluster_pt[p.family] for p in pool)
                source.base_criteria[4] = min(1., sum(j.size for j in jobs)
                    / env.job_cluster_pt[source.family] / reference)
        candidates.append(env._proposal_features(requester, source))
    features = np.stack(candidates)
    context = env._global_context_features()
    machines = []
    for machine in env.machines:
        maintenance = np.mean([env._maintenance_features(machine, p)
                               for p in proposals], axis=0)
        machines.append(np.concatenate((env._policy_machine_feature(
            machine, requester_id=requester.machine_id), context,
            maintenance, features.mean(axis=0))))
    return JointObservation(features, np.asarray(machines, dtype=np.float32),
                            requester.machine_id, mask)
