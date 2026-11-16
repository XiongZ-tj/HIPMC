"""Four-action MMA policies without modifying the standard HIPMC actors."""

import numpy as np
import torch
from torch import nn

from methods.hierarchical.agents import HierarchicalMAPPO


WAIT = 3  # Existing Process=0, IPM=1 and PM=2 semantics remain unchanged.


class IdleMaintenanceMAPPO(HierarchicalMAPPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for actor in self.mma_actors:
            previous = actor.model[-1]
            actor.model[-1] = nn.Linear(previous.in_features, 4).to(self.device)
        # Rebuild before any learning; optimizers must own the replacement heads.
        self.optimizer = torch.optim.Adam(
            [*self.baa_actor.parameters(), *self.mma_actors.parameters(),
             *self.critic.parameters()], lr=kwargs["learning_rate"])

    def _rollout_diagnostic(self, head_id, transitions, raw_advantages,
                            normalized_advantages, returns):
        result = super()._rollout_diagnostic(
            head_id, transitions, raw_advantages, normalized_advantages, returns)
        if head_id:
            for health in range(self.health_state_count):
                positions = [i for i, tr in enumerate(transitions)
                             if tr.state.health_state == health and int(tr.action) == WAIT]
                prefix = f"health_{health}_action_{WAIT}"
                result[f"{prefix}_count"] = len(positions)
                result[f"{prefix}_reward_mean"] = (
                    float(np.mean([transitions[i].reward for i in positions])) if positions else 0.)
                result[f"{prefix}_elapsed_time_mean"] = (
                    float(np.mean([transitions[i].elapsed_time for i in positions])) if positions else 0.)
        return result
