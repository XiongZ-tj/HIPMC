"""Shared observation encoders for baselines using the common simulator."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn



def _tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)






def compute_smdp_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        elapsed_times: list[float] | tuple[float, ...],
        terminated: list[bool] | tuple[bool, ...],
        *,
        gamma: float,
        gae_lambda: float,
        time_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Episode-safe GAE with fixed per-transition gamma.

    elapsed_times and time_scale remain accepted for caller compatibility;
    duration metadata is not used to weight returns.
    """

    if not (
            rewards.ndim == values.ndim == next_values.ndim == 1
            and rewards.shape == values.shape == next_values.shape):
        raise ValueError("Reward/value tensors must be aligned vectors.")
    count = rewards.numel()
    if len(elapsed_times) != count or len(terminated) != count:
        raise ValueError("Transition metadata must align with rewards.")
    if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Invalid GAE discount parameters.")

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in reversed(range(count)):
        discount = (
            0.0 if terminated[index]
            else gamma
        )
        delta = (
            rewards[index]
            + discount * next_values[index]
            - values[index]
        )
        # A terminal transition has discount=0, which cuts the trace from a
        # later episode while preserving its own delta for preceding steps in
        # the same episode.
        gae = delta + discount * gae_lambda * gae
        advantages[index] = gae
    return advantages, advantages + values


def pad_joint_observations(observations, device):
    batch = len(observations)
    max_proposals = max(item.proposal_features.shape[0] for item in observations)
    max_machines = max(item.machine_features.shape[0] for item in observations)
    proposal_dim = observations[0].proposal_features.shape[1]
    machine_dim = observations[0].machine_features.shape[1]
    proposals = np.zeros(
        (batch, max_proposals, proposal_dim), dtype=np.float32)
    machines = np.zeros(
        (batch, max_machines, machine_dim), dtype=np.float32)
    machine_mask = np.zeros((batch, max_machines), dtype=np.bool_)
    action_mask = np.zeros((batch, max_proposals + 2), dtype=np.bool_)
    requester = np.zeros(batch, dtype=np.int64)
    for index, observation in enumerate(observations):
        proposal_count = observation.proposal_features.shape[0]
        machine_count = observation.machine_features.shape[0]
        proposals[index, :proposal_count] = observation.proposal_features
        machines[index, :machine_count] = observation.machine_features
        machine_mask[index, :machine_count] = True
        local_actions = observation.action_mask
        if local_actions.size != proposal_count + 2:
            raise ValueError(
                "Joint action mask must contain K process actions plus IPM/PM.")
        action_mask[index, :proposal_count] = local_actions[:proposal_count]
        action_mask[index, max_proposals:] = local_actions[proposal_count:]
        requester[index] = observation.requester_index
    return (
        _tensor(proposals, device),
        _tensor(machines, device),
        torch.as_tensor(machine_mask, dtype=torch.bool, device=device),
        torch.as_tensor(action_mask, dtype=torch.bool, device=device),
        torch.as_tensor(requester, dtype=torch.long, device=device),
    )


class JointActionNetwork(nn.Module):
    def __init__(
            self,
            proposal_dim=13,
            machine_dim=14,
            hidden_dim=64) -> None:
        super().__init__()
        self.proposal_encoder = nn.Sequential(
            nn.Linear(proposal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.machine_encoder = nn.Sequential(
            nn.Linear(machine_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, 4, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.maintenance_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward_context(self, observations, device):
        proposals, machines, machine_mask, action_mask, requester = (
            pad_joint_observations(observations, device))
        proposal_tokens = self.proposal_encoder(proposals)
        machine_tokens = self.machine_encoder(machines)
        context, _ = self.attention(
            proposal_tokens,
            machine_tokens,
            machine_tokens,
            key_padding_mask=~machine_mask,
            need_weights=False,
        )
        indices = torch.arange(len(observations), device=device)
        requester_token = machine_tokens[indices, requester]
        requester_expanded = requester_token[:, None, :].expand(
            -1, proposal_tokens.shape[1], -1)
        joint = torch.cat(
            (proposal_tokens, context, requester_expanded), dim=-1)
        process_logits = self.actor(joint).squeeze(-1)
        valid_machine_count = machine_mask.sum(dim=1, keepdim=True)
        global_machine = (
            machine_tokens * machine_mask[:, :, None]
        ).sum(dim=1) / valid_machine_count
        maintenance_logits = self.maintenance_actor(torch.cat(
            (requester_token, global_machine), dim=-1))
        logits = torch.cat((process_logits, maintenance_logits), dim=1)
        logits = logits.masked_fill(~action_mask, -torch.inf)
        return logits, requester_token, global_machine

    def forward_batch(self, observations, device):
        logits, _, _ = self.forward_context(observations, device)
        return logits

    def forward_observation(self, observation, device):
        logits = self.forward_batch([observation], device)
        return logits[0, :observation.action_mask.size]






def padded_joint_actions(observations, actions, *, device):
    """Map ragged local K+2 action indices to a padded batch tensor."""

    max_proposals = max(
        observation.proposal_features.shape[0]
        for observation in observations)
    mapped = []
    for observation, action in zip(observations, actions, strict=True):
        proposal_count = observation.proposal_features.shape[0]
        value = int(action)
        if value >= proposal_count:
            value = max_proposals + (value - proposal_count)
        mapped.append(value)
    return torch.as_tensor(mapped, dtype=torch.long, device=device)


def observation_summary(observation, device):
    proposal_mean = _tensor(
        observation.proposal_features.mean(axis=0), device)
    machines = _tensor(observation.machine_features, device)
    requester = machines[observation.requester_index]
    machine_mean = machines.mean(dim=0)
    count = torch.as_tensor(
        [observation.proposal_features.shape[0] / 4],
        dtype=torch.float32,
        device=device,
    )
    return torch.cat((proposal_mean, requester, machine_mean, count))
