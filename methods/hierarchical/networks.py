"""Heterogeneous MAPPO actors and a centralized global-state critic."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .types import BAAObservation, MMAObservation
from .features import GLOBAL_CONTEXT_FEATURES


def _tensor(array, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.unsqueeze(-1)
    count = valid.sum(dim=1).clamp_min(1)
    mean = (tokens * valid).sum(dim=1) / count
    maximum = tokens.masked_fill(~valid, -torch.inf).max(dim=1).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return mean, maximum


def _pad_baa(observations: list[BAAObservation], device: torch.device):
    batch = len(observations)
    max_proposals = max(item.proposal_features.shape[0] for item in observations)
    max_machines = max(item.machine_features.shape[0] for item in observations)
    proposal_dim = observations[0].proposal_features.shape[1]
    machine_dim = observations[0].machine_features.shape[1]
    proposals = np.zeros((batch, max_proposals, proposal_dim), dtype=np.float32)
    machines = np.zeros((batch, max_machines, machine_dim), dtype=np.float32)
    proposal_mask = np.zeros((batch, max_proposals), dtype=np.bool_)
    machine_mask = np.zeros((batch, max_machines), dtype=np.bool_)
    requester = np.zeros(batch, dtype=np.int64)
    contexts = np.stack([item.context_features for item in observations])
    for index, item in enumerate(observations):
        proposal_count = item.proposal_features.shape[0]
        machine_count = item.machine_features.shape[0]
        proposals[index, :proposal_count] = item.proposal_features
        machines[index, :machine_count] = item.machine_features
        proposal_mask[index, :proposal_count] = item.proposal_mask
        machine_mask[index, :machine_count] = item.machine_mask
        requester[index] = item.requester_index
    return (
        _tensor(proposals, device),
        _tensor(machines, device),
        torch.as_tensor(proposal_mask, dtype=torch.bool, device=device),
        torch.as_tensor(machine_mask, dtype=torch.bool, device=device),
        torch.as_tensor(requester, dtype=torch.long, device=device),
        _tensor(contexts, device),
    )


class BAAActor(nn.Module):
    """Per-candidate scoring actor: one logit per batch proposal.

    The batch-assignment policy scores every candidate with the same
    three-tower encoder plus a scalar scoring head (proposal token, requester
    machine token, and global context), producing a categorical distribution
    over the proposals.  This mirrors the centralized PPO proposal head while
    keeping the BAA layer's per-requester local decision structure.
    """

    def __init__(
            self,
            proposal_dim: int = 6,
            machine_dim: int = 19,
            context_dim: int = len(GLOBAL_CONTEXT_FEATURES),
            hidden_dim: int = 64) -> None:
        super().__init__()
        self.proposal_encoder = nn.Sequential(
            nn.Linear(proposal_dim, hidden_dim), nn.Tanh())
        self.machine_encoder = nn.Sequential(
            nn.Linear(machine_dim, hidden_dim), nn.Tanh())
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.Tanh())
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_batch(
            self,
            observations: list[BAAObservation],
            device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-candidate logits (B, max_K) and their boolean mask."""
        proposals, machines, proposal_mask, _machine_mask, requester, contexts = (
            _pad_baa(observations, device))
        proposal_tokens = self.proposal_encoder(proposals)
        machine_tokens = self.machine_encoder(machines)
        indices = torch.arange(len(observations), device=device)
        requester_token = machine_tokens[indices, requester]
        context_token = self.context_encoder(contexts)
        max_candidates = proposal_tokens.shape[1]
        requester_expanded = requester_token[:, None, :].expand(
            -1, max_candidates, -1)
        context_expanded = context_token[:, None, :].expand(
            -1, max_candidates, -1)
        scores = self.score_head(torch.cat((
            proposal_tokens,
            requester_expanded,
            context_expanded,
        ), dim=-1)).squeeze(-1)
        logits = scores.masked_fill(~proposal_mask, -torch.inf)
        return logits, proposal_mask

    def forward_observation(
            self,
            observation: BAAObservation,
            device: torch.device) -> torch.Tensor:
        return self.forward_batch([observation], device)[0][0]


class MMAActor(nn.Module):
    """A machine-class-specific categorical actor over Process/IPM/PM."""

    def __init__(
            self,
            batch_dim: int = 6,
            machine_dim: int = 19,
            context_dim: int = len(GLOBAL_CONTEXT_FEATURES),
            maintenance_dim: int = 5,
            hidden_dim: int = 64) -> None:
        super().__init__()
        self.batch_encoder = nn.Sequential(
            nn.Linear(batch_dim, hidden_dim), nn.Tanh())
        self.machine_encoder = nn.Sequential(
            nn.Linear(machine_dim, hidden_dim), nn.Tanh())
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.Tanh())
        self.maintenance_encoder = nn.Sequential(
            nn.Linear(maintenance_dim, hidden_dim), nn.Tanh())
        self.model = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

    def forward_batch(
            self,
            observations: list[MMAObservation],
            device: torch.device) -> torch.Tensor:
        batches = torch.stack([
            _tensor(item.batch_features, device) for item in observations])
        requesters = torch.stack([
            _tensor(item.machine_features, device) for item in observations])
        contexts = torch.stack([
            _tensor(item.context_features, device) for item in observations])
        maintenance = torch.stack([
            _tensor(item.maintenance_features, device) for item in observations])
        global_machines = [
            _tensor(item.global_machine_features, device) for item in observations]
        machine_tokens = [self.machine_encoder(item) for item in global_machines]
        machine_means = torch.stack([item.mean(dim=0) for item in machine_tokens])
        machine_maxima = torch.stack([item.max(dim=0).values for item in machine_tokens])
        features = torch.cat((
            self.batch_encoder(batches),
            self.machine_encoder(requesters),
            machine_means,
            machine_maxima,
            self.context_encoder(contexts),
            self.maintenance_encoder(maintenance),
        ), dim=-1)
        logits = self.model(features)
        mask = torch.as_tensor(np.stack([
            item.action_mask for item in observations
        ]), dtype=torch.bool, device=device)
        return logits.masked_fill(~mask, -torch.inf)

    def forward_observation(
            self,
            observation: MMAObservation,
            device: torch.device) -> torch.Tensor:
        return self.forward_batch([observation], device)[0]


class CentralCritic(nn.Module):
    """Central critic with one shared encoder and four agent-type value heads."""

    def __init__(
            self,
            proposal_dim: int = 6,
            machine_dim: int = 19,
            context_dim: int = len(GLOBAL_CONTEXT_FEATURES),
            maintenance_dim: int = 5,
            hidden_dim: int = 64,
            head_count: int = 4) -> None:
        super().__init__()
        self.maintenance_dim = int(maintenance_dim)
        summary_dim = (
            proposal_dim * 2 + machine_dim * 3 + context_dim + maintenance_dim)
        self.encoder = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(head_count)
        ])

    def _summary(self, observation, device: torch.device) -> torch.Tensor:
        if isinstance(observation, BAAObservation):
            proposals = _tensor(observation.proposal_features, device)
            valid = torch.as_tensor(
                observation.proposal_mask, dtype=torch.bool, device=device)
            proposals = proposals[valid]
            machines = _tensor(observation.machine_features, device)
            requester = machines[observation.requester_index]
            context = _tensor(observation.context_features, device)
            maintenance = torch.zeros(
                self.maintenance_dim, dtype=torch.float32, device=device)
        elif isinstance(observation, MMAObservation):
            batch = _tensor(observation.batch_features, device)
            proposals = batch[None, :]
            machines = _tensor(observation.global_machine_features, device)
            requester = machines[observation.requester_index]
            context = _tensor(observation.context_features, device)
            maintenance = _tensor(observation.maintenance_features, device)
        else:
            raise TypeError(f"Unsupported critic observation: {type(observation)!r}")
        return torch.cat((
            proposals.mean(dim=0),
            proposals.max(dim=0).values,
            machines.mean(dim=0),
            machines.max(dim=0).values,
            requester,
            context,
            maintenance,
        ))

    def forward_batch(
            self,
            observations: list[BAAObservation | MMAObservation],
            head_ids: list[int] | np.ndarray | torch.Tensor,
            device: torch.device) -> torch.Tensor:
        summaries = torch.stack([
            self._summary(item, device) for item in observations])
        encoded = self.encoder(summaries)
        ids = torch.as_tensor(head_ids, dtype=torch.long, device=device)
        values = torch.empty(len(observations), dtype=torch.float32, device=device)
        for head_id, head in enumerate(self.heads):
            positions = torch.nonzero(ids == head_id, as_tuple=False).flatten()
            if positions.numel():
                values[positions] = head(encoded[positions]).squeeze(-1)
        return values

    def forward_observation(
            self,
            observation: BAAObservation | MMAObservation,
            head_id: int,
            device: torch.device) -> torch.Tensor:
        return self.forward_batch([observation], [head_id], device)[0]
