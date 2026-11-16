"""Independent, reproducible random streams for training episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingSeeds:
    episode: int
    instance_seed: int
    simulation_seed: int
    policy_seed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingSeedProtocol:
    master_seed: int

    def __post_init__(self) -> None:
        if not 0 <= self.master_seed <= 2**32 - 1:
            raise ValueError("master_seed must be in [0, 2**32 - 1].")

    def _stream(self, episode: int, stream: int) -> int:
        if episode < 0:
            raise ValueError("episode must be nonnegative.")
        return int(np.random.SeedSequence(
            [self.master_seed, episode, stream]
        ).generate_state(1, dtype=np.uint32)[0])

    def for_episode(self, episode: int) -> TrainingSeeds:
        return TrainingSeeds(
            episode=episode,
            instance_seed=self._stream(episode, 0),
            simulation_seed=self._stream(episode, 1),
            policy_seed=self._stream(episode, 2),
        )
