"""Frozen full-factorial hyperparameter grid and naming contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


from experiment.revision_protocol import LEARNING_RATES,CLIP_RATIOS,DISCOUNT_FACTORS
EXPECTED_CONFIGURATION_COUNT = 12


def _decimal_id(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", "p")


def _learning_rate_id(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent):+03d}"


@dataclass(frozen=True)
class SensitivityConfiguration:
    "One full-factorial configuration."

    learning_rate: float
    clip_ratio: float
    discount_factor: float

    @property
    def configuration_id(self) -> str:
        return (
            f"lr-{_learning_rate_id(self.learning_rate)}_"
            f"clip-{_decimal_id(self.clip_ratio, 2)}_"
            f"gamma-{_decimal_id(self.discount_factor, 2)}"
        )

    def to_dict(self) -> dict:
        return {
            "configuration_id": self.configuration_id,
            **asdict(self),
            "gamma_baa": self.discount_factor,
            "gamma_mma": self.discount_factor,
        }


def full_factorial_grid() -> tuple[SensitivityConfiguration, ...]:
    grid = tuple(
        SensitivityConfiguration(learning_rate, clip_ratio, discount_factor)
        for learning_rate, clip_ratio, discount_factor in product(
            LEARNING_RATES, CLIP_RATIOS, DISCOUNT_FACTORS
        )
    )
    identifiers = [configuration.configuration_id for configuration in grid]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Sensitivity configuration identifiers are not unique.")
    return grid


def select_configurations(
        identifiers: list[str] | tuple[str, ...] | None,
) -> tuple[SensitivityConfiguration, ...]:
    grid = full_factorial_grid()
    if not identifiers:
        return grid
    requested = tuple(identifiers)
    if len(set(requested)) != len(requested):
        raise ValueError("--configuration-ids contains duplicates.")
    by_id = {configuration.configuration_id: configuration for configuration in grid}
    unknown = [identifier for identifier in requested if identifier not in by_id]
    if unknown:
        raise ValueError(
            "Unknown sensitivity configuration(s): " + ", ".join(unknown)
        )
    return tuple(by_id[identifier] for identifier in requested)


