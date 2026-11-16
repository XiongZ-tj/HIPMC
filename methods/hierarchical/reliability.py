"Uninterrupted batch-completion probability under state-dependent degradation and failure."

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


COMPLETION_PROBABILITY_PROTOCOL = "work_domain_multistate_failure_survival"


def _vector(values, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector.")
    return tuple(float(x) for x in array)


def uninterrupted_completion_probabilities(
        remaining_work: float,
        processing_rates,
        degradation_rates,
        unexpected_failure_rates) -> tuple[float, ...]:
    """Return survival from every initial health state (healthy first).

    ``remaining_work`` has nominal-time units. Processing speeds must be
    positive; degradation and total unexpected-failure rates nonnegative.
    The final degradation rate leads to terminal failure, not a fifth
    operational state. Returned tuples cannot corrupt cached results.
    """

    work = float(remaining_work)
    if not np.isfinite(work) or work < 0:
        raise ValueError("Remaining nominal work must be finite and nonnegative.")
    speeds = _vector(processing_rates, "Processing rates")
    degradation = _vector(degradation_rates, "Degradation rates")
    failures = _vector(unexpected_failure_rates, "Unexpected failure rates")
    if not len(speeds) == len(degradation) == len(failures):
        raise ValueError("Every operational state needs speed and both hazard rates.")
    if min(speeds) <= 0 or min(degradation) < 0 or min(failures) < 0:
        raise ValueError("Speeds must be positive and hazard rates nonnegative.")
    return _cached_survival(work, speeds, degradation, failures)


@lru_cache(maxsize=4096)
def _cached_survival(
        work: float,
        speeds: tuple[float, ...],
        degradation: tuple[float, ...],
        failures: tuple[float, ...]) -> tuple[float, ...]:
    # Key every physical parameter and exact work value. No instance identity,
    # rounded work, random stream, or implicit mutable configuration is cached.
    count = len(speeds)
    if work == 0:
        return (1.0,) * count
    rho = np.asarray(speeds, dtype=np.float64)
    alpha = np.asarray(degradation, dtype=np.float64)
    lam = np.asarray(failures, dtype=np.float64)
    generator = np.diag(-(alpha + lam) / rho)
    if count > 1:
        indices = np.arange(count - 1)
        generator[indices, indices + 1] = alpha[:-1] / rho[:-1]
    scaled = generator * work
    if not np.isfinite(scaled).all():
        raise ValueError("Work-domain generator exceeds finite numeric range.")
    with torch.no_grad():
        matrix = torch.as_tensor(scaled, dtype=torch.float64, device="cpu")
        probability = torch.linalg.matrix_exp(matrix).sum(dim=1).numpy()
    if (not np.isfinite(probability).all()
            or np.any(probability < -1e-10)
            or np.any(probability > 1.0 + 1e-10)):
        raise ArithmeticError("Multistate completion probability violates [0, 1].")
    # Clip roundoff only, never materially invalid probabilities.
    return tuple(float(x) for x in np.clip(probability, 0.0, 1.0))
