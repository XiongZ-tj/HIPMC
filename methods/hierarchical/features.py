"""Bounded, semantically normalized feature contracts for the proposed method."""

from __future__ import annotations

import numpy as np


FEATURE_SCHEMA_VERSION = 5

# A declared size unit, not an estimate from the complete current instance.
REFERENCE_JOB_SIZE = 20.0

POLICY_MACHINE_FEATURES = (
    "health_healthy",
    "health_minor",
    "health_moderate",
    "health_severe",
    "phase_idle",
    "phase_processing",
    "phase_maintaining",
    "phase_repairing",
    "remaining_duration_saturation",
    "capacity_minmax",
    "class_0",
    "class_1",
    "class_2",
    "reserved_batch_fill",
    "historical_processing_utilization",
    "historical_maintenance_utilization",
    "breakdown_burden_saturation",
    "processing_interruption_probability",
    "is_requester",
)

GLOBAL_CONTEXT_FEATURES = (
    "available_job_count_pressure",
    "available_processing_work_pressure",
    "allocated_capacity_fraction",
    "overdue_available_fraction",
    "urgent_not_overdue_available_fraction",
    "mean_available_urgency",
    "maximum_available_urgency",
    "mean_waiting_saturation",
    "maximum_waiting_saturation",
    "idle_machine_fraction",
    "processing_machine_fraction",
    "maintaining_machine_fraction",
    "repairing_machine_fraction",
    "eligible_family_fraction",
    "capacity_weighted_health_severity",
    "severe_health_capacity_fraction",
)

MMA_MAINTENANCE_FEATURES = (
    "health_severity",
    "processing_interruption_probability",
    "ipm_duration_saturation",
    "pm_duration_saturation",
    "ipm_one_step_recovery_probability",
)


def bounded_fraction(numerator: float, denominator: float) -> float:
    """Normalize a fraction without runtime assertions."""

    numerator = float(numerator)
    denominator = float(denominator)
    value = numerator / denominator
    return float(np.clip(value, 0.0, 1.0))


def positive_saturation(value: float, scale: float) -> float:
    """Map a nonnegative quantity monotonically to ``[0, 1)``."""

    value = float(value)
    scale = float(scale)
    return value / (value + scale)


def checked_unit_features(
        name: str,
        values,
        *,
        expected_dim: int | None = None) -> np.ndarray:
    "Convert and clip features; retain the name for existing imports."

    array = np.asarray(values, dtype=np.float32)
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)
