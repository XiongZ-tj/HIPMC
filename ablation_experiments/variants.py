"Identities and component settings for nine frozen-policy replacements."

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from methods.hierarchical.types import HandshakeMode


class AblationVariant(str, Enum):
    MMU_BATCH_SELECTION = "mmu_batch_selection"
    EDD_B_BATCH_SELECTION = "eddb_batch_selection"
    MAST_BATCH_SELECTION = "mast_batch_selection"
    PERIODIC_MAINTENANCE_POLICY = "periodic_maintenance_policy"
    HEALTH_STATE_THRESHOLD_MAINTENANCE_POLICY = (
        "health_state_threshold_maintenance_policy")
    COMPOSITE_BATCH_SELECTION = "tuned_composite_batch_selection"
    EDD_JOB_BATCH_FORMATION = "edd_job_batch_formation"
    FFD_BATCH_FORMATION = "ffd_batch_formation"
    CORRECTIVE_ONLY_MAINTENANCE = "corrective_only_maintenance_policy"


ABLATION_VARIANTS = tuple(item.value for item in AblationVariant)

# SDR names match the shared rules; Composite and CM are local replacements.
BAA_RULES = (None, "MMU", "EDD-B", "MAST", "Composite")
MMA_RULES = (None, "Period", "Threshold", "CM")


@dataclass(frozen=True)
class VariantSpec:
    variant: AblationVariant
    publication_name: str
    target_component: str
    handshake_mode: HandshakeMode
    baa_policy: str
    mma_policy: str
    baa_rule: str | None
    mma_rule: str | None
    formation_rule: str = "CH"


_SPECS = {
    AblationVariant.MMU_BATCH_SELECTION: VariantSpec(
        variant=AblationVariant.MMU_BATCH_SELECTION,
        publication_name="maximum-utilization-based batch selection",
        target_component="baa_batch_evaluation",
        handshake_mode=HandshakeMode.INTERACTIVE,
        baa_policy="maximum_machine_utilization_rule",
        mma_policy="learned_masked_categorical_process_maintenance_policy",
        baa_rule="MMU",
        mma_rule=None,
    ),
    AblationVariant.EDD_B_BATCH_SELECTION: VariantSpec(
        variant=AblationVariant.EDD_B_BATCH_SELECTION,
        publication_name="earliest-due-date batch selection",
        target_component="baa_batch_evaluation",
        handshake_mode=HandshakeMode.INTERACTIVE,
        baa_policy="earliest_due_date_batch_rule",
        mma_policy="learned_masked_categorical_process_maintenance_policy",
        baa_rule="EDD-B",
        mma_rule=None,
    ),
    AblationVariant.MAST_BATCH_SELECTION: VariantSpec(
        variant=AblationVariant.MAST_BATCH_SELECTION,
        publication_name="minimum-average-slack batch selection",
        target_component="baa_batch_evaluation",
        handshake_mode=HandshakeMode.INTERACTIVE,
        baa_policy="minimum_average_slack_batch_rule",
        mma_policy="learned_masked_categorical_process_maintenance_policy",
        baa_rule="MAST",
        mma_rule=None,
    ),
    AblationVariant.PERIODIC_MAINTENANCE_POLICY: VariantSpec(
        variant=AblationVariant.PERIODIC_MAINTENANCE_POLICY,
        publication_name="periodic-maintenance policy",
        target_component="mma_process_maintenance_policy",
        handshake_mode=HandshakeMode.INTERACTIVE,
        baa_policy="learned_categorical_batch_scoring",
        mma_policy="periodic_ipm_policy",
        baa_rule=None,
        mma_rule="Period",
    ),
    AblationVariant.HEALTH_STATE_THRESHOLD_MAINTENANCE_POLICY: VariantSpec(
        variant=AblationVariant.HEALTH_STATE_THRESHOLD_MAINTENANCE_POLICY,
        publication_name="health-state-threshold maintenance policy",
        target_component="mma_process_maintenance_policy",
        handshake_mode=HandshakeMode.INTERACTIVE,
        baa_policy="learned_categorical_batch_scoring",
        mma_policy="health_state_threshold_maintenance_policy",
        baa_rule=None,
        mma_rule="Threshold",
    ),
}

for variant,name,component,baa_rule,mma_rule,formation in (
    (AblationVariant.COMPOSITE_BATCH_SELECTION,
     "tuned composite batch-priority selection","baa_batch_evaluation","Composite",None,"CH"),
    (AblationVariant.EDD_JOB_BATCH_FORMATION,
     "earliest-due-date job-based batch formation","batch_formation",None,None,"EDD-J"),
    (AblationVariant.FFD_BATCH_FORMATION,
     "first-fit-decreasing batch formation","batch_formation",None,None,"FFD"),
    (AblationVariant.CORRECTIVE_ONLY_MAINTENANCE,
     "corrective-only maintenance policy","mma_process_maintenance_policy",None,"CM","CH"),
):
    _SPECS[variant]=VariantSpec(variant,name,component,HandshakeMode.INTERACTIVE,
        "tuned_composite_batch_priority" if baa_rule else "learned_categorical_batch_scoring",
        "corrective_only_maintenance" if mma_rule else "learned_masked_categorical_process_maintenance_policy",
        baa_rule,mma_rule,formation)


def variant_spec(variant: str | AblationVariant) -> VariantSpec:
    key = AblationVariant(variant)
    try:
        return _SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown ablation variant: {variant}") from exc
