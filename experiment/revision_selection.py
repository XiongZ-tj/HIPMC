"""Author-selected production-pressure strata; no instance or seed regeneration."""

GENERALIZATION_SCENES = ('S1', 'S2', 'S3', 'S4', 'S5')
COMPONENT_SCENES = ('S0', *GENERALIZATION_SCENES)
REPRESENTATIVE_CONDITION_IDS = (
    'rhoF-0p90_due-3_jobs-100_initial-20',
    'rhoF-1p00_due-2p5_jobs-100_initial-20',
    'rhoF-1p00_due-2p5_jobs-150_initial-30',
    'rhoF-1p10_due-2p5_jobs-150_initial-30',
    'rhoF-1p10_due-2_jobs-200_initial-40',
)


def selected_conditions(condition_ids, scene_id, *, components=False, limit=None):
    """S0 main comparison stays complete; generalization/components use five strata.

    Non-generated domains (e.g. industrial) pass through unchanged. Component
    evaluation includes S0, using the same five strata as S1-S5.
    """
    result=list(condition_ids)
    if components or scene_id in GENERALIZATION_SCENES:
        result=[c for c in result if c in REPRESENTATIVE_CONDITION_IDS]
    return result[:limit]


def include_generated_result(condition_id, scene_id, family):
    if family=='components':
        return scene_id in COMPONENT_SCENES and condition_id in REPRESENTATIVE_CONDITION_IDS
    return scene_id not in GENERALIZATION_SCENES or condition_id in REPRESENTATIVE_CONDITION_IDS
