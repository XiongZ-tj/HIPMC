"""Change within-family packing, reuse canonical batch feature calculations."""
from methods.baselines.sdr.rules import form_batches
from methods.hierarchical.candidates import build_proposals


def formation_proposals(jobs,capacity,clock,processing_times,rule):
    batches=form_batches(jobs,capacity,clock,processing_times,rule)
    selected={j for _,ids in batches for j in ids}
    # Each selected family already fits. Canonical CH cannot change membership
    # when repacking it, and recomputes all criteria on the actual batch pool.
    return build_proposals([j for j in jobs if j.job_id in selected],
                           capacity,clock,processing_times)
