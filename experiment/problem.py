"""Shared physical and instance parameters for every comparative method."""

JOB_SIZE_INTERVAL = (10, 30)
MACHINE_CAPACITIES = (60, 80, 100)
DEGRADATION_STATES = 4
FAILURE_TYPES = 3
DEGRADATION_RATE_RANGES = (
    (0.001, 0.005),
    (0.001, 0.003),
    (0.001, 0.002),
)
UNEXPECTED_FAILURE_RATE_RANGES = (
    (0.0001, 0.0003),
    (0.0001, 0.0003),
    (0.0001, 0.0003),
)
TERMINAL_DEGRADATION_RATES = (0.005, 0.003, 0.002)
FAILURE_TYPE_PROPORTIONS = (0.6, 0.3, 0.1)
# State-dependent IPM restoration rates beta[c][s - 1] (1/time unit).
# Health state 0 is as-good-as-new and therefore has no restoration rate.
# The rows correspond to machine classes and the columns to degraded states
# s=1, 2, 3.  Rates are calibrated so that an isolated IPM has one-step
# departure probabilities (0.75, 0.70, 0.35), respectively.
IPM_RECOVERY_RATES = (
    (0.06931471805599453, 0.06019864021629608, 0.02153914580462271),
    (0.046209812037329684, 0.04013242681086405, 0.014359430536415141),
    (0.03465735902799726, 0.03009932010814804, 0.010769572902311356),
)
IPM_TIMES = (20, 30, 40)
PM_TIMES = (60, 90, 120)
CORRECTIVE_TIME_RANGES = ((20, 40), (40, 60), (60, 80))
MINIMAL_REPAIR_TIMES = (80, 120, 160)
PRODUCTION_RATE_RANGE = (0.9, 1.0)
