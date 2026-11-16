"Common experimental budgets, seeds and hyperparameter grid."
TRAINING_SEEDS = (101, 202, 303)
LEARNED_METHODS = ('ours_categorical_baa', 'ddqnb', 'psa_mappo', 'hdrl_option')
STANDARD_EPISODES = 500
EXTENDED_EPISODES = 3000
FINETUNING_EPISODES = 200
VALIDATION_INSTANCES = 10
VALIDATION_INTERVAL = 10
CHECKPOINT_INTERVAL = 25
TEST_INSTANCES_PER_CONDITION = 10
# Original stored manifest size, independent of the formal evaluation subset.
FROZEN_TEST_INSTANCES_PER_CONDITION = 20
SIMULATION_REPLICATIONS = 10
LEARNING_RATES = (5e-5, 1e-4, 2e-4)
CLIP_RATIOS = (0.20,)
DISCOUNT_FACTORS = (0.90, 0.95, 0.99, 1.00)
DISCOUNT_PROTOCOL = "fixed_per_decision_transition"
# Select from validation evidence later; do not silently freeze existing defaults.
MAIN_HYPERPARAMETERS = None
