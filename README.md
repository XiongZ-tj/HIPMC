# HIPMC

Research code and datasets for **Hierarchical interactive production–maintenance coordination for batch scheduling in aerospace heat treatment under machine degradation**.

## Background

Aerospace heat treatment requires dynamically arriving jobs to be grouped into compatible batches and assigned to non-identical parallel-batch furnaces. Machine degradation affects processing efficiency and failure risk, while preventive maintenance causes downtime. Scheduling therefore needs to balance timely order delivery against equipment deterioration.

Hierarchical interactive production–maintenance coordination (HIPMC) combines a constructive batch-formation heuristic, learned candidate batch evaluation, and machine-class-specific maintenance policies. An event-driven simulator represents job arrivals, processing completion, degradation, failures, and maintenance. The optimization objective is total tardiness (TTD).

This repository supports the experiments in the manuscript and Supplementary Material. It includes source code, frozen datasets, experiment configurations, and calibrated reward scales. Trained checkpoints and experimental outputs are generated locally and are not distributed with the repository.

## Project contents

| Directory | Contents |
| --- | --- |
| `methods/` | HIPMC; Double-DQN-based batch scheduling and maintenance (DDQNB); processing-stage-agent MAPPO (PSA-MAPPO); option-based hierarchical deep reinforcement learning (HDRL); and rule-based controllers. |
| `experiment/` | Instance construction, event-driven simulation support, observable features, reward normalization, physical scenarios, seeds, and training records. |
| `comparison_runners/` | Training and evaluation entry points, composite-rule tuning, offline memetic algorithm (MA) planning, and rolling-horizon adaptive large neighborhood search (RH-ALNS). |
| `ablation_experiments/` | Deployment-time replacements of batch selection, maintenance control, and batch formation. |
| `idle_maintenance_experiments/` | Independently trained extension permitting preventive maintenance during empty-queue periods. |
| `params_analysis/` | Hyperparameter search and extended-budget training. |
| `result_analysis/` | Training summaries, paired comparisons, confidence intervals, and computational-time summaries. |
| `MILP/` | Deterministic production-scheduling mixed-integer linear programming (MILP) benchmark for small industrial instances. |
| `datasets/` | Frozen generated and industrial order-based datasets, including small industrial instances. |
| `experiment_manifests/` | Instance selections, selected hyperparameters, composite-rule weights, and result-source configuration. |
| `reward_scales/` | Independently calibrated, fixed reward-normalization scales. |
| `tests/` | Unit tests and small-instance regression checks. |

The non-learning comparisons include 18 dispatching-and-maintenance rule combinations, a tuned composite batch-priority rule with health-state-threshold maintenance, MA, and RH-ALNS.

## Installation

Clone the repository and run all commands from its root directory. Examples below use Windows Command Prompt.

```cmd
git clone https://github.com/XiongZ-tj/HIPMC.git
cd HIPMC
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-analysis.txt
```

`requirements.txt` installs the core versions specified in `requirements-lock.txt`. The analysis dependencies support statistics and plotting. For GPU execution, use a PyTorch build compatible with the installed CUDA environment. `--device auto` selects CUDA when available; `--device cpu` requests CPU execution.

Additional requirements for optimization baselines:

- MA and RH-ALNS require a C++20-capable compiler, such as `g++` or `clang++`, available on `PATH`. Their native backends are compiled when needed.
- The small-instance MILP requires `gurobipy` and a valid Gurobi license.

```cmd
python -m pip install gurobipy
```

## Experimental protocol

The generated benchmark crosses nominal loads of 0.90, 1.00, and 1.10, due-date factors of 2.0, 2.5, and 3.0, and job counts of 100, 150, and 200, giving 27 production conditions. Training uses the condition `(load, due-date factor, jobs) = (1.00, 2.5, 150)`.

- Generated-data evaluation uses the first ten frozen instances per selected condition and ten simulation seeds per instance.
- Learned methods use three training seeds: 101, 202, and 303.
- Standard training uses 500 episodes. Validation uses ten fixed instances, evaluated initially and every ten episodes; checkpoints are selected by validation performance.
- The industrial order-based semi-realistic case contains 89 orders and 60 arrival realizations across three load profiles, with ten simulation seeds per realization.

The repository includes the datasets, calibrated reward scales, selected hyperparameters, and tuned composite-rule weights used by the experiments. These inputs can be reused directly. Training checkpoints must be produced before learned-policy evaluation.

The common task launcher is:

```cmd
python run_experiment.py --help
```

Add `--dry-run` to a task command to inspect its underlying command without starting an experiment. Method aliases accepted by this launcher are `hipmc`, `ddqnb`, `psa_mappo`, and `hdrl`. Some underlying scripts and output directories retain the historical identifier `ours_categorical_baa` for HIPMC and `hdrl_option` for HDRL.

## Main-manuscript experiments

### 1. Training and hyperparameter sensitivity

Each of the four learned methods receives the same twelve configurations: learning rates `5e-5`, `1e-4`, and `2e-4`, crossed with discount factors `0.90`, `0.95`, `0.99`, and `1.00`. Each configuration is trained for 500 episodes with all three seeds.

```cmd
python run_experiment.py train-grid --device auto
python run_experiment.py summarize-generated-training
```

To train one method at a time, add, for example, `--methods hipmc` to `train-grid`. Evaluation loads the method-specific selections recorded in `experiment_manifests/selected_generated_hyperparameters.json`: HIPMC and PSA-MAPPO use learning rate `1e-4` and discount factor `0.90`; DDQNB and HDRL use `5e-5` and `0.99`.

### 2. Generated-benchmark performance and computational efficiency

Evaluate the learned policies and non-learning baselines under the reference scenario S0:

```cmd
python run_experiment.py evaluate-generated --scenes S0 --device auto --instances 10
python run_experiment.py evaluate-nonlearning --methods sdr composite --scenes S0 --instances 10
python run_experiment.py evaluate-nonlearning --methods ma --phase plan --scenes S0 --ma-threads 4 --instances 10
python run_experiment.py evaluate-nonlearning --methods ma --phase evaluate --scenes S0 --instances 10
python run_experiment.py evaluate-nonlearning --methods rh_alns --scenes S0 --instances 10
python run_experiment.py summarize-results --instances 10
```

`sdr` evaluates all 18 rule combinations. The composite rule loads the published weights from `experiment_manifests/tuned_composite_priority_weights.json`. MA first constructs a full-information static plan and then executes it in the stochastic simulator. RH-ALNS replans online using up to 8,000 objective evaluations per reoptimization.

Evaluation records computation time per online decision and per complete episode. The job-count levels already present in S0 provide the computational-size comparison. Search and reoptimization summaries are available through:

```cmd
python run_experiment.py summarize-ma --instances 10
python run_experiment.py summarize-rh --instances 10
```

### 3. Generalization across degradation stress and equipment configurations

The trained policies and composite-rule weights are reused under five additional scenarios:

| Scenario | Change from S0 |
| --- | --- |
| S1 | Double degradation and breakdown rates. |
| S2 | Double maintenance durations. |
| S3 | Double both event rates and maintenance durations. |
| S4 | Change class-specific machine capacities from `(60, 80, 100)` to `(70, 90, 120)`. |
| S5 | Change class-specific machine counts from `(2, 2, 1)` to `(2, 1, 2)`. |

Each scenario uses five representative production conditions defined in `experiment/revision_selection.py`.

```cmd
python run_experiment.py evaluate-generated --scenes S1 S2 S3 S4 S5 --device auto --instances 10
python run_experiment.py evaluate-nonlearning --methods sdr composite --scenes S1 S2 S3 S4 S5 --instances 10
python run_experiment.py evaluate-nonlearning --methods ma --phase plan --scenes S1 S2 S3 S4 S5 --ma-threads 4 --instances 10
python run_experiment.py evaluate-nonlearning --methods ma --phase evaluate --scenes S1 S2 S3 S4 S5 --instances 10
python run_experiment.py evaluate-nonlearning --methods rh_alns --scenes S1 S2 S3 S4 S5 --instances 10
python run_experiment.py summarize-results --instances 10
```

### 4. Component contributions

Evaluate nine component replacements under S0–S5 using the same five representative conditions:

- Batch selection: maximum utilization, minimum average slack, earliest batch due date, or tuned composite priority.
- Maintenance: health-state-threshold, periodic, or corrective-only maintenance.
- Batch formation: earliest-due-date job packing or first-fit-decreasing packing.

Each replacement retains the other trained components. The HIPMC reference is reused from the preceding evaluations.

```cmd
python run_experiment.py evaluate-ablations --scenes S0 S1 S2 S3 S4 S5 --device auto --instances 10
python run_experiment.py summarize-results --instances 10
```

The crossed batch-selection/maintenance comparisons in the Supplementary Material use these same outputs: HIPMC, composite batch selection with learned maintenance, learned batch selection with threshold maintenance, and composite batch selection with threshold maintenance. For this comparison, the S0 reference results are restricted to the same five representative conditions as the component replacements.

### 5. Industrial order-based validation

Three deployment modes are evaluated: zero-shot transfer of generated-data policies, training from scratch for 500 episodes, and fine-tuning from generated-data checkpoints for 200 episodes.

```cmd
python run_experiment.py train-industrial --mode scratch --device auto
python run_experiment.py summarize-industrial-training --mode scratch
python run_experiment.py train-industrial --mode finetune --device auto
python run_experiment.py summarize-industrial-training --mode finetune
python run_experiment.py evaluate-industrial --device auto
```

Evaluate the common non-learning baselines:

```cmd
python run_experiment.py evaluate-industrial-nonlearning --methods sdr composite
python run_experiment.py evaluate-industrial-nonlearning --methods ma --phase plan --ma-threads 4
python run_experiment.py evaluate-industrial-nonlearning --methods ma --phase evaluate
python run_experiment.py evaluate-industrial-nonlearning --methods rh_alns
python run_experiment.py summarize-results --instances 10
```

Industrial evaluation uses all 60 arrival realizations. `--instances 10` in the summary command controls generated-data selection.

## Supplementary experiments

### Composite-rule weight search

The supplied composite-rule weights can be used directly for all comparisons above. To reproduce their selection, run the search on the generated-data training condition:

```cmd
python comparison_runners/composite_priority/tune.py --optimization-seeds 101 --search-instances 30 --selection-instances 30 --search-replications 3 --selection-replications 10 --random-candidates 128 --local-per-step 16 --finalists 10 --workers 4 --output-dir runs/composite_priority_tuning
```

The search evaluates single-feature and equal-weight anchors, random weight vectors, and local perturbations. The final ten candidates are compared on an independent selection set. Search records and selected weights are written to `runs/composite_priority_tuning/`. To use a newly reproduced selection in subsequent evaluations, copy its `tuned_composite_priority_weights.json` into `experiment_manifests/` before starting those evaluations. The same selected weights are used across generated scenarios, component replacements, and the industrial case.

### Extended training budget

Continue the selected configurations to a total of 3,000 episodes, summarize training and validation, and evaluate their selected checkpoints:

```cmd
python run_experiment.py continue-extended --device auto
python run_experiment.py summarize-extended-training
python run_experiment.py evaluate-extended --scenes S0 --device auto --instances 10
python run_experiment.py summarize-results --instances 10
```

Budget-specific snapshots preserve the 500-episode models used in the main experiments.

### Discount-consistent potential shaping

Train and validate HIPMC with the discount-consistent potential-shaping formulation at discount factor `0.90`, using the same 500-episode budget and three training seeds:

```cmd
python comparison_runners/revision_followup.py train-shaping --device auto
python comparison_runners/revision_followup.py summarize-shaping
```

This experiment reports training and validation performance. Its outputs are stored separately under `runs/revision_reward_and_rule_followup/`. The discount-factor-1.00 results are part of the common hyperparameter grid.

### Preventive maintenance during empty-queue periods

Train the extension that allows an idle machine to choose preventive maintenance or waiting when the released, unassigned queue is empty. Compare it with the standard HIPMC policies:

```cmd
python run_experiment.py train-idle --device auto
python run_experiment.py evaluate-idle --scenes S0 S1 S2 S3 S4 S5 --device auto --instances 10
python run_experiment.py summarize-results --instances 10
```

### Small-scale industrial MILP benchmark

With Gurobi installed and licensed, solve the supplied 10- and 20-job instances. Then evaluate static-plan execution and the industrial policies trained from scratch:

```cmd
python MILP/run_all_instances.py
python run_experiment.py evaluate-small-milp
python run_experiment.py evaluate-small-drl --device auto
python run_experiment.py summarize-results --instances 10
```

The MILP optimizes the deterministic production-scheduling subproblem. Static-plan execution is subsequently evaluated with stochastic degradation and corrective repairs.

## Outputs and analysis

| Output | Default location |
| --- | --- |
| Generated-data training grid | `runs/revision_hyperparameter_training/` |
| Generated-data learned-policy evaluation | `runs/revision_generated_drl_evaluation/` |
| Generated-data non-learning evaluation | `runs/revision_nonlearning_evaluation/` |
| Component replacements | `runs/revision_component_ablation/` |
| Industrial training | `runs/revision_industrial_training/` |
| Industrial learned-policy evaluation | `runs/revision_industrial_evaluation/` |
| Industrial non-learning evaluation | `runs/revision_industrial_sdr/`, `runs/revision_industrial_composite/`, `runs/revision_industrial_ma/`, `runs/revision_industrial_rh_alns/` |
| Extended-budget evaluation | `runs/revision_extended_budget_evaluation/` |
| Empty-queue maintenance evaluation | `runs/revision_idle_maintenance_evaluation/` |
| Aggregated statistics | `runs/revision_result_statistics/` |

Training outputs include episode records, validation records, and checkpoints. Evaluation outputs include partitioned episode-level results and computational-time measurements. Job-level records are disabled by default.

`summarize-results` reads the sources listed in `experiment_manifests/revision_analysis_sources.json`. The summaries include mean TTD with 95% bootstrap confidence intervals, paired relative reductions, instance-level wins/losses, signed-rank-biserial effect sizes, and Holm-adjusted significance tests. Simulation outcomes are averaged within each instance and trained model before instance-level comparison.

Existing completed partitions and supported training checkpoints can be reused when continuing a run. Keep outputs from different configurations in separate directories. If an evaluation output path is changed, update the analysis-source configuration accordingly.

## Regression tests

For a focused check of composite-rule maintenance control and reward-shaping behavior:

```cmd
python -m unittest tests.test_composite_priority tests.test_composite_retuning tests.test_reward_shaping_followup tests.test_revision_statistics
```
