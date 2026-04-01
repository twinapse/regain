# Experimental Framework

This document specifies the benchmark protocol, controller regimes, evaluation behavior, and metric logging
organization used to measure **retrieval-correctable forgetting** on supported class-incremental
learning (CIL) scenarios (**SplitCIFAR-100**, **Split CUB-200**, **Split Tiny-ImageNet**, and **Split ImageNet-R**).

---

## Contents

- [0) Overview](#0-overview)
- [1) Benchmark & Scenario](#1-benchmark--scenario)
- [2) Model & Training Protocol](#2-model--training-protocol)
- [3) Controllers & Evaluation Behavior](#3-controllers--evaluation-behavior)
- [4) Repair Data (Disjoint from Training)](#4-repair-data-disjoint-from-training)
- [5) Recorded Accuracies](#5-recorded-accuracies)
- [6) Forgetting & Repair Metrics](#6-forgetting--repair-metrics)
- [7) Metric Logging & Reporting (MLflow)](#7-metric-logging--reporting-mlflow)
- [8) REGAIN analysis tool: Curves, Frontiers, and Predictive Associations](#8-regain-analysis-tool-curves-frontiers-and-predictive-associations)

---

## 0) Overview

**Goal.** Quantify how much forgetting in continual learning is **retrieval-correctable** by a controller, using per-task
accuracies and derived analysis metrics.

**High-level pipeline.**
1. Build the configured continual learning scenario (`scenario`, `num_experiences`).
2. Train a single-head classifier sequentially over experiences.
3. Fit and/or apply the configured controller (depending on controller type).
4. Evaluate and log metrics.
5. Compute analysis artifacts: $A_{\text{ref}}, A_{\text{post}}$, plus $A_{\text{ctrl}}, \rho$ and aggregates when a
   repair controller is present.

`regain/cli/run_experiment.py` executes this pipeline and logs metrics to MLflow. The REGAIN analysis tool
(`regain/cli/run_analysis.py`) consumes the logged metrics to generate tables, curves, frontiers, and predictive
association summaries.

Experiment execution code is organized under `regain/experiments/`:
`orchestrator.py` (run flow), `config.py` (YAML parsing/validation), `builders.py` (scenario/model/strategy builders),
`backbone.py` (reserved backbone-run reuse helpers), `logging.py` (MLflow params/summary logging), and `utils.py`.

**Mental model.** Experiments write per-experience metrics and analysis artifacts to MLflow. The analysis tool is a
read-only post-processing step that aggregates those logged runs into tidy tables, recoverability/calibration/latency
curves, efficiency frontiers, and predictive-correlation summaries.

---

## 1) Benchmark & Scenario

### Dataset and split
- Scenario selection is configuration-driven through `scenario`.
- `num_experiences` controls the class-incremental partitioning and is configurable per experiment.
- Evaluation setting: **single-head class-incremental** (no task ID at test time)

### Supported scenarios

#### `split_cifar100`
- Dataset: **CIFAR-100** (100 classes, 32x32 images)
- Scenario builder: **Avalanche `SplitCIFAR100`**
- Data source: Managed automatically by Avalanche (download and dataset handling).

#### `split_tiny_imagenet`
- Dataset: **Tiny-ImageNet-200** (200 classes, 64x64 images)
- Scenario builder: **Avalanche `SplitTinyImageNet`**
- Data source: Managed automatically by Avalanche (download and dataset handling).

#### `split_cub200`
- Dataset: **CUB-200-2011** (200 bird species, variable-resolution natural images)
- Scenario builder: **Avalanche `SplitCUB200`**
- Data source: Downloaded from the CaltechDATA CUB-200-2011 archive.

#### `split_imagenet_r`
- Dataset: **ImageNet-R** (200 classes)
- Scenario builder: **custom `SplitImageNetR`**
- Data source: Downloaded from the Berkeley ImageNet-R release ([Hendrycks et al.](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)).
- Expected split layout under dataset root: `train/<class_name>/*` and `test/<class_name>/*`.
- If explicit split folders are not found, a deterministic per-class train/test holdout is built from a single-root class folder layout using `seed`.

### Task definition (Avalanche)
- For `split_cifar100`, we use `SplitCIFAR100(n_experiences=<num_experiences>, return_task_id=False, seed=<seed>, class_ids_from_zero_from_first_exp=True)`.
- For `split_tiny_imagenet`, we use `SplitTinyImageNet(n_experiences=<num_experiences>, return_task_id=False, seed=<seed>, class_ids_from_zero_from_first_exp=True)`.
- For `split_cub200`, we use `SplitCUB200(n_experiences=<num_experiences>, classes_first_batch=<computed>, return_task_id=False, seed=<seed>, class_ids_from_zero_from_first_exp=True)` with scenario-managed dataset download.
- For `split_imagenet_r`, we resolve raw train/test datasets and then build a class-incremental NC benchmark with `return_task_id=False`.
- Each experience corresponds to one task $T_i$.
- The class partition/order is determined by `seed` and scenario generator logic.
- Assumption: class IDs are contiguous starting from 0 across the full benchmark; this is enforced by the runner
  (`verify_classes` in `regain/experiments/builders.py`).

### Label-space regimes (important!)
We record accuracies under two different “label space” regimes:

1) **Seen-classes-only evaluation (for reference accuracies)**  
   Right after learning $T_i$, predictions are restricted to the set of classes seen up to $T_i$.  
   (Implemented via a seen-classes masking plugin during reference evaluation.)

2) **All-classes evaluation (for post-sequence + controller accuracies)**  
   After training all experiences, evaluation is performed over all benchmark classes.

---

## 2) Model & Training Protocol

### Model
- Backbone: `resnet18` | `vit_small` | `vit_base` (default `resnet18`)
- Normalization: **BatchNorm** (standard ResNet-18 by default)
- Classifier head: linear layer to **`num_classes(scenario)`** logits (single head)

> ℹ️ **Class label invariant:** in this single-head setup, logit index `c` corresponds to global class ID `c` (class IDs
> are contiguous from 0), and downstream analyses/controllers are expected to preserve this ordering.

### Pretraining
- Backbone initialization: **random initialization (no pretraining)**

### Incremental training protocol
- Offline / multi-epoch training per experience.
- After completing experience $T_k$, training proceeds to $T_{k+1}$.
- Raw data from previous experiences are not reused (unless the chosen Avalanche strategy uses replay/memory internally).
- The classifier remains single-head over all benchmark classes throughout.
- Typical backbone schedules are expressed explicitly in config:
  - `resnet18` commonly uses `backbone.training.num_epochs: 50` per experience.
  - `vit_small` and `vit_base` commonly use `backbone.training.num_epochs: 100` per experience.

---

## 3) Controllers & Evaluation Behavior

This codebase distinguishes when a controller acts and how evaluation is executed.

### 3.1 Controller types

**Prevention (training-time) controllers** (`PreventionController`)
- Affect training dynamics and therefore the final learned model.
- Examples:
  - modify the training loss (implements `TrainingObjectiveControllerInterface`)
  - modify model internals in a way that affects training (implements `BackboneControllerInterface`, e.g., normalization replacement)

**Repair (post-training) controllers** (`RepairController`)
- Intended for *post-hoc* / *post-training* repair.
- They should not change the backbone’s training dynamics.
- They may:
  - observe repair data during training (via the repair stream), and/or
  - fit after an experience or after training completes, and/or
  - correct logits during evaluation (inference-time correction).
- In code, repair controllers receive post-training hooks (`on_train_experience_end`, `on_train_end`), plus evaluation
  hooks (`on_eval_*`) and `correct_outputs` during evaluation. They do **not** receive training-time hooks.

### 3.2 Repair/fitting protocol by controller type

**`PreventionController` (training-time intervention)**
- Acts during training; it may change:
  - the loss surface and optimization trajectory, and/or
  - the backbone architecture/normalization behavior.
- It is not purely post-hoc, so this pipeline does not expose a fair base/ctrl toggle for prevention runs.

**`RepairController` (post-hoc repair)**
- After each training experience, the repair stream provides that experience’s fixed repair set.
- Controller fitting uses a deterministic stratified subset (`repair.budget_fraction`) of that set.
- Fitting schedule:
  - For repair-controller runs, `repair.fit_schedule` must be set explicitly.
  - If `repair.fit_schedule=per_experience`, fit after each experience.
  - If `repair.fit_schedule=final_only`, fit once after the full training sequence.
  - Controllers that require per-experience fitting (`requires_per_experience_fitting()`) will error when set to
    `final_only`.
- During fitting, backbone parameters $\phi$ are frozen (inference-only forwards); the controller trains on samples from
  the **repair stream** (aggregated across experiences).
- The test set is never used for fitting $g_\theta$.

### 3.3 Evaluation

Evaluation flow per run:

- After each training experience, an end-of-experience base accuracy point ($A_{\text{exp,base}}$) is computed and recorded.
- For repair-controller runs with `repair.fit_schedule=per_experience`, posthoc raw evaluation also runs after each
  experience.
- At training end, final posthoc raw evaluation is guaranteed:
  - if final checkpoint was already covered by per-experience posthoc evaluation, no extra pass is needed;
  - otherwise, one final posthoc raw evaluation pass is executed.
- In repair-controller runs, base baselines ($A_{\text{exp,base}}$, $A_{\text{base}}$) are inherited from the
  reserved `backbone` run. In backbone/prevention runs, they are computed from the current run.

Within that flow, evaluation cadence is controlled by two knobs:

1. `evaluation.avalanche_schedule` controls only Avalanche built-in evaluation logs (`run.eval.*`):
   - `per_experience` maps to `eval_every=0` (evaluate after each experience).
   - `final_only` maps to `eval_every=-1` (evaluate only at the end).
   - Forward-transfer metrics are emitted only when `evaluation.avalanche_schedule=per_experience`.
2. `repair.fit_schedule` (repair controllers) controls repair fitting cadence and posthoc raw evaluation cadence:
   - `per_experience`: fit after each experience; posthoc raw evaluation runs after each experience.
   - `final_only`: fit once after training; posthoc raw evaluation runs once after training.
   - If the final checkpoint was not already evaluated in `per_experience` mode, one final posthoc raw evaluation pass
     is executed.

>ℹ️ Metric placement and metric namespaces are defined in [section 7](#7-metric-logging--reporting-mlflow).

### 3.4 Shared backbone execution for controller runs
When one or more configured controller runs are present:
- A base `backbone` trajectory must be available, either by:
  - automatically executing that dedicated reserved `backbone` run in the current experiment when it does not exist, or
  - loading an existing `backbone` run from `backbone.source_experiment`.
- `backbone.source_experiment` reuse requires that source `backbone` run to contain
  backbone checkpoint artifacts (i.e., checkpoints must have been saved in that source run), and base
  baseline vectors (`acc.exp.base`, `acc.final.base`) available as either metrics or in `analysis_artifacts.json`.
- Repair-run downstream analysis enforces a baseline-only policy:
  - diagnostic vectors and analysis calibration vectors (`run.calibration.ece`, `run.calibration.aece`, `run.calibration.nll`) are sourced from
    base `analysis_artifacts.json`;
  - run-level repair `run.calibration.max_ece` is derived from baseline `run.calibration.ece` vectors;
  - ctrl per-task run metrics for those keys are ignored.
- When executed in the current experiment, the `backbone` run performs the only backbone training pass and writes one
  checkpoint per experience.
- Each repair-controller run then reuses those checkpoints (the loader restores each experience checkpoint before the
  experience hooks), so controller behavior is compared on the same trained backbone trajectory.
- The `backbone` run is also the source of base analysis vectors (`acc.exp.base`, `acc.final.base`) reused by repair runs.
- `name: backbone` is reserved in `runs` (always).
- An experiment cannot contain multiple runs named `backbone`.
- If a local `backbone` run already exists while `backbone` config is non-null, execution is rejected
  (including configurations that set `backbone.source_experiment`).
- If `checkpoints_enabled: true`, shared checkpoints are also logged to MLflow artifacts.

### 3.5 Run Configuration Constraints
- Required top-level fields:
  - `experiment_name`
  - `scenario`
  - `num_experiences`
  - `repair` (mapping)
  - `evaluation` (mapping; defaults are applied only within this mapping)
- `backbone` configuration must define exactly one of:
  - `backbone.training` (train from scratch in the current experiment), or
  - `backbone.source_experiment` (reuse a reserved `backbone` run from another experiment).
- `backbone.training.optimizer.name` supports `sgd` and `adamw`.
- `backbone.training.lr_scheduler.name` supports `multi_step` and `warmup_cosine`.
- `backbone.training.grad_clip_max_norm` is optional; when set, gradients are clipped before each optimizer update.
- For `adamw`, provide `backbone.training.optimizer.kwargs.betas` as a YAML sequence such as `[0.9, 0.999]`.
- For `warmup_cosine`, provide `backbone.training.lr_scheduler.kwargs.warmup_epochs`; `min_lr` is optional.
- When `backbone.source_experiment` is provided, it must be the only field under `backbone`.
- `backbone.source_experiment` must be different from `experiment_name` (same-experiment reuse is rejected).
- `repair` configuration is mandatory. All runs in an experiment (backbone, prevention controllers, repair controllers)
  share the exact same repair split defined by:
  - required `repair.split_fraction` (float in `[0, 1)`) as the fraction of each training experience excluded from
    backbone training.
- `repair.budget_fraction` is the fitting fraction consumed from each fixed repair set.
  For each experience with repair set size `r_exp = floor(repair.split_fraction * n_exp)`,
  fitting uses `floor(repair.budget_fraction * r_exp)` samples.
- If any run in `runs` is a repair controller, the following fields are required explicitly:
  - `repair.budget_fraction` (float in `(0, 1]`)
  - `repair.fit_schedule` (`per_experience` or `final_only`)
  - `repair.num_epochs`
  - `repair.batch_size`
- Every user-configured run in `runs` must define a controller.
- Run-specific config blocks are not allowed to override experiment-level training/runtime parameters
  (for example epochs, batch sizes, replay memory size, device, eval frequency, repair budget/fit schedule,
  and checkpoint-saving flags).
- Each run logs `controller.type` with one of `none`, `prevention`, or `repair`.
  Analysis uses this parameter as the authoritative controller-kind signal.
- Controller runs do not log `backbone.*` parameters, except `backbone.source_experiment.id` and
  `backbone.source_experiment.name` when source reuse is configured. `backbone.source_experiment.name` is a run-time
  snapshot and is not synchronized if the source experiment is renamed later.
- If `runs` is omitted, `null`, or empty, the run executes backbone-only mode:
  - if `backbone.source_experiment` is not set, it logs only the reserved `backbone` run (creating it only when absent);
  - if `backbone.source_experiment` is set, it reuses the source `backbone` run without creating a new local run
    (requires source backbone checkpoint artifacts and `acc.exp.base`/`acc.final.base` baselines).
  - if a local `backbone` run already exists while `backbone` config is non-null, execution is rejected.
- If `backbone.training` is not set and user-configured runs are present, all configured runs must be repair-controller
  runs (non-repair runs require local backbone training).

---

## 4) Repair Data (Disjoint from Training)

We use two related but distinct concepts alongside the sequential training stream:

- **Repair stream**: a parallel Avalanche stream whose experiences contain the repair subsets. This is a sequence of
  experiences, not maintained storage.
- **Repair buffer**: an optional maintained storage set populated from the repair stream and used for controller
  fitting when you need a fixed-capacity memory and do not know how many classes will appear in the future. **The 
  default experimental pipeline does not use this buffer**.

- For each experience’s original training dataset, we split samples into:
  - a **training subset** used by the continual learner, and
  - a **repair set** (the full held-out repair subset for that experience).
- The split is deterministic under the experiment seed.
- Repair samples are never used for backbone training.
- The split is controlled by `repair.split_fraction`:
  - per experience with `n_exp` samples, the repair set size is
    `floor(repair.split_fraction * n_exp)`.
  - a guard enforces at least one remaining training sample per class.
- Setting `repair.split_fraction: 0.0` disables the repair stream entirely.
- For repair controllers, fitting uses a deterministic stratified **repair fit subset** sampled from each repair set:
  - per experience with repair set size `r_exp`, used samples are `floor(repair.budget_fraction * r_exp)`;
  - class proportions follow the repair-set distribution as closely as possible.
- Dataset split indices are logged as `splits.tar.gz`, with entries named `{stream}/exp_###.txt`.
- Analysis derives `repair_set_total` as:
  - `0` when `repair.split_fraction == 0.0`;
  - otherwise, the exact total non-empty line count over `repair/exp_*.txt` in `splits.tar.gz`.
- **Important:** The `repair` configuration section is **mandatory** for all experiments, even backbone-only runs.
  This forces an explicit decision about data splitting. To use the full dataset (no repair hold-out), 
  you must explicitly set `repair.split_fraction: 0.0`. This prevents unintentional data leakage where a 
  backbone sees data that should have been held out for a downstream repair controller.

---

## 5) Recorded Accuracies

Let $T_i$ be the i-th Avalanche experience.

### Reference accuracy (seen-classes-only, right after learning $T_i$)
$$
A_{\text{ref}}(T_i) =
\text{Acc}\Big(\text{test}(T_i)\ \text{with predictions restricted to seen classes up to } T_i \Big)
$$

### Post-sequence accuracy (all-classes, after all experiences)
$$
A_{\text{post}}(T_i) =
\text{Acc}\Big(\text{test}(T_i)\ \text{with label space } \{0,\dots,C-1\}\Big)
$$

>ℹ️ With prevention controllers, $A_{\text{post}}$ is the posthoc evaluation of the model trained under the controller
> (no eval-time toggling). With repair controllers, $A_{\text{post}}$ is the shared base baseline taken from
> the reserved `backbone` run for the same checkpoint trajectory.

### Controller accuracy (all-classes)
$$
A_{\text{ctrl}}(T_i;\theta,b) =
\text{Acc}\Big(\text{test}(T_i)\ \text{with label space } \{0,\dots,C-1\}\Big)
$$
>ℹ️ $A_{\text{ctrl}}$ and $\rho$ are logged only for repair controllers, because prevention controllers do not define
> a post-hoc, toggleable evaluation-time correction.

### Final global accuracy (all classes)
$$
A_{\text{final}} = \text{Acc}\Big(\text{test}(\{0,\dots,C-1\})\Big)
$$
>ℹ️ In practice this corresponds to the posthoc stream accuracy (e.g., `Top1_Acc_Stream`). It is also surfaced as a
> normalized `run.summary.accuracy.final.avg.base`; see [section 7](#7-metric-logging--reporting-mlflow).

---

## 6) Forgetting & Repair Metrics

### Total forgetting
$$
F_{\text{total}}(T_i) = A_{\text{ref}}(T_i) - A_{\text{post}}(T_i)
$$

### Residual forgetting after repair
$$
F_{\text{res}}(T_i;\theta,b) = A_{\text{ref}}(T_i) - A_{\text{ctrl}}(T_i;\theta,b)
$$

### Absolute recovery
$$
\Delta A(T_i;\theta,b)=A_{\text{ctrl}}(T_i;\theta,b)-A_{\text{post}}(T_i)
$$

### Retrieval-correctable fraction
$$
\rho(T_i;\theta,b) =
\frac{A_{\text{ctrl}}(T_i;\theta,b) - A_{\text{post}}(T_i)}
     {A_{\text{ref}}(T_i) - A_{\text{post}}(T_i)}
$$
>ℹ️ $\rho$ is reported only for repair controllers.

### Valid vs invalid tasks (for $\rho$)
A task $T_i$ is considered **valid for retrieval-correctable fraction computations** if:

$$
F_{\text{total}}(T_i) = A_{\text{ref}}(T_i) - A_{\text{post}}(T_i) > \epsilon
$$

where $\epsilon = 10^{-4}$.

- If $F_{\text{total}}(T_i) \le \epsilon$, then $\rho(T_i;\theta,b)$ is treated as **invalid** and set to `None`.
- Invalid tasks are **excluded** from mean/aggregate $\rho$ computations.
- All accuracies $A_{\text{ref}}, A_{\text{post}}, A_{\text{ctrl}}$ are recorded for every task.

---

## 7) Metric Logging & Reporting (MLflow)

This section describes **how metrics are organized in MLflow** and **what we report**.

>ℹ️ Evaluation cadence rules are defined in section 3.3.

### 7.1 Metric namespaces
Metric keys are normalized and namespaced as:

- `run.train.<metric>` for training-time logs
- `run.eval.<metric>` for Avalanche built-in evaluation logs
- `run.exp###.<metric>` for per-experience posthoc evaluations (repair runs with `per_experience` schedule)
- `run.final.<metric>` for the final posthoc evaluation
- `run.calibration.<...>` for calibration metrics (for example per-task `run.calibration.<metric>.exp###` and run-level `run.calibration.max_ece`)
- `run.diagnostics.<...>` for task-level diagnostic metrics (for example `run.diagnostics.<metric>.exp###`)
- `run.latency.<...>` for base/ctrl latency and throughput
- `run.repair.seconds` / `run.repair.steps` for repair fit-time resources
- `run.summary.<...>` for duplicated run-level summary scalars used in dashboards

Cross-run analysis artifacts (CSV columns) use `analysis.<...>` keys.
Examples: `analysis.repair.rho.avg`, `analysis.accuracy.final.avg.ctrl`, `analysis.latency.ms_ratio.avg`.

**Important implications**
- Runs do not emit paired base/ctrl posthoc metric streams in the same run.
- In repair-controller runs, base analysis baselines (`acc.exp.base`, `acc.final.base`) are inherited from the `backbone`
  run; posthoc raw metrics in repair runs correspond to ctrl evaluation.
- `run.train.*` metrics always correspond to the actual training procedure that was executed:
  - If a training-time controller was used, `run.train.*` metrics reflect that controller-influenced training.
  - `run.train.*` is not a base/ctrl comparison; it is simply “what happened during training”.

### 7.2 MLflow run structure
Each experiment run creates a single MLflow run. Run creation rules (reserved `backbone` run,
`backbone.source_experiment` reuse, and rejection conditions) are defined in section 3.5.

Analysis collection requires each run to provide:
- `controller.type` in `{none, prevention, repair}`;
- `repair.split_fraction`.

Runs that do not satisfy these analysis requirements are skipped during collection and recorded as run-level failures.
`regain/cli/run_analysis.py` can still publish successful outputs when invoked with `--allow-partial`.
When zero runs are successfully collected, analysis outputs are not published and the command exits with failure.

All metrics live on the run. The final posthoc evaluation is always prefixed with `final.`. For repair-controller
runs with `repair.fit_schedule=per_experience`, per-experience checkpoint metrics are additionally prefixed with the
checkpoint name (`exp000.`, `exp001.`, ...).

Metric-family placement summary:

| Metric family                          | Key pattern                                         |
|----------------------------------------|-----------------------------------------------------|
| Training metrics                       | `run.train.<metric>`                                |
| Avalanche scheduled evaluation metrics | `run.eval.<metric>`                                 |
| Posthoc raw evaluation metrics         | `run.final.<metric>` or `run.exp###.<metric>`       |
| Calibration per-task metrics           | `run.calibration.<metric>.exp###`                         |
| Diagnostic per-task metrics            | `run.diagnostics.<metric>.exp###`                          |
| Repair fit resources                   | `run.repair.seconds`, `run.repair.steps`, and suffixed keys |
| Latency overhead                       | `run.latency.<...>`                                 |
| Summary duplicate metrics              | `run.summary.<...>`                                 |
| Cross-run analysis columns             | `analysis.<...>`                                    |

### 7.3 Where the analysis artifacts live
After each experience, the evaluation plugin logs:

- Per-task:
  - `run.accuracy.exp.exp###.base`

At the end of training, when reference accuracies are complete, the evaluation plugin logs:

- Per-task:
  - `run.accuracy.final.exp###.base`
- Per-task (repair controllers only):
  - `run.accuracy.final.exp###.ctrl`
  - `run.repair.rho.exp###` (only for valid tasks)
- Aggregates:
  - `run.accuracy.final.avg.base`
  - `run.summary.accuracy.final.avg.base`
  - `run.summary.accuracy.exp.avg.base`
- Aggregates (repair controllers only):
  - `run.repair.rho.avg`
  - `run.summary.repair.rho.avg`
  - `run.accuracy.final.avg.ctrl`
  - `run.summary.accuracy.final.avg.ctrl`
- Calibration aggregates:
  - `run.calibration.max_ece`:
    - non-repair runs: maximum per-task ECE over the latest completed eval pass;
    - repair runs: maximum over baseline artifact `run.calibration.ece` vectors (`max_i run.calibration.ece[i]`).
- Additional persisted vectors in `analysis_artifacts.json` (when available):
  - `run.diagnostics.out_of_task_rate`
  - `run.diagnostics.avg_conf`
  - `run.diagnostics.avg_entropy`
  - `run.calibration.ece`
  - `run.calibration.aece`
  - `run.calibration.nll`
  - `run.diagnostics.logit_avg_drift`
- Additional persisted scalar in `analysis_artifacts.json` (when available):
  - `run.calibration.max_ece`
- For repair-controller runs, analysis outputs (`runs_table`, curves, predictive summaries) enforce baseline-only
  consumption of diagnostic values and analysis calibration values (`run.calibration.ece`, `run.calibration.aece`, `run.calibration.nll`,
  run-level `run.calibration.max_ece`) from `analysis_artifacts.json`.
  For `run.calibration.max_ece`, repair runs use vector-derived `max(run.calibration.ece)`.
- Additional run-level efficiency metrics:
  - `run.repair.seconds` / `run.repair.steps` (cumulative) and `run.repair.seconds.exp###` / `run.repair.steps.exp###`
    or `.final` (per-fit event)
  - `run.latency.ms_per_sample.base`
  - `run.latency.samples_per_sec.base`
  - `run.latency.ms_per_sample.ctrl` (controller runs)
  - `run.latency.samples_per_sec.ctrl` (controller runs)
  - `run.latency.ms_ratio` (controller runs; on/off latency ratio)
- If reference accuracies are incomplete, the plugin logs no `run.accuracy.final` / `run.repair.rho`
  vectors and instead writes a JSON artifact (`analysis_artifacts.json`) containing:
  - `status` (set to `incomplete_acc_exp_base`)
  - `expected_num_experiences`
  - `observed_num_exp_points`
  - `run.eps`
  - partial `acc.exp.base` vector
- A flag metric `run.status.incomplete_acc_exp_base=1.0` is logged when reference accuracies are incomplete.

### 7.4 What gets reported
We report mean ± std across seeds (common configs use **3 seeds**) for:

- $A_{\text{final}}$
- $\{A_{\text{post}}(T_i)\}_{i=1}^{N}$ (where $N=\text{num\_experiences}$)
- $\bar{\rho}(\theta,b)$ for each repair controller and budget
- Optionally: $\overline{A}_{\text{ctrl}}(\theta,b)$ for each repair controller and budget
- Optionally: calibration, diagnostic, and efficiency summaries (for example `run.calibration.max_ece`, latency and repair-cost
  metrics) in budget/controller-level tables and curves

**Persisted artifacts**
- Per-task accuracies $(A_{\text{ref}}, A_{\text{post}})$ in runs with complete reference vectors
- Per-task accuracies $(A_{\text{ctrl}})$ for repair controllers with complete reference vectors
- Per-task $\rho$ for valid tasks in repair-controller runs (invalid tasks are omitted from `run.repair.rho.exp###`)
- Per-task calibration/diagnostic vectors for base post-sequence behavior (when available)
- Aggregate summaries and curve/frontier/predictive inputs

### Metric definitions (calibration, overhead, diagnostics, analysis)

The definitions below are the normative interpretation of the metric families used by REGAIN.

#### Calibration metrics

- `run.calibration.nll`: Negative log-likelihood (cross-entropy). Mean `-log p(y_true)` over samples. Lower is better.
- `run.calibration.brier`: Multiclass Brier score. Mean squared error between predicted probability vectors and one-hot labels.
  Lower is better.
- `run.calibration.ece`: Expected calibration error with fixed-width confidence bins. Weighted mean absolute gap between
  per-bin accuracy and mean confidence. Lower is better.
- `run.calibration.aece`: Adaptive ECE with approximately equal-count confidence bins. Lower is better.
- `run.calibration.mce`: Maximum calibration error. Largest absolute per-bin gap between accuracy and confidence. Lower is better.
- `run.calibration.max_ece`: Worst-task ECE.
  - Non-repair runs: worst task in the latest completed evaluation pass, i.e. `max_i run.calibration.ece.exp{i}`.
  - Repair runs: baseline-only value `max_i run.calibration.ece[i]` derived from base artifact vectors.
  Lower is better.

#### Resource / overhead metrics

- Conceptual `perf.ms_per_sample`: latency (milliseconds per sample), measured on timed forward passes after
  warm-up.
  - Logged as `run.latency.ms_per_sample.base` (base) and `run.latency.ms_per_sample.ctrl` (ctrl).
- Conceptual `perf.samples_per_sec`: throughput (samples/sec) on the same timed passes.
  - Logged as `run.latency.samples_per_sec.base` and `run.latency.samples_per_sec.ctrl`.
- Conceptual `perf.latency_ratio`: ctrl/base latency ratio.
  - Logged as `run.latency.ms_ratio = run.latency.ms_per_sample.ctrl / run.latency.ms_per_sample.base`.
- Conceptual `repair.repair_steps`: total repair optimization steps, typically
  `epochs * ceil(N_repair / batch_size)`.
  - Logged as cumulative `run.repair.steps` plus per-fit event keys (`run.repair.steps.exp###` or `run.repair.steps.final`).
- Conceptual `repair.repair_seconds`: total repair wall-clock fitting time in seconds.
  - Logged as cumulative `run.repair.seconds` plus per-fit event keys (`run.repair.seconds.exp###` or `run.repair.seconds.final`).

#### Diagnostic-layer metrics (task-level diagnostics)

- `run.diagnostics.out_of_task_rate`: For task `T_i`, fraction of predictions outside that task's class set `S_i`.
  Higher indicates stronger off-task prediction bias.
- `run.diagnostics.avg_conf`: Mean of max predicted probability on task `T_i`.
- `run.diagnostics.avg_entropy`: Mean predictive entropy on task `T_i`; higher indicates less peaked predictions.
- Conceptual `run.diagnostics.logit_avg_exp`: average logit vector on task `T_i` after training experience `i`.
- Conceptual `run.diagnostics.logit_avg_base`: average logit vector on task `T_i` at final base evaluation.
- Conceptual `run.diagnostics.logit_avg_drift_l2`: `||mu_i_exp - mu_i_base||_2`.
  - Logged as `run.diagnostics.logit_avg_drift`.
- Optional `run.diagnostics.logit_cov_drift_fro`: Frobenius drift between reference and post-sequence logit covariance matrices.
  This metric is not produced in the default pipeline.

#### Analysis outputs

- Conceptual `predict.pearson_r(<pred>, rho)`: Pearson correlation between diagnostic values and `rho(T_i)` over tasks
  where both are defined. Reported in `predictive_correlations.csv` column `pearson_r`.
- Conceptual `predict.spearman_r(<pred>, rho)`: Spearman rank correlation between diagnostic values and `rho(T_i)`.
  Reported in column `spearman_r`.
- Conceptual `predict.r2(<preds> -> rho)` (optional): coefficient of determination for predicting `rho`.
  The analysis computes univariate `R^2` per diagnostic, reported in column `r2`.

---

## 8) REGAIN analysis tool: Curves, Frontiers, and Predictive Associations

### 8.1 Recoverability curves
For each repair controller capacity level, we compute curves over repair budget fraction $b$:

- Mean recoverable fraction:
  $$
  \bar{\rho}(\theta,b)=\mathbb{E}_{i:\,F_{\text{total}}(T_i)>\epsilon}\big[\rho(T_i;\theta,b)\big]
  $$
- Mean controller accuracy across tasks:
  $$
  \overline{A}_{\text{ctrl}}(\theta,b)=\mathbb{E}_{i}[A_{\text{ctrl}}(T_i;\theta,b)]
  $$

We also report $\rho(T_i;\theta,b)$ as a function of task index $i$ (“task age”).

The `curves` analysis step writes:
- `recoverability_curve.csv`
- `task_age_rho.csv`
- `calibration_vs_budget.csv` (mean/std of `run.calibration.max_ece` by controller and budget)
- `latency_vs_budget.csv` (mean/std of latency ratio and base/ctrl latency summaries by controller and budget)

### 8.2 Repair efficiency frontier
We place controller configurations on frontiers defined by:

- **Data cost:** repair-fit fraction $b$, where $b = \text{repair.budget\_fraction}$
  (total consumed repair examples are computed from repair-set size and $b$)
- **Parameter cost:** controller parameter count $|\theta|$

Frontiers plot recovered performance (e.g., $\bar{\rho}$, $\overline{A}_{\text{ctrl}}$) against these costs.

### 8.3 Predictive associations
The `predictive` analysis step assesses the predictive power of diagnostic signals for repairability (`rho`). For 
each `(controller, budget)` group it computes correlation-based measures and writes
`predictive/predictive_correlations.csv` with:

- diagnostic key
- Pearson correlation (`pearson_r`)
- Spearman correlation (`spearman_r`)
- simple linear-fit coefficient of determination (`r2`)
- number of valid tasks used (`n_valid_tasks`)
