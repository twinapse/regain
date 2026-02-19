# Experimental Framework

This document specifies the benchmark protocol, controller regimes, evaluation behavior, and metric logging
organization used to measure **retrieval-correctable forgetting** on **SplitCIFAR-100** in a
**single-head class-incremental learning (CIL)** setting.

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
- [8) REGAIN analysis tool: Curves & Frontiers](#8-regain-analysis-tool-curves--frontiers)

---

## 0) Overview

**Goal.** Quantify how much forgetting in continual learning is **retrieval-correctable** by a controller, using per-task
accuracies and derived analysis metrics.

**High-level pipeline.**
1. Build a SplitCIFAR-100 continual learning scenario (`num_experiences`, default 10).
2. Train a single-head classifier sequentially over experiences.
3. Fit and/or apply the configured controller (depending on controller type).
4. Evaluate and log metrics.
5. Compute analysis artifacts: $A_{\text{ref}}, A_{\text{post}}$, plus $A_{\text{ctrl}}, \rho$ and aggregates when a
   repair controller is present.

`regain/cli/run_experiment.py` executes this pipeline and logs metrics to MLflow. The REGAIN analysis tool
(`regain/cli/run_analysis.py`) consumes the logged metrics to generate curves and frontiers.

Experiment execution code is organized under `regain/experiments/`:
`orchestrator.py` (run flow), `config.py` (YAML parsing/validation), `builders.py` (scenario/model/strategy builders),
`backbone.py` (reserved backbone-run reuse helpers), `logging.py` (MLflow params/summary logging), and `utils.py`.

**Mental model.** Experiments write per-experience metrics and analysis artifacts to MLflow. The analysis tool is a
read-only post-processing step that aggregates those logged runs into tables, recoverability curves, and efficiency
frontiers.

---

## 1) Benchmark & Scenario

### Dataset and split
- Dataset: **CIFAR-100**
- Image size: **32×32**
- Number of classes: **100**
- Scenario builder: **Avalanche `SplitCIFAR100`**
- Split protocol: **default 10 experiences × 10 classes** (disjoint class subsets; `num_experiences` configurable)
- Evaluation setting: **single-head class-incremental** (no task ID at test time)

### Task definition (Avalanche)
- We use `SplitCIFAR100(n_experiences=<num_experiences>, return_task_id=False, seed=<seed>, class_ids_from_zero_from_first_exp=True)`.
- Each experience corresponds to one task $T_i$.
- The class partition/order is determined by Avalanche under `seed` (tasks are defined by Avalanche).
- Assumption: class IDs are contiguous starting from 0 across the full benchmark; this is enforced by the runner
  (`verify_classes` in `regain/experiments/builders.py`).

### Label-space regimes (important!)
We record accuracies under two different “label space” regimes:

1) **Seen-classes-only evaluation (for reference accuracies)**  
   Right after learning $T_i$, predictions are restricted to the set of classes seen up to $T_i$.  
   (Implemented via a seen-classes masking plugin during reference evaluation.)

2) **All-classes evaluation (for post-sequence + controller accuracies)**  
   After training all experiences, evaluation is performed over all 100 classes.

---

## 2) Model & Training Protocol

### Model
- Backbone: **ResNet-18**
- Normalization: **BatchNorm** (standard ResNet-18 by default)
- Classifier head: linear layer to **100** logits (single head)

> ℹ️ **Class label invariant:** in this single-head setup, logit index `c` corresponds to global class ID `c` (class IDs
> are contiguous from 0), and downstream analyses/controllers are expected to preserve this ordering.

### Pretraining
- Backbone initialization: **random initialization (no pretraining)**

### Incremental training protocol
- Offline / multi-epoch training per experience.
- After completing experience $T_k$, training proceeds to $T_{k+1}$.
- Raw data from previous experiences are not reused (unless the chosen Avalanche strategy uses replay/memory internally).
- The classifier remains single-head over all 100 classes throughout.

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
- It is not purely post-hoc, so this pipeline does not expose a fair controller-off/controller-on toggle for prevention runs.

**`RepairController` (post-hoc repair)**
- After each training experience, the repair stream provides that experience’s repair subset.
- Fitting schedule:
  - If `repair.fit_schedule=per_experience` (default), fit after each experience.
  - If `repair.fit_schedule=final_only`, fit once after the full training sequence.
  - Controllers that require per-experience fitting (`requires_per_experience_fitting()`) will error when set to
    `final_only`.
- During fitting, backbone parameters $\phi$ are frozen (inference-only forwards); the controller trains on samples from
  the **repair stream** (aggregated across experiences).
- The test set is never used for fitting $g_\theta$.

### 3.3 Evaluation

Evaluation flow per run:

- After each training experience, a reference seen-classes accuracy point ($A_{\text{ref}}$) is computed and recorded.
- For repair-controller runs with `repair.fit_schedule=per_experience`, posthoc raw evaluation also runs after each
  experience.
- At training end, final posthoc raw evaluation is guaranteed:
  - if final checkpoint was already covered by per-experience posthoc evaluation, no extra pass is needed;
  - otherwise, one final posthoc raw evaluation pass is executed.
- In repair-controller runs, controller-off baselines ($A_{\text{ref}}$, $A_{\text{post}}$) are inherited from the
  reserved `backbone` run. In backbone/prevention runs, they are computed from the current run.

Within that flow, evaluation cadence is controlled by two knobs:

1. `evaluation.avalanche_schedule` controls only Avalanche built-in evaluation logs (`eval.*`):
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
- A controller-off `backbone` trajectory must be available, either by:
  - automatically executing that dedicated reserved `backbone` run in the current experiment when it does not exist, or
  - loading an existing `backbone` run from `backbone.source_experiment`.
- `backbone.source_experiment` reuse requires that source `backbone` run to contain
  backbone checkpoint artifacts (i.e., checkpoints must have been saved in that source run), and controller-off
  baseline vectors (`a_ref`, `a_post`) available as either metrics or in `analysis_artifacts.json`.
- When executed in the current experiment, the `backbone` run performs the only backbone training pass and writes one
  checkpoint per experience.
- Each repair-controller run then reuses those checkpoints (the loader restores each experience checkpoint before the
  experience hooks), so controller behavior is compared on the same trained backbone trajectory.
- The `backbone` run is also the source of controller-off analysis vectors (`a_ref`, `a_post`) reused by repair runs.
- `name: backbone` is reserved in `runs` (always).
- An experiment cannot contain multiple runs named `backbone`.
- If a local `backbone` run already exists while `backbone` config is non-null, execution is rejected
  (including configurations that set `backbone.source_experiment`).
- If `checkpoints_enabled: true`, shared checkpoints are also logged to MLflow artifacts.

### 3.5 Run Configuration Constraints
- `backbone` configuration must define exactly one of:
  - `backbone.training` (train from scratch in the current experiment), or
  - `backbone.source_experiment` (reuse a reserved `backbone` run from another experiment).
- When `backbone.source_experiment` is provided, it must be the only field under `backbone`.
- `backbone.source_experiment` must be different from `experiment_name` (same-experiment reuse is rejected).
- `repair` configuration is mandatory. All runs in an experiment (backbone, prevention controllers, repair controllers)
  share the exact same data split defined by `repair.budget_per_class`.
  - If `budget > 0`, all runs train on `Full - Budget`.
  - If `budget = 0`, all runs train on `Full`.
- Every user-configured run in `runs` must define a controller.
- Run-specific config blocks are not allowed to override experiment-level training/runtime parameters
  (for example epochs, batch sizes, replay memory size, device, eval frequency, repair budget/fit schedule,
  and checkpoint-saving flags).
- Controller runs do not log `backbone.*` parameters, except `backbone.source_experiment.id` and
  `backbone.source_experiment.name` when source reuse is configured. `backbone.source_experiment.name` is a run-time
  snapshot and is not synchronized if the source experiment is renamed later.
- If `runs` is omitted, `null`, or empty, the run executes backbone-only mode:
  - if `backbone.source_experiment` is not set, it logs only the reserved `backbone` run (creating it only when absent);
  - if `backbone.source_experiment` is set, it reuses the source `backbone` run without creating a new local run
    (requires source backbone checkpoint artifacts and `a_ref`/`a_post` baselines).
  - if a local `backbone` run already exists while `backbone` config is non-null, execution is rejected.
- If `backbone.training` is not set and user-configured runs are present, all configured runs must be repair-controller
  runs (non-repair runs require local backbone training).

---

## 4) Repair Data (Disjoint from Training)

We use two related but distinct concepts alongside the sequential training stream:

- **Repair stream**: a parallel Avalanche stream whose experiences contain the repair subsets. This is a sequence of
  experiences, not maintained storage.
- **Repair buffer**: an optional maintained storage pool populated from the repair stream and used for controller
  fitting when you need a fixed-capacity memory and do not know how many classes will appear in the future. **The 
  default experimental pipeline does not use this buffer**.

- For each experience’s original training dataset, we split samples into:
  - a **training subset** used by the continual learner, and
  - a **repair subset** used only for controller fitting.
- The split is stratified per class within the experience and controlled by the experiment seed.
- Repair samples are never used for backbone training.
- Per class, the repair subset size is `max(1, min(repair.budget_per_class, n_class - 1))`.
  Setting `repair.budget_per_class: 0` disables the repair stream entirely.
- **Important:** The `repair` configuration section is **mandatory** for all experiments, even backbone-only runs.
  This forces an explicit decision about data splitting. To use the full dataset (no repair hold-out), 
  you must explicitly set `repair.budget_per_class: 0`. This prevents unintentional data leakage where a 
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
\text{Acc}\Big(\text{test}(T_i)\ \text{with label space } \{0,\dots,99\}\Big)
$$

>ℹ️ With prevention controllers, $A_{\text{post}}$ is the posthoc evaluation of the model trained under the controller
> (no eval-time toggling). With repair controllers, $A_{\text{post}}$ is the shared controller-off baseline taken from
> the reserved `backbone` run for the same checkpoint trajectory.

### Controller accuracy (all-classes)
$$
A_{\text{ctrl}}(T_i;\theta,b) =
\text{Acc}\Big(\text{test}(T_i)\ \text{with label space } \{0,\dots,99\}\Big)
$$
>ℹ️ $A_{\text{ctrl}}$ and $\rho$ are logged only for repair controllers, because prevention controllers do not define
> a post-hoc, toggleable evaluation-time correction.

### Final global accuracy (all classes)
$$
A_{\text{final}} = \text{Acc}\Big(\text{test}(\{0,\dots,99\})\Big)
$$
>ℹ️ In practice this corresponds to the posthoc stream accuracy (e.g., `Top1_Acc_Stream`). It is also surfaced as a
> normalized `summary.final_a_post_mean`; see [section 7](#7-metric-logging--reporting-mlflow).

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

> ℹ️ **SQLite backend required:** REGAIN forces MLflow to use a SQLite backend. Provide a tracking URI like
> `sqlite:///path/to/mlflow.db` (or a filesystem path such as `./mlflow.db`). Non-SQLite tracking URIs are rejected.

### 7.1 Metric namespaces
Metric keys are normalized and namespaced as:

- `train.<metric>` for training-time logs
- `eval.<metric>` for Avalanche built-in evaluation logs
- `analysis.<...>` for analysis metrics computed by the evaluation plugin
- `summary.<...>` for end-of-run summary scalars (including normalized posthoc eval metrics)
- `<metric>` (no additional prefix) for raw posthoc evaluation logs

**Important implications**
- Runs do not emit paired controller-off/controller-on posthoc metric streams in the same nested run.
- In repair-controller runs, controller-off analysis baselines (`a_ref`, `a_post`) are inherited from the `backbone`
  run; posthoc raw metrics in repair runs correspond to controller-on evaluation.
- Posthoc raw metrics are unprefixed and should be interpreted in the context of their run placement
  (parent run or nested `exp###` / `final`).
- `train.*` metrics always correspond to the actual training procedure that was executed:
  - If a training-time controller was used, `train.*` metrics reflect that controller-influenced training.
  - `train.*` is not a controller-off/controller-on comparison; it is simply “what happened during training”.

### 7.2 MLflow run structure
Each experiment run creates:

- A **parent MLflow run** for each executed run.
- Parent-run creation rules (reserved `backbone` run, `backbone.source_experiment` reuse, and rejection conditions) are
  defined in section 3.5.
- **Nested MLflow runs** are created only for repair-controller runs with `repair.fit_schedule=per_experience`:
  - Per-experience checkpoint runs: `exp###`
  - Optional fallback final run: `final`
- Nested runs store only unprefixed posthoc raw metrics (`<metric>`).
- `analysis.*` and `summary.*` metrics always remain in the parent run.

Metric-family placement summary:

| Metric family                          | Key pattern      | Run placement                           |
|----------------------------------------|------------------|-----------------------------------------|
| Training metrics                       | `train.<metric>` | Parent run                              |
| Avalanche scheduled evaluation metrics | `eval.<metric>`  | Parent run                              |
| Analysis vectors                       | `analysis.<...>` | Parent run                              |
| Summary metrics                        | `summary.<...>`  | Parent run                              |
| Posthoc raw evaluation metrics         | `<metric>`       | Parent run or nested `exp###` / `final` |

### 7.3 Where the analysis artifacts live
After each experience, the evaluation plugin logs:

- Per-task:
  - `analysis.a_ref.exp###` (to the parent run)

At the end of training, when reference accuracies are complete, the evaluation plugin logs:

- Per-task:
  - `analysis.a_post.exp###`
- Per-task (repair controllers only):
  - `analysis.a_ctrl.exp###`
  - `analysis.rho.exp###` (only for valid tasks)
- Aggregates:
  - `summary.final_a_post_mean`
- Aggregates (repair controllers only):
  - `analysis.rho_mean`
  - `summary.final_rho_mean`
  - `summary.final_a_ctrl_mean`
- If reference accuracies are incomplete, the plugin logs no `analysis.a_post` / `analysis.a_ctrl` / `analysis.rho`
  vectors and instead writes a JSON artifact (`analysis_artifacts.json`) containing:
  - `status` (set to `incomplete_a_ref`)
  - `expected_num_experiences`
  - `observed_num_reference_points`
  - `eps`
  - partial `a_ref` vector
- A flag metric `analysis.incomplete_a_ref=1.0` is logged when reference accuracies are incomplete.

### 7.4 What gets reported
We report mean ± std across seeds (common configs use **3 seeds**) for:

- $A_{\text{final}}$
- $\{A_{\text{post}}(T_i)\}_{i=1}^{N}$ (where $N=\text{num\_experiences}$)
- $\bar{\rho}(\theta,b)$ for each repair controller and budget
- Optionally: $\overline{A}_{\text{ctrl}}(\theta,b)$ for each repair controller and budget

**Persisted artifacts**
- Per-task accuracies $(A_{\text{ref}}, A_{\text{post}})$ in runs with complete reference vectors
- Per-task accuracies $(A_{\text{ctrl}})$ for repair controllers with complete reference vectors
- Per-task $\rho$ for valid tasks in repair-controller runs (invalid tasks are omitted from `analysis.rho.exp###`)
- Aggregate summaries and curve/frontier inputs

---

## 8) REGAIN analysis tool: Curves & Frontiers

### 8.1 Recoverability curves
For each repair controller capacity level, we compute curves over repair budget $b$:

- Mean recoverable fraction:
  $$
  \bar{\rho}(\theta,b)=\mathbb{E}_{i:\,F_{\text{total}}(T_i)>\epsilon}\big[\rho(T_i;\theta,b)\big]
  $$
- Mean controller accuracy across tasks:
  $$
  \overline{A}_{\text{ctrl}}(\theta,b)=\mathbb{E}_{i}[A_{\text{ctrl}}(T_i;\theta,b)]
  $$

We also report $\rho(T_i;\theta,b)$ as a function of task index $i$ (“task age”).

### 8.2 Repair efficiency frontier
We place controller configurations on frontiers defined by:

- **Data cost:** shots/class $b$, where $b = \text{repair.budget\_per\_class}$
  (total repair examples = `repair.budget_per_class * num_classes`)
- **Parameter cost:** controller parameter count $|\theta|$

Frontiers plot recovered performance (e.g., $\bar{\rho}$, $\overline{A}_{\text{ctrl}}$) against these costs.
