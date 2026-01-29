# Experimental Framework

This document specifies the benchmark protocol, controller regimes, **evaluation modes**, and **metric logging organization**
used to measure **retrieval-correctable forgetting** on **SplitCIFAR-100** in a **single-head class-incremental learning (CIL)** setting.

---

## Contents

- [0) Overview](#0-overview)
- [1) Benchmark & Scenario](#1-benchmark--scenario)
- [2) Model & Training Protocol](#2-model--training-protocol)
- [3) Controllers & Evaluation Modes](#3-controllers--evaluation-modes)
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
3. Optionally fit and/or apply a controller (depending on controller type).
4. Evaluate and log metrics (with MLflow parent + nested evaluation runs).
5. Compute analysis artifacts: $A_{\text{ref}}, A_{\text{post}}$, plus $A_{\text{ctrl}}, \rho$ and aggregates when a
   repair controller is present.

`regain/cli/run_experiment.py` executes this pipeline and logs metrics to MLflow. The REGAIN analysis tool
(`regain/cli/run_analysis.py`) consumes the logged metrics to generate curves and frontiers.

**Mental model.** Experiments write per-experience metrics and analysis artifacts to MLflow. The analysis tool is a
read-only post-processing step that aggregates those logged runs into tables, recoverability curves, and efficiency
frontiers.

**What varies across runs.**
- Seed
- Strategy (naive / replay)
- Controller (optional)
- `eval_mode` ("single" or "compare"; subject to constraints)

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
  (`_verify_classes` in `regain/experiments/core.py`).

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

## 3) Controllers & Evaluation Modes

This codebase distinguishes *when* a controller acts:

### 3.1 Controller types (code-level meaning)

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
  hooks (`on_eval_*`) and `correct_outputs` when enabled. They do **not** receive training-time hooks.

### 3.2 Repair/fitting protocol by controller type

**`PreventionController` (training-time intervention)**
- Acts during training; it may change:
  - the loss surface and optimization trajectory, and/or
  - the backbone architecture/normalization behavior.
- It is not purely post-hoc, so “base vs ctrl” comparisons must be treated carefully (see `eval_mode` constraints).

**`RepairController` (post-hoc repair)**
- After each training experience, the repair stream provides that experience’s repair subset.
- Fitting schedule:
  - If `repair_after_experience=True` (default), fit after each experience.
  - If `repair_after_experience=False`, fit once after the full training sequence.
  - Controllers that require per-experience fitting (`requires_per_experience_fitting()`) will error if the flag is off.
- During fitting, backbone parameters $\phi$ are frozen (inference-only forwards); the controller trains on samples from
  the **repair stream** (aggregated across experiences).
- The test set is never used for fitting $g_\theta$.

### 3.3 `eval_mode` ("single" vs "compare") and constraints

Posthoc evaluations (the ones that log `base-*` / `ctrl-*` metrics) are executed inside **nested** MLflow runs. Reference
evaluation for $A_{\text{ref}}$ runs without a nested run and logs only `analysis-*` metrics to the parent run.

**`eval_mode="single"`**
- One nested evaluation run is created:
  - `run_name="ctrl"` when any controller is present (suffix `_exp###` when running per-experience posthoc evals).
  - `run_name="base"` when no controller is configured (suffix `_exp###` when running per-experience posthoc evals).
- Behavior:
  - If a repair controller plugin exists, it is enabled for evaluation in this mode (prevention controllers are baked
    into the trained model).
  - Metrics are logged under the `ctrl-*` namespace when a controller is configured, otherwise `base-*` (see logging section).

**`eval_mode="compare"`**
- Two nested evaluation runs are created:
  - `run_name="base"`: controller disabled during evaluation (suffix `_exp###` when running per-experience posthoc evals)
  - `run_name="ctrl"`: controller enabled during evaluation (suffix `_exp###` when running per-experience posthoc evals)
- Metrics are logged separately under `base-*` and `ctrl-*` namespaces.
- Summary metrics from both are also logged to the parent run (prefixed accordingly).

**Constraint (enforced in code)**
- `eval_mode="compare"` requires a controller and is only allowed for **toggleable repair controllers**.
- In code terms: `compare` is disallowed if the controller is `None` or a `PreventionController`.

Rationale:
- Training-time controllers change the learned model itself, so you cannot fairly obtain a “base” model by simply toggling
  evaluation-time application.

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
- Per class, the repair subset size is `min(repair_budget_per_class, n_class - 1)`, clamped to `[1, n_class - 1]`.
  Setting `repair_budget_per_class <= 0` disables the repair stream entirely.

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
> (no eval-time toggling). With repair controllers, $A_{\text{post}}$ is the repair-disabled evaluation of the standard
> trained model. Both are "no eval-time correction" baselines, but they are not directly comparable across controller
> regimes without context.

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
>ℹ️ In practice this corresponds to the posthoc stream accuracy (e.g., `Top1_Acc_Stream`) and is logged under `base-*` or
> `ctrl-*` in the nested eval run, then surfaced in the parent run as a normalized `summary-*` metric.

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
- All accuracies $A_{\text{ref}}, A_{\text{post}}, A_{\text{ctrl}}$ are still recorded for every task.

---

## 7) Metric Logging & Reporting (MLflow)

This section describes **how metrics are organized in MLflow** and **what we report**.

### 7.1 MLflow run structure (parent + nested evaluation runs)
Each experiment run creates:

- A **parent MLflow run** (the main run configured by `run_name`).
- One or more **nested MLflow runs** used for evaluation:
  - Always at least one nested run (even in `eval_mode="single"`).
  - In `eval_mode="compare"`, there are two nested runs (`base` and `ctrl`).
  - If `repair_after_experience=True` with a repair controller, additional nested runs are created per experience
    (named `base_exp###` and/or `ctrl_exp###`).
- Reference evaluation for $A_{\text{ref}}$ does not use nested runs; it logs `analysis-a_ref-exp###` directly to the
  parent run.

Nested evaluation runs are used so that metrics from different evaluation regimes do not collide.

### 7.2 Metric namespaces (how keys look)
Metric keys are normalized and namespaced as:

- `train-<metric>` for training-time logs
- `eval-<metric>` for Avalanche built-in evaluation logs (when `eval_every` is enabled)
- `ctrl-<metric>` for `eval_mode="single"` evaluation logs when a controller is configured
- `base-<metric>` for `eval_mode="single"` evaluation logs when no controller is configured
- `base-<metric>` for `eval_mode="compare"` base evaluation logs
- `ctrl-<metric>` for `eval_mode="compare"` controller evaluation logs
- `analysis-<...>` for analysis metrics computed by the evaluation plugin
- `summary-<...>` for end-of-run summary scalars written to the parent run (including normalized posthoc eval metrics)

**Important implications**
- In `eval_mode="compare"`, you should compare `base-*` vs `ctrl-*` metrics.
- In `eval_mode="single"`, evaluation metrics live under `ctrl-*` when a controller is configured, otherwise `base-*`.
- `train-*` metrics always correspond to the actual training procedure that was executed:
  - If a training-time controller was used, `train-*` metrics reflect that controller-influenced training.
  - `train-*` is not “base vs ctrl”; it is simply “what happened during training”.

### 7.3 Where the analysis artifacts live
After each experience, the evaluation plugin logs:

- Per-task:
  - `analysis-a_ref-exp###` (to the parent run)

At the end of training, the evaluation plugin logs:

- Per-task:
  - `analysis-a_post-exp###`
- Per-task (repair controllers only):
  - `analysis-a_ctrl-exp###`
  - `analysis-rho-exp###` (only for valid tasks)
- Aggregates:
  - `summary-final_a_post_mean`
- Aggregates (repair controllers only):
  - `analysis-rho_mean`
  - `summary-final_rho_mean`
  - `summary-final_a_ctrl_mean`
- If reference accuracies are incomplete, a JSON artifact (`analysis_artifacts.json`) is logged containing:
  - `status` (set to `incomplete_a_ref`)
  - `expected_num_experiences`
  - `observed_num_reference_points`
  - `eps`
  - partial `a_ref` vector
- A flag metric `analysis-incomplete_a_ref=1.0` is logged when reference accuracies are incomplete.

### 7.4 What gets reported
We report mean ± std across seeds (common configs use **3 seeds**) for:

- $A_{\text{final}}$
- $\{A_{\text{post}}(T_i)\}_{i=1}^{N}$ (where $N=\text{num\_experiences}$)
- $\bar{\rho}(\theta,b)$ for each repair controller and budget
- Optionally: $\overline{A}_{\text{ctrl}}(\theta,b)$ for each repair controller and budget

**Persisted artifacts**
- Per-task accuracies $(A_{\text{ref}}, A_{\text{post}})$ in all runs
- Per-task accuracies $(A_{\text{ctrl}})$ for repair controllers
- Per-task $\rho$ with invalid tasks marked for repair controllers
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

- **Data cost:** shots/class $b$, where $b = \text{repair\_budget\_per\_class}$
  (total repair examples = `repair_budget_per_class * num_classes`)
- **Parameter cost:** controller parameter count $|\theta|$

Frontiers plot recovered performance (e.g., $\bar{\rho}$, $\overline{A}_{\text{ctrl}}$) against these costs.
